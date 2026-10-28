# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Import a HuggingFace-trained MaxToki regression (timelapse/aging) model into BioNeMo.

The HF regression model is a custom wrapper (state-dict prefix ``cf.``) around a Llama
backbone (``cf.model.*``, ``cf.lm_head.weight``) with an explicit regression head
``linear1`` = ``Linear(hidden, 1)``. Unlike the base MaxToki LM, it is not a plain
``LlamaForCausalLM`` and ships without a ``config.json``, so this importer reads the
weights directly from the safetensors file, strips the ``cf.`` prefix, and maps:

    cf.model.*   -> Llama backbone (QKV interleave + gate/up merge, as in import_hf)
    cf.lm_head   -> output_layer
    linear1.*    -> regression_head.linear_fc2.*  (on FinetuneLlamaModel)

The backbone hyperparameters that cannot be inferred from weight shapes (head count,
RoPE base, norm epsilon) are taken from a Llama-style ``config.json`` passed via
``--hf-config`` (e.g. the matching ``MaxToki-217M-HF/config.json`` from the hub); the
vocab size is taken from the checkpoint itself.

Example:
    python -m bionemo.maxtoki.import_regression_hf \\
        --model-safetensors /path/aging_trained_maxtoki_217M_HF/model.safetensors \\
        --hf-config /path/MaxToki-217M-HF/config.json \\
        --tokenizer-path /path/token_dictionary.json \\
        --train-data-path /path/heldout_ages.dataset \\
        --val-data-path /path/heldout_ages.dataset \\
        --test-data-path /path/heldout_ages.dataset \\
        --seq-length 16384 \\
        --result-dir ./results --experiment-name maxtoki_aging_import
"""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.utils import logging
from safetensors.torch import load_file

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import setup_nemo_lightning_logger
from bionemo.maxtoki.datamodule import MaxTokiDataModule
from bionemo.maxtoki.lightning import maxtoki_lightning_module
from bionemo.maxtoki.model import MaxTokiFineTuneConfig, maxtoki_forward_step
from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer


__all__: Sequence[str] = ("build_regression_config", "convert_regression_state", "get_parser", "main")

CF_PREFIX = "cf."

# The RMSNorm epsilon the HF MaxToki models were trained with. The trained checkpoints ship without a
# config.json, so `--hf-config` is usually a stand-in written by hand or taken from a sibling model —
# and a stand-in carrying the Llama default of 1e-5 silently produces a subtly wrong conversion:
# every RMSNorm in every layer is off, and the epsilon is then baked into the saved checkpoint.
# The failure is quiet (predictions stay ~99.99% correlated with the reference)
MAXTOKI_RMS_NORM_EPS = 1e-6


def _read_safetensors_header_shapes(path: Path) -> dict:
    """Read the safetensors header (without loading tensors) to infer shapes."""
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def build_regression_config(
    model_safetensors: Path,
    hf_config_path: Path,
    seq_length: int,
    precision_dtype: torch.dtype = torch.float32,
    rms_norm_eps: float = MAXTOKI_RMS_NORM_EPS,
) -> MaxTokiFineTuneConfig:
    """Build a MaxTokiFineTuneConfig (FinetuneLlamaModel + regression head) for the HF model.

    Backbone hyperparameters come from ``hf_config_path`` (a Llama config.json); the
    number of layers / hidden / ffn sizes and the vocab size are taken from the actual
    checkpoint so the config always matches the weights being loaded.

    SDPA attention is forced (rather than relying on the runtime auto-switch, which only
    triggers when ``initial_ckpt_path`` is set) because the head dim is frequently not a
    multiple of 8.

    ``rms_norm_eps`` sets the norm epsilon (default ``MAXTOKI_RMS_NORM_EPS``). The supplied config's
    ``rms_norm_eps`` is not used; a warning is logged if it differs.
    """
    with open(hf_config_path) as f:
        hf = json.load(f)

    shapes = _read_safetensors_header_shapes(model_safetensors)
    layer_ids = set()
    for k in shapes:
        parts = k.split("layers.")
        if len(parts) > 1:
            layer_ids.add(int(parts[1].split(".")[0]))
    num_layers = max(layer_ids) + 1
    vocab_size, hidden_size = shapes[f"{CF_PREFIX}model.embed_tokens.weight"]["shape"]
    ffn_hidden_size = shapes[f"{CF_PREFIX}model.layers.0.mlp.gate_proj.weight"]["shape"][0]

    num_attention_heads = hf["num_attention_heads"]
    head_dim = hf.get("head_dim") or (hidden_size // num_attention_heads)

    config_eps = hf.get("rms_norm_eps")
    if config_eps is not None and not math.isclose(config_eps, rms_norm_eps, rel_tol=1e-9):
        logging.warning(
            f"{hf_config_path} sets rms_norm_eps={config_eps!r}; using --rms-norm-eps={rms_norm_eps!r} instead. "
            f"MaxToki HF models were trained with {MAXTOKI_RMS_NORM_EPS!r}."
        )

    # Every derived hyperparameter is logged so a bad --hf-config is visible in the run log rather
    # than silently baked into the checkpoint.
    logging.info(
        f"Derived backbone config: num_layers={num_layers} hidden_size={hidden_size} "
        f"ffn_hidden_size={ffn_hidden_size} num_attention_heads={num_attention_heads} "
        f"num_query_groups={hf['num_key_value_heads']} kv_channels={head_dim} vocab_size={vocab_size} "
        f"rotary_base={hf['rope_theta']} layernorm_epsilon={rms_norm_eps} "
        f"(config.json said {config_eps!r})"
    )

    # RoPE is derived from the HF config's rope_scaling rather than assumed. Two context-extension
    # strategies exist for MaxToki: the BioNeMo temporal finetune (train.py) uses llama3 scaling
    # (--rope-scaling-factor, scale_factor > 1), while the HF-trained models here use rope_scaling=null
    # (standard RoPE, θ=rope_theta) and rely on pure extrapolation for the long context. Which one a
    # given checkpoint used is recorded in its config, so we read it here:
    #   - llama3 block -> scale_factor = its factor, old_context_len = its original_max_position_embeddings
    #   - null/absent  -> scale_factor = 1.0 (apply_rope_scaling with factor 1.0 is identity)
    # For the aging model this yields scale_factor=1.0, empirically the only setting that reproduces its
    # HF reference on the longest (16k-token) sequences (no-scaling MAE ~0.3 vs llama3-factor-4 MAE ~117).
    rope_scaling = hf.get("rope_scaling")
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        scale_factor = float(rope_scaling.get("factor", 8.0))
        old_context_len = int(rope_scaling.get("original_max_position_embeddings", hf["max_position_embeddings"]))
        low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
    else:
        scale_factor = 1.0
        old_context_len = int(hf.get("max_position_embeddings", seq_length))
        low_freq_factor, high_freq_factor = 1.0, 4.0

    config = MaxTokiFineTuneConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=hf["num_key_value_heads"],
        kv_channels=head_dim,
        init_method_std=hf.get("initializer_range", 0.02),
        layernorm_epsilon=rms_norm_eps,
        seq_length=seq_length,
        rotary_base=hf["rope_theta"],
        gated_linear_unit=True,
        make_vocab_size_divisible_by=True,
        share_embeddings_and_output_weights=hf.get("tie_word_embeddings", False),
        vocab_size=vocab_size,
        scale_factor=scale_factor,
        old_context_len=old_context_len,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        rope_scaling_factor=scale_factor,
        # The regression head ships in this checkpoint; do NOT skip it on load.
        initial_ckpt_skip_keys_with_these_prefixes=[],
        # Force SDPA: head_dim is typically not divisible by 8 (TE flash attention requires it).
        transformer_layer_spec=sdpa_layer_spec,
        persist_layer_norm=False,
        attention_softmax_in_fp32=False,
        masked_softmax_fusion=True,
        fp16=False,
        bf16=precision_dtype == torch.bfloat16,
        params_dtype=precision_dtype,
        autocast_dtype=precision_dtype,
    )
    return config


def convert_regression_state(target, source_path: Path):
    """Map the cf-prefixed HF regression checkpoint onto the BioNeMo FinetuneLlamaModel.

    Reuses the same backbone mapping as ``HFMaxTokiImporter.convert_state`` (embedding /
    output padding, per-query-group QKV interleave, gate/up merge), then additionally maps
    the ``linear1`` regression head to ``regression_head.linear_fc2`` and writes layer-norm
    weights under the key layout the target actually uses (TE-fused vs SDPA-separate).

    Args:
        target: Target NeMo model (MegatronParallel-wrapped FinetuneLlamaModel).
        source_path: Path to the HF model's ``model.safetensors`` file (or its directory).

    Returns:
        The target model with weights loaded.
    """
    source_path = Path(source_path)
    if source_path.is_dir():
        source_path = source_path / "model.safetensors"

    raw = load_file(str(source_path))
    # Strip the cf. wrapper so backbone keys match the standard HF Llama layout.
    source_sd = {}
    for k, v in raw.items():
        if k.startswith(CF_PREFIX):
            source_sd[k[len(CF_PREFIX) :]] = v
        else:
            source_sd[k] = v

    # Unwrap the (MegatronParallel -> Lightning -> DDP -> Float16Module -> ...) wrappers down to
    # the actual GPT model. The nesting depth varies with the DDP/precision config, so we unwrap
    # until we reach the module that owns the decoder/embedding rather than hardcoding a depth.
    # Use the module's OWN child modules (``_modules``) rather than ``hasattr``: wrappers like DDP
    # delegate attribute access to their child, so ``hasattr(ddp, "decoder")`` is True and would
    # stop unwrapping at a level whose state_dict keys carry a ``module.`` prefix.
    def _owns_gpt(m):
        children = getattr(m, "_modules", {})
        return "decoder" in children and "embedding" in children

    target_model = target
    for _ in range(10):
        if _owns_gpt(target_model):
            break
        if hasattr(target_model, "module"):
            target_model = target_model.module
        else:
            break
    if not _owns_gpt(target_model):
        raise RuntimeError("Could not unwrap target down to the GPT model (no .decoder/.embedding found).")
    target_sd = target_model.state_dict()
    config = target_model.config

    num_layers = config.num_layers
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = config.hidden_size
    head_size = config.kv_channels

    # Whether the target fuses layer norms into the linear layers (TE) or keeps them
    # as separate modules (SDPA / local specs).
    te_fused_ln = any("linear_qkv.layer_norm_weight" in k for k in target_sd)

    def _to_tgt(src_tensor, tgt_key):
        return src_tensor.to(target_sd[tgt_key].dtype)

    # Embedding with padding (HF vocab may be smaller than NeMo's padded vocab).
    src_emb = source_sd["model.embed_tokens.weight"]
    tgt_emb = target_sd["embedding.word_embeddings.weight"].clone()
    tgt_emb[: src_emb.shape[0], :] = src_emb.to(tgt_emb.dtype)
    target_sd["embedding.word_embeddings.weight"] = tgt_emb

    # LM head with padding (only when output weights are untied).
    if "output_layer.weight" in target_sd and "lm_head.weight" in source_sd:
        src_lmh = source_sd["lm_head.weight"]
        tgt_lmh = target_sd["output_layer.weight"].clone()
        tgt_lmh[: src_lmh.shape[0], :] = src_lmh.to(tgt_lmh.dtype)
        target_sd["output_layer.weight"] = tgt_lmh

    target_sd["decoder.final_layernorm.weight"] = _to_tgt(
        source_sd["model.norm.weight"], "decoder.final_layernorm.weight"
    )

    for i in range(num_layers):
        tgt_key = f"decoder.layers.{i}.self_attention.linear_proj.weight"
        target_sd[tgt_key] = _to_tgt(source_sd[f"model.layers.{i}.self_attn.o_proj.weight"], tgt_key)

        tgt_key = f"decoder.layers.{i}.mlp.linear_fc2.weight"
        target_sd[tgt_key] = _to_tgt(source_sd[f"model.layers.{i}.mlp.down_proj.weight"], tgt_key)

        # Layer norms: target layout depends on TE-fused vs separate (SDPA) modules.
        if te_fused_ln:
            qkv_ln_key = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"
            mlp_ln_key = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"
        else:
            qkv_ln_key = f"decoder.layers.{i}.input_layernorm.weight"
            mlp_ln_key = f"decoder.layers.{i}.pre_mlp_layernorm.weight"
        target_sd[qkv_ln_key] = _to_tgt(source_sd[f"model.layers.{i}.input_layernorm.weight"], qkv_ln_key)
        target_sd[mlp_ln_key] = _to_tgt(source_sd[f"model.layers.{i}.post_attention_layernorm.weight"], mlp_ln_key)

        # QKV merge: interleave q, k, v per query group (Megatron layout).
        q = source_sd[f"model.layers.{i}.self_attn.q_proj.weight"].view(head_num, head_size, -1)
        k = source_sd[f"model.layers.{i}.self_attn.k_proj.weight"].view(num_query_groups, head_size, -1)
        v = source_sd[f"model.layers.{i}.self_attn.v_proj.weight"].view(num_query_groups, head_size, -1)
        qkv_chunks = []
        for g in range(num_query_groups):
            qkv_chunks.append(q[g * heads_per_group : (g + 1) * heads_per_group, :, :])
            qkv_chunks.append(k[g : g + 1, :, :])
            qkv_chunks.append(v[g : g + 1, :, :])
        qkv = torch.cat(qkv_chunks).reshape(head_size * (head_num + 2 * num_query_groups), hidden_size)
        tgt_key = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
        target_sd[tgt_key] = qkv.to(target_sd[tgt_key].dtype)

        # MLP gate/up merge into FC1.
        gate = source_sd[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = source_sd[f"model.layers.{i}.mlp.up_proj.weight"]
        tgt_key = f"decoder.layers.{i}.mlp.linear_fc1.weight"
        target_sd[tgt_key] = torch.cat([gate, up], dim=0).to(target_sd[tgt_key].dtype)

    # Regression head: linear1 -> regression_head.linear_fc2.
    for src_name, tgt_name in (
        ("linear1.weight", "regression_head.linear_fc2.weight"),
        ("linear1.bias", "regression_head.linear_fc2.bias"),
    ):
        if src_name not in source_sd:
            raise KeyError(f"Expected regression head weight '{src_name}' not found in HF checkpoint.")
        if tgt_name not in target_sd:
            raise KeyError(
                f"Target model has no '{tgt_name}'. Is the config using FinetuneLlamaModel (regression head)?"
            )
        target_sd[tgt_name] = _to_tgt(source_sd[src_name], tgt_name)

    # Sanity: every parameter in the target model must have been assigned from the source.
    # load_state_dict(strict=False) silently skips mismatched keys, which would leave weights
    # at random init and produce garbage outputs, so we check coverage explicitly.
    real_keys = set(target_model.state_dict().keys())
    written = set(target_sd.keys()) & real_keys
    IGNORE = ("_extra_state", "rotary_pos_emb", "inv_freq")
    # Keys we placed in target_sd that do not exist on the model (typos / wrong layout).
    spurious = sorted(set(target_sd.keys()) - real_keys)
    # Params that were initialized but never overwritten by a source weight.
    param_names = {n for n, _ in target_model.named_parameters()}
    missing_params = sorted(n for n in param_names if n not in written and not any(tok in n for tok in IGNORE))
    if spurious:
        print(f"[convert_regression_state] WARNING: {len(spurious)} keys not on model: {spurious[:8]}")
    if missing_params:
        raise KeyError(
            f"{len(missing_params)} target parameters were not assigned a source weight "
            f"(would stay at init -> garbage). First few: {missing_params[:12]}"
        )

    # Preserve main_grad references (used by the distributed optimizer).
    main_grad_map = {n: p.main_grad for n, p in target_model.named_parameters() if hasattr(p, "main_grad")}
    target_model.load_state_dict(target_sd, strict=False)
    for n, p in target_model.named_parameters():
        if n in main_grad_map:
            p.main_grad = main_grad_map[n]

    return target


def main(
    model_safetensors: Path,
    hf_config_path: Path,
    tokenizer_path: Path,
    train_data_path: Path,
    val_data_path: Path,
    test_data_path: Path,
    result_dir: Path,
    experiment_name: str,
    seq_length: int,
    devices: int = 1,
    num_nodes: int = 1,
    micro_batch_size: int = 1,
    precision: PrecisionTypes = "32",
    num_dataset_workers: int = 0,
    limit_val_batches: int = 1,
    compute_seq_length: int = 512,
    rms_norm_eps: float = MAXTOKI_RMS_NORM_EPS,
) -> None:
    """Import the HF regression model and save a BioNeMo checkpoint (1 step at lr=0).

    Runs in fp32 with the distributed optimizer disabled so the converted weights, loaded by
    the model_transform, are the model's only copy and are written verbatim to the checkpoint
    (a mixed-precision master copy would otherwise overwrite them at the lr=0 step).

    The single throwaway train step runs on sequences truncated to ``compute_seq_length`` to keep
    memory bounded; the saved config still carries the full ``seq_length`` (rotary supports it), so
    prediction can run on the long held-out sequences.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=False,
        ckpt_async_save=False,
    )
    trainer = nl.Trainer(
        devices=devices,
        max_steps=1,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        val_check_interval=1,
        num_nodes=num_nodes,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision=precision),
        enable_checkpointing=True,
    )

    with open(tokenizer_path) as f:
        token_dictionary = {k: int(v) for k, v in json.load(f).items()}
    tokenizer = MaxTokiTokenizer(token_dictionary=token_dictionary)

    data = MaxTokiDataModule(
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        # Truncate the throwaway training step's data; the model config keeps the full seq_length.
        seq_length=compute_seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        rampup_batch_size=None,
        seed=42,
        num_workers=num_dataset_workers,
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        pretrain=True,
    )

    config = build_regression_config(
        model_safetensors=Path(model_safetensors),
        hf_config_path=Path(hf_config_path),
        seq_length=seq_length,
        rms_norm_eps=rms_norm_eps,
    )

    model = maxtoki_lightning_module(
        config,
        forward_step=maxtoki_forward_step,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=0.0,
                optimizer="adam",
                use_distributed_optimizer=False,
                fp16=config.fp16,
                bf16=config.bf16,
            ),
        ),
    )

    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=1,
        always_save_context=True,
    )
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=False,
        wandb_config=None,
        ckpt_callback=checkpoint_callback,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(resume_if_exists=False, resume_ignore_no_checkpoint=True),
        model_transform=partial(convert_regression_state, source_path=Path(model_safetensors)),
    )


def get_parser():
    """CLI parser for importing the HF regression model."""
    p = argparse.ArgumentParser(description="Import a HF MaxToki regression (timelapse) model into BioNeMo.")
    p.add_argument(
        "--model-safetensors", type=Path, required=True, help="Path to the HF model.safetensors (or its directory)."
    )
    p.add_argument(
        "--hf-config",
        type=Path,
        required=True,
        dest="hf_config_path",
        help="Llama-style config.json supplying backbone hyperparameters (head count, rope_theta, rms_norm_eps). "
        "Its rms_norm_eps is not used; see --rms-norm-eps.",
    )
    p.add_argument(
        "--rms-norm-eps",
        type=float,
        default=MAXTOKI_RMS_NORM_EPS,
        help=f"RMSNorm epsilon of the model being converted. Defaults to {MAXTOKI_RMS_NORM_EPS}, the value the "
        f"MaxToki HF models were trained with. --hf-config's rms_norm_eps is not used (a warning is logged if it "
        f"differs), because the trained checkpoints ship without a config.json and a stand-in carrying the Llama "
        f"default of 1e-5 silently produces a wrong conversion.",
    )
    p.add_argument("--tokenizer-path", type=Path, required=True, help="Path to token_dictionary.json.")
    p.add_argument("--train-data-path", type=Path, required=True)
    p.add_argument("--val-data-path", type=Path, required=True)
    p.add_argument("--test-data-path", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, default=Path("./results"))
    p.add_argument("--experiment-name", type=str, default="maxtoki_aging_import")
    p.add_argument(
        "--seq-length",
        type=int,
        required=True,
        help="Baked into the saved config; must be >= the longest sequence used at prediction (e.g. 16512).",
    )
    p.add_argument(
        "--compute-seq-length",
        type=int,
        default=512,
        help="Sequence length for the single throwaway import step (kept small to bound memory).",
    )
    p.add_argument("--num-gpus", type=int, default=1, dest="devices")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--num-dataset-workers", type=int, default=0)
    p.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Import precision. Defaults to fp32 so converted weights are saved verbatim.",
    )
    return p


def entrypoint():
    """Command-line entrypoint."""
    args = get_parser().parse_args()
    main(
        model_safetensors=args.model_safetensors,
        hf_config_path=args.hf_config_path,
        tokenizer_path=args.tokenizer_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        result_dir=args.result_dir,
        experiment_name=args.experiment_name,
        seq_length=args.seq_length,
        devices=args.devices,
        num_nodes=args.num_nodes,
        micro_batch_size=args.micro_batch_size,
        precision=args.precision,
        num_dataset_workers=args.num_dataset_workers,
        compute_seq_length=args.compute_seq_length,
        rms_norm_eps=args.rms_norm_eps,
    )


if __name__ == "__main__":
    entrypoint()
