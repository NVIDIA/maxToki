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

import argparse
import json

from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Type, get_args

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.base import default_layer_spec
from nemo.lightning import resume, teardown
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.maxtoki.datamodule import MaxTokiDataModule
from bionemo.maxtoki.model import (
    MaxTokiConfig,
    MaxTokiMultitaskFineTuneConfig,
    maxtoki_forward_step,
)
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.maxtoki.lightning import maxtoki_lightning_module
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


__all__: Sequence[str] = ("get_parser", "main")


def main(
    tokenizer_path: Path | None,
    train_data_path: Path,
    val_data_path: Path,
    test_data_path: Path,
    num_nodes: int,
    devices: int,
    seq_length: int,
    result_dir: Path,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    num_dataset_workers: int,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    experiment_name: str,
    resume_if_exists: bool,
    hf_model_path: Path,
    precision: PrecisionTypes,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: List[str] | None = None,
    wandb_group: Optional[str] = None,
    wandb_job_type: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: bool = False,
    wandb_log_model: bool = False,
    create_tensorboard_logger: bool = False,
    create_checkpoint_callback: bool = True,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    config_class: Type[MaxTokiConfig] = MaxTokiConfig,
    log_every_n_steps: int = 50,
    aligned_megatron_ddp: bool = False,
    recompilation_check: bool = False,
    label_scalar: float = 200.0,
    additive_penalty: float = 10.0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    timelapse_loss: Literal["mse", "ce"] = "ce",
) -> None:
    """Imports a huggingface model to NeMo format by importing it, and running a single step with zero learning rate, then saving a checkpoint.

    Args:
        tokenizer_path (Path): Path to the tokenizer file. Used with HF dataset. If not present, assumes the file exists in the parent directory to the train-data-path.
        train_data_path (Path): Path to the train dataset
        val_data_path (Path): Path to the val dataset
        test_data_path (Path): Path to the test dataset
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save
        num_dataset_workers (int): num dataset workers
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        cosine_rampup_frac (float): fraction of steps at the beginning of the run to ramp up the learning rate
        cosine_hold_frac (float): fraction of steps to hold the minimum learning rate at the end of the run
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        accumulate_grad_batches (int): if requested, gradients are only updated every `accumulate_grad_batches` steps.
        config_class (Type[MaxTokiConfig]): which model config do you want to train?
        metric_to_monitor_for_checkpoints (str): which metric do you want to monitor for checkpoints?
        precision (str): desired training precision
        save_last_checkpoint (bool): if you want the last checkpoint saved
        save_top_k (int): if you want the top k checkpoints all saved.
        resume_if_exists (bool): attempt to resume if the checkpoint exists
        wandb_entity (str): The team posting this run (default: your username or your default team)
        wandb_project (str): The name of the project to which this run will belong.
        wandb_tags (List[str]): Tags associated with this run.
        wandb_group (str): A unique string shared by all runs in a given group
        wandb_job_type (Optional[str]): Type of run, which is useful when you're grouping runs together into larger experiments using group.
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_id (str): Sets the version, mainly used to resume a previous run.
        wandb_anonymous (bool): Enables or explicitly disables anonymous logging.
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers.
        create_tensorboard_logger (bool): create the tensorboard logger
        create_checkpoint_callback (bool): create a ModelCheckpoint callback and attach it to the pytorch lightning trainer
        restore_from_checkpoint_path (path): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
        num_layers (int): Number of layers in MaxToki. Default to 16.
        hidden_size (int): Hidden size in MaxToki. Default to 2048.
        ffn_hidden_size (int): Feedforward hidden size in MaxToki. Default to 8192.
        num_attention_heads (int): Number of attention heads in MaxToki. Default to 32.
        log_every_n_steps (int): log at this interval.
        nsys_profiling (bool): Whether to enable the nsys profiling callback hooks.
        nsys_start_step (int): Step to start profiling.
        nsys_ranks (list[int]): GPU/node ranks to profile. Defaults to [0] (only main gpu.)
        nsys_end_step (int): Step to stop profiling.
        gc_interval (int): if a value > 0 is provided, this will turn off automatic garbage collection and only run
            at this requested interval of train/val steps.
        aligned_megatron_ddp (bool): if activated, this will activate a number of communication optimizations that are
            good for clusters.
        recompilation_check (bool): enable a recompilation check (only do on a small run) to verify that fused gpu
            kernels are not being regularly recompiled.
        include_unrecognized_vocab_in_dataset (bool): If set to True, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab.
        pretrain (bool): If true- uses a collate function for pretraining, otherwise uses a special collate function for finetuning.
        label_scalar (float): Scalar used to downscale regression labels. e.g. 200 => label = label / 200. This is inverted in the outputs.
        freeze_params_until_key_suffix (Optional[str]): Will disable grads for all parameters in the model parameter list until the key suffix is reached.
            For example, if the key suffix is "decoder.layers", then all parameters in the model parameter list will be frozen until a parameter name ends with "decoder.layers".
            Often parameters will have various levels of module nesting, such as "module.module.module.decoder.layers.0.weight", using the suffix allows for a more general pattern.
    """
    # Create the result directory if it does not exist.
    if wandb_tags is None:
        wandb_tags = []
    result_dir.mkdir(parents=True, exist_ok=True)
    val_check_interval = min(val_check_interval, num_steps)  # Training will fail if val_check_interval > num_steps

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    if aligned_megatron_ddp:
        ddp: str | DistributedDataParallelConfig = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        )
    else:
        ddp = "megatron"  # this will launch DistributedDataParallelConfig(check_for_nan_in_grad=True).

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp=ddp,
        progress_interval=log_every_n_steps,
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        gradient_as_bucket_view=True,
        ckpt_async_save=False,
        ckpt_parallel_load=True,
    )

    wandb_options: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            job_type=wandb_job_type,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )
    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=[],
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision=precision),  # type: ignore
        enable_checkpointing=create_checkpoint_callback,
    )

    if tokenizer_path is None:
        tokenizer_path = train_data_path.parent / "token_dictionary.json"

    with open(tokenizer_path) as f:
        token_dictionary = json.load(f)
    # Fixes dtype serialization error when checkpointing (token dictionary format may differ).
    token_dictionary = {k: int(v) for k, v in token_dictionary.items()}

    tokenizer = MaxTokiTokenizer(token_dictionary=token_dictionary)

    data = MaxTokiDataModule(
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        rampup_batch_size=None,
        seed=42,
        num_workers=num_dataset_workers,
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        # Since we arent actually training, we want to enable the collator that isnt so picky.
        pretrain=True,
    )

    # Optionally pass through Finetuning specific kwargs.
    finetuning_kwargs = (
        {
            "label_scalar": label_scalar,
            "additive_penalty": additive_penalty,
            "timelapse_loss": timelapse_loss,  # mse or ce
        }
        if issubclass(config_class, MaxTokiMultitaskFineTuneConfig)
        else {}
    )

    if issubclass(config_class, MaxTokiMultitaskFineTuneConfig):
        train_metrics = [
            TorchmetricsConfig(
                class_path="bionemo.maxtoki.model.UnreducedMSELoss",
                task="multitask",
                metric_name="train_mse_loss",
                kwargs={},
            ),
            TorchmetricsConfig(
                class_path="bionemo.maxtoki.model.UnreducedCELoss",
                task="multitask",
                metric_name="train_ce_loss",
                kwargs={},
            ),
            TorchmetricsConfig(
                class_path="bionemo.maxtoki.model.RegressionAvgPred",
                task="multitask",
                metric_name="train_regression_preds",
                kwargs={},
            ),
        ]
        valid_metrics = [
            TorchmetricsConfig(
                class_path="bionemo.maxtoki.model.UnreducedMSELoss",
                task="multitask",
                metric_name="valid_mse_loss",
                kwargs={},
            ),
            TorchmetricsConfig(
                class_path="bionemo.maxtoki.model.UnreducedCELoss",
                task="multitask",
                metric_name="valid_ce_loss",
                kwargs={},
            ),
        ]
        metric_kwargs = {"train_metric": train_metrics, "valid_metric": valid_metrics}
    else:
        metric_kwargs = {}

    # Instantiate the model config from the HF model config.
    hf_config = HFMaxTokiImporter.config(hf_model_path, tokenizer.vocab_size)
    # Create the actual lightning object.
    model = HFMaxTokiImporter.init(hf_config, data.tokenizer, tokenizer.vocab_size)

    # Configure our custom Checkpointer
    if create_checkpoint_callback:
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,
            filename="{epoch}-{val_loss:.2f}-{step}-{consumed_samples}",
        )
    else:
        checkpoint_callback = None

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_options,
        ckpt_callback=checkpoint_callback,
    )
    if recompilation_check:
        torch._dynamo.config.error_on_recompile = True  # type: ignore

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,
            resume_ignore_no_checkpoint=True,
        ),
        model_transform=partial(HFMaxTokiImporter.convert_state, source_path=hf_model_path),
    )


def sanity_check(model: torch.nn.Module, data: MaxTokiDataModule, hf_model_path: Path, seq_length: int) -> None:
    """Compare NeMo and HF model outputs on a sample."""
    nemo_model = model.to("cuda").eval()
    first_example = torch.tensor([data._train_ds[0]["input_ids"]]).to(model.device)
    first_example_position_ids = torch.arange(seq_length).unsqueeze(0).to(model.device)
    hf_model = load_and_predict_hf(hf_model_path, None, None)
    nemo_model = nemo_model.eval()
    nemo_output_real = nemo_model(first_example, first_example_position_ids, attention_mask=None)

    hf_model = hf_model.eval()
    hf_output_real = hf_model(first_example, first_example_position_ids, use_cache=False)
    print("done with the hf forward pass")


class HFMaxTokiImporter:  # io.ModelConnector["LlamaForCausalLM", MaxTokiConfig]):
    """Importer for converting Hugging Face Llama models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's LlamaModel format, including weight mapping and configuration translation.
    """

    @staticmethod
    def init(config, tokenizer, vocab_size) -> BionemoLightningModule:
        """Initialize a NeMo LlamaModel instance.

        Returns:
            Model: Initialized NeMo Llama Lightning module model with the appropriate configuration
                and tokenizer.
        """
        model = maxtoki_lightning_module(
            config,
            # Only used for vocab size.
            forward_step=maxtoki_forward_step,
            tokenizer=tokenizer,
            optimizer=MegatronOptimizerModule(
                config=OptimizerConfig(
                    lr=0.00,  # No learning during import-only runs.
                    optimizer="adam",
                    use_distributed_optimizer=True,
                    # Assuming import is actually fp32
                    fp16=config.fp16,
                    bf16=config.bf16,
                ),
            ),
        )
        return model

    def _apply(self, output_path: Path) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(str(self))
        source = AutoModelForCausalLM.from_pretrained(str(self), torch_dtype="auto", device_map=str(target_device))

        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Llama model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @staticmethod
    def convert_state(target, source_path):
        """Convert state dict from HF format to NeMo format.

        Manually maps weights from the HF model to the NeMo model, handling
        embedding/output padding, QKV merging, and MLP gate/up merging. Uses
        direct state_dict manipulation instead of io.apply_transforms to avoid
        dtype assertion issues with Transformer Engine lazy parameter init.

        Args:
            target: Target NeMo model (MegatronParallel-wrapped)
            source_path: Path to the HF model directory

        Returns:
            The target model with weights loaded from the HF source
        """
        from transformers import AutoModelForCausalLM

        target_model = target.module.module.module
        target_sd = target_model.state_dict()
        config = target_model.config

        source = AutoModelForCausalLM.from_pretrained(str(source_path), torch_dtype="auto")
        source.to("cuda")
        source_sd = source.state_dict()

        num_layers = config.num_layers
        head_num = config.num_attention_heads
        num_query_groups = config.num_query_groups
        heads_per_group = head_num // num_query_groups
        hidden_size = config.hidden_size
        head_size = config.kv_channels

        def _to_target_dtype(src_tensor, tgt_key):
            return src_tensor.to(target_sd[tgt_key].dtype)

        # Embedding with padding (HF vocab may be smaller than NeMo's padded vocab)
        src_emb = source_sd["model.embed_tokens.weight"]
        tgt_emb = target_sd["embedding.word_embeddings.weight"].clone()
        tgt_emb[: src_emb.shape[0], :] = src_emb.to(tgt_emb.dtype)
        target_sd["embedding.word_embeddings.weight"] = tgt_emb

        # LM head with padding (only when output weights are untied)
        if "output_layer.weight" in target_sd and "lm_head.weight" in source_sd:
            src_lmh = source_sd["lm_head.weight"]
            tgt_lmh = target_sd["output_layer.weight"].clone()
            tgt_lmh[: src_lmh.shape[0], :] = src_lmh.to(tgt_lmh.dtype)
            target_sd["output_layer.weight"] = tgt_lmh

        # Final layernorm
        target_sd["decoder.final_layernorm.weight"] = _to_target_dtype(
            source_sd["model.norm.weight"], "decoder.final_layernorm.weight"
        )

        for i in range(num_layers):
            # Attention output projection
            tgt_key = f"decoder.layers.{i}.self_attention.linear_proj.weight"
            target_sd[tgt_key] = _to_target_dtype(source_sd[f"model.layers.{i}.self_attn.o_proj.weight"], tgt_key)

            # Input layernorm (TE fuses this into the QKV projection)
            tgt_key = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"
            target_sd[tgt_key] = _to_target_dtype(source_sd[f"model.layers.{i}.input_layernorm.weight"], tgt_key)

            # Post-attention layernorm (TE fuses this into MLP FC1)
            tgt_key = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"
            target_sd[tgt_key] = _to_target_dtype(
                source_sd[f"model.layers.{i}.post_attention_layernorm.weight"], tgt_key
            )

            # MLP down projection
            tgt_key = f"decoder.layers.{i}.mlp.linear_fc2.weight"
            target_sd[tgt_key] = _to_target_dtype(source_sd[f"model.layers.{i}.mlp.down_proj.weight"], tgt_key)

            # QKV merge: interleave q, k, v per query group (matches Megatron layout)
            q = source_sd[f"model.layers.{i}.self_attn.q_proj.weight"]
            k = source_sd[f"model.layers.{i}.self_attn.k_proj.weight"]
            v = source_sd[f"model.layers.{i}.self_attn.v_proj.weight"]
            q = q.view(head_num, head_size, -1)
            k = k.view(num_query_groups, head_size, -1)
            v = v.view(num_query_groups, head_size, -1)
            qkv_chunks = []
            for g in range(num_query_groups):
                qkv_chunks.append(q[g * heads_per_group : (g + 1) * heads_per_group, :, :])
                qkv_chunks.append(k[g : g + 1, :, :])
                qkv_chunks.append(v[g : g + 1, :, :])
            qkv = torch.cat(qkv_chunks).reshape(head_size * (head_num + 2 * num_query_groups), hidden_size)
            tgt_key = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
            target_sd[tgt_key] = qkv.to(target_sd[tgt_key].dtype)

            # MLP gate/up merge into FC1
            gate = source_sd[f"model.layers.{i}.mlp.gate_proj.weight"]
            up = source_sd[f"model.layers.{i}.mlp.up_proj.weight"]
            fc1 = torch.cat([gate, up], dim=0)
            tgt_key = f"decoder.layers.{i}.mlp.linear_fc1.weight"
            target_sd[tgt_key] = fc1.to(target_sd[tgt_key].dtype)

        # Preserve main_grad references (used by distributed optimizer)
        main_grad_map = {}
        for name, param in target_model.named_parameters():
            if hasattr(param, "main_grad"):
                main_grad_map[name] = param.main_grad

        # Load the mapped state dict back into the target model
        target_model.load_state_dict(target_sd, strict=False)

        # Restore main_grad references
        for name, param in target_model.named_parameters():
            if name in main_grad_map:
                param.main_grad = main_grad_map[name]

        return target

    @staticmethod
    def config(hf_model_path, vocab_size) -> MaxTokiConfig:
        """Create a NeMo LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import AutoConfig

        source = AutoConfig.from_pretrained(str(hf_model_path))

        # RoPE: HF rope_scaling may be null; original_context_len should match max_position_embeddings.
        # There is no rope_type in the HF model at the moment, so the rope_type check will not match.
        if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
            # Apply Llama3.1 customize rope scaling
            cls = partial(MaxTokiConfig, scale_factor=source.rope_scaling.get("factor", 8.0))
        else:
            cls = partial(MaxTokiConfig, scale_factor=1.0)

        args = {}
        output = MaxTokiConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=(
                source.intermediate_size
                if not getattr(source, "intermediate_size_mlp", None)
                else source.intermediate_size_mlp
            ),
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=True,
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            # Could be useful for changing dtype, but we got good results with bf16 from fp32.
            # fp16=(dtype_from_hf(source) == torch.float16),
            # bf16=(dtype_from_hf(source) == torch.bfloat16),
            # params_dtype=dtype_from_hf(source),
            # autocast_dtype=dtype_from_hf(source),
            fp16=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            vocab_size=vocab_size,
            kv_channels=getattr(source, "head_dim", None),
            # This sets the attention backend, but the transformer_layer_spec sets the actual backend.
            # attention_backend="local",
            transformer_layer_spec=default_layer_spec,
            persist_layer_norm=False,
            attention_softmax_in_fp32=False,
            masked_softmax_fusion=True,
            scale_factor=1.0,
            old_context_len=source.max_position_embeddings,
            **args,
        )
        if output.scale_factor != 1.0:
            raise Exception("Scale factor is not 1.0")

        if output.old_context_len != source.max_position_embeddings:
            raise Exception("Old context length is not the same as the source max position embeddings")

        return output


def extract_devices(ckpt):
    """Extracts device from the input iterator
    ckpt can be module.named_parameters or module.state_dict().items()
    """
    devices = {}
    for key, val in ckpt:
        if hasattr(val, "device"):
            devices[key] = val.device
    return devices


def get_parser():
    """Return the cli parser for this tool."""
    parser = argparse.ArgumentParser(description="Import HF model to MaxToki format.")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=False,
        help="Path to the tokenizer file. Used with HF dataset. If not present, assumes the file exists in the parent directory to the train-data-path.",
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        required=True,
        help="Path to the data base directory",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        required=True,
        help="Path to the val dataset",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        required=True,
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-4,
        help="Learning rate for training. Default is 1e-4.",
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    parser.add_argument(
        "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument(
        "--experiment-name", type=str, required=False, default="maxtoki", help="Name of the experiment."
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=[], help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="Type of run, useful when grouping runs together into larger experiments.",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
    parser.add_argument(
        "--cosine-rampup-frac",
        type=float,
        required=False,
        default=0.01,
        help="Fraction of steps in which to ramp up the learning rate. Default is 0.01.",
    )
    parser.add_argument(
        "--include-unrecognized-vocab-in-dataset",
        action="store_true",
        help="If set to true, verify all gene identifers are in the tokenizer vocab.",
    )
    parser.add_argument(
        "--cosine-hold-frac",
        type=float,
        required=False,
        default=0.05,
        help="Fraction of final steps in which to hold the minimum LR. Default is 0.05.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of dataset workers. Default is 0.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps between validation checks. Default is 10000.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        required=False,
        default=50,
        help="Number of steps between logging. Default is 50.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=True,
        default=2048,
        help="Sequence length of cell. Default is 2048.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float_or_int_or_none,
        required=False,
        default=2,
        help="Number of global batches used for validation if int. Fraction of validation dataset if float. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=1,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        required=False,
        default=1,
        help="Gradient accumulation steps. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        default=True,
        help="Save the best checkpoint based on the metric to monitor.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action="store_true",
        default=True,
        help="Save the last checkpoint.",
    )
    parser.add_argument(
        "--metric-to-monitor-for-checkpoints",
        type=str,
        required=False,
        default="val_loss",
        help="The metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        required=False,
        default=2,
        help="Save the top k checkpoints.",
    )
    parser.add_argument(
        "--restore-from-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the checkpoint directory to restore from.",
    )
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers in MaxToki. Default to 16.")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size in MaxToki. Default to 2048.")
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=8192, help="Feedforward hidden size in MaxToki. Default to 8192."
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=32, help="Number of attention heads in MaxToki. Default to 32."
    )

    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range.",
    )
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )

    parser.add_argument(
        "--gc-interval",
        type=int,
        required=False,
        default=0,
        help="Run garbage collection on the cluster every --gc-interval steps, 0 to disable (default).",
    )

    parser.add_argument(
        "--aligned-megatron-ddp",
        action="store_true",
        default=False,
        help="Enable MegatronDDP communication optimizations for better cluster performance.",
    )
    parser.add_argument(
        "--recompilation-check",
        action="store_true",
        default=False,
        help="Enable recompilation checks for fused GPU kernels.",
    )
    parser.add_argument(
        "--use-finetuning-config",
        action="store_true",
        default=False,
        help="Uses the MaxTokiFineTuneConfig for finetuning instead of the MaxTokiConfig.",
    )
    parser.add_argument(
        "--initial-ckpt-path",
        type=str,
        required=False,
        default=None,
        help="Path to a saved checkpoint. Uses the weights from this checkpoint to initialize the model.",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        default=False,
        help="Enable pretraining.",
    )
    parser.add_argument(
        "--penalty-factor",
        type=float,
        required=False,
        default=10.0,
        help="Constant factor added to MSE loss during finetuning.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size.",
    )
    parser.add_argument("--rope-scale-factor", type=float, required=True, default=8.0, help="Rope scaling factor.")
    parser.add_argument(
        "--non-numeric-token-regression-penalty",
        type=float,
        required=False,
        default=10.0,
        dest="additive_penalty",
        help="Additive penalty for non-numeric token predictions in regression tasks weighed by their liklihood.",
    )
    parser.add_argument(
        "--label-scalar",
        type=float,
        required=False,
        default=200.0,
        help="Label scalar for regression tasks. e.g. 200 => label = label / 200. This is inverted in the outputs.",
    )
    parser.add_argument(
        "--freeze-params-until-key-suffix",
        type=str,
        required=False,
        default=None,
        help="Will disable grads for all parameters in the model parameter list until the key suffix is reached. For example, if the key suffix is 'decoder.layers', then all parameters in the model parameter list will be frozen until a parameter name ends with 'decoder.layers'.",
    )
    parser.add_argument(
        "--output-weights",
        choices=["tied", "separate"],
        required=True,
        default="tied",
        help="Whether to share the embeddings and output weights. Tied means the embedding and output weights are shared. Separate means new and separate parameters are used for the model outputs.",
    )
    parser.add_argument(
        "--initial-ckpt-skip-keys-with-these-prefixes",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Prefixes of keys to skip when loading the initial checkpoint. Using when loading a checkpoint that has a"
        "different structure than the model, such as in finetuning. If finetuning from a model with shared "
        "embedding/output weights, and youd like to add output weights, pass output_layer.weight as a prefix.",
    )
    parser.add_argument(
        "--timelapse-loss",
        choices=["mse", "ce"],
        required=False,
        default="mse",
        help="The loss to use for the timelapse task. When using cross entropy, logging outputs will still report MSE for timelapse, but the loss is computed with cross entropy.",
    )

    parser.add_argument(
        "--hf-model-path",
        type=Path,
        required=True,
        help="Path to the HuggingFace model to load weights and config into the model.",
    )
    return parser


def entrypoint():
    """Command-line entrypoint for HF model import."""
    parser = get_parser()
    # Parse the arguments and pull them out into local variables for ease of future refactor
    args = parser.parse_args()
    # Switch through possible configs at runtime.
    if args.use_finetuning_config:
        config_class = MaxTokiMultitaskFineTuneConfig
    else:
        config_class = MaxTokiConfig
    print("Using config class:", config_class)
    main(
        tokenizer_path=args.tokenizer_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        seq_length=args.seq_length,
        result_dir=args.result_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_dataset_workers=args.num_dataset_workers,
        micro_batch_size=args.micro_batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        create_checkpoint_callback=args.create_checkpoint_callback,
        config_class=config_class,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        log_every_n_steps=args.log_every_n_steps,
        aligned_megatron_ddp=args.aligned_megatron_ddp,
        recompilation_check=args.recompilation_check,
        create_tensorboard_logger=args.create_tensorboard_logger,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        additive_penalty=args.additive_penalty,
        label_scalar=args.label_scalar,
        timelapse_loss=args.timelapse_loss,
        hf_model_path=args.hf_model_path,
    )


if __name__ == "__main__":
    entrypoint()
