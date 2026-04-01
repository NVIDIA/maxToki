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

### Run with:
# apptainer exec \
#     --nv \
#     --pwd "/workspace/bionemo2" \
#     --bind /rootdir/cellformer/results:/workspace/results \
#     --bind /rootdir/bionemo-framework-private:/workspace/bionemo2 \
#     --env TMPDIR=/tmp/ \
#     --env NUMBA_CACHE_DIR=/tmp/ \
#     --env PYTHONPATH=/workspace/bionemo2 \
#     /rootdir/cellformer_bionemo/bionemo_dev-bionemo2-7022a8aae0c9eae5c30fab8ab529b57019320f70.sif \
#     python /workspace/bionemo2/nemo_convert.py

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, override

import lightning.pytorch as pl
import torch
from nemo.collections.llm.gpt.model.llama import HFLlamaExporter
from nemo.collections.llm.gpt.model.ssm import AutoModelForCausalLM


if TYPE_CHECKING:
    from transformers import LlamaForCausalLM

from bionemo.maxtoki.datamodule import MaxTokiDataModule
from bionemo.maxtoki.tokenizer import MaxTokiPreCollator, MaxTokiTokenizer


logger = logging.getLogger(__name__)


class _ExportPreCollator(MaxTokiPreCollator):
    @property
    def bos_id(self):
        return self.token_dictionary.get("<bos>", self.token_dictionary["<cls>"])

    @property
    def eos_id(self):
        return self.token_dictionary["<eos>"]


class HFMaxTokiExporter(HFLlamaExporter):
    def __init__(self, model_path: Path, token_dictionary_path: Path):
        super().__init__(model_path)
        self.token_dictionary_path = token_dictionary_path

    @override
    def convert_state(self, source, target, source_config=None):
        """Convert NeMo state dict to HF format, handling both TE and SDPA layer norm layouts.

        With TransformerEngine (TE), layer norms are fused into the QKV linear:
            decoder.layers.*.self_attention.linear_qkv.layer_norm_weight
        With SDPA (non-TE), layer norms are separate modules in memory:
            decoder.layers.*.input_layernorm.weight

        NeMo's parent convert_state only maps the TE format. For SDPA models we remap
        keys to match the TE convention before delegating to the parent.
        """
        from megatron.core.transformer.module import MegatronModule
        from nemo.lightning.io.state import _ModelState

        # Mirror apply_transforms' unwrapping logic to get the actual source state dict.
        _src = source.module if (hasattr(source, "module") and isinstance(source.module, MegatronModule)) else source
        src_state = _src.state_dict()

        if any(".input_layernorm.weight" in k for k in src_state):
            patched = {}
            for k, v in src_state.items():
                if ".input_layernorm.weight" in k:
                    patched[k.replace(".input_layernorm.weight", ".self_attention.linear_qkv.layer_norm_weight")] = v
                elif ".pre_mlp_layernorm.weight" in k:
                    patched[k.replace(".pre_mlp_layernorm.weight", ".mlp.linear_fc1.layer_norm_weight")] = v
                else:
                    patched[k] = v
            source = _ModelState(patched, getattr(_src, "config", source_config))

        return super().convert_state(source, target, source_config)

    def init(self, dtype=torch.bfloat16) -> "LlamaForCausalLM":
        """Initialize a HF LlamaForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            LlamaForCausalLM: Initialized HF Llama model
        """
        from transformers import LlamaForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return LlamaForCausalLM(self.config)

    @override
    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF model
        """
        source, _ = self.nemo_load(str(self))
        from nemo.collections.llm.gpt.model.base import torch_dtype_from_mcore_config

        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            logging.warning("Skipping tokenizer save.")
            pass
        except Exception as e:
            logging.warning("Failed to save tokenizer")
            raise e

        logging.info(f"Saved model to {output_path}")
        return output_path

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            with open(self.token_dictionary_path) as f:
                token_dictionary = json.load(f)
            self._tokenizer = MaxTokiTokenizer(token_dictionary=token_dictionary)
        return self._tokenizer

    def nemo_load(
        self, path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True
    ) -> Tuple[pl.LightningModule, pl.Trainer]:
        """Loads a model from the specified path.

        Args:
            path (Path): The path from which the model will be loaded.
            trainer (Optional[pl.Trainer]): The trainer to be used, if not provided a new one will be created.
            cpu (bool): If True, the model will be loaded with a CPU-focused strategy.

        Returns:
        -------
            Tuple[pl.LightningModule, pl.Trainer]: The loaded model and the trainer configured with the model.
        """
        from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib
        from nemo.lightning.io.api import load_context

        model = load_context(path, subpath="model")

        model.tokenizer = self.tokenizer

        # disable FP8 model loading for LoRA export
        # (base model loaded in FP8 during training but can be loaded in BF16 during export)
        # FP8 SFT model export is not supported
        model.config.fp8 = None
        model.config.fp8_param = False

        # skip initialization since a checkpoint is loaded in this function
        model.config.perform_initialization = False

        is_peft_ckpt = model.model_transform is not None
        callbacks = []
        if is_peft_ckpt:
            callbacks.append(model.model_transform)

        _trainer = trainer or Trainer(
            devices=1,
            accelerator="cpu" if cpu else "gpu",
            strategy=MegatronStrategy(ddp="pytorch", setup_optimizers=False),
            callbacks=callbacks,
            max_epochs=1,
            max_steps=10,
        )

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()
        # TODO: Fix cpu initialization
        if not model.state_dict():
            if cpu:
                # TODO: Make this more generic
                with _strategy_lib.megatron_cpu_init_context(model.config):
                    model.configure_model()
            else:
                model.configure_model()

        _trainer.strategy.setup(_trainer)
        if is_peft_ckpt:
            from nemo.lightning.io.pl import ckpt_to_weights_subdir

            model.trainer = _trainer
            model = model.model_transform(model)
            load_path = ckpt_to_weights_subdir(path, is_saving=False)
            sharded_sd_metadata = _trainer.strategy.unwrapped_checkpoint_io.load_content_metadata(load_path)
            adapter_sharded_state_dict = {
                k: v
                for k, v in _trainer.strategy.megatron_parallel.sharded_state_dict(
                    metadata=sharded_sd_metadata
                ).items()
                if ".adapter." in k
            }
            adapter_state = _trainer.strategy.checkpoint_io.load_checkpoint(
                load_path, sharded_state_dict=adapter_sharded_state_dict
            )
            _trainer.strategy.load_model_state_dict(adapter_state, strict=False)
        else:
            _trainer.strategy.load_checkpoint(path)

        return model, _trainer


def load_model(model_path: Path, token_dictionary_path: Path):
    return HFMaxTokiExporter(model_path, token_dictionary_path=token_dictionary_path)


def convert_model(model: HFLlamaExporter, output_path: Path):
    model.apply(output_path)
    logger.warning("Tokenizer not needed to save for MaxToki model conversion.")
    return model


def get_parser():
    p = argparse.ArgumentParser(description="Convert a NeMo MaxToki checkpoint to HuggingFace format.")
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the pretrained NeMo checkpoint directory.",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory where the converted HuggingFace model will be saved.",
    )
    p.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Path to token_dictionary.pkl used by the model.",
    )
    p.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run NeMo vs HF logit comparison on sample data after conversion.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to HF dataset for sanity check (required if --sanity-check).",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of examples to use for sanity check (default: 8).",
    )
    return p


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.sanity_check and args.data_path is None:
        parser.error("--data-path is required when --sanity-check is set")
    return args


def main():
    args = parse_args()
    assert args.model_path.exists(), f"Model path {args.model_path} does not exist"
    assert args.tokenizer_path.exists(), f"Tokenizer path {args.tokenizer_path} does not exist"

    exporter = load_model(args.model_path, args.tokenizer_path)

    if args.sanity_check:
        # Load the NeMo model once and reuse it for both conversion and sanity check.
        # Calling nemo_load twice in the same process re-initializes Megatron, which fails.
        # Use cpu=False so TransformerEngine layers have CUDA available during the forward pass.
        assert args.data_path.exists(), f"Data path {args.data_path} does not exist"
        nemo_model, nemo_trainer = exporter.nemo_load(str(exporter), cpu=False)
        from nemo.collections.llm.gpt.model.base import torch_dtype_from_mcore_config
        target = exporter.init(torch_dtype_from_mcore_config(nemo_model.config))
        target = exporter.convert_state(nemo_model, target)
        target = target.cpu()
        target.save_pretrained(args.output_path)
        logging.info(f"Saved model to {args.output_path}")

        hf_model = AutoModelForCausalLM.from_pretrained(args.output_path)
        data = load_data(args.data_path, args.tokenizer_path, nemo_trainer)
        mae, max_abs = sanity_check(nemo_model, data, hf_model, nemo_trainer, num_examples=args.num_examples)
        if mae > 0.05:
            raise SystemExit(
                f"Logit MAE {mae:.6f} (max_abs={max_abs:.6f}) exceeds threshold 0.05 — conversion may be incorrect"
            )
    else:
        convert_model(exporter, args.output_path)

    print(f"Model saved to {args.output_path}")


def load_data(data_path, tokenizer_path, trainer):
    with open(tokenizer_path) as f:
        token_dictionary = json.load(f)
    tokenizer = MaxTokiTokenizer(token_dictionary=token_dictionary)

    data = MaxTokiDataModule(
        tokenizer=tokenizer,
        train_dataset_path=None,
        val_dataset_path=None,
        test_dataset_path=None,
        predict_dataset_path=data_path,
        seq_length=4096,  # just for sanity
        micro_batch_size=1,
        global_batch_size=1,
        rampup_batch_size=None,
        seed=42,
        pin_memory=False,
        # Since we arent actually training, we want to enable the collator that isnt so picky.
        pretrain=True,
        # One batch should be enough.
        limit_predict_batches_to_n=16,
    )
    data.trainer = trainer
    data.setup()
    return data


def sanity_check(
    model: torch.nn.Module, data: MaxTokiDataModule, hf_model, trainer, num_examples: int = 8
) -> tuple[float, float]:
    """Compare NeMo and HF model logits on a small set of examples.

    The NeMo model must have been loaded with nemo_load(cpu=False) so that TransformerEngine
    layers have CUDA available. The HF model and input tensors are moved to the same device.

    Returns:
        (mae, max_abs_err) averaged over num_examples.
    """
    nemo_model = model.eval()
    device = next(nemo_model.parameters()).device
    hf_model = hf_model.to(device).eval()
    nemo_module = nemo_model.module
    vocab_size = hf_model.config.vocab_size

    num_examples = min(num_examples, len(data._predict_ds))
    mae_list = []
    max_abs_list = []

    for i in range(num_examples):
        input_ids = torch.tensor([data._predict_ds[i]["input_ids"]]).to(device)
        position_ids = torch.arange(len(input_ids[0])).unsqueeze(0).to(device)

        with torch.no_grad():
            # attention_mask=None: Megatron uses its default causal mask (attn_mask_type="causal").
            nemo_output = nemo_module(input_ids, position_ids, attention_mask=None)
            # Pretrain models (MCoreGPTModel) return a Tensor; finetune models return a dict.
            if isinstance(nemo_output, dict):
                nemo_logits = nemo_output["lm_outputs"].float()
            else:
                nemo_logits = nemo_output.float()
            nemo_logits = nemo_logits[:, :, :vocab_size]

            hf_output = hf_model(input_ids, position_ids=position_ids, use_cache=False)
            hf_logits = hf_output.logits.float()

        diff = (nemo_logits - hf_logits).abs()
        mae_list.append(diff.mean().item())
        max_abs_list.append(diff.max().item())

    mae = sum(mae_list) / len(mae_list)
    max_abs = max(max_abs_list)
    print(f"NeMo vs HF logits: MAE={mae:.6f}, max_abs_err={max_abs:.6f} (over {num_examples} examples)")
    return mae, max_abs


if __name__ == "__main__":
    main()
