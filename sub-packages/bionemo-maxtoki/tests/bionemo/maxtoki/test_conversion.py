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

import pytest
import torch

from bionemo.testing.subprocess_utils import run_command_in_subprocess

from .conftest import PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH


TINY_MODEL_ARGS = (
    "--num-layers 2 "
    "--hidden-size 256 "
    "--ffn-hidden-size 512 "
    "--num-attention-heads 8 "
    "--seq-length 128 "
    "--micro-batch-size 2 "
    "--precision bf16-mixed "
    "--rope-scaling-factor 1.0 "
    "--num-dataset-workers 0 "
    "--output-weights separate"
)

EXPERIMENT_NAME = "cellformer_test"


def _pretrain_tiny_model(tmp_path):
    result_dir = tmp_path / "pretrain_results"
    cmd = (
        f"python -m bionemo.maxtoki.train "
        f"--train-data-path {PRETRAIN_DATA_PATH} "
        f"--val-data-path {PRETRAIN_DATA_PATH} "
        f"--test-data-path {PRETRAIN_DATA_PATH} "
        f"--tokenizer-path {PRETRAIN_TOKENIZER_PATH} "
        f"--result-dir {result_dir} "
        f"--experiment-name {EXPERIMENT_NAME} "
        f"--num-steps 4 "
        f"--val-check-interval 5 "
        f"--limit-val-batches 1 "
        f"--pretrain "
        f"{TINY_MODEL_ARGS}"
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    ckpt_root = result_dir / EXPERIMENT_NAME / "dev" / "checkpoints"
    candidates = sorted(ckpt_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    assert candidates, f"No checkpoints found in {ckpt_root}"
    return candidates[0]


def _convert_bionemo_to_hf(ckpt_path, hf_output, tmp_path):
    cmd = (
        f"python -m bionemo.maxtoki.export_hf "
        f"--model-path {ckpt_path} "
        f"--output-path {hf_output} "
        f"--tokenizer-path {PRETRAIN_TOKENIZER_PATH}"
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    assert (hf_output / "config.json").exists(), "HF config.json not created"
    return hf_output


def _compare_hf_logits(hf_path_1, hf_path_2, mae_threshold=0.05):
    """Load two HF models and assert their logits are close on a fixed synthetic input."""
    from transformers import AutoModelForCausalLM

    model1 = AutoModelForCausalLM.from_pretrained(hf_path_1).cpu().eval()
    model2 = AutoModelForCausalLM.from_pretrained(hf_path_2).cpu().eval()

    vocab_size = model1.config.vocab_size
    gen = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 32), generator=gen)
    position_ids = torch.arange(32).unsqueeze(0)

    with torch.no_grad():
        logits1 = model1(input_ids, position_ids=position_ids, use_cache=False).logits.float()
        logits2 = model2(input_ids, position_ids=position_ids, use_cache=False).logits.float()

    mae = (logits1 - logits2).abs().mean().item()
    assert mae < mae_threshold, f"HF logit MAE {mae:.6f} exceeds threshold {mae_threshold}"


def _skip_if_no_pretrain_data():
    if not PRETRAIN_DATA_PATH.exists():
        pytest.skip(f"Pretrain data not found at {PRETRAIN_DATA_PATH}")
    if not PRETRAIN_TOKENIZER_PATH.exists():
        pytest.skip(f"Pretrain tokenizer not found at {PRETRAIN_TOKENIZER_PATH}")


def _pretrain_tiny_model_sdpa(tmp_path):
    """Pretrain a tiny model with explicit SDPA attention (non-TE separate layer norms)."""
    result_dir = tmp_path / "pretrain_sdpa_results"
    cmd = (
        f"python -m bionemo.maxtoki.train "
        f"--train-data-path {PRETRAIN_DATA_PATH} "
        f"--val-data-path {PRETRAIN_DATA_PATH} "
        f"--test-data-path {PRETRAIN_DATA_PATH} "
        f"--tokenizer-path {PRETRAIN_TOKENIZER_PATH} "
        f"--result-dir {result_dir} "
        f"--experiment-name {EXPERIMENT_NAME} "
        f"--num-steps 4 "
        f"--val-check-interval 5 "
        f"--limit-val-batches 1 "
        f"--pretrain "
        f"--use-sdpa "
        f"{TINY_MODEL_ARGS}"
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    ckpt_root = result_dir / EXPERIMENT_NAME / "dev" / "checkpoints"
    candidates = sorted(ckpt_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    assert candidates, f"No checkpoints found in {ckpt_root}"
    return candidates[0]


@pytest.mark.slow
def test_sdpa_bionemo_to_hf_export(tmp_path):
    """Export a checkpoint trained with SDPA (separate input_layernorm) to HF.

    SDPA models store layer norms as separate modules (input_layernorm.weight /
    pre_mlp_layernorm.weight) rather than fused into the QKV linear
    (linear_qkv.layer_norm_weight). This verifies that convert_state remaps
    these keys correctly so the export does not raise an IndexError.
    """
    _skip_if_no_pretrain_data()
    ckpt_path = _pretrain_tiny_model_sdpa(tmp_path)
    hf_output = tmp_path / "hf_sdpa_model"
    _convert_bionemo_to_hf(ckpt_path, hf_output, tmp_path)
    weight_files = list(hf_output.glob("*.safetensors")) + list(hf_output.glob("*.bin"))
    assert len(weight_files) > 0, "No weight files found in SDPA HF output"


@pytest.mark.slow
def test_bionemo_to_hf_conversion(tmp_path):
    """Convert a pretrained BioNeMo checkpoint to HF and verify logit equivalence."""
    _skip_if_no_pretrain_data()
    ckpt_path = _pretrain_tiny_model(tmp_path)

    hf_output = tmp_path / "hf_model"
    cmd = (
        f"python -m bionemo.maxtoki.export_hf "
        f"--model-path {ckpt_path} "
        f"--output-path {hf_output} "
        f"--tokenizer-path {PRETRAIN_TOKENIZER_PATH} "
        f"--sanity-check "
        f"--data-path {PRETRAIN_DATA_PATH} "
        f"--num-examples 2"
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    assert (hf_output / "config.json").exists(), "HF config.json not created"
    weight_files = list(hf_output.glob("*.safetensors")) + list(hf_output.glob("*.bin"))
    assert len(weight_files) > 0, "No weight files found in HF output"


@pytest.mark.slow
def test_hf_to_bionemo_conversion(tmp_path):
    """Round-trip BioNeMo -> HF -> BioNeMo, then verify the re-exported HF logits match the original."""
    _skip_if_no_pretrain_data()

    # Step 1: Pretrain a tiny BioNeMo model
    ckpt_path = _pretrain_tiny_model(tmp_path)

    # Step 2: Convert BioNeMo -> HF
    hf_output = tmp_path / "hf_model"
    _convert_bionemo_to_hf(ckpt_path, hf_output, tmp_path)

    # Step 3: Import HF -> BioNeMo via import_hf
    import_result_dir = tmp_path / "import_results"
    cmd = (
        f"python -m bionemo.maxtoki.import_hf "
        f"--hf-model-path {hf_output} "
        f"--train-data-path {PRETRAIN_DATA_PATH} "
        f"--val-data-path {PRETRAIN_DATA_PATH} "
        f"--test-data-path {PRETRAIN_DATA_PATH} "
        f"--tokenizer-path {PRETRAIN_TOKENIZER_PATH} "
        f"--result-dir {import_result_dir} "
        f"--experiment-name {EXPERIMENT_NAME} "
        f"--seq-length 128 "
        f"--num-steps 1 "
        f"--val-check-interval 2 "
        f"--limit-val-batches 1 "
        f"--micro-batch-size 2 "
        f"--num-dataset-workers 0 "
        f"--precision bf16-mixed "
        f"--output-weights separate "
        f"--rope-scale-factor 1.0"
    )
    run_command_in_subprocess(cmd, str(tmp_path))

    import_ckpt_root = import_result_dir / EXPERIMENT_NAME / "dev" / "checkpoints"
    assert import_ckpt_root.exists(), f"No checkpoints directory at {import_ckpt_root}"
    import_candidates = sorted(import_ckpt_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    assert import_candidates, f"No checkpoints found in {import_ckpt_root}"
    import_ckpt = import_candidates[0]

    # Step 4: Re-export the re-imported BioNeMo checkpoint -> HF
    hf_reimport_output = tmp_path / "hf_from_reimport"
    _convert_bionemo_to_hf(import_ckpt, hf_reimport_output, tmp_path)

    # Step 5: Compare HF B (from original pretrain) vs HF D (from re-imported BioNeMo).
    # import_hf runs 1 training step, so a small logit difference is expected; threshold is generous.
    _compare_hf_logits(hf_output, hf_reimport_output, mae_threshold=1.0)
