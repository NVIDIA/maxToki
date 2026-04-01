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

import shlex
import subprocess
import sys

import torch
import pytest

from bionemo.testing.subprocess_utils import run_command_in_subprocess

from .conftest import PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH, FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH


# Real checkpoints from the HF→NeMo conversion + subsequent finetune.
# Tests using these paths are skipped if the files are not present.
_CONVERTED_CKPT = "/workspaces/bionemo-framework-private/results/hf-weight-conversion/dev/checkpoints/epoch=0-val_loss=0.00-step=0-consumed_samples=0-last-v3"
_FINETUNED_CKPT = "/workspaces/bionemo-framework-private/results/finetune-hfconv/maxtoki/dev/checkpoints/epoch=0-val_loss=3.07-step=499-consumed_samples=500.0-last"
# Known-harmless missing key: TE FP8 state absent in non-FP8 checkpoints.
_EXPECTED_MISSING_KEYS = {"module.decoder.layers.self_attention.core_attention._extra_state"}


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


def _build_pretrain_cmd(data_path, tokenizer_path, result_dir, num_steps=4, model_args=TINY_MODEL_ARGS):
    return (
        f"python -m bionemo.maxtoki.train "
        f"--train-data-path {data_path} "
        f"--val-data-path {data_path} "
        f"--test-data-path {data_path} "
        f"--tokenizer-path {tokenizer_path} "
        f"--result-dir {result_dir} "
        f"--experiment-name {EXPERIMENT_NAME} "
        f"--num-steps {num_steps} "
        f"--val-check-interval {num_steps + 1} "
        f"--limit-val-batches 1 "
        f"--pretrain "
        f"{model_args}"
    )


def _build_finetune_cmd(data_path, tokenizer_path, result_dir, initial_ckpt_path, num_steps=4):
    return (
        f"python -m bionemo.maxtoki.train "
        f"--train-data-path {data_path} "
        f"--val-data-path {data_path} "
        f"--test-data-path {data_path} "
        f"--tokenizer-path {tokenizer_path} "
        f"--result-dir {result_dir} "
        f"--experiment-name {EXPERIMENT_NAME} "
        f"--num-steps {num_steps} "
        f"--val-check-interval {num_steps + 1} "
        f"--limit-val-batches 1 "
        f"--use-finetuning-config "
        f"--initial-ckpt-path {initial_ckpt_path} "
        f"{TINY_MODEL_ARGS}"
    )


def _build_predict_cmd(data_path, tokenizer_path, ckpt_path, output_dir):
    return (
        f"python -m bionemo.maxtoki.predict "
        f"--data-path {data_path} "
        f"--tokenizer-path {tokenizer_path} "
        f"--initial-ckpt-path {ckpt_path} "
        f"--output-dir {output_dir} "
        f"--seq-length 8192 "
        f"--micro-batch-size 1 "
        f"--precision bf16-mixed "
        f"--generate-next-cell "
        f"--naive-benchmarking-only "
        f"--max-tokens-to-generate 8 "
        f"--limit-predict-batches-to-n 4 "
        f"--write-interval batch"
    )


def _build_regression_predict_cmd(data_path, tokenizer_path, ckpt_path, output_dir):
    return (
        f"python -m bionemo.maxtoki.predict "
        f"--data-path {data_path} "
        f"--tokenizer-path {tokenizer_path} "
        f"--initial-ckpt-path {ckpt_path} "
        f"--output-dir {output_dir} "
        f"--seq-length 8192 "
        f"--micro-batch-size 1 "
        f"--precision bf16-mixed "
        f"--limit-predict-batches-to-n 4 "
        f"--write-interval batch"
    )


def _find_checkpoint(result_dir, experiment_name=EXPERIMENT_NAME):
    ckpt_root = result_dir / experiment_name / "dev" / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints directory at {ckpt_root}")
    candidates = sorted(ckpt_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_root}")
    return candidates[0]


def _validate_prediction_files(prediction_files, mode="regression"):
    assert len(prediction_files) > 0, "No prediction files found"
    nonempty_count = 0
    for pf in prediction_files:
        data = torch.load(pf, weights_only=True)
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"

        if mode == "regression":
            for key in ("regression_preds", "timelapse_token_preds"):
                assert key in data, f"Missing key '{key}' in {pf.name}"
                t = data[key]
                if t.numel() == 0:
                    continue
                nonempty_count += 1
                assert not torch.isnan(t).any(), f"'{key}' contains NaN in {pf.name}"
                assert not torch.isinf(t).any(), f"'{key}' contains Inf in {pf.name}"
                if t.numel() > 1:
                    assert not (t == t.flatten()[0]).all(), f"'{key}' is constant in {pf.name}"
        elif mode == "generation":
            # Generation uses PassthroughEverythingLossReduction which returns a list,
            # so PredictionWriter wraps it as {"batch_idx": ..., "prediction": [list_of_dicts]}.
            assert "prediction" in data, f"Missing 'prediction' key in {pf.name}"
            pred_list = data["prediction"]
            assert isinstance(pred_list, list) and len(pred_list) > 0, f"Empty prediction list in {pf.name}"
            for entry in pred_list:
                assert isinstance(entry, dict), f"Expected dict entry, got {type(entry)}"
                if entry == {}:
                    continue
                nonempty_count += 1
                for key in ("generated_tokens", "lengths", "full_sequence"):
                    assert key in entry, f"Missing key '{key}' in generation output in {pf.name}"
                lengths = entry["lengths"]
                assert lengths.numel() > 0, f"'lengths' is empty in {pf.name}"
                assert (lengths > 0).all(), f"Some sequences have zero length in {pf.name}"
                gen = entry["generated_tokens"]
                assert gen.numel() > 0, f"'generated_tokens' is empty in {pf.name}"
                assert not torch.isnan(gen.float()).any(), f"'generated_tokens' contains NaN in {pf.name}"

    assert nonempty_count > 0, "All prediction batches were empty"


def _skip_if_no_pretrain_data():
    if not PRETRAIN_DATA_PATH.exists():
        pytest.skip(f"Pretrain data not found at {PRETRAIN_DATA_PATH}")
    if not PRETRAIN_TOKENIZER_PATH.exists():
        pytest.skip(f"Pretrain tokenizer not found at {PRETRAIN_TOKENIZER_PATH}")


def _skip_if_no_finetune_data():
    if not FINETUNE_DATA_PATH.exists():
        pytest.skip(f"Finetune data not found at {FINETUNE_DATA_PATH}")
    if not FINETUNE_TOKENIZER_PATH.exists():
        pytest.skip(f"Finetune tokenizer not found at {FINETUNE_TOKENIZER_PATH}")


@pytest.mark.slow
def test_pretrain_completes(tmp_path):
    _skip_if_no_pretrain_data()
    result_dir = tmp_path / "pretrain_results"
    cmd = _build_pretrain_cmd(PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH, result_dir, num_steps=4)
    run_command_in_subprocess(cmd, str(tmp_path))
    ckpt = _find_checkpoint(result_dir)
    assert ckpt.exists()


@pytest.mark.slow
def test_pretrain_then_finetune(tmp_path):
    _skip_if_no_pretrain_data()
    _skip_if_no_finetune_data()

    # Pretrain
    pretrain_dir = tmp_path / "pretrain_results"
    cmd = _build_pretrain_cmd(PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH, pretrain_dir, num_steps=4)
    run_command_in_subprocess(cmd, str(tmp_path))
    pretrain_ckpt = _find_checkpoint(pretrain_dir)

    # Finetune
    finetune_dir = tmp_path / "finetune_results"
    cmd = _build_finetune_cmd(
        FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, finetune_dir, pretrain_ckpt, num_steps=4
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    finetune_ckpt = _find_checkpoint(finetune_dir)
    assert finetune_ckpt.exists()


@pytest.mark.slow
def test_pretrain_then_finetune_then_generate(tmp_path):
    _skip_if_no_pretrain_data()
    _skip_if_no_finetune_data()

    # Pretrain
    pretrain_dir = tmp_path / "pretrain_results"
    cmd = _build_pretrain_cmd(PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH, pretrain_dir, num_steps=4)
    run_command_in_subprocess(cmd, str(tmp_path))
    pretrain_ckpt = _find_checkpoint(pretrain_dir)

    # Finetune
    finetune_dir = tmp_path / "finetune_results"
    cmd = _build_finetune_cmd(
        FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, finetune_dir, pretrain_ckpt, num_steps=4
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    finetune_ckpt = _find_checkpoint(finetune_dir)

    # Predict / Generate (uses finetune data since predict collate expects <eoq> tokens)
    predict_output = tmp_path / "predictions"
    cmd = _build_predict_cmd(
        FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, finetune_ckpt, predict_output
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    prediction_files = sorted(predict_output.glob("predictions__rank_*.pt"))
    _validate_prediction_files(prediction_files, mode="generation")


@pytest.mark.slow
def test_pretrain_then_finetune_then_regression_predict(tmp_path):
    """Full pipeline: pretrain → finetune → regression predict (TimeBetweenCells)."""
    _skip_if_no_pretrain_data()
    _skip_if_no_finetune_data()

    pretrain_dir = tmp_path / "pretrain_results"
    cmd = _build_pretrain_cmd(PRETRAIN_DATA_PATH, PRETRAIN_TOKENIZER_PATH, pretrain_dir, num_steps=4)
    run_command_in_subprocess(cmd, str(tmp_path))
    pretrain_ckpt = _find_checkpoint(pretrain_dir)

    finetune_dir = tmp_path / "finetune_results"
    cmd = _build_finetune_cmd(
        FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, finetune_dir, pretrain_ckpt, num_steps=4
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    finetune_ckpt = _find_checkpoint(finetune_dir)

    predict_output = tmp_path / "regression_predictions"
    cmd = _build_regression_predict_cmd(
        FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, finetune_ckpt, predict_output
    )
    run_command_in_subprocess(cmd, str(tmp_path))
    prediction_files = sorted(predict_output.glob("predictions__rank_*.pt"))
    _validate_prediction_files(prediction_files, mode="regression")


def _run_predict_capture_stderr(cmd, cwd):
    """Run a predict command and return (stdout, stderr). Raises on non-zero exit."""
    import os
    from lightning.fabric.plugins.environments.lightning import find_free_network_port
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(find_free_network_port())
    result = subprocess.run(shlex.split(cmd), shell=False, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")
    assert result.returncode == 0, f"Command failed: {cmd}"
    return result.stdout, result.stderr


def _assert_no_unexpected_missing_keys(stderr):
    import re
    matches = re.findall(r"Missing keys.*?:\s*\{([^}]+)\}", stderr, re.DOTALL)
    for match in matches:
        keys = {k.strip().strip("'\"") for k in match.split(",")}
        unexpected = keys - _EXPECTED_MISSING_KEYS
        assert not unexpected, f"Unexpected missing checkpoint keys: {unexpected}"


@pytest.mark.slow
def test_deterministic_generate(tmp_path):
    """Generate against the real finetuned checkpoint — validates output sanity and no unexpected missing keys."""
    from pathlib import Path
    if not Path(_FINETUNED_CKPT).exists():
        pytest.skip(f"Finetuned checkpoint not found: {_FINETUNED_CKPT}")
    if not FINETUNE_DATA_PATH.exists():
        pytest.skip(f"Finetune data not found: {FINETUNE_DATA_PATH}")

    predict_output = tmp_path / "predictions"
    cmd = _build_predict_cmd(FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, _FINETUNED_CKPT, predict_output)
    stdout, stderr = _run_predict_capture_stderr(cmd, str(tmp_path))

    _assert_no_unexpected_missing_keys(stderr)
    prediction_files = sorted(predict_output.glob("predictions__rank_*.pt"))
    _validate_prediction_files(prediction_files, mode="generation")


@pytest.mark.slow
def test_deterministic_regression_predict(tmp_path):
    """Regression predict against the real finetuned checkpoint — validates output sanity and no unexpected missing keys."""
    from pathlib import Path
    if not Path(_FINETUNED_CKPT).exists():
        pytest.skip(f"Finetuned checkpoint not found: {_FINETUNED_CKPT}")
    if not FINETUNE_DATA_PATH.exists():
        pytest.skip(f"Finetune data not found: {FINETUNE_DATA_PATH}")

    predict_output = tmp_path / "regression_predictions"
    cmd = _build_regression_predict_cmd(FINETUNE_DATA_PATH, FINETUNE_TOKENIZER_PATH, _FINETUNED_CKPT, predict_output)
    stdout, stderr = _run_predict_capture_stderr(cmd, str(tmp_path))

    _assert_no_unexpected_missing_keys(stderr)
    prediction_files = sorted(predict_output.glob("predictions__rank_*.pt"))
    _validate_prediction_files(prediction_files, mode="regression")

