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

"""Unit tests for generate_utils.py.

These tests mock DynamicInferenceContext and a minimal model stub so they run
without a GPU and without real checkpoints.  Their purpose is to validate the
generate logic and lock in the Megatron DynamicInferenceContext API contract so
that a version bump that changes the interface (e.g. CVE-2025-23360 / NeMo
2.7.2 which removed keyword args from add_request) will be caught immediately.
"""

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams

from bionemo.maxtoki.generate_utils import maxtoki_generate_predict_step, setup_default_sampling_mask


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_model(tokenizer, num_layers=2, hidden=64, kv_channels=8, num_query_groups=2):
    model = MagicMock()
    model.tokenizer = tokenizer
    model.config.num_layers = num_layers
    model.config.kv_channels = kv_channels
    model.config.num_query_groups = num_query_groups
    model.config.tensor_model_parallel_size = 1
    return model


def _make_batch(batch_size, seq_len, eoq_id, device="cpu"):
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    # Place <eoq> at mid-point so truncation has something to do
    eoq_pos = seq_len // 2
    tokens[:, eoq_pos] = eoq_id
    return {"tokens": tokens}


# ---------------------------------------------------------------------------
# setup_default_sampling_mask
# ---------------------------------------------------------------------------


def test_setup_default_sampling_mask_excludes_eos_bos(tokenizer):
    mask = setup_default_sampling_mask(tokenizer)
    # <eos> and <bos> must NOT be in the mask (they are valid generation tokens)
    assert tokenizer.special_tokens["<eos>"] not in mask
    assert tokenizer.special_tokens["<bos>"] not in mask


def test_setup_default_sampling_mask_includes_other_specials(tokenizer):
    mask = setup_default_sampling_mask(tokenizer)
    # <pad>, <mask>, <boq>, <eoq> must be blocked
    for name in ("<pad>", "<mask>", "<boq>", "<eoq>"):
        assert tokenizer.special_tokens[name] in mask, f"{name} should be in the sampling mask"


def test_setup_default_sampling_mask_includes_numeric_tokens(tokenizer):
    mask = setup_default_sampling_mask(tokenizer)
    for tid in tokenizer.numeric_tokens.values():
        assert tid in mask, f"Numeric token {tid} should be in the sampling mask"


# ---------------------------------------------------------------------------
# maxtoki_generate_predict_step — empty batch fast-path
# ---------------------------------------------------------------------------


def test_generate_predict_step_empty_batch_returns_empty(tokenizer):
    model = _make_model(tokenizer)
    result = maxtoki_generate_predict_step(model, {})
    assert result == {}


# ---------------------------------------------------------------------------
# maxtoki_generate_predict_step — DynamicInferenceContext API contract
#
# We mock DynamicInferenceContext so no GPU allocation happens, then assert
# that add_request is called with a DynamicInferenceRequest (not bare kwargs).
# This is the regression test for the NeMo 2.7.2 API change.
# ---------------------------------------------------------------------------


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_add_request_called_with_dynamic_inference_request(MockContext, tokenizer):
    """add_request must receive a DynamicInferenceRequest, not bare keyword args."""
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False  # terminate immediately

    batch_size = 2
    seq_len = 16
    eoq_id = tokenizer.special_tokens["<eoq>"]
    batch = _make_batch(batch_size, seq_len, eoq_id)

    model = _make_model(tokenizer)
    max_tokens = 4

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=max_tokens)

    assert ctx.add_request.call_count == batch_size
    for c in ctx.add_request.call_args_list:
        args, kwargs = c
        # Must be called as add_request(req) or add_request(req, ...) — positional
        assert len(args) >= 1, "add_request must receive a positional DynamicInferenceRequest argument"
        req = args[0]
        assert isinstance(req, DynamicInferenceRequest), (
            f"add_request must be called with a DynamicInferenceRequest, got {type(req)}"
        )


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_add_request_num_tokens_to_generate_in_sampling_params(MockContext, tokenizer):
    """num_tokens_to_generate must be forwarded via SamplingParams on the request."""
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False

    eoq_id = tokenizer.special_tokens["<eoq>"]
    batch = _make_batch(1, 16, eoq_id)
    model = _make_model(tokenizer)
    max_tokens = 7

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=max_tokens)

    req = ctx.add_request.call_args[0][0]
    assert req.sampling_params is not None
    assert req.sampling_params.num_tokens_to_generate == max_tokens


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_add_request_prompt_tokens_match_input(MockContext, tokenizer):
    """prompt_tokens on the request must match the input token tensor."""
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False

    eoq_id = tokenizer.special_tokens["<eoq>"]
    seq_len = 16
    batch = _make_batch(1, seq_len, eoq_id)
    model = _make_model(tokenizer)

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=4)

    req = ctx.add_request.call_args[0][0]
    assert req.prompt_tokens is not None
    assert req.prompt_tokens.shape[0] == seq_len


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_add_request_prompt_truncated_when_using_pretrain_dataset(MockContext, tokenizer):
    """When using_pretrain_dataset=True, prompt must be truncated at <eoq>."""
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False

    eoq_id = tokenizer.special_tokens["<eoq>"]
    seq_len = 20
    eoq_pos = 8  # place <eoq> at position 8 → truncated length should be 9
    batch = {"tokens": torch.zeros(1, seq_len, dtype=torch.long)}
    batch["tokens"][0, eoq_pos] = eoq_id

    model = _make_model(tokenizer)

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=4, using_pretrain_dataset=True)

    req = ctx.add_request.call_args[0][0]
    assert req.prompt_tokens.shape[0] == eoq_pos + 1, (
        f"Expected truncated length {eoq_pos + 1}, got {req.prompt_tokens.shape[0]}"
    )


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_correct_number_of_requests_added(MockContext, tokenizer):
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False

    eoq_id = tokenizer.special_tokens["<eoq>"]
    batch_size = 3
    batch = _make_batch(batch_size, 16, eoq_id)
    model = _make_model(tokenizer)

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=4)

    assert ctx.add_request.call_count == batch_size


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_current_input_and_position_ids_called_not_split_methods(MockContext, tokenizer):
    """generate loop must use current_input_and_position_ids(), not the removed
    current_input_ids() / current_position_ids() methods."""
    ctx = MockContext.return_value
    # One iteration then stop
    ctx.has_unfinished_requests.side_effect = [True, False]
    ctx.current_input_and_position_ids.return_value = (
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    ctx.request_ids = torch.tensor([0])
    ctx.paused_request_count = 0
    ctx.total_request_count = 1
    ctx.get_active_sequence_lengths.return_value = torch.tensor([1])
    ctx.get_max_sequence_lengths.return_value = torch.tensor([8])

    eoq_id = tokenizer.special_tokens["<eoq>"]
    batch = _make_batch(1, 8, eoq_id)
    model = _make_model(tokenizer)
    vocab_size = tokenizer.vocab_size
    model.forward.return_value = {"lm_outputs": torch.zeros(1, 1, vocab_size)}

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=4)

    assert ctx.current_input_and_position_ids.called, (
        "current_input_and_position_ids() must be called each generation step"
    )
    assert not hasattr(ctx, "current_input_ids") or not ctx.current_input_ids.called, (
        "current_input_ids() is removed in NeMo 2.7.2 and must not be called"
    )
    assert not hasattr(ctx, "current_position_ids") or not ctx.current_position_ids.called, (
        "current_position_ids() is removed in NeMo 2.7.2 and must not be called"
    )


@patch("bionemo.maxtoki.generate_utils.DynamicInferenceContext")
def test_labels_removed_from_batch(MockContext, tokenizer):
    """Labels key must be stripped from the batch before generation."""
    ctx = MockContext.return_value
    ctx.has_unfinished_requests.return_value = False

    eoq_id = tokenizer.special_tokens["<eoq>"]
    batch = _make_batch(1, 16, eoq_id)
    batch["labels"] = batch["tokens"].clone()
    model = _make_model(tokenizer)

    maxtoki_generate_predict_step(model, batch, max_tokens_to_generate=4)

    assert "labels" not in batch
