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

import json
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bionemo.maxtoki.import_regression_hf import MAXTOKI_RMS_NORM_EPS, build_regression_config
from bionemo.maxtoki.model import maxtoki_regression_predict_step
from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec


def _write_tiny_cf_model(path, num_layers=2, hidden=16, ffn=32, vocab=40, kv_heads=4):
    """Write a tiny cf-prefixed safetensors mimicking the HF aging model layout."""
    sd = {
        "cf.model.embed_tokens.weight": torch.zeros(vocab, hidden),
        "cf.lm_head.weight": torch.zeros(vocab, hidden),
        "cf.model.norm.weight": torch.zeros(hidden),
        "linear1.weight": torch.zeros(1, hidden),
        "linear1.bias": torch.zeros(1),
    }
    for i in range(num_layers):
        sd[f"cf.model.layers.{i}.input_layernorm.weight"] = torch.zeros(hidden)
        sd[f"cf.model.layers.{i}.post_attention_layernorm.weight"] = torch.zeros(hidden)
        sd[f"cf.model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"cf.model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"cf.model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"cf.model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"cf.model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(ffn, hidden)
        sd[f"cf.model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(ffn, hidden)
        sd[f"cf.model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, ffn)
    save_file(sd, str(path))


def test_build_regression_config_reads_shapes_and_hyperparams(tmp_path):
    """Config dims come from the safetensors; heads and rope come from the HF config.json; eps is pinned."""
    model_path = tmp_path / "model.safetensors"
    _write_tiny_cf_model(model_path, num_layers=3, hidden=16, ffn=32, vocab=40, kv_heads=4)

    hf_config = {
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "rope_theta": 12345.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "initializer_range": 0.02,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(hf_config))

    cfg = build_regression_config(model_path, cfg_path, seq_length=2048)

    assert cfg.num_layers == 3
    assert cfg.hidden_size == 16
    assert cfg.ffn_hidden_size == 32
    assert cfg.vocab_size == 40
    assert cfg.num_attention_heads == 8
    assert cfg.num_query_groups == 4
    assert cfg.kv_channels == 16 // 8  # head_dim = hidden / num_heads when not given
    assert cfg.rotary_base == 12345.0
    assert cfg.layernorm_epsilon == 1e-6
    # Standard RoPE (no llama3 scaling), SDPA forced, regression head not skipped on load.
    assert cfg.scale_factor == 1.0
    assert cfg.transformer_layer_spec is sdpa_layer_spec
    assert cfg.persist_layer_norm is False
    assert cfg.initial_ckpt_skip_keys_with_these_prefixes == []


def test_build_regression_config_respects_explicit_head_dim(tmp_path):
    """An explicit head_dim in the HF config overrides hidden/num_heads."""
    model_path = tmp_path / "model.safetensors"
    _write_tiny_cf_model(model_path, hidden=16)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 154, "rope_theta": 1e4, "rms_norm_eps": 1e-5}))
    cfg = build_regression_config(model_path, cfg_path, seq_length=128)
    assert cfg.kv_channels == 154


def _eps_config(tmp_path, **extra):
    model_path = tmp_path / "model.safetensors"
    _write_tiny_cf_model(model_path, hidden=16)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"num_attention_heads": 8, "num_key_value_heads": 8, "rope_theta": 1e4, **extra}))
    return model_path, cfg_path


@pytest.mark.parametrize("extra", [{"rms_norm_eps": 1e-5}, {"rms_norm_eps": 1e-6}, {}])
def test_build_regression_config_defaults_to_known_eps(tmp_path, extra):
    cfg = build_regression_config(*_eps_config(tmp_path, **extra), seq_length=128)
    assert cfg.layernorm_epsilon == MAXTOKI_RMS_NORM_EPS == 1e-6


@pytest.mark.parametrize("extra", [{"rms_norm_eps": 1e-6}, {}])
def test_build_regression_config_uses_given_eps(tmp_path, extra):
    cfg = build_regression_config(*_eps_config(tmp_path, **extra), seq_length=128, rms_norm_eps=1e-5)
    assert cfg.layernorm_epsilon == 1e-5


def test_build_regression_config_warns_when_config_eps_differs(tmp_path, monkeypatch):
    from bionemo.maxtoki import import_regression_hf

    warnings = []
    monkeypatch.setattr(import_regression_hf.logging, "warning", lambda msg, *a, **k: warnings.append(msg))
    build_regression_config(*_eps_config(tmp_path, rms_norm_eps=1e-5), seq_length=128)
    build_regression_config(*_eps_config(tmp_path, rms_norm_eps=1e-6), seq_length=128)
    assert len(warnings) == 1 and "rms_norm_eps=1e-05" in warnings[0]


def test_cli_rms_norm_eps():
    from bionemo.maxtoki.import_regression_hf import get_parser

    base = ["--model-safetensors", "m", "--hf-config", "c", "--tokenizer-path", "t", "--train-data-path", "d",
            "--val-data-path", "d", "--test-data-path", "d", "--result-dir", "r", "--seq-length", "128"]  # fmt: skip
    assert get_parser().parse_args(base).rms_norm_eps == MAXTOKI_RMS_NORM_EPS
    assert get_parser().parse_args([*base, "--rms-norm-eps", "1e-5"]).rms_norm_eps == 1e-5


def test_build_regression_config_rope_scaling_from_config(tmp_path):
    """RoPE is derived from the config: null -> no scaling; llama3 -> its factor/context length."""
    model_path = tmp_path / "model.safetensors"
    _write_tiny_cf_model(model_path, hidden=16)
    base = {"num_attention_heads": 8, "num_key_value_heads": 8, "rope_theta": 1e4, "rms_norm_eps": 1e-5, "max_position_embeddings": 4096}

    null_cfg = tmp_path / "null.json"
    null_cfg.write_text(json.dumps({**base, "rope_scaling": None}))
    c = build_regression_config(model_path, null_cfg, seq_length=128)
    assert c.scale_factor == 1.0
    assert c.old_context_len == 4096

    l3_cfg = tmp_path / "llama3.json"
    l3_cfg.write_text(
        json.dumps(
            {
                **base,
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 4.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 2048,
                },
            }
        )
    )
    c2 = build_regression_config(model_path, l3_cfg, seq_length=128)
    assert c2.scale_factor == 4.0
    assert c2.old_context_len == 2048
    assert c2.high_freq_factor == 4.0


class _FakeModel:
    """Minimal stand-in exposing forward_step + tokenizer for the predict-step test."""

    def __init__(self, regression_outputs, eoq_id):
        self._out = regression_outputs
        self.tokenizer = types.SimpleNamespace(token_dictionary={"<eoq>": eoq_id})

    def forward_step(self, batch):
        return {"regression_outputs": self._out}


def test_regression_predict_step_gathers_at_eoq_and_scales():
    """The predict step reads the head output at <eoq> and multiplies by label_scalar."""
    eoq_id = 99
    seq_len, batch = 6, 2
    # regression_outputs[s, b, 0] = s, so gathering at the eoq position returns that index.
    reg = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1).repeat(1, batch, 1)
    # tokens [B, S]; place <eoq> at positions 3 and 4 for the two batch elements.
    tokens = torch.zeros(batch, seq_len, dtype=torch.long)
    tokens[0, 3] = eoq_id
    tokens[1, 4] = eoq_id

    model = _FakeModel(reg, eoq_id)
    out = maxtoki_regression_predict_step(model, {"tokens": tokens}, eoq_token_id=eoq_id, label_scalar=1000.0)

    preds = out["regression_preds"].flatten().tolist()
    assert np.allclose(preds, [3 * 1000.0, 4 * 1000.0])


def test_regression_predict_step_uses_tokenizer_eoq_by_default():
    """When eoq_token_id is None, the step falls back to the tokenizer's <eoq> id."""
    eoq_id = 7
    reg = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
    tokens = torch.zeros(1, 5, dtype=torch.long)
    tokens[0, 2] = eoq_id
    model = _FakeModel(reg, eoq_id)
    out = maxtoki_regression_predict_step(model, {"tokens": tokens}, label_scalar=1.0)
    assert out["regression_preds"].flatten().item() == 2.0


def test_regression_predict_step_schema_parity_without_logits():
    """With no lm_outputs, the head step still emits both keys ([1, B]) — a -1 token sentinel."""
    eoq_id = 4
    reg = torch.arange(5, dtype=torch.float32).view(5, 1, 1).repeat(1, 2, 1)
    tokens = torch.zeros(2, 5, dtype=torch.long)
    tokens[0, 2] = eoq_id
    tokens[1, 3] = eoq_id
    out = maxtoki_regression_predict_step(_FakeModel(reg, eoq_id), {"tokens": tokens}, label_scalar=1.0)
    assert set(out) == {"regression_preds", "timelapse_token_preds"}
    assert out["regression_preds"].shape == (1, 2)
    assert out["timelapse_token_preds"].shape == (1, 2)
    assert out["timelapse_token_preds"].flatten().tolist() == [-1, -1]


def test_regression_predict_step_emits_timelapse_token_preds_from_logits():
    """When lm_outputs are present, timelapse_token_preds is the numeric-token argmax at <eoq>.

    Matches the headless path's schema so the two prediction files are interchangeable.
    """
    eoq_id, seq_len, batch, vocab = 4, 5, 2, 8
    reg = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1).repeat(1, batch, 1)  # [S, B, 1]

    tokens = torch.zeros(batch, seq_len, dtype=torch.long)
    tokens[0, 2] = eoq_id  # eoq at s=2 for batch 0
    tokens[1, 3] = eoq_id  # eoq at s=3 for batch 1

    # LM logits [S, B, V]. Numeric tokens are ids 5,6,7. At each <eoq> the largest *numeric* logit
    # is the expected answer; a larger *non-numeric* logit must be masked out.
    lm = torch.full((seq_len, batch, vocab), -10.0)
    lm[2, 0, 6] = 5.0
    lm[2, 0, 1] = 100.0  # non-numeric, must be ignored
    lm[3, 1, 7] = 5.0
    lm[3, 1, 0] = 100.0  # non-numeric, must be ignored

    numeric = torch.zeros(vocab, dtype=torch.bool)
    numeric[5:8] = True

    class _Tok:
        def __init__(self):
            self.token_dictionary = {"<eoq>": eoq_id}

        def build_numeric_mask(self, vocab_size=None):
            return numeric

    class _Model:
        def __init__(self):
            self.tokenizer = _Tok()

        def forward_step(self, batch):
            # Like Megatron's postprocess: with labels the LM output is the [B, S] per-token loss, not logits.
            lm_out = torch.zeros(batch["tokens"].shape) if batch.get("labels") is not None else lm
            return {"regression_outputs": reg, "lm_outputs": lm_out}

    out = maxtoki_regression_predict_step(_Model(), {"tokens": tokens, "labels": tokens.clone()}, label_scalar=1.0)
    assert set(out) == {"regression_preds", "timelapse_token_preds"}
    assert out["regression_preds"].shape == (1, batch)
    assert out["timelapse_token_preds"].shape == (1, batch)
    assert out["timelapse_token_preds"].flatten().tolist() == [6, 7]
