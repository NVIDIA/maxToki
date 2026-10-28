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
from types import SimpleNamespace

import pytest

from bionemo.maxtoki import import_hf
from bionemo.maxtoki.import_hf import HFMaxTokiImporter, get_parser, resolve_rms_norm_eps


def _write_llama_config(path, **overrides):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "vocab_size": 128,
        **overrides,
    }
    config = {k: v for k, v in config.items() if v is not ...}
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture()
def stub_maxtoki_config(monkeypatch):
    # Instantiating the real MaxTokiConfig marks it as io-wrapped in NeMo, and its subclasses inherit that flag, so
    # later MaxTokiFineTuneConfig instances in the same process would be created without `__io__`.
    monkeypatch.setattr(import_hf, "MaxTokiConfig", lambda **kwargs: SimpleNamespace(**kwargs))


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_resolve_uses_config_value(eps):
    assert resolve_rms_norm_eps({"rms_norm_eps": eps}) == eps


def test_resolve_override_wins():
    assert resolve_rms_norm_eps({"rms_norm_eps": 1e-5}, 1e-6) == 1e-6


def test_resolve_override_when_config_missing():
    assert resolve_rms_norm_eps({}, 1e-6) == 1e-6


def test_resolve_missing_without_override_raises():
    with pytest.raises(ValueError, match="rms_norm_eps"):
        resolve_rms_norm_eps({})


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_importer_config_preserves_config_eps(tmp_path, stub_maxtoki_config, eps):
    config = HFMaxTokiImporter.config(_write_llama_config(tmp_path, rms_norm_eps=eps), vocab_size=128)
    assert config.layernorm_epsilon == eps


def test_importer_config_missing_eps_raises(tmp_path, stub_maxtoki_config):
    # transformers would silently default this to 1e-6.
    with pytest.raises(ValueError, match="rms_norm_eps"):
        HFMaxTokiImporter.config(_write_llama_config(tmp_path, rms_norm_eps=...), vocab_size=128)


def test_importer_config_override(tmp_path, stub_maxtoki_config):
    hf_dir = _write_llama_config(tmp_path, rms_norm_eps=1e-5)
    assert HFMaxTokiImporter.config(hf_dir, vocab_size=128, rms_norm_eps=1e-6).layernorm_epsilon == 1e-6


def test_cli_rms_norm_eps():
    base = [
        "--train-data-path", "x", "--val-data-path", "x", "--test-data-path", "x",
        "--hf-model-path", "x", "--rope-scale-factor", "1.0", "--output-weights", "separate", "--seq-length", "128",
    ]  # fmt: skip
    assert get_parser().parse_args(base).rms_norm_eps is None
    assert get_parser().parse_args([*base, "--rms-norm-eps", "1e-6"]).rms_norm_eps == 1e-6
