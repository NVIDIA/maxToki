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

from bionemo.llm.model import config as bionemo_config
from bionemo.maxtoki.model import MaxTokiFineTuneConfig, MaxTokiMultitaskFineTuneConfig
from bionemo.maxtoki.predict import get_parser, rope_scaling_config_kwargs


CONFIG_CLASSES = [MaxTokiMultitaskFineTuneConfig, MaxTokiFineTuneConfig]


def _load_with_checkpoint_scale(monkeypatch, config, ckpt_scale_factor):
    ckpt_config = type(config)(scale_factor=ckpt_scale_factor)
    monkeypatch.setattr(bionemo_config.io, "load_context", lambda path, subpath: ckpt_config)
    config.load_settings_from_checkpoint("unused")
    return config


def test_multitask_default_overrides_ignore_checkpoint_scale(monkeypatch):
    config = _load_with_checkpoint_scale(monkeypatch, MaxTokiMultitaskFineTuneConfig(), 1.0)
    assert config.scale_factor == 8.0


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
def test_default_uses_checkpoint_scale_factor(monkeypatch, config_class):
    kwargs = rope_scaling_config_kwargs(config_class, None)
    assert "scale_factor" not in kwargs["override_parent_fields"]
    config = _load_with_checkpoint_scale(monkeypatch, config_class(**kwargs), 1.0)
    assert config.scale_factor == 1.0


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
def test_user_scale_factor_overrides_checkpoint(monkeypatch, config_class):
    kwargs = rope_scaling_config_kwargs(config_class, 4.0)
    assert "scale_factor" in kwargs["override_parent_fields"]
    config = _load_with_checkpoint_scale(monkeypatch, config_class(**kwargs), 1.0)
    assert config.scale_factor == 4.0


def test_cli_flag_defaults_to_none():
    required = ["--initial-ckpt-path", "x", "--data-path", "x", "--tokenizer-path", "x", "--output-dir", "x"]
    parser = get_parser()
    assert parser.parse_args(required).rope_scaling_factor is None
    assert parser.parse_args([*required, "--rope-scaling-factor", "2.5"]).rope_scaling_factor == 2.5
