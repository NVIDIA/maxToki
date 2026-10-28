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

from types import SimpleNamespace

import pytest
import torch

from bionemo.maxtoki.callbacks import WeightSetupGuard


class _Module:
    """Stand-in for the Lightning module: configure_model builds ``self.module``."""

    def __init__(self, use_distributed_optimizer=True):
        self.module = None
        self.optim = SimpleNamespace(config=SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer))

    def configure_model(self):
        torch.manual_seed(0)
        self.module = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.LayerNorm(32), torch.nn.Linear(32, 4))


def _built(stage="fit"):
    guard, pl_module = WeightSetupGuard(), _Module()
    guard.setup(None, pl_module, stage)
    pl_module.configure_model()
    return guard, pl_module


def test_unchanged_weights_pass():
    guard, pl_module = _built()
    guard.on_fit_start(None, pl_module)


def test_precision_cast_passes():
    guard, pl_module = _built()
    pl_module.module.to(torch.bfloat16)
    guard.on_fit_start(None, pl_module)


def test_corrupted_weight_is_named():
    guard, pl_module = _built()
    with torch.no_grad():
        pl_module.module[1].weight[0] = 8.7e31
    with pytest.raises(RuntimeError, match=r"(?s)1 model parameter\(s\) changed.*1\.weight: sum\|w\| 32"):
        guard.on_fit_start(None, pl_module)


def test_nan_weight_is_caught():
    guard, pl_module = _built()
    with torch.no_grad():
        pl_module.module[0].bias[3] = float("nan")
    with pytest.raises(RuntimeError, match=r"0\.bias"):
        guard.on_fit_start(None, pl_module)


def test_predict_is_untouched():
    guard, pl_module = WeightSetupGuard(), _Module()
    configure_model = pl_module.configure_model
    guard.setup(None, pl_module, "predict")
    assert pl_module.configure_model == configure_model
    pl_module.configure_model()
    guard.on_fit_start(None, pl_module)
