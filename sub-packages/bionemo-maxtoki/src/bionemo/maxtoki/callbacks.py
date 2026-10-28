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

from typing import List, Sequence, Tuple

import torch
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging


__all__: Sequence[str] = ("WeightSetupGuard",)

_Fingerprint = List[Tuple[str, torch.nn.Parameter, float, float]]


def _modules(module) -> list:
    return list(module) if isinstance(module, (list, tuple)) else [module]


def _fingerprint(module) -> _Fingerprint:
    """Per-parameter (sum |w|, max |w|), cheap enough to take for every tensor and kept as plain floats."""
    out = []
    with torch.no_grad():
        for m in _modules(module):
            for name, param in m.named_parameters():
                w = param.detach().float().abs()
                out.append((name, param, float(w.sum(dtype=torch.float64)), float(w.max()) if w.numel() else 0.0))
    return out


class WeightSetupGuard(Callback):
    """Makes sure the model weights survive Megatron's distributed setup before training starts.

    With the distributed optimizer, Megatron DDP moves every parameter into a contiguous buffer after the model (and
    any checkpoint weights) has been built. A race in that setup can leave a parameter holding garbage, which surfaces
    later as a NaN in the first forward pass. This callback synchronizes the device once setup has finished and then
    checks every parameter against a fingerprint taken right after the model was built. The check is tolerant of the
    precision cast (bf16 rounding moves the fingerprint by well under 1%) and raises on anything larger.
    """

    def __init__(self, rtol: float = 1e-2):
        """Create the guard.

        Args:
            rtol: Allowed relative change of each parameter's sum and max of absolute values across setup.
        """
        self.rtol = rtol
        self._reference: _Fingerprint | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        """Wrap ``configure_model`` so the fingerprint is taken right after the model is built."""
        if stage != "fit" or getattr(pl_module, "_weight_guard_wrapped", False):
            return
        configure_model = pl_module.configure_model

        def configure_model_and_fingerprint(*args, **kwargs):
            configure_model(*args, **kwargs)
            if self._reference is None and getattr(pl_module, "module", None) is not None:
                self._reference = _fingerprint(pl_module.module)

        pl_module.configure_model = configure_model_and_fingerprint
        pl_module._weight_guard_wrapped = True

    def on_fit_start(self, trainer, pl_module) -> None:
        """Synchronize after distributed setup, then verify every parameter against the fingerprint."""
        if torch.cuda.is_available() and self._uses_distributed_optimizer(pl_module):
            torch.cuda.synchronize()
        if self._reference is None:
            return
        bad = []
        with torch.no_grad():
            for name, param, ref_sum, ref_max in self._reference:
                w = param.detach().float().abs()
                cur_sum = float(w.sum(dtype=torch.float64))
                cur_max = float(w.max()) if w.numel() else 0.0
                ok = (
                    torch.isfinite(w).all()
                    and abs(cur_sum - ref_sum) <= self.rtol * ref_sum + 1e-6
                    and cur_max <= ref_max * (1 + self.rtol) + 1e-6
                )
                if not ok:
                    bad.append(f"{name}: sum|w| {ref_sum:.6g} -> {cur_sum:.6g}, max|w| {ref_max:.6g} -> {cur_max:.6g}")
        self._reference = None
        if bad:
            raise RuntimeError(
                f"{len(bad)} model parameter(s) changed during distributed setup, before any training step, so the "
                "weights in memory no longer match the model that was built or loaded:\n  " + "\n  ".join(bad)
            )
        logging.info("Model weights verified after distributed setup.")

    @staticmethod
    def _uses_distributed_optimizer(pl_module) -> bool:
        config = getattr(getattr(pl_module, "optim", None), "config", None)
        return bool(getattr(config, "use_distributed_optimizer", True))
