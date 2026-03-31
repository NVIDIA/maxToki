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

"""Callable transform for loading HuggingFace weights into BioNeMo models.

This module provides a callable class that loads HuggingFace MaxToki
weights into a BioNeMo model during training initialization, enabling finetuning from
pretrained HF models without intermediate checkpoint conversion.

Usage:
    from bionemo.maxtoki.hf_weights import HFMaxTokiWeightLoader

    # Create the transform
    hf_loader = HFMaxTokiWeightLoader("/path/to/hf/model")

    # Pass to llm.train via model_transform
    llm.train(model, data, trainer, model_transform=hf_loader)
"""

from pathlib import Path

import torch.nn as nn
from nemo.collections.llm.gpt.model.llama import HFLlamaImporter


class HFMaxTokiWeightLoader:
    """Callable transform that loads HuggingFace MaxToki weights.

    This class implements a callable transform that can be passed to llm.train()
    via the model_transform parameter. It loads weights from a HuggingFace model
    and applies them to the BioNeMo model.

    Args:
        hf_model_path: Path to the HuggingFace model directory

    Example:
        >>> transform = HFMaxTokiWeightLoader("./my_hf_model")
        >>> llm.train(model, data, trainer, model_transform=transform)
    """

    def __init__(self, hf_model_path: str | Path):
        self.hf_model_path = Path(hf_model_path)

    def __call__(self, model: nn.Module) -> nn.Module:
        """Load HF weights into the model.

        This method is called by the ModelTransform callback with trainer.model.

        Args:
            model: The model to load weights into (typically MegatronParallel wrapper)

        Returns:
            The same model with HF weights loaded
        """
        print(f"Loading HuggingFace MaxToki weights from {self.hf_model_path}")

        from transformers import AutoModelForCausalLM

        target_model = self._get_target_model(model)
        target_device = next(target_model.module.module.parameters()).device

        source = AutoModelForCausalLM.from_pretrained(
            str(self.hf_model_path), torch_dtype="auto", device_map=str(target_device)
        )

        # Preserve main_grad attributes across weight loading
        main_grad_map = {}
        for name, param in target_model.module.module.named_parameters():
            if hasattr(param, "main_grad"):
                main_grad_map[name] = param.main_grad

        HFLlamaImporter.convert_state(None, source, target_model.module.module)

        for name, param in target_model.module.module.named_parameters():
            if name in main_grad_map:
                param.main_grad = main_grad_map[name]

        return model

    def _get_target_model(self, model: nn.Module) -> nn.Module:
        """Unwrap the model to get to the right level for convert_state.

        The model hierarchy in NeMo/BioNeMo training is:
        - MegatronParallel (trainer wrapper)
          -> BionemoLightningModule (our Lightning module)
            -> DDP (distributed data parallel wrapper)
              -> MCoreGPTModel (the actual Megatron Core model)

        The convert_state method with apply_transforms expects either:
        1. A module with .module attribute pointing to a MegatronModule, OR
        2. Directly a MegatronModule

        BionemoLightningModule has .module pointing to DDP (which is a MegatronModule),
        so we pass the BionemoLightningModule.
        """
        from megatron.core.transformer.module import MegatronModule

        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "module") and isinstance(model.module, MegatronModule):
            return model

        return model
