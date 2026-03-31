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

from dataclasses import dataclass
from typing import Sequence, Type

from nemo.collections.llm.gpt.model.base import MaskedTokenLossReduction
from nemo.collections.llm.gpt.model.llama import Llama32Config1B, LlamaModel
from nemo.lightning.megatron_parallel import MegatronLossReduction

from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "MaxTokiBaseConfig",
    "MaxTokiModel",
)

MaxTokiModel = LlamaModel


@dataclass
class MaxTokiBaseConfig(Llama32Config1B, iom.IOMixinWithGettersSetters):
    """A specialized MaxToki config for training Llama models on biological data.

    This config extends Llama32Config1B from NeMo and adds a leaf-level iomixin.
    It's specifically designed to work with the SingleCellDataModule from Geneformer.
    """

    enable_autocast: bool = False
    model_cls: Type[MaxTokiModel] = MaxTokiModel
    loss_reduction_class: Type[MegatronLossReduction] = MaskedTokenLossReduction  # type: ignore
