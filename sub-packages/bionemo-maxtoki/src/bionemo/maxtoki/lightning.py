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

from typing import Callable, List, Optional, Sequence, Tuple

from megatron.core.packed_seq_params import PackedSeqParams
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import GPTModel
from nemo.collections.llm.gpt.model.base import MCoreGPTModel, gpt_data_step, gpt_forward_step
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from torch import Tensor, zeros
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from bionemo.maxtoki.api import MaxTokiBaseConfig
from bionemo.llm.lightning import (
    BionemoLightningModule,
    DataStep,
    ForwardStep,
    default_megatron_optimizer,
)


__all__: Sequence[str] = (
    "maxtoki_lightning_module",
    "PassthroughEverythingLossReduction",
)


class PassthroughEverythingLossReduction(MegatronLossReduction):
    """A loss reduction that passes through everything."""

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[Tensor, DataT]:
        return zeros((1, 1)), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        return forward_out


def maxtoki_lightning_module(
    config: MaxTokiBaseConfig,
    optimizer: Optional[MegatronOptimizerModule] = None,
    tokenizer: Optional[TokenizerSpec | PreTrainedTokenizerBase] = None,
    data_step: DataStep = gpt_data_step,
    forward_step: ForwardStep = gpt_forward_step,
    predict_step: Optional[Callable[[DataT], DataT]] = None,
    predict_loss_reduction: Optional[Callable[[DataT], DataT]] = None,
    freeze_params_until_key_suffix: Optional[str] = None,
    model_transform: Optional[Callable] = None,
    **model_construct_args,
) -> BionemoLightningModule[MCoreGPTModel, MegatronLossReduction]:
    """Construct a BionemoLightningModule for MaxToki.

    Args:
        config: Model configuration.
        optimizer: Optional optimizer module.
        tokenizer: Optional tokenizer.
        data_step: Data step function.
        forward_step: Forward step function.
        predict_step: Optional predict step function.
        predict_loss_reduction: Overrides the default passthrough loss reduction.
        freeze_params_until_key_suffix: Optional suffix to freeze params up to.
        model_transform: Optional model transform callable.
        **model_construct_args: Additional model construction arguments.
    """
    module = BionemoLightningModule(
        config=config,
        optimizer=optimizer if optimizer is not None else default_megatron_optimizer(),
        data_step=data_step,
        forward_step=forward_step,
        predict_step=predict_step,
        tokenizer=tokenizer,
        model_transform=model_transform,
        freeze_params_until_key_suffix=freeze_params_until_key_suffix,
        **model_construct_args,
    )
    if predict_loss_reduction is not None:
        module.predict_loss_reduction = predict_loss_reduction

    module.get_inference_wrapper = GPTModel.get_inference_wrapper
    return module
