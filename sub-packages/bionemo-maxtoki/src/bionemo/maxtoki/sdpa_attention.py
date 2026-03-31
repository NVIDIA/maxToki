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

"""SDPA-based attention backend for Megatron.

This module provides a drop-in replacement for Megatron's DotProductAttention
that uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) as the
core attention computation. PyTorch's SDPA auto-selects the most efficient
backend (Flash Attention, Memory-Efficient Attention, or Math) based on the
input dimensions and hardware capabilities.

**Why this exists:** TransformerEngine's attention kernels require
``head_dim % 8 == 0`` for Flash Attention. When this condition is not met
(e.g., head_dim=154 from hidden_size=1232 / 8 heads), TE falls back to an
unfused O(N^2) implementation that materializes the full attention matrix.
At seq_length=16384, this uses ~33 GB per layer and causes OOM.

PyTorch's SDPA Flash Attention backend handles non-aligned head dimensions
via internal padding, avoiding the O(N^2) fallback.

Usage::

    from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec

    config = MaxTokiConfig(
        ...,
        transformer_layer_spec=sdpa_layer_spec,
    )
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from nemo.collections.llm.gpt.model.base import GPTConfig

import torch
import torch.nn.functional as F

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt import gpt_layer_specs
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec


class SDPADotProductAttention(DotProductAttention):
    """Drop-in replacement for Megatron's DotProductAttention using torch SDPA.

    Inherits from DotProductAttention to maintain the same ``__init__``
    signature and config handling. Overrides ``forward()`` to use
    ``F.scaled_dot_product_attention`` instead of ``baddbmm`` + fused softmax.

    This enables Flash Attention for head dimensions that are not divisible
    by 8 (e.g., head_dim=154), since PyTorch's SDPA backend handles internal
    padding transparently.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Compute attention using torch SDPA.

        Args:
            query: Query tensor of shape ``[sq, b, np, hn]``.
            key: Key tensor of shape ``[sk, b, ng, hn]``.
            value: Value tensor of shape ``[sk, b, ng, hn]``.
            attention_mask: Unused when ``is_causal=True`` (kept for API compat).
            attn_mask_type: Override for the mask type set in ``__init__``.
            attention_bias: Not supported; must be ``None``.
            packed_seq_params: Not supported; must be ``None``.

        Returns:
            Context tensor of shape ``[sq, b, hp]``.
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by SDPADotProductAttention. "
            "Use TEDotProductAttention instead."
        )
        assert attention_bias is None, (
            "Attention bias is not supported by SDPADotProductAttention."
        )

        # GQA: expand KV heads to match query heads if needed
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim=2,
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim=2,
            )

        # Megatron layout [sq, b, np, hn] -> SDPA layout [b, np, sq, hn]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        # Determine causal masking
        effective_mask_type = attn_mask_type if attn_mask_type is not None else self.attn_mask_type
        is_causal = effective_mask_type == AttnMaskType.causal

        # Core SDPA call -- PyTorch auto-selects Flash/Efficient/Math backend
        dropout_p = self.config.attention_dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )

        # Back to Megatron layout: [b, np, sq, hn] -> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] -> [sq, b, hp] to match DotProductAttention output shape
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)
        return context


class SDPASpecProvider(LocalSpecProvider):
    """Spec provider that uses SDPA attention with local (non-TE) linear layers.

    Extends ``LocalSpecProvider`` (Megatron's built-in local backend) and
    overrides only ``core_attention()`` to return ``SDPADotProductAttention``.
    All other components (linear layers, layer norms, MLP) remain unchanged.
    """

    def core_attention(self) -> type:
        """Return the SDPA-based attention module."""
        return SDPADotProductAttention


def sdpa_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a layer specification that uses SDPA attention with local layers.

    This is a drop-in replacement for ``local_layer_spec`` that substitutes
    Megatron's ``DotProductAttention`` with ``SDPADotProductAttention``.
    All other components (linear layers, layer norms, MLP) use Megatron's
    standard local (non-TE) implementations.

    Args:
        config: GPT configuration object (from NeMo/Megatron).

    Returns:
        ModuleSpec: Module specification using SDPA attention.
    """
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

    backend = SDPASpecProvider()

    # Layer norm selection: match local_layer_spec behavior
    if config.normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    # MLP spec
    mlp = gpt_layer_specs.get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=layer_norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),  # SDPADotProductAttention
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
