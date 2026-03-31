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

"""Tests for SDPA attention backend equivalence.

These tests verify that our SDPADotProductAttention produces outputs equivalent
to:
  1. Manual baddbmm + fp32 softmax (the Megatron unfused fallback path)
  2. torch.nn.functional.scaled_dot_product_attention (directly)
  3. TransformerEngine DotProductAttention (when available)
"""

import math

import pytest
import torch
import torch.nn.functional as F


DEVICE = "cuda"
DTYPE = torch.bfloat16
# bf16 has ~3 decimal digits of mantissa precision. Softmax and the quadratic
# attention computation amplify errors. 5e-2 is a practical tolerance for
# comparing two numerically-different implementations in bf16.
ATOL = 5e-2
RTOL = 5e-2


def _reference_baddbmm_attention(q, k, v, scale, causal=True):
    """Reference implementation: Megatron's unfused baddbmm + fp32 softmax path.

    All operations are autograd-compatible (no in-place on graph tensors).

    Args:
        q: [sq, b, np, hn]
        k: [sk, b, np, hn]
        v: [sk, b, np, hn]
        scale: softmax scale factor (1/sqrt(head_dim))
        causal: whether to apply causal mask

    Returns:
        context: [sq, b, np, hn]
    """
    sq, b, np_, hn = q.shape
    sk = k.shape[0]

    # Reshape to [b*np, sq, hn] / [b*np, sk, hn]
    q_2d = q.permute(1, 2, 0, 3).reshape(b * np_, sq, hn)
    k_2d = k.permute(1, 2, 0, 3).reshape(b * np_, sk, hn)
    v_2d = v.permute(1, 2, 0, 3).reshape(b * np_, sk, hn)

    # Attention scores: [b*np, sq, sk]
    scores = torch.bmm(q_2d, k_2d.transpose(1, 2)) * scale

    if causal:
        mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool, device=q.device), diagonal=1)
        # Use torch.where instead of masked_fill_ to keep autograd clean
        scores = torch.where(mask.unsqueeze(0), torch.tensor(float("-inf"), device=q.device, dtype=scores.dtype), scores)

    # fp32 softmax (matches Megatron's fallback behavior)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # Weighted sum: [b*np, sq, hn]
    context = torch.bmm(probs, v_2d)

    # Back to [sq, b, np, hn]
    context = context.reshape(b, np_, sq, hn).permute(2, 0, 1, 3).contiguous()
    return context


def _sdpa_attention(q, k, v, scale, causal=True):
    """SDPA implementation: permute to [b, np, sq, hn], call F.scaled_dot_product_attention.

    This is what our SDPADotProductAttention.forward should do.

    Args:
        q: [sq, b, np, hn]
        k: [sk, b, np, hn]
        v: [sk, b, np, hn]
        scale: softmax scale factor
        causal: whether to apply causal mask

    Returns:
        context: [sq, b, np, hn]
    """
    # [sq, b, np, hn] -> [b, np, sq, hn]
    q_sdpa = q.permute(1, 2, 0, 3)
    k_sdpa = k.permute(1, 2, 0, 3)
    v_sdpa = v.permute(1, 2, 0, 3)

    out = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        is_causal=causal,
        scale=scale,
    )

    # [b, np, sq, hn] -> [sq, b, np, hn]
    return out.permute(2, 0, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Parametrize over the problematic (154) and fixed (160) head dimensions,
# plus a few sequence lengths. Short seq for fast tests, long seq for realism.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_dim", [154, 160])
@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize("num_heads", [8])
class TestAttentionEquivalence:
    """Verify output equivalence between baddbmm reference and SDPA."""

    @pytest.fixture(autouse=True)
    def setup(self, head_dim, seq_len, num_heads):
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.batch = 2
        self.scale = 1.0 / math.sqrt(head_dim)

        # Fixed seed for reproducibility
        torch.manual_seed(42)
        # Megatron layout: [sq, b, np, hn]
        self.q = torch.randn(
            seq_len, self.batch, num_heads, head_dim,
            dtype=DTYPE, device=DEVICE,
        )
        self.k = torch.randn(
            seq_len, self.batch, num_heads, head_dim,
            dtype=DTYPE, device=DEVICE,
        )
        self.v = torch.randn(
            seq_len, self.batch, num_heads, head_dim,
            dtype=DTYPE, device=DEVICE,
        )

    def test_sdpa_matches_reference_forward(self):
        """SDPA forward output should match the baddbmm reference."""
        ref_out = _reference_baddbmm_attention(self.q, self.k, self.v, self.scale)
        sdpa_out = _sdpa_attention(self.q, self.k, self.v, self.scale)

        assert ref_out.shape == sdpa_out.shape, (
            f"Shape mismatch: ref={ref_out.shape}, sdpa={sdpa_out.shape}"
        )
        torch.testing.assert_close(sdpa_out, ref_out, atol=ATOL, rtol=RTOL)

    def test_sdpa_matches_reference_backward(self):
        """Gradients from SDPA should be close to gradients from baddbmm reference."""
        # Clone with grad tracking
        q_ref = self.q.clone().detach().requires_grad_(True)
        k_ref = self.k.clone().detach().requires_grad_(True)
        v_ref = self.v.clone().detach().requires_grad_(True)

        q_sdpa = self.q.clone().detach().requires_grad_(True)
        k_sdpa = self.k.clone().detach().requires_grad_(True)
        v_sdpa = self.v.clone().detach().requires_grad_(True)

        # Forward + backward for reference
        ref_out = _reference_baddbmm_attention(q_ref, k_ref, v_ref, self.scale)
        ref_out.sum().backward()

        # Forward + backward for SDPA
        sdpa_out = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, self.scale)
        sdpa_out.sum().backward()

        torch.testing.assert_close(q_sdpa.grad, q_ref.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_sdpa.grad, k_ref.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v_sdpa.grad, v_ref.grad, atol=ATOL, rtol=RTOL)

    def test_sdpa_output_is_finite(self):
        """SDPA output should contain no NaN or Inf values."""
        sdpa_out = _sdpa_attention(self.q, self.k, self.v, self.scale)
        assert torch.isfinite(sdpa_out).all(), "SDPA output contains NaN or Inf"

    def test_sdpa_causal_vs_noncausal(self):
        """Causal and non-causal should differ (sanity check that masking works)."""
        causal_out = _sdpa_attention(self.q, self.k, self.v, self.scale, causal=True)
        noncausal_out = _sdpa_attention(self.q, self.k, self.v, self.scale, causal=False)
        # They should NOT be equal (unless seq_len=1, which we don't test)
        assert not torch.allclose(causal_out, noncausal_out, atol=1e-6), (
            "Causal and non-causal outputs are identical -- masking is not working"
        )


class TestAttentionWithGQA:
    """Test SDPA with grouped-query attention (num_query_groups < num_heads)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.seq_len = 128
        self.batch = 2
        self.num_heads = 8
        self.num_kv_heads = 4  # GQA: 2 query heads per KV head
        self.head_dim = 154
        self.scale = 1.0 / math.sqrt(self.head_dim)
        torch.manual_seed(123)

    def _expand_kv(self, kv, num_heads, num_kv_heads):
        """Expand KV heads for GQA: [sq, b, ng, hn] -> [sq, b, np, hn]."""
        repeats = num_heads // num_kv_heads
        return kv.repeat_interleave(repeats, dim=2)

    def test_sdpa_with_gqa(self):
        """SDPA should produce correct output with GQA head expansion."""
        q = torch.randn(
            self.seq_len, self.batch, self.num_heads, self.head_dim,
            dtype=DTYPE, device=DEVICE,
        )
        k = torch.randn(
            self.seq_len, self.batch, self.num_kv_heads, self.head_dim,
            dtype=DTYPE, device=DEVICE,
        )
        v = torch.randn(
            self.seq_len, self.batch, self.num_kv_heads, self.head_dim,
            dtype=DTYPE, device=DEVICE,
        )

        # Expand KV for both methods
        k_expanded = self._expand_kv(k, self.num_heads, self.num_kv_heads)
        v_expanded = self._expand_kv(v, self.num_heads, self.num_kv_heads)

        ref_out = _reference_baddbmm_attention(q, k_expanded, v_expanded, self.scale)
        sdpa_out = _sdpa_attention(q, k_expanded, v_expanded, self.scale)

        torch.testing.assert_close(sdpa_out, ref_out, atol=ATOL, rtol=RTOL)


class TestTEDotProductAttention:
    """Compare our SDPA implementation against TE DotProductAttention.

    TE's DotProductAttention requires more setup but is the actual production path.
    We compare at the raw tensor level using TE's class directly.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.seq_len = 128
        self.batch = 1
        self.num_heads = 8
        self.dtype = DTYPE
        torch.manual_seed(999)

    @pytest.mark.parametrize("head_dim", [154, 160])
    def test_sdpa_matches_te_attention(self, head_dim):
        """SDPA output should match TE DotProductAttention output."""
        try:
            from transformer_engine.pytorch import DotProductAttention as TEAttn
        except ImportError:
            pytest.skip("TransformerEngine not available")

        scale = 1.0 / math.sqrt(head_dim)

        te_attn = TEAttn(
            num_attention_heads=self.num_heads,
            kv_channels=head_dim,
            attention_dropout=0.0,
            attn_mask_type="causal",
            num_gqa_groups=self.num_heads,
            qkv_format="sbhd",
        ).to(device=DEVICE, dtype=self.dtype).eval()

        # TE sbhd format: [sq, b, np, hn]
        q = torch.randn(
            self.seq_len, self.batch, self.num_heads, head_dim,
            dtype=self.dtype, device=DEVICE,
        )
        k = q.clone()
        v = q.clone()

        with torch.no_grad():
            te_out = te_attn(q, k, v)
            sdpa_out = _sdpa_attention(q, k, v, scale, causal=True)

        # TE returns [sq, b, hidden_size] (heads merged), SDPA returns [sq, b, np, hn].
        # Reshape SDPA to match TE's merged-head output.
        sq, b, np_, hn = sdpa_out.shape
        sdpa_out_merged = sdpa_out.reshape(sq, b, np_ * hn)

        # TE may use different precision internally, so use wider tolerance
        torch.testing.assert_close(sdpa_out_merged, te_out, atol=5e-2, rtol=5e-2)


class TestSDPAMegatronLayerEquivalence:
    """Verify that a full TransformerLayer produces equivalent outputs with SDPA vs local (DotProductAttention) backend.

    This test exercises the actual Megatron module classes (not just raw tensor
    ops), catching integration bugs like output shape mismatches between
    core_attention and linear_proj.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hidden_size = 1232  # head_dim = 1232/8 = 154 (not % 8)
        self.num_heads = 8
        self.ffn_hidden_size = 2464
        self.seq_len = 128
        self.batch = 1

    def _build_config(self):
        """Build a minimal TransformerConfig for testing."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        return TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            ffn_hidden_size=self.ffn_hidden_size,
            normalization="RMSNorm",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            bf16=True,
            params_dtype=DTYPE,
            pipeline_dtype=DTYPE,
        )

    def test_sdpa_layer_matches_local_layer_forward(self):
        """Full TransformerLayer forward: SDPA backend output should match local backend."""
        from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
        from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec
        from nemo.collections.llm.gpt.model import local_layer_spec
        from megatron.core.transformer.spec_utils import build_module

        with distributed_model_parallel_state():
            config = self._build_config()

            local_spec = local_layer_spec(config)
            sdpa_spec = sdpa_layer_spec(config)

            local_layer = build_module(local_spec, config=config, layer_number=1).to(device=DEVICE, dtype=DTYPE)
            sdpa_layer = build_module(sdpa_spec, config=config, layer_number=1).to(device=DEVICE, dtype=DTYPE)

            # Copy weights from local to sdpa so outputs are comparable
            sdpa_layer.load_state_dict(local_layer.state_dict())

            # Input: [seq, batch, hidden]
            torch.manual_seed(42)
            hidden_states = torch.randn(
                self.seq_len, self.batch, self.hidden_size,
                dtype=DTYPE, device=DEVICE,
            )

            local_layer.eval()
            sdpa_layer.eval()

            with torch.no_grad():
                local_out, local_ctx = local_layer(hidden_states)
                sdpa_out, sdpa_ctx = sdpa_layer(hidden_states)

            assert local_out.shape == sdpa_out.shape, (
                f"Shape mismatch: local={local_out.shape}, sdpa={sdpa_out.shape}"
            )
            assert local_out.shape == (self.seq_len, self.batch, self.hidden_size), (
                f"Unexpected output shape: {local_out.shape}"
            )
            torch.testing.assert_close(sdpa_out, local_out, atol=ATOL, rtol=RTOL)

    def test_sdpa_layer_matches_local_layer_backward(self):
        """Gradients through SDPA TransformerLayer should match local backend."""
        from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
        from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec
        from nemo.collections.llm.gpt.model import local_layer_spec
        from megatron.core.transformer.spec_utils import build_module

        with distributed_model_parallel_state():
            config = self._build_config()

            local_spec = local_layer_spec(config)
            sdpa_spec = sdpa_layer_spec(config)

            local_layer = build_module(local_spec, config=config, layer_number=1).to(device=DEVICE, dtype=DTYPE)
            sdpa_layer = build_module(sdpa_spec, config=config, layer_number=1).to(device=DEVICE, dtype=DTYPE)

            sdpa_layer.load_state_dict(local_layer.state_dict())

            torch.manual_seed(42)
            h_local = torch.randn(
                self.seq_len, self.batch, self.hidden_size,
                dtype=DTYPE, device=DEVICE, requires_grad=True,
            )
            h_sdpa = h_local.clone().detach().requires_grad_(True)

            local_layer.train()
            sdpa_layer.train()

            local_out, _ = local_layer(h_local)
            local_out.sum().backward()

            sdpa_out, _ = sdpa_layer(h_sdpa)
            sdpa_out.sum().backward()

            # Input gradients should be close
            torch.testing.assert_close(h_sdpa.grad, h_local.grad, atol=ATOL, rtol=RTOL)

            # Weight gradients accumulate numerical differences across all
            # sequence positions, so they need a wider tolerance than forward
            # outputs. With bf16, ~0.003% outliers at atol=0.05 is normal.
            WEIGHT_GRAD_ATOL = 0.15
            WEIGHT_GRAD_RTOL = 0.15
            for (name_l, p_l), (name_s, p_s) in zip(
                local_layer.named_parameters(), sdpa_layer.named_parameters()
            ):
                assert name_l == name_s
                if p_l.grad is not None:
                    torch.testing.assert_close(
                        p_s.grad, p_l.grad, atol=WEIGHT_GRAD_ATOL, rtol=WEIGHT_GRAD_RTOL,
                        msg=lambda m: f"Grad mismatch for {name_l}: {m}",
                    )


class TestSDPAMemoryBehavior:
    """Verify that SDPA uses sub-quadratic memory (i.e., flash attention kicks in)."""

    def test_sdpa_memory_is_subquadratic(self):
        """At seq_len=4096 with head_dim=154, SDPA should use << N^2 memory."""
        import gc
        seq_len = 4096
        batch = 1
        num_heads = 8
        head_dim = 154
        scale = 1.0 / math.sqrt(head_dim)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q = torch.randn(seq_len, batch, num_heads, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)
        k = torch.randn(seq_len, batch, num_heads, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)
        v = torch.randn(seq_len, batch, num_heads, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)

        baseline_mem = torch.cuda.max_memory_allocated()

        out = _sdpa_attention(q, k, v, scale, causal=True)
        out.sum().backward()

        peak_mem = torch.cuda.max_memory_allocated()
        used_mb = (peak_mem - baseline_mem) / 1024**2

        # O(N^2) for attention matrix alone would be:
        # 8 heads * 4096^2 * 2 bytes (bf16) = 256 MB, plus fp32 softmax = 512 MB
        # Flash attention should use much less -- under 100 MB for the attention part
        # Set a generous threshold: if SDPA uses less than 200 MB, flash is working
        quadratic_estimate_mb = num_heads * seq_len * seq_len * 2 / 1024**2
        assert used_mb < quadratic_estimate_mb, (
            f"SDPA used {used_mb:.1f} MB, which is >= the O(N^2) estimate of "
            f"{quadratic_estimate_mb:.1f} MB. Flash attention may not be active."
        )
