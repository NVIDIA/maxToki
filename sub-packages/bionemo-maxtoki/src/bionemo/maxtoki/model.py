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
import contextlib
import logging
from dataclasses import dataclass, field
from functools import partial

from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, Type, TypeVar

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.mlp import HAVE_TE
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params
from nemo.collections.llm import Llama32Config1B, MaskedTokenLossReduction
from nemo.collections.llm.gpt.model.base import get_packed_seq_params, mtp_block_spec
from nemo.collections.llm.gpt.model.llama import apply_rope_scaling

from nemo.lightning.base import get_vocab_size
from torch import Tensor
from torch.nn import Linear
from torchmetrics import Metric

from bionemo.maxtoki.tokenizer import (
    MaxTokiTokenizer,
    find_eoq_indices,
)
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.config import (
    _OVERRIDE_BIONEMO_CONFIG_DEFAULTS,
    MegatronBioNeMoTrainableModelConfig,
    TorchmetricsConfig,
)
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "FineTuneLlamaLanguageModelLossWithReduction",
    "FinetuneLlamaModel",
    "MaxTokiFineTuneConfig",
    "MaxTokiLossWithReduction",
    "MaxTokiRegressionHead",
    "UnreducedCELoss",
    "UnreducedMSELoss",
)


class UnreducedMSELoss(Metric):
    """Tracks unreduced MSE loss for multitask regression."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("sum_mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mse_loss: Tensor, loss_mask: Tensor) -> None:
        if mse_loss.shape == loss_mask.t().shape:
            loss_mask = loss_mask.t()

        masked_loss = mse_loss * loss_mask
        self.sum_mse_loss += masked_loss.sum()
        self.total_tokens += loss_mask.sum().long()

    def compute(self) -> Tensor:
        if self.total_tokens == 0:
            return torch.tensor(0.0, device=self.sum_mse_loss.device)
        return self.sum_mse_loss.float() / self.total_tokens.float()


class UnreducedCELoss(Metric):
    """Tracks unreduced CE loss for multitask language modeling."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("sum_ce_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ce_loss: Tensor, loss_mask: Tensor) -> None:
        if ce_loss.shape == loss_mask.t().shape:
            loss_mask = loss_mask.t()
        masked_loss = ce_loss * loss_mask
        self.sum_ce_loss += masked_loss.sum()
        self.total_tokens += loss_mask.sum().long()

    def compute(self) -> Tensor:
        if self.total_tokens == 0:
            return torch.tensor(0.0, device=self.sum_ce_loss.device)
        return self.sum_ce_loss.float() / self.total_tokens.float()


class RegressionAvgPred(Metric):
    """Average prediction produced by the model."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("sum_regression_avg_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, regression_preds: Tensor, loss_mask: Tensor) -> None:
        if regression_preds.shape == loss_mask.t().shape:
            loss_mask = loss_mask.t()
        masked_preds = regression_preds * loss_mask
        self.sum_regression_avg_pred += masked_preds.sum()
        self.total_tokens += masked_preds.numel()

    def compute(self) -> Tensor:
        if self.total_tokens == 0:
            return torch.tensor(0.0, device=self.sum_regression_avg_pred.device)
        return self.sum_regression_avg_pred.float() / self.total_tokens


class MaxTokiLossWithReduction(MaskedTokenLossReduction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Mixed loss reduction that combines the timelapse regression loss with 'next cell' lm loss.

        Expects batch to be produced using collate_batch_multitask (TODO: location of this function) with the following structure:
        {
            'tokens': Tensor,
            'labels': Tensor,
            'loss_mask': Tensor,
            'position_ids': Tensor,
        }
        """
        _, normal_forward_out = forward_out["regression_preds"], forward_out["lm_outputs"]

        # Apply our masks.
        task_mask = self.get_task_mask(batch)
        timelapse_task_mask = torch.tensor(task_mask, device=batch["labels"].device)
        # Per-task loss masks that prevent task leak (e.g. NextCell entities wont be includedfor MSE)
        # NOTE: This might be more efficient using labels instead of loss mask.
        timelapse_loss_mask = batch["loss_mask"] * timelapse_task_mask
        nextcell_loss_mask = batch["loss_mask"] * ~timelapse_task_mask

        # pullout the base loss_mask, replace it with the composed loss mask.
        loss_mask = batch["loss_mask"]
        batch["loss_mask"] = nextcell_loss_mask
        ce_loss_per_token_sums = super().forward(batch, normal_forward_out)

        # Put it back to prevent side effects.
        batch["loss_mask"] = loss_mask

        # Once again we need to filter for tasks, this time for the TimeBetweenCells task.
        loss_mask = batch["loss_mask"].t()

        per_token_mse_loss = forward_out["mse_loss"]
        per_token_mse_loss = per_token_mse_loss * timelapse_loss_mask.t()
        # Now we pass in numeric_regression_labels to make this a bit easier.

        mse_loss_sum = per_token_mse_loss.sum()
        mse_num_valid_tokens_sum = timelapse_loss_mask.sum()

        # Mixture parameter that re-weights examples by normalizing 'token counts'

        mixture_ratio = 0.5
        # Can always disable the mixing parameters. Unfortunately this happens outside the configs, so we cant really parameterize it at the top level.
        # mixture_ratio = None
        if mixture_ratio is not None:
            # Total number of tokens.
            reference_count = ce_loss_per_token_sums[1] + mse_num_valid_tokens_sum
            if mse_num_valid_tokens_sum == 0:
                mse_ratio = 0.0
            else:
                # Reduces the loss sums to loss-per-token.
                mse_ratio = mse_loss_sum / mse_num_valid_tokens_sum

            # Since reference count is dominated by CE tokens, this will mostly be the same as scaling both by (.5 * ce_num_valid_tokens)
            scaled_mse_sum = mixture_ratio * mse_ratio * reference_count

            scaled_ce_sum = (
                (1 - mixture_ratio) * (ce_loss_per_token_sums[0] / ce_loss_per_token_sums[1]) * reference_count
            )
            if ce_loss_per_token_sums[1] == 0:
                ce_ratio = 0.0
            else:
                ce_ratio = ce_loss_per_token_sums[0] / ce_loss_per_token_sums[1]
            scaled_ce_sum = (1 - mixture_ratio) * ce_ratio * reference_count

            combined_sum = scaled_ce_sum + scaled_mse_sum
            combined_count = reference_count

            combined_loss_sums = (combined_sum, combined_count)
        else:
            combined_sum = mse_loss_sum + ce_loss_per_token_sums[0]
            combined_count = ce_loss_per_token_sums[1] + mse_num_valid_tokens_sum

        combined_loss_sums_tensor = torch.cat(
            [combined_loss_sums[0].clone().detach().view(1), combined_loss_sums[1].clone().detach().view(1)]
        )

        combined_loss = (
            combined_loss_sums[0],
            combined_loss_sums[1].to(ce_loss_per_token_sums[1].dtype),
            {
                "loss_sum_and_ub_size": combined_loss_sums_tensor,
            },
        )

        return combined_loss

    @staticmethod
    def infer_task_type(batch: Dict[str, Tensor]) -> Literal["TimeBetweenCells", "NextCell"]:
        types = []
        for b in batch["loss_mask"]:
            if b.sum() <= 1:
                types.append("TimeBetweenCells")
            elif b.sum() > 1:
                types.append("NextCell")
            else:
                raise ValueError("Invalid task type.")
        return types

    @staticmethod
    def get_task_mask(batch: Dict[str, Tensor]) -> List[List[bool]]:
        """Returns a mask for the task type.

        True indicates that the task is a TimeBetweenCells task.
        False indicates that the task is a NextCell task.
        The output is a list of singleton lists to facilitate element-wise multiplication with the downstream loss mask.
        """
        tasks = MaxTokiLossWithReduction.infer_task_type(batch)
        mask = [[x == "TimeBetweenCells"] for x in tasks]
        return mask

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Only used for logging. This reduce should be equivalent to the mcore reduce used in loss."""
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                # legacy behavior, average over the number of microbatches
                avg = [x["avg"] for x in losses_reduced_per_micro_batch]
                loss = torch.cat(avg).mean()
                return loss

            from megatron.core import parallel_state

            loss_sum_and_ub_size = [
                x["loss_sum_and_ub_size"] for x in losses_reduced_per_micro_batch if x["loss_sum_and_ub_size"][1] > 0
            ]
            loss = (
                torch.vstack(loss_sum_and_ub_size).sum(dim=0)
                if len(loss_sum_and_ub_size) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            torch.distributed.all_reduce(
                loss,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # average over the total number of tokens across the global batch.
            loss = loss[0] / loss[1]
            return loss

        return torch.tensor(0.0, device=torch.cuda.current_device())


class MaxTokiLossCEOnlyWithReduction(MaskedTokenLossReduction):
    """CE-only loss reduction for multitask training. Uses cross-entropy loss for both NextCell
    and TimeBetweenCells tasks (no MSE regression), with a mixture ratio to balance task contributions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        _, normal_forward_out = forward_out["regression_preds"], forward_out["lm_outputs"]
        task_mask = MaxTokiLossWithReduction.get_task_mask(batch)
        timelapse_task_mask = torch.tensor(task_mask, device=batch["labels"].device)

        timelapse_loss_mask = batch["loss_mask"] * timelapse_task_mask
        nextcell_loss_mask = batch["loss_mask"] * ~timelapse_task_mask
        loss_mask = batch["loss_mask"]

        batch["loss_mask"] = nextcell_loss_mask
        nextcell_ce_loss_per_token_sums = super().forward(batch, normal_forward_out)

        batch["loss_mask"] = timelapse_loss_mask
        timelapse_ce_loss_per_token_sums = super().forward(batch, normal_forward_out)
        batch["loss_mask"] = loss_mask

        nextcell_loss_sum, nextcell_num_valid_tokens = (
            nextcell_ce_loss_per_token_sums[0],
            nextcell_ce_loss_per_token_sums[1],
        )
        timelapse_loss_sum, timelapse_num_valid_tokens = (
            timelapse_ce_loss_per_token_sums[0],
            timelapse_ce_loss_per_token_sums[1],
        )

        mixture_ratio = 0.5
        if mixture_ratio is not None:
            reference_count = nextcell_num_valid_tokens + timelapse_num_valid_tokens

            if timelapse_num_valid_tokens == 0:
                timelapse_ratio = 0.0
            else:
                timelapse_ratio = timelapse_loss_sum / timelapse_num_valid_tokens

            if nextcell_num_valid_tokens == 0:
                nextcell_ratio = 0.0
            else:
                nextcell_ratio = nextcell_loss_sum / nextcell_num_valid_tokens

            scaled_timelapse_sum = mixture_ratio * timelapse_ratio * reference_count
            scaled_nextcell_sum = (1 - mixture_ratio) * nextcell_ratio * reference_count

            combined_sum = scaled_nextcell_sum + scaled_timelapse_sum
            combined_count = reference_count

            combined_loss_sums = (combined_sum, combined_count)
        else:
            combined_sum = nextcell_loss_sum + timelapse_loss_sum
            combined_count = nextcell_num_valid_tokens + timelapse_num_valid_tokens
            combined_loss_sums = (combined_sum, combined_count)

        combined_loss_sums_tensor = torch.cat(
            [combined_loss_sums[0].clone().detach().view(1), combined_loss_sums[1].clone().detach().view(1)]
        )

        combined_loss = (
            combined_loss_sums[0],
            combined_loss_sums[1].to(nextcell_ce_loss_per_token_sums[1].dtype),
            {
                "loss_sum_and_ub_size": combined_loss_sums_tensor,
            },
        )

        return combined_loss


class MaxTokiFineTuneModel(MCoreGPTModel):
    """Megatron model for MaxToki fine-tuning. Adds additional parameters for the numeric mask and vocab to numeric map.

    Customization is applied such that a custom regression head or regression loss is taken from the model and passed to the loss.
    """

    def __init__(
        self,
        config,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        numeric_mask: Tensor = None,
        vocab_to_numeric_map: Tensor = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "mrope", "none"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
        penalty_factor: float | None = None,
        additive_penalty: float = 10.0,
        label_scalar: float = 200.0,
    ) -> None:
        """Copypasted the entire signature from GPTModel, this exposes whats actually there, but in practice could
        make this easier on the eyes by using the kwargs/args parameters to get the params we care about.

        additive_penalty: Additive penalty applied for penalizing non-numeric token predictions in regression tasks.
        """
        # Instantiate our custom task head.
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        self.numeric_mask = numeric_mask
        self.vocab_to_numeric_map = vocab_to_numeric_map

        # Penalty parameter used in the regression head.
        if penalty_factor is None:
            penalty_factor = 1.0  # no extra penalty

        self.penalty_factor = penalty_factor
        # Attach custom task head.

        self.label_scalar = label_scalar
        # Additive penalty applied for penalizing non-numeric token predictions in regression tasks.
        # per_token_mse_loss * scale_factor_a + (scale_factor_b * additive_penalty)
        # Where scale_factor_b is the weighted probability of non-numeric tokens.
        self.additive_penalty = additive_penalty

    @staticmethod
    def apply_penalty(
        per_token_mse_loss: Tensor,
        scale_factor_a: float = 1.0,
        scale_factor_b: float = 0.0,
        additive_penalty: float = 0.0,
    ) -> Tensor:
        """Apply a penalty to the MSE loss using the following formula:

        penalized_loss = per_token_mse_loss * scale_factor_a + (scale_factor_b * additive_penalty)

        Args:
            per_token_mse_loss: The MSE loss per token. In practice, loss mask should only contain one non-zero.
            scale_factor_a: Scalar applied to the loss.
            scale_factor_b: Scalar applied to additive penalty. Useful for dynamic rescaling of the loss.
            additive_penalty: Constant penalty to apply.
        """
        return per_token_mse_loss * scale_factor_a + (scale_factor_b * additive_penalty)

    @staticmethod
    def _headless_timelapse(
        hidden_states: Tensor, numeric_mask: Tensor, vocab_to_numeric_map: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        prediction_logits = hidden_states
        prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1).float()

        # Need a boolean mask
        # NOTE: ~ is not supported for tensors, so we need to convert to a boolean mask.
        numeric_mask = numeric_mask.bool().to(prediction_probs.device)

        # Calculate the total probability mass assigned to non-numeric tokens
        non_numeric_prob_mass = torch.sum(prediction_probs * ~numeric_mask, axis=-1)
        # applies a mask column wise
        # Calculate the expected value using only the distribution over numeric tokens
        numeric_probs = prediction_probs * numeric_mask.to(prediction_probs.device)
        numeric_prob_sum = torch.sum(numeric_probs, axis=-1)

        valid_mask = numeric_prob_sum > 1e-9
        # replace bad rows with one to make division safe
        numeric_prob_sum = torch.where(
            numeric_prob_sum > 1e-9,
            input=numeric_prob_sum,
            other=torch.ones_like(numeric_prob_sum, device=prediction_probs.device),
        )
        renormalized_numeric_probs = numeric_probs / numeric_prob_sum[..., None]

        # Zero out bad rows.
        renormalized_numeric_probs = torch.where(
            valid_mask[..., None], renormalized_numeric_probs, torch.zeros_like(renormalized_numeric_probs)
        )

        # element wise dot product to get back into numeric space.
        predicted_value = torch.matmul(renormalized_numeric_probs, vocab_to_numeric_map.to(prediction_probs.device))

        # Also compute the argmax of the logits after applying numeric mask
        # Mask out non-numeric tokens by setting their logits to -inf
        masked_logits = torch.where(
            numeric_mask, prediction_logits, torch.tensor(float("-inf"), device=prediction_logits.device)
        )
        argmax_prediction = torch.argmax(masked_logits, dim=-1)

        # shape = [S, B]
        return predicted_value, non_numeric_prob_mass, argmax_prediction

    def headless_timelapse(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Using the logits in the final state, make a prediction for the timelapse token.

        This is done by applying a softmax over the logits and transforming the discrete tokens back into their numeric values.
        First a softmax is applied to the logits, then masked to only include dimensions which represent numeric values.
        The softmax is then rescaled using only numeric values to sum to one.

        The values each token represents is then weighted by the estimated probability of the token and summed to get the expected value.

        The expected numeric value is then computed and returned, along with the non-numeric probability mass and the argmax prediction (masked).
        """
        return self._headless_timelapse(hidden_states, self.numeric_mask, self.vocab_to_numeric_map)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        next_token_only: bool = False,
    ) -> dict:
        # Check sequence length set internally here.

        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        standard_output = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        headless_timelapse_output = None, None, None
        regression_output = None

        if not next_token_only:
            token_logits, _ = self.output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
            )
            headless_timelapse_output = self.headless_timelapse(token_logits)
            regression_output = headless_timelapse_output[0]

        if labels is not None and not next_token_only:
            regression_labels = labels
            numeric_map = self.vocab_to_numeric_map.to(regression_labels.device)
            # Good to point out here that 0 is used for non-numeric tokens and for the actual value 0.
            #    we rely on our loss mask to avoid this.
            numeric_regression_labels = numeric_map[regression_labels].t()
            headless_mse_loss = torch.nn.functional.mse_loss(
                headless_timelapse_output[0].float(),
                numeric_regression_labels.float() / self.label_scalar,
                reduction="none",
            )
            mse_loss = self.apply_penalty(
                headless_mse_loss,
                scale_factor_a=1.0,
                scale_factor_b=headless_timelapse_output[1],
                additive_penalty=self.additive_penalty,
            )
        else:
            # No labels, no MSE loss.
            mse_loss = None
            numeric_regression_labels = None

        return {
            "regression_preds": regression_output * self.label_scalar if regression_output is not None else None,
            "lm_outputs": standard_output,  # Contains logits at inference time allegedly.
            "mse_loss": mse_loss,
            "non_numeric_prob_mass": headless_timelapse_output[1],
            "timelapse_token_preds": headless_timelapse_output[2],
            "loss_mask": loss_mask,
        }


class FineTuneLlamaLanguageModelLossWithReduction(MaskedTokenLossReduction):
    """Custom loss reduction used for fine tuning. Example class.

    The key insight is to connect some output from the underlying model that can be used by the forward
    method in this class. In this example, we combine regression loss with language model loss. The forward
    pass of the underlying gpt model returns per-token language model loss. We have setup the regression head
    to return plain outputs, thus the loss is computed here.

    Per-token losses are summed in the `forward` implementation, as well as the number of tokens. This occurs
    independently for both the regression loss and the language model loss. The combining of these four terms
    is an implementation detail, and in this example, it is done naively with a simple sum of all terms.

    `forward()` then must return a three element tuple, where the first element is the loss sum, the second element is the number of summed items,
    and the third element is a dictionary containing a tensor of the loss sum and the number of summed items.

    The actual reduction occurs inside Megatron's get_fwd_backward function, which simply gathers the terms across all
    devices (microbatches), sums them, and then divides the losses by the total number of summed items.

    `reduce()` is only used for logging the validation loss. In this example, its copy pasted from the parent class for
    exposure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Responsible for outputting the loss as per-microbatch, with the number of summed items.

        the language model loss (ce_loss_per_token_sums) is an output from MCoreGPTModel.
        the regression loss is computed here, where the output of the regression head is the input here.

        Here, the per-token losses should be filtered to only those which should be included in the total loss.
        loss_mask is used for computing the sum, set tokens to zero to exclude them from the total loss in masked_token_loss.

        The output structure is as follows:
            (loss_sum, summed_items, {'loss_sum_and_ub_size': (loss_sum, summed_items)}
            with shapes:
            (tensor(1,), tensor(1,), {'loss_sum_and_ub_size': tensor((2,), device='cuda:0')})

        summed_items refers to the denominator when computing the loss total. Naive case is the number of 'valid tokens'
            used in the loss.
        """
        regression_output, normal_forward_out = forward_out["regression_outputs"], forward_out["lm_outputs"]
        # NOTE: do any filtering of tokens before this step.
        #    forward_outclass contains per-token, cross-entropy, loss.
        #    parent class simply multiplies them by the loss mask and sums.
        ce_loss_per_token_sums = super().forward(batch, normal_forward_out)

        # NOTE: example:  for simplicity we will take a random number between 20-90
        regression_label = (
            torch.normal(35.0, 20.0, size=regression_output.shape).clamp(min=0).to(regression_output.device)
        )

        # Compute the per-token MSE loss, using regression labels as the targets.
        # !! IMPORTANT: Must cast this to float to function in the same way as cross entropy loss. Gradient dtype errors will occur if this is skipped.
        regression_output = regression_output.float()
        per_token_mse_loss = torch.nn.functional.mse_loss(regression_output, regression_label, reduction="none")

        # NOTE: optionally could scale mse_loss per token here.
        per_token_mse_loss = per_token_mse_loss / 10

        # Now weighted average the losses per token
        mse_loss_mask = batch["loss_mask"]  # Tokens eligible for mse_loss

        # NOTE: function name implies its a masked_token_loss, but its simply a sum over both parameters.
        # mse_loss_sum = masked_token_loss(per_token_mse_loss, mse_loss_mask)
        # This is the same as the masked_token_loss function above, but without the function call.
        mse_loss_sum = torch.sum(per_token_mse_loss.view(-1).float() * mse_loss_mask.view(-1).float())
        mse_num_valid_tokens_sum = mse_loss_mask.sum()

        """
        Both Losses now have the following structure:
        (loss_sum, summed_items, {'loss_sum_and_ub_size': (loss_sum, summed_items)})
        with shapes:
        (tensor(1,), tensor(1,), {'loss_sum_and_ub_size': tensor((2,), device='cuda:0')})
        """

        combined_loss_sums = (
            ce_loss_per_token_sums[0] + mse_loss_sum,
            ce_loss_per_token_sums[1] + mse_num_valid_tokens_sum,
        )
        # pack into tensor for compatibility with megatron
        weighted_loss_sum_tensor = torch.cat(
            [combined_loss_sums[0].clone().detach().view(1), combined_loss_sums[1].clone().detach().view(1)]
        )
        combined_loss = (
            combined_loss_sums[0],
            combined_loss_sums[1].to(ce_loss_per_token_sums[1].dtype),
            {"loss_sum_and_ub_size": weighted_loss_sum_tensor},
        )
        """
        !!IMPORTANT!!

        The returned tuples represent the numerator and the denominator of the loss for this microbatch.
        These are summed across ALL microbatches, and then divided. The above naively combines the per-token losses
        between the cross-entropy and the mse loss.

        Any transformations to the loss should be done with this in-mind.
        """
        return combined_loss

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main
        /nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L535-L552 .

        We expose it here because it is used for logging. Consumes the output from `forward` as output.

        ONLY effects the validation loss in the logs.
        """
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                # legacy behavior, average over the number of microbatches
                avg = [x["avg"] for x in losses_reduced_per_micro_batch]
                loss = torch.cat(avg).mean()
                return loss

            from megatron.core import parallel_state

            loss_sum_and_ub_size = [
                x["loss_sum_and_ub_size"] for x in losses_reduced_per_micro_batch if x["loss_sum_and_ub_size"][1] > 0
            ]
            loss = (
                torch.vstack(loss_sum_and_ub_size).sum(dim=0)
                if len(loss_sum_and_ub_size) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            torch.distributed.all_reduce(
                loss,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # average over the total number of tokens across the global batch.
            loss = loss[0] / loss[1]
            return loss

        return torch.tensor(0.0, device=torch.cuda.current_device())


class MaxTokiRegressionHead(MegatronModule):
    """A Megatron MLP head. Used in the finetuned model.

    torch layers may be used here as well. Megatron's ColumnParallelLinear is used here as an example
    and is fully functional with tensor parallelism.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.linear_fc2 = Linear(
            config.hidden_size, 1, bias=True
        )  # Alternatively could use ColumnParallelLinear with gather_output=True
        with torch.no_grad():
            torch.nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=1.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Consider- summing up to EOQ instead of just using the last token.
        # Note that this would require considerable changes to how the loss is used downstream.
        # Hidden states is [L, B, H]
        _logits = self.linear_fc2(hidden_states)
        # Second element is only non-None when gather_output is False and bias = True
        return _logits


class FinetuneLlamaModel(MCoreGPTModel):
    """Megatron model for llama fine-tuning."""

    def __init__(
        self,
        config,  # TransformerConfig, MaxTokiConfig probably...
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "mrope", "none"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        """Copy pasted the entire signature from GPTModel, this exposes whats actually there, but in practice could
        make this easier on the eyes by using the kwargs/args parameters to get the params we care about.

        post_process=True => do the post_process call (which computes loss per token)
        """
        # Instantiate our custom task head.
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        # Attach custom task head.
        self.regression_head = MaxTokiRegressionHead(config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> dict:
        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        standard_output = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )
        if not self.post_process:
            # In the middle of a pipeline, just return the standard output.
            return standard_output

        # Add a regression head or whatever is of interest.
        regression_output = self.regression_head(hidden_states)

        # Re-pack the outputs in a dictionary for our loss reduction.
        return {
            "regression_outputs": regression_output,
            "lm_outputs": standard_output,
        }


MegatronMaxTokiModelType = TypeVar("MegatronMaxTokiModelType", bound=MCoreGPTModel)


@dataclass
class MaxTokiConfig(
    Llama32Config1B,
    MegatronBioNeMoTrainableModelConfig[MegatronMaxTokiModelType, MegatronLossType],
    iom.IOMixinWithGettersSetters,
    Generic[MegatronMaxTokiModelType, MegatronLossType],
):
    # metric logging - required for our ligthning stuff...
    train_metric: Optional[TorchmetricsConfig] = None
    valid_metric: Optional[TorchmetricsConfig] = None

    # Dont expose this. This is the wrong parameter?
    rope_scaling_factor: float = 8.0

    # scale_factor is the thing we care about.
    scale_factor: float = 8.0

    # Loss reduction, also required to get custom loss.
    loss_reduction_class: Type[MegatronLossType] = MaskedTokenLossReduction

    # Model cls, allows generic usage.
    model_cls: Type[MegatronMaxTokiModelType] = MCoreGPTModel  # type: ignore

    def configure_model(
        self, tokenizer, pre_process=None, post_process=None, vp_stage=None, **kwargs
    ) -> MegatronMaxTokiModelType:
        """Configures and instantiates the underlying model using the saved checkpoint, denoted with self.model_cls.

        This method is lifted from GPTModel.configure_model as it was not designed to take an arbitrary child of MCoreGPTModel.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            model (MCoreGPTModel-like): Configured Megatron Core GPT model instance or variant.
        """
        if self.initial_ckpt_path:
            # overrides config settings using the config inside the pre-trained checkpoint.
            self.load_settings_from_checkpoint(self.initial_ckpt_path)

            # TE Flash Attention requires head_dim % 8 == 0. When this fails, both TE and
            # Megatron's fallback use O(N^2) unfused attention (~33 GB/layer at seq_len=16384).
            # Auto-switch to SDPA which handles non-aligned head dims via internal padding.
            head_dim = self.hidden_size // self.num_attention_heads
            if head_dim % 8 != 0:
                from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec
                self.transformer_layer_spec = sdpa_layer_spec
                self.persist_layer_norm = False
                logging.info(
                    f"Auto-enabling SDPA attention: head_dim={head_dim} "
                    f"(hidden_size={self.hidden_size} / num_attention_heads={self.num_attention_heads}) "
                    f"is not divisible by 8. TE Flash Attention requires head_dim % 8 == 0."
                )

        # Stuff from MCoreGPTModel...
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        import inspect

        from megatron.core import parallel_state

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is not None:
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device="meta")

        if "mtp_block_spec" in inspect.signature(MCoreGPTModel.__init__).parameters:
            kwargs = {"mtp_block_spec": mtp_block_spec(self, vp_stage=vp_stage)}
        else:
            kwargs = {}
        with model_init_device_context():
            # NOTE: this keeps the class generic while still allowing children to define the type associations.
            if issubclass(self.model_cls, MaxTokiFineTuneModel):
                if not isinstance(tokenizer, MaxTokiTokenizer):
                    raise ValueError("Tokenizer must be a MaxTokiTokenizer for MaxToki fine-tuning models.")
                kwargs["numeric_mask"] = tokenizer.build_numeric_mask(vocab_size)
                kwargs["vocab_to_numeric_map"] = tokenizer.build_numeric_vocab_to_numeric_map(vocab_size)
                kwargs["penalty_factor"] = self.penalty_factor

            model = self.model_cls(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rope_scaling_factor=self.rope_scaling_factor,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process
                or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=post_process
                or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                vp_stage=vp_stage,
                **kwargs,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for module in self.get_model_module_list():
                    for index, child in enumerate(module.modules()):
                        if index == 0:
                            continue
                        if hasattr(child, "set_context_parallel_group"):
                            child.set_context_parallel_group(
                                parallel_state.get_context_parallel_group(),
                                parallel_state.get_context_parallel_global_ranks(),
                                cp_stream,
                            )

        # NOTE(SKH) - this overrides the scaling applied in mcore. the config attribute rope_scaling controls this.
        model.rotary_pos_emb.inv_freq = apply_rope_scaling(
            model.rotary_pos_emb.inv_freq,
            factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )

        if self.initial_ckpt_path is not None:
            logging.info(f"Loading weights from {self.initial_ckpt_path}")
            self.update_model_from_checkpoint(model, self.initial_ckpt_path)
            logging.info(f"Done!! Weights restored from {self.initial_ckpt_path}")

        return model

    def get_loss_reduction_class(self) -> Type[MegatronLossType]:
        # You could optionally return a different loss reduction class here based on the config settings.
        return self.loss_reduction_class


@dataclass
class MaxTokiFineTuneConfig(
    MaxTokiConfig[FinetuneLlamaModel, FineTuneLlamaLanguageModelLossWithReduction],
    iom.IOMixinWithGettersSetters,
):
    """MaxToki fine-tuning model configuration. Enables the ability to skip the loading of certain model weights as well
    as attaching a custom loss reduction class to the underlying model.
    """

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[FinetuneLlamaModel] = FinetuneLlamaModel

    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(
        default_factory=lambda: ["regression_head", "max_sequence_length"]
    )

    def get_loss_reduction_class(self) -> Type[FineTuneLlamaLanguageModelLossWithReduction]:
        """Attaches the custom loss reduction class for finetuning."""
        return FineTuneLlamaLanguageModelLossWithReduction


@dataclass
class MaxTokiMultitaskFineTuneConfig(
    MaxTokiConfig[MaxTokiFineTuneModel, MaxTokiLossWithReduction],
    iom.IOMixinWithGettersSetters,
):
    """MaxToki multitask fine-tuning model configuration. Connects the MaxTokiFineTuneModel with our custom loss reduction
    used by multitask finetuning.
    """

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[MaxTokiFineTuneModel] = MaxTokiFineTuneModel

    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])
    # Update the default behavior to respect sequence length- relevant for RoPE scaling.
    override_parent_fields: List[str] = field(
        default_factory=lambda: _OVERRIDE_BIONEMO_CONFIG_DEFAULTS + ["seq_length", "scale_factor", "transformer_layer_spec", "persist_layer_norm"]
    )

    penalty_factor: float | None = None
    # first order scalar applied to the mse loss
    label_scalar: float = 1.0

    # Multiplied by the weight(probability) of non-numeric tokens and then added to the loss.
    additive_penalty: float = 10.0
    timelapse_loss: Literal["mse", "ce"] = "ce"

    def get_loss_reduction_class(self) -> Type[MaxTokiLossWithReduction]:
        """Attaches the custom loss reduction class for finetuning."""
        return MaxTokiLossCEOnlyWithReduction if self.timelapse_loss == "ce" else MaxTokiLossWithReduction


def maxtoki_headless_predict_step(
    model, batch, batch_idx: Optional[int] = None, using_pretrain_dataset: bool = True
) -> Dict[str, Any]:
    # Hacky stuff for verifying correctness.
    if using_pretrain_dataset:
        new_batch = filter_pretraining_batch(batch, task_type="TimeBetweenCells")

        # NOTE: filtering isnt strictly necessary- as the lack of filtering will simply truncate to EOQ, which does the right thing anyway.
        new_batch = batch
        if new_batch == {}:
            return {}
        batch = new_batch

    if "attention_mask" not in batch:
        batch["attention_mask"] = None

    outputs = model.forward_step(batch)

    # 2025-11-20: Manually verified that we get the same predictions using predict_step as we do when computing the loss.
    #               model appears to collapse and always predict close to the mean.
    eoqs = find_eoq_indices(batch["tokens"], model.tokenizer.token_dictionary["<eoq>"])
    predictions = torch.gather(outputs["regression_preds"], dim=0, index=eoqs.unsqueeze(0))
    timelapse_token_preds = torch.gather(outputs["timelapse_token_preds"], dim=0, index=eoqs.unsqueeze(0))
    return {
        # [1, B]
        "regression_preds": predictions,
        "timelapse_token_preds": timelapse_token_preds,
    }


def filter_pretraining_batch(
    batch: Dict[str, Tensor], task_type: Literal["TimeBetweenCells", "NextCell"]
) -> Dict[str, Tensor]:
    """Filters a batch to only include the inputs for the given task type.

    Args:
        batch: The batch to filter.
        task_type: The type of task to filter for.

    Returns:
        The filtered batch.
    """
    retain_idxs = []
    if batch == {} or "loss_mask" not in batch:
        return {}
    for i, b in enumerate(batch["loss_mask"]):
        if b.sum() <= 1 and task_type == "TimeBetweenCells":
            retain_idxs.append(i)
        elif b.sum() > 1 and task_type == "NextCell":
            retain_idxs.append(i)

    if retain_idxs == []:
        return {}

    retain = {}
    for k, v in batch.items():
        if v is None:
            continue
        retain[k] = v[retain_idxs]

    return retain


def maxtoki_forward_step(model, batch, **kwargs) -> torch.Tensor:
    """Execute a forward step for the GPT model.

    This function prepares the arguments needed for the model's forward pass
    and handles both normal and packed sequence processing.

    Lifted (mostly) from NeMo/nemo/collections/llm/gpt/model/base.py:gpt_forward_step

    Args:
        model: The GPT model
        batch: The input batch containing tokens, positions, and other required inputs

    Returns:
        torch.Tensor: Output tensor from the model forward pass


    # There are some defaults we need to be aware of.
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
    ) -> torch.Tensor:

    call chain
        LM = lightning module
        LM.forward_step() -> LM._forward_step() -> model.forward()

    """
    forward_kwargs = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"] if "position_ids" in batch else None,
        "labels": batch["labels"] if "labels" in batch else None,
    }

    if "attention_mask" not in batch:
        assert HAVE_TE, (
            "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
        )

        forward_kwargs["attention_mask"] = None
    else:
        forward_kwargs["attention_mask"] = batch["attention_mask"]

    if "cu_seqlens" in batch:
        forward_kwargs["packed_seq_params"] = get_packed_seq_params(batch)

    forward_kwargs.update({})

    # can we do a predict step here just for fun
    return model(**forward_kwargs)
