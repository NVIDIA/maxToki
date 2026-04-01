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


"""This captures the 'generate' workflow for the MaxToki model.

Given a prompt ([some input]<boq><prompt><eoq>):
    - generate the next cells gene expression vector.
    - the input may contain multiple cells, the boq/eoq pair wrap a timespan, which is the target for what will be predicted.


Implementation notes:
- modifications are needed for prompt handling
- this fits into the lightning workflow as a predict step
- need to be mindful/respectful of the input dataset class


Two paths:
    1) implement generate directly - requires background research and ensuring correctness
    2) use inferencewrapper/generate to go through the mcore generate engine. unclear exactly what constraints are required & how we benefit.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import tqdm
from megatron.core.inference.contexts import DynamicInferenceContext, TokenOverflowError
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from nemo.utils.exp_manager import TimingCallback

from bionemo.maxtoki.tokenizer import find_eoq_indices


if TYPE_CHECKING:
    from bionemo.maxtoki.tokenizer import MaxTokiTokenizer


def setup_default_sampling_mask(tokenizer: MaxTokiTokenizer):
    special_token_ids = [v for k, v in tokenizer.special_tokens.items() if k != "<eos>" and k != "<bos>"]
    numeric_token_ids = [v for v in tokenizer.numeric_tokens.values()]
    return special_token_ids + numeric_token_ids


def maxtoki_generate_predict_step_naive(
    model,
    batch,
    batch_idx: Optional[int] = None,
    max_tokens_to_generate: int = 4096,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    using_pretrain_dataset: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Naive implementation of the generate step. No sequence packing, no KV cache. Useful mostly
    for benchmarking and testing correctness. Not recommended for any real usecase.
    """
    if batch == {}:
        return {}

    if eos_token_id is None:
        eos_token_id = model.tokenizer.special_tokens["<eos>"]
    pad_id = model.tokenizer.pad_token_id
    eoq_token_id = model.tokenizer.special_tokens["<eoq>"]

    sampling_params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

    tokens = batch["tokens"]
    device = tokens.device
    B = tokens.size(0)

    # Build per-request prompt token lists (same behavior as the optimized path)
    eoq_indices = find_eoq_indices(tokens, eoq_token_id)

    tokens_list: List[torch.Tensor] = []
    if using_pretrain_dataset:
        for i in range(B):
            prompt_len = int(eoq_indices[i].item()) + 1
            tokens_list.append(tokens[i, :prompt_len])
    else:
        tokens_list = [tokens[i] for i in range(B)]

    # Per-request "don't generate these" mask (matches the optimized logic)
    sampling_masks = [setup_default_sampling_mask(model.tokenizer) for _ in range(B)]

    generated_sequences = {i: [] for i in range(B)}
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    total_generated = 0

    pbar = setup_tqdm(max_tokens_to_generate, B)
    step = 0
    for i in range(B):
        seq = tokens_list[i].clone()

        for _step in range(max_tokens_to_generate):
            # Full forward over the entire sequence-so-far (naive)
            input_ids = seq.unsqueeze(0)  # [1, L]
            pos_ids = torch.arange(seq.numel(), device=device).unsqueeze(0)  # [1, L]

            out = model.forward(
                input_ids,
                pos_ids,
                runtime_gather_output=True,
                inference_context=None,
                next_token_only=False,  # doesn't matter much here; full prefix is computed
            )
            logits = out["lm_outputs"].squeeze(0)  # [1, L, V] -> [L, V] or [L,V] depending wrapper
            last_logits = logits[-1].unsqueeze(0)  # [1, V]

            # Apply "invalid ids" mask for this request
            bad_ids = sampling_masks[i]
            last_logits[:, bad_ids] = float("-inf")

            next_tok = sample_from_logits(
                last_logits,
                sampling_params=sampling_params,
                vocab_size=model.tokenizer.vocab_size,
            )[0].item()
            total_generated += 1

            generated_sequences[i].append(next_tok)
            sampling_masks[i].append(next_tok)

            seq = torch.cat([seq, torch.tensor([next_tok], device=device, dtype=seq.dtype)], dim=0)

            step += 1
            update_pbar(pbar, 1, step)

            if next_tok == eos_token_id:
                finished[i] = True
                break

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            pbar.close()

    # Pack outputs like the optimized function
    max_gen = max((len(v) for v in generated_sequences.values()), default=0)
    gen = torch.full((B, max_gen), pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    finished_naturally = torch.zeros(B, dtype=torch.bool, device=device)

    for i in range(B):
        toks = generated_sequences[i]
        if toks:
            gen[i, : len(toks)] = torch.tensor(toks, device=device, dtype=torch.long)
            lengths[i] = len(toks)
            finished_naturally[i] = toks[-1] == eos_token_id

    full = []
    for i in range(B):
        prompt = tokens_list[i]
        full_i = torch.cat([prompt, gen[i, : lengths[i]]], dim=0)
        full.append(full_i)

    max_full = max((t.numel() for t in full), default=0)
    full_t = torch.full((B, max_full), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(full):
        full_t[i, : t.numel()] = t

    return {
        "generated_tokens": gen.cpu(),
        "lengths": lengths.cpu(),
        "finished_naturally": finished_naturally.cpu(),
        "full_sequence": full_t.cpu(),
    }


def maxtoki_generate_predict_step(
    model,
    batch,
    batch_idx: Optional[int] = None,
    max_tokens_to_generate: int = 4096,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    using_pretrain_dataset: bool = False,
    # Inference context parameters.
    buffer_size_gb=20.0,
    buffer_guaranteed_fraction=0.1,
    chunk_size_tokens=4096,
    buffer_overflow_factor=1.0,
) -> Dict[str, Any]:
    """Generate gene expression predictions autoregressively until EOS or max length with KV caching.

    Uses DynamicInferenceContext for efficient KV caching and sequence packing. Continues generating until EOS or max length is reached.
    Tokens may not be generated more than once, special tokens other than <BOS>/<EOS> are not allowed to be generated.
    Output is a dictionary of tensors which includes a batch dimension. Tensor is padded with <PAD>.

    Prompt is everything up to and including <eoq> token.

    Args:
        model: LightningModule holding the finetuned MaxToki model.
        batch: Input mini-batch containing atleast the element 'tokens'.
            Unless using the pretraining dataset- this should be an unpadded sequence that ends with <EOQ>.
        max_tokens_to_generate: Maximum NEW tokens to generate (stops here if EOS not reached)
        eos_token_id: EOS token ID to stop generation. If None, uses model.tokenizer.eos_id
        temperature/top_k/top_p: Sampling parameters (ignored if sampling_params provided)
        invalid_token_ids: Set of token IDs that should never be sampled
        using_pretrain_dataset: If true, assumes `batch` has the same structure as the pretraining dataset. This means:
            - 'tokens' tensor contains labels after <eoq>
            - 'tokens' tensor contains padding.
            This flag will truncate the sequence after <eoq>.
        sampling_params: Optional SamplingParams object. If provided, overrides temperature/top_k/top_p

    Returns:
        Dictionary containing:
            - 'generated_tokens': [batch_size, variable_length] generated tokens (padded with EOS)
            - 'lengths': [batch_size] actual length of each generated sequence
            - 'finished': [batch_size] boolean indicating which sequences hit EOS vs max_length
            - 'full_sequence': [batch_size, prompt_len + generated_len] full sequence (prompt + generated)

    """
    if batch == {}:
        # Sometimes this can happen when DDP is enabled and there is an unbalanced batch.
        return {}

    # NOTE- is there a better place for this? should it be passed in? owned by the tokenizer?
    #     This initializes by disabling all special tokens other than bos/eos and disabling all numeric tokens.
    sampling_masks = [setup_default_sampling_mask(model.tokenizer) for _ in range(batch["tokens"].size(0))]

    # Get EOS token ID
    if eos_token_id is None:
        eos_token_id = model.tokenizer.special_tokens["<eos>"]

    # Ensure eos_token_id is valid
    assert eos_token_id is not None, "eos_token_id cannot be None"
    assert model.tokenizer.pad_token_id is not None, "pad_token_id cannot be None"

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    """ Some assumptions that I must pay attention to:
    - dataset we are using includes the labels in the input sequence
    - there is some overflow limit
    - 16389 is the max sequence length, inherited from pretraining.
        4096 * 4 (4 sequences)
        + 4 (eos tokens)
        + 1 (timelapse label)

    special_tokens:
    {'<bos>': 2, '<eos>': 3, '<eoq>': 23276, '<boq>': 23275, '<pad>': 0, '<mask>': 1}

    Is this an issue with timelapse vs next cell prediction?
    """
    tokens = batch["tokens"]
    tokens_list = []
    batch_size, initial_seq_len = tokens.shape
    device = tokens.device

    # Find <eoq> token indices - this marks end of prompt
    eoq_token_id = model.tokenizer.special_tokens["<eoq>"]
    eoq_indices = find_eoq_indices(tokens, eoq_token_id)  # [batch_size] - indices along seq dim

    # If using pretraining dataset, truncate to prompt only (up to and including <eoq>)
    if using_pretrain_dataset:
        # Update eoq_indices since we truncated
        eoq_indices = find_eoq_indices(tokens, eoq_token_id)

        # Regardless of pretrain.
        initial_seq_len = tokens.shape[1]

        # Truncate each sequence individually
        for i in range(batch_size):
            actual_length = eoq_indices[i] + 1
            actual_tokens = tokens[i, :actual_length]
            tokens_list.append(actual_tokens)
    else:
        # In this case we assume the sequence ends in <EOQ> and contains no padding.
        tokens_list = [tokens[i] for i in range(len(tokens))]

    if "labels" in batch:
        # Remove labels as we're generating, not training
        del batch["labels"]

    max_seq_length = initial_seq_len + max_tokens_to_generate

    # Sets up KV caching and sequence packing.
    # num_attention_heads - actually wants the number of query groups.
    inference_context = DynamicInferenceContext(
        params_dtype=torch.bfloat16,
        num_layers=model.config.num_layers,
        kv_channels=model.config.kv_channels,
        num_attention_heads=model.config.num_query_groups,
        buffer_size_gb=buffer_size_gb,
        buffer_guaranteed_fraction=buffer_guaranteed_fraction,
        chunk_size_tokens=chunk_size_tokens,
        buffer_overflow_factor=buffer_overflow_factor,
        tensor_model_parallel_size=model.config.tensor_model_parallel_size,
        max_sequence_length=max_seq_length,
    )

    try:
        for i in range(batch_size):
            req = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=tokens_list[i],
                sampling_params=SamplingParams(num_tokens_to_generate=max_tokens_to_generate),
            )
            inference_context.add_request(req)
    except TokenOverflowError as e:
        raise TokenOverflowError(
            f"Token overflow error, likely due to insufficient memory. Decrease batch size or increase buffer size: {e}"
        ) from e

    pbar = setup_tqdm(max_tokens_to_generate, batch_size)
    generated_sequences = {i: [] for i in range(batch_size)}
    mask = None
    step = 0

    while inference_context.has_unfinished_requests():
        inference_context.initialize_attention_state()

        # NOTE: These are packed - thanks dynamic inference context!
        input_ids, pos_ids = inference_context.current_input_and_position_ids()

        outputs = model.forward(
            input_ids, pos_ids, runtime_gather_output=True, inference_context=inference_context, next_token_only=True
        )

        logits = outputs["lm_outputs"].squeeze(0)  # [1, B, vocab_size] => [B, vocab_size]

        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Get current active request IDs (maps position in next_tokens to original request_id)
        current_request_ids = inference_context.request_ids[
            inference_context.paused_request_count : inference_context.total_request_count
        ]

        for idx, request_id in enumerate(current_request_ids):
            rid = int(request_id.item())
            mask[idx, sampling_masks[rid]] = True
        logits[mask] = float("-inf")

        # Sample next tokens
        next_tokens = sample_from_logits(
            logits, sampling_params=sampling_params, vocab_size=model.tokenizer.vocab_size
        )

        # Store generated tokens for each active request
        for idx, request_id in enumerate(current_request_ids):
            rid = int(request_id.item())
            generated_sequences[rid].append(next_tokens[idx].item())
            # Add to sampling mask for this request
            sampling_masks[rid].append(next_tokens[idx].item())

        # Keep going until we hit the max sequence length or generate EOS.
        active_mask = next_tokens != eos_token_id
        active_sequence_lengths = inference_context.get_active_sequence_lengths()
        max_sequence_lengths = inference_context.get_max_sequence_lengths()

        active_mask = (next_tokens != eos_token_id).byte() & torch.less(
            active_sequence_lengths, max_sequence_lengths
        ).byte()
        inference_context.update_requests(active_mask, next_tokens)

        step += 1
        update_pbar(pbar, active_mask.sum().item(), step)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            pbar.close()

    max_generated_length = max(len(seq) for seq in generated_sequences.values()) if generated_sequences else 0

    # We could return a list of tensors or a list of dicts rather than padding.
    generated_tokens_tensor = torch.full(
        (batch_size, max_generated_length), model.tokenizer.pad_token_id, dtype=torch.long, device=device
    )

    generation_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished_naturally = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for request_id, token_list in generated_sequences.items():
        if len(token_list) > 0:
            generated_tokens_tensor[request_id, : len(token_list)] = torch.tensor(
                token_list, dtype=torch.long, device=device
            )
            generation_lengths[request_id] = len(token_list)
            # Check if last token is EOS (finished naturally) or hit max length
            finished_naturally[request_id] = token_list[-1] == eos_token_id

    # Reconstruct full sequences (prompt + generated)
    full_sequences = []
    for i in range(batch_size):
        prompt = tokens_list[i]
        generated = generated_tokens_tensor[i, : generation_lengths[i]]
        full_seq = torch.cat([prompt, generated])
        full_sequences.append(full_seq)

    # Pad full sequences to same length
    max_full_length = max(len(seq) for seq in full_sequences)
    full_sequence_tensor = torch.full(
        (batch_size, max_full_length), model.tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    for i, seq in enumerate(full_sequences):
        full_sequence_tensor[i, : len(seq)] = seq

    return {
        "generated_tokens": generated_tokens_tensor.cpu(),
        "lengths": generation_lengths.cpu(),
        "finished_naturally": finished_naturally.cpu(),
        "full_sequence": full_sequence_tensor.cpu(),
    }


def sample_from_logits(
    last_token_logits: torch.Tensor,
    sampling_params: Optional[SamplingParams] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Samples the logits to generate outputs

    Lifted from mcore: 3rdparty/Megatron-LM/megatron/core/inference/text_generation_controllers/text_generation_controller.py:sample_from_logits
    Given the logits of the last token, this function samples it
    according to the parameters defined in sampling_params
    and returns the samples. If sampling parameters top_n_logprobs > 0
    at each step it also updates the top_n_logprobs dict.

    Args:
        last_token_logits (torch.Tensor): The last token logits. A tensor of
            size [batch_size, vocab_size]
        sampling_params (SamplingParams): The parameters to use for inference.
        vocab_size (int): Obtained from the tokenizer. Defaults to None

    Returns:
        sampled_logits (torch.Tensor): 1D tensor with [batch_size] elements
    """
    # Ensure sampling_params is not None
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=1.0, top_k=0, top_p=0.0)

    top_p = sampling_params.top_p
    top_k = sampling_params.top_k
    temperature = sampling_params.temperature

    assert isinstance(top_p, float)
    assert isinstance(top_k, int)
    assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
    assert top_p <= 1.0, "top-p should be in (0,1]"

    def modify_logits_for_top_k_filtering(logits, top_k):
        """Set the logits for none top-k values to -inf."""
        filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits.masked_fill_(filter_, float("-Inf"))

    def modify_logits_for_top_p_filtering(logits, top_p):
        """Set the logits for none top-p values to -inf."""
        # First sort and calculate cumulative sum of probabilities.
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Filteration based on the cumulative sum.
        filter_ = cumulative_probs > top_p
        # This shift by 1 is weird and I cannot justify it. This existed
        # in the original implementation:
        #   https://github.com/ari-holtzman/degen/blob/master/gen.py
        # and I guess it is needed so keeping it for now.
        filter_[:, 1:] = filter_[:, :-1].clone()
        # Make sure we at least have one token to select from.
        filter_[..., 0] = 0

        # Fill in the filtered part
        filter_ = filter_.scatter(1, sorted_indices, filter_)
        logits.masked_fill_(filter_, float("-Inf"))

    # Greedy sampling
    if top_k == 1:
        sampled_logits = torch.argmax(last_token_logits, dim=-1)
    else:
        last_token_logits = last_token_logits.clone()
        if temperature != 1.0:
            last_token_logits.div_(temperature)
        if top_k > 1:
            assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
            if vocab_size:
                assert top_k < vocab_size, "top-k is larger than vocab size."
            modify_logits_for_top_k_filtering(last_token_logits, top_k)

        elif top_p > 0.0:
            modify_logits_for_top_p_filtering(last_token_logits, top_p)

        # After filtering, we need to recalculate the distribution.
        probabilities = last_token_logits.softmax(dim=-1)

        sampled_logits = torch.multinomial(probabilities, num_samples=1).view(-1)

        # If vocab size is provided, make sure the samples are in in the range [0, vocab-size).
        if vocab_size:
            sampled_logits = torch.clamp(sampled_logits, min=0, max=(vocab_size - 1))

    return sampled_logits


def setup_tqdm(max_tokens_to_generate: int, batch_size: int):
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    max_possible_steps = max_tokens_to_generate * batch_size  # Upper bound
    pbar = tqdm.tqdm(
        total=max_possible_steps,
        desc="Generating tokens",
        disable=not is_main_process,  # Only show on rank 0
        dynamic_ncols=True,
        unit="tok",
    )
    return pbar


def update_pbar(pbar: tqdm.tqdm, items, step: int):
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    if is_main_process:
        # Update by number of active requests that generated a token
        pbar.update(items)
        pbar.set_postfix(
            {
                "remaining": items,
                "step": step,
            }
        )


class PredictTimingCallback(TimingCallback):
    def __init__(self, *args, show_batch_progress: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer_name = "predict_step_timing"
        self.show_batch_progress = show_batch_progress
        self._pbar = None

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """on_predict_batch_start"""
        self._on_batch_start(self.timer_name)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """on_predict_batch_end"""
        self.timer.stop(self.timer_name)
        if self.log_tokens_per_sec:
            if "text" in batch:
                batch["tokens"] = batch["text"]
            # Generation outputs have a "lengths" key; regression outputs do not.
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], dict) and "lengths" in outputs[0]:
                self.tokens_generated += outputs[0]["lengths"].sum().item()
        if self._pbar is not None:
            self._pbar.update(1)

    def on_predict_start(self, trainer, pl_module):
        self.tokens_generated = torch.tensor(0, device=pl_module.device)
        if self.show_batch_progress:
            is_main_process = True
            if torch.distributed.is_initialized():
                is_main_process = torch.distributed.get_rank() == 0
            total = sum(trainer.num_predict_batches) if trainer.num_predict_batches else None
            self._pbar = tqdm.tqdm(
                total=total,
                desc="Predicting (regression)",
                disable=not is_main_process,
                dynamic_ncols=True,
                unit="batch",
            )

    def on_predict_end(self, trainer, pl_module):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.tokens_generated, op=torch.distributed.ReduceOp.SUM)
            print(f"Total tokens generated: {self.tokens_generated}")
            print(f"Time taken: {self.timer[self.timer_name]}")
            print(f"Tokens per second: {self.tokens_generated / self.timer[self.timer_name]}")
            print(
                f"Tokens per second per GPU: {self.tokens_generated / self.timer[self.timer_name] / torch.distributed.get_world_size()}"
            )
        else:
            print(f"Total tokens generated: {self.tokens_generated}")
            print(f"Time taken: {self.timer[self.timer_name]}")
            print(f"Tokens per second: {self.tokens_generated / self.timer[self.timer_name]}")
            print(
                f"Tokens per second per GPU: {self.tokens_generated / self.timer[self.timer_name] / torch.distributed.get_world_size()}"
            )
