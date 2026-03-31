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

import torch
import pytest

from bionemo.maxtoki.tokenizer import MaxTokiTokenizer, find_eoq_index
from bionemo.llm.data.collate import hf_llama_padding_collate_fn


# ---------------------------------------------------------------------------
# CellformerTokenizer init
# ---------------------------------------------------------------------------


def test_special_tokens_keys(tokenizer):
    expected_keys = {"<bos>", "<eos>", "<boq>", "<eoq>", "<pad>", "<mask>"}
    assert set(tokenizer.special_tokens.keys()) == expected_keys


def test_special_tokens_ids(tokenizer, synthetic_token_dictionary):
    for name in ("<bos>", "<eos>", "<boq>", "<eoq>", "<pad>", "<mask>"):
        assert tokenizer.special_tokens[name] == synthetic_token_dictionary[name]


def test_numeric_tokens_are_string_ints(tokenizer):
    for token_str in tokenizer.numeric_tokens:
        int(token_str)  # must not raise


def test_numeric_tokens_content(tokenizer, synthetic_token_dictionary):
    expected = {str(i): 16 + i for i in range(10)}
    assert tokenizer.numeric_tokens == expected


def test_numeric_token_ids_inverse(tokenizer):
    for token_id, token_str in tokenizer.numeric_token_ids.items():
        assert tokenizer.numeric_token_ids[token_id] == token_str


def test_vocab_size(tokenizer, synthetic_token_dictionary):
    assert tokenizer.vocab_size == len(synthetic_token_dictionary)


def test_shorthand_properties(tokenizer, synthetic_token_dictionary):
    assert tokenizer.bos_id == synthetic_token_dictionary["<bos>"]
    assert tokenizer.eos_id == synthetic_token_dictionary["<eos>"]
    assert tokenizer.eoq_id == synthetic_token_dictionary["<eoq>"]
    assert tokenizer.boq_id == synthetic_token_dictionary["<boq>"]
    assert tokenizer.pad_id == synthetic_token_dictionary["<pad>"]
    assert tokenizer.mask_id == synthetic_token_dictionary["<mask>"]


# ---------------------------------------------------------------------------
# build_numeric_mask / build_numeric_vocab_to_numeric_map
# ---------------------------------------------------------------------------


def test_build_numeric_mask_shape_and_dtype(tokenizer):
    mask = tokenizer.build_numeric_mask()
    assert mask.shape == (tokenizer.vocab_size,)
    assert mask.dtype == torch.bool


def test_build_numeric_mask_values(tokenizer):
    mask = tokenizer.build_numeric_mask()
    for token_id in tokenizer.numeric_token_ids:
        assert mask[token_id].item() is True
    non_numeric_count = mask.sum().item()
    assert non_numeric_count == len(tokenizer.numeric_tokens)


def test_build_numeric_mask_padded(tokenizer):
    padded_size = 32
    mask = tokenizer.build_numeric_mask(vocab_size=padded_size)
    assert mask.shape == (padded_size,)
    assert mask[tokenizer.vocab_size:].any().item() is False


def test_build_numeric_vocab_to_numeric_map_shape(tokenizer):
    m = tokenizer.build_numeric_vocab_to_numeric_map()
    assert m.shape == (tokenizer.vocab_size,)
    assert m.dtype == torch.float


def test_build_numeric_vocab_to_numeric_map_values(tokenizer):
    m = tokenizer.build_numeric_vocab_to_numeric_map()
    for token_id, token_str in tokenizer.numeric_token_ids.items():
        assert m[token_id].item() == float(token_str)
    # Non-numeric positions should be 0.0
    for i in range(tokenizer.vocab_size):
        if i not in tokenizer.numeric_token_ids:
            assert m[i].item() == 0.0


def test_build_numeric_vocab_to_numeric_map_padded(tokenizer):
    padded_size = 32
    m = tokenizer.build_numeric_vocab_to_numeric_map(vocab_size=padded_size)
    assert m.shape == (padded_size,)
    assert (m[tokenizer.vocab_size:] == 0.0).all()


# ---------------------------------------------------------------------------
# find_eoq_index
# ---------------------------------------------------------------------------


def test_find_eoq_index_present(tokenizer):
    #  <bos> GENE0 GENE1 <eos> <boq> <eoq> <bos> GENE2 <eos>
    seq = [2, 4, 5, 3, 14, 15, 2, 6, 3]
    idx = tokenizer.find_eoq_index(seq)
    assert idx == 5


def test_find_eoq_index_missing(tokenizer):
    seq = [2, 4, 5, 3]
    with pytest.raises(ValueError):
        tokenizer.find_eoq_index(seq)


def test_find_eoq_index_standalone_function(synthetic_token_dictionary):
    eoq_token = synthetic_token_dictionary["<eoq>"]
    seq = [2, 4, 15, 3]
    assert find_eoq_index(seq, eoq_token) == 2


# ---------------------------------------------------------------------------
# determine_task_type
# ---------------------------------------------------------------------------


def test_determine_task_type_time_between_cells(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> "5" "3"
    numeric_5 = tokenizer.numeric_tokens["5"]
    numeric_3 = tokenizer.numeric_tokens["3"]
    seq = [2, 4, 3, 14, 15, numeric_5, numeric_3]
    eoq_idx = 4
    assert tokenizer.determine_task_type(seq, eoq_idx) == "TimeBetweenCells"


def test_determine_task_type_next_cell(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> <bos> GENE1 <eos>
    seq = [2, 4, 3, 14, 15, 2, 5, 3]
    eoq_idx = 4
    assert tokenizer.determine_task_type(seq, eoq_idx) == "NextCell"


def test_determine_task_type_invalid(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> GENE1  (GENE1 is neither numeric nor <bos>)
    seq = [2, 4, 3, 14, 15, 4]
    eoq_idx = 4
    with pytest.raises(ValueError, match="Invalid grammar"):
        tokenizer.determine_task_type(seq, eoq_idx)


# ---------------------------------------------------------------------------
# create_loss_mask
# ---------------------------------------------------------------------------


def test_loss_mask_time_between_cells(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> "5" "3"
    numeric_5 = tokenizer.numeric_tokens["5"]
    numeric_3 = tokenizer.numeric_tokens["3"]
    seq = [2, 4, 3, 14, 15, numeric_5, numeric_3]
    mask = tokenizer.create_loss_mask(seq, "TimeBetweenCells", eoq_index=4)
    assert len(mask) == len(seq)
    # All zeros except penultimate
    assert mask == [0, 0, 0, 0, 0, 1, 0]


def test_loss_mask_next_cell_mask_bos_true(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> <bos> GENE1 GENE2 <eos>
    seq = [2, 4, 3, 14, 15, 2, 5, 6, 3]
    eoq_idx = 4
    mask = tokenizer.create_loss_mask(seq, "NextCell", eoq_idx, mask_bos_next_cell=True)
    assert len(mask) == len(seq)
    # prefix through eoq: all 0
    assert mask[:5] == [0, 0, 0, 0, 0]
    # postfix: bos masked (0), then 1s, last masked (0)
    assert mask[5] == 0  # <bos> masked
    assert mask[6] == 1
    assert mask[7] == 1
    assert mask[8] == 0  # last token always masked


def test_loss_mask_next_cell_mask_bos_false(tokenizer):
    # <bos> GENE0 <eos> <boq> <eoq> <bos> GENE1 GENE2 <eos>
    seq = [2, 4, 3, 14, 15, 2, 5, 6, 3]
    eoq_idx = 4
    mask = tokenizer.create_loss_mask(seq, "NextCell", eoq_idx, mask_bos_next_cell=False)
    assert len(mask) == len(seq)
    assert mask[:5] == [0, 0, 0, 0, 0]
    assert mask[5] == 1  # <bos> NOT masked
    assert mask[6] == 1
    assert mask[7] == 1
    assert mask[8] == 0  # last still masked


def test_loss_mask_last_token_always_zero(tokenizer):
    for task in ("TimeBetweenCells", "NextCell"):
        if task == "TimeBetweenCells":
            numeric_5 = tokenizer.numeric_tokens["5"]
            seq = [2, 4, 3, 14, 15, numeric_5, numeric_5]
        else:
            seq = [2, 4, 3, 14, 15, 2, 5, 3]
        eoq_idx = 4
        mask = tokenizer.create_loss_mask(seq, task, eoq_idx)
        assert mask[-1] == 0


# ---------------------------------------------------------------------------
# create_position_ids_simple
# ---------------------------------------------------------------------------


def test_position_ids_simple_default(tokenizer):
    seq = [2, 4, 5, 3]
    pos = tokenizer.create_position_ids_simple(seq)
    assert torch.equal(pos, torch.arange(4))


def test_position_ids_simple_offset(tokenizer):
    seq = [2, 4, 5, 3]
    pos = tokenizer.create_position_ids_simple(seq, begin_rank_with=10)
    assert torch.equal(pos, torch.arange(10, 14))


# ---------------------------------------------------------------------------
# collate_batch_multitask (finetuning collate)
# ---------------------------------------------------------------------------


def _make_next_cell_sample(tokenizer):
    # <bos> GENE0 GENE1 <eos> <boq> <eoq> <bos> GENE2 GENE3 <eos>
    return {"input_ids": [2, 4, 5, 3, 14, 15, 2, 6, 7, 3]}


def _make_tbc_sample(tokenizer):
    # <bos> GENE0 GENE1 <eos> <boq> <eoq> "5" "3"
    n5 = tokenizer.numeric_tokens["5"]
    n3 = tokenizer.numeric_tokens["3"]
    return {"input_ids": [2, 4, 5, 3, 14, 15, n5, n3]}


def test_collate_multitask_output_keys(tokenizer):
    batch = [_make_next_cell_sample(tokenizer)]
    result = tokenizer.collate_batch_multitask(batch, padding_value=0)
    assert set(result.keys()) >= {"tokens", "position_ids", "loss_mask", "labels"}


def test_collate_multitask_shapes(tokenizer):
    batch = [_make_next_cell_sample(tokenizer), _make_next_cell_sample(tokenizer)]
    result = tokenizer.collate_batch_multitask(batch, padding_value=0, min_length=16)
    for key in ("tokens", "position_ids", "loss_mask", "labels"):
        assert result[key].shape[0] == 2
        assert result[key].shape[1] >= 16


def test_collate_multitask_padding(tokenizer):
    short = {"input_ids": [2, 4, 3, 14, 15, 2, 5, 3]}  # length 8
    result = tokenizer.collate_batch_multitask([short], padding_value=0, min_length=12)
    tokens = result["tokens"][0]
    assert tokens.shape[0] >= 12
    # Padded region should be padding_value
    assert (tokens[8:] == 0).all()


def test_collate_multitask_loss_mask_padding_region(tokenizer):
    short = {"input_ids": [2, 4, 3, 14, 15, 2, 5, 3]}
    result = tokenizer.collate_batch_multitask([short], padding_value=0, min_length=12)
    loss_mask = result["loss_mask"][0]
    # Padded region loss_mask should be 0
    assert (loss_mask[8:] == 0).all()


def test_collate_multitask_labels_are_rolled(tokenizer):
    sample = _make_next_cell_sample(tokenizer)
    result = tokenizer.collate_batch_multitask([sample], padding_value=0)
    tokens = result["tokens"][0]
    labels = result["labels"][0]
    seq_len = len(sample["input_ids"])
    # Where loss_mask is 1, labels should be the next token
    loss_mask = result["loss_mask"][0]
    for i in range(seq_len - 1):
        if loss_mask[i] == 1:
            assert labels[i].item() == tokens[i + 1].item()


def test_collate_multitask_last_label_is_padding(tokenizer):
    sample = _make_next_cell_sample(tokenizer)
    pad_val = 0
    result = tokenizer.collate_batch_multitask([sample], padding_value=pad_val)
    seq_len = len(sample["input_ids"])
    labels = result["labels"][0]
    assert labels[seq_len - 1].item() == pad_val


# ---------------------------------------------------------------------------
# hf_llama_padding_collate_fn (pretraining collate)
# ---------------------------------------------------------------------------


def test_hf_llama_collate_output_keys():
    batch = [{"input_ids": [10, 20, 30, 40]}]
    result = hf_llama_padding_collate_fn(batch, padding_value=0)
    assert set(result.keys()) >= {"tokens", "position_ids", "loss_mask", "labels"}


def test_hf_llama_collate_loss_mask_all_ones():
    batch = [{"input_ids": [10, 20, 30, 40]}]
    result = hf_llama_padding_collate_fn(batch, padding_value=0)
    loss_mask = result["loss_mask"][0]
    assert (loss_mask[:4] == 1.0).all()


def test_hf_llama_collate_labels_rolled():
    ids = [10, 20, 30, 40]
    batch = [{"input_ids": ids}]
    result = hf_llama_padding_collate_fn(batch, padding_value=0)
    labels = result["labels"][0]
    assert labels[0].item() == 20
    assert labels[1].item() == 30
    assert labels[2].item() == 40
    assert labels[3].item() == 0  # last = padding_value


def test_hf_llama_collate_position_ids():
    batch = [{"input_ids": [10, 20, 30]}]
    result = hf_llama_padding_collate_fn(batch, padding_value=0, begin_rank_with=5)
    pos = result["position_ids"][0]
    assert torch.equal(pos[:3], torch.tensor([5, 6, 7]))


def test_hf_llama_collate_padding():
    short = [{"input_ids": [10, 20]}]
    long = [{"input_ids": [10, 20, 30, 40]}]
    batch = short + long
    result = hf_llama_padding_collate_fn(batch, padding_value=0)
    tokens = result["tokens"]
    assert tokens.shape == (2, 4)
    # First sample is padded
    assert tokens[0, 2].item() == 0
    assert tokens[0, 3].item() == 0
    # Padding region loss_mask = 0
    loss_mask = result["loss_mask"]
    assert loss_mask[0, 2].item() == 0.0
    assert loss_mask[0, 3].item() == 0.0


def test_hf_llama_collate_min_length():
    batch = [{"input_ids": [10, 20]}]
    result = hf_llama_padding_collate_fn(batch, padding_value=0, min_length=8)
    assert result["tokens"].shape[1] >= 8
