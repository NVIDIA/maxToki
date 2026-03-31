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

"""Shared utilities for the MaxToki data preparation pipeline."""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from typing import Any

import datasets
import numpy as np
from datasets import Dataset, Features, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def load_and_filter(input_data_file, filter_data=None, filter_out_times=None, index=None, nproc=1):
    """Load a dataset from disk and apply inclusion/exclusion filters."""
    data = load_from_disk(input_data_file)

    if index is not None:
        data = data.add_column("unfiltered_cell_indices", list(range(len(data))))
        data = data.add_column("unfiltered_dataset_indices", [index] * len(data))

    filter_data = filter_data or {}
    filter_out_times = filter_out_times or {}

    filter_data = {k: v for k, v in filter_data.items() if v is not None}
    filter_out_times = {k: v for k, v in filter_out_times.items() if v is not None}

    if filter_data:
        data = filter_by_dict(data, filter_data, nproc, exclude=False)

    if filter_out_times:
        data = filter_by_dict(data, filter_out_times, nproc, exclude=True)

    return data


def filter_example(example, key, value, exclude=False):
    """Return True if example passes the filter for a single key/value pair."""
    if exclude:
        return example[key] not in value
    else:
        return example[key] in value


def filter_by_dict(data, filter_data, nproc=1, exclude=False):
    """Filter dataset rows by a dictionary of {column: allowed_values}."""
    if not filter_data:
        return data

    for key, value in filter_data.items():
        if value is None:
            continue
        data = data.filter(
            lambda example, k=key, v=value, ex=exclude: filter_example(example, k, v, ex),
            num_proc=nproc,
        )
    return data


def smart_concatenate(dataset_list):
    """Concatenate datasets, aligning column feature types before merging.

    Plain ``concatenate_datasets`` raises an error when datasets share the same
    columns but with different dtypes (e.g. int32 vs int64 produced by different
    AnnData files). This function casts every dataset to the dtype of the first
    dataset for each column before concatenating.

    All datasets must have the same column names; missing columns are not added.
    """
    if not dataset_list:
        return dataset_list

    master_features = {}
    for ds in dataset_list:
        for col_name, feature in ds.features.items():
            if col_name not in master_features:
                master_features[col_name] = feature

    aligned = []
    for ds in dataset_list:
        target = Features({col: master_features[col] for col in ds.column_names})
        aligned.append(ds.cast(target) if ds.features != target else ds)

    return concatenate_datasets(aligned)


def load_datasets(
    data_files,
    filter_data=None,
    filter_out_times=None,
    time_column=None,
    nproc=1,
    group_by_dataset=False,
    time_column_conversion_dict=None,
):
    """Load one or more datasets, apply filters, and optionally concatenate."""
    valid_data_files = []
    dataset_list = []

    def _ensure_int_time_column(ds, time_column, time_column_conversion_dict, data_file_label):
        current_type = ds.features[time_column].dtype

        if "int" not in current_type and time_column_conversion_dict is not None:

            def map_to_int(example):
                val = example[time_column]
                if val in time_column_conversion_dict:
                    example[time_column] = time_column_conversion_dict[val]
                return example

            ds = ds.map(map_to_int)

        current_type = ds.features[time_column].dtype

        if "int" not in current_type:
            warnings.warn(
                f"Dataset {data_file_label}. Column '{time_column}' is of type '{current_type}' "
                f"and is being forcibly cast to 'int64'. Data loss (truncation) may occur.",
                UserWarning,
                stacklevel=2,
            )
            ds = ds.cast_column(time_column, datasets.Value("int64"))
            print(f"Dataset column '{time_column}' successfully cast to int64.")

        return ds

    if len(data_files) > 1:
        for i, f in enumerate(tqdm(data_files, desc="Loading and/or filtering")):
            ds = load_and_filter(
                filter_data=filter_data,
                filter_out_times=filter_out_times,
                nproc=nproc,
                input_data_file=f,
                index=i,
            )

            if time_column is not None:
                ds = _ensure_int_time_column(ds, time_column, time_column_conversion_dict, f)

            dataset_list.append(ds)
            valid_data_files.append(f)

        if len(dataset_list) == 0:
            raise ValueError("No valid datasets found after filtering. Please check your data and filters.")
        if not group_by_dataset:
            logger.info(
                f"Multiple .dataset files found. Concatenating {len(valid_data_files)} valid .dataset files "
                f"into a single pool."
            )
            dataset_list = [smart_concatenate(dataset_list)]

    else:
        valid_data_files = [data_files[0]]
        ds = load_and_filter(
            filter_data=filter_data,
            filter_out_times=filter_out_times,
            nproc=nproc,
            input_data_file=data_files[0],
            index=0,
        )

        if time_column is not None:
            ds = _ensure_int_time_column(ds, time_column, time_column_conversion_dict, data_files[0])

        if len(ds) == 0:
            raise ValueError(
                f"Dataset {data_files} has no valid examples after filtering. Please check your data and filters."
            )
        else:
            dataset_list = [ds]
    return dataset_list, valid_data_files


def flatten_list(megalist):
    """Flatten a list of lists into a single list."""
    return [item for sublist in megalist for item in sublist]


def get_weighted_probs(item_lens, balance_items):
    """Compute balanced sampling probabilities from item lengths."""
    probs_weighted = [item_len * (1 - balance_items) + (1 / len(item_lens)) * balance_items for item_len in item_lens]
    probs_weighted_sum = sum(probs_weighted)
    probs = [probs_weighted_i / probs_weighted_sum for probs_weighted_i in probs_weighted]
    return probs


def get_dataset_choice_list(num_examples, balance_datasets, dataset_timegroup_timechoice_dict):
    """Generate a list of dataset indices for sampling, with optional balancing."""
    valid_indexes = list(dataset_timegroup_timechoice_dict.keys())
    num_datasets = len(valid_indexes)

    if num_datasets > 1:
        print("Generating dataset order list")
        if balance_datasets == 1:
            if num_examples > num_datasets:
                replace = True
            else:
                replace = False
            order = np.random.choice(valid_indexes, size=num_examples, replace=replace)
        else:
            dataset_lens = []
            for ds_idx in tqdm(valid_indexes, desc="Generating dataset order list"):
                total_ds_len = 0
                for timegroup_name in dataset_timegroup_timechoice_dict[ds_idx]:
                    indices_dict = dataset_timegroup_timechoice_dict[ds_idx][timegroup_name].get("indices")
                    total_ds_len += sum(len(idx_list) for idx_list in indices_dict.values())

                dataset_lens.append(total_ds_len)

            dataset_lens_sum = sum(dataset_lens)

            probs = [dataset_len / dataset_lens_sum for dataset_len in dataset_lens]

            if balance_datasets != 0:
                probs = get_weighted_probs(dataset_lens, balance_datasets)

            order = np.random.choice(valid_indexes, size=num_examples, replace=True, p=probs)
    else:
        order = np.full(num_examples, valid_indexes[0], dtype=int)
    return [int(i) for i in order]


def get_timegroup_choice_list(dataset_order, dataset_timegroup_timechoice_dict, balance_timegroups):
    """Generate a list of timegroup selections for each dataset index in the order."""
    if balance_timegroups == 1:
        print("Generating timegroup order list")
        return [np.random.choice(list(dataset_timegroup_timechoice_dict[i].keys())) for i in tqdm(dataset_order)]

    else:
        timegroup_list = []

        for ds_idx in tqdm(dataset_order, desc="Generating timegroup order list"):
            timegroup_lens = []
            timegroup_options = []
            for timegroup_name in dataset_timegroup_timechoice_dict[ds_idx]:
                timegroup_options = timegroup_options + [timegroup_name]
                indices_dict = dataset_timegroup_timechoice_dict[ds_idx][timegroup_name].get("indices")
                timegroup_lens = timegroup_lens + [sum(len(idx_list) for idx_list in indices_dict.values())]

            timegroup_lens_sum = sum(timegroup_lens)

            probs = [timegroup_len / timegroup_lens_sum for timegroup_len in timegroup_lens]

            if balance_timegroups != 0:
                probs = get_weighted_probs(timegroup_lens, balance_timegroups)

            timegroup_order_i = np.random.choice(timegroup_options, p=probs)
            timegroup_list.append(str(timegroup_order_i))

        return timegroup_list


def get_dataset_timechoice_dict_no_groups(dataset_list, time_column):
    """Build per-dataset time->index mapping without time groups.

    Returns ``{dataset_index: {"_all_": {times: [...], indices: {t: [idx, ...]}}}}``
    """
    result = {}
    for ds_idx in tqdm(range(len(dataset_list)), desc="Generating dataset timechoice dictionary (no groups)"):
        ds_idx = int(ds_idx)
        df = dataset_list[ds_idx].to_pandas()
        if not df.empty:
            times = df[time_column].unique().tolist()
            indices_dict = {t: df.index[df[time_column] == t].tolist() for t in times}
            result[ds_idx] = {"_all_": {"times": times, "indices": indices_dict}}
    return result


def get_dataset_timegroup_timechoice_dict(dataset_list, time_column):
    """Build per-dataset, per-timegroup time->index mapping."""
    dataset_timegroup_timechoice_dict = {}

    for unfiltered_dataset_indices in tqdm(range(len(dataset_list)), desc="Generating dataset timegroup dictionary"):
        unfiltered_dataset_indices = int(unfiltered_dataset_indices)
        df = dataset_list[unfiltered_dataset_indices].to_pandas()
        if not df.empty:
            df["master_time_group"] = [str(i) for i in df["master_time_group"]]

            timegroup_dict = {}
            for tg, subdf in df.groupby("master_time_group"):
                times = subdf[time_column].unique().tolist()
                indices_dict = {t: subdf.index[subdf[time_column] == t].tolist() for t in times}
                timegroup_dict[tg] = {"times": times, "indices": indices_dict}

            dataset_timegroup_timechoice_dict[unfiltered_dataset_indices] = timegroup_dict
    return dataset_timegroup_timechoice_dict


def filter_dataset_timegroup_timechoice_dict(
    dataset_timegroup_timechoice_dict,
    min_times=0,
    necessary_timepoints=None,
    necessary_timegroups=None,
    data_files=None,
    exclude_times=None,
):
    """Filter timegroup dict by minimum times, required timepoints/groups, and excluded times."""
    for ds_idx in tqdm(list(dataset_timegroup_timechoice_dict.keys()), desc="Filtering dataset timegroup dictionary"):
        removed_groups = []

        for tg_name in list(dataset_timegroup_timechoice_dict[ds_idx].keys()):
            has_necessary = True

            if necessary_timegroups is not None:
                if tg_name not in necessary_timegroups:
                    has_necessary = False

            if necessary_timepoints is not None:
                has_necessary = has_necessary and any(
                    t in dataset_timegroup_timechoice_dict[ds_idx][tg_name]["times"] for t in necessary_timepoints
                )

            if exclude_times is not None:
                dataset_timegroup_timechoice_dict[ds_idx][tg_name]["times"] = [
                    i for i in dataset_timegroup_timechoice_dict[ds_idx][tg_name]["times"] if i not in exclude_times
                ]
                dataset_timegroup_timechoice_dict[ds_idx][tg_name]["indices"] = {
                    t: v
                    for t, v in dataset_timegroup_timechoice_dict[ds_idx][tg_name]["indices"].items()
                    if t not in exclude_times
                }

            is_too_small = len(dataset_timegroup_timechoice_dict[ds_idx][tg_name]["times"]) < min_times

            if is_too_small or not has_necessary:
                removed_groups.append(tg_name)
                del dataset_timegroup_timechoice_dict[ds_idx][tg_name]

        if removed_groups:
            if data_files is not None:
                print(f"Dataset {ds_idx} ({data_files[ds_idx]}): Removed groups {removed_groups} ")
            else:
                print(f"Dataset {ds_idx}: Removed groups {removed_groups} ")

        if not dataset_timegroup_timechoice_dict[ds_idx]:
            if data_files is not None:
                print(f"Removing empty dataset index {ds_idx}: {data_files[ds_idx]}")
            else:
                print(f"Removing empty dataset index {ds_idx}")
            del dataset_timegroup_timechoice_dict[ds_idx]
    return dataset_timegroup_timechoice_dict


def random_choice_with_limit(
    input_list,
    number_of_choices,
    max_repetitions,
    max_repeats_per_chosen_ele,
    allow_zero_timelapses,
    allow_negative_timelapses,
    match_time_tolerance=None,
    alt_times=None,
):
    """Select time choices from input_list with constraints on repetition and ordering."""
    if match_time_tolerance is not None:
        input_list = sorted(input_list)
        question_input_list = [i for i in input_list if any(abs(i - t) <= match_time_tolerance for t in alt_times)]

        if (max_repetitions == 0) and (max_repeats_per_chosen_ele == 0):
            if allow_negative_timelapses:
                question = np.random.choice(question_input_list, 1)
                context_input_list = [i for i in input_list if i != question[0]]
                context = np.random.choice(context_input_list, number_of_choices - 1, replace=False)
            else:
                question = np.random.choice(question_input_list[number_of_choices - 1 :], 1)
                context_input_list = [i for i in input_list if i < question[0]]
                context = np.random.choice(context_input_list, number_of_choices - 1, replace=False)
                context.sort()
            return np.concatenate([context, question])

        if allow_zero_timelapses == "question":
            if allow_negative_timelapses:
                question = np.random.choice(question_input_list, 1)
                context = np.random.choice(input_list, number_of_choices - 1, replace=False)
            else:
                question = np.random.choice(question_input_list[number_of_choices - 1 :], 1)
                context_input_list = [i for i in input_list if i <= question[0]]
                context = np.random.choice(context_input_list, number_of_choices - 1, replace=False)
                context.sort()
            return np.concatenate([context, question])

        if allow_zero_timelapses == "context_and_question":
            if (max_repetitions is None) and (max_repeats_per_chosen_ele is None):
                question = np.random.choice(question_input_list, 1)
                context = np.random.choice(input_list, number_of_choices - 1, replace=True)
            else:
                max_repeats_per_chosen_ele = (
                    min(max_repeats_per_chosen_ele, max_repetitions) if max_repetitions else max_repeats_per_chosen_ele
                )
                question = np.random.choice(question_input_list, 1)
                available_context_pool = flatten_list(
                    [[element] * max_repeats_per_chosen_ele for element in input_list]
                )
                for idx, val in enumerate(available_context_pool):
                    if val == question[0]:
                        available_context_pool.pop(idx)
                        break
                context = np.random.choice(available_context_pool, number_of_choices - 1, replace=False)

            if allow_negative_timelapses:
                return np.concatenate([context, question])
            else:
                context = np.array([c for c in context if c <= question[0]])
                results = np.concatenate([context, question])
                results.sort()
                return results

    if (max_repetitions == 0) and (max_repeats_per_chosen_ele == 0):
        time_choices = np.random.choice(input_list, min(number_of_choices, len(input_list)), replace=False)
        if allow_negative_timelapses:
            return time_choices
        else:
            time_choices.sort()
            return time_choices

    if allow_zero_timelapses == "question":
        context = np.random.choice(input_list, min(number_of_choices - 1, len(input_list)), replace=False)
        if allow_negative_timelapses:
            question = np.random.choice(input_list, 1)
        else:
            context.sort()
            available_choices = [x for x in input_list if x >= context[-1]]
            question = np.random.choice(available_choices, 1)
        return np.concatenate((context, question))

    if allow_zero_timelapses == "context_and_question":
        if (max_repetitions is None) and (max_repeats_per_chosen_ele is None):
            time_choices = np.random.choice(input_list, number_of_choices, replace=True)
        else:
            max_possible_choices = len(input_list) + len(input_list) * max_repetitions
            number_of_choices = min(number_of_choices, max_possible_choices)
            max_repeats_per_chosen_ele = min(max_repeats_per_chosen_ele, max_repetitions)
            available_choices = flatten_list([[element] * max_repeats_per_chosen_ele for element in input_list])
            time_choices = np.random.choice(available_choices, number_of_choices, replace=False)
        if allow_negative_timelapses:
            return time_choices
        else:
            time_choices.sort()
            return time_choices


def get_timestep(input_list):
    """Compute successive differences between adjacent elements."""
    return [j - i for i, j in itertools.pairwise(input_list)]


def truncate_before_first_eos(prompt, total_length, model_input_size, eos_token):
    """Truncate tokens before the first EOS to fit within model_input_size."""
    if eos_token not in prompt:
        raise ValueError("No <eos> token in prompt.")

    first_eos_index = prompt.index(eos_token)
    max_removable = max(0, first_eos_index - 3)
    tokens_to_remove = total_length - model_input_size

    if tokens_to_remove > max_removable:
        raise ValueError(
            f"Cannot truncate enough tokens before first <eos> without removing protected tokens. "
            f"Tokens to remove: {tokens_to_remove}, max removable: {max_removable}"
        )

    start = first_eos_index - tokens_to_remove
    prompt = prompt[:start] + prompt[first_eos_index:]
    return prompt


def compose_example(
    dataset_i,
    cell_indices_i,
    timesteps_i,
    time_choices_i,
    token_dictionary,
    predict_time_or_cell,
    is_train,
    model_input_size,
    clip_first_cell,
):
    """Compose a context/question/response example from cell indices and timesteps."""
    cell_list = [dataset_i[j]["input_ids"] for j in cell_indices_i]
    unfiltered_cell_indices = [dataset_i[j]["unfiltered_cell_indices"] for j in cell_indices_i]
    unfiltered_dataset_indices = [dataset_i[j]["unfiltered_dataset_indices"] for j in cell_indices_i]
    timestep_list = [[token_dictionary[t]] for t in timesteps_i]
    last_time_choice = time_choices_i[-1]

    context = []
    for cell, timestep in zip(cell_list[:-1], timestep_list):
        context.extend(cell)
        context.extend(timestep)
    context = context[:-1]  # remove last timestep (will be added in question)

    if predict_time_or_cell == "cell":
        question = compose_question(timestep_list[-1], token_dictionary)
        response = cell_list[-1]
    elif predict_time_or_cell == "time":
        question = compose_question(cell_list[-1], token_dictionary)
        response = timestep_list[-1]

    prompt = context + question

    total_length = len(prompt) + len(response)

    if total_length > model_input_size:
        if not clip_first_cell:
            raise ValueError("Prompt and response length exceeds model input size.")
        prompt = truncate_before_first_eos(prompt, total_length, model_input_size, token_dictionary["<eos>"])

    if is_train is False:
        response_len = len(response)
        max_response_len = model_input_size - len(prompt)
        return (
            prompt,
            context,
            question,
            response,
            response_len,
            max_response_len,
            last_time_choice,
            unfiltered_cell_indices,
            unfiltered_dataset_indices,
        )
    else:
        combined = prompt + response
        response_len = len(response)
        return combined, response_len


def compose_question(question_content, token_dictionary):
    """Wrap question content with <boq> and <eoq> tokens."""
    return [token_dictionary["<boq>"]] + question_content + [token_dictionary["<eoq>"]]


def filter_query_datasets(
    query_dataset_list: list[Dataset],
    query_subset_dict: dict[str, list[Any]],
    nproc: int = 1,
) -> list[Dataset]:
    """Filter a list of HuggingFace datasets by allowed column values."""

    def filter_row_by_dict(example: dict[str, Any]) -> bool:
        for column, allowed_values in query_subset_dict.items():
            if column in example:
                if example[column] not in allowed_values:
                    return False
        return True

    filtered_dataset_list = []

    for dataset in query_dataset_list:
        filtered_dataset = dataset.filter(filter_row_by_dict, num_proc=nproc)
        filtered_dataset_list.append(filtered_dataset)

    return filtered_dataset_list


def convert_token_dictionary_keys(token_dictionary: dict) -> dict:
    """Convert string numeric keys in a token dictionary to integers.

    Token dictionaries loaded from pickle may have numeric keys stored as strings
    (e.g., ``"3"`` or ``"3.0"``). This function converts those to ``int`` keys while
    leaving gene Ensembl IDs and special tokens (``<bos>``, ``<eos>``, etc.) unchanged.

    Args:
        token_dictionary: The raw token dictionary.

    Returns:
        A new dictionary with numeric string keys converted to integers.
    """
    converted = {}
    conversion_count = 0

    for key, value in token_dictionary.items():
        if isinstance(key, str):
            try:
                float_key = float(key)
                if float_key == math.floor(float_key):
                    converted[int(float_key)] = value
                    conversion_count += 1
                    continue
                else:
                    warnings.warn(
                        f"String key '{key}' had a non-zero decimal part and was skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    converted[key] = value
                    continue
            except ValueError:
                converted[key] = value
                continue

        converted[key] = value

    if conversion_count > 0:
        print(f"\nSuccessfully converted {conversion_count} numeric string keys to integers.")
    else:
        print("\nNo numeric string keys found for conversion.")

    return converted
