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

"""MaxToki timeseries query assembler.

Builds evaluation query datasets by replacing query cells in assembled cell paragraphs
with cells from experimental/alternative data sources.

**Usage:**

.. code-block :: python

    >>> from bionemo.maxtoki.data_prep import QueryAssembler
    >>> qa = QueryAssembler(match_time_tolerance=5, nproc=4)
    >>> qa.build_eval_cell_query_dataset(
    ...     "blueprint_dictionary_file",
    ...     "time_token_dictionary_file",
    ...     "cell_paragraph_dataset_file",
    ...     query_alt_directory="path/to/alt_data",
    ...     output_directory="output_directory",
    ...     output_prefix="output_prefix",
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from tqdm import tqdm

from . import dataset_utils as du


logger = logging.getLogger(__name__)


class QueryAssembler:
    """Builds evaluation query datasets from cell paragraphs and alternative data."""

    def __init__(
        self,
        query_subset_dict=None,
        match_time_tolerance=None,
        time_column="time",
        time_group_columns=None,
        allow_zero_timelapses="question",
        allow_negative_timelapses=True,
        exclude_times=None,
        filter_data=None,
        nproc=1,
        model_input_size=16_384,
        max_timelapse=None,
        model_version="V1",
        necessary_timepoints=None,
        time_column_conversion_dict=None,
        seed=42,
    ):
        """Initialize query assembler.

        Args:
            query_subset_dict: Dictionary of eval groups to include in the query.
            match_time_tolerance: Maximum time difference for matching control/experimental cells.
            time_column: Column name for cell time along trajectory. Must be integers.
            time_group_columns: Column names for trajectory grouping.
            allow_zero_timelapses: One of "none", "question", or "context_and_question".
            allow_negative_timelapses: Whether to allow negative timelapses.
            exclude_times: List of times to exclude.
            filter_data: Dictionary of filter criteria.
            nproc: Number of processes for dataset mapping.
            model_input_size: Max model input size.
            max_timelapse: Maximum absolute timelapse allowed.
            model_version: Model version (currently "V1").
            necessary_timepoints: Required timepoints for dataset inclusion.
            time_column_conversion_dict: Mapping to convert time column values to integers.
            seed: Random seed.
        """
        self.necessary_timepoints = necessary_timepoints
        self.time_column_conversion_dict = time_column_conversion_dict
        self.query_subset_dict = query_subset_dict
        self.match_time_tolerance = match_time_tolerance
        self.time_column = time_column
        self.time_group_columns = time_group_columns
        self.allow_negative_timelapses = allow_negative_timelapses
        valid_zero_timelapse_options = {"none", "question", "context_and_question"}
        if allow_zero_timelapses not in valid_zero_timelapse_options:
            raise ValueError(
                f"allow_zero_timelapses must be one of {valid_zero_timelapse_options}, got '{allow_zero_timelapses}'"
            )
        self.allow_zero_timelapses = allow_zero_timelapses
        self.exclude_times = exclude_times
        self.filter_data = filter_data
        self.max_timelapse = max_timelapse
        self.nproc = nproc
        self.model_input_size = model_input_size
        self.model_version = model_version
        self.seed = seed

    def build_eval_cell_query_dataset(
        self,
        blueprint_dictionary_file,
        time_token_dictionary_file,
        cell_paragraph_dataset_file,
        query_data_files=None,
        query_alt_directory=None,
        output_directory=None,
        output_prefix=None,
    ):
        """Build query dataset for specific eval cells as query.

        Args:
            blueprint_dictionary_file: Path to cell paragraph blueprint JSON file.
            time_token_dictionary_file: Path to time token dictionary JSON file.
            cell_paragraph_dataset_file: Path to .dataset with cell paragraphs.
            query_data_files: List of alternative dataset paths.
            query_alt_directory: Path to directory with alternative query datasets.
            output_directory: Path for saving output .dataset.
            output_prefix: Prefix for output .dataset.

        Returns:
            The query paragraphs dataset.
        """
        with open(blueprint_dictionary_file, "r") as fp:
            blueprint_dictionary = json.load(fp)

        alt_query_dataset_timegroup_timechoice_dict = None
        alt_query_dataset_list = None

        if output_prefix is None:
            output_prefix = blueprint_dictionary.get("output_prefix_complete", "query_output")

        if query_data_files is None:
            query_data_files = blueprint_dictionary["data_files"]

        if query_alt_directory is not None:
            if isinstance(query_alt_directory, str):
                if query_alt_directory.lower().endswith(".dataset"):
                    query_alt_directory = [query_alt_directory]
                elif os.path.isdir(query_alt_directory):
                    path_obj = Path(query_alt_directory)
                    query_alt_directory = [
                        str(p) for p in path_obj.iterdir() if p.is_dir() and p.name.endswith(".dataset")
                    ]

        def get_dataset_list(query_data_files):
            query_dataset_list, _ = du.load_datasets(
                data_files=query_data_files,
                filter_data=self.filter_data,
                time_column=self.time_column,
                nproc=self.nproc,
                group_by_dataset=blueprint_dictionary["args"]["group_by_dataset"],
                time_column_conversion_dict=self.time_column_conversion_dict,
            )

            if self.query_subset_dict is not None:
                query_dataset_list = du.filter_query_datasets(
                    query_dataset_list,
                    self.query_subset_dict,
                    self.nproc,
                )

            for data in query_dataset_list:
                if self.time_column not in data.column_names:
                    raise ValueError(f"'{self.time_column}' column missing from dataset")
                if self.time_group_columns is not None:
                    if "master_time_group" in data.column_names:
                        raise ValueError(
                            "'master_time_group' column already exists. "
                            "Please rename or remove it before assembling cell paragraphs."
                        )
                    if not isinstance(self.time_group_columns, list):
                        raise TypeError("time_group_columns must be a list of column names.")
                    missing_cols = [col for col in self.time_group_columns if col not in data.column_names]
                    if missing_cols:
                        raise ValueError(f"Dataset is missing the following timegroup columns: {missing_cols}")

            if self.time_group_columns is not None:
                logger.info(
                    f"Creating master time group column for grouping trajectories by {self.time_group_columns}."
                )

                def create_master_time_group_column(example):
                    time_group = "_".join([str(example[col]) for col in self.time_group_columns])
                    example["master_time_group"] = time_group
                    return example

                query_dataset_list = [
                    dataset.map(create_master_time_group_column, num_proc=self.nproc)
                    for dataset in tqdm(query_dataset_list, desc="Generating query master_time_group_column")
                ]

            return query_dataset_list

        query_dataset_list = get_dataset_list(query_data_files)

        if self.time_group_columns is None:
            query_dataset_timegroup_timechoice_dict = du.get_dataset_timechoice_dict_no_groups(
                query_dataset_list, self.time_column
            )
        else:
            query_dataset_timegroup_timechoice_dict = du.get_dataset_timegroup_timechoice_dict(
                query_dataset_list, self.time_column
            )

        if self.exclude_times is not None:
            query_dataset_timegroup_timechoice_dict = du.filter_dataset_timegroup_timechoice_dict(
                query_dataset_timegroup_timechoice_dict, exclude_times=self.exclude_times
            )

        if query_alt_directory is not None:
            alt_query_dataset_list = get_dataset_list(query_alt_directory)
            if self.time_group_columns is None:
                alt_query_dataset_timegroup_timechoice_dict = du.get_dataset_timechoice_dict_no_groups(
                    alt_query_dataset_list, self.time_column
                )
            else:
                alt_query_dataset_timegroup_timechoice_dict = du.get_dataset_timegroup_timechoice_dict(
                    alt_query_dataset_list, self.time_column
                )
            alt_query_dataset_timegroup_timechoice_dict = du.filter_dataset_timegroup_timechoice_dict(
                alt_query_dataset_timegroup_timechoice_dict, exclude_times=self.exclude_times
            )
        else:
            alt_query_dataset_timegroup_timechoice_dict = None

        cell_paragraphs_dataset = du.load_and_filter(input_data_file=cell_paragraph_dataset_file, nproc=self.nproc)

        with open(time_token_dictionary_file, "r") as fp:
            token_dictionary = json.load(fp)

        token_dictionary = du.convert_token_dictionary_keys(token_dictionary)

        random.seed(self.seed)

        blueprint_no_time_group = any(
            t == "__NO_TIME_GROUP__" for t in blueprint_dictionary.get("timegroup_order", [])
        )
        if blueprint_no_time_group and self.time_group_columns is not None:
            raise ValueError(
                "Blueprint was built without time groups (timegroup_order is __NO_TIME_GROUP__). "
                "Set time_group_columns=None in QueryAssembler to build queries for this blueprint."
            )
        use_no_time_group = self.time_group_columns is None or blueprint_no_time_group

        def generate_query_data(sample, idx):
            dataset_i = blueprint_dictionary["dataset_order"][idx]
            if use_no_time_group:
                timegroup_i = "_all_"
            else:
                timegroup_i = str(blueprint_dictionary["timegroup_order"][idx])

            if alt_query_dataset_timegroup_timechoice_dict is not None:
                matching_datasets = [
                    ds_idx
                    for ds_idx, content in alt_query_dataset_timegroup_timechoice_dict.items()
                    if timegroup_i in content
                ]
                if matching_datasets:
                    dataset_i = random.choice(matching_datasets)
                else:
                    raise ValueError(
                        f"No matching timegroups found. Timegroup '{timegroup_i}' from the blueprint "
                        f"must exist in the query dataset. Ensure cell paragraphs were built with the "
                        f"same time_group_columns as this query assembler ({self.time_group_columns})."
                    )
                query_dataset_timegroup_timechoice_dict_i = alt_query_dataset_timegroup_timechoice_dict
            else:
                query_dataset_timegroup_timechoice_dict_i = query_dataset_timegroup_timechoice_dict

            if not use_no_time_group:
                if (
                    dataset_i not in query_dataset_timegroup_timechoice_dict_i
                    or timegroup_i not in query_dataset_timegroup_timechoice_dict_i[dataset_i]
                ):
                    raise ValueError(
                        f"No matching timegroups found. Timegroup '{timegroup_i}' from the blueprint "
                        f"must exist in the query dataset. Ensure cell paragraphs were built with the "
                        f"same time_group_columns as this query assembler ({self.time_group_columns})."
                    )

            timechoices_available = query_dataset_timegroup_timechoice_dict_i[dataset_i][timegroup_i]["times"]

            current_time_choice = sample["time_choices"][-2]
            original_time_choice = sample["time_choices"][-1]

            if not self.allow_negative_timelapses:
                timechoices_available = [i for i in timechoices_available if i >= current_time_choice]

            if self.max_timelapse is not None:
                timechoices_available = [
                    i for i in timechoices_available if abs(i - current_time_choice) <= self.max_timelapse
                ]

            if self.allow_zero_timelapses == "none":
                timechoices_available = [i for i in timechoices_available if i != current_time_choice]

            if self.match_time_tolerance is not None:
                timechoices_available = [
                    i for i in timechoices_available if abs(i - original_time_choice) <= self.match_time_tolerance
                ]

            if not timechoices_available:
                raise RuntimeError(f"No valid times for dataset order {dataset_i} and timegroup order {timegroup_i}.")

            timechoice_selected = random.choice(timechoices_available)
            new_timestep = int(timechoice_selected - sample["time_choices"][-1] + sample["timesteps"][-1])

            if new_timestep not in token_dictionary:
                raise KeyError(
                    f"Computed timestep token {new_timestep} not found in token_dictionary. "
                    f"timechoice_selected={timechoice_selected}, "
                    f"original_time_choice={sample['time_choices'][-1]}, "
                    f"original_timestep={sample['timesteps'][-1]}."
                )

            indices_available = query_dataset_timegroup_timechoice_dict_i[dataset_i][timegroup_i]["indices"][
                timechoice_selected
            ]

            index_selected = random.choice(indices_available)
            if alt_query_dataset_list is not None:
                selected_query_data = alt_query_dataset_list[dataset_i][index_selected]
            else:
                selected_query_data = query_dataset_list[dataset_i][index_selected]
            new_cell = selected_query_data["input_ids"]
            len_new_cell = len(new_cell)
            max_len_new_cell = self.model_input_size - len(sample["context"]) - 3  # 2 for boq/eoq + 1 for timestep
            if len(new_cell) > max_len_new_cell and max_len_new_cell >= 3:
                new_cell = new_cell[: max_len_new_cell - 1] + [new_cell[-1]]
                print(
                    f"WARNING: Substitute cell is too long, shortening it.\n"
                    f"Original length {len_new_cell}, new length {len(new_cell)}, "
                    f"max length {max_len_new_cell}"
                )
                len_new_cell = len(new_cell)
            elif max_len_new_cell < 3:
                raise ValueError("Maximum allowed length is too short for adding any new gene token ids")

            new_unfiltered_dataset_index = selected_query_data["unfiltered_dataset_indices"]
            new_unfiltered_cell_index = selected_query_data["unfiltered_cell_indices"]

            output = {}

            bos_token = token_dictionary["<bos>"]
            boq_token = token_dictionary["<boq>"]
            eoq_token = token_dictionary["<eoq>"]

            output["context"] = sample["context"]

            if sample["response"][0] == bos_token:
                output["response"] = new_cell
                output["old_response"] = sample["response"]
                output["question"] = [boq_token] + [token_dictionary[new_timestep]] + [eoq_token]
                output["old_question"] = sample["question"]
            else:
                output["question"] = [boq_token] + new_cell + [eoq_token]
                output["old_question"] = sample["question"]
                output["response"] = [token_dictionary[new_timestep]]
                output["old_response"] = sample["response"]

            output["input_ids"] = output["context"] + output["question"]
            output["old_input_ids"] = output["context"] + output["old_question"]

            output["old_unfiltered_dataset_indices"] = sample["unfiltered_dataset_indices"]
            output["new_unfiltered_dataset_indices"] = sample["unfiltered_dataset_indices"][:-1] + [
                new_unfiltered_dataset_index
            ]
            output["dataset_order"] = dataset_i
            output["dataset_order_old"] = sample["dataset_order"]
            output["timegroup_order"] = timegroup_i
            output["timegroup_order_old"] = sample["timegroup_order"]
            output["unfiltered_cell_indices_old"] = sample["unfiltered_cell_indices"]
            output["unfiltered_cell_indices"] = sample["unfiltered_cell_indices"][:-1] + [new_unfiltered_cell_index]
            output["time_choices_old"] = sample["time_choices"]
            output["time_choices"] = sample["time_choices"][:-1] + [timechoice_selected]
            output["timesteps_old"] = sample["timesteps"]
            output["timesteps"] = sample["timesteps"][:-1] + [new_timestep]

            return output

        query_paragraphs_dataset = cell_paragraphs_dataset.map(
            generate_query_data,
            with_indices=True,
            desc="Generating Query Paragraphs",
            num_proc=1,
        )

        save_path = os.path.join(output_directory, f"{output_prefix}.dataset")
        query_paragraphs_dataset.save_to_disk(str(save_path))

        print(f"Dataset saved successfully to: {save_path}")

        return query_paragraphs_dataset
