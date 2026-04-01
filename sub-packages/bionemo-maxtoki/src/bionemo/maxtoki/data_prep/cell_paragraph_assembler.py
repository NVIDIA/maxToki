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

"""MaxToki timeseries cell paragraph assembler.

**Input data:**

| Input data file:
| *Required format:* Hugging Face .dataset file with the following columns:
| *Required columns:* "time" (must be integers only unless providing time_column_conversion_dict)
| *Optional columns:* "time_group"

| Time token dictionary file:
| *Required format:* Pickle file containing a dictionary of gene and time tokens.
| *Required special tokens:* <bos>, <eos>, <boq>, <eoq>

**Usage:**

.. code-block :: python

    >>> from bionemo.maxtoki.data_prep import CellParagraphAssembler
    >>> cpa = CellParagraphAssembler(time_group_columns=["cell_type", "sex"], nproc=4)
    >>> dataset_list, blueprint_dictionary = cpa.generate_cell_paragraph_blueprint(
    ...     "data_directory", "output_directory", "output_prefix")
    >>> output_prefix_complete = blueprint_dictionary["output_prefix_complete"]
    >>> blueprint_file = f"{output_directory}/{output_prefix_complete}_cell_paragraph_blueprint.json"
    >>> time_token_dictionary = cpa.generate_time_dictionary(output_directory, output_prefix, max_timepoint)
    >>> time_dict_file = f"{output_directory}/{output_prefix_complete}_time_dictionary.json"
    >>> cpa.assemble_cell_paragraphs(
    ...     dataset_list, blueprint_file, time_dict_file, "output_directory", "output_prefix", is_train=True)
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from . import dataset_utils as du


logger = logging.getLogger(__name__)


class CellParagraphAssembler:
    """Assembles tokenized single-cell datasets into training cell paragraphs."""

    def __init__(
        self,
        num_examples=10_000_000,
        time_column="time",
        time_group_columns=None,
        group_by_dataset=False,
        balance_datasets=0.1,
        balance_timegroups=0.1,
        task_ratio=0.5,
        min_timepoints=3,
        max_timepoints=4,
        max_repeat_timepoints=1,
        max_repeats_specific_timepoint=1,
        allow_zero_timelapses="question",
        allow_negative_timelapses=True,
        exclude_times=None,
        filter_data=None,
        nproc=1,
        model_input_size=16_384,
        model_version="V1",
        necessary_timepoints=None,
        necessary_timegroups=None,
        time_column_conversion_dict=None,
        seed=42,
    ):
        """Initialize cell paragraph assembler.

        Args:
            num_examples: Number of cell paragraphs to assemble.
            time_column: Column name in dataset for cell timepoint along trajectory.
            time_group_columns: Column names for trajectory grouping. If None, pools all cells.
            group_by_dataset: Whether to group cell paragraphs by dataset.
            balance_datasets: Factor for balancing examples across datasets.
            balance_timegroups: Factor for balancing examples across time groups.
            task_ratio: Ratio of task 1 (timelapse) to task 2 (cell state) paragraphs.
            min_timepoints: Minimum timepoints per cell paragraph.
            max_timepoints: Maximum timepoints per cell paragraph.
            max_repeat_timepoints: Maximum times any timepoint may repeat.
            max_repeats_specific_timepoint: Maximum repeats of a specific timepoint.
            allow_zero_timelapses: One of "none", "question", or "context_and_question".
            allow_negative_timelapses: Whether to allow negative timelapses.
            exclude_times: List of times to exclude.
            filter_data: Dictionary of filter criteria {column: values_to_keep}.
            nproc: Number of processes for dataset mapping.
            model_input_size: Max model input size.
            model_version: Model version (currently "V1").
            necessary_timepoints: Required timepoints for dataset inclusion.
            necessary_timegroups: Required timegroups for dataset inclusion.
            time_column_conversion_dict: Mapping to convert time column values to integers.
            seed: Random seed.
        """
        self.num_examples = num_examples
        self.time_column = time_column
        self.time_group_columns = time_group_columns
        self.group_by_dataset = group_by_dataset
        self.balance_datasets = balance_datasets
        self.balance_timegroups = balance_timegroups
        self.task_ratio = task_ratio
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints
        self.max_repeat_timepoints = max_repeat_timepoints
        self.max_repeats_specific_timepoint = max_repeats_specific_timepoint
        self.necessary_timepoints = necessary_timepoints
        self.necessary_timegroups = necessary_timegroups
        self.time_column_conversion_dict = time_column_conversion_dict

        if (self.max_repeats_specific_timepoint == 0 and self.max_repeat_timepoints != 0) or (
            self.max_repeat_timepoints == 0 and self.max_repeats_specific_timepoint != 0
        ):
            raise ValueError(
                "If max_repeats_specific_timepoint is 0, then max_repeat_timepoints must be 0, and viceversa"
            )
        valid_zero_timelapse_options = {"none", "question", "context_and_question"}
        if allow_zero_timelapses not in valid_zero_timelapse_options:
            raise ValueError(
                f"allow_zero_timelapses must be one of {valid_zero_timelapse_options}, got '{allow_zero_timelapses}'"
            )
        self.allow_zero_timelapses = allow_zero_timelapses
        if self.allow_zero_timelapses == "question":
            self.max_repeat_timepoints = 1
            self.max_repeats_specific_timepoint = 1
            logger.info(
                "allow_zero_timelapses == 'question'. "
                "Resetting max_repeat_timepoints and max_repeats_specific_timepoint to 1."
            )
        elif self.allow_zero_timelapses == "none":
            self.max_repeat_timepoints = 0
            self.max_repeats_specific_timepoint = 0
            logger.info(
                "allow_zero_timelapses == 'none'. "
                "Resetting max_repeat_timepoints and max_repeats_specific_timepoint to 0."
            )

        self.allow_negative_timelapses = allow_negative_timelapses
        self.exclude_times = exclude_times
        self.filter_data = filter_data
        self.nproc = nproc
        self.model_input_size = model_input_size
        self.model_version = model_version
        self.seed = seed

    def generate_cell_paragraph_blueprint(
        self,
        data_directory,
        output_directory,
        output_prefix,
        query_alt_directory=None,
        match_time_tolerance=None,
    ):
        """Generate cell paragraph blueprint.

        Args:
            data_directory: Path to .dataset file or directory with multiple .dataset files.
            output_directory: Path to directory for saving blueprint JSON file.
            output_prefix: Prefix for output files.
            query_alt_directory: Alternative data path for control vs. experimental queries.
            match_time_tolerance: Maximum time difference for matching control/experimental cells.

        Returns:
            Tuple of (dataset_list, blueprint_dictionary).
        """
        np.random.seed(self.seed)
        alt_time_per_timegroups = None

        if match_time_tolerance is not None and query_alt_directory is None:
            raise ValueError("match_time_tolerance requires query_alt_directory to be set.")

        if output_directory[-1] != "/":
            output_directory = output_directory + "/"

        output_prefix_complete = (
            f"{output_prefix}_n{self.num_examples}_minT{self.min_timepoints}_maxT{self.max_timepoints}"
            f"_maxRP{self.max_repeats_specific_timepoint}_zeroT{str(self.allow_zero_timelapses)}"
            f"_rev{str(self.allow_negative_timelapses)}_taskR{self.task_ratio}_seed{self.seed}"
        )

        if data_directory.endswith(".dataset"):
            data_files = [data_directory]
        else:
            data_files = [
                os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".dataset")
            ]

        dataset_list, data_files = du.load_datasets(
            data_files=data_files,
            filter_data=self.filter_data,
            time_column=self.time_column,
            nproc=self.nproc,
            group_by_dataset=self.group_by_dataset,
            time_column_conversion_dict=self.time_column_conversion_dict,
        )

        for data in dataset_list:
            if self.time_column not in data.column_names:
                raise ValueError(f"'{self.time_column}' column missing from dataset")
            if self.time_group_columns is not None and "master_time_group" in data.column_names:
                raise ValueError(
                    "'master_time_group' column already exists. "
                    "Please rename or remove it before assembling cell paragraphs."
                )
            if self.time_group_columns is not None:
                if not isinstance(self.time_group_columns, list):
                    raise TypeError("time_group_columns must be a list of column names.")
                missing_cols = [col for col in self.time_group_columns if col not in data.column_names]
                if missing_cols:
                    raise ValueError(f"Dataset is missing the following timegroup columns: {missing_cols}")

        if self.time_group_columns is not None:
            logger.info(f"Creating master time group column for grouping trajectories by {self.time_group_columns}.")

            def create_master_time_group_column(example):
                time_group = "_".join([str(example[col]) for col in self.time_group_columns])
                example["master_time_group"] = time_group
                return example

            dataset_list = [
                dataset.map(create_master_time_group_column, num_proc=self.nproc)
                for dataset in tqdm(dataset_list, desc="Generating master_time_group_column")
            ]

        if query_alt_directory is not None and self.time_group_columns is None:
            logger.info("query_alt_directory and match_time_tolerance are ignored when time_group_columns is None.")
        if query_alt_directory is not None and self.time_group_columns is not None:
            if query_alt_directory.endswith(".dataset"):
                alt_data_files = [query_alt_directory]
            else:
                alt_data_files = [
                    os.path.join(query_alt_directory, f)
                    for f in os.listdir(query_alt_directory)
                    if f.endswith(".dataset")
                ]

            alt_dataset_list, _ = du.load_datasets(
                data_files=alt_data_files,
                filter_data=self.filter_data,
                time_column=self.time_column,
                nproc=self.nproc,
                group_by_dataset=self.group_by_dataset,
                time_column_conversion_dict=self.time_column_conversion_dict,
            )

            for data in alt_dataset_list:
                if self.time_column not in data.column_names:
                    raise ValueError(f"'{self.time_column}' column missing from dataset")
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

            logger.info(f"Creating master time group column for grouping trajectories by {self.time_group_columns}.")

            def create_master_time_group_column(example):
                time_group = "_".join([str(example[col]) for col in self.time_group_columns])
                example["master_time_group"] = time_group
                return example

            alt_dataset_list = [
                dataset.map(create_master_time_group_column, num_proc=self.nproc)
                for dataset in tqdm(alt_dataset_list, desc="Generating alt master_time_group_column")
            ]

            nested_uniques = [dataset.unique("master_time_group") for dataset in alt_dataset_list]
            new_unique_groups = list(set().union(*nested_uniques))

            if self.necessary_timegroups is not None:
                self.necessary_timegroups = list(set(self.necessary_timegroups).union(new_unique_groups))
            else:
                self.necessary_timegroups = new_unique_groups

            if match_time_tolerance is not None:
                alt_time_per_timegroups_sets = defaultdict(set)

                for dataset_i in alt_dataset_list:
                    groups_array = np.array(dataset_i["master_time_group"])
                    times_array = np.array(dataset_i[self.time_column])

                    if self.exclude_times is not None:
                        valid_indices = ~np.isin(times_array, self.exclude_times)
                        times_array = times_array[valid_indices]
                        groups_array = groups_array[valid_indices]

                    unique_groups_in_ds = np.unique(groups_array)

                    groups_to_process = [g for g in unique_groups_in_ds if g in self.necessary_timegroups]

                    for tg_name in groups_to_process:
                        mask = groups_array == tg_name
                        unique_ages_here = np.unique(times_array[mask])
                        alt_time_per_timegroups_sets[tg_name].update(unique_ages_here.tolist())

                alt_time_per_timegroups = {tg: sorted(ages) for tg, ages in alt_time_per_timegroups_sets.items()}

        logger.info("Generating dataset_timegroup_timechoice_dict")

        if self.time_group_columns is not None:
            dataset_timegroup_timechoice_dict = du.get_dataset_timegroup_timechoice_dict(
                dataset_list, self.time_column
            )
            dataset_time_choices = [list(set(dataset[self.time_column])) for dataset in dataset_list]
            max_timepoint = max(du.flatten_list(dataset_time_choices))
            n_timepoints_options = list(range(self.min_timepoints, self.max_timepoints + 1))
            min_times = self.min_timepoints - self.max_repeat_timepoints

            if match_time_tolerance is not None:
                dataset_keys = list(dataset_timegroup_timechoice_dict.keys())

                for ds_idx in dataset_keys:
                    timegroups_in_ds = list(dataset_timegroup_timechoice_dict[ds_idx].keys())

                    for tg in timegroups_in_ds:
                        alt_times = np.array(alt_time_per_timegroups.get(tg, []))
                        base_times = np.array(dataset_timegroup_timechoice_dict[ds_idx][tg]["times"])

                        diff_matrix = np.abs(alt_times[:, np.newaxis] - base_times)
                        ref_idx, _ = np.where(diff_matrix <= match_time_tolerance)
                        alt_times_within_limits = alt_times[ref_idx]

                        if self.allow_negative_timelapses is False:
                            is_smaller = alt_times_within_limits[:, None] > base_times
                            counts = np.sum(is_smaller, axis=1)
                            alt_times_within_limits = alt_times_within_limits[counts >= min_times]

                            if alt_times_within_limits.size > 0:
                                dataset_timegroup_timechoice_dict[ds_idx][tg]["times"] = [
                                    i
                                    for i in dataset_timegroup_timechoice_dict[ds_idx][tg]["times"]
                                    if i <= max(alt_times_within_limits)
                                ]

                                if len(dataset_timegroup_timechoice_dict[ds_idx][tg]["times"]) >= min_times:
                                    tg_data = dataset_timegroup_timechoice_dict[ds_idx][tg]
                                    dataset_timegroup_timechoice_dict[ds_idx][tg]["indices"] = {
                                        t: tg_data["indices"][t] for t in tg_data["times"] if t in tg_data["indices"]
                                    }
                            else:
                                del dataset_timegroup_timechoice_dict[ds_idx][tg]

                        existing = np.array(alt_time_per_timegroups.get(tg, []))
                        alt_time_per_timegroups[tg] = np.union1d(existing, alt_times_within_limits)

                        if tg in dataset_timegroup_timechoice_dict[ds_idx] and alt_times_within_limits.size == 0:
                            del dataset_timegroup_timechoice_dict[ds_idx][tg]
                            continue

                        if (
                            tg in dataset_timegroup_timechoice_dict[ds_idx]
                            and len(dataset_timegroup_timechoice_dict[ds_idx][tg]["times"]) < min_times
                        ):
                            del dataset_timegroup_timechoice_dict[ds_idx][tg]
                            continue

                    if not dataset_timegroup_timechoice_dict[ds_idx]:
                        del dataset_timegroup_timechoice_dict[ds_idx]

            dataset_timegroup_timechoice_dict = du.filter_dataset_timegroup_timechoice_dict(
                dataset_timegroup_timechoice_dict,
                min_times=min_times,
                necessary_timepoints=self.necessary_timepoints,
                necessary_timegroups=self.necessary_timegroups,
                data_files=data_files,
                exclude_times=self.exclude_times,
            )

            dataset_order = du.get_dataset_choice_list(
                self.num_examples, self.balance_datasets, dataset_timegroup_timechoice_dict
            )

            timegroup_order = du.get_timegroup_choice_list(
                dataset_order, dataset_timegroup_timechoice_dict, self.balance_timegroups
            )

            if len(timegroup_order) == 0:
                raise ValueError("After subsetting by the master timegroup, no timegroups have enough time points")

            logger.info("Building time choices and cell indices.")

            time_choices = []
            cell_indices = []

            for j in tqdm(range(len(dataset_order))):
                n_timepoints = np.random.choice(n_timepoints_options)

                dataset_indices = int(dataset_order[j])
                timegroup_idx = str(timegroup_order[j])

                available_times = dataset_timegroup_timechoice_dict[dataset_indices][timegroup_idx]["times"]

                if alt_time_per_timegroups is not None:
                    chosen_times = du.random_choice_with_limit(
                        available_times,
                        n_timepoints,
                        self.max_repeat_timepoints,
                        self.max_repeats_specific_timepoint,
                        self.allow_zero_timelapses,
                        self.allow_negative_timelapses,
                        match_time_tolerance,
                        alt_time_per_timegroups[timegroup_idx],
                    )
                else:
                    chosen_times = du.random_choice_with_limit(
                        available_times,
                        n_timepoints,
                        self.max_repeat_timepoints,
                        self.max_repeats_specific_timepoint,
                        self.allow_zero_timelapses,
                        self.allow_negative_timelapses,
                        match_time_tolerance,
                    )

                time_choices.append(chosen_times)

                cell_indices_j = []
                indices_dict = dataset_timegroup_timechoice_dict[dataset_indices][timegroup_idx]["indices"]

                for t in chosen_times:
                    chosen_index = int(np.random.choice(indices_dict[t]))
                    cell_indices_j.append(chosen_index)

                cell_indices.append(cell_indices_j)

            logger.info("Time choices and cell indices successfully built.")

        else:
            logger.info("Building dataset timegroup dictionary (no time groups, one group per dataset).")
            dataset_timegroup_timechoice_dict = {}
            for ds_idx in tqdm(range(len(dataset_list)), desc="Generating dataset timegroup dictionary"):
                ds_idx = int(ds_idx)
                df = dataset_list[ds_idx].to_pandas()
                if not df.empty:
                    times = df[self.time_column].unique().tolist()
                    indices_dict = {t: df.index[df[self.time_column] == t].tolist() for t in times}
                    dataset_timegroup_timechoice_dict[ds_idx] = {"_all_": {"times": times, "indices": indices_dict}}

            n_timepoints_options = list(range(self.min_timepoints, self.max_timepoints + 1))
            min_times = self.min_timepoints - self.max_repeat_timepoints
            dataset_timegroup_timechoice_dict = du.filter_dataset_timegroup_timechoice_dict(
                dataset_timegroup_timechoice_dict,
                min_times=min_times,
                necessary_timepoints=self.necessary_timepoints,
                necessary_timegroups=None,
                data_files=data_files,
                exclude_times=self.exclude_times,
            )

            if len(dataset_timegroup_timechoice_dict) == 0:
                raise ValueError(
                    "After filtering (exclude_times, necessary_timepoints, min_times), "
                    "no datasets have enough time points."
                )

            dataset_order = du.get_dataset_choice_list(
                self.num_examples, self.balance_datasets, dataset_timegroup_timechoice_dict
            )

            timegroup_order = ["__NO_TIME_GROUP__" for _ in dataset_order]
            max_timepoint = max(
                du.flatten_list(
                    [dataset_timegroup_timechoice_dict[d]["_all_"]["times"] for d in dataset_timegroup_timechoice_dict]
                )
            )

            logger.info("Building time choices and cell indices.")
            time_choices = []
            cell_indices = []
            for j in tqdm(range(len(dataset_order)), desc="Building time choices and cell indices"):
                n_timepoints = np.random.choice(n_timepoints_options)
                dataset_indices = int(dataset_order[j])
                timegroup_idx = "_all_"
                available_times = dataset_timegroup_timechoice_dict[dataset_indices][timegroup_idx]["times"]
                chosen_times = du.random_choice_with_limit(
                    available_times,
                    n_timepoints,
                    self.max_repeat_timepoints,
                    self.max_repeats_specific_timepoint,
                    self.allow_zero_timelapses,
                    self.allow_negative_timelapses,
                )
                time_choices.append(chosen_times)
                indices_dict = dataset_timegroup_timechoice_dict[dataset_indices][timegroup_idx]["indices"]
                cell_indices_j = [int(np.random.choice(indices_dict[t])) for t in chosen_times]
                cell_indices.append(cell_indices_j)

        timesteps = [du.get_timestep(time_choice_list) for time_choice_list in time_choices]
        blueprint_dictionary = {
            "data_files": data_files,
            "dataset_order": dataset_order,
            "timegroup_order": timegroup_order,
            "time_choices": time_choices,
            "cell_indices": cell_indices,
            "timesteps": timesteps,
            "max_timepoint": max_timepoint,
            "output_prefix_complete": output_prefix_complete,
            "args": {
                "time_column": self.time_column,
                "time_group_columns": self.time_group_columns,
                "group_by_dataset": self.group_by_dataset,
                "balance_datasets": self.balance_datasets,
                "balance_timegroups": self.balance_timegroups,
                "task_ratio": self.task_ratio,
                "min_timepoints": self.min_timepoints,
                "max_timepoints": self.max_timepoints,
                "max_repeat_timepoints": self.max_repeat_timepoints,
                "max_repeats_specific_timepoint": self.max_repeats_specific_timepoint,
                "allow_zero_timelapses": self.allow_zero_timelapses,
                "allow_negative_timelapses": self.allow_negative_timelapses,
                "exclude_times": self.exclude_times,
                "filter_data": self.filter_data,
                "nproc": self.nproc,
                "model_input_size": self.model_input_size,
                "model_version": self.model_version,
                "seed": self.seed,
            },
        }

        filepath_cell_paragraph_blueprint = f"{output_directory}/{output_prefix_complete}_cell_paragraph_blueprint.json"
        with open(filepath_cell_paragraph_blueprint, "w") as fp:
            json.dump(blueprint_dictionary, fp)
            logger.info(f"Saved cell paragraph blueprint to {filepath_cell_paragraph_blueprint}")

        return dataset_list, blueprint_dictionary

    def generate_time_dictionary(
        self,
        output_directory,
        output_prefix,
        max_timepoint,
        token_dictionary,
    ):
        """Generate time token dictionary.

        Args:
            output_directory: Path to directory for saving time dictionary JSON file.
            output_prefix: Prefix for output file.
            max_timepoint: Maximum timepoint to include.
            token_dictionary: Path to base token dictionary pickle file.

        Returns:
            The augmented token dictionary with time and query tokens.
        """
        if not isinstance(max_timepoint, int):
            raise TypeError("max_timepoint must be an integer.")

        with open(token_dictionary, "r") as fp:
            token_dictionary = json.load(fp)

        if self.allow_negative_timelapses:
            min_time = -max_timepoint
        else:
            min_time = 0

        token_dictionary["<boq>"] = len(token_dictionary)
        token_dictionary["<eoq>"] = len(token_dictionary)

        for i in range(min_time, max_timepoint + 1):
            if i not in token_dictionary.keys():
                token_dictionary[i] = len(token_dictionary)

        os.makedirs(output_directory, exist_ok=True)

        with open(f"{output_directory}/{output_prefix}_time_dictionary.json", "w") as fp:
            json.dump({str(k): v for k, v in token_dictionary.items()}, fp)

        return token_dictionary

    def assemble_cell_paragraphs(
        self,
        dataset_list,
        blueprint_dictionary_file,
        time_token_dictionary_file,
        output_directory,
        output_prefix_complete=None,
        clip_timesteps=False,
        clip_first_cell=True,
        is_train=True,
    ):
        """Assemble cell paragraphs from blueprint and tokenized datasets.

        Args:
            dataset_list: Loaded dataset list (output of generate_cell_paragraph_blueprint).
            blueprint_dictionary_file: Path to blueprint pickle or loaded dict.
            time_token_dictionary_file: Path to time token dictionary pickle or loaded dict.
            output_directory: Path to directory for saving output .dataset.
            output_prefix_complete: Complete prefix for output .dataset.
            clip_timesteps: Whether to clip timesteps out of token dictionary range.
            clip_first_cell: Whether to clip first cell tokens if exceeding model input size.
            is_train: Whether assembling for training or evaluation.

        Returns:
            The assembled cell paragraph dataset.
        """
        if isinstance(time_token_dictionary_file, dict):
            token_dictionary = time_token_dictionary_file
        else:
            with open(time_token_dictionary_file, "r") as fp:
                token_dictionary = json.load(fp)

        token_dictionary = du.convert_token_dictionary_keys(token_dictionary)

        if isinstance(blueprint_dictionary_file, dict):
            blueprint_dictionary = blueprint_dictionary_file
        else:
            with open(blueprint_dictionary_file, "r") as fp:
                blueprint_dictionary = json.load(fp)
        data_files = blueprint_dictionary["data_files"]
        dataset_order = blueprint_dictionary["dataset_order"]
        time_choices = blueprint_dictionary["time_choices"]
        cell_indices = blueprint_dictionary["cell_indices"]
        timesteps = blueprint_dictionary["timesteps"]
        if output_prefix_complete is None:
            output_prefix_complete = blueprint_dictionary["output_prefix_complete"]
        timegroup_order = blueprint_dictionary["timegroup_order"]

        if output_directory[-1] != "/":
            output_directory = output_directory + "/"

        if dataset_list is None:
            dataset_list, data_files = du.load_datasets(data_files, filter_data=self.filter_data, nproc=self.nproc)

        timestep_keys = [k for k in token_dictionary.keys() if isinstance(k, int)]
        if len(timestep_keys) == 0:
            raise ValueError(
                "Token dictionary does not contain any integer keys. "
                "Please ensure that token dictionary contains integer keys for timesteps."
            )

        for i, sublist in enumerate(timesteps):
            if not isinstance(sublist, list):
                raise ValueError(f"Expected a list at index {i}, got {type(sublist)}")
            for j, timestep in enumerate(sublist):
                if isinstance(timestep, int):
                    continue
                elif isinstance(timestep, np.integer):
                    timesteps[i][j] = int(timestep)
                else:
                    raise ValueError(
                        f"Timestep at position [{i}][{j}] is not an integer: {timestep} ({type(timestep)})"
                    )

        max_timestep = max(timestep_keys)
        min_timestep = min(timestep_keys)
        if any(timestep < min_timestep or timestep > max_timestep for timestep in du.flatten_list(timesteps)):
            if not clip_timesteps:
                raise ValueError(
                    f"Timesteps out of range of token dictionary. "
                    f"Please ensure all timesteps are between {min_timestep} and {max_timestep}."
                )
            else:
                logger.info(
                    f"Clipping timesteps out of range of token dictionary. "
                    f"{min_timestep} <= timestep <= {max_timestep}"
                )
                timesteps = [[max(min(t, max_timestep), min_timestep) for t in sublist] for sublist in timesteps]

        timesteps_not_in_token_dict = [
            timestep for timestep in du.flatten_list(timesteps) if timestep not in timestep_keys
        ]
        if len(timesteps_not_in_token_dict) > 0:
            raise ValueError(
                f"Timesteps not present in token dictionary. Missing timesteps: {timesteps_not_in_token_dict}"
            )

        def paragraphs_fn(example):
            idx = example["dataset_order"]
            dataset_i = dataset_list[idx]
            predict_time_or_cell_i = np.random.choice(["time", "cell"], p=[self.task_ratio, 1 - self.task_ratio])

            if is_train is True:
                input_ids_i, response_len_i = du.compose_example(
                    dataset_i,
                    example["cell_indices"],
                    example["timesteps"],
                    example["time_choices"],
                    token_dictionary,
                    predict_time_or_cell_i,
                    True,
                    self.model_input_size,
                    clip_first_cell,
                )
                return {"input_ids": input_ids_i, "response_len": response_len_i}
            else:
                (
                    prompt_i,
                    context_i,
                    question_i,
                    response_i,
                    response_len_i,
                    max_response_len_i,
                    last_time_choice_i,
                    unfiltered_cell_indices_i,
                    unfiltered_dataset_indices_i,
                ) = du.compose_example(
                    dataset_i,
                    example["cell_indices"],
                    example["timesteps"],
                    example["time_choices"],
                    token_dictionary,
                    predict_time_or_cell_i,
                    False,
                    self.model_input_size,
                    clip_first_cell,
                )
                return {
                    "input_ids": prompt_i,
                    "context": context_i,
                    "question": question_i,
                    "response": response_i,
                    "response_len": response_len_i,
                    "max_response_len": max_response_len_i,
                    "last_time_choice": last_time_choice_i,
                    "time_choices": example["time_choices"],
                    "timegroup_order": example["timegroup_order"],
                    "dataset_order": example["dataset_order"],
                    "unfiltered_cell_indices": unfiltered_cell_indices_i,
                    "unfiltered_dataset_indices": unfiltered_dataset_indices_i,
                }

        dataset = Dataset.from_dict(
            {
                "dataset_order": dataset_order,
                "timegroup_order": timegroup_order,
                "cell_indices": cell_indices,
                "timesteps": timesteps,
                "time_choices": time_choices,
            }
        )

        logger.info("Generating dataset with prompts and answers.")
        cell_paragraph_dataset = dataset.map(paragraphs_fn, num_proc=1)

        logger.info("Saving dataset.")

        cell_paragraph_dataset.save_to_disk(str(os.path.join(output_directory, f"{output_prefix_complete}.dataset")))

        return cell_paragraph_dataset
