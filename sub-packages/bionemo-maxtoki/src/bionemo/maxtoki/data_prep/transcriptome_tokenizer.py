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

"""MaxToki transcriptome tokenizer.

Converts raw scRNA-seq h5ad data into rank-value encoded HuggingFace datasets.

**Input data:**

| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required var gene attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required obs cell attribute:* "n_counts"; total read counts in that cell.

| *Optional obs cell attribute:* "filter_pass"; binary indicator of whether cell should be tokenized
  based on user-defined filtering criteria.
| *Optional obs cell attributes:* any other cell metadata can be passed on to the tokenized dataset
  as a custom attribute dictionary as shown below.

**Usage:**

.. code-block :: python

    >>> from bionemo.maxtoki.data_prep import TranscriptomeTokenizer
    >>> tk = TranscriptomeTokenizer("gene_median.json", "token_dictionary.json", custom_attr_name_dict={"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
    >>> tk.tokenize_data("data_directory", "output_directory", "output_prefix")
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from datasets import Dataset
from tqdm import tqdm



warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

logger = logging.getLogger(__name__)


def rank_genes(gene_vector, gene_tokens):
    """Rank gene expression vector by descending value."""
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def tokenize_cell(gene_vector, gene_tokens):
    """Convert normalized gene expression vector to tokenized rank value encoding."""
    nonzero_mask = np.nonzero(gene_vector)[0]
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


def sum_ensembl_ids(
    data_directory,
    collapse_gene_ids,
    gene_mapping_dict,
    gene_token_dict,
    custom_attr_name_dict,
    use_h5ad_index,
    file_format="h5ad",
    chunk_size=512,
):
    """Map Ensembl IDs via gene mapping dict and collapse duplicates by summing counts."""
    if file_format == "h5ad":
        data = sc.read_h5ad(str(data_directory))

        if use_h5ad_index:
            data.var["ensembl_id"] = list(data.var.index)

        assert "ensembl_id" in data.var.columns, "'ensembl_id' column missing from data.var"

        assert "ensembl_id_collapsed" not in data.var.columns, (
            "'ensembl_id_collapsed' column already exists in data.var"
        )
        assert "n_counts" in data.obs.columns, "'n_counts' column missing from data.obs"

        if custom_attr_name_dict is not None:
            for label in custom_attr_name_dict:
                assert label in data.obs.columns, f"Attribute `{label}` not present in data.obs"

        ensembl_ids = data.var.ensembl_id
        if not collapse_gene_ids:
            ensembl_id_check = [gene for gene in ensembl_ids if gene in gene_token_dict.keys()]
            if len(ensembl_id_check) == len(set(ensembl_id_check)):
                return data_directory
            else:
                raise ValueError("Error: data Ensembl IDs non-unique.")

        genes_in_map_dict = [gene for gene in ensembl_ids if gene in gene_mapping_dict.keys()]
        vals_from_map_dict = [gene_mapping_dict.get(gene) for gene in genes_in_map_dict]

        if len(set(genes_in_map_dict)) == len(set(vals_from_map_dict)):
            data.var["ensembl_id_collapsed"] = data.var.ensembl_id.str.upper().map(gene_mapping_dict)
            return data
        else:
            data.var["ensembl_id_collapsed"] = data.var.ensembl_id.str.upper().map(gene_mapping_dict)
            data.var_names = data.var["ensembl_id_collapsed"]
            data = data[:, ~data.var.index.isna()]
            dup_genes = [idx for idx, count in Counter(data.var_names).items() if count > 1]

            num_chunks = int(np.ceil(data.shape[0] / chunk_size))

            processed_genes = []
            for i in tqdm(range(num_chunks)):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, data.shape[0])
                data_chunk = data[start_idx:end_idx, :]

                processed_chunks = []
                for dup_gene in dup_genes:
                    data_dup_gene = data_chunk[:, data_chunk.var_names == dup_gene]
                    df = pd.DataFrame.sparse.from_spmatrix(
                        sp.csc_matrix(data_dup_gene.X),
                        index=data_dup_gene.obs_names,
                        columns=data_dup_gene.var_names,
                    )
                    df_sum = pd.DataFrame(df.sum(axis=1))
                    df_sum.columns = [dup_gene]
                    df_sum.index = data_dup_gene.obs.index
                    processed_chunks.append(df_sum)

                processed_chunks = pd.concat(processed_chunks, axis=1)
                processed_genes.append(processed_chunks)
            processed_genes = pd.concat(processed_genes, axis=0)
            var_df = pd.DataFrame({"ensembl_id_collapsed": processed_genes.columns})
            var_df.index = processed_genes.columns
            processed_genes = sc.AnnData(X=processed_genes, obs=data.obs, var=var_df)

            data_dedup = data[:, ~data.var.index.isin(dup_genes)]
            data_dedup = sc.concat([data_dedup, processed_genes], axis=1)
            data_dedup.obs = data.obs
            return data_dedup
    else:
        raise ValueError(f"File format {file_format} not supported.")


class TranscriptomeTokenizer:
    """Tokenizes raw scRNA-seq h5ad data into rank-value encoded HuggingFace datasets."""

    def __init__(
        self,
        gene_median_file,
        token_dictionary_file,
        custom_attr_name_dict=None,
        time_column="time",
        unique_cell_id_column="unique_cell_id",
        time_group_column="time_group",
        nproc=1,
        chunk_size=512,
        model_input_size=4096,
        collapse_gene_ids=True,
        use_h5ad_index=False,
        keep_counts=False,
        model_version="V1",
        gene_mapping_file=None,
    ):
        """Initialize tokenizer.

        Args:
            custom_attr_name_dict: Dictionary of custom attributes to be added to the dataset.
                Keys are the names of the attributes in the h5ad file.
                Values are the names of the attributes in the dataset.
            time_column: Obs attribute name in .h5ad file for cell timepoint.
            unique_cell_id_column: Obs attribute name for unique cell ID.
            time_group_column: Obs attribute name for time group ID. If provided,
                will be retained in tokenized dataset for cell paragraph assembly.
            nproc: Number of processes to use for dataset mapping.
            chunk_size: Chunk size for anndata tokenizer.
            model_input_size: Max input size of model to truncate input to.
            collapse_gene_ids: Whether to collapse gene IDs based on gene mapping dictionary.
            use_h5ad_index: Use index as Ensembl IDs (only available for h5ad).
            keep_counts: Whether to keep normalized gene counts as a dataset column.
            model_version: Model version string (currently "V1").
            gene_median_file: Path to pickle file with non-zero median gene expression values.
            token_dictionary_file: Path to pickle file with token dictionary.
            gene_mapping_file: Path to pickle file with gene ID mapping dictionary.
        """
        self.custom_attr_name_dict = custom_attr_name_dict
        if self.custom_attr_name_dict is None:
            self.custom_attr_name_dict = {}

        self.time_column = time_column
        self.unique_cell_id_column = unique_cell_id_column
        self.time_group_column = time_group_column

        self.custom_attr_name_dict[self.time_column] = "time"
        self.custom_attr_name_dict[self.unique_cell_id_column] = "unique_cell_id"
        if self.time_group_column is not None:
            self.custom_attr_name_dict[self.time_group_column] = "time_group"

        self.nproc = nproc
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size

        self.model_version = model_version
        if self.model_version not in ["V1"]:
            logger.error("Unrecognized model version. Current options: V1: models pretrained on ~175M cells.")

        with open(gene_median_file, "r") as f:
            self.gene_median_dict = json.load(f)

        with open(token_dictionary_file, "r") as f:
            self.gene_token_dict = json.load(f)

        if ("<bos>" not in self.gene_token_dict) or ("<eos>" not in self.gene_token_dict):
            raise ValueError("<bos> and <eos> required in gene_token_dict.")

        self.collapse_gene_ids = collapse_gene_ids
        self.use_h5ad_index = use_h5ad_index
        self.keep_counts = keep_counts

        if gene_mapping_file is not None:
            with open(gene_mapping_file, "r") as f:
                self.gene_mapping_dict = json.load(f)
        else:
            self.gene_mapping_dict = {k: k for k, _ in self.gene_token_dict.items()}

        self.gene_keys = list(self.gene_token_dict.keys())

        gene_keys_set = set(self.gene_token_dict.keys())
        self.gene_mapping_dict = {k: v for k, v in self.gene_mapping_dict.items() if v in gene_keys_set}

        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["h5ad"] = "h5ad",
        input_identifier: str = "",
        use_generator: bool = False,
    ):
        """Tokenize .h5ad files in data_directory and save as tokenized .dataset in output_directory.

        Args:
            data_directory: Path to directory containing h5ad files.
            output_directory: Path to directory where tokenized data will be saved.
            output_prefix: Prefix for output .dataset.
            file_format: Format of input files (currently only "h5ad").
            input_identifier: Substring identifier; only matching .h5ad files are tokenized.
            use_generator: Whether to use generator or dict for tokenization.
        """
        tokenized_cells, cell_metadata, tokenized_counts = self.tokenize_files(
            Path(data_directory), file_format, input_identifier
        )

        tokenized_dataset = self.create_dataset(
            tokenized_cells,
            cell_metadata,
            tokenized_counts,
            use_generator=use_generator,
        )

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(str(output_path))

    def tokenize_files(self, data_directory, file_format: Literal["h5ad"] = "h5ad", input_identifier: str = ""):
        """Tokenize all matching files in data_directory."""
        tokenized_cells = []
        tokenized_counts = []
        if self.custom_attr_name_dict is not None:
            cell_attr = list(self.custom_attr_name_dict.keys())
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        tokenize_file_fn = self.tokenize_anndata
        if input_identifier == "":
            file_match = f"*.{file_format}"
        else:
            file_match = f"*{input_identifier}*.{file_format}"
        file_found = 0
        for file_path in data_directory.glob(file_match):
            file_found = 1
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata, file_tokenized_counts = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            tokenized_counts += file_tokenized_counts
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

        if file_found == 0:
            raise FileNotFoundError(f"No .{file_format} files found in directory {data_directory}.")
        return tokenized_cells, cell_metadata, tokenized_counts

    def tokenize_anndata(self, adata_file_path, target_sum=10_000):
        """Tokenize a single anndata file into rank-value encoded cells."""
        adata = sum_ensembl_ids(
            adata_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            self.custom_attr_name_dict,
            self.use_h5ad_index,
            file_format="h5ad",
            chunk_size=self.chunk_size,
        )

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        coding_miRNA_loc = np.where([self.genelist_dict.get(i, False) for i in adata.var["ensembl_id_collapsed"]])[0]
        norm_factor_vector = np.array(
            [self.gene_median_dict[i] for i in adata.var["ensembl_id_collapsed"][coding_miRNA_loc]]
        )
        coding_miRNA_ids = adata.var["ensembl_id_collapsed"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array([self.gene_token_dict[i] for i in coding_miRNA_ids])

        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
        elif not var_exists:
            print(f"{adata_file_path} has no obs attribute 'filter_pass'; tokenizing all cells.")
            filter_pass_loc = np.array(list(range(adata.shape[0])))

        tokenized_cells = []
        tokenized_counts = []

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            idx = filter_pass_loc[i : i + self.chunk_size]

            n_counts = adata[idx].obs["n_counts"].values[:, None]
            X_view0 = adata[idx, :].X
            X_view = X_view0[:, coding_miRNA_loc]
            X_norm_unscaled = X_view / n_counts * target_sum
            X_norm = X_norm_unscaled / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)
            X_log1p = np.log1p(X_norm_unscaled)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices]) for i in range(X_norm.shape[0])
            ]

            if self.keep_counts:
                X_log1p = sp.csr_matrix(X_log1p)
                tokenized_counts += [rank_genes(X_norm[i].data, X_log1p[i].data) for i in range(X_norm.shape[0])]

            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        empty_cell_indices = [i for i, cell in enumerate(tokenized_cells) if cell.size == 0]
        if len(empty_cell_indices) > 0:
            logger.warning(
                "Warning: cells without any genes in token dictionary detected. "
                "This is unusual and may indicate empty droplets or otherwise invalid cells "
                "within the input data. Consider further QC prior to tokenization. "
                "Proceeding with excluding empty cells."
            )
            empty_cell_indices.sort(reverse=True)
            for index in empty_cell_indices:
                del tokenized_cells[index]
                if self.keep_counts:
                    del tokenized_counts[index]
            for k, v in file_cell_metadata.items():
                for index in empty_cell_indices:
                    del v[index]
                file_cell_metadata[k] = v

        return tokenized_cells, file_cell_metadata, tokenized_counts

    def create_dataset(
        self,
        tokenized_cells,
        cell_metadata,
        tokenized_counts,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        """Create HuggingFace Dataset from tokenized cells with BOS/EOS wrapping and truncation."""
        print("Creating dataset.")
        dataset_dict = {"input_ids": tokenized_cells}
        if self.keep_counts:
            dataset_dict["counts"] = tokenized_counts

        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        dataset_dict["input_ids"] = [i.tolist() for i in dataset_dict["input_ids"]]
        if self.keep_counts:
            dataset_dict["counts"] = [i.tolist() for i in dataset_dict["counts"]]

        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)

        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_cell_features(example):
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            example["input_ids"] = example["input_ids"][0 : self.model_input_size - 2]

            example["input_ids"] = [self.gene_token_dict.get("<bos>")] + example["input_ids"]
            example["input_ids"] = example["input_ids"] + [self.gene_token_dict.get("<eos>")]

            if self.keep_counts:
                example["counts"] = example["counts"][0 : self.model_input_size - 2]
                example["counts"] = [0.0] + example["counts"]
                example["counts"] = example["counts"] + [0.0]

            example["length"] = len(example["input_ids"])

            return example

        output_dataset_truncated = output_dataset.map(format_cell_features, num_proc=self.nproc)
        return output_dataset_truncated
