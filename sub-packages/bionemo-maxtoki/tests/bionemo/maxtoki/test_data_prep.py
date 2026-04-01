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

import json

import numpy as np
import pytest

from bionemo.maxtoki.data_prep import dataset_utils as du


# ---------------------------------------------------------------------------
# Synthetic resource fixtures
# ---------------------------------------------------------------------------

NUM_GENES = 50


def _gene_ids(n=NUM_GENES):
    return [f"ENSG{i:011d}" for i in range(n)]


@pytest.fixture()
def synthetic_token_dict(tmp_path):
    d = {}
    for i, gene in enumerate(_gene_ids()):
        d[gene] = i
    d["<bos>"] = NUM_GENES
    d["<eos>"] = NUM_GENES + 1
    d["<pad>"] = NUM_GENES + 2
    path = tmp_path / "token_dictionary.json"
    with open(path, "w") as f:
        json.dump(d, f)
    return d, path


@pytest.fixture()
def synthetic_gene_median_dict(tmp_path):
    d = {gene: float(np.random.uniform(0.01, 1.0)) for gene in _gene_ids()}
    path = tmp_path / "gene_median_dictionary.json"
    with open(path, "w") as f:
        json.dump(d, f)
    return d, path


@pytest.fixture()
def synthetic_ensembl_mapping(tmp_path):
    d = {gene: gene for gene in _gene_ids()}
    path = tmp_path / "ensembl_mapping_dict.json"
    with open(path, "w") as f:
        json.dump(d, f)
    return d, path


# ---------------------------------------------------------------------------
# Synthetic h5ad fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_h5ad(tmp_path):
    anndata = pytest.importorskip("anndata")
    scipy_sparse = pytest.importorskip("scipy.sparse")

    n_cells = 30
    gene_ids = _gene_ids()
    n_genes = len(gene_ids)

    rng = np.random.default_rng(42)
    X = scipy_sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=rng)
    X.data = np.abs(rng.poisson(5, size=X.data.shape)).astype(np.float32)

    import pandas as pd

    obs = pd.DataFrame(
        {
            "n_counts": np.array(X.sum(axis=1)).flatten(),
            "time": rng.choice([0, 1, 2, 3, 4], size=n_cells),
            "unique_cell_id": [f"cell_{i}" for i in range(n_cells)],
            "time_group": rng.choice(["groupA", "groupB"], size=n_cells),
        }
    )
    var = pd.DataFrame({"ensembl_id": gene_ids}, index=gene_ids)

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    h5ad_dir = tmp_path / "h5ad_data"
    h5ad_dir.mkdir()
    h5ad_path = h5ad_dir / "test_data.h5ad"
    adata.write_h5ad(h5ad_path)

    return h5ad_dir, h5ad_path


# ---------------------------------------------------------------------------
# Unit tests: pure utility functions
# ---------------------------------------------------------------------------


class TestDatasetUtils:
    def test_flatten_list(self):
        assert du.flatten_list([[1, 2], [3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
        assert du.flatten_list([]) == []

    def test_get_timestep(self):
        assert du.get_timestep([0, 5, 10, 20]) == [5, 5, 10]
        assert du.get_timestep([10, 5, 0]) == [-5, -5]

    def test_get_weighted_probs(self):
        probs = du.get_weighted_probs([100, 100], 0.5)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_compose_question(self):
        token_dict = {"<boq>": 100, "<eoq>": 101}
        result = du.compose_question([1, 2, 3], token_dict)
        assert result == [100, 1, 2, 3, 101]

    def test_truncate_before_first_eos(self):
        eos = 99
        prompt = [1, 2, 3, 4, 5, eos, 7, 8]
        result = du.truncate_before_first_eos(prompt, len(prompt) + 2, len(prompt), eos)
        assert eos in result
        assert len(result) < len(prompt)

    def test_truncate_before_first_eos_no_eos(self):
        with pytest.raises(ValueError, match="No <eos> token"):
            du.truncate_before_first_eos([1, 2, 3], 5, 3, 99)

    def test_convert_token_dictionary_keys(self):
        d = {"ENSG001": 0, "<bos>": 1, "3": 2, "5.0": 3, 7: 4}
        result = du.convert_token_dictionary_keys(d)
        assert "ENSG001" in result
        assert "<bos>" in result
        assert 3 in result
        assert 5 in result
        assert 7 in result
        assert "3" not in result
        assert "5.0" not in result

    def test_convert_token_dictionary_keys_no_numeric(self):
        d = {"ENSG001": 0, "<bos>": 1}
        result = du.convert_token_dictionary_keys(d)
        assert result == d

    def test_smart_concatenate_mismatched_dtypes(self):
        """smart_concatenate succeeds when datasets have the same columns but different dtypes.

        This is the bug fixed by the external team: plain concatenate_datasets raises an error
        when different .dataset files are produced from different AnnData sources and end up with
        e.g. int32 vs int64 for the same numeric column.
        """
        import datasets as hf_datasets

        ds_int32 = hf_datasets.Dataset.from_dict(
            {"values": [1, 2, 3]},
            features=hf_datasets.Features({"values": hf_datasets.Value("int32")}),
        )
        ds_int64 = hf_datasets.Dataset.from_dict(
            {"values": [4, 5, 6]},
            features=hf_datasets.Features({"values": hf_datasets.Value("int64")}),
        )

        # Verify the bug: plain concatenate_datasets fails on mismatched dtypes.
        with pytest.raises(Exception):
            hf_datasets.concatenate_datasets([ds_int32, ds_int64])

        # Verify the fix: smart_concatenate succeeds and preserves all rows.
        result = du.smart_concatenate([ds_int32, ds_int64])
        assert len(result) == 6
        assert result["values"] == [1, 2, 3, 4, 5, 6]

    def test_smart_concatenate_empty(self):
        result = du.smart_concatenate([])
        assert result == []

    def test_smart_concatenate_matching_dtypes(self):
        import datasets as hf_datasets

        ds1 = hf_datasets.Dataset.from_dict({"x": [1, 2]})
        ds2 = hf_datasets.Dataset.from_dict({"x": [3, 4]})
        result = du.smart_concatenate([ds1, ds2])
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Unit tests: class initialization
# ---------------------------------------------------------------------------


class TestClassInitialization:
    def test_cell_paragraph_assembler_init(self):
        from bionemo.maxtoki.data_prep.cell_paragraph_assembler import CellParagraphAssembler

        cpa = CellParagraphAssembler(num_examples=100, seed=0)
        assert cpa.num_examples == 100
        assert cpa.seed == 0

    def test_cell_paragraph_assembler_invalid_zero_timelapses(self):
        from bionemo.maxtoki.data_prep.cell_paragraph_assembler import CellParagraphAssembler

        with pytest.raises(ValueError, match="allow_zero_timelapses"):
            CellParagraphAssembler(allow_zero_timelapses="invalid")

    def test_query_assembler_init(self):
        from bionemo.maxtoki.data_prep.query_assembler import QueryAssembler

        qa = QueryAssembler(seed=123)
        assert qa.seed == 123

    def test_query_assembler_invalid_zero_timelapses(self):
        from bionemo.maxtoki.data_prep.query_assembler import QueryAssembler

        with pytest.raises(ValueError, match="allow_zero_timelapses"):
            QueryAssembler(allow_zero_timelapses="bogus")


# ---------------------------------------------------------------------------
# Unit tests: TranscriptomeTokenizer initialization
# ---------------------------------------------------------------------------


class TestTranscriptomeTokenizer:
    def test_init(self, synthetic_token_dict, synthetic_gene_median_dict, synthetic_ensembl_mapping):
        from bionemo.maxtoki.data_prep.transcriptome_tokenizer import TranscriptomeTokenizer

        _, tok_path = synthetic_token_dict
        _, med_path = synthetic_gene_median_dict
        _, map_path = synthetic_ensembl_mapping

        tk = TranscriptomeTokenizer(
            gene_median_file=med_path,
            token_dictionary_file=tok_path,
            gene_mapping_file=map_path,
        )
        assert len(tk.gene_token_dict) == NUM_GENES + 3  # genes + bos + eos + pad
        assert "<bos>" in tk.gene_token_dict
        assert "<eos>" in tk.gene_token_dict

    def test_rank_genes(self):
        from bionemo.maxtoki.data_prep.transcriptome_tokenizer import rank_genes

        values = np.array([0.1, 0.5, 0.3])
        tokens = np.array([10, 20, 30])
        result = rank_genes(values, tokens)
        assert result[0] == 20  # highest value

    def test_tokenize_cell(self):
        from bionemo.maxtoki.data_prep.transcriptome_tokenizer import tokenize_cell

        values = np.array([0.0, 0.5, 0.0, 0.3])
        tokens = np.array([10, 20, 30, 40])
        result = tokenize_cell(values, tokens)
        assert len(result) == 2  # only nonzero genes
        assert result[0] == 20  # highest nonzero value


# ---------------------------------------------------------------------------
# E2E test: full tokenize → assemble pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestE2EPipeline:
    def test_tokenize_and_assemble(
        self,
        tmp_path,
        synthetic_h5ad,
        synthetic_token_dict,
        synthetic_gene_median_dict,
        synthetic_ensembl_mapping,
    ):
        from bionemo.maxtoki.data_prep.cell_paragraph_assembler import CellParagraphAssembler
        from bionemo.maxtoki.data_prep.transcriptome_tokenizer import TranscriptomeTokenizer

        h5ad_dir, _ = synthetic_h5ad
        tok_dict, tok_path = synthetic_token_dict
        _, med_path = synthetic_gene_median_dict
        _, map_path = synthetic_ensembl_mapping

        output_dir = tmp_path / "tokenized"
        output_dir.mkdir()

        # Stage 1: Tokenize
        tk = TranscriptomeTokenizer(
            gene_median_file=med_path,
            token_dictionary_file=tok_path,
            gene_mapping_file=map_path,
            model_input_size=128,
            chunk_size=16,
        )
        tk.tokenize_data(str(h5ad_dir), str(output_dir), "test")

        tokenized_path = output_dir / "test.dataset"
        assert tokenized_path.exists()

        # Stage 2: Blueprint
        paragraph_dir = tmp_path / "paragraphs"
        paragraph_dir.mkdir()

        cpa = CellParagraphAssembler(
            num_examples=20,
            min_timepoints=2,
            max_timepoints=3,
            seed=42,
            model_input_size=4096,
            allow_zero_timelapses="question",
        )

        dataset_list, blueprint = cpa.generate_cell_paragraph_blueprint(
            str(tokenized_path), str(paragraph_dir), "test"
        )
        assert "max_timepoint" in blueprint
        assert len(blueprint["dataset_order"]) == 20

        # Stage 2b: Time dictionary
        time_dict = cpa.generate_time_dictionary(
            str(paragraph_dir), "test", blueprint["max_timepoint"], token_dictionary=tok_path
        )
        assert "<boq>" in time_dict
        assert "<eoq>" in time_dict

        # Stage 2c: Assemble
        output_prefix = blueprint["output_prefix_complete"]
        blueprint_file = str(paragraph_dir / f"{output_prefix}_cell_paragraph_blueprint.json")
        time_dict_file = str(paragraph_dir / "test_time_dictionary.json")

        result = cpa.assemble_cell_paragraphs(
            dataset_list,
            blueprint_file,
            time_dict_file,
            str(paragraph_dir),
            output_prefix,
            is_train=True,
        )

        assert "input_ids" in result.column_names
        assert "response_len" in result.column_names
        assert len(result) == 20


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parser_creates(self):
        from bionemo.maxtoki.data_prep.__main__ import get_parser

        parser = get_parser()
        assert parser is not None

    def test_tokenize_help(self):
        from bionemo.maxtoki.data_prep.__main__ import get_parser

        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["tokenize", "--help"])
