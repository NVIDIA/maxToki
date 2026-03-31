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

"""MaxToki data preparation pipeline.

Three-stage pipeline for preparing single-cell RNA-seq data for MaxToki training:

1. **TranscriptomeTokenizer**: Raw h5ad scRNAseq -> HuggingFace .dataset
2. **CellParagraphAssembler**: Tokenized datasets -> training cell paragraphs
3. **QueryAssembler**: Cell paragraphs + alternative data -> evaluation query datasets

Resource pickle files (token dictionary, gene medians, ensembl mapping) must be placed
in the ``resources/`` subdirectory. See ``resources/README.md`` for format details.
"""

from pathlib import Path


RESOURCES_DIR = Path(__file__).parent / "resources"

ENSEMBL_MAPPING_FILE = RESOURCES_DIR / "ensembl_mapping_dict_v1.pkl"
GENE_MEDIAN_FILE = RESOURCES_DIR / "gene_median_dictionary_v1.pkl"
GENE_NAME_ID_FILE = RESOURCES_DIR / "gene_name_id_dict_v1.pkl"
TOKEN_DICTIONARY_FILE = RESOURCES_DIR / "token_dictionary_v1.pkl"

from bionemo.maxtoki.data_prep.cell_paragraph_assembler import CellParagraphAssembler
from bionemo.maxtoki.data_prep.query_assembler import QueryAssembler
from bionemo.maxtoki.data_prep.transcriptome_tokenizer import TranscriptomeTokenizer


__all__ = [
    "ENSEMBL_MAPPING_FILE",
    "GENE_MEDIAN_FILE",
    "GENE_NAME_ID_FILE",
    "RESOURCES_DIR",
    "TOKEN_DICTIONARY_FILE",
    "CellParagraphAssembler",
    "QueryAssembler",
    "TranscriptomeTokenizer",
]
