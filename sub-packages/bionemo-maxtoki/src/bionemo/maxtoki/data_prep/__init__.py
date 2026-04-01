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

Resource JSON files (token dictionary, gene medians, ensembl mapping) must be placed
in the ``resources/`` subdirectory. See ``resources/README.md`` for format details.
"""

from bionemo.maxtoki.data_prep.cell_paragraph_assembler import CellParagraphAssembler
from bionemo.maxtoki.data_prep.query_assembler import QueryAssembler
from bionemo.maxtoki.data_prep.transcriptome_tokenizer import TranscriptomeTokenizer


__all__ = [
    "CellParagraphAssembler",
    "QueryAssembler",
    "TranscriptomeTokenizer",
]
