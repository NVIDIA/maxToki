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
from pathlib import Path

import pytest

from bionemo.maxtoki.tokenizer import MaxTokiTokenizer


PRETRAIN_DATA_PATH = Path("/home/ubuntu/data/cellformer-pretrain/genecorpus_XM_250708_update_shuffled_test100k.dataset")
PRETRAIN_TOKENIZER_PATH = Path("/home/ubuntu/data/cellformer-pretrain/token_dictionary.json")
FINETUNE_DATA_PATH = Path("/home/ubuntu/data/cellformer")
# TODO: update to .json once the collaborator-provided token dictionary is migrated.
# See PICKLE-REMEDIATION.md for the full migration plan.
FINETUNE_TOKENIZER_PATH = Path("/home/ubuntu/data/cellformer/token_dictionary.json")


@pytest.fixture()
def synthetic_token_dictionary():
    d = {
        "<pad>": 0,
        "<mask>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    for i in range(10):
        d[f"GENE{i}"] = 4 + i
    d["<boq>"] = 14
    d["<eoq>"] = 15
    for i in range(10):
        d[str(i)] = 16 + i
    return d


@pytest.fixture()
def tokenizer(synthetic_token_dictionary):
    return MaxTokiTokenizer(synthetic_token_dictionary)


@pytest.fixture()
def real_token_dictionary():
    path = FINETUNE_TOKENIZER_PATH.with_suffix(".json")
    if not path.exists():
        pytest.skip(f"Real token dictionary not found at {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture()
def real_tokenizer(real_token_dictionary):
    return MaxTokiTokenizer(real_token_dictionary)
