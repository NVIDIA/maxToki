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

"""End-to-end comparison of the HuggingFace and BioNeMo regression (timelapse) inference paths.

This mirrors the conversion tests in ``test_conversion.py``: there, ``_compare_hf_logits`` checks
two HF models agree; here we check the *two methods* agree — the original HuggingFace forward pass
(``cf.model`` Llama backbone + the ``linear1`` head) versus the BioNeMo path
(``import_regression_hf`` -> ``predict --regression-head``) — on the same held-out ``time`` rows.

Requires the trained aging artifacts, which are external (not in CI), so the test skips unless they
are present. Point it at them with env vars (defaults under ``/home/ubuntu/data``):

    MAXTOKI_AGING_MODEL_SAFETENSORS  path to aging model.safetensors
    MAXTOKI_AGING_HF_CONFIG          path to a matching Llama config.json (e.g. MaxToki-217M-HF)
    MAXTOKI_AGING_DATASET            path to a HuggingFace dataset with `time` rows + input_ids
    MAXTOKI_AGING_TOKENIZER          path to the extended token_dictionary.json (optional; a
                                     stand-in is built if absent)
    MAXTOKI_AGING_EOQ                <eoq> token id (default 23276)
    MAXTOKI_AGING_LABEL_SCALAR       label scalar (default 1000)
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bionemo.maxtoki.import_regression_hf import MAXTOKI_RMS_NORM_EPS
from bionemo.maxtoki.verify_regression import compare, load_predictions
from bionemo.testing.subprocess_utils import run_command_in_subprocess


AGING_MODEL_SAFETENSORS = Path(
    os.environ.get(
        "MAXTOKI_AGING_MODEL_SAFETENSORS",
        "/home/ubuntu/data/cellformer-aging/aging_trained_maxtoki_217M_HF/model.safetensors",
    )
)
AGING_HF_CONFIG = Path(
    os.environ.get("MAXTOKI_AGING_HF_CONFIG", "/home/ubuntu/data/cellformer-aging/MaxToki-217M-HF/config.json")
)
AGING_DATASET = Path(
    os.environ.get("MAXTOKI_AGING_DATASET", "/home/ubuntu/data/cellformer-aging/heldout_ages_100k.dataset")
)
# The real extended (23277-token) dictionary. If present, the test uses it and lets the
# <eoq> position come from the tokenizer; otherwise it builds a structurally-valid stand-in and
# passes --eoq-token-id (the gene<->id mapping is irrelevant for pre-tokenized inputs).
AGING_TOKENIZER = Path(
    os.environ.get("MAXTOKI_AGING_TOKENIZER", "/home/ubuntu/data/cellformer/token_dictionary.json")
)
EOQ_TOKEN_ID = int(os.environ.get("MAXTOKI_AGING_EOQ", "23276"))
LABEL_SCALAR = float(os.environ.get("MAXTOKI_AGING_LABEL_SCALAR", "1000"))

N_COMPARE = 16
SEQ_LENGTH = 16512
CF_PREFIX = "cf."


def _skip_if_no_aging_artifacts():
    for p in (AGING_MODEL_SAFETENSORS, AGING_HF_CONFIG, AGING_DATASET):
        if not p.exists():
            pytest.skip(f"Aging artifact not found: {p}")


def _first_n_time_rows(n):
    """Return the first ``n`` `time`-task rows from the held-out dataset, in order."""
    from datasets import load_from_disk

    ds = load_from_disk(str(AGING_DATASET))
    rows = []
    for i in range(len(ds)):
        if ds[i]["cell_or_time"] == "time":
            rows.append(ds[i])
            if len(rows) >= n:
                break
    return rows


def _build_time_subset(tmp_path, rows):
    """Persist the selected rows as a HuggingFace dataset for the BioNeMo predict subprocess."""
    from datasets import Dataset

    cols = {k: [r[k] for r in rows] for k in rows[0]}
    out = tmp_path / "time_subset.dataset"
    Dataset.from_dict(cols).save_to_disk(str(out))
    return out


def _write_token_dictionary(tmp_path, vocab_size, eoq_id):
    """Write a structurally-valid token dictionary.

    The inputs are pre-tokenized, so only the vocab size and the special-token ids matter for the
    regression-head path (gene<->id mapping is never used). ``<eoq>`` is placed at ``eoq_id`` and
    ``<pad>`` at 0 to match the trained model.
    """
    specials = {"<pad>": 0, "<eos>": 3, "<eoq>": eoq_id}
    used = set(specials.values())
    # Assign the remaining required specials to the first free low ids.
    free = (i for i in range(1, vocab_size) if i not in used)
    for name in ("<mask>", "<bos>", "<boq>"):
        specials[name] = next(free)
        used.add(specials[name])
    td = {f"GENE_{i}": i for i in range(vocab_size) if i not in used}
    td.update(specials)
    assert sorted(td.values()) == list(range(vocab_size))
    path = tmp_path / "token_dictionary.json"
    path.write_text(json.dumps(td))
    return path


def _hf_regression_predict(rows):
    """Reference predictions from the original HuggingFace forward pass.

    Rebuilds the Llama backbone from the ``cf.model.*`` weights plus the ``linear1`` head, reads the
    final hidden state at the ``<eoq>`` position, applies the head, and rescales by ``LABEL_SCALAR``.
    """
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaModel

    sd = load_file(str(AGING_MODEL_SAFETENSORS))
    hf = json.loads(Path(AGING_HF_CONFIG).read_text())
    vocab, hidden = sd[f"{CF_PREFIX}model.embed_tokens.weight"].shape

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=sd[f"{CF_PREFIX}model.layers.0.mlp.gate_proj.weight"].shape[0],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"],
        max_position_embeddings=hf["max_position_embeddings"],
        rms_norm_eps=MAXTOKI_RMS_NORM_EPS,
        rope_theta=hf["rope_theta"],
        rope_scaling=hf.get("rope_scaling"),
        attention_bias=False,
        tie_word_embeddings=hf.get("tie_word_embeddings", False),
    )
    model = LlamaModel(cfg).float().cuda().eval()
    model.load_state_dict(
        {k[len(f"{CF_PREFIX}model.") :]: v for k, v in sd.items() if k.startswith(f"{CF_PREFIX}model.")}, strict=True
    )
    w = sd["linear1.weight"].float().cuda()
    b = sd["linear1.bias"].float().cuda()

    preds = []
    for r in rows:
        ids = torch.tensor([r["input_ids"]]).cuda()
        eoq_pos = (ids[0] == EOQ_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
        with torch.no_grad():
            h = model(ids, use_cache=False).last_hidden_state[0, eoq_pos]
            preds.append(((h @ w.t() + b).item()) * LABEL_SCALAR)
    return np.array(preds)


@pytest.mark.slow
def test_hf_vs_bionemo_regression_inference(tmp_path):
    """Import the HF regression model into BioNeMo, predict, and assert it matches the HF forward."""
    _skip_if_no_aging_artifacts()

    rows = _first_n_time_rows(N_COMPARE)
    assert rows, "No `time` rows found in the dataset."
    hf_preds = _hf_regression_predict(rows)

    subset = _build_time_subset(tmp_path, rows)

    # Prefer the real extended dictionary; fall back to a structurally-valid stand-in.
    if AGING_TOKENIZER.exists():
        tokenizer = AGING_TOKENIZER
        eoq_flag = ""  # <eoq> comes from the tokenizer
    else:
        from safetensors.torch import load_file

        vocab = load_file(str(AGING_MODEL_SAFETENSORS))[f"{CF_PREFIX}model.embed_tokens.weight"].shape[0]
        tokenizer = _write_token_dictionary(tmp_path, vocab, EOQ_TOKEN_ID)
        eoq_flag = f"--eoq-token-id {EOQ_TOKEN_ID}"

    # Import HF -> BioNeMo.
    result_dir = tmp_path / "converted"
    run_command_in_subprocess(
        f"python -m bionemo.maxtoki.import_regression_hf "
        f"--model-safetensors {AGING_MODEL_SAFETENSORS} "
        f"--hf-config {AGING_HF_CONFIG} "
        f"--tokenizer-path {tokenizer} "
        f"--train-data-path {subset} --val-data-path {subset} --test-data-path {subset} "
        f"--seq-length {SEQ_LENGTH} --result-dir {result_dir} --experiment-name aging_test",
        str(tmp_path),
    )
    ckpts = sorted((result_dir / "aging_test" / "dev" / "checkpoints").glob("*last*"))
    assert ckpts, "Import did not produce a checkpoint."

    # Predict with the explicit regression head.
    preds_dir = tmp_path / "preds"
    run_command_in_subprocess(
        f"python -m bionemo.maxtoki.predict "
        f"--initial-ckpt-path {ckpts[-1]} --data-path {subset} --tokenizer-path {tokenizer} "
        f"--output-dir {preds_dir} --seq-length {SEQ_LENGTH} --write-interval batch --precision fp32 "
        f"--regression-head --label-scalar {LABEL_SCALAR} {eoq_flag}",
        str(tmp_path),
    )

    bionemo_preds = load_predictions(preds_dir)
    metrics = compare(bionemo_preds, hf_preds)
    print(f"HF vs BioNeMo regression: {metrics}")

    # fp32 against the fp32 HF forward: ~0.001. A wrong RMSNorm epsilon gives ~2.2, which a bf16 comparison (~1.2)
    # cannot reliably separate from a correct conversion.
    assert metrics["n"] == N_COMPARE
    assert metrics["pearson_vs_hf"] > 0.99999, metrics
    assert metrics["mae_vs_hf"] < 0.05, metrics
