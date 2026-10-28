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

r"""Compare BioNeMo regression-head predictions against HF-generated reference predictions.

Reads the prediction files written by ``bionemo.maxtoki.predict --regression-head`` (one
``predictions__rank_*__batch_*.pt`` per batch) and compares the per-sample timelapse
predictions to a reference pickle produced by the original HuggingFace code. The reference
pickle is a dict with ``generated_answer_timelapse`` (the HF model's predictions) and,
optionally, ``ground_truth_answer_timelapse``.

Predictions are aligned to the reference by batch order (the predict dataset must be the
``time`` rows in the same order the reference was generated, e.g. via ``--write-interval batch``
on a single GPU).

Example:
    python -m bionemo.maxtoki.verify_regression \\
        --predictions-dir ./predictions \\
        --reference base_heldout_ages_217M_100M.pickle
"""

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


__all__: Sequence[str] = ("compare", "load_predictions", "main")


def load_predictions(predictions_dir: Path) -> np.ndarray:
    """Load and order per-batch regression predictions written by PredictionWriter.

    Each file is a dict containing ``regression_preds`` (shape [1, batch]) and ``batch_idx``.
    Files are sorted by ``batch_idx`` so the output matches the dataset/reference order.
    """
    files = list(Path(predictions_dir).glob("predictions__rank_*.pt"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    # batch_idx restarts per rank, so multi-rank output cannot be ordered by it alone.
    ranks = {f.name.split("__")[1] for f in files}
    if len(ranks) > 1:
        raise ValueError(
            f"Found predictions from {len(ranks)} ranks ({sorted(ranks)}) in {predictions_dir}. "
            "Alignment to the reference requires single-GPU prediction; re-run with devices=1."
        )

    batches = []
    for f in files:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        preds = obj["regression_preds"].float().flatten()
        batch_idx = int(obj["batch_idx"].item()) if "batch_idx" in obj else len(batches)
        batches.append((batch_idx, preds))

    batches.sort(key=lambda x: x[0])
    return torch.cat([p for _, p in batches]).numpy()


def compare(preds: np.ndarray, reference: np.ndarray, ground_truth: np.ndarray | None = None) -> dict:
    """Compute agreement metrics between BioNeMo preds and the HF reference predictions."""
    n = min(len(preds), len(reference))
    preds, reference = preds[:n], reference[:n]

    diff = preds - reference
    metrics = {
        "n": n,
        "mae_vs_hf": float(np.mean(np.abs(diff))),
        "max_abs_vs_hf": float(np.max(np.abs(diff))),
        "rmse_vs_hf": float(np.sqrt(np.mean(diff**2))),
        "pearson_vs_hf": float(np.corrcoef(preds, reference)[0, 1]),
    }
    if ground_truth is not None:
        gt = np.asarray(ground_truth[:n], dtype=float)
        metrics["mae_bionemo_vs_gt"] = float(np.mean(np.abs(preds - gt)))
        metrics["mae_hf_vs_gt"] = float(np.mean(np.abs(reference - gt)))
        metrics["pearson_bionemo_vs_gt"] = float(np.corrcoef(preds, gt)[0, 1])
    return metrics


def main():
    """CLI entrypoint for verification."""
    p = argparse.ArgumentParser(description="Compare BioNeMo regression preds vs HF reference.")
    p.add_argument("--predictions-dir", type=Path, required=True, help="Dir with predictions__rank_*.pt files.")
    p.add_argument("--reference", type=Path, required=True, help="HF reference pickle.")
    p.add_argument(
        "--reference-key", type=str, default="generated_answer_timelapse", help="Key for HF predictions in the pickle."
    )
    p.add_argument(
        "--ground-truth-key",
        type=str,
        default="ground_truth_answer_timelapse",
        help="Key for ground-truth values in the pickle (optional).",
    )
    p.add_argument(
        "--mae-threshold",
        type=float,
        default=0.05,
        help="Fail if MAE vs HF exceeds this. The default assumes predictions made with --precision fp32: a "
        "correct conversion lands ~0.001, while one with the wrong RMSNorm epsilon lands ~2.2. bf16 predictions "
        "of a correct conversion land ~1.2, too close to the wrong-epsilon value to tell the two apart.",
    )
    args = p.parse_args()

    preds = load_predictions(args.predictions_dir)
    with open(args.reference, "rb") as f:
        ref = pickle.load(f)
    reference = np.asarray(ref[args.reference_key], dtype=float)
    ground_truth = np.asarray(ref[args.ground_truth_key], dtype=float) if args.ground_truth_key in ref else None

    metrics = compare(preds, reference, ground_truth)
    print("=== BioNeMo vs HuggingFace regression predictions ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if ground_truth is not None:
        print(
            "  note: the *_vs_gt metrics measure the aging model's own accuracy, not conversion\n"
            "        fidelity — compare mae_bionemo_vs_gt against mae_hf_vs_gt, which should match.\n"
            "        Conversion fidelity is mae_vs_hf / pearson_vs_hf."
        )

    print("  sample (bionemo | hf | gt):")
    for i in range(min(8, metrics["n"])):
        gt_s = f"{ground_truth[i]:9.2f}" if ground_truth is not None else "    n/a"
        print(f"    {preds[i]:9.3f} | {reference[i]:9.3f} | {gt_s}")

    if metrics["mae_vs_hf"] > args.mae_threshold:
        raise SystemExit(
            f"MAE vs HF {metrics['mae_vs_hf']:.4f} exceeds threshold {args.mae_threshold}. If the predictions were "
            "made in bf16, rerun predict with --precision fp32; otherwise the conversion is likely incorrect."
        )
    print(f"\nPASS: MAE vs HF {metrics['mae_vs_hf']:.4f} within threshold {args.mae_threshold}")


if __name__ == "__main__":
    main()
