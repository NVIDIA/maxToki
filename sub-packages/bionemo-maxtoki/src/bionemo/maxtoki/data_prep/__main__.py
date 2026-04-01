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

"""CLI entry point for the MaxToki data preparation pipeline.

Run as::

    python -m bionemo.maxtoki.data_prep tokenize --help
    python -m bionemo.maxtoki.data_prep assemble-paragraphs --help
    python -m bionemo.maxtoki.data_prep assemble-queries --help
"""

from __future__ import annotations

import argparse
import json


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the data preparation CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m bionemo.maxtoki.data_prep",
        description="MaxToki data preparation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- tokenize ---
    tok = subparsers.add_parser("tokenize", help="Tokenize scRNAseq h5ad files into HuggingFace datasets.")
    tok.add_argument("--data-directory", required=True, help="Directory containing .h5ad files.")
    tok.add_argument("--output-directory", required=True, help="Directory for output .dataset.")
    tok.add_argument("--output-prefix", required=True, help="Prefix for output .dataset.")
    tok.add_argument("--gene-median-file", default=None, help="Path to gene median JSON file.")
    tok.add_argument("--token-dictionary-file", default=None, help="Path to token dictionary JSON file.")
    tok.add_argument("--gene-mapping-file", default=None, help="Path to ensembl mapping JSON file.")
    tok.add_argument(
        "--custom-attrs", default=None, help='JSON dict of custom attributes, e.g. \'{"cell_type": "cell_type"}\'.'
    )
    tok.add_argument("--nproc", type=int, default=1, help="Number of processes.")
    tok.add_argument("--model-input-size", type=int, default=4096, help="Max model input size.")
    tok.add_argument("--chunk-size", type=int, default=512, help="Chunk size for anndata processing.")
    tok.add_argument("--no-collapse-gene-ids", action="store_true", help="Disable gene ID collapsing.")
    tok.add_argument("--use-h5ad-index", action="store_true", help="Use h5ad var index as ensembl_id.")
    tok.add_argument("--keep-counts", action="store_true", help="Keep normalized gene counts column.")

    # --- assemble-paragraphs ---
    ap = subparsers.add_parser("assemble-paragraphs", help="Assemble cell paragraphs from tokenized datasets.")
    ap.add_argument("--data-directory", required=True, help="Path to .dataset file or directory.")
    ap.add_argument("--output-directory", required=True, help="Directory for output files.")
    ap.add_argument("--output-prefix", required=True, help="Prefix for output files.")
    ap.add_argument("--token-dictionary-file", default=None, help="Path to token dictionary JSON file.")
    ap.add_argument("--max-timepoint", type=int, required=True, help="Maximum timepoint for time dictionary.")
    ap.add_argument("--num-examples", type=int, default=10_000_000, help="Number of cell paragraphs.")
    ap.add_argument("--min-timepoints", type=int, default=3, help="Minimum timepoints per paragraph.")
    ap.add_argument("--max-timepoints", type=int, default=4, help="Maximum timepoints per paragraph.")
    ap.add_argument("--task-ratio", type=float, default=0.5, help="Ratio of timelapse vs cell tasks.")
    ap.add_argument("--time-group-columns", nargs="*", default=None, help="Column names for trajectory grouping.")
    ap.add_argument("--nproc", type=int, default=1, help="Number of processes.")
    ap.add_argument("--model-input-size", type=int, default=16_384, help="Max model input size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--is-train", action="store_true", default=True, help="Assemble for training (default).")
    ap.add_argument("--is-eval", action="store_true", help="Assemble for evaluation.")

    # --- assemble-queries ---
    aq = subparsers.add_parser("assemble-queries", help="Assemble evaluation query datasets.")
    aq.add_argument("--blueprint-dictionary-file", required=True, help="Path to blueprint JSON file.")
    aq.add_argument("--time-token-dictionary-file", required=True, help="Path to time token dictionary JSON file.")
    aq.add_argument("--cell-paragraph-dataset-file", required=True, help="Path to cell paragraph .dataset.")
    aq.add_argument("--output-directory", required=True, help="Directory for output .dataset.")
    aq.add_argument("--output-prefix", default=None, help="Prefix for output .dataset.")
    aq.add_argument("--query-alt-directory", default=None, help="Directory with alternative query datasets.")
    aq.add_argument("--match-time-tolerance", type=int, default=None, help="Time tolerance for matching.")
    aq.add_argument("--time-group-columns", nargs="*", default=None, help="Column names for trajectory grouping.")
    aq.add_argument("--nproc", type=int, default=1, help="Number of processes.")
    aq.add_argument("--model-input-size", type=int, default=16_384, help="Max model input size.")
    aq.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def main(argv=None):
    """Run the data preparation CLI."""
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.command == "tokenize":
        from bionemo.maxtoki.data_prep import TranscriptomeTokenizer

        if not args.gene_median_file:
            raise ValueError("--gene-median-file is required.")
        if not args.token_dictionary_file:
            raise ValueError("--token-dictionary-file is required.")

        custom_attrs = json.loads(args.custom_attrs) if args.custom_attrs else None
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=custom_attrs,
            nproc=args.nproc,
            chunk_size=args.chunk_size,
            model_input_size=args.model_input_size,
            collapse_gene_ids=not args.no_collapse_gene_ids,
            use_h5ad_index=args.use_h5ad_index,
            keep_counts=args.keep_counts,
            gene_median_file=args.gene_median_file,
            token_dictionary_file=args.token_dictionary_file,
            gene_mapping_file=args.gene_mapping_file,
        )
        tk.tokenize_data(args.data_directory, args.output_directory, args.output_prefix)

    elif args.command == "assemble-paragraphs":
        from bionemo.maxtoki.data_prep import CellParagraphAssembler

        if not args.token_dictionary_file:
            raise ValueError("--token-dictionary-file is required.")

        is_train = not args.is_eval

        cpa = CellParagraphAssembler(
            num_examples=args.num_examples,
            time_group_columns=args.time_group_columns,
            min_timepoints=args.min_timepoints,
            max_timepoints=args.max_timepoints,
            task_ratio=args.task_ratio,
            nproc=args.nproc,
            model_input_size=args.model_input_size,
            seed=args.seed,
        )

        dataset_list, blueprint_dictionary = cpa.generate_cell_paragraph_blueprint(
            args.data_directory, args.output_directory, args.output_prefix
        )

        cpa.generate_time_dictionary(
            args.output_directory, args.output_prefix, args.max_timepoint, token_dictionary=args.token_dictionary_file
        )

        output_prefix_complete = blueprint_dictionary["output_prefix_complete"]
        blueprint_file = f"{args.output_directory}/{output_prefix_complete}_cell_paragraph_blueprint.json"
        time_dict_file = f"{args.output_directory}/{args.output_prefix}_time_dictionary.json"

        cpa.assemble_cell_paragraphs(
            dataset_list,
            blueprint_file,
            time_dict_file,
            args.output_directory,
            output_prefix_complete,
            is_train=is_train,
        )

    elif args.command == "assemble-queries":
        from bionemo.maxtoki.data_prep import QueryAssembler

        qa = QueryAssembler(
            match_time_tolerance=args.match_time_tolerance,
            time_group_columns=args.time_group_columns,
            nproc=args.nproc,
            model_input_size=args.model_input_size,
            seed=args.seed,
        )
        qa.build_eval_cell_query_dataset(
            args.blueprint_dictionary_file,
            args.time_token_dictionary_file,
            args.cell_paragraph_dataset_file,
            query_alt_directory=args.query_alt_directory,
            output_directory=args.output_directory,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()
