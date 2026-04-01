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

import argparse
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Sequence, get_args

import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.maxtoki.datamodule import MaxTokiDataModule
from bionemo.maxtoki.generate_utils import (
    PredictTimingCallback,
    maxtoki_generate_predict_step,
    maxtoki_generate_predict_step_naive,
)
from bionemo.maxtoki.model import (
    MaxTokiMultitaskFineTuneConfig,
    maxtoki_forward_step,
    maxtoki_headless_predict_step,
)
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.llm.lightning import BionemoLightningModule, PassthroughLossReduction, default_megatron_optimizer
from bionemo.maxtoki.lightning import PassthroughEverythingLossReduction
from bionemo.llm.utils.callbacks import PredictionWriter
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size


__all__: Sequence[str] = ("get_parser", "predict")


def predict(
    ckpt_dir: str,
    output_dir: Path,
    data_path: Path | str,
    tokenizer_path: Path | str,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    work_dir: Path | None = None,
    micro_batch_size: int = 1,
    no_sequence_parallel: bool = False,
    seq_length: int = 2048,
    use_headless_mse_regression: bool = False,
    precision: PrecisionTypes = "bf16-mixed",
    generate_next_cell: bool = False,
    write_interval: Literal["epoch", "batch"] = "epoch",
    devices: int = 1,
    # Only used for Generation.
    max_tokens_to_generate: int = 4096,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    buffer_size_gb: float = 20.0,
    buffer_guaranteed_fraction: float = 0.1,
    chunk_size_tokens: int = 4096,
    buffer_overflow_factor: float = 1.0,
    using_pretrain_dataset: bool = False,
    naive_benchmarking_only: bool = False,
    limit_predict_batches_to_n: Optional[int] = None,
):
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())

    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists, files will be written here.
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size

    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=1,
        devices=devices,
        accumulate_grad_batches=1,
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )
    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=tensor_parallel_size > 1 and sequence_parallel,
            ckpt_load_strictness="log_all",
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                seq_len=8192,
                output_log=False,  # this is needed for predict step to work
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        # These keys are for the TimeLapse prediction task.
        # To add extra metadata from the batch, the predict_step_function needs to propagate them forward.
        callbacks=[
            PredictionWriter(
                output_dir=output_dir,
                write_interval=write_interval,
                batch_dim_key_defaults={"regression_preds": 1},
                seq_dim_key_defaults={"regression_preds": 0},
                collate_batch=not generate_next_cell,  # Generate next cell has a different prediction structure.
            ),
            PredictTimingCallback(log_tokens_per_sec=True, show_batch_progress=not generate_next_cell),
        ],
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
        ),
    )

    config = MaxTokiMultitaskFineTuneConfig(
        # I think these can basically all use defaults because we load them from the underlying checkpoint
        initial_ckpt_path=ckpt_dir,
        seq_length=seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
    )

    trainer.strategy._setup_optimizers = False

    nemo_logger = NeMoLogger(log_dir=work_dir)
    nemo_logger.setup(trainer, resume_if_exists=True)

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        restore_config=nl.RestoreConfig(
            path=str(ckpt_dir),  # NeMo expects a string path.
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )

    with open(tokenizer_path) as f:
        token_dictionary = json.load(f)
    # NOTE: this fixes a dtype serialization error when checkpointing. Suspect we have different token dictionaries.
    token_dictionary = {k: int(v) for k, v in token_dictionary.items()}

    tokenizer = MaxTokiTokenizer(token_dictionary=token_dictionary)

    if naive_benchmarking_only:
        base_pred_step_func = maxtoki_generate_predict_step_naive
    else:
        base_pred_step_func = maxtoki_generate_predict_step

    if generate_next_cell:
        if using_pretrain_dataset:
            predict_step_func = partial(
                base_pred_step_func,
                using_pretrain_dataset=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                buffer_size_gb=buffer_size_gb,
                buffer_guaranteed_fraction=buffer_guaranteed_fraction,
                chunk_size_tokens=chunk_size_tokens,
                buffer_overflow_factor=buffer_overflow_factor,
                max_tokens_to_generate=max_tokens_to_generate,
            )
        else:
            predict_step_func = partial(
                base_pred_step_func,
                using_pretrain_dataset=False,  # Only difference in this conditional.
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                buffer_size_gb=buffer_size_gb,
                buffer_guaranteed_fraction=buffer_guaranteed_fraction,
                chunk_size_tokens=chunk_size_tokens,
                buffer_overflow_factor=buffer_overflow_factor,
                max_tokens_to_generate=max_tokens_to_generate,
            )
    else:
        # 'maxtoki_headless_predict_step' applies a filter to the batch to only include inputs for the TimeBetweenCells task.
        #    This is because it produces a single output per input, thus KVC is not really needed. That said, it is possible
        #      to unify this with the generate approach, but generated tokens would need to be capped based on the task, and it would
        #      need to know which task it is performing.
        predict_step_func = maxtoki_headless_predict_step

    forward_step = maxtoki_forward_step

    model = BionemoLightningModule(
        config=config,
        optimizer=default_megatron_optimizer(),
        data_step=llm.gpt_data_step,
        forward_step=forward_step,
        predict_step=predict_step_func,
        # Generate next cell uses a different prediction structure, does not rely on collation. 'PassthroughLossReduction' applies collation.
        predict_loss_reduction_class=PassthroughEverythingLossReduction
        if generate_next_cell
        else PassthroughLossReduction,
        tokenizer=tokenizer,
        model_transform=None,
    )

    resume.setup(trainer, model)

    datamodule = MaxTokiDataModule(
        tokenizer=tokenizer,
        train_dataset_path=None,
        val_dataset_path=None,
        test_dataset_path=None,
        predict_dataset_path=data_path,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        limit_predict_batches_to_n=limit_predict_batches_to_n,
    )

    trainer.predict(model, datamodule=datamodule)


def get_parser():
    """Return the cli parser for this tool."""
    parser = argparse.ArgumentParser(description="Predict with MaxToki on single cell data.")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=False,
        help="Path to the tokenizer file. Used with HF dataset. If not present, assumes the file exists in the parent directory to the train-data-path.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the data base directory",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--work-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of dataset workers. Default is 0.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=False,
        default=2048,
        help="Sequence length of cell. Default is 2048.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=1,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--initial-ckpt-path",
        type=str,
        required=False,
        default=None,
        help="Path to a saved checkpoint. Uses the weights from this checkpoint to initialize the model.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--generate-next-cell",
        action="store_true",
        default=False,
        help="Generate next cell predictions.",
    )
    parser.add_argument(
        "--write-interval",
        required=False,
        default="epoch",
        choices=["epoch", "batch"],
        help="Write interval. Default is 'epoch'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=0,
        help="Use top-k tokens when sampling from the logits. Only used for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.0,
        help="Top-p. Only used for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="Temperature for token sampling. Only used for generation.",
    )
    parser.add_argument(
        "--buffer-size-gb",
        type=float,
        required=False,
        default=20.0,
        help="Buffer size in GB for KV caching. Only used for generation.",
    )
    parser.add_argument(
        "--buffer-guaranteed-fraction",
        type=float,
        required=False,
        default=0.1,
        help="Buffer guaranteed fraction for KV caching. Only used for generation.",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        required=False,
        default=4096,
        help="Chunk size in tokens for KV caching. Only used for generation.",
    )
    parser.add_argument(
        "--buffer-overflow-factor",
        type=float,
        required=False,
        default=50.0,
        help="Buffer overflow factor for KV caching. Weird parameter, smaller values leave larger unused space in the KV cache. 50.0 uses the max tokens allocated for the token limit check.",
    )
    parser.add_argument(
        "--max-tokens-to-generate",
        type=int,
        required=False,
        default=4096,
        help="Maximum number of tokens to generate. Only used for generation.",
    )
    parser.add_argument(
        "--using-pretrain-dataset",
        action="store_true",
        default=False,
        help="Tells the model the finetuning dataset is being used for pretraining (pretrain is a misnomer.)",
    )
    parser.add_argument(
        "--naive-benchmarking-only",
        action="store_true",
        default=False,
        help="Only run the naive benchmarking function. This is useful for benchmarking the performance of the model without the dynamic inference context.",
    )
    parser.add_argument(
        "--limit-predict-batches-to-n",
        type=int,
        required=False,
        default=None,
        help="Limit the number of predict batches to n. This is useful for benchmarking the performance of the model.",
    )
    return parser


def entrypoint():
    """Command-line entrypoint for MaxToki prediction."""
    parser = get_parser()
    # Parse the arguments and pull them out into local variables for ease of future refactor
    args = parser.parse_args()
    predict(
        ckpt_dir=args.initial_ckpt_path,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tokenizer_path=args.tokenizer_path,
        devices=args.num_gpus,
        work_dir=args.work_dir,
        micro_batch_size=args.micro_batch_size,
        data_path=args.data_path,
        precision=args.precision,
        seq_length=args.seq_length,
        generate_next_cell=args.generate_next_cell,
        write_interval=args.write_interval,
        max_tokens_to_generate=args.max_tokens_to_generate,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        buffer_size_gb=args.buffer_size_gb,
        buffer_guaranteed_fraction=args.buffer_guaranteed_fraction,
        chunk_size_tokens=args.chunk_size_tokens,
        buffer_overflow_factor=args.buffer_overflow_factor,
        using_pretrain_dataset=args.using_pretrain_dataset,
        naive_benchmarking_only=args.naive_benchmarking_only,
        limit_predict_batches_to_n=args.limit_predict_batches_to_n,
    )


if __name__ == "__main__":
    entrypoint()
