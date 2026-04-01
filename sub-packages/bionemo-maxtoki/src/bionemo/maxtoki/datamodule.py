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


import functools
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import datasets
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bionemo.core.data.multi_epoch_dataset import MultiEpochDatasetResampler
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test", "predict"]

__all__: Sequence[str] = ("MaxTokiDataModule",)


class TruncatedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_length: int):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return self.max_length

    def __getitem__(self, index):
        item = self.dataset[index]
        return item


class MaxTokiDataModule(MegatronDataModule):
    """LightningDataModule wrapper of `SingleCellDataset`

    Args:
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa
        collator: Used to batch samples
        process_item: Function defining how each item should be processed
        num_workers (int): Number of workers to use
        num_mask_per_sample (int): Number of masked versions of a single sample to be returned by each worker
        train_batch_size (int): Batch size for training
        val_batch_size (int): Batch size for validation
        begin_position_rank_with (int): The value to begin the position rank with when using gene expression rank for position ids.
        pretrain (bool): If true- uses a collate function for pretraining, otherwise uses a special collate function for finetuning.

    Attributes:
        cfg (Config): Configuration object
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        median_dict (dict): Dictionary containing median values
        tokenizer (Tokenizer): Tokenizer object
        setup_called (bool): Flag indicating if the setup method has been called
        dataset (SingleCellDataset): Single-cell dataset object

    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        train_dataset_path: str | Path | None = None,
        val_dataset_path: str | Path | None = None,
        test_dataset_path: str | Path | None = None,
        predict_dataset_path: str | Path | None = None,
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        begin_position_rank_with: int = 0,
        pretrain: bool = False,
        limit_predict_batches_to_n: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = seq_length
        self.seq_length = seq_length
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.pretrain = pretrain
        self.begin_position_rank_with = begin_position_rank_with

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.predict_dataset_path = predict_dataset_path

        self._train_dataset_ori = (
            datasets.load_from_disk(self.train_dataset_path) if self.train_dataset_path is not None else None
        )
        self._val_dataset_ori = (
            datasets.load_from_disk(self.val_dataset_path) if self.val_dataset_path is not None else None
        )

        self._test_dataset_ori = (
            datasets.load_from_disk(self.test_dataset_path) if self.test_dataset_path is not None else None
        )
        self._predict_dataset_ori = (
            datasets.load_from_disk(self.predict_dataset_path) if self.predict_dataset_path is not None else None
        )

        if limit_predict_batches_to_n is not None and self._predict_dataset_ori is not None:
            self._predict_dataset_ori = TruncatedDataset(
                self._predict_dataset_ori, limit_predict_batches_to_n * global_batch_size
            )

        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        assert getattr(self, "trainer", None) is not None, "Please only call setup after trainer is attached."

        if self._train_dataset_ori is not None:
            assert self._val_dataset_ori is not None and self._test_dataset_ori is not None
            max_train_steps = self.trainer.max_steps
            if self.trainer.max_epochs > 1:
                logging.warning(
                    "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used in each. Instead set max_epochs to 1 and increase the number of max_steps."
                )
            assert max_train_steps > 0, "Please specify trainer.max_steps"

            num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)

            self._train_ds = MultiEpochDatasetResampler(
                self._train_dataset_ori,
                num_samples=num_train_samples,
                shuffle=True,
                seed=self.seed,
                normal_idx_getitem_behavior=True,
            )

            if self.trainer.limit_val_batches == 0:
                logging.info("Skip creating validation dataset because trainer.limit_val_batches=0.")
            else:
                num_val_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_val_batches,
                    num_samples_in_dataset=len(self._val_dataset_ori),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="val",
                )
                self._validation_ds = MultiEpochDatasetResampler(
                    self._val_dataset_ori,
                    num_samples=num_val_samples,
                    shuffle=False,
                    seed=self.seed,
                    normal_idx_getitem_behavior=True,
                )
            if self.trainer.limit_test_batches == 0:
                logging.info("Skip creating test dataset because trainer.limit_test_batches=0.")

            else:
                num_test_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_test_batches,
                    num_samples_in_dataset=len(self._test_dataset_ori),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="test",
                )
                self._test_ds = MultiEpochDatasetResampler(
                    self._test_dataset_ori,
                    num_samples=num_test_samples,
                    shuffle=False,
                    seed=self.seed,
                    normal_idx_getitem_behavior=True,
                )
        else:
            assert self._predict_dataset_ori is not None
            self._predict_ds = MultiEpochDatasetResampler(
                self._predict_dataset_ori,
                shuffle=False,
                seed=self.seed,
                normal_idx_getitem_behavior=True,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds, mode="test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._predict_ds, mode="predict", drop_last=False)

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        if self.pretrain:
            collate_fn = functools.partial(
                collate.hf_llama_padding_collate_fn,
                padding_value=self.tokenizer.pad_token_id,
                begin_rank_with=self.begin_position_rank_with,
                min_length=self.max_len,
                max_length=self.max_len,
            )
        else:
            collate_fn = functools.partial(
                self.tokenizer.collate_batch_multitask,
                padding_value=self.tokenizer.pad_token_id,
                begin_rank_with=self.begin_position_rank_with,
                min_length=self.max_len,
                max_length=self.max_len,
                keep_regression_labels_as_tokens=True,
            )
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
