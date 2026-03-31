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

import contextlib
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, Type, TypeVar, Union

import lightning.pytorch as pl
import torch.distributed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from megatron.core import parallel_state
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import gpt_data_step, gpt_forward_step
from nemo.collections.llm.gpt.model.base import MCoreGPTModel
from nemo.lightning import io as nlio
from nemo.lightning.megatron_parallel import (
    DataT,
    MegatronLossReduction,
    ReductionT,
)
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from torch import Tensor

from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.llm.api import MegatronLossType, MegatronModelType


__all__: Sequence[str] = (
    "BionemoLightningModule",
    "LightningPassthroughPredictionMixin",
    "PassthroughLossReduction",
    "batch_collator",
    "default_megatron_optimizer",
    "get_dtype_device",
)


T = TypeVar("T")
BatchT = TypeVar("BatchT")


def some_first(seq: Iterable[Optional[T]]) -> T:
    """Returns the first non-None value from the sequence or fails"""  # noqa: D415
    for s in seq:
        if s is not None:
            return s
    raise ValueError("non-None value not found")


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:  # noqa: D103
    match torch_object:
        case []:
            raise ValueError("Looking up dtype on an empty list")
        case {**data} if not data:
            raise ValueError("Looking up dtype on an empty dict")
        case Tensor(dtype=dtype, device=device):
            return dtype, device
        case torch.nn.Module() as m:
            try:
                p = next(m.parameters())
            except StopIteration as e:
                raise ValueError("Cannot get dtype on a torch module with no parameters.") from e
            return p.dtype, p.device
        case dict(keys=_, values=values):
            val = some_first(values())
            return get_dtype_device(val)
        case list() as l:
            val = some_first(l)
            return get_dtype_device(val)
        case _:
            raise TypeError("Got something we didnt expect")


# NOTE(SKH): These types are all wrong, but are close. The inner type must always be a Tensor, but the outer container should be generic.
def batch_collator(
    batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]],
    batch_dim: int = 0,
    seq_dim: int = 1,
    batch_dim_key_defaults: dict[str, int] = {"token_logits": 1},
    seq_dim_key_defaults: dict[str, int] = {"token_logits": 0},
    preferred_gpu: int = 0,
) -> Optional[ReductionT]:
    """Takes a sequence of batches and collates them into a single batch.

        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    IMPORTANT: The underlying data primitive _must_ be a torch Tensor. The input to this function is a recurisve type,
    there can be any amount of nesting between dictionaries, tuples, and lists, as long as the inner type is a n-d Tensor.

    Examples:
        Outer container = Dict:
            [{'a': Tensor([1]), 'b': Tensor([2])}, {'a': Tensor([2]), 'b': Tensor([3])}] -> {'a': Tensor([1, 2]), 'b': Tensor([2, 3])}
        Outer container = List:
            [[Tensor([1]), Tensor([2])], [Tensor([2]), Tensor([3])]] -> [Tensor([1, 2]), Tensor([2, 3])]
        Outer container = Tuple:
            ([Tensor([1]), Tensor([2])], [Tensor([2]), Tensor([3])]) -> (Tensor([1, 2]), Tensor([2, 3]))

    Args:
        batches (Optional[Sequence[ReductionT]]): sequence of batches to collate into a single batch.
        batch_dim: If you know that the batch dim for the batch you are concatenating is not the 0th dimension (for
            example it is sequence first) then supply that dimension.
        seq_dim: If you know that the sequence dim for the batch you are concatenating is not the 1st dimension (for
            example it is sequence first) then supply that dimension. This is used for padding to the max length.
        batch_dim_key_defaults (dictionary of keys to integers): If your batch is a dictionary and you know that some
            keys have non-standard (0) batch dimensions, supply those here. By default "token_logits" has batch dim 1
            and otherwise all keys are assumed to have batch dim 0.
        seq_dim_key_defaults (dictionary of keys to integers): If your batch is a dictionary and you know that some
            keys have non-standard (1) sequence dimensions, supply those here. By default "token_logits" has seq dim 0
            and otherwise all keys are assumed to have seq dim 1.
        preferred_gpu: If any of the tensors are on any GPU, all of them will be moved to this GPU. 0 by default.

    Returns:
        A single batch of the same type as the elements of your input sequence.
    """
    match batches:
        # Handle base-cases for batch concatenation, either a list of None or a list of tensors
        case [None, *_]:
            return None
        case [Tensor(), *_]:
            # If any tensor is on a GPU, move all to preferred GPU
            if any(t.is_cuda for t in batches):
                device = torch.device(f"cuda:{preferred_gpu}")
                batches = [t.to(device) for t in batches]
            # First shortcut if all tensors are 1D (they have at least one batch dim, and it must be at 0)
            if len(batches) > 0 and isinstance(batches[0], Tensor) and batches[0].ndim == 1:
                return torch.cat(batches, dim=0)
            # Find max sequence length across all tensors
            max_seq_len = max(batch.size(seq_dim) for batch in batches)
            # Pad each tensor to max length along seq_dim
            padded_batches = []
            for batch in batches:
                # Initialize padding tuple - needs 2 values per dim, starting from last dim
                # e.g. for 3D tensor: [left_pad_dim2, right_pad_dim2, left_pad_dim1, right_pad_dim1, left_pad_dim0, right_pad_dim0]
                pad_size = [0] * (2 * batch.ndim)
                # Calculate padding needed at end of sequence dimension
                pad_amount = max_seq_len - batch.size(seq_dim)
                # Pad end of sequence dimension by putting padding amount in correct position
                # For seq_dim=1 in 3D tensor: [0, 0, 0, pad_amount, 0, 0]
                pad_size[2 * (batch.ndim - 1 - seq_dim) + 1] = pad_amount
                padded_batch = torch.nn.functional.pad(batch, tuple(pad_size))
                padded_batches.append(padded_batch)
            padded_batch = torch.cat(padded_batches, dim=batch_dim)
            assert padded_batch.size(seq_dim) == max_seq_len
            return padded_batch
        # Next 3 calls are the recursive calls into the sub-structures of the batch. We handle dictionaries, tuples, and lists
        case [dict(), *_]:
            return {
                key: batch_collator(
                    [batch[key] for batch in batches],
                    batch_dim=batch_dim_key_defaults.get(key, batch_dim),
                    seq_dim=seq_dim_key_defaults.get(key, seq_dim),
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for key in batches[0]
            }
        case [tuple(), *_]:
            return tuple(
                batch_collator(
                    [batch[i] for batch in batches],
                    batch_dim=batch_dim,
                    seq_dim=seq_dim,
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for i in range(len(batches[0]))
            )
        case [list(), *_]:
            return [
                batch_collator(
                    [batch[i] for batch in batches],
                    batch_dim=batch_dim,
                    seq_dim=seq_dim,
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for i in range(len(batches[0]))
            ]
        # Final cases shouldn't happen, an empty sequence (no batches), or "other".
        case []:
            raise ValueError("Cannot process an empty sequence")
        case _:
            raise ValueError("Unsupported input structure in batch_collator")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
# TODO(@skothenhill): Re-think the generics here- the way that `batch_collator` is expressed, `batches` should be a recursive generic type.
class PassthroughLossReduction(MegatronLossReduction, Generic[DataT]):
    """A workaround for nemo/megatron to perform inference.

    Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is
    expected to return a loss. This class hijacks that mechanism to instead pass through the forward output unperturbed
    as the loss (to enable inference in the predict step), and then the reduce method is used to collate the batch of
    forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple, or list of
    tensors. The inner type _must always be a Tensor_.
    """

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[Tensor, DataT]:
        """Passes through the `forward_out` value as the 2nd tuple element.

        Args:
            batch: The batch of data that was passed through the model to generate output. NOTE: this value is ignored.
            forward_out: The output from your model's forward pass.

        Returns:
            A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """
        return torch.zeros((1, 1)), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        """Collates list of model's outputs into a single output."""
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """A mixin that allows your model to do inference on the predict step by hijacking nemo's loss reduction mechanism."""

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        """For the predict step, pass through the forward pass output."""
        return PassthroughLossReduction()


ForwardStep = Callable[[MegatronModelType, DataT], DataT]
"""Megatron-compatible forward pass function.
"""

DataStep = Callable[[Iterator[DataT]], DataT]
"""Batches together an iterator of individual examples.

Necessary for compatability with Megatron. This function type is similiar to the collate function of PyTorch.

A `DataStep` function takes an iterator over individual examples. Each example may be a tensor, sequence of tensors,
or a set of named tensors (provided as a `dict` mapping `str` names to each `Tensor`). Each iteration must
yield the same type.

The output of this function will mirror the same structure of each yielded example. It will be a concatenation of all
of the examples in the iterator.
"""


class BionemoLightningModule(
    Generic[MegatronModelType, MegatronLossType],
    pl.LightningModule,
    nlio.IOMixin,
    nlio.ConnectorMixin,
    # LightningPassthroughPredictionMixin,
):
    """Reusable PyTorch Lightning module for Megatron models that is compatible with NeMo's conventions."""

    def __init__(
        self,
        config: BionemoTrainableModelConfig[MegatronModelType, MegatronLossType],
        forward_step: ForwardStep,
        data_step: DataStep,
        optimizer: MegatronOptimizerModule,
        predict_step: Optional[Callable[[DataT], DataT]] = None,
        model_transform: Optional[Callable[[MegatronModelType], MegatronModelType]] = None,
        configure_init_model_parallel: bool = False,
        model_context_managers: Optional[list] = [],
        freeze_params_until_key_suffix: Optional[str] = None,
        tokenizer = None,
        predict_loss_reduction_class: Type[MegatronLossType] = PassthroughLossReduction,
        **model_construct_args,
    ) -> None:
        """Constructor.

        Args:
            config: Serializable configuration object that allows one to construct a new model instance and loss
                function. Necessary for Megatron-based training as the model itself cannot be serialized and
                distributed to nodes. Instead, we serialize the procedure for making the model and distribute that.
            forward_step: Performs forward pass using the model and a batch of data.
            data_step: Custom batch-creating function for the model.
            optimizer: Megatron-compatible distributed optimizer instance. Defaults to using ADAM with a 1e-4 learning
                rate.
            model_construct_args: Optional. Any arguments necessary to construct the model in the `config`'s
                `configure_model` method.
            model_transform: Optional. The model transform function.
            freeze_params_until_key_suffix: Optional. Will disable grads for all parameters in the model parameter list until the key suffix is reached.
                For example, if the key suffix is "decoder.layers", then all parameters in the model parameter list will be frozen until a parameter name ends with "decoder.layers".
                Often parameters will have various levels of module nesting, such as "module.module.module.decoder.layers.0.weight", using the suffix allows for a more general pattern.
            configure_init_model_parallel: Optional. Whether to initialize the model parallel at configuration time.
            **model_construct_args: Optional. Arguments necessary for the supplied model configuration's
                `configure_model` method, which will make an instance of the model.
        """
        super().__init__()
        self.config = config
        self.module_construct_args: Optional[dict[str, Any]] = model_construct_args
        # ***must** be set up in configure_model() -- megatron constraint
        # also, must be called `module`: nemo expects the actual model to be stored this way
        self.module: Optional[MegatronModelType] = None
        self.loss_reduction_class: type[MegatronLossType] = config.get_loss_reduction_class()
        self.optim = optimizer
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self._data_step = data_step
        self._forward_step = forward_step
        self._predict_step = predict_step if predict_step is not None else self._default_predict_step
        self.model_transform = model_transform
        self.configure_init_model_parallel = configure_init_model_parallel
        self.model_context_managers = model_context_managers
        self.freeze_params_until_key_suffix = freeze_params_until_key_suffix

        self.predict_loss_reduction_class = predict_loss_reduction_class
        # TODO(SKH): Fix this- Used for vocab size in some models.. (GPT), shouldnt really need the whole tokenizer.
        self.tokenizer = tokenizer
        self.train_metric = self._configure_metrics(self.config.train_metric, "train") if self.config.train_metric else None
        self.valid_metric = self._configure_metrics(self.config.valid_metric, "valid") if self.config.valid_metric else None
    
    def _configure_metrics(self, metric_config, prefix):
        # TODO(SKH): Only allow lists.
        if metric_config is None:
            return None
        if isinstance(metric_config, list):
            metrics = []
            for i, m_config in enumerate(metric_config):
                metric = m_config.get_instance()
                setattr(self, f"{prefix}_metric_{i}", metric)
                metrics.append(metric)
            return metrics
        else:
            return metric_config.get_instance()
    
    def _move_metrics_to_device(self):
        if self.module is None:
            return
        
        try:
            device = next(self.module.parameters()).device
        except StopIteration:
            return
        
        # TODO(SKH): Only allow lists.
        if self.train_metric is not None:
            if isinstance(self.train_metric, list):
                for metric in self.train_metric:
                    metric.to(device)
            else:
                self.train_metric.to(device)
        
        if self.valid_metric is not None:
            if isinstance(self.valid_metric, list):
                for metric in self.valid_metric:
                    metric.to(device)
            else:
                self.valid_metric.to(device)

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        # TODO(SKH): reconcile with implementation in main
        from nemo.collections.llm.modelopt.model_utils import restore_modelopt_state

        if self.configure_init_model_parallel:
            self.trainer.strategy._init_model_parallel = True

        if self.module is None:
            with contextlib.ExitStack() as stack:
                # Apply requested context managers for this block
                for cm in self.model_context_managers:
                    stack.enter_context(cm)

                self.module = self.config.configure_model(self.tokenizer, vp_stage=vp_stage)

            # Restore ModelOpt state if it exists.
            # NOTE: Also called in MegatronStrategy.load_checkpoint but we do it for GPTModel here first,
            # for transformations which add new parameters to the model that need to be included in the optimizer.
            # TODO: Add to other models when needed.
            restore_modelopt_state(self.module, trainer=self._trainer)

            self._move_metrics_to_device()


    def is_on_logging_device(self):
        """Return True if last stage of pipeline parallel and first tensor parallel rank."""
        return parallel_state.is_pipeline_last_stage() and parallel_state.get_tensor_model_parallel_rank() == 0

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the GPT model.

        Args:
            *args: Positional arguments to pass to the model's forward pass
            **kwargs: Keyword arguments to pass to the model's forward pass
        Returns:
            torch.Tensor: Output tensor from the model
        """
        self.configure_model()
        assert self.module is not None, "ERROR: configure_model() method has been incorrectly overridden!"
        prediction = self.module(*args, **kwargs)  # for now just pass through to the underlying model
        return prediction



    def data_step(self, dataloader_iter: Iterator[DataT]) -> DataT:  # noqa: D102
        return self._data_step(dataloader_iter)

    def forward_step(self, batch, kwargs: Optional[Dict[str, Any]] = None) -> Tensor:
        """Megatron-required: the training forward step for the model, which is required to produce the loss.

        Normally, the forward pass of a model means its inference. Loss is computed using the predictions
        from the forward pass against labels. Megatron unfortunately conflates these two different concepts
        and instead has models "forward" method produce the loss. See the Megatron docs for details:
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L170

        To get actual predictions, use the :func:`forward` method instead.
        """
        # safe to do because configure_model is idempotent
        self.configure_model()
        assert self.module is not None
        if kwargs is None:
            kwargs = {}

        return self._forward_step(self.module, batch, **kwargs)

    def update_metric(
        self, batch, outputs, metric, task: Literal["pretraining", "classification", "regression", "multitask"]
    ) -> None:
        """Update metric for logging."""
        match task:
            case "pretraining":
                logits = outputs["token_logits"].detach().transpose(0, 1)  #  [s, b, v] -> [b, s, v]
                metric(logits, batch["labels"])
            case "classification":
                classification_output = outputs["classification_output"]
                num_classes = classification_output.shape[-1]
                labels = batch["labels"]
                if classification_output.ndim == 3:  # token-level classification
                    classification_output = classification_output.reshape(-1, num_classes)[
                        batch["loss_mask"].view(-1)
                    ]  # shape [-1, num_classes]
                    assert classification_output.ndim == 2

                    labels = batch["labels"].reshape(-1)[batch["loss_mask"].view(-1)]
                metric(
                    classification_output.reshape(-1, num_classes),
                    labels.reshape(-1),
                )
            case "regression":
                regression_output = outputs["regression_output"]
                metric(regression_output, batch["labels"])
            case "multitask":
                metric_type = metric.__class__.__name__
                if metric_type == "UnreducedMSELoss":
                    # NOTE: this is 'unpenalized loss.' Meaning the ratio of tokens weighting is not applied - ONLY the mse loss over the labels.
                    loss_mask = batch["loss_mask"]
                    mse_loss = outputs["mse_loss"]
                    
                    # Create task mask: True for MSE tasks (TimeBetweenCells)
                    # MSE tasks have only one unmasked token per sequence
                    task_mask = torch.tensor(
                        [[mask_seq.sum() <= 1] for mask_seq in loss_mask],
                        dtype=loss_mask.dtype,
                        device=loss_mask.device
                    )
                    mse_task_mask = (loss_mask * task_mask).to(mse_loss.device)
                    metric.update(mse_loss, mse_task_mask)
                elif metric_type == "UnreducedCELoss":
                    loss_mask = batch["loss_mask"]
                    ce_loss = outputs["lm_outputs"]
                    task_mask = torch.tensor(
                        [[mask_seq.sum() > 1] for mask_seq in loss_mask],
                        dtype=loss_mask.dtype,
                        device=loss_mask.device
                    )
                    ce_task_mask = (loss_mask * task_mask).to(ce_loss.device)
                    metric.update(ce_loss, ce_task_mask)
                elif metric_type == "RegressionPreds":
                    regression_preds = outputs["regression_output"]
                    metric.update(regression_preds, batch["loss_mask"])
                elif metric_type == "RegressionAvgPred":
                    regression_preds = outputs["regression_preds"]
                    loss_mask = batch["loss_mask"]
                    task_mask = torch.tensor(
                        [[mask_seq.sum() <= 1] for mask_seq in loss_mask],
                        dtype=loss_mask.dtype,
                        device=loss_mask.device
                    )
                    mse_task_mask = (loss_mask * task_mask).to(regression_preds.device)
                    metric.update(regression_preds, mse_task_mask)
                else:
                    raise ValueError(f"Unsupported metric type for multitask: {metric_type}")
            case _:
                raise NotImplementedError(f"unrecognized task {task}")

    def training_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        outputs = self.forward_step(batch)
        
        if self.train_metric is not None and self.is_on_logging_device():
            if isinstance(self.train_metric, list):
                for metric, metric_config in zip(self.train_metric, self.config.train_metric):
                    self.update_metric(batch, outputs, metric, metric_config.task)
            else:
                self.update_metric(batch, outputs, self.train_metric, self.config.train_metric.task)

        return outputs

    def on_train_start(self):
        super().on_train_start()
        # Optionally handle parameter freezing if specified.
        params = dict(self.module.named_parameters())

        if self.freeze_params_until_key_suffix is None:
            return

        for name, param in params.items():
            # Might need to be smarter about the pattern.
            if name.endswith(self.freeze_params_until_key_suffix):
                break
            else:
                param.requires_grad = False

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.train_metric is not None:
            # Handle both single metrics and lists of metrics
            # TODO(SKH): Only allow lists.
            if isinstance(self.train_metric, list):
                for metric, metric_config in zip(self.train_metric, self.config.train_metric):
                    computed_value = metric.compute()
                    self.log(
                        metric_config.metric_name,
                        computed_value,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        batch_size=1,
                    )
                    metric.reset()
            else:
                computed_value = self.train_metric.compute()
                self.log(
                    self.config.train_metric.metric_name,
                    computed_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=1,
                )
                self.train_metric.reset()

    def validation_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        outputs = self.forward_step(batch)
        
        if self.valid_metric is not None and self.is_on_logging_device():
            if isinstance(self.valid_metric, list):
                for metric, metric_config in zip(self.valid_metric, self.config.valid_metric):
                    self.update_metric(batch, outputs, metric, metric_config.task)
            else:
                self.update_metric(batch, outputs, self.valid_metric, self.config.valid_metric.task)

        return outputs

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        return self._predict_step(self, batch)

    def _default_predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor | None:
        """Alias for forward_step."""
        if len(batch) == 0:
            return
        
        ret = self.forward_step(batch)
        return ret

    def training_loss_reduction(self) -> MegatronLossType:
        """This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss."""
        return self.loss_reduction_class()

    def validation_loss_reduction(self) -> MegatronLossType:  # noqa: D102
        return self.loss_reduction_class(validation_step=True)

    def test_loss_reduction(self) -> MegatronLossType:  # noqa: D102
        return self.loss_reduction_class(validation_step=True)

    def predict_loss_reduction(self) -> MegatronLossType:
        return self.predict_loss_reduction_class()

    def on_validation_epoch_end(self):
        if self.valid_metric is None:
            return

        if self.trainer.sanity_checking:
            if isinstance(self.valid_metric, list):
                for metric in self.valid_metric:
                    metric.reset()
            else:
                self.valid_metric.reset()
            return

        if isinstance(self.valid_metric, list):
            for metric, metric_config in zip(self.valid_metric, self.config.valid_metric):
                self.log(
                    metric_config.metric_name,
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                metric.reset()
        else:
            self.log(
                self.config.valid_metric.metric_name,
                self.valid_metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.valid_metric.reset()


def default_megatron_optimizer() -> MegatronOptimizerModule:
    """Default distributed optimizer uses Adam with a 1e-4 learning rate."""
    return MegatronOptimizerModule(
        config=OptimizerConfig(lr=1e-4, optimizer="adam", use_distributed_optimizer=True),
    )
