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

import collections
import json
import warnings
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from nemo.lightning.io import IOMixin
from packaging import version
from torch.utils.data.sampler import RandomSampler
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    SpecialTokensMixin,
    Trainer,
)
from transformers.file_utils import is_datasets_available, is_sagemaker_dp_enabled
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from transformers.utils import is_tf_available, is_torch_available, logging, to_py_obj
from transformers.utils.generic import _is_tensorflow, _is_torch

from bionemo.llm.data.collate import padding_collate_fn


logger = logging.get_logger(__name__)
EncodedInput = List[int]
VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

if is_sagemaker_dp_enabled():
    pass
else:
    pass

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True


class ExplicitEnum(Enum):
    """Enum with more explicit error message for missing values."""

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TruncationStrategy(ExplicitEnum):
    """Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class PaddingStrategy(ExplicitEnum):
    """Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class MaxTokiPreCollator(SpecialTokensMixin, IOMixin):
    def __init__(self, token_dictionary: dict, pad_to_multiple_of: int = 8) -> None:
        super().__init__(mask_token="<mask>", pad_token="<pad>")
        self.token_dictionary = token_dictionary
        self.padding_side = "right"
        self.model_input_names = ["input_ids"]

    @property
    def vocab_size(self):
        return len(self.token_dictionary)

    def convert_ids_to_tokens(self, value):
        return self.token_dictionary.get(value)

    def __call__(self, tokens: List[str] | str) -> List[int] | int:
        return self.convert_tokens_to_ids(tokens)

    def _get_padding_truncation_strategies(
        self,
        padding=False,
        truncation=False,
        max_length=None,
        pad_to_multiple_of=None,
        verbose=True,
        **kwargs,
    ):
        """Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, "
                        "please use `truncation=True` to explicitly truncate examples to max length. "
                        "Defaulting to 'longest_first' truncation strategy. "
                        "If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy "
                        "more precisely by providing a specific strategy to `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, "
                    "use `truncation=True` to truncate examples to a max length. You can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the "
                    "maximal input size of the model (e.g. 512 for Bert). "
                    " If you have pairs of inputs, you can give a specific truncation strategy selected among "
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.
        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)
        .. note::
            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.
                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.
                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(len(v) == batch_size for v in encoded_inputs.values()), (
            "Some items in the output dictionary have a different batch size than others."
        )

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        return encoded_inputs

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of ids of the second sequence.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument."
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property

        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self.token_dictionary.get(token)

    def __len__(self):
        return len(self.token_dictionary)


class MaxTokiTokenizer(IOMixin):
    """List of ways this is used:

    padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),

    probably want a way to convert new sequences... but then again, maybe not since we dont have medians?

    """

    def __init__(self, token_dictionary: dict, pad_token: str = "<pad>") -> None:
        super().__init__()
        self.token_dictionary = token_dictionary
        self.token_ids = {v: k for k, v in self.token_dictionary.items()}
        self._build_numeric_tokens()
        self._build_special_tokens()
        self.n_reserved_entity_ids = 2  # <boq> and numeric tokens.
        self.pad_token = pad_token
        self.pad_token_id = self.token_dictionary[self.pad_token]

    def _build_numeric_tokens(self):
        self.numeric_tokens: dict[str, int] = {}
        self.numeric_token_ids: dict[int, str] = {}
        for token, _id in self.token_dictionary.items():
            try:
                int(token)
                self.numeric_tokens[token] = _id
                self.numeric_token_ids[_id] = token
            except ValueError:
                # Only looking for valid ints.
                continue

    def _build_special_tokens(self):
        self.special_tokens = {
            "<bos>": self.token_dictionary["<bos>"],
            "<eos>": self.token_dictionary["<eos>"],
            "<eoq>": self.token_dictionary["<eoq>"],
            "<boq>": self.token_dictionary["<boq>"],
            "<pad>": self.token_dictionary["<pad>"],
            "<mask>": self.token_dictionary["<mask>"],
        }
        self.bos_id = self.special_tokens["<bos>"]
        self.eos_id = self.special_tokens["<eos>"]
        self.eoq_id = self.special_tokens["<eoq>"]
        self.boq_id = self.special_tokens["<boq>"]
        self.pad_id = self.special_tokens["<pad>"]
        self.mask_id = self.special_tokens["<mask>"]

        self.special_token_ids = {v: k for k, v in self.special_tokens.items()}

    def build_numeric_mask(self, vocab_size=None):
        """Build a boolean mask of the numeric tokens.
        If vocab_size is provided, pad the mask to the vocab size. Useful when padding vocab to be divisible by 8.
        """
        numeric_mask = torch.zeros(len(self.token_dictionary), dtype=torch.bool)
        for i in range(len(numeric_mask)):
            if i in self.numeric_token_ids:
                numeric_mask[i] = True

        if vocab_size is None:
            return numeric_mask
        else:
            return self.pad_to_vocab_size(numeric_mask, vocab_size, pad_value=False)

    def build_numeric_vocab_to_numeric_map(self, vocab_size=None):
        """Build a map of the numeric tokens to their numeric values.
        If vocab_size is provided, pad the map to the vocab size. Useful when padding vocab to be divisible by 8.
        """
        vocab_to_numeric_map = torch.zeros(len(self.token_dictionary), dtype=torch.float)
        for _id, token in self.numeric_token_ids.items():
            vocab_to_numeric_map[_id] = float(token)

        if vocab_size is None:
            return vocab_to_numeric_map
        else:
            return self.pad_to_vocab_size(vocab_to_numeric_map, vocab_size, pad_value=0.0)

    @staticmethod
    def pad_to_vocab_size(tensor, vocab_size, pad_value=0):
        diff = vocab_size - tensor.size(0)
        if diff <= 0:
            return tensor
        pad_shape = (diff,)
        pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad])

    def text_to_tokens(self, text):
        return text.split()

    def tokens_to_text(self, tokens):
        return " ".join(tokens)

    def tokens_to_ids(self, tokens):
        return [self.token_dictionary.get(token) for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.token_ids.get(id) for id in ids]

    @property
    def vocab_size(self):
        return len(self.token_dictionary)

    def __call__(self, tokens):
        return self.tokens_to_ids(tokens)

    def find_boq_index(self, token_ids: List[int]) -> Optional[int]:
        """Return index of <boq> token or None if missing."""
        return 10
        return token_ids.index(self.special_tokens["<boq>"])

    # --- per-sample inspectors ---
    def find_eoq_index(self, token_ids: List[int] | torch.Tensor) -> Optional[int]:
        """Return index of <eoq> token or None if missing."""
        return find_eoq_index(token_ids, self.special_tokens["<eoq>"])

    def determine_task_type(self, token_ids: List[int], eoq_index: int) -> Literal["TimeBetweenCells", "NextCell"]:
        # eoq_index = self.find_eoq_index(token_ids)
        if eoq_index is None:
            raise ValueError("No <eoq> token found in sequence.")
        elif token_ids[eoq_index + 1] in self.numeric_token_ids:
            return "TimeBetweenCells"
        elif token_ids[eoq_index + 1] is self.special_tokens["<bos>"]:
            return "NextCell"
        else:
            raise ValueError("Invalid grammar found in sequence.")

    # TODO rename.
    def categorize_token_spans(self, token_ids: List[int]) -> Dict[str, List[Tuple[int, int]]]:
        """Find all bos/eos pairs, boq/eoq pair, and numeric tokens."""
        bos_tokens = []
        eos_tokens = []
        boq_tokens = []
        eoq_tokens = []
        numeric_tokens = []

        for i, v in enumerate(token_ids):
            if v == self.special_tokens["<bos>"]:
                bos_tokens.append(i)
            elif v == self.special_tokens["<eos>"]:
                eos_tokens.append(i)
            elif v == self.special_tokens["<boq>"]:
                boq_tokens.append(i)
            elif v == self.special_tokens["<eoq>"]:
                eoq_tokens.append(i)
            elif v in self.numeric_token_ids:
                numeric_tokens.append(i)

        if not len(bos_tokens) == len(eos_tokens):
            raise ValueError("Mismatch in number of bos/eos tokens")
        if len(bos_tokens) == 0:
            raise ValueError("No bos/eos tokens found in sequence.")
        if not len(boq_tokens) == len(eoq_tokens):
            raise ValueError("Mismatch in number of boq/eoq tokens")

        return {
            "bos_eos": list(zip(bos_tokens, eos_tokens)),
            "boq_eoq": list(zip(boq_tokens, eoq_tokens)),
            "numeric": numeric_tokens,
        }

    def create_position_ids_simple(self, token_ids: List[int], begin_rank_with: int = 0) -> List[torch.Tensor]:
        """Uniform ranks across all entries."""
        positions = torch.arange(begin_rank_with, len(token_ids) + begin_rank_with)
        return positions

    def create_position_ids_with_entity_ids(
        self, token_ids: List[int], begin_rank_with: int = 0
    ) -> List[Tuple[int, int]]:
        """Creates a vector of position ids for the sequence using the following rules:

        - reset ranks between entities.
        - unique identifiers for bos/eos by entity (e.g. one per gene expression vector)
        - constant used for timelapses
        - constant used for boq/eoqs (they can be the same)

        """
        # Hardcoding these since we dont know which direction we will go nor if this should be generic.
        eoq_reserved_id = -3
        boq_reserved_id = -2
        numeric_reserved_id = -1
        n_reserved_entity_ids = 3

        spans = self.categorize_token_spans(token_ids)

        positions = torch.zeros(len(token_ids))
        # Spans includes the query potentially.
        for i, (start, end) in enumerate(spans["bos_eos"]):
            entity_id = -1 * (i + 1 + n_reserved_entity_ids)
            positions[start] = entity_id
            positions[end] = entity_id
            positions[start + 1 : end] = torch.arange(begin_rank_with, (end - (start + 1)) + begin_rank_with)

        # Also need to find, boq, eoq, and numeric tokens.

        for start, end in spans["boq_eoq"]:
            positions[start] = boq_reserved_id
            positions[end] = eoq_reserved_id
            # Unlike bos/eos, anything can be between these.

        for i, (pos) in enumerate(spans["numeric"]):
            positions[pos] = numeric_reserved_id

        return positions

    def visualize_position_ids(self, positions: List[int], token_ids: List[int]):
        """Visualize the position ids as a compact string. RVEs are collapsed into a single token."""
        representation = []
        spans = self.categorize_token_spans(token_ids)
        # Really we should just do the start/end for RVE, then special tokens.
        for i, value in enumerate(positions):
            """ if its a special token, print the token and position id
            if its a numeric token, print the token and position id
            if its a RVE span, print the first and last token and position id
            """
            token_id = token_ids[i]

            # Check if it's a special token
            if token_id in self.special_token_ids:
                token_name = self.special_token_ids[token_id]
                representation.append(f"{token_name}({value})")
            # Check if it's a numeric token
            elif token_id in self.numeric_token_ids:
                token_value = self.numeric_token_ids[token_id]
                representation.append(f"{token_value}({value})")
            # Otherwise it's part of an RVE span
            else:
                # For RVE spans, we want to show first and last tokens with position ids
                # Check if this is the start of a new RVE span
                if i == 0 or token_ids[i - 1] in self.special_token_ids or token_ids[i - 1] in self.numeric_token_ids:
                    # This is the start of an RVE span, find the end
                    span_end = i
                    while (
                        span_end < len(token_ids) - 1
                        and token_ids[span_end + 1] not in self.special_token_ids
                        and token_ids[span_end + 1] not in self.numeric_token_ids
                    ):
                        span_end += 1

                    if span_end == i:
                        # Single token RVE span
                        representation.append(f"RVE({value})")
                    else:
                        # Multi-token RVE span, show first and last
                        representation.append(f"RVE({value}...{positions[span_end]})")

        return representation

    def create_loss_mask(
        self,
        token_ids: List[int],
        task_type: Literal["TimeBetweenCells", "NextCell"],
        eoq_index: int,
        mask_bos_next_cell: bool = True,
    ) -> List[int]:
        """TimeBetweenCells, mask everything except the last token.
        NextCell, mask everything before <eoq> and mask <bos> optionally (?).

        Args:
            token_ids: The token ids of the sequence.
            task_type: The type of task.
            eoq_index: The index of the <eoq> token, zero indexed.
            mask_bos_next_cell: Whether to mask the <bos> token in the postfix. This is only relevant for NextCell tasks.

        Some simple assertions:
            len(mask) == len(token_ids)
            mask[eoq_index] == 1
            mask[eoq_index + 1] == 0 if task_type == "TimeBetweenCells"
            mask[eoq_index + 2] == 0 if task_type == "NextCell"
            mask[eoq_index + 1] == 1 if task_type == "NextCell"

            token_ids[-1] == eos if task_type == "NextCell"
            token_ids[-1] is_numeric if task_type == "TimeBetweenCells"
        """
        if task_type == "TimeBetweenCells":
            # +1 because we want to mask eoq
            # mask = [0] * (eoq_index + 1) + [1] * (len(token_ids) - eoq_index - 1)
            # NOTE: more simple implementation.
            # the last token is the thing we want to predict, since we setup labels with auto-regression, and want to
            #   to use the logits from the prior token, we mask the last token and only keep the second to last token unmasked.
            #   Since this occurs before padding, its safe.
            mask = [0] * (len(token_ids) - 2) + [1, 0]
            return mask
        elif task_type == "NextCell":
            # +2 if we want to mask eoq and bos
            # +1 if we want to mask eoq and still predict bos

            # The last token remains masked since it doesn't actually have a label (nothing after!). In collate we actually put a pad for the label so this happens automatically as well.

            # Wait how does this mask work.
            # eoq_index is going to be 0 indexed, so + 1.

            # token after eoq is bos, so + 1 ? why do we mask bos?
            prefix_size = eoq_index + 1  # because its 0 indexed, to get to eoq, we need to add one.
            postfix_size = len(token_ids) - prefix_size
            # Mask up to and including eoq.
            prefix_mask = [0] * (prefix_size)
            # Do not mask postfix.
            postfix_mask = [1] * (postfix_size)
            # Optionally mask bos in the postfix.
            if mask_bos_next_cell:
                postfix_mask[0] = 0
            # Mask the last token.
            postfix_mask[-1] = 0
            return prefix_mask + postfix_mask
        else:
            raise ValueError("Invalid task type.")

    def sequence_as_ascii_art(self, token_ids: List[int]):
        """Return a string of the sequence as ASCII representation we care about.

        basically, where are the special tokens? (indicate the gap between them)
        Where are the numeric tokens?
        """
        representation = []
        last_i = 0
        for i, token_id in enumerate(token_ids):
            # Might be nice to also track the length of gene expression vectors.
            # Are these already padded?
            if token_id in self.special_token_ids:
                representation.append(str(self.special_token_ids[token_id]) + f"({last_i - i})")
                last_i = i
            elif token_id in self.numeric_token_ids:
                representation.append(str(self.numeric_token_ids[token_id]) + f"({last_i - i})")
                last_i = i
            else:
                if len(representation) > 0 and representation[-1] != "<RVE>":
                    representation.append("<RVE>")
                # If we are already within a RVE span, do nothing for seeing another gene token.
                # if we are not in a gene expression span, add a new RVE token.

        return representation

    def collate_batch_multitask(
        self,
        batch: List[Dict],
        padding_value: int,
        begin_rank_with: int = 0,
        min_length: int | None = None,
        max_length: int | None = None,
        use_special_token_masked_labels: bool = True,
        keep_regression_labels_as_tokens: bool = True,
    ) -> Dict:
        """Padding collate function for Llama dataloaders. Used in finetuning tasks where multiple tokens and MSE loss
        may be used.

        Args:
            batch (list): List of samples.
            padding_value (int, optional): The tokenizer's pad token ID.
            begin_rank_with (int, optional): The value to begin the rank with.
            min_length: Minimum length of the output batch; tensors will be padded to this length. If not
                provided, no extra padding beyond the max_length will be added.
            max_length: Maximum length of the sequence. If not provided, tensors will be padded to the
                longest sequence in the batch.
            use_special_token_masked_labels (bool, optional): Whether to use the magic constant, -100, for masked labels. This allows CUDA to make optimizations.
            keep_regression_labels_as_tokens (bool): When true, token id's as-is get passed through the network as labels. When false, the ids are converted to values inside the labels tensor.
        """
        new_batch = []
        for item in batch:
            if not isinstance(item, dict):
                raise ValueError(f"Item in batch is not a dictionary: {item}")
            new_item = {}
            new_item["tokens"] = torch.tensor(
                item["input_ids"], dtype=torch.long
            )  # Has two elements that are duplicates.

            eoq_index = self.find_eoq_index(item["input_ids"])
            task_type = self.determine_task_type(item["input_ids"], eoq_index)

            # TODO these should probably be done on __getitem__ to benefit from parallelism. For now we will leave it here though.
            new_item["loss_mask"] = torch.tensor(
                self.create_loss_mask(item["input_ids"], task_type, eoq_index), dtype=torch.float
            )
            new_item["position_ids"] = torch.tensor(
                self.create_position_ids_simple(item["input_ids"], begin_rank_with=begin_rank_with), dtype=torch.long
            )
            # set labels. need to think about this for the fine-tuning task since we do things a little differently.
            token_ids = new_item["tokens"]
            # These are the standard pretraining labels, used within megatron. we can choose to ignore the results in loss reduction
            labels = torch.roll(token_ids, shifts=-1, dims=0)
            # Never predict the last token since there is no 'next' token

            if use_special_token_masked_labels:
                labels[new_item["loss_mask"] == 0] = padding_value

            labels[-1] = padding_value  # use the padding value anyway for the last token.

            # NOTE that mcore/nemo/torch uses -100 here.
            new_item["labels"] = labels

            if task_type == "TimeBetweenCells" and not keep_regression_labels_as_tokens:
                # do lookup on token => value
                label = float(self.numeric_token_ids[item["input_ids"][-1]])
                labels[-2] = label

            new_batch.append(new_item)

        padding_values = {
            "tokens": padding_value,
            "labels": padding_value,
            "loss_mask": False,
            "position_ids": 0,
        }
        to_return = padding_collate_fn(
            batch=new_batch,  # type: ignore[assignment]
            padding_values=padding_values,
            min_length=min_length,
            max_length=max_length,
        )

        # NOTE: GPT_data_step filters out keys it doesnt care about
        return to_return


class GeneformerPretrainer(Trainer):
    def __init__(self, *args, **kwargs):
        data_collator = kwargs.get("data_collator", None)
        token_dictionary = kwargs.pop("token_dictionary")
        mlm = kwargs.pop("mlm", True)
        mlm_probability = kwargs.pop("mlm_probability", 0.15)

        if data_collator is None:
            precollator = MaxTokiPreCollator(token_dictionary=token_dictionary)

            # # Data Collator Functions
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=precollator, mlm=mlm, mlm_probability=mlm_probability
            )
            kwargs["data_collator"] = data_collator

        # load previously saved length vector for dataset to speed up LengthGroupedSampler
        # pre-obtained with [dataset[i]["length"] for i in range(len(dataset))]
        example_lengths_file = kwargs.pop("example_lengths_file")
        if example_lengths_file:
            with open(example_lengths_file, "r") as f:
                self.example_lengths = json.load(f)
        else:
            raise Exception(
                "example_lengths_file is required; e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048_sorted_lengths.pkl"
            )
        super().__init__(*args, **kwargs)

    # updated to not use distributed sampler since Trainer now distributes with accelerate
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = self.example_lengths
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )

        else:
            if _is_torch_generator_available:
                return RandomSampler(self.train_dataset, generator=generator)
            return RandomSampler(self.train_dataset)


def find_eoq_index(token_ids: List[int] | torch.Tensor, eoq_token: int) -> int:
    """Return index of eoq_token token or None if missing."""
    return token_ids.index(eoq_token)


def find_eoq_indices(token_ids: torch.Tensor, eoq_token: int) -> "Tensor":
    """Returns all of the positions of <eoq>, when input is a tensor."""
    _values, indices = torch.where(token_ids == eoq_token)

    # There must be exactly 1 EOQ token in each batch.
    # First expression checks that # matches is the same as batch size, the second makes sure they come from different batches.
    if indices.numel() > token_ids.shape[0] and _values.unique() != token_ids.shape[0]:
        raise ValueError("Multiple <eoq> tokens found in sequence.")
    else:
        return indices.detach()


def find_first_pad_or_last_token_index(token_ids: torch.Tensor, pad_token: int) -> "Tensor":
    """Find the first pad token or the last token in the sequence."""
    batch_size, seq_len = token_ids.shape
    pad_mask = token_ids == pad_token
    indices = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)

    # Replace non-pad positions with seq_len (larger than any valid index)
    # Keep pad positions with their actual indices
    masked_indices = torch.where(pad_mask, indices, seq_len)

    # Take the minimum along sequence dimension - gives first pad index or seq_len if no pad
    first_pad_or_end = masked_indices.min(dim=1).values

    # If result is seq_len (no pad found), use seq_len - 1 (last valid index)
    result = torch.where(first_pad_or_end < seq_len, first_pad_or_end, seq_len - 1)

    return result.detach()
