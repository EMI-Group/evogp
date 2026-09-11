from typing import Optional, Tuple, Callable, List, Tuple, Union
import numbers
import time
import torch
from torch import Tensor
import numpy as np

from evogp.tree.utils import check_tensor

from .forest import Forest
from .descriptor import GenerateDescriptor
from .combined_tree import CombinedTree


# --------------------------------------------------------------------------- #
# The input-dict contract (shared by CombinedForest and CombinedTree)
# --------------------------------------------------------------------------- #


def _describe_input(value) -> str:
    """Describe an offending input value: its shape if it has one, else its type."""
    if isinstance(value, torch.Tensor):
        return f"shape {tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"a numpy array of shape {tuple(value.shape)}"
    return f"a {type(value).__name__}"


def _input_column(out_name: str, name: str, value) -> Tensor:
    """Normalize the ``x[name]`` entry to a 1-D ``(K,)`` column of ``K`` values.

    Accepted forms: a Python/NumPy scalar number or a 0-D tensor (``K == 1``), a
    1-D ``(K,)`` tensor, and a 2-D ``(K, 1)`` tensor (column form, equivalent).
    Anything else raises ``ValueError`` naming ``name``, its shape and the
    ``out_name`` sub-forest that requires it.
    """
    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
        if shape == ():
            return value.reshape(1)
        if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
            if shape[0] == 0:
                raise ValueError(
                    f"x[{name!r}] (input of {out_name!r}) is empty "
                    f"({_describe_input(value)}); a data point needs at least one value"
                )
            # ``(K,)`` and ``(K, 1)`` are the same column of K values
            return value.reshape(-1)
    elif isinstance(value, numbers.Number) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    ):
        return torch.as_tensor(value).reshape(1)

    raise ValueError(
        f"x[{name!r}] (input of {out_name!r}) must be a scalar (0-D) or a 1-D "
        f"(K,) / 2-D (K, 1) tensor, but got {_describe_input(value)}"
    )


def _stack_columns(columns: List[Tensor]) -> Tensor:
    """Stack one sub-forest's ``(K,)`` input columns into a ``(K, len(columns))`` matrix.

    The columns are placed on the first column's device, so a dict mixing plain
    numbers (which become CPU tensors) with device tensors still assembles; the
    interior ``Forest`` methods relocate the result to its own device anyway.
    """
    device = columns[0].device
    return torch.stack([column.to(device) for column in columns], dim=1)


def normalize_inputs(
    data_info: dict,
    x: dict,
    single_point: bool = False,
) -> Tuple[dict, int]:
    """Read ``x`` into per-sub-forest input columns (the containers' one shape contract).

    Shared by ``CombinedForest.forward`` / ``CombinedForest.batch_forward`` and by
    ``CombinedTree.forward`` (which uses the returned batch size to pick between
    them), so every bit of dict-shape validation lives in exactly one place.

    ``data_info`` is ``{out_name: [input_name, ...]}``, so each sub-forest consumes
    its own list of input columns, in that order. Only those names are read --
    extra ``x`` entries are ignored completely (never read, never validated).

    Every required name must describe the same number ``K`` of data points, in one
    of these forms: a Python/NumPy scalar number or a 0-D tensor (one value), a 1-D
    ``(K,)`` tensor, or a 2-D ``(K, 1)`` tensor (column form, equivalent). Values
    are never broadcast, so a single value is not silently stretched over a
    ``K > 1`` batch.

    Args:
        data_info: The ``{out_name: [input_name, ...]}`` input description.
        x: The ``{input_name: values}`` dict to read.
        single_point: When True, every required name must hold exactly one value
            (the single-point contract of ``CombinedForest.forward``) and a
            ``K > 1`` value raises instead of being accepted as a batch.

    Returns:
        ``(columns, K)``, where ``columns[out_name]`` is the list of that
        sub-forest's ``(K,)`` input columns in ``data_info`` order.

    Raises:
        ValueError: for a missing required name, an unsupported value shape
            (``ndim > 2``, a 2-D second dimension other than 1, or an empty
            value), a disagreement on ``K``, or a ``K > 1`` value in
            ``single_point`` mode. Every message names the offending input name,
            its shape and the sub-forest that requires it.
    """
    columns = {}
    batch_size = None
    for out_name, names in data_info.items():
        out_columns = []
        for name in names:
            if name not in x:
                raise ValueError(
                    f"x is missing the input {name!r} required by {out_name!r}"
                )
            value = x[name]
            column = _input_column(out_name, name, value)
            n_values = int(column.shape[0])
            if batch_size is None:
                batch_size = n_values
            elif n_values != batch_size:
                raise ValueError(
                    f"x[{name!r}] (input of {out_name!r}) holds {n_values} values "
                    f"({_describe_input(value)}), but the other input names hold "
                    f"{batch_size}; all input names must agree on the number of "
                    "data points (values are never broadcast)"
                )
            if single_point and n_values != 1:
                raise ValueError(
                    f"x[{name!r}] (input of {out_name!r}) holds {n_values} values "
                    f"({_describe_input(value)}), but forward() evaluates a single "
                    "data point for the whole population; use batch_forward() for "
                    "a batch of data points"
                )
            out_columns.append(column)
        columns[out_name] = out_columns

    return columns, batch_size


class CombinedForest:
    def __init__(
        self,
        forests: List[Forest],
        data_info: dict,
    ):
        self.data_info = data_info
        self.forests = forests
        self.output_names = list(data_info.keys())
        self.input_names = []
        for vals in data_info.values():
            self.input_names.extend(vals)
        # deduplicate while keeping the data_info order (``set()`` order is not stable
        # across processes, which made ``input_len`` look nondeterministic)
        self.input_names = list(dict.fromkeys(self.input_names))

        self.input_len = len(self.input_names)
        self.output_len = len(self.output_names)

        self.pop_size = forests[0].pop_size

    @staticmethod
    def random_generate(
        pop_size: int,
        data_info: dict,
        descriptors: Union[List, GenerateDescriptor],
    ):
        if isinstance(descriptors, GenerateDescriptor):
            descriptors = [descriptors] * len(data_info)

        assert isinstance(descriptors, list) and len(descriptors) == len(
            data_info
        ), f"there are {len(data_info)} sub_forests, but got {len(descriptors)} descriptors"

        for i, (key, vals) in enumerate(data_info.items()):
            # check input_len
            assert descriptors[i].input_len == len(vals), "input size not match"
            # check output_len
            assert descriptors[i].output_len == 1, "output size mush be 1"

        forests = [
            Forest.random_generate(pop_size=pop_size, descriptor=d) for d in descriptors
        ]

        return CombinedForest(
            forests=forests,
            data_info=data_info,
        )

    # (pop_size, input_len) -> (pop_size, output_len)
    def forward(
        self,
        x: dict[str, torch.Tensor],
    ):
        """Evaluate every individual of the population at ONE shared data point.

        The single-point (population-broadcast) counterpart of
        :meth:`batch_forward`: ``x`` describes one data point, which is replicated
        for all ``pop_size`` individuals -- individual ``i`` of every sub-forest is
        evaluated at the very same point, which is what ``Forest.forward``'s
        ``(pop_size, input_len)`` contract means.

        Args:
            x: ``{input_name: values}``. Every name that ``data_info`` requires
                must hold exactly one value, in one of these forms:

                * a Python/NumPy scalar number or a 0-D tensor,
                * a 1-D tensor of shape ``(1,)``,
                * a 2-D tensor of shape ``(1, 1)`` (column form, equivalent).

                Every other name of ``x`` is ignored completely (never read, so a
                dict carrying extra entries -- e.g. a loss target -- is accepted
                as-is).

        Returns:
            ``{out_name: tensor}``, one ``(pop_size, output_len_sub)`` tensor per
            ``data_info`` key (every sub-forest has ``output_len == 1``).

        Raises:
            ValueError: if ``x`` misses an input name required by ``data_info``,
                if a value has an unsupported shape (``ndim > 2``, a 2-D second
                dimension other than 1, or an empty value), or if a value holds
                more than one data point -- a batch belongs to
                :meth:`batch_forward`. Nothing is broadcast here: every required
                name must hold exactly one value.
        """
        columns, _ = normalize_inputs(self.data_info, x, single_point=True)

        outputs = {}
        for i, f in enumerate(self.forests):
            out_name = self.output_names[i]
            # one-point (1, input_len) matrix, replicated for every individual
            point = _stack_columns(columns[out_name])
            inputs_torch = point.expand(self.pop_size, -1).contiguous()
            outputs[out_name] = f.forward(inputs_torch)

        return outputs

    # (batch_size, input_len) -> (pop_size, batch_size, output_len)
    def batch_forward(
        self,
        x: dict[str, torch.Tensor],
    ):
        """Evaluate every individual of the population on each of ``K`` data points.

        The batch counterpart of :meth:`forward`: ``x`` describes ``K >= 1`` data
        points and every sub-forest is evaluated on all of them, which is what
        ``Forest.batch_forward``'s ``(N, input_len) -> (pop_size, N, output_len)``
        contract means. ``K == 1`` is a perfectly good batch (a single-point batch
        of size one); batches are never extrapolated from single values.

        Args:
            x: ``{input_name: values}``. Every name that ``data_info`` requires
                must describe the same number ``K`` of data points, in one of
                these forms:

                * a Python/NumPy scalar number or a 0-D tensor (``K == 1``),
                * a 1-D tensor of shape ``(K,)``,
                * a 2-D tensor of shape ``(K, 1)`` (column form, equivalent).

                Every other name of ``x`` is ignored completely (never read, so a
                dict carrying extra entries -- e.g. a loss target -- is accepted
                as-is).

        Returns:
            ``{out_name: tensor}``, one ``(pop_size, K, output_len_sub)`` tensor per
            ``data_info`` key (every sub-forest has ``output_len == 1``).

        Raises:
            ValueError: if ``x`` misses an input name required by ``data_info``,
                if a value has an unsupported shape (``ndim > 2``, a 2-D second
                dimension other than 1, or an empty value), or if the required
                names disagree on ``K``. There is no broadcasting: a lone
                scalar/single-point value next to ``K > 1`` names is an error
                (use :meth:`forward` for a single shared point).
        """
        columns, _ = normalize_inputs(self.data_info, x)

        outputs = {}
        for i, f in enumerate(self.forests):
            out_name = self.output_names[i]
            inputs_torch = _stack_columns(columns[out_name])
            outputs[out_name] = f.batch_forward(inputs_torch)

        return outputs

    def __getitem__(self, index):
        if isinstance(index, int):
            return CombinedTree(
                data_info=self.data_info,
                trees=[f[index] for f in self.forests],
            )
        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            return CombinedForest(
                data_info=self.data_info,
                forests=[f[index] for f in self.forests],
            )
        else:
            raise Exception("Do not support index type {}".format(type(index)))

    def __setitem__(self, index, value):
        if isinstance(index, int):
            assert isinstance(
                value, CombinedTree
            ), f"value should be Tree when index is int, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.trees[i]

        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            assert isinstance(
                value, CombinedForest
            ), f"value should be Forest when index is slice, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.forests[i]
        else:
            raise NotImplementedError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < self.pop_size:
            return self[self.iter_index]
        else:
            raise StopIteration

    def __len__(self):
        return self.pop_size

    def __add__(self, other):
        assert self.data_info == other.data_info
        if isinstance(other, CombinedForest):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.forests[i])
            return CombinedForest(new_forests, self.data_info)
        if isinstance(other, CombinedTree):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.tree[i])
            return CombinedForest(new_forests, self.data_info)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)
