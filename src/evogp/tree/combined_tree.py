from typing import List, Union
import torch

from .descriptor import GenerateDescriptor


class CombinedTree:
    def __init__(self, trees, data_info):
        self.trees = trees
        self.data_info = data_info
        self.output_names = list(data_info.keys())
        self.input_names = []
        for vals in data_info.values():
            self.input_names.extend(vals)
        self.input_names = list(set(self.input_names))

        self.input_len = len(self.input_names)
        self.output_len = len(self.output_names)

        for i, names in enumerate(self.output_names):
            setattr(self, names, self.trees[i])

    @staticmethod
    def random_generate(
        descriptors: Union[List, GenerateDescriptor],
        data_info: dict,
    ):
        from .combined_forest import CombinedForest

        return CombinedForest.random_generate(
            pop_size=1,
            descriptors=descriptors,
            data_info=data_info,
        )[0]

    def forward(self, x: dict[str, torch.Tensor]):
        """Evaluate this individual on one data point or on a batch of them.

        A pop-free wrapper around the two ``CombinedForest`` methods (the
        individual is first promoted to a one-individual ``CombinedForest`` by
        :meth:`to_combined_forest`): the number ``K`` of data points described by
        ``x`` selects the method -- ``K == 1`` is a single point (delegated to
        ``CombinedForest.forward``, i.e. the same one point for the whole
        population) and ``K > 1`` is a batch (delegated to
        ``CombinedForest.batch_forward``).

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
            ``{out_name: tensor}`` with one ``(output_len_sub,)`` tensor per
            ``data_info`` key for a single point, or one ``(K, output_len_sub)``
            tensor per key for a batch (every sub-tree has ``output_len == 1``).

        Raises:
            ValueError: if ``x`` misses an input name required by ``data_info``,
                if a value has an unsupported shape (``ndim > 2``, a 2-D second
                dimension other than 1, or an empty value), or if the required
                names disagree on ``K``. There is no broadcasting: a lone
                scalar/single-point value next to ``K > 1`` names is an error.
        """
        # Lazy import: ``.combined_forest`` imports this module at module level
        from .combined_forest import normalize_inputs

        _, batch_size = normalize_inputs(self.data_info, x)

        forest = self.to_combined_forest()
        if batch_size == 1:
            pop_res = forest.forward(x)
        else:
            pop_res = forest.batch_forward(x)

        # ``to_combined_forest()`` returns a pop_size == 1 forest, and
        # ``CombinedForest.forward``/``batch_forward`` return ``{out_name: tensor}``
        # with that leading pop dimension: strip it from every output, i.e.
        # ``{out_name: (output_len,)}`` (or ``{out_name: (batch_size, output_len)}``).
        return {out_name: res[0] for out_name, res in pop_res.items()}

    def to_combined_forest(self):
        from .combined_forest import CombinedForest

        return CombinedForest(
            forests=[tree.to_forest() for tree in self.trees],
            data_info=self.data_info,
        )