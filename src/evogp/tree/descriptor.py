import warnings
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from .utils import MAX_STACK, MAX_FULL_DEPTH, dict2prob, check_tensor, FUNCS_NAMES, Func


def check_tree_length(max_tree_len, using_funcs, max_layer_cnt, layer_leaf_prob):
    max_operents_for_funcs = 0
    for func in using_funcs:
        idx = FUNCS_NAMES.index(func)
        if idx == Func.IF:
            max_operents_for_funcs = 3
        elif idx <= Func.GE:
            max_operents_for_funcs = max(max_operents_for_funcs, 2)
        else:
            max_operents_for_funcs = max(max_operents_for_funcs, 1)

    # Size for a complete-N tree with height h
    if max_operents_for_funcs > 1:
        max_tree_len_should_be = int(
            (max_operents_for_funcs**max_layer_cnt - 1) / (max_operents_for_funcs - 1)
        )
    else:
        max_tree_len_should_be = max_layer_cnt

    assert max_tree_len >= max_tree_len_should_be, (
        f"max_tree_len={max_tree_len} is too small\n"
        f"max_tree_len should >={max_tree_len_should_be}\n"
        f"as the max arity of funcs is {max_operents_for_funcs} and the max layer is {max_layer_cnt}."
    )

    non_leaf_layer_cnt = max_layer_cnt - 1
    depth2leaf_probs = torch.tensor(
        [layer_leaf_prob] * non_leaf_layer_cnt
        + [1.0] * (MAX_FULL_DEPTH - non_leaf_layer_cnt),
        device="cuda",
    )
    return depth2leaf_probs


class GenerateDescriptor:
    def __init__(
        self,
        max_tree_len: int,
        input_len: int,
        output_len: int,
        const_prob: float = 0.5,
        out_prob: float = 0.5,
        depth2leaf_probs: Optional[Tensor] = None,
        roulette_funcs: Optional[Tensor] = None,
        const_samples: Optional[Union[list, Tensor]] = None,
        using_funcs: Optional[Union[dict, list]] = None,
        max_layer_cnt: Optional[int] = None,
        layer_leaf_prob: Optional[float] = 0.2,
        const_range: Optional[Tuple[float, float]] = None,
        sample_cnt: Optional[int] = None,
    ):

        self.__params = {key: value for key, value in locals().items() if key != "self"}

        assert (
            max_tree_len <= MAX_STACK
        ), f"max_tree_len={max_tree_len} is too large, MAX_STACK={MAX_STACK}"

        assert (
            isinstance(input_len, int) and input_len > 0
        ), "input_len should be a positive integer"
        assert (
            isinstance(output_len, int) and output_len > 0
        ), "output_len should be a positive integer"
        assert (
            const_prob >= 0.0 and const_prob <= 1.0
        ), "const_prob should be in [0.0, 1.0]"

        assert out_prob >= 0.0 and out_prob <= 1.0, "out_prob should be in [0.0, 1.0]"

        if output_len > 1 and out_prob == 0.0:
            warnings.warn(
                f"output_len={output_len} > 1, but out_prob={out_prob} is 0.0."
            )

        if depth2leaf_probs is None:
            assert (
                max_layer_cnt is not None
            ), "max_layer_cnt should not be None when depth2leaf_probs is None"
            assert (
                layer_leaf_prob is not None
            ), "layer_leaf_prob should not be None when depth2leaf_probs is None"

            depth2leaf_probs = check_tree_length(
                max_tree_len, using_funcs, max_layer_cnt, layer_leaf_prob
            )

        if roulette_funcs is None:
            assert (
                using_funcs is not None
            ), "func_prob should not be None when roulette_funcs is None"
            assert isinstance(using_funcs, dict) or isinstance(
                using_funcs, list
            ), "func_prob should be a dictionary or a list"

            if isinstance(using_funcs, list):
                using_funcs = {f: 1.0 for f in using_funcs}

            func_prob = dict2prob(using_funcs)
            roulette_funcs = torch.cumsum(
                func_prob,
                dim=0,
                dtype=torch.float32,
            ).to("cuda")

            tfunc_prob = torch.zeros_like(func_prob)
            tfunc_prob[Func.TF_START : Func.BF_START] = func_prob[
                Func.TF_START : Func.BF_START
            ]
            roulette_tfuncs = torch.cumsum(
                tfunc_prob,
                dim=0,
                dtype=torch.float32,
            ).to("cuda")

            bfunc_prob = torch.zeros_like(func_prob)
            bfunc_prob[Func.BF_START : Func.UF_START] = func_prob[
                Func.BF_START : Func.UF_START
            ]
            roulette_bfuncs = torch.cumsum(
                bfunc_prob,
                dim=0,
                dtype=torch.float32,
            ).to("cuda")

            ufunc_prob = torch.zeros_like(func_prob)
            ufunc_prob[Func.UF_START : Func.END] = func_prob[Func.UF_START : Func.END]
            roulette_ufuncs = torch.cumsum(
                ufunc_prob,
                dim=0,
                dtype=torch.float32,
            ).to("cuda")

        if const_samples is None:
            assert (
                const_range is not None
            ), "const_range should not be None when const_samples is None"
            assert (
                sample_cnt is not None
            ), "sample_cnt should not be None when const_samples is None"
            const_samples = (
                torch.rand(sample_cnt, device="cuda")
                * (const_range[1] - const_range[0])
                + const_range[0]
            )

        if isinstance(const_samples, list):
            const_samples = torch.tensor(
                const_samples, dtype=torch.float32, device="cuda", requires_grad=False
            )

        check_tensor(depth2leaf_probs)
        check_tensor(roulette_funcs)
        check_tensor(const_samples)

        assert depth2leaf_probs.shape == (
            MAX_FULL_DEPTH,
        ), f"depth2leaf_probs shape should be ({MAX_FULL_DEPTH}), but got {depth2leaf_probs.shape}"

        assert roulette_funcs.shape == (
            Func.END,
        ), f"roulette_funcs shape should be ({Func.END}), but got {roulette_funcs.shape}"
        assert (
            const_samples.dim() == 1
        ), f"const_samples dim should be 1, but got {const_samples.dim()}"

        self.max_tree_len = max_tree_len
        self.input_len = input_len
        self.output_len = output_len
        self.const_prob = const_prob
        self.out_prob = out_prob
        self.depth2leaf_probs = depth2leaf_probs
        self.roulette_funcs = roulette_funcs
        self.roulette_ufuncs = roulette_ufuncs
        self.roulette_bfuncs = roulette_bfuncs
        self.roulette_tfuncs = roulette_tfuncs
        self.const_samples = const_samples

    def update(self, **kwargs):
        self.__params.update(kwargs)
        return self.__class__(**self.__params)

    def __str__(self):
        return (
            f"max_tree_len: {self.max_tree_len}\n"
            f"input_len: {self.input_len}\n"
            f"output_len: {self.output_len}\n"
            f"const_prob: {self.const_prob}\n"
            f"out_prob: {self.out_prob}\n"
            f"depth2leaf_probs: {self.depth2leaf_probs}\n"
            f"roulette_funcs: {self.roulette_funcs}\n"
            f"const_samples: {self.const_samples}\n"
        )
