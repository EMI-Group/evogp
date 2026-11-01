"""Frozen, torch-free configuration objects of the evoxc GP stack.

Everything here follows the Evox 2.0 idiom this subtree copies (see
``CONTEXT.md``/``DESIGN.md``): a config holds **static leaves only** -- Python
scalars and tuples -- and is never mutated and never traced.  The traced graphs
in :mod:`evogp.gp.encoding` close over a config, so a config value is baked into
the compiled program; changing one produces a different graph.

Two objects:

``GPTreeConfig``
    The descriptor: the tree-buffer width (``max_nodes``), the function set and
    its weights, the depth schedule, the constant/leaf probabilities and the
    constant table.  It derives -- once, in ``__post_init__`` -- every static
    table a traced graph needs, as plain tuples: the function cdf
    (``roulette_funcs``), the depth -> leaf-probability schedule
    (``depth2leaf_probs``), the per-id arity (``func_arity``) and node type
    (``func_type``), the resolved constant samples and the padded-layout shape
    (``max_arity``, ``levels``).  The rules are the deleted
    ``GenerateDescriptor``'s (``descriptor.py``) -- same ids, same cdf
    construction, same complete-tree feasibility check, same depth schedule --
    but expressed without torch and raising ``ValueError`` instead of an
    ``assert``.

``GPConfig``
    Population size and operator rates.  Deliberately minimal: only the fields
    the operator layer (``operators.py``, to come) will consume.

The ids (``MAX_STACK``/``MAX_FULL_DEPTH``/``NType``/``Func``/``FUNCS_NAMES``)
come from :mod:`evogp._tables`, the torch-free single source of truth, through
their re-export in :mod:`evogp.gp._constants` -- this module never duplicates
them and never imports torch.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple, Union

import numpy as np

from ._constants import (
    FUNC_ARITY,
    FUNC_IDS,
    FUNC_TYPE,
    FUNCS_NAMES,
    MAX_FULL_DEPTH,
    MAX_STACK,
    Func,
)

__all__ = ["GPTreeConfig", "GPConfig", "SELECTION_MODES", "MUTATION_KINDS"]

#: ``using_funcs`` in either accepted spelling.
UsingFuncs = Union[Tuple[str, ...], Tuple[Tuple[str, float], ...]]

#: The selection modes :mod:`evogp.gp.selection` implements (the canonical list
#: lives here so a config can validate itself without importing the operator).
SELECTION_MODES = ("tournament", "rank", "roulette")

#: The mutation operators :mod:`evogp.gp.mutation` implements.
MUTATION_KINDS = ("point", "subtree", "hoist")


def _normalize_funcs(using_funcs) -> Tuple[Tuple[str, float], ...]:
    """``(name, weight)`` pairs, validated and order-preserving.

    Accepts the two spellings the deleted ``GenerateDescriptor`` accepted: a
    sequence of names (all weight 1.0) or a mapping / sequence of
    ``(name, weight)`` pairs.  Duplicates are rejected instead of silently
    merged, because the weight of a duplicated name would otherwise depend on
    dict ordering.
    """
    if isinstance(using_funcs, dict):
        pairs = list(using_funcs.items())
    else:
        try:
            items = list(using_funcs)
        except TypeError as exc:
            raise ValueError(
                "using_funcs must be a sequence of function names or of "
                f"(name, weight) pairs, got {type(using_funcs).__name__}"
            ) from exc
        if not items:
            raise ValueError("using_funcs must not be empty")
        pairs = []
        for item in items:
            if isinstance(item, str):
                pairs.append((item, 1.0))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError(
                    "using_funcs entries must be a function name or a "
                    f"(name, weight) pair, got {item!r}"
                )

    if not pairs:
        raise ValueError("using_funcs must not be empty")

    normalized = []
    seen = set()
    for name, weight in pairs:
        if name not in FUNC_IDS:
            raise ValueError(
                f"unknown function name {name!r}; the function set is {FUNCS_NAMES}"
            )
        if name in seen:
            raise ValueError(f"duplicate function {name!r} in using_funcs")
        seen.add(name)
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"the weight of {name!r} must be a number, got {weight!r}"
            ) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"the weight of {name!r} must be finite and >= 0.0, got {weight}"
            )
        normalized.append((name, weight))
    if sum(w for _, w in normalized) <= 0.0:
        raise ValueError(
            "the function weights of using_funcs must not sum to 0.0 "
            f"(got {[w for _, w in normalized]})"
        )
    return tuple(normalized)


def _roulette(pairs: Tuple[Tuple[str, float], ...]) -> Tuple[float, ...]:
    """The function cdf of the deleted ``GenerateDescriptor``: ``cumsum`` of
    the weights.

    Identical construction (``dict2prob`` + ``torch.cumsum``): weights are
    normalized over the *whole* function set (a slot that is not used keeps
    probability 0), then summed in id order.  The last entry is 1.0 up to
    float rounding, which is what makes ``searchsorted(cdf, r, right=True)``
    -- reproduced in the traced graph as ``sum(cdf <= r)`` -- land on a used
    function for every ``r < 1``.
    """
    weights = np.zeros(Func.END, dtype=np.float64)
    for name, weight in pairs:
        weights[FUNC_IDS[name]] = weight
    weights = weights / weights.sum()
    return tuple(float(v) for v in np.cumsum(weights))


def _depth_schedule(max_layer_cnt: int, layer_leaf_prob: float) -> Tuple[float, ...]:
    """The deleted ``GenerateDescriptor.check_tree_length``'s depth ->
    leaf-probability schedule."""
    non_leaf_layer_cnt = max_layer_cnt - 1
    return tuple(
        [float(layer_leaf_prob)] * non_leaf_layer_cnt
        + [1.0] * (MAX_FULL_DEPTH - non_leaf_layer_cnt)
    )


def _max_arity(pairs: Tuple[Tuple[str, float], ...]) -> int:
    """Largest arity any function of the set can produce (bounds the layout)."""
    return max(FUNC_ARITY[FUNC_IDS[name]] for name, _ in pairs)


def _levels(depth2leaf_probs: Tuple[float, ...]) -> int:
    """Number of depth levels a generated tree can reach.

    Mirrors the deleted ``generation.py::_descriptor_plan``: the first depth
    whose leaf probability is ``1.0`` ends the schedule (every node there is a
    leaf), so that depth is the last populated level.
    """
    for depth, prob in enumerate(depth2leaf_probs):
        if prob >= 1.0:
            return depth + 1
    return len(depth2leaf_probs) + 1


def _complete_tree_size(arity: int, levels: int) -> int:
    """Node count of a complete ``arity``-ary tree with ``levels`` levels."""
    if arity > 1:
        return (arity**levels - 1) // (arity - 1)
    return levels


@dataclasses.dataclass(frozen=True)
class GPTreeConfig:
    """The static description of one tree population (torch-free).

    Args:
        max_nodes: Row width of the encoding channels -- the tree layer's
            ``max_tree_len``.  Must be ``<= MAX_STACK`` and large enough for a
            complete ``max_arity``-ary tree of ``max_layer_cnt`` levels.
        input_len: Number of input variables (``VAR`` payloads are ``% input_len``).
        using_funcs: Function names or ``(name, weight)`` pairs, positional with
            the ids of :class:`evogp.gp._constants.Func`.
        output_len: Number of outputs.  Multi-output is not implemented yet
            (``out_prob``/``OUT_NODE`` tagging); only ``1`` is accepted by
            :func:`evogp.gp.encoding.init_population`.
        max_layer_cnt: Depth of the schedule (levels a tree can reach).
        layer_leaf_prob: Leaf probability of every non-terminal depth.
        const_prob: Probability that a leaf is a constant (else a variable).
        out_prob: Probability that a function node carries an output tag
            (multi-output; stored for later, unused while ``output_len == 1``).
        const_samples: The constant table.  ``None`` draws ``sample_cnt`` values
            uniformly from ``const_range`` with ``np.random.default_rng(const_seed)``,
            so the config stays a pure function of its fields.
        const_range: ``(low, high)`` of the constant draw.
        sample_cnt: Number of constants to draw.
        const_seed: Seed of the constant draw (only used when ``const_samples``
            is ``None``).

    Attributes:
        roulette_funcs: The function cdf, length ``Func.END``.
        func_arity: Arity per function id, length ``Func.END``.
        func_type: ``NType`` marker per function id, length ``Func.END``.
        depth2leaf_probs: The depth schedule, length ``MAX_FULL_DEPTH``.
        max_arity: Largest arity of the function set (padded-layout width).
        levels: Number of populated depth levels.
    """

    max_nodes: int
    input_len: int
    using_funcs: UsingFuncs
    output_len: int = 1
    max_layer_cnt: int = 4
    layer_leaf_prob: float = 0.2
    const_prob: float = 0.5
    out_prob: float = 0.5
    const_samples: Optional[Tuple[float, ...]] = None
    const_range: Optional[Tuple[float, float]] = None
    sample_cnt: Optional[int] = None
    const_seed: int = 0

    # ---- derived, static, never passed in ---------------------------------
    roulette_funcs: Tuple[float, ...] = dataclasses.field(init=False, repr=False)
    func_arity: Tuple[int, ...] = dataclasses.field(init=False, repr=False)
    func_type: Tuple[int, ...] = dataclasses.field(init=False, repr=False)
    depth2leaf_probs: Tuple[float, ...] = dataclasses.field(init=False, repr=False)
    max_arity: int = dataclasses.field(init=False, repr=False)
    levels: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__

        if not isinstance(self.max_nodes, int) or isinstance(self.max_nodes, bool):
            raise ValueError(f"max_nodes must be an int, got {self.max_nodes!r}")
        if not 0 < self.max_nodes <= MAX_STACK:
            raise ValueError(
                f"max_nodes={self.max_nodes} must be in (0, {MAX_STACK}] "
                f"(MAX_STACK={MAX_STACK})"
            )
        if not isinstance(self.input_len, int) or self.input_len <= 0:
            raise ValueError(
                f"input_len must be a positive integer, got {self.input_len!r}"
            )
        if not isinstance(self.output_len, int) or self.output_len <= 0:
            raise ValueError(
                f"output_len must be a positive integer, got {self.output_len!r}"
            )
        if not 0.0 <= float(self.const_prob) <= 1.0:
            raise ValueError(
                f"const_prob must be in [0.0, 1.0], got {self.const_prob}"
            )
        if not 0.0 <= float(self.out_prob) <= 1.0:
            raise ValueError(f"out_prob must be in [0.0, 1.0], got {self.out_prob}")
        if not 0.0 <= float(self.layer_leaf_prob) <= 1.0:
            raise ValueError(
                f"layer_leaf_prob must be in [0.0, 1.0], got {self.layer_leaf_prob}"
            )
        if not isinstance(self.max_layer_cnt, int) or isinstance(
            self.max_layer_cnt, bool
        ):
            raise ValueError(
                f"max_layer_cnt must be an int, got {self.max_layer_cnt!r}"
            )
        if not 1 <= self.max_layer_cnt <= MAX_FULL_DEPTH + 1:
            raise ValueError(
                f"max_layer_cnt={self.max_layer_cnt} must be in [1, "
                f"{MAX_FULL_DEPTH + 1}] (the depth schedule has "
                f"MAX_FULL_DEPTH={MAX_FULL_DEPTH} entries)"
            )

        pairs = _normalize_funcs(self.using_funcs)
        set_(self, "using_funcs", pairs)
        max_arity = _max_arity(pairs)
        set_(self, "max_arity", max_arity)

        # The deleted GenerateDescriptor.check_tree_length: a complete tree of
        # the deepest level schedule must fit the row width, otherwise
        # generation can legitimately overflow max_nodes.
        needed = _complete_tree_size(max_arity, self.max_layer_cnt)
        if self.max_nodes < needed:
            raise ValueError(
                f"max_nodes={self.max_nodes} is too small: the function set's "
                f"max arity is {max_arity} and max_layer_cnt is "
                f"{self.max_layer_cnt}, so a complete tree needs {needed} nodes."
            )

        set_(self, "roulette_funcs", _roulette(pairs))
        set_(self, "func_arity", tuple(FUNC_ARITY))
        set_(self, "func_type", tuple(FUNC_TYPE))
        schedule = _depth_schedule(self.max_layer_cnt, float(self.layer_leaf_prob))
        set_(self, "depth2leaf_probs", schedule)
        set_(self, "levels", _levels(schedule))
        set_(self, "const_samples", self._resolve_const_samples())

    def _resolve_const_samples(self) -> Tuple[float, ...]:
        """The constant table as a tuple, drawing it when it was not given."""
        samples = self.const_samples
        if samples is not None:
            values = tuple(float(v) for v in samples)
            if not values:
                raise ValueError("const_samples must not be empty")
            return values
        if self.const_range is None or self.sample_cnt is None:
            raise ValueError(
                "const_samples is None: either pass const_samples or both "
                "const_range and sample_cnt"
            )
        low, high = (float(v) for v in self.const_range)
        if self.sample_cnt <= 0:
            raise ValueError(f"sample_cnt must be positive, got {self.sample_cnt}")
        rng = np.random.default_rng(self.const_seed)
        return tuple(float(v) for v in rng.uniform(low, high, int(self.sample_cnt)))

    # ------------------------------------------------------------------ misc
    @property
    def max_tree_len(self) -> int:
        """The tree layer's name for :attr:`max_nodes`."""
        return self.max_nodes

    def padded_widths(self) -> Tuple[int, ...]:
        """Slots per depth of the padded complete-tree layout."""
        return tuple(self.max_arity**depth for depth in range(self.levels))

    def padded_total(self) -> int:
        """Total padded slots of one row while the tree grows."""
        return sum(self.padded_widths())

    def depth2leaf_prob_at(self, depth: int) -> float:
        """Leaf probability of ``depth`` (``1.0`` past the schedule)."""
        if depth < len(self.depth2leaf_probs):
            return self.depth2leaf_probs[depth]
        return 1.0

    def replace(self, **changes) -> "GPTreeConfig":
        """A new config with ``changes`` applied (the receiver is unchanged)."""
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass(frozen=True)
class GPConfig:
    """Population size and operator rates (static leaves only).

    Args:
        pop_size: Number of trees per generation.
        elite_count: Number of best individuals copied unchanged into the next
            generation.
        tournament_size: Number of candidates drawn by tournament selection.
        crossover_prob: Per-individual probability of being recombined.
        mutation_prob: Per-individual probability of being mutated.
        selection: The selection mode, one of :data:`SELECTION_MODES` --
            ``"tournament"`` (default, the pressure knob is
            ``tournament_size``), ``"rank"`` (linear rank distribution, the
            pressure knob is ``selection_pressure``) or ``"roulette"``
            (fitness-proportional).
        selection_pressure: Rank selection's ``sp`` in ``[0, 1]`` (``0`` = no
            pressure, ``1`` = the best individual is twice as likely as the
            worst).  Unused by the other modes.
        mutation_kind: The mutation operator, one of :data:`MUTATION_KINDS` --
            ``"subtree"`` (default: replace one subtree with a fresh random
            tree), ``"point"`` (one same-arity node edit) or ``"hoist"``
            (shrink to one subtree).
    """

    pop_size: int
    elite_count: int = 1
    tournament_size: int = 2
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    selection: str = "tournament"
    selection_pressure: float = 0.5
    mutation_kind: str = "subtree"

    def __post_init__(self) -> None:
        if not isinstance(self.pop_size, int) or isinstance(self.pop_size, bool):
            raise ValueError(f"pop_size must be an int, got {self.pop_size!r}")
        if self.pop_size <= 0:
            raise ValueError(f"pop_size must be positive, got {self.pop_size}")
        if not isinstance(self.elite_count, int) or isinstance(
            self.elite_count, bool
        ):
            raise ValueError(
                f"elite_count must be an int, got {self.elite_count!r}"
            )
        if not 0 <= self.elite_count < self.pop_size:
            raise ValueError(
                f"elite_count={self.elite_count} must be in [0, pop_size="
                f"{self.pop_size})"
            )
        if not isinstance(self.tournament_size, int) or self.tournament_size < 1:
            raise ValueError(
                f"tournament_size must be an int >= 1, got {self.tournament_size!r}"
            )
        for name in ("crossover_prob", "mutation_prob", "selection_pressure"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
        if self.selection not in SELECTION_MODES:
            raise ValueError(
                f"selection={self.selection!r} is not one of {SELECTION_MODES}"
            )
        if self.mutation_kind not in MUTATION_KINDS:
            raise ValueError(
                f"mutation_kind={self.mutation_kind!r} is not one of "
                f"{MUTATION_KINDS}"
            )

    def replace(self, **changes) -> "GPConfig":
        """A new config with ``changes`` applied (the receiver is unchanged)."""
        return dataclasses.replace(self, **changes)
