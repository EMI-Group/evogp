"""``evogp.gp`` -- the torch-free, evoxc-based functional GP stack.

The new Evox-2.0-style EvoGP: a functional genetic-programming stack built only
on ``evoxc`` (EvoX Compiler) tensors and numpy host glue.  It replaces the
torch stack in ``../tree/``, ``../algorithm/``, ``../problem/`` and
``../pipeline/`` once parity is established; the torch stack stays in place,
unchanged, as the reference implementation and the migration source.

The idiom is Evox 2.0's (``/root/evox/src/evox/DESIGN.md``): frozen config
dataclasses holding static leaves only (:mod:`evogp.gp.config`), state as a
dataclass whose leaves are evoxc tensors (:mod:`evogp.gp.state`), and plain
module-level functions in the ``init(config, key) -> state`` /
``step(config, state, evaluate) -> state`` protocol.

The stack, bottom up:

``config`` / ``state`` / ``encoding``
    the tree encoding: a ``[pop_size, max_nodes]`` prefix-order channel triple
    (``opcode`` / ``payload`` / ``size``), one traced generation graph per
    static configuration (:func:`init_population`), host-side validation and
    the bridge to ``../tree``'s legacy triple (:func:`to_legacy_triple`).
``selection`` / ``crossover`` / ``mutation`` / ``variation``
    the operator layer: fitness-vector selection, splice crossover, the point /
    subtree / hoist mutation operators, and :func:`apply_variation` -- one whole
    generation in one traced graph, including the elite splice when
    ``elite_count`` is given (``0``, the operator-layer default, is
    offspring-only).  The driver passes ``GPConfig.elite_count``, which is why
    a device generation needs no readback.
``evaluator`` / ``problem``
    the numpy tree evaluator (:func:`predict`) and the problem layer
    (:class:`SymbolicRegression`, :class:`CustomLoss`, plus the in-graph
    :func:`sr_fitness_graph` / :func:`predict_graph`).
``graph_eval`` / ``fused_eval``
    the device-resident in-graph evaluator (:class:`GraphEvaluator`,
    :func:`sr_fitness_channels`, :func:`readback_fitness`) -- what a device run
    plugs in as ``evaluate=`` -- and the optional fused (kernel) backend
    (:class:`FusedEvaluator`, :func:`fused_available`), which resolves its
    dependencies at call time only.
``workflow``
    the driver: :func:`init_gp`, :func:`step` (one generation, elitism included,
    one ``evaluate`` call), :func:`run` (the loop) and :class:`GPResult`.  The
    whole loop is device-resident when the population is
    (``backend="iree", device="rocm"``) and ``evaluate`` returns a device
    fitness tensor; a numpy objective drives the host loop unchanged.

Canonical flow::

    problem = SymbolicRegression(x, y)
    config = (GPTreeConfig(...), GPConfig(pop_size=200))
    result = run(config, problem.evaluate, generations=100, key=0)
    result.best_fitness, result.best_channels

Device-resident flow (same driver, same result object)::

    evaluator = GraphEvaluator(x, y, used_ids=config[0],
                               backend="iree", device="rocm")
    result = run(config, evaluator, generations=100, key=0,
                 backend="iree", device="rocm")
    result.best_fitness_history, result.best_channels   # host rows

Two names are spelled differently at this level than in their modules:
``apply_variation`` is :func:`evogp.gp.variation.variation` and
``apply_crossover`` is :func:`evogp.gp.crossover.crossover`, because the bare
names are the submodules themselves (``evogp.gp.variation`` must stay importable
as a module).

No module of this subtree carries a torch import statement
(``tests/test_gp_stack_purity.py`` pins that; the optional fused backend resolves
torch through ``importlib`` inside its functions, at call time only), and nothing
imports ``evoxc`` at module scope either, so ``import evogp.gp`` works on a plain
checkout; the graphs themselves need evoxc at call time.
"""

from .config import GPConfig, GPTreeConfig, MUTATION_KINDS, SELECTION_MODES
from .crossover import crossover as apply_crossover
from .encoding import (
    from_legacy_triple,
    init_population,
    is_leaf,
    live_mask,
    row_length,
    to_legacy_triple,
    validate_channels,
)
from .evaluator import predict
from .fused_eval import FusedEvaluator, fused_available
from .graph_eval import GraphEvaluator, readback_fitness, sr_fitness_channels
from .mutation import hoist_mutation, mutate, point_mutation, subtree_mutation
from .problem import (
    BaseProblem,
    CustomLoss,
    SymbolicRegression,
    predict_graph,
    sr_fitness_graph,
)
from .selection import select
from .state import (
    CHANNEL_NAMES,
    UNEVALUATED,
    GPState,
    channels,
    max_nodes,
    population_size,
    replace,
)
from .variation import variation as apply_variation
from .workflow import GPResult, init_gp, run, step

__all__ = [
    # config
    "GPTreeConfig",
    "GPConfig",
    "SELECTION_MODES",
    "MUTATION_KINDS",
    # state
    "GPState",
    "channels",
    "replace",
    "population_size",
    "max_nodes",
    "CHANNEL_NAMES",
    "UNEVALUATED",
    # encoding
    "init_population",
    "live_mask",
    "row_length",
    "is_leaf",
    "validate_channels",
    "to_legacy_triple",
    "from_legacy_triple",
    # operator layer
    "select",
    "apply_crossover",
    "mutate",
    "point_mutation",
    "subtree_mutation",
    "hoist_mutation",
    "apply_variation",
    # problem layer
    "BaseProblem",
    "SymbolicRegression",
    "CustomLoss",
    "predict",
    "predict_graph",
    "sr_fitness_graph",
    # device-resident evaluation (the device driver's evaluate=)
    "GraphEvaluator",
    "sr_fitness_channels",
    "readback_fitness",
    # optional fused (kernel) evaluation backend
    "FusedEvaluator",
    "fused_available",
    # driver
    "init_gp",
    "step",
    "run",
    "GPResult",
    # cache helpers
    "cache_stats",
    "clear_cache",
]


# --------------------------------------------------------------------------- #
# the whole-stack cache surface
# --------------------------------------------------------------------------- #
#: ``layer name -> (stats getter, clear function)`` of every layer that memoizes
#: a traced graph.  ``problem`` keeps its own compile cache (the in-graph
#: evaluator), the other five share the one ``evoxc.CompileCache`` of the
#: encoding layer.
_LAYERS = (
    ("encoding", "cache_stats", "clear_cache"),
    ("selection", "cache_stats", "clear_cache"),
    ("crossover", "cache_stats", "clear_cache"),
    ("mutation", "cache_stats", "clear_cache"),
    ("variation", "cache_stats", "clear_cache"),
    ("workflow", "cache_stats", "clear_cache"),
    ("problem", "graph_cache_stats", "clear_graph_cache"),
)

#: The DEVICE-evaluation layers, reported and cleared alongside :data:`_LAYERS`
#: but kept in their own bucket: they only memoize a graph on the device route,
#: so a host run reports them empty, and a caller reading the per-layer table of
#: a host run keeps seeing exactly the seven layers above.
#:
#: :mod:`evogp.gp.fused_eval` is absent on purpose: it memoizes nothing in this
#: process (the fused kernel cache belongs to the sibling engine, which owns its
#: own reset), so a stats entry here would always be a fabricated zero.
_DEVICE_LAYERS = (
    ("graph_eval", "device_graph_stats", "clear_device_graph_cache"),
)


def _layer_modules():
    """The operator/encoding layer modules, imported on demand."""
    from . import (
        crossover,
        encoding,
        mutation,
        problem,
        selection,
        variation,
        workflow,
    )

    return {
        "encoding": encoding,
        "selection": selection,
        "crossover": crossover,
        "mutation": mutation,
        "variation": variation,
        "workflow": workflow,
        "problem": problem,
    }


def cache_stats() -> dict:
    """Graph-memo counters of the whole stack, per layer.

    Returns:
        ``{"graphs": <sum over layers>, "compile_cache_size": ...,
        "compile_cache_maxsize": ..., "layers": {layer name -> that layer's
        counters}, "device_layers": {...}, "device_graphs": <sum over those>}``.
        ``graphs`` summed over ``layers`` is the total number of memoized traced
        graphs of the host stack; ``compile_cache_*`` are the shared
        ``evoxc.CompileCache``'s (owned by :mod:`evogp.gp.encoding`).  The
        device-resident evaluation layer (``graph_eval``) is reported in
        ``device_layers`` and summed into ``device_graphs``: it only memoizes a
        graph on the device route, and keeping it out of ``layers`` leaves that
        table a stable seven entries for host runs.
    """
    modules = _layer_modules()
    layers = {
        name: getattr(modules[name], stats_name)()
        for name, stats_name, _clear_name in _LAYERS
    }
    device_layers = {
        name: getattr(_device_module(name), stats_name)()
        for name, stats_name, _clear_name in _DEVICE_LAYERS
    }
    encoding_stats = layers["encoding"]
    return {
        "graphs": sum(layer["graphs"] for layer in layers.values()),
        "compile_cache_size": encoding_stats["compile_cache_size"],
        "compile_cache_maxsize": encoding_stats["compile_cache_maxsize"],
        "layers": layers,
        "device_layers": device_layers,
        "device_graphs": sum(layer["graphs"] for layer in device_layers.values()),
    }


def _device_module(name: str):
    """One of the device-evaluation layer modules, imported on demand."""
    from . import graph_eval

    return {"graph_eval": graph_eval}[name]


def clear_cache() -> None:
    """Drop every memoized graph of the stack (the compile cache too)."""
    modules = _layer_modules()
    for name, _stats_name, clear_name in _LAYERS:
        getattr(modules[name], clear_name)()
    for name, _stats_name, clear_name in _DEVICE_LAYERS:
        getattr(_device_module(name), clear_name)()
