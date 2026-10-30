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

Nothing here imports torch (``tests/test_gp_stack_no_torch.py`` pins that), and
nothing imports ``evoxc`` at module scope either, so ``import evogp.gp`` works on
a plain checkout; the graphs themselves need evoxc at call time.

Current surface: the encoding foundation (config, state, generation, channel
helpers, validation and the bridge to the tree layer's legacy triple).  The
operator layer (selection/crossover/mutation), the problem layer, the numpy
evaluator and the driver loop are the next milestones.
"""

from .config import GPConfig, GPTreeConfig
from .encoding import (
    cache_stats,
    clear_cache,
    from_legacy_triple,
    init_population,
    is_leaf,
    live_mask,
    row_length,
    to_legacy_triple,
    validate_channels,
)
from .state import GPState, channels, population_size, replace

__all__ = [
    # config
    "GPTreeConfig",
    "GPConfig",
    # state
    "GPState",
    "channels",
    "replace",
    "population_size",
    # encoding
    "init_population",
    "live_mask",
    "row_length",
    "is_leaf",
    "validate_channels",
    "to_legacy_triple",
    "from_legacy_triple",
    "cache_stats",
    "clear_cache",
]
