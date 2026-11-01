# Changelog

Notable changes to EvoGP, newest first.
The `2.0` line is a pre-release series aligned with `evox` 2.0.0a1; entries here carry no date, only the release.

## 2.0.0a2 — pre-release: the single-stack release

`evogp.gp` is now the whole library.
The torch-native stack is deleted, the GP engine is torch-free (`evoxc` graphs + numpy host glue), and it is
presented as `evogp.gp` (the engine), `evogp.fused` (the optional device-native Triton evaluator) and
`evogp.evox_ext` (the EvoX 2.0 `StdWorkflow` adapter).

### Removed

- The torch-native stack, with no shims: the modules `evogp.tree`, `evogp.algorithm`, `evogp.problem`,
  `evogp.pipeline`, `evogp.sr_test` and `evogp.tests` are gone.
- The objects built on it: `Forest`, `Tree`, `CombinedForest`, `GenerateDescriptor`, `GeneticProgramming`,
  `StandardPipeline`, `DefaultSelection`, `DefaultMutation`, `SympyExpression`, `Tree.to_png`, and the classification
  / transformation / robotics problem classes.
- The evaluation-route policy: `EVOGP_EVAL_MODE` and the `auto` / `dense` / `specialized` / `batch` / `fused`
  selection behind it, the dense engine, the per-structure `specialized` (`evoxc_engine`) engine, the opt-in merged
  `batch` engine, and their knobs (`EVOGP_DENSE_*`, `EVOGP_EVOXC_CACHE_SIZE`, `EVOGP_EVOXC_ARTIFACT_DIR`,
  `EVOGP_BATCH_*`, `EVOGP_FUSED_MIN_POP`). The route is now whatever the caller passes as `evaluate=`.
- The optional-dependency surface that went with them: the `torch`, `brax` and `visualize` extras and the
  `scikit_learn`, `networkx` and `sympy` requirements. `install_requires` is `evoxc>=0.1.0` + `numpy`.
- The ten torch-stack example scripts and the three torch-stack tutorials, replaced by five `evogp.gp` scripts and
  one notebook (see below).

### Added

- `evogp.fused`: the device-native Triton evaluation route as its own package — ONE JIT-compiled kernel launch per
  (tree, row block), batch/multi-output value paths and both SR loss paths, a memory envelope, a prepared-call memo,
  the geometry knobs (`EVOGP_FUSED_BLOCK_ROWS`, `EVOGP_FUSED_NUM_WARPS`, `EVOGP_FUSED_MAX_SCRATCH_MB`,
  `EVOGP_FUSED_DISPATCH`, `EVOGP_FUSED_HOST_MEMO`), a 45-case pytest net and the `python -m evogp.fused.check`
  battery (`parity` / `multi` / `goldens` / `declines` / `timing` / `tuning`).
- `evogp.gp.fused_eval.FusedDeviceEvaluator`: the fused kernel as an `evaluate=` plug-in — the population's device
  channels cross to torch as zero-copy DLPack views, the kernel writes the `(pop_size,)` fitness buffer through
  `out=`, and one fence orders it, so a generation is one variation-graph run with no readback. Declines cleanly
  (`interop_available()` / `fused_available()`, or a print of the reason) on a box without the capability.
- `evogp.gp.graph_eval.GraphEvaluator`: the torch-free device-resident in-graph route (two `evoxc.run` calls per
  generation, zero host readbacks), plus `sr_fitness_channels` and `readback_fitness`.
- `evogp.evox_ext`: the EvoX 2.0 adapter — `GPAlgorithmConfig` / `make_gp_algorithm`, `SRProblemConfig` /
  `SRProblemState` / `make_sr_problem`, `make_stdworkflow(...)` (forwarding `StdWorkflow(cache_dir=...)` when the
  installed `evox` declares it, feature-detected), the readbacks `generation_fitness` / `best_so_far` / `best_row` /
  `host_channels`, a runnable demo and its contract suite.
- `evogp.gp` as the public engine surface (44 names): `GPTreeConfig` / `GPConfig` / `SELECTION_MODES` /
  `MUTATION_KINDS`, `GPState` and the channel helpers, `init_population` / `validate_channels` /
  `to_legacy_triple` / `from_legacy_triple`, the operators, `SymbolicRegression` / `CustomLoss` / `predict`,
  `init_gp` / `step` / `run` / `GPResult`, and `cache_stats()` / `clear_cache()`.
- The torch-free single sources of truth `evogp._tables` (node types, function ids, arities) and
  `evogp._multi_output` (the one `OUT_NODE` refusal rule).
- The measurement harnesses `python -m evogp.gp.measure_fused` / `measure_popscale` / `measure_generation`, their
  committed evidence under `src/evogp/gp/evidence/`, and the reports of record `src/evogp/gp/POPSCALE.md` and
  `src/evogp/gp/DEVICE_LOOP.md`.
- `example/gp_sr.py`, `example/gp_custom_loss.py`, `example/gp_uci_sr.py`, `example/gp_device.py`,
  `example/evox_workflow.py`; `tutorial/gp_intro.ipynb`; the dev-only numeric oracle at `test/oracle/`.
- Extras `fused` (torch + triton), `iree` (iree-base-compiler / iree-base-runtime), `evox` (`evox>=2.0.0a1`),
  `dev` (pytest) and `examples` (scikit-learn, for `example/gp_uci_sr.py` only); `MANIFEST.in` now ships
  `CHANGELOG.md` as well as `README.md`.

### Changed

- Version `2.0.0a1` → `2.0.0a2`; the package stays pure Python with no build step and no compiled extension.
- Determinism is key-based: `run(..., key=0)` replaces `torch.manual_seed(0)`. The RNG is an `evoxc` splitmix64 key
  carried in `GPState` (the key of the NEXT generation), so a run is reproducible from one integer with no global
  seed and no torch.
- The population is three prefix-order `[pop_size, max_nodes]` channels (`opcode` int32, `payload` float32, `size`
  int32) instead of a `Forest` of `Tree` objects; the winner is reported as `GPResult.best_channels`.
- Evaluation is three explicit routes — host numpy, the in-graph `iree` `GraphEvaluator`, and the fused Triton
  `FusedDeviceEvaluator` — instead of a policy-selected engine. Nothing switches routes behind a call.
- The fitness contract is repo-wide: `evaluate(channels)` returns a fresh `(pop_size,)` float32 vector that is
  MAXIMIZED, and every non-finite entry is mapped to `-inf` before ranking.
- Importing a submodule is now cheap everywhere: `import evogp.gp` needs neither torch, `evox` nor `evoxc` at
  module scope (the purity is pinned by a source-grep test), and `import evogp` stays empty.
- `README.md` was rewritten as a single-stack document (install, quick start, the three routes with their
  measured numbers, the API, the migration list, layout, validation, licence).

### Known limitations

- **Symbolic regression is the whole problem layer.** No classification, transformation or robotics problems.
- **Single-output only in the stack**: `init_population` raises `NotImplementedError` for `output_len > 1` and every
  evaluator refuses an `OUT_NODE`-tagged population; `evogp.fused` itself serves `output_len` 2..8 on the legacy
  triple, and `CustomLoss` is the escape hatch for any other objective.
- **No export and no population container**: nothing renders a tree to sympy, to an infix string or to a PNG, and
  there is no `Forest`-like object protocol — the population is the three channel rows.
- **Per call, the fused objective as `evogp.gp` calls it is 9.0×–19.8× dearer than the deleted torch stack's own
  pre-bound fused route** on the same trees (`src/evogp/gp/POPSCALE.md` §1.4/§5(a)): it pays the channel-triple
  crossing. The win is the generation loop (1.83×–2.31× against the torch stack, 1.69×–2.34× against it with its
  fused kernel on), where the operator layer is one traced graph.
- **The in-graph `iree` body materializes `[pop_size, n_rows]` intermediates per tree column**, so it is far dearer
  than the host numpy evaluator at large `n_rows` (16× at `N=1000`, pop 200, `max_nodes=64`) and the iree lowering
  declines a 64-wide channel graph on the reference box (`max_nodes=32` compiles).
- **The fused route's residual corners**: the multi-output LOSS path accumulates in float32 per row block
  (≤1.2e-07 relative, non-bitwise by construction, where the value path is bitwise); `rows ≤ 64` at 8–32-node trees
  is unmeasured; `output_len > 8` declines; the envelope is a budget rather than a guarantee; and there is no
  per-stage timer surface.
- **`FusedDeviceEvaluator` returns ONE reused device fitness buffer**, overwritten by the next device call — copy it
  before the next call if the numbers must be kept.
- **Channel validation is a Python loop** (~0.5 µs per live node), so `validate=False` is the only mitigation.
- **No CI**: the three suites (`src/evogp/gp/tests`, `src/evogp/fused/tests`, `src/evogp/evox_ext/tests`) and the
  fused battery are run by hand, quoted per path.
- The two measurement records (`src/evogp/gp/POPSCALE.md`, `src/evogp/gp/DEVICE_LOOP.md`) and the fused route's
  `src/evogp/fused/FUSED.md` were measured on the pre-2.0.0a2 tree with torch comparison lanes that this release no
  longer carries; their numbers are read as records, not as a reproducible baseline of this tree.

## 2.0.0a1 — pre-release: the two-stack line

First release aligned with `evox` 2.0.0a1: two stacks over one `evoxc` engine — the torch-native
`evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline` and the torch-free `evogp.gp` — with four
selectable evaluation routes (`fused`, `dense`, `specialized`, opt-in `batch`) behind `EVOGP_EVAL_MODE`; the torch
stack ships in that release and is recoverable at commit `6720723`.
