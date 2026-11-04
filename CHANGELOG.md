# Changelog

Notable changes to EvoGP, newest first.
`1.0` is the first public pre-release series; `1.0.0a1` is the new `evoxc`-based engine.

## 1.0.0a1 — 1.0-alpha (pre-release): the evoxc-based engine

`evogp.gp` is now the whole engine.
The pre-1.0 torch-native stack is deleted, the GP engine is torch-free (`evoxc` graphs + numpy host glue), and it is
presented as `evogp.gp` (the engine), `evogp.fused` (the optional device-native Triton evaluator) and
`evogp.evox_ext` (the EvoX 2.0 `StdWorkflow` adapter).

### Removed

- The pre-1.0 torch-native stack, with no shims: the modules `evogp.tree`, `evogp.algorithm`, `evogp.problem`,
  `evogp.pipeline`, `evogp.sr_test` and `evogp.tests` are gone.
- The objects built on it: `Forest`, `Tree`, `CombinedForest`, `GenerateDescriptor`, `GeneticProgramming`,
  `StandardPipeline`, `DefaultSelection`, `DefaultMutation`, `SympyExpression`, `Tree.to_png`, and the classification
  / transformation / robotics problem classes.
- The evaluation-route policy: `EVOGP_EVAL_MODE` and the `auto` / `dense` / `specialized` / `batch` / `fused`
  selection behind it, the dense engine, the per-structure `specialized` (`evoxc_engine`) engine, the opt-in merged
  `batch` engine, and their knobs (`EVOGP_DENSE_*`, `EVOGP_EVOXC_CACHE_SIZE`, `EVOGP_EVOXC_ARTIFACT_DIR`,
  `EVOGP_BATCH_*`, `EVOGP_FUSED_MIN_POP`). The route is now whatever the caller passes as `evaluate=`.
- The optional-dependency surface that went with them: the `torch`, `brax` and `visualize` extras and the
  `scikit_learn`, `networkx` and `sympy` requirements. The runtime dependencies are `evoxc>=0.1.0` + `numpy`.
- The ten torch-stack example scripts and the three torch-stack tutorials, replaced by five `evogp.gp` scripts and
  one notebook (see below).
- `setup.py` and `MANIFEST.in`: both are replaced by the `pyproject.toml` metadata and the
  `[tool.uv.build-backend]` inclusion rules.

### Added

- `evogp.fused`: the device-native Triton evaluation route as its own package — ONE JIT-compiled kernel launch per
  (tree, row block), batch/multi-output value paths and both SR loss paths, a memory envelope, a prepared-call memo,
  the geometry knobs (`EVOGP_FUSED_BLOCK_ROWS`, `EVOGP_FUSED_NUM_WARPS`, `EVOGP_FUSED_MAX_SCRATCH_MB`,
  `EVOGP_FUSED_DISPATCH`, `EVOGP_FUSED_HOST_MEMO`), a 45-case pytest net and the `python -m evogp.fused.check`
  battery (`parity` / `multi` / `goldens` / `declines` / `timing` / `tuning`). It is the one subtree of the package
  that imports `torch` at module scope, so `import evogp.fused` needs the `fused` extra (`torch` + `triton`);
  `triton` itself stays lazy, inside `fused/kernels.py`, and `evogp.gp` stays torch-free.
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
- The release pipeline `.github/workflows/release.yml`, on `release: published` plus a `workflow_dispatch` dry run:
  it verifies the tag `v<version>` against `[project].version`, runs
  `uv run --with pytest --with evoxc --with numpy python -m pytest src/evogp/gp/tests -q` (CPU-only), runs
  `uv build`, smoke-installs the wheel in a clean venv and runs `python -m evogp.gp.smoke` from outside the
  checkout, publishes the distributions to PyPI through Trusted Publishing, and attaches `dist/*` to the release.
- Extras `fused` (torch + triton), `iree` (iree-base-compiler / iree-base-runtime 3.11), `evox` (`evox>=2.0.0a1`),
  `dev` (pytest) and `examples` (scikit-learn, for `example/gp_uci_sr.py` only), declared in
  `[project.optional-dependencies]`; the in-package Markdown ships in both archives, while the pytest suites stay
  in the repository and the `gp/evidence/*.json` files are sdist-only.

### Changed

- `1.0.0a1` is the first public pre-release of the new engine; the package stays pure Python with no build step and
  no compiled extension, and the distributions are built by the `uv_build` backend from `pyproject.toml` (`uv build`).
- `pyproject.toml` is the single packaging source: `[project]` carries the metadata (version, `requires-python`,
  SPDX `license`, extras, dependencies, URLs) and `[tool.uv.build-backend]` carries the inclusion rules — the whole
  module directory under `src`, `source-exclude = ["tests"]` and `wheel-exclude = ["gp/evidence"]`.
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
- `README.md` was rewritten for the 1.0-alpha release (install, quick start, the new-vs-old comparison, the three
  routes with their measured numbers, the API, the migration list, layout, validation, releasing and licence).

### Known limitations

- **Symbolic regression is the whole problem layer.** No classification, transformation or robotics problems.
- **Single-output only in the stack**: `init_population` raises `NotImplementedError` for `output_len > 1` and every
  evaluator refuses an `OUT_NODE`-tagged population; `evogp.fused` itself serves `output_len` 2..8 on the legacy
  triple, and `CustomLoss` is the escape hatch for any other objective.
- **No export and no population container**: nothing renders a tree to sympy, to an infix string or to a PNG, and
  there is no `Forest`-like object protocol — the population is the three channel rows.
- **Per call, the fused objective as `evogp.gp` calls it is 9.0×–19.8× dearer than the pre-1.0 torch-native stack's
  own pre-bound fused route** on the same trees (`src/evogp/gp/POPSCALE.md` §1.4/§5(a)): it pays the channel-triple
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
- **CI is release-only**: nothing runs on a pull request or a push, and the release pipeline gates on
  `src/evogp/gp/tests` alone; the three suites (`src/evogp/gp/tests`, `src/evogp/fused/tests`,
  `src/evogp/evox_ext/tests`) and the fused battery are otherwise run by hand, quoted per path.
- The two measurement records (`src/evogp/gp/POPSCALE.md`, `src/evogp/gp/DEVICE_LOOP.md`) and the fused route's
  `src/evogp/fused/FUSED.md` were measured on the pre-1.0 torch-native tree with torch comparison lanes that this
  release no longer carries; their numbers are read as records, not as a reproducible baseline of this tree.

## Development line (unreleased, superseded): the two-stack 2.0.x

The two-stack `2.0.x` line was never published, and its version naming is retired in favour of `1.0.0a1`.
It described two stacks over one `evoxc` engine — the torch-native `evogp.tree` / `evogp.algorithm` / `evogp.problem`
/ `evogp.pipeline` and the torch-free `evogp.gp` — with four selectable evaluation routes (`fused`, `dense`,
`specialized`, opt-in `batch`) behind `EVOGP_EVAL_MODE`; the torch stack it shipped is recoverable at commit
`6720723`.
