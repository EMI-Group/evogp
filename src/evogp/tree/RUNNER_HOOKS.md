# `RUNNER_HOOKS.md` — build-once / run-many benchmark runners

Design document of record for the two **benchmark-support** runner APIs:

| builder | module | route |
| --- | --- | --- |
| `build_dense_runner(population_or_forest, inputs, *, ...)` | `dense_engine.py` | the ONE compiled dense graph (`plan_dense` + `build_dense`) |
| `build_specialized_runner(population_or_forest, inputs, *, ...)` | `etl_engine.py` | the per-structure specialized engine (`decode_population` + `_executable` + one `etl.run` per tree) |

They are **not** part of the public route: nothing in the library calls them, `Forest`/`Tree` are
untouched, the training path never sees them, and they do not change `evaluate` / `sr_fitness` /
`evaluate_dense` / `sr_fitness_dense`. They exist so a harness can time **the compiled forward pass
alone** (the warm-path floor of `test/benchmark/REPORT.md` §8 levers 1–3) without the per-call host
glue the engine's own entry points necessarily carry: no re-plan, no re-decode, no `build_cached` key
derivation, no `_executable` lookup, no input re-preparation/re-upload, no readback, no torch conversion.

Both builders construct the same object as calling the class directly (`DenseRunner(...)` /
`SpecializedRunner(...)`); **the constructor is the build phase**.

## 1. Frozen signatures

```python
# src/evogp/tree/dense_engine.py
def build_dense_runner(
    population_or_forest, inputs, *, per_row=False,
    backend=None, iree_device_index=None,
    max_tree_len=None, budget_mb=None, update_mode=None,
) -> DenseRunner: ...

# src/evogp/tree/etl_engine.py
def build_specialized_runner(
    population_or_forest, inputs, *, per_row=False,
    backend=None, iree_device_index=None,
) -> SpecializedRunner: ...
```

The names, the positional order and the `.run()` semantics are **frozen** (the benchmark harness in
`test/benchmark/` is the consumer). Everything is keyword-only after `inputs`; the extra dense keywords
are exactly `evaluate_dense`'s knobs (`max_tree_len` = the trace cap, `budget_mb` = `EVOGP_DENSE_BUDGET_MB`,
`update_mode`), and `backend` / `iree_device_index` override `EVOGP_ETL_BACKEND` / `EVOGP_ETL_IREE_DEVICE`
per runner instead of per call.

* `population_or_forest` — duck-typed on `batch_node_value` / `batch_node_type` / `batch_subtree_size` /
  `input_len` / `output_len` (`etl_engine._population_inputs`). Any `Forest` works; the container is never
  imported (that would be an import cycle: `forest`/`tree` import the engine, not the reverse). A
  non-Forest-like object raises `TypeError` naming the missing attributes.
* `inputs` — `x` with the engine's own validation semantics (`etl_engine._input_to_numpy`): a
  `(N, input_len)` tensor/array converted to contiguous float32 **once, at build** (`N == pop_size` when
  `per_row=True`, where tree `i` sees row `i`). Wrong columns / a `per_row` row-count mismatch raise
  `ValueError` at build time.
* Both builders deliberately **ignore `EVOGP_EVAL_MODE`**: the route is chosen by choosing the builder.
  There is no silent fallback — an unsupported dense population raises `DenseUnsupported` instead of
  quietly running on the specialized engine. Neither builder writes to the environment.

## 2. Build-vs-run split (what costs what)

| phase | dense runner | specialized runner |
| --- | --- | --- |
| duck-type + `inputs` validation/conversion | build | build |
| host decode (`decode_population`) | **never** (the dense path works from the tensors) | build, exactly **once** |
| `plan_dense` (host preprocessing + chunk shape + graph specs) | build | — |
| compile | build: `build_dense` = **ONE** graph per shape signature (1 cache miss for the whole population) | build: `_executable` per **non-constant tree**, i.e. one `build_cached` miss per **distinct program** |
| input staging | build: the 7 `(L, P)` node columns + `x` for the single chunk, uploaded once on a device backend (`_iter_plan_chunks` with an injected payload memo) | build: `x` once (`per_row=False`) or one `(1, input_len)` row per non-constant tree (`per_row=True`) |
| warm-up | build: one `etl.run` that fills the payload memo | none (no graph is run at build) |
| `.run()` | **one `etl.run`** over the staged payloads | **`n_nonconst` `etl.run` calls** over the staged payloads |
| `.run()` never | re-plans, re-derives a `build_cached` key, looks up `_executable`, re-prepares/re-uploads a payload, reads a result back, converts to torch | decodes, looks up `_executable`, re-uploads `x`, reads a result back, converts to torch, assembles host rows |

Constant trees (folded at decode, never compiled) are handled without a graph: the dense path has none,
and the specialized runner returns `None` for those slots (their values are staged read-only in
`.constant_rows`, a `(n_rows, output_len)` broadcast view of the float32 constant).

`E.clear_cache()` / `E.cache_stats()` bracket both phases in `runner_check.py`; after the build,
**every** `cache_stats()` delta of repeated `.run()` calls is `{}` (0 misses, 0 hits, 0 entries).
The `_EXECUTABLE_MEMO` is not a `CompileCache` field, so its activity is measured as a delta
(`memo_hits` +1 per lookup, `memo_misses` unchanged after the build).

## 3. `.run()` semantics and return contract, per backend

`DenseRunner.run() -> Any` — the **raw `etl.run` result** of this plan's single chunk: an `etl` runtime
tensor of shape `(pop_size, n_rows)` (`n_rows == 1` for `per_row=True`).

* `iree` — **device-resident** (`etl.core.Device("cuda", index)`), no readback, no host assembly, no
  torch conversion. The consumer reads it back itself (`raw.to(etl.core.Device("cpu", 0)).numpy()`);
  that copy is the consumer's step, not a runner step, and belongs **outside** the timed region.
* `numpy` — host-backed by construction (the reference interpreter never leaves the host): the result
  *is* a host array, nothing is copied, and `raw.numpy()` is a zero-copy view. This is the numpy
  analogue of "no readback" and is what `device_resident=True` deliberately **refuses** (a
  "device-resident" result on numpy is meaningless, and the flag also requires a single-chunk plan).
* Chunking is **refused at build**: `DenseUnsupported` when `plan.p_chunks > 1 or plan.n_chunks > 1`,
  because a chunked result cannot be assembled without a host (or a second compiled graph) step, which
  is outside this API's contract. Raise `EVOGP_DENSE_BUDGET_MB` to keep a benchmark shape in one chunk.
  Multi-output populations, over-trace-cap trees and malformed trees raise `DenseUnsupported` /
  `TypeError` / `ValueError` from `plan_dense` (same taxonomy as `evaluate_dense`).

`SpecializedRunner.run() -> Tuple[Optional[Any], ...]` — a `pop_size`-long tuple; entry `i` is the **raw
`etl.run` result of tree `i`** (an `etl` runtime tensor of shape `(n_rows, output_len)`: host-backed on
numpy, device-resident on iree), or `None` for a constant tree.

* **No host assembly** — the per-tree results stay exactly where the engine produced them; there is no
  `(P, N, out)` stitch, no numpy reduction, no torch conversion. A caller that needs assembled rows
  does that itself (see `runner_check._assemble_specialized`, which also stitches `constant_rows`).
* The static program is re-passed by value on every `etl.run`, exactly as the frozen engine does, so
  every call still goes through `etl.run`'s run boundary and its per-call validation of the statics —
  now against the ETL-side **static-validation fast path** (a per-executable content fingerprint instead
  of an O(#tokens) per-leaf walk: ETL bench, numpy, a 60-token program 303 → 67–71 µs per `etl.run`,
  170 tokens 758 → 87–96 µs; see §5).

Repeatability: the executables and the staged payloads are re-used as-is, so repeated `.run()` calls are
bit-identical and issue exactly the same number of `etl.run` calls (1 / `n_nonconst`) with **zero** host
crossings of any kind — measured, not assumed (`runner_check.py` stage `dense` / `specialized`).

Diagnostics (read-only attributes; a harness may print them but must not write them):
`DenseRunner`: `.plan` (`.trace_len` / `.pop_size` / `.n_rows` / chunk shape / `.peak_bytes` / variant —
`describe_plan`; the population / row counts live **only** here, on the plan), `.executable`, `.staged`
(the payload memo), `.backend_name`, `.device`, `.device_resident`, `.per_row`, `.input_len`,
`.output_len`, `.invocations` (== 1).
`SpecializedRunner`: `.population` (decoded), `.executables`, `.payloads`, `.constant_rows`,
`.backend_name`, `.device`, `.options`, `.per_row`, `.input_len`, `.output_len`, `.pop_size`, `.n_rows`,
`.invocations` (== non-constant trees).

## 4. What is timed

```text
build region   build_dense_runner(...)  /  build_specialized_runner(...)   # compile + stage
timed region   runner.run()                                               # forward pass ONLY
```

The timer must bracket `.run()` and nothing else; in particular **never** put a readback or a
`.numpy()` / `torch.from_numpy(...)` conversion inside the timed region, and do not reuse one runner
across different `pop_size` / `n_rows` / `input_len` shapes (that re-compiles). One runner per
`(population shape, data shape, backend)`; call `.run()` many times for the median.

## 5. Adoption points for the ETL-side work (marked, NOT implemented here)

`SpecializedRunner.run()` carries an `ADOPTION POINT` comment on the single `etl.run(executable,
payload, tree.program)` line. The ETL-side facility that line swaps to has since landed:
`etl.jit(fn, *, backend=..., device=...)` returns a function-entry compile cache (`JitFunction` /
`BoundJit`, exported from `etl/__init__.py`) whose `f.bind(program=prog)` pre-bound runner takes only
the tensor argument on the call path and dispatches the already-compiled executable — no per-call cache
key derivation for the bound statics (ETL bench, numpy, 60-token program: bound dispatch 320 µs against
314 µs for the raw `etl.run`, i.e. within ~2 %). What is left to adopt here is therefore the
consumer-side plumbing, not the two costs this section used to name: ETL's per-executable run-path
static-validation fast path has already replaced the O(#tokens) per-leaf walk (60-token program
303 → 67–71 µs per `etl.run`, 170 tokens 758 → 87–96 µs), and its multi-slot `CompileCache`
content-fingerprint memo replays a repeated program's key (~17–25 µs at 30–170 tokens; ~32 µs per call
in a steady-state K=50 cycle of 60-token programs, against ~660 µs before). All of that landed ETL-side
on branch `evogit-agent-T1-A160`, on top of the `main` checkout the venv installs — nothing is emulated
here and `/mnt/local-ssd/bchuang/etl` is **not modified** by this repository. The dense runner has no
equivalent point: its `cfg` is one static block per chunk and the same model applies there
(`_iter_plan_chunks` re-passes it by value once per `.run()`).

Also deferred (documented, deliberately not built):

* **chunked dense plans** — refused (`DenseUnsupported`); assembly needs a host step or a second
  compiled graph fed with the chunk results.
* **device-side assembly for the specialized route** — needs ETL multi-program execution or a
  zero-copy device export (`core.Tensor.__dlpack__` raises for payload-backed tensors today); the
  per-tree dispatch/readback structure is a `tree/CONTEXT.md` known issue, not a runner limitation.
* **numpy "device residency"** — meaningless (the interpreter is host-only); the numpy contract above
  is the documented semantics instead.

## 6. Example (numpy, `ETL_DISABLE_LOCATIONS=1`, RTX A6000 box — measured)

```python
import statistics, time, torch
from evogp.tree import Forest, GenerateDescriptor
from evogp.tree.dense_engine import build_dense_runner
from evogp.tree.etl_engine import build_specialized_runner

torch.manual_seed(0)
descriptor = GenerateDescriptor(
    max_tree_len=16, input_len=3, output_len=1,
    using_funcs=["+", "-", "*", "/"], max_layer_cnt=4,
    const_range=(-1.0, 1.0), sample_cnt=100,
)
forest = Forest.random_generate(pop_size=64, descriptor=descriptor)
x = torch.randn(1024, 3, device="cuda")          # the engine is device-agnostic; cuda matches the forest


def timed(build, repeat=5):
    start = time.perf_counter()
    runner = build()                             # build phase
    build_ms = (time.perf_counter() - start) * 1e3
    calls = []
    for _ in range(repeat):                      # timed region: .run() only
        start = time.perf_counter()
        runner.run()
        calls.append((time.perf_counter() - start) * 1e3)
    return runner, build_ms, statistics.median(calls)


dense, dense_build, dense_run = timed(lambda: build_dense_runner(forest, x, backend="numpy"))
spec, spec_build, spec_run = timed(lambda: build_specialized_runner(forest, x, backend="numpy"))
```

Recorded for `P=64`, `N=1024`, `L=16` (numpy; one process; `trace_len=16`):

| runner | build | `.run()` median | per `.run()` |
| --- | --- | --- | --- |
| dense | 1 943.2 ms | 340.666 ms | 1 `etl.run`, 1 compile |
| specialized | 154.4 ms | 8.251 ms | 51 `etl.run` (51 non-constant trees of 64), 51 compiles at build |

The dense build is compile-dominated (ONE graph for the whole population) while the specialized build
pays one (cheap, numpy) compile per **distinct** program; the two `.run()` numbers are direct: one dense
batch launch against 51 per-tree launches, which is exactly the "collapse the K per-tree invocations"
lever of `test/benchmark/REPORT.md` §8 in a controlled setting (the specialized runner times *only* the
launch side — its build already paid decode, cache hits and `x` upload; so the gap here is a **floor**
for the dense win, not a replacement for the end-to-end harness numbers).

Smaller battery fixture (`runner_check.py` defaults, `P=8`, `N=37`, `L=16`, one constant tree injected,
numpy): dense build 1050.8 ms / `.run()` 40.657 ms; specialized build 12.6 ms (5 compiles for 5 distinct
programs) / `.run()` 0.549 ms (5 trees per call). iree (`P=4`, `N=8`, one process,
`CUDA_VISIBLE_DEVICES=6 --gpu 0`): dense build 6 866.4 ms / `.run()` 2.818 ms; specialized build 835.3 ms
(2 compiles) / `.run()` 0.505 ms. Both iree runs are staged one config per process (the recorded
teardown-only nanobind leak closes the process *after* the JSON/OK line; exit code stays 0).

## 7. Evidence battery

`src/evogp/tree/runner_check.py` — dev CLI, `__main__`-guarded, imported by nothing, NOT
pytest-collected (it reuses fixtures/helpers from the sibling batteries, so it is not importable as a
library either). It counts host↔device traffic by monkeypatching the conversion / compile / run sites
inside a `with Traffic()` block and tagging each record with its **caller frame** (name + file), because
the runner entry point is itself named `run` and ETL's interpreter internals call `.numpy()` too
(6 111 such internal calls on a numpy dense `.run()`, 0 from runner/seam frames — hence the filter).

```bash
# numpy: all three stages, must be ALL-GREEN
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python src/evogp/tree/runner_check.py
# -> == OK: 48/48 checks passed ==

# iree leg (one config per process, visible device 0 == CUDA_VISIBLE_DEVICES=6, <=8 GB)
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python src/evogp/tree/runner_check.py \
  --backend iree --gpu 0 --pop 4 --rows 8
# -> == OK: 32/32 checks passed ==   (nanobind teardown-only leak notice afterwards; exit 0)
```

Checks per leg: (1) build phase — exactly ONE dense compile / one decode + one `_executable` per
non-constant tree, one compile per distinct program, payloads staged once, no graph run at the
specialized build, single-chunk plan; (2) `.run()` — exactly 1 / `n_nonconst` `etl.run` calls and
**zero** readbacks, uploads, preparations, torch conversions, `build_cached` calls, `_executable`
lookups and decodes, as measured by the patched counters; (3) repeatability — N further calls
bit-identical with the same traffic and **zero** cache deltas, staged payload memo re-used by reference;
(4) numerics vs `etl_engine.evaluate` (and `Forest.forward` / `batch_forward` on numpy) for
`per_row=False` / `per_row=True` / multi-output, `rtol=atol=1e-4` on finite values with exact NaN/±inf
masks (`dense_integration_check.compare`); (5) **positive controls** — the same counters must fire for
`run_plan` (per-chunk `_to_host` readbacks + a FRESH seam call re-staging all 8 payloads) and for
`evaluate` (one `decode_population` + per-tree `_run_tree` readbacks), so a filter that silently stopped
matching cannot pass as a green zero; (6) build-time contracts — `DenseUnsupported` for multi-output and
for a chunked plan, `ValueError` for wrong columns / `per_row` row mismatch / a malformed tree, `TypeError`
for a non-Forest-like container, and both builders ignoring `EVOGP_EVAL_MODE` without writing the
environment.

Recorded failure-first state: commit `6b232fe` (the battery, with `build_dense_runner` /
`build_specialized_runner` absent → `== FAILED: 0/2 checks passed ==`), then the implementation.
