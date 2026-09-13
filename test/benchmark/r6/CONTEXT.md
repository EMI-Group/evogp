# `test/benchmark/r6/` — Triton fused-kernel evaluation prototype (feasibility + measurement round)

## Intent
Round 6 tests the r5 RCA hypothesis that a custom kernel replicating the legacy CUDA engine's fused design restores legacy-class performance for EvoGP's single-output SR case.
Context of record: `../r5/RCA.md` (§9) — the ≈8–11× gap decomposes into per-call host glue (≈4.2–4.3 ms of the ~5.4–5.8 ms dense call) plus a dispatch-bound 310-op dense graph (249 dispatches + 61 DtoD copies, 125-deep chain, ~1.4 µs kernels) versus the legacy engine's 3 device ops (memset + fused tree-walk kernel + averaging kernel).
This round is feasibility + measurement: the prototype lives in `test/` (this node) only — no `src/` changes, no foreign-repo writes, no edits to committed r2/r4/r5 artifacts or shared harnesses.
A decisive negative (documented no-go with specific evidence) is an acceptable outcome; fabricated numbers are not.

## Deliverables (the round's acceptance criteria)
1. Prototype: `triton_eval.py` + drivers — a Triton fused evaluator for single-output SR consuming the same staged inputs as the dense route; one compile per shape; legacy-style runtime node interpreter; target 1–2 launches/call; zero per-call uploads/readback; build-once/run-many.
2. Numeric validation: triton vs `src/evogp/tests/reference_eval.py` (plain-torch oracle) and vs the `test/goldens/` legacy captures, on fixture64 + fixture200; rtol/atol 1e-4, equal_nan; report exactness (+ − * bit-equal; / last-ulp) and declare op coverage explicitly (fixture ops {+,-,*,/} required; full 29-op table is a stretch goal).
3. Same-window measurement campaign (one continuous window; GPU 6; one fresh process per cell; fcntl-serialized): legacy f64+f200, dense-iree f64+f200, triton f64+f200 — type-1 evaluate envelope; type-2 pure-forward for dense + triton; triton first-call/compile economics + cache behavior; immutable JSON+log records with provenance in `results/<window>/`; operator log.
4. `SUMMARY.md`: per-call decomposition (triton kernel-vs-host; light legacy decomposition if cheap) + verdict answering Q1–Q5: Q1 semantics correct? Q2 how fast vs legacy/dense same-window? Q3 what remains of the gap + what would close it? Q4 coverage/limits + integration path + effort estimate? Q5 does the evidence support "same implementation ⇒ legacy-class performance"? (yes/no + the flat-host-cost and 3-op-vs-310-op caveats).
5. `check_r6.py`: stdlib-only, GPU-free gate recomputing every headline from stored records + validating envelope/provenance identities (25 calls, call 0 cold, median warm, routes, provenance fields); house standard ≈ hundreds of checks / 0 failures.
6. Docs: `README.md` (protocol, session-scoped env/CPATH fix, exact repro commands), this file kept current; parent `../CONTEXT.md` is updated by the parent agent, not from here.

## Status
- Skeleton created (mission brief below); no artifacts yet — keep this section and the Routing Table current as phases land.
- Campaign runner in place and offline-validated: `run_cells.sh` (`--list`, `--dry-run`, `--only`, `--window`, freshness gate; the `check` group's dependency-pre-flight body exercised with a stub interpreter and its report/gate semantics verified) — **no cell has been executed on the GPU yet**, so `results/` is still empty and the window is unrecorded; the campaign runs later in one continuous window (`bash run_cells.sh --group check --window <label>` then `--group all --window <label>`).

## Environment (session-scoped; verify, don't assume)
- Python: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2, torch 2.6.0+cu124, triton 3.2.0 with its own ptxas); no `python`/`python3` on PATH with torch.
- Triton runtime compile needs `Python.h`, missing at `/usr/include/python3.11`; working fix (~5 s, no root): download the `libpython3.11-dev` deb into `$TMPDIR`, `dpkg-deb -x` it, then `export CPATH="$TMPDIR/py311hdr/usr/include/python3.11:$TMPDIR/py311hdr/usr/include"` (both entries required); a ready header tree exists at `/tmp/triton_feas/py311hdr` — reuse if still present.
- Set `TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache` for reproducible compile caching; keep the cache out of the repo (checksum the cache only if useful).
- GPU: `CUDA_VISIBLE_DEVICES=6`; never GPU 7 (hardware ECC fault); check `nvidia-smi` before every timed run; GPU 6 may carry a co-tenant's committed memory (idle at times) — never kill foreign processes, just measure and note contention.
- `flock` *binary* hangs in this sandbox — serialize ALL GPU work with a Python `fcntl.flock(LOCK_EX)` helper on `/tmp/gpbench_critical.lock` (check r4/r5 runners for the established pattern and reuse it); never nest locks; keep `timeout` inside the lock.
- Legacy engine (read-only): `/tmp/evogp_legacy_bench`; run with `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`; rebuild recipe in `../legacy/BUILD_NOTES.md` + `../legacy/setup_legacy.sh` if wiped.
- Lock helper (fallback if no repo helper is found; usage: `python run_under_lock.py timeout -s INT -k 30 <cmd...>`):
```python
import fcntl, os, subprocess, sys, time
fd = os.open("/tmp/gpbench_critical.lock", os.O_CREAT | os.O_RDWR, 0o644)
deadline = time.time() + 7200
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
    except BlockingIOError:
        if time.time() > deadline: sys.exit("lock timeout after 7200s")
        time.sleep(5)
try:
    sys.exit(subprocess.run(sys.argv[1:]).returncode)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
```
- Disk: `/tmp` (== `/`) is ~97 % full (~15 GB free) — keep outputs small, no giant binary logs.

## Contracts & semantics (single-output SR)
- Output contract: positive `(pop,)` float32 loss; the caller negates (`forest.SR_fitness` → `SymbolicRegression.evaluate`); `problem.evaluate(forest)` is the public API the benchmark times.
- Forest schema: `(pop, max_tree_len)` prefix (pre-order) encoded; `batch_node_value` f32, `batch_node_type` i16, `batch_subtree_size` i16; only `[0, subtree_size[0])` meaningful; arity from base type (VAR=0/CONST=1 leaves, UFUNC=2, BFUNC=3, TFUNC=4); `MAX_STACK=1024`; node 0 = root.
- Op semantics of record: `src/evogp/tree/ENGINE_SPEC.md` §3; `/` = `var2==0 ? NaN : var1/var2` (select, not trap); `+ - *` f32-exact; NaN/±inf propagate, never sanitize.
- Dense-route staging: `src/evogp/tree/dense_engine.py` `prepare_dense(...)` → 7 `(L,P)` columns `[opcode, func_id, const_value, var_index, child1, child2, child3]` (+ `tree_len`); reuse for byte-identical staging, or decode from the raw schema in-kernel — declare which was used.
- Legacy blueprint (`b613503`, `src/evogp/cuda/forward.cu`): reversed-prefix prefix-row walk where the same array doubles as the operand stack (children before parents; arity from base type; no child indices); op dispatch by function id; fitness path = `cudaMemsetAsync` + per-(tree,datapoint)-thread fused kernel (shared-mem reduction + atomicAdd) + averaging kernel = 3 device ops; no per-structure caching, recompute every call, no readback.
- Type-1 triton envelope: runtime-patch the dense seam `evogp.tree.dense_engine.sr_fitness_dense` (call site `src/evogp/tree/forest.py` ~L338-361) with `EVOGP_EVAL_MODE=dense`, then drive the unmodified public API `problem.evaluate(forest)`; the patched function must return exactly what `sr_fitness_dense` returned (positive loss); state exactly what ran; if the patch proves fragile, replicate the bench_gp eval64 loop faithfully and document the equivalence.
- Fixtures: fixture64 = `test/goldens/forest_random.npz` (64 trees, N=64 rows, x (64,2), labels = x0²−x1); fixture200 = `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, same 64 rows bitwise); loader `test/benchmark/bench_gp.py::build_eval_case` (~L768-819).

## Protocol conventions (mirror r4/r5)
- Type-1 cells: `bench_gp.py --task eval64 --fixture <F> --eval-reps 25`; 25 sync-bracketed `problem.evaluate(forest)` calls; call 0 cold; `median_warm_ms = median(calls 1..24)`; one fresh process per cell.
- House command shape: `env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 [EVOGP_*] <venv>/bin/python test/benchmark/bench_gp.py <args> --tag <TAG> --json-out <out>.json`, all inside the fcntl lock with `timeout` inside.
- Dense cells env: `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1` (+ `ulimit -n 65536`; stock-ETL fd-leak note in `../CONTEXT.md`).
- Legacy cells env: legacy PYTHONPATH only; no `EVOGP_*`/`ETL_*` vars (inert).
- Type-2 protocol: `bench_gp.py --task eval-pure --mode dense --backend iree --reps 5 --warmup 10 --invocations 100 --e2e-companion 11 --max-cold-builds 400 --output-checksum` (dense side) — see `../r4/`; triton type-2 = tight build-once/run-many loop on the same primitives (`test/benchmark/pure_forward.py`: `time_block`, `summarize`, `PureRunner`, `digest_blocks`).
- Same-window anchors (r5 `../r5/results/q4_window/`): legacy 0.6461/0.6372 ms; dense-iree e2e 5.3823/5.7912 ms; dense pure 1.2142/1.4612 ms (f64/f200).
- Records: one JSON + one log per cell with provenance (`tag, side, task, fixture, env, gpu, evogp_commit, evogp_file, route, ...`); immutable — fresh `results/<dir>/` per run; quote medians with the ±10 % shared-box band (contention ⇒ 2–3× inflation).
- Gate patterns to copy: `../r4/check_r4.py`, `../r5/check_r5.py` (stdlib-only, GPU-free, recompute headlines from stored arrays).

## Key facts / expectations (from the session's feasibility probe; do not re-derive)
- Probe-verified working: runtime-bound loops, data-dependent scalar branches, masked 2D grids, `triton.language.extra.cuda.libdevice` math, atomics/`tl.sum`, `tl.static_range` unrolling, 12+ kernel args, gather/scatter.
- Compiler trap: per-lane vector `if` aborts the compiler process (SIGABRT, uncatchable) — use scalar conditions or `tl.where`; during kernel development, compile new kernels in a subprocess.
- Compile economics: interpreter-style kernel ≈0.3 s; unrolled per-node ≈6 ms/node; recompiles only on new constexpr/signature; in-process relaunch free; launch ≈28 µs; `torch.cuda.CUDAGraph` capture of Triton works (optional).
- Warm micro-bench of a 64-tree × 63-node interpreter skeleton: 27.6–36.2 µs/call (num_warps=4, one program per tree) — expect a triton warm call to be launch/host-bound, not kernel-bound; measure and decompose both.
- Legacy interpretation caveat: legacy warm per-call is ≈flat from pop 64→1000 (0.604→0.603 ms) ⇒ largely a fixed host-side component (auto-dispatch, 2 launches, memset, alloc, neg); do not read 0.6 ms as the legacy kernel's cost; a light kineto decomposition of the legacy call is a P1 nicety.
- Integration context for SUMMARY Q4: placement (a) EvoGP-side runner replacing the dense executor body for single-output SR (recommended, no ETL interop) vs (b) ETL-side `external_call` + `register_external_kernel(device_resident=True)` — on iree 3.11/cuda the device payloads have no public raw-pointer/DLPack export (D2D + private HAL binding needed) and ETL has no fusion/codegen infrastructure and no triton usage anywhere; summarize only, do not implement.

## Routing Table
- (planned; update as artifacts land) `triton_eval.py` → Triton fused evaluator + build/staging; `run_triton_eval64.py` → type-1 cell driver (patched seam); `validate_triton.py` → oracle/goldens validation; `run_cells.sh` → the same-window campaign runner: cell table `check|legacy|dense|triton|pure|all` with stable ids `check_deps leg_f64 leg_f200 dense_f64 dense_f200 triton_f64 triton_f200 densepure_f64 densepure_f200 tritonpure_f64 tritonpure_f200`, flags `--group/--only/--window/--timeout/--list/--dry-run`, one fresh process per cell serialized through `run_under_lock.py` with `timeout` inside the lock, records into a fresh `results/<window>/` (`<cellid>.json` + `<cellid>.log`, plus `check_deps.txt`/`check_deps.log` and `window_meta.txt`), `--group all` = the 10 measurement cells in table order; `results/` → immutable campaign records; `check_r6.py` → the stdlib-only gate; `SUMMARY.md` → verdict; `README.md` → protocol + repro commands; `RUNLOG.md` → operator log.
- Read-only dependencies: `../r5/RCA.md`, `../r4/` (protocol), `../goldens/` + `../fixtures/`, `../../src/evogp/tree/{dense_engine.py,forest.py,ENGINE_SPEC.md,RUNNER_HOOKS.md}`, `../../src/evogp/tests/reference_eval.py`.

## Notes for Agents
- Never edit `SPEC.md`, committed r2/r4/r5 records/harnesses, `bench_gp.py`, `pure_forward.py`, `fixtures/`, `goldens/`, anything under `src/`, or the foreign repos; a new round gets its own copies.
- `python`/`python3` on PATH lack torch — always use the project venv; run everything from the repo root with `PYTHONPATH=src`.
- Keep files <~1000 lines; scratch in `$TMPDIR`; no giant binary artifacts in the repo.
- Commit after each phase; update this file whenever an interface/semantic decision lands so nested agents inherit it.
- Validation sequence for the round: `check_r6.py` (GPU-free gate) → numeric validation script (GPU, under lock) → spot-check records vs `SUMMARY.md`.
