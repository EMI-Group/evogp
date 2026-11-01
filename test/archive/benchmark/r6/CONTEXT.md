> **ARCHIVED at the EvoGP 2.0.0a2 pre-release.** The prototype kernels and checkers this file routes to were deleted with the torch stack; `SUMMARY.md` and the round's records stay here.
> Numbers are as measured on the pre-2.0.0a2 tree; the code is recoverable at commit `6720723`.
> Entry point: `test/archive/README.md`.
# `test/benchmark/r6/` — Triton fused-kernel evaluation prototype (feasibility + measurement round)

## Intent
Round 6 tested the r5 RCA hypothesis that a custom kernel replicating the legacy CUDA engine's fused design restores legacy-class performance for EvoGP's single-output SR case; the answer is yes and exceeded (`SUMMARY.md` §10/Q5).
Context of record: `../r5/RCA.md` (§9) — the ≈8–11× gap decomposes into per-call host glue (≈4.2–4.3 ms of the ~5.4–5.8 ms dense call) plus a dispatch-bound 310-op dense graph (249 dispatches + 61 DtoD copies, 125-deep chain, ~1.4 µs kernels) versus the legacy engine's 3 device ops (memset + fused tree-walk kernel + averaging kernel).
This round is feasibility + measurement: the prototype lives in `test/` (this node) only — no `src/` changes, no foreign-repo writes, no edits to committed r2/r4/r5 artifacts or shared harnesses.
A decisive negative (documented no-go with specific evidence) is an acceptable outcome; fabricated numbers are not.

## Status
- **Round complete** — all six deliverables landed: prototype (`triton_eval.py` + `run_triton_eval64.py` + `validate_triton.py`), same-window campaign, P1 extras, `SUMMARY.md` (analysis of record), `check_r6.py` (gate), `README.md` + this file.
- Campaign complete: one continuous session (2026-09-14, GPU 6); primary window **`results/w1d_20260914/`** (10/10 cells exit 0, deps pre-flight `READY (20 ok / 0 warn / 0 missing)`, every median inside −11.1 %…+9.9 % of the r5 `q4_window` anchors); **`results/w1c_20260914/`** is the superseded anomaly window (whole-window inflation, kept as evidence, never edited).
- Headline warm medians (w1d, ms, f64/f200): legacy e2e **0.6364/0.6269**; dense-iree e2e **4.7860/6.3179** (7.52×/10.08× legacy); triton e2e **0.1314/0.1379** (36.4×/45.8× faster than dense, 4.8×/4.6× faster than legacy); dense pure **1.3339/1.4796**; triton pure **0.05470/0.05472** per invocation (24.4×/27.0× faster than dense pure).
- All four triton cells: counter gate PASSED (`triton_call_count`/`triton_runs` exact, `fallbacks=0`, `builds=1`, `cache_hits=24`); triton pure = exactly 1 kernel launch/invocation (device 5.8/9.1 µs vs host residual 50.4/45.5 µs ⇒ launch/host-bound).
- Gate `check_r6.py` green: **14 records / 1440 checks / 0 failures** (+ 10 anomaly cells, 5 kernel-host fact sheets, 2 window files, 30 cell logs, 1 RUNLOG).
- P1 extras under `results/w1d_20260914/extra/`: `kernel_host/` kineto decomposition via `extra_trace.py` (legacy = 3 device ops of substance 6.4/11.7 µs vs 617/576 µs host residual); `compile_econ/` warm-vs-fresh `TRITON_CACHE_DIR` (first call 732.7 ms warm / **1785.4 ms cold-cache**, medians unaffected); re-staging counterfactual (`--e2e-companion` − pure: triton 0.084/0.086 ms vs dense 7.26/7.04 ms).
- `results/w1d_20260914/**`, `results/w1c_20260914/**`, `results/validation/**` are **immutable evidence**; a re-run uses a new window label (`run_cells.sh` freshness gate enforces one write per label).

## Deliverables (the round's acceptance criteria)
1. Prototype: `triton_eval.py` + drivers — a Triton fused evaluator for single-output SR consuming the same staged inputs as the dense route; one compile per shape; legacy-style runtime node interpreter; target 1–2 launches/call; zero per-call uploads/readback; build-once/run-many.
2. Numeric validation: triton vs `src/evogp/tests/reference_eval.py` (plain-torch oracle) and vs the `test/goldens/` legacy captures, on fixture64 + fixture200; rtol/atol 1e-4, equal_nan; report exactness (+ − * bit-equal; / last-ulp) and declare op coverage explicitly (fixture ops {+,-,*,/} required; full op table stretched to all 31 dispatch entries and met).
3. Same-window measurement campaign (one continuous window; GPU 6; one fresh process per cell; fcntl-serialized): legacy f64+f200, dense-iree f64+f200, triton f64+f200 — type-1 evaluate envelope; type-2 pure-forward for dense + triton; triton first-call/compile economics + cache behavior; immutable JSON+log records with provenance in `results/<window>/`; operator log.
4. `SUMMARY.md`: per-call decomposition (triton kernel-vs-host; light legacy decomposition if cheap) + verdict answering Q1–Q5: Q1 semantics correct? Q2 how fast vs legacy/dense same-window? Q3 what remains of the gap + what would close it? Q4 coverage/limits + integration path + effort estimate? Q5 does the evidence support "same implementation ⇒ legacy-class performance"? (yes/no + the flat-host-cost and 3-op-vs-310-op caveats).
5. `check_r6.py`: stdlib-only, GPU-free gate recomputing every headline from stored records + validating envelope/provenance identities (25 calls, call 0 cold, median warm, routes, provenance fields); house standard ≈ hundreds of checks / 0 failures — delivered at 1 440 checks.
6. Docs: `README.md` (protocol, session-scoped env/CPATH fix, exact repro commands), this file kept current; parent `../CONTEXT.md` is updated by the parent agent, not from here.

## Known Issues / caveats
- **`RUNLOG.md` §6.5 prose error (kept, not edited — log immutability).** It says the fresh-cache "call 1 is 1.39 ms"; the records' call 1 is 0.5302 ms (warm extra), 0.5657 ms (campaign), 0.5428 ms (fresh cache) — the same RUNLOG's §4b table already prints 0.5428. Trust the records; `SUMMARY.md` §8 documents the record-vs-prose note.
- **Validation run 2 is an uncommitted kernel revision.** `results/validation/run2/validation.json` declares `triton_eval.py` sha256 `e155dd13…`, which exists in no git object; run 1 (`results/validation/validation.json`) pins the committed file (`163b5ffd…`) and is the **record of record** for the committed kernel. Asserted by `check_r6.py` (`provenance:run2_*` + NOTE); operator story in `RUNLOG.md` §8.
- **`densepure_*` provenance drift (report upward).** The r6 pure records carry `pure.runner_source="frozen-api:build_dense_runner"` / `compile.path="frozen"` with `build_ms`/`decode_plan_ms` null — not the `local-fallback:*` label the r4 node text implies; the records are what the installed harness emits (`SUMMARY.md` §11.4).
- **Four un-landed deviation classes** (FTZ denormals, transcendental last-ULP, `neg` zero sign, CONST −0.0 staging) — all fixture-irrelevant, candidate fixes enumerated in `SUMMARY.md` §11.2.

## Environment (session-scoped; verify, don't assume)
- Python: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2, torch 2.6.0+cu124, triton 3.2.0 with its own ptxas); no `python`/`python3` on PATH with torch.
- Triton runtime compile needs `Python.h`, missing at `/usr/include/python3.11`; working fix (~5 s, no root): download the `libpython3.11-dev` deb into `$TMPDIR`, `dpkg-deb -x` it, then `export CPATH="$TMPDIR/py311hdr/usr/include/python3.11:$TMPDIR/py311hdr/usr/include"` (both entries required); a ready header tree exists at `/tmp/triton_feas/py311hdr` — reuse if still present.
- Set `TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache` for reproducible compile caching; keep the cache out of the repo (checksum the cache only if useful).
- GPU: `CUDA_VISIBLE_DEVICES=6`; never GPU 7 (hardware ECC fault); check `nvidia-smi` before every timed run; GPU 6 may carry a co-tenant's committed memory (idle at times) — never kill foreign processes, just measure and note contention.
- `flock` *binary* hangs in this sandbox — serialize ALL GPU work with a Python `fcntl.flock(LOCK_EX)` helper on `/tmp/gpbench_critical.lock`; never nest locks; keep `timeout` inside the lock.
- The node's `run_under_lock.py` implements that wrapper; usage: `run_under_lock.py timeout -s INT -k 30 <cap> <cmd...>`.
- Legacy engine (read-only): `/tmp/evogp_legacy_bench`; run with `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`; rebuild recipe in `../legacy/BUILD_NOTES.md` + `../legacy/setup_legacy.sh` if wiped.
- Multi-minute phases: launch via `systemd-run --user --unit=<name> --collect -p LimitNOFILE=65536 --setenv=CUDA_VISIBLE_DEVICES=6 bash -c '…' "$PWD"` — `nohup`/`setsid`/`disown` backgrounding inside a tool call dies with the call's cgroup (two early campaign attempts were reaped this way).
- Disk: `/tmp` (== `/`) is ~97 % full (~15 GB free) — keep outputs small, no giant binary logs.

## Contracts & semantics (single-output SR)
- Output contract: positive `(pop,)` float32 loss; the caller negates (`forest.SR_fitness` → `SymbolicRegression.evaluate`); `problem.evaluate(forest)` is the public API the benchmark times.
- Forest schema: `(pop, max_tree_len)` prefix (pre-order) encoded; `batch_node_value` f32, `batch_node_type` i16, `batch_subtree_size` i16; only `[0, subtree_size[0])` meaningful; arity from base type (VAR=0/CONST=1 leaves, UFUNC=2, BFUNC=3, TFUNC=4); `MAX_STACK=1024`; node 0 = root.
- Op semantics of record: `src/evogp/tree/ENGINE_SPEC.md` §3; `/` = `var2==0 ? NaN : var1/var2` (select, not trap); `+ - *` f32-exact; NaN/±inf propagate, never sanitize.
- Dense-route staging: `src/evogp/tree/dense_engine.py` `prepare_dense(...)` → 7 `(L,P)` columns `[opcode, func_id, const_value, var_index, child1, child2, child3]` (+ `tree_len`); the prototype reuses it for byte-identical staging (declared).
- Legacy blueprint (`b613503`, `src/evogp/cuda/forward.cu`): reversed-prefix prefix-row walk where the same array doubles as the operand stack (children before parents; arity from base type; no child indices); op dispatch by function id; fitness path = `cudaMemsetAsync` + per-(tree,datapoint)-thread fused kernel (shared-mem reduction + atomicAdd) + averaging kernel = 3 device ops; no per-structure caching, recompute every call, no readback.
- Type-1 triton envelope: runtime-patch the dense seam `evogp.tree.dense_engine.sr_fitness_dense` (module-attribute binding; call site `src/evogp/tree/forest.py` ~L338-361) with `EVOGP_EVAL_MODE=dense`, then drive the unmodified public API `problem.evaluate(forest)`; the patched function returns exactly what `sr_fitness_dense` returned (positive loss).
- Fixtures: fixture64 = `test/goldens/forest_random.npz` (64 trees, N=64 rows, x (64,2), labels = x0²−x1); fixture200 = `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, same 64 rows bitwise); loader `test/benchmark/bench_gp.py::build_eval_case` (~L768-819).

## Protocol conventions (mirror r4/r5)
- Type-1 cells: 25 sync-bracketed `problem.evaluate(forest)` calls through the unmodified public API (triton cells via the patched seam); call 0 cold; `median_warm_ms = median(calls 1..24)`; one fresh process per cell.
- Dense cells env: `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1` (+ `ulimit -n 65536`; stock-ETL fd-leak note in `../CONTEXT.md`).
- Legacy cells env: legacy PYTHONPATH only; no `EVOGP_*`/`ETL_*` vars (inert).
- Type-2 protocol: 5 timed blocks × 100 resident invocations after a 10-invocation warmup + an 11-call `--e2e-companion` block; triton type-2 = tight build-once/run-many loop (`run_triton_eval64.py --mode pure`).
- Same-window anchors (r5 `../r5/results/q4_window/`): legacy 0.6461/0.6372 ms; dense-iree e2e 5.3823/5.7912 ms; dense pure 1.2142/1.4612 ms (f64/f200); both windows' r6 cells land inside the shared-box band around them.
- Records: one JSON + one log per cell with provenance (`tag, side, task, fixture, env, gpu, evogp_commit, evogp_file, route, ...`); immutable — fresh `results/<dir>/` per run; quote medians with the ±10 % shared-box band (contention ⇒ 2–3× inflation).

## Key facts / expectations (from the session's feasibility probe; do not re-derive)
- Probe-verified working: runtime-bound loops, data-dependent scalar branches, masked 2D grids, `triton.language.extra.cuda.libdevice` math, atomics/`tl.sum`, `tl.static_range` unrolling, 12+ kernel args, gather/scatter.
- Compiler trap: per-lane vector `if` aborts the compiler process (SIGABRT, uncatchable) — use scalar conditions or `tl.where`; during kernel development, compile new kernels in a subprocess.
- Compile economics: interpreter-style kernel ≈0.3 s; unrolled per-node ≈6 ms/node; recompiles only on new constexpr/signature; in-process relaunch free; launch ≈28 µs; `torch.cuda.CUDAGraph` capture of Triton works (optional).
- Legacy interpretation caveat (now measured, `extra/kernel_host/`): legacy warm per-call is ≈flat from pop 64→1000 ⇒ largely a fixed host-side component (617/576 µs residual vs 6.4/11.7 µs of device ops); do not read 0.6 ms as the legacy kernel's cost.
- Integration context for SUMMARY Q4: placement (a) EvoGP-side runner replacing the dense executor body for single-output SR (recommended, no ETL interop) vs (b) ETL-side `external_call` + `register_external_kernel(device_resident=True)` — on iree 3.11/cuda the device payloads have no public raw-pointer/DLPack export (D2D + private HAL binding needed) and ETL has no fusion/codegen infrastructure and no triton usage anywhere; summarized only, not implemented.

## Routing Table
- `triton_eval.py` → the Triton fused evaluator + build/staging (877 lines); `run_triton_eval64.py` → type-1 cell driver (patched seam, counter gates) + `--mode pure` type-2 loop; `validate_triton.py` → oracle/goldens/per-op validation (writes `results/validation/**`); `extra_trace.py` → P1 extras probe (kineto kernel-vs-host decomposition, `--target triton-pure|triton-e2e|legacy-e2e`).
- `run_cells.sh` → the same-window campaign runner: cell table `check|legacy|dense|triton|pure|all` with stable ids (`check_deps leg_f64 leg_f200 dense_f64 dense_f200 triton_f64 triton_f200 densepure_f64 densepure_f200 tritonpure_f64 tritonpure_f200`), flags `--group/--only/--window/--timeout/--list/--dry-run`, one fresh process per cell serialized via `run_under_lock.py` (`timeout` inside the lock), records into a fresh `results/<window>/`; `run_under_lock.py` → the `fcntl.flock` wrapper for every GPU phase.
- `check_r6.py` → the stdlib-only, GPU-free gate (`python3 test/benchmark/r6/check_r6.py` → 14 records / 1440 checks / 0 failures; also 10 anomaly cells, 5 kernel-host fact sheets, 2 window files, 2 deps files, 30 cell logs, 1 RUNLOG); ~1 831 lines by design — per-schema recomputation over 10 primary cells × 5 record shapes + anomaly + validation + fact sheets (header comment explains).
- `SUMMARY.md` → the analysis of record (verdict Q1–Q5, per-call decomposition, compile economics, anomaly window, open items, artifact index); `README.md` → protocol/repro entry point (session-scoped env fixes, exact commands, immutability rules); `RUNLOG.md` → the operator log of the executed campaign.
- `results/w1d_20260914/` (primary window: 10 cells + `extra/{kernel_host,compile_econ}`) + `results/w1c_20260914/` (superseded anomaly window) + `results/validation/` (+ `run2/`) → immutable records.
- Read-only dependencies: `../r5/RCA.md`, `../r4/` (protocol), `../goldens/` + `../fixtures/`, `../../src/evogp/tree/{dense_engine.py,forest.py,ENGINE_SPEC.md,RUNNER_HOOKS.md}`, `../../src/evogp/tests/reference_eval.py`.

## Notes for Agents
- Never edit `SPEC.md`, committed r2/r4/r5 records/harnesses, `bench_gp.py`, `pure_forward.py`, `fixtures/`, `goldens/`, anything under `src/`, or the foreign repos; a new round gets its own copies.
- `python`/`python3` on PATH lack torch — always use the project venv; run everything from the repo root with `PYTHONPATH=src`.
- Keep files <~1000 lines (the gate is the documented exception); scratch in `$TMPDIR`; no giant binary artifacts in the repo.
- Commit after each phase; update this file whenever an interface/semantic decision lands so nested agents inherit it.
- Validation sequence for the round: `check_r6.py` (GPU-free gate) → numeric validation script (GPU, under lock) → spot-check `SUMMARY.md` vs records; a re-run needs fresh window labels and the session-scoped deps re-checked (`README.md`).
