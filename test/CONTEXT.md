# `test/` — dev tooling (a numeric oracle + one evoxc probe) and the archived records of the pre-2.0.0a2 stack
## Intent
This directory holds the two things that deliberately are **not** part of the `evogp` package: the independent numerics oracle the fused route is checked against (`./oracle/`), and the frozen performance evidence of the deleted torch stack (`./archive/`), plus one surviving probe (`./evoxc_jit_probe.py`, evoxc-only economics).
It is dev tooling, **not** a pytest suite: no shipped runtime module imports it — only the dev gates do, by appending `test/oracle` to `sys.path` (`src/evogp/fused/check.py`, `src/evogp/fused/tests/conftest.py`) — and `pytest test` collects nothing (the harness's two torch-free test modules were deleted with the harness).
`./oracle/` and the standalone `./evoxc_jit_probe.py` are the two things here that run today; the oracle runs as a *consumer* of the library's own gates — `src/evogp/fused/tests/` and `python -m evogp.fused.check`.
Everything under `./archive/` is read-only evidence: its producing code was deleted with the torch stack and is recoverable at commit `6720723`.
The shipped-stack benchmarks are library code now, not `test/` code: `src/evogp/gp/measure_popscale.py`, `src/evogp/gp/measure_generation.py`, `src/evogp/gp/measure_fused.py` and `python -m evogp.fused.check --stage all`.
Their reports of record are `src/evogp/gp/POPSCALE.md`, `src/evogp/gp/DEVICE_LOOP.md` and `src/evogp/fused/FUSED.md`.
## API Surface / tree
- `./oracle/` → the independent plain-torch evaluator `reference_eval.py` + `goldens/` (7 frozen `.npz` cases); DEV-only, imported by no library module, added to `sys.path` by its dev consumers; described in `./oracle/CONTEXT.md`.
- `./archive/` → the records of record of the pre-2.0.0a2 tree: markdown reports, raw JSON timing records, run logs and the rounds' captures, in the old `benchmark/` layout (`archive/benchmark/{evoxc,r2,r4,r5,r6,legacy,new,xval,fixtures}`); entry page `./archive/README.md`, which indexes every report of record (the counts, the provenance and the index live there, not here).
- `./evoxc_jit_probe.py` → standalone `evoxc` build/run economics probe (imports `evoxc` + numpy, never `evogp`; a docstring table plus its own PASS/FAIL checks, exit 1 on failure).
- Nothing else. There is no `fixtures/`, no `goldens/`, no notebook and no harness script at this level.
## Run commands (exact)
Interpreter is the project venv — `python`/`python3` on `PATH` lack torch, and the fused legs need one; `evoxc` comes from the sibling checkout `/root/evoxc` and `evox` from `/root/evox/src`.
```bash
PY=/tmp/rocvenv/bin/python          # py3.12, torch 2.13.0+rocm7.1, triton, evoxc from /root/evoxc
export PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1

# 1. the oracle's only real gate — the fused route's golden stage (reads test/oracle/goldens): 6/6
$PY -m evogp.fused.check --stage goldens

# 2. the oracle's pytest consumer (parity / multi-output / contract): 45 passed
$PY -m pytest src/evogp/fused/tests -q

# 3. the surviving probe, numpy legs (~8 s; sections b,d,e and the iree-cuda legs of f,g need an NVIDIA driver)
EVOXC_TESTS=/root/evoxc/tests $PY test/evoxc_jit_probe.py --sections a

# 4. the torch-free population-scale lanes of the shipped stack (not test/ code — quoted here because this
#    directory is where performance questions used to be answered); lanes are numpy|gp|gp-fused|torch-batched|torch-per-tree
$PY -m evogp.gp.measure_popscale --pops 64,256 --lanes numpy,gp --rows 256 --reps 3 --warmup 1
```
Measured on the reference box at this commit, for the commands above:
- `--stage goldens`: `GOLDENS: 6/6 comparisons inside 1e-4`, exit 0 — oracle-only, because `test/goldens/` (the CUDA-era captures) is gone.
- `pytest src/evogp/fused/tests -q`: `45 passed in 10.02s`.
- `evoxc_jit_probe.py --sections a`: all numpy checks pass (first `build_cached` 28.868 ms, cache hit 0.255 ms, distinct-tree build 7.428 ms, run N=1000 0.408 ms/tree, max err 1.439e-07 vs the numpy reference).
- `measure_popscale --lanes numpy,gp`: `gp` reads warm 0.570 ms `evaluate` at pop 64 / rows 256 (cold 2 654.801 ms, the compile) and 0.555 ms at pop 256 (cold 2 597.008 ms), i.e. 3.62× / 10.46× the `numpy` lane's 2.062 / 5.806 ms; parity `gp.evaluate` vs `numpy.evaluate` max_rel 2.09e-07 (pop 64) and 4.27e-07 (pop 256), 0 finiteness mismatches.
## Constraints
- `./archive/` is evidence: never edit a number, never re-render, never re-gate, and never "fix" a record against today's engine.
- Its reports name paths as `test/benchmark/...`; the same file now lives at `test/archive/benchmark/<same>`, and the `*.py`/`*.sh` they cite were deleted (recover at `6720723`).
- `./oracle/` may import `torch` and is the one directory where that is correct; it must import nothing from `evogp`, so it can catch the library's bugs instead of inheriting them.
- No new dependency, no build step and no compiled extension: everything here is plain Python over the venv's torch/evoxc/numpy.
- There is no CI and no pytest ini in the repo; nothing in this directory is collected by `pytest src/evogp/...`.
## Known Issues
- `evoxc_jit_probe.py`'s iree-cuda legs cannot be green on a ROCm box: iree reports `CUDA driver library 'libcuda.so' not available`, so `--sections abcdefgh` fails in (b)/(d)/(e) and in the iree-cuda parts of (f)/(g) while every numpy leg passes.
- Its section-(f) DLPack expectation is stale: it still expects `torch.from_dlpack` of an `evoxc` device tensor to be refused, and prints `UNEXPECTEDLY ACCEPTED`, because `/root/evoxc` `474bf72` added the native device export the `evogp.gp` fused objective depends on.
- Its section-(g) source path comes from `$EVOXC_TESTS` (default `/root/evoxc/tests`, the local checkout); on a host without that checkout the section fails at the copy step.
- The archive cannot re-derive anything: the R2/R3/R4 `etl`-shadow halves, the legacy CUDA build recipe (`/tmp/evogp_legacy_bench`, CUDA 12.6 nvcc) and every harness gate are gone with the code, and several stored records embed absolute paths from other session worktrees.
- `src/evogp/fused/CONTEXT.md` still quotes the goldens leg as `10/10`; with `test/goldens/` deleted the same stage reads `6/6` (the oracle fixtures alone). Those files are outside this node.
## Routing Table
- `./oracle/` → The independent plain-torch numeric oracle and its 7 frozen fixtures; the reference behind `src/evogp/fused/tests/` and `evogp.fused.check --stage parity|goldens|multi`.
- `./archive/` → Read-only records of record of the pre-2.0.0a2 tree (torch stack + the four `EVOGP_EVAL_MODE` routes), with `./archive/README.md` as the single entry point and an index of every report of record.
- `./evoxc_jit_probe.py` → The standalone evoxc build/run economics probe (no `evogp` import).
## Notes for Agents
- Performance questions about the *shipped* stack are answered in `../src/evogp/gp/` and `../src/evogp/fused/`, never here; read `src/evogp/gp/POPSCALE.md` / `DEVICE_LOOP.md` / `src/evogp/fused/FUSED.md` and run the `measure_*` scripts.
- Need a number from the old four-route engine? Read it out of `./archive/` and quote it with its provenance (pre-2.0.0a2 tree, commit `6720723` for the code) — do not present it as a current measurement.
- Numeric disagreements go to `./oracle/reference_eval.py`: it is the semantics of record for op placement, NaN/±inf and the multi-output accumulation rule, and it is slow by design.
## See Also
Sibling paths (read-only from this node — escalate any needed write to the parent `./`): `../src/evogp/` (the shipped library: `gp/`, `fused/`, `evox_ext/`), `../example/`, `../tutorial/`.
Foreign repositories: `/root/evoxc` (EvoX Compiler — the evaluation engine, read by `evoxc_jit_probe.py`) and `/root/evox` (EvoX 2.0, whose idiom `evogp.gp` follows).
