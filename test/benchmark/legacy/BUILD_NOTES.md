# `test/benchmark/legacy/BUILD_NOTES.md` — legacy (pre-ETL, custom-CUDA) checkout: build + smoke record

Side **OLD** of `../SPEC.md` (legacy CUDA engine vs ETL "true jit" engine). Everything below was
executed by hand on the shared box on **2026-09-11 (HKT)**; raw output is committed under
`results/logs/`. Driver: `./setup_legacy.sh` (idempotent; safe to re-run).

## TL;DR status

| item | value |
|---|---|
| materialize | OK — `git archive b613503` into `/tmp/evogp_legacy_bench` (no `.git` there, by design) |
| build | OK — first try after the env recipe below, **67 s** wall (20:59:53 → 21:01:00) |
| extension | `/tmp/evogp_legacy_bench/src/evogp/evogp_cuda.cpython-311-x86_64-linux-gnu.so` — 1 112 488 B, sha256 `d7af153c4d12843dc218e3616985173db9e294c3fc3a04f9dbab457b30d715a4` |
| smoke import | OK — `import evogp.tree; import evogp.evogp_cuda` → `ok` |
| cu-scale smoke (`evogp.sr_test`, shipped defaults) | **100/100 generations, exit 0, no timeout**; 8 s wall; steady median **10.90 ms**/gen, steady max 15.73 ms, gen 0 (cold) 572.35 ms; best fitness **-0.2257** |
| GPU | **device 4** (RTX A6000) — device 7, the idle first choice, is **hardware-faulted** (ECC), see "Gotchas" |
| artifact left in place | yes — the built checkout stays at `/tmp/evogp_legacy_bench` for `run_matrix.py`/later agents |

## 1. Materialization

Executed from the EvoGP repo root (the archive is written by this repo's git, the destination is a
plain directory — `/tmp/evogp_legacy_bench` is **not** a git checkout afterwards):

```bash
mkdir -p /tmp/evogp_legacy_bench && git archive b613503 | tar -x -C /tmp/evogp_legacy_bench
```

* Full commit id: `b613503c3107db23222f69bfaad2a329639f28fe` — *"Update CONTEXT.md for ./src/evogp/problem"*,
  2026-09-11 12:21:16 +0800. Last revision before the kernel-removal rewrite.
* Sanity checks done after extraction: `src/evogp/cuda/forward.cu` (+ `generate.cu`, `mutation.cu`,
  `torch_wrapper.cu`, `defs.h`, `kernel.h`) and `setup.py` containing
  `CUDAExtension(name="evogp.evogp_cuda", …)` are present. `setup_legacy.sh` re-asserts both and
  fails loudly if they are missing.
* The build produces `evogp/evogp_cuda.<abi>.so` **directly under `src/evogp/`** (that is where
  `src/evogp/tree/__init__.py` picks it up), plus a `build/` tree (6.4 MB total). A partial `build/`
  left by an interrupted attempt was deleted before the successful build.

## 2. Environment recipe (known good; reproduced exactly by `setup_legacy.sh`)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python          # py3.11.2, torch 2.6.0+cu124
CUDA_ENV=/opt/miniforge3/envs/wenet_tiktoken             # CUDA 12.6 toolchain (system nvcc 11.8 is rejected by torch cu124)
export CUDA_HOME=$CUDA_ENV/targets/x86_64-linux
export PATH=$CUDA_ENV/bin:$CUDA_ENV/targets/x86_64-linux/nvvm/bin:$PATH   # cicc/ptxas live in nvvm/bin
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"          # /usr/bin/g++ -> g++-12 (CUDA 12.6 allows <= 13)
export CPATH=/tmp/evogp_t1a263/pydev/usr/include/python3.11:/tmp/evogp_t1a263/pydev/usr/include:/tmp/evogp_t1a263/pydev/usr/include/x86_64-linux-gnu
export CPLUS_INCLUDE_PATH=$CPATH
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8                                        # see gotcha 3: setup.py overrides this
```

Toolchain actually used (copied from the run log / `nvcc --version`):

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
   nvcc path : /opt/miniforge3/envs/wenet_tiktoken/bin/nvcc  (-> targets/x86_64-linux/bin/nvcc)
   host cc   : /usr/bin/g++ -> g++-12 (Debian 12.2.0) via NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"
   python    : /mnt/local-ssd/bchuang/evox/.venv/bin/python (Python 3.11.2), torch 2.6.0+cu124
```

Why the fallbacks: `/usr/include/python3.11/Python.h` does **not** exist on this box (no
python3.11-dev), so the recovered headers under `/tmp/evogp_t1a263/pydev/usr/include/python3.11` are
put on `CPATH`; and `siteutils/` (`/tmp/evogp_t1a263/siteutils`, sklearn 1.7.2 + joblib +
threadpoolctl + cloudpickle) is what `evogp.problem` needs at *run* time — there is no internet, so
nothing can be installed.

## 3. Build

```bash
cd /tmp/evogp_legacy_bench && $PY setup.py build_ext --inplace     # 4 TUs: torch_wrapper/generate/mutation/forward .cu
```

* Runs under one `flock -w 7200 /tmp/gpbench_critical.lock bash -c '<cmd>'` (host-wide lock shared
  with the NEW-engine side of the benchmark; never nested).
* Result: 20:59:53 start → `.so` copied into `src/evogp` at 21:00 → smoke at 21:00:58 → **67 s** end-to-end.
* Compiler output (18 KB, includes the `ptxas -v` register/spill report and the full link line):
  `results/logs/build_20260911_2059.log`. Full driver run: `results/logs/setup_legacy_build_run.log`.
* All kernels build for `sm_86` (`TORCH_CUDA_ARCH_LIST=8.6`); `-use_fast_math -maxrregcount=32` are
  hard-coded in `setup.py` (the fast-math origin of the golden-capture tolerance, see
  `../../goldens/README.md`).

## 4. Smoke imports

```bash
# (a) import-only smoke — phase (d) of setup_legacy.sh, one flock, ~2 s
cd /tmp/evogp_legacy_bench && PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils \
    $PY -c "import evogp.tree; import evogp.evogp_cuda as m; print('ok')"      # -> ok, exit 0

# (b) cu-scale smoke — the shipped install-validation script, unmodified, under the shared lock
CUDA_VISIBLE_DEVICES=4 flock -w 7200 /tmp/gpbench_critical.lock bash -c \
  "cd /tmp/evogp_legacy_bench && CUDA_VISIBLE_DEVICES=4 \
   PYTHONPATH=src:/tmp/evogp_t1a263/siteutils timeout 1500 $PY -u -m evogp.sr_test"
   # -> exit 0 (no timeout), log: results/logs/sr_test_defaults.log
```

`sr_test.py` at `b613503` is the shipped defaults (no CLI args, work starts at import time):
`torch.random.manual_seed(0)` + `torch.cuda.manual_seed(0)`; `SymbolicRegression(func, num_inputs=2,
num_data=1000, lower_bounds=-5, upper_bounds=5)`; `GenerateDescriptor(max_tree_len=128, …,
max_layer_cnt=5)`; `pop_size=1000`; `mutation_rate=0.2`; `survival_rate=0.3, elite_rate=0.01`;
`generation_limit=100`; `is_show_details=True` (the per-generation `Cost time: Xms` line *is* the
timing trace). `-u` keeps the output unbuffered so a timeout would still leave usable partial data.

## 5. cu-scale smoke result (real captured output — also a C2-size datapoint)

All 100 generations ran (indices 0–99), the run ended with `Generation limit reached!`, **exit code 0**
(no `timeout` kill). Wall clock of the whole locked command: **8 s** (a few seconds waiting on the
lock — it was recorded as held by the sibling at start — plus ~1.5 s interpreter/model import,
dataset generation, and 1.60 s of summed generation time).

| metric | value |
|---|---|
| generations completed | 100 / 100 (gen 0 … gen 99) |
| per-generation time, generation 0 (initial population — cold CUDA module load / first `cudaMalloc`) | **572.35 ms** |
| per-generation time, generations 1–99 (n=99) | **median 10.90 ms**, mean 10.40 ms, min 9.00 ms, **max 15.73 ms** |
| per-generation time over all 100 | median 10.91 ms, max 572.35 ms |
| summed generation time | 1602.1 ms (1.60 s) |
| best fitness (max of the per-generation population maxima) | **-0.2257**, first reached at generation 2 and held to generation 99 (gen 0/1: -0.2511) |
| traceback / warnings | none |

The per-generation fitness line itself is spiky (`min`/`mean` reach -1e7 because deep random trees
divide by near-zero constants — expected for this problem); `valid cnt` counts fitnesses inside the
pipeline's ±1e8 validity boundary (944 at gen 0, ≈990 later).

Host was shared while measuring (other tenants on GPUs 0–6, see the snapshot below), so treat these
numbers as "captured on the shared box", not as a quiescent-machine benchmark; the sibling's
`run_matrix.py` owns the normalized matrix.

## 6. GPU used + `nvidia-smi` snapshot

**Device 4** was used, not the idle-looking device 7 (see gotchas). Snapshot at run start
(2026-09-11 21:02:04 HKT, full record in the head/tail of `results/logs/sr_test_defaults.log`):

```
index, utilization.gpu, memory.used, memory.total, temperature.gpu, power.draw
0, 42 %, 1119 MiB, 49140 MiB, 64, 199.08 W
1, 100 %, 7152 MiB, 49140 MiB, 70, 292.64 W
2, 27 %, 17702 MiB, 49140 MiB, 64, 182.12 W
3, 41 %, 4564 MiB, 49140 MiB, 55, 189.43 W
4,  1 %, 4238 MiB, 49140 MiB, 36,  73.66 W     <-- used (least busy: ~0 % util, ~30-75 W, 3.3 GB resident from another user)
5,  0 %, 24399 MiB, 49140 MiB, 32,  35.87 W    (idle but a 24 GB vLLM engine resident)
6, 100 %, 24948 MiB, 49140 MiB, 76, 295.86 W
7,  0 %,     1 MiB, 46068 MiB, 32,  29.00 W    <-- IDLE BUT BROKEN (ECC); do not use
```

After the run, device 4 showed 22 % / 4153 MiB / 79.55 W (the state left in the log tail).

## 7. Gotchas hit

1. **GPU 7 is hardware-faulted — do not schedule work there.** It is the only truly idle device
   (0 %, 1 MiB, 46 068 MiB total because ECC is enabled), but any CUDA context dies with
   `RuntimeError: CUDA error: uncorrectable ECC error encountered` — even a bare
   `torch.rand(1000, 2, device="cuda")`. `nvidia-smi -q -i 7` reports
   `DRAM Uncorrectable: 2263 volatile / 2689 aggregate`, `Remapped Rows … Pending: Yes`,
   `Uncorrectable Error: 2`. It needs an out-of-band GPU reset (root); until then it is unusable.
   Evidence: `results/logs/gpu7_ecc_evidence.txt` (full `nvidia-smi -q -i 7`) and the failed first
   attempt `results/logs/sr_test_defaults.gpu7_ecc_failure.log` (exit 1 after 4 s, traceback at
   `torch.full(...)` inside `SymbolicRegression.generate_data`). The re-run on device 4 passed.
2. **Long phases must run in the foreground.** On this host each command is executed inside its own
   transient `systemd-run --user` unit; when the launching call returns (or times out) the unit's
   cgroup is torn down and **every** child — including `nohup setsid … &` daemons — is killed. A first
   build launched in the background was killed mid-link (`results/logs/build_20260911_2057.interrupted.log`),
   leaving `forward.o`/`generate.o`/`mutation.o` but no `.so`; that stale `build/` dir was removed
   before the clean rebuild. Run `setup_legacy.sh` in the foreground with a generous wall-clock budget
   (the build needs ~70 s, plus up to 2 h if a sibling holds the lock).
3. `MAX_JOBS=8` is cosmetic here: `setup.py` itself sets `os.environ["MAX_JOBS"] = str(os.cpu_count())`
   (= 128 on this box) before calling `build_ext`, so ninja used 128 workers. Harmless — there are
   only 4 translation units, and the successful build is the evidence.
4. `nvcc` on `PATH` by default is the **system CUDA 11.8**, which torch 2.6.0+cu124 refuses. Always
   export the 12.6 `CUDA_HOME`/`PATH` above. Omitting `…/nvvm/bin` from `PATH` breaks `cicc`/`ptxas`
   (a documented failure mode of earlier build logs), and omitting `CPATH` breaks `#include <Python.h>`.
5. The build only emits a warning about the CUDA 12.6-vs-12.4-torch minor mismatch; it is benign.
6. `/.` is at 97 % (16 GB free) — the checkout + build tree is only 6.4 MB, but keep an eye on /tmp.

## 8. Idempotency verification

`setup_legacy.sh` was run a second time (21:02:30 → 21:02:37, exit 0). It re-materialized nothing and
skipped the build:

```
SKIP materialize: checkout already present (/tmp/evogp_legacy_bench/src/evogp/cuda/forward.cu)
=== build SKIPPED — extension already built ===
/tmp/evogp_legacy_bench/src/evogp/evogp_cuda.cpython-311-x86_64-linux-gnu.so   (1112488 bytes, 21:00)
=== smoke import (one lock acquisition) ===
ok
```

Log: `results/logs/setup_legacy_rerun_idempotency.log`. Each invocation takes exactly two lock
acquisitions (build, smoke) and never nests them.

## 9. State left behind for later agents (reuse this — do not rebuild)

* **`/tmp/evogp_legacy_bench` is a complete, built, importable legacy checkout** (6.4 MB). It was
  materialized with `git archive b613503` and is **not a git repository** (no `.git`, no remote).
* Run anything against it with:
  `CUDA_VISIBLE_DEVICES=<i> PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils $PY …`
  (order matters: `src` first so the *legacy* `evogp` wins over any installed/new copy) and from
  `cd /tmp/evogp_legacy_bench` when relative asset paths are involved.
* The extension is the *only* prerequisite for `import evogp.tree` at `b613503` — there is no CPU
  fallback; if the `.so` is missing nothing works.
* `bash test/benchmark/legacy/setup_legacy.sh` re-creates the checkout and rebuilds only if the `.so`
  is gone; it is safe to call from any cwd (`EVOGP_LEGACY_DIR` / `EVOGP_LEGACY_REF` /
  `EVOGP_BENCH_LOGDIR` override the defaults).
* Verified .so identity: 1 112 488 B, sha256
  `d7af153c4d12843dc218e3616985173db9e294c3fc3a04f9dbab457b30d715a4` (detect a stale/replaced artifact
  with this hash, or just re-run the script).

## 10. Committed logs (`results/logs/`)

| file | what it is |
|---|---|
| `setup_legacy_build_run.log` | full driver output of the successful (re)build invocation: env dump, materialize skip, build, smoke — 20:59:53 → 21:01:00 |
| `build_20260911_2059.log` | raw `setup.py build_ext` compiler output of that successful build (ptxas register/spill report + link line) |
| `build_20260911_2057.interrupted.log` | truncated compiler output of the first attempt, killed externally mid-link (kept as evidence, not a valid build) |
| `sr_test_defaults.log` | **the cu-scale smoke record**: dates, lock state, `nvidia-smi` before/after, all 100 `Generation: N, Cost time: Xms` lines + fitness lines, `Generation limit reached!`, exit code 0, wall 8 s |
| `sr_test_defaults.gpu7_ecc_failure.log` | the discarded first attempt on GPU 7: ECC traceback, exit code 1, wall 4 s |
| `gpu7_ecc_evidence.txt` | full `nvidia-smi -q -i 7` dump documenting the broken device |
| `setup_legacy_rerun_idempotency.log` | second invocation of `setup_legacy.sh`: build skipped, smoke `ok`, exit 0 |

No timing in these files was inferred or edited — every number quoted above is read off those logs.
(The repo's root `.gitignore` has a blanket `*.log` rule, so these files were added with `git add -f`:
they are committed on purpose, as raw evidence for the benchmark.)
