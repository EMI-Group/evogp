# test/benchmark/r2/round3/results/armB/windows.md — Arm B run windows

Shared-box context for every Arm B measurement window. Arm B re-measures the R3 "Arm A"
configs against the patched ETL checkout (`PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141`,
see `etl_import_probe.txt` and `../new/<id>.log`'s `etl:` line). Because the box is shared and
GPU/host contention is the dominant error source on this benchmark, each window records its own
snapshot — a number is only quotable next to the window it was taken in.

## Append format (add a new entry per window; never edit an existing one)

```
## Window N — <date/time UTC> (<local time>) — <purpose>
- purpose / phase: <agreement pre-flight | matrix group X | re-attempt | tooling>
- commands: <exact commands launched in this window>
- cwd / interpreter: <repo root of the window>, /mnt/local-ssd/bchuang/evox/.venv/bin/python
- PYTHONPATH: <src:... >   (Arm B: src:/tmp/evogp_etl_patched_51a7141)
- CUDA_VISIBLE_DEVICES: <n>   (never 7 — hardware ECC fault)
- nvidia-smi (ALL GPUs):
  ```text
  <nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits>
  ```
- compute apps (pid → user, MiB): <nvidia-smi --query-compute-apps=... + ps owner lookup>
- uptime / loadavg: <uptime; cat /proc/loadavg>
- ulimit -n (after the runner's raise): <value>
- contention notes: <foreign tenants, lock waits, GPU 6/0 occupancy, anything that makes a
  number non-comparable; state whether the window's numbers are quotable or noise-only>
```

Rules of thumb carried over from the R2/R3 rounds: treat medians as ±10 %; a contended window
inflates launch-bound configs 2–3×; serialize every heavy phase on exactly one
`flock -w 7200 /tmp/gpbench_critical.lock` (the runner does this) and never charge lock wait
time to a config.

---

## Window 0 — 2026-09-12 14:49 UTC (22:49 HKT) — Arm B tooling window (no heavy run)

- purpose / phase: tooling creation only — the Arm B runner/validator scripts, the manifest and
  this evidence scaffolding were written and self-checked (`--list`, `--dry-run`, `--bash -n`,
  `py_compile`, the offline checker selftest and the ETL import probe). **No `bench_gp.py`, no
  agreement run, no timed region, nothing on a GPU was executed in this window**; the JSON/log
  directories under `results/armB/` are therefore intentionally still empty apart from these
  evidence files.
- commands: `bash -n test/benchmark/r2/round3/run_all_armB.sh`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --list|--dry-run`,
  `bash test/benchmark/r2/round3/run_agreement_armB.sh --list|--dry-run`,
  `python3 test/benchmark/r2/round3/check_agreement_armB.py` (expected graceful FAIL: no records yet),
  `python3 test/benchmark/r2/round3/selftest_check_agreement_armB.py`, plus the read-only ETL probes
  recorded in `etl_import_probe.txt`.
- cwd / interpreter: the Arm B tooling worktree of this repo, `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141` (shadow verified: `etl.__file__` =
  `/tmp/evogp_etl_patched_51a7141/etl/__init__.py`)
- CUDA_VISIBLE_DEVICES: unset for the probes (nothing touched a GPU); the runners default to 6
- nvidia-smi (ALL GPUs), 2026-09-12T22:49:21+08:00:
  ```text
  0, NVIDIA RTX A6000, 100, 21034, 49140
  1, NVIDIA RTX A6000, 100, 22846, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 43, 15377, 49140
  4, NVIDIA RTX A6000, 37, 15389, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 26, 30455, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
- compute apps (pid → user, MiB), same snapshot:
  - GPU 0 (`GPU-7ca10c94…`): 2394552 → hyqiu 19706, 2398977 → hyqiu 1318
  - GPU 1 (`GPU-38ba30b9…`): 2394555 → hyqiu 20052, 2398881 → hyqiu 2784
  - GPU 2 (`GPU-01343fc6…`): 1228200 → whhu (VLLM::EngineCore, 35 d) 16454
  - GPU 3 (`GPU-25828e67…`): 1285079 → hyqiu (pt_main_thread, 1 d 11 h) 15320
  - GPU 4 (`GPU-098b70f2…`): 1284680 → hyqiu 15300
  - GPU 5 (`GPU-a4c47eb9…`): 1228191 → whhu (VLLM) 24274
  - GPU 6 (`GPU-b8253071…`): 2246336 → syfang (python, up 2 h 54 m) 30406 — **still the same
    foreign tenant the Arm A first window ran alongside (R2/R3 runlog's `syfang` video-attention
    training), now ≈30 GB of the 49 GB card**
  - GPU 7 (`GPU-0f5f6bfd…`): no compute apps (1 MiB resident) — the known hardware-ECC-fault card,
    never used
- uptime / loadavg: `up 252 days, 21:04, 46 users, load average: 84.58, 105.73, 103.66`
  (`/proc/loadavg` 84.58 105.73 103.66 84/16035)
- ulimit -n (raised by the runner): 65536
- contention notes: host is heavily loaded (loadavg ≈85–105, 46 logged-in users, GPUs 0–6 all
  carrying foreign tenant memory) and **GPU 6 — the runner default — holds ≈30 GB of `syfang`'s
  job**, i.e. at least as contended as Arm A's first window (which inflated the launch-bound legacy
  ids ≈3× until the light group was re-run on GPU 0). Consequently: the upcoming Arm B heavy
  windows should be watched for the same inflation, and if a companion re-run is needed it should
  follow the Arm A precedent of moving to a tenant-free card (GPU 0 was only occupied by hyqiu's
  ≈21 GB job at this snapshot, so a GPU choice must be re-checked per window, not assumed).
  Nothing measured in this window is a benchmark number — there are none.

---
## Window 1 — 2026-09-12 14:54 UTC (22:54 HKT) — Arm B agreement pre-flight (gate (i), six arms)
- purpose / phase: agreement pre-flight — the six Arm B arms (`fixture64|fixture200` ×
  `numpy|iree|iree_spec`) against the PATCHED ETL checkout, one fresh
  `check_backend_agreement.py` process per arm, serialized on the shared benchmark lock.
  Correctness gate only: no R3-style headline number comes out of this window, so shared-host
  contention can slow it, but cannot make it wrong.
- commands: `bash test/benchmark/r2/round3/run_agreement_armB.sh` (one invocation, all six arms,
  launched with `nohup … > /tmp/armB_evidence/armB_agreement_console.log 2>&1 &` from the repo root
  and polled; no wrapper lock, no `timeout` — the runner owns both), followed by
  `python3 test/benchmark/r2/round3/check_agreement_armB.py`; plus the read-only ETL import probes
  appended to `etl_import_probe.txt` at window start.
- cwd / interpreter: repository root of the Arm B pre-flight worktree
  (`/mnt/local-ssd/bchuang/evogp/.genesis/workers/worker_T3_A255`),
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141`   (Arm B: the patched checkout shadows the stock
  editable install — re-verified in this window, see `etl_import_probe.txt`)
- CUDA_VISIBLE_DEVICES: 6   (the runner's default, kept because GPU 6 was at **0 % utilization** in
  the snapshot below; never 7 — hardware ECC fault). All six arms used the one chosen device.
- nvidia-smi (ALL GPUs), 2026-09-12T22:54:14+08:00 (UTC 14:54:14):
  ```text
  0, NVIDIA RTX A6000, 100, 22510, 49140
  1, NVIDIA RTX A6000, 100, 22856, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 0, 15377, 49140
  4, NVIDIA RTX A6000, 0, 15389, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 0, 30455, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
- compute apps (pid → user, MiB), same snapshot:
  - GPU 0 (`GPU-7ca10c94…`): 2394552 → hyqiu 19706, 2398977 → yzhou (nnUNetv2_predic) 2794
  - GPU 1 (`GPU-38ba30b9…`): 2394555 → hyqiu 20052, 2398881 → yzhou (nnUNetv2_predic) 2794
  - GPU 2 (`GPU-01343fc6…`): 1228200 → whhu (VLLM::EngineCore, 35 d) 16454
  - GPU 3 (`GPU-25828e67…`): 1285079 → hyqiu (pt_main_thread, 1 d 11 h) 15320
  - GPU 4 (`GPU-098b70f2…`): 1284680 → hyqiu 15300
  - GPU 5 (`GPU-a4c47eb9…`): 1228191 → whhu (VLLM) 24274
  - GPU 6 (`GPU-b8253071…`): 2246336 → syfang (python, up ≈2 h 59 m) 30406 — the same foreign tenant
    the tooling window and Arm A's first window ran alongside (≈30 GB of the 49 GB card, **idle in
    this snapshot**)
  - GPU 7 (`GPU-0f5f6bfd…`): no compute apps (1 MiB resident) — the known hardware-ECC-fault card,
    never used
- uptime / loadavg: `22:54:14 up 252 days, 21:09, 46 users, load average: 125.64, 120.33, 110.51`
  (`/proc/loadavg` 125.64 120.33 110.51 11/15866) — the host is under heavy load (≈126, 46 users)
- ulimit -n (after the runner's raise): 65536
- contention notes: GPU 6 was **0 % utilized / ≈30 GB resident** (syfang's job) when the device was
  chosen, so the default card was kept; the arms' own `gpu (before)/(after)` snapshots in their logs
  show that tenant's utilization fluctuating (0 → ≈97 %) while the six arms ran, i.e. the window is
  shared exactly like Arm A's. The agreement gate compares fitness vectors, not times, so this window
  is quotable as a *correctness* verdict regardless of the contention; any wall time quoted from it
  carries the usual shared-box inflation and is not a benchmark number. Lock discipline is the
  runner's: exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per arm, never nested (other
  agents share that lock), with no `timeout` wrapper inside it.

---
## Window 1b — 2026-09-12 15:01 UTC (23:01 HKT) — Arm B agreement pre-flight, the EXECUTED batch (same device as Window 1)
- purpose / phase: agreement pre-flight — the re-launch of Window 1's batch. Window 1's detached
  (`nohup … &`) invocation was reaped together with its own parent shell at 22:54:24, mid-arm
  (`fixture64_numpy` had already finished cleanly with hash `2e5c0fa1…`; `fixture64_iree` was left
  header-only in its append-only log). This window runs the same six arms **in the foreground of one
  long-budget invocation**, so the batch completes inside a single process lifetime; the runner is
  idempotent per arm and the workload is RNG-free, so the re-measured fixtures are the same ones. The
  records committed under `results/armB/backend_agreement/` are the ones this window produced.
- commands: `bash test/benchmark/r2/round3/run_agreement_armB.sh` (one invocation, six arms
  sequentially, stdout tee'd to `/tmp/armB_evidence/armB_agreement_console.log`), then
  `python3 test/benchmark/r2/round3/check_agreement_armB.py` (no GPU) — both from the repo root.
- cwd / interpreter: repository root of the Arm B pre-flight worktree,
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141`
- CUDA_VISIBLE_DEVICES: 6   (unchanged from Window 1: GPU 6 was idle at snapshot time and the gate is a
  correctness gate; one device for all six arms, never 7)
- nvidia-smi (ALL GPUs), 2026-09-12T23:01:00+08:00 (UTC 15:01:00):
  ```text
  0, NVIDIA RTX A6000, 98, 20025, 49140
  1, NVIDIA RTX A6000, 98, 20059, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 9, 15377, 49140
  4, NVIDIA RTX A6000, 38, 15389, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 0, 30455, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
- compute apps: unchanged tenant set from Window 1's snapshot (hyqiu on GPUs 0/1/3/4, yzhou's
  nnUNetv2_predic on GPUs 0/1, whhu's VLLM::EngineCore on GPUs 2/5, **`syfang` pid 2246336 on GPU 6,
  ≈30 GB, ~3 h up**); no process of this round appears in the snapshot because the batch starts after it.
- uptime / loadavg: `23:01:01 up 252 days, 21:16, 46 users, load average: 88.66, 99.35, 103.97`
- ulimit -n (after the runner's raise): 65536
- contention notes: the six arms ran 23:01:00 → 23:04:05 HKT (3 min 5 s total) while GPU 6's foreign
  tenant `syfang` oscillated 0 → 97 % utilization (per-arm `gpu (before)/(after)` lines, all in the
  committed arm logs) and the host carried loadavg ≈88–126. Every arm took its own single
  `flock -w 7200 /tmp/gpbench_critical.lock` with no observable wait, and the batch needed no retry:
  `ok=6 skipped=0 failed=0`, `gate: PASS (38 checks, 0 failures)`. Because this window decides
  *quotability* (fitness-vector equality), the contention inflation quoted above is a wall-clock caveat
  only; the window's correctness verdict is quotable.

---
## Window 2 — 2026-09-12 15:08 UTC (23:08 HKT) — matrix group `iree-spec` (3 configs, GPU 6)
- purpose / phase: matrix group `iree-spec` — the three `--backend iree` / `EVOGP_EVAL_MODE=specialized`
  configs (`armB_iree_fixture64_n64_spec`, `armB_iree_fixture200_n64_spec`, `armB_iree_sr_a_spec`)
  against the PATCHED ETL checkout, i.e. the first measured Arm B window (Arm B's quotability is
  already established by gate (i), which passed in window 1b: all six arms bit-identical to Arm A).
  This window's numbers are the Arm B side of the Arm A-vs-Arm B tables.
- commands: one foreground invocation per config, in run order —
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64_spec`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64_spec`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a_spec`
  (each console tee'd to `results/armB/status_<id>.txt`; the runner owns `ulimit`/`flock`/`timeout` —
  one `flock -w 7200 /tmp/gpbench_critical.lock` per config with `timeout -s INT -k 30 3600` inside it).
- cwd / interpreter: repository root of the Arm B matrix worktree,
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141`   (shadow re-verified at window start, see
  `etl_import_probe.txt` third block; patched-file hashes unchanged since the agreement window)
- CUDA_VISIBLE_DEVICES: 6   (the runner default, **kept because GPU 6 showed 0 % utilization** in the
  snapshot below — ≈30 GB of `syfang`'s job resident, i.e. memory pressure but no compute load; the
  switch-to-a-free-card rule was not triggered, and GPU 7 — hardware ECC fault — is never used)
- nvidia-smi (ALL GPUs), 2026-09-12T23:08:41+08:00 (UTC 15:08:41):
  ```text
  0, NVIDIA RTX A6000, 0, 6283, 49140
  1, NVIDIA RTX A6000, 97, 19941, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 0, 15377, 49140
  4, NVIDIA RTX A6000, 0, 15389, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 0, 30447, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
- compute apps (pid → user, MiB), same snapshot:
  - GPU 0 (`GPU-7ca10c94…`): 2394552 → hyqiu 6276
  - GPU 1 (`GPU-38ba30b9…`): 2394555 → hyqiu 19934
  - GPU 2 (`GPU-01343fc6…`): 1228200 → whhu (VLLM::EngineCore) 16454
  - GPU 3 (`GPU-25828e67…`): 1285079 → hyqiu 15320
  - GPU 4 (`GPU-098b70f2…`): 1284680 → hyqiu 15300
  - GPU 5 (`GPU-a4c47eb9…`): 1228191 → whhu (VLLM) 24274
  - GPU 6 (`GPU-b8253071…`): 2246336 → syfang (python) 30406 — the same ≈30 GB foreign tenant as
    windows 0/1/1b; **0 % utilization** at snapshot time
  - GPU 7 (`GPU-0f5f6bfd…`): no compute apps (1 MiB resident) — the known hardware-ECC-fault card,
    never used
- uptime / loadavg: `23:08:43 up 252 days, 21:24, 46 users, load average: 99.44 99.15 102.41`
  (`/proc/loadavg` 99.44 99.15 102.41 9/13653)
- ulimit -n (after the runner's raise): 65536
- contention notes: the box is as shared as in windows 1/1b (loadavg ≈99, 46 users, foreign tenants on
  GPUs 0–5) and GPU 6 again carries the ≈30 GB `syfang` tenant whose utilization oscillates — the
  device was kept because the switch rule (sustained ≥90 % foreign utilization in the snapshot) was not
  met. Per-config contention evidence is the `gpu (before)/(after)` pair in each committed
  `results/armB/new/<id>.log`; treat the window's medians as ±10 % and re-read a number only together
  with this window.

---
## Window 3 — 2026-09-12 15:17 UTC (23:17 HKT) — matrix group `iree` (3 configs, GPU 6)
- purpose / phase: matrix group `iree` — the three `--backend iree` / `EVOGP_EVAL_MODE=auto` configs
  (`armB_iree_fixture64_n64`, `armB_iree_fixture200_n64`, `armB_iree_sr_a`) against the PATCHED ETL
  checkout. These are the DENSE default route (`auto` picks the wave-2 dense evaluator on iree), and
  they are the second measured Arm B window after `iree-spec` (window 2); the gate that makes Arm B
  IREE numbers quotable — the six-arm A→B `values_sha256` identity pre-flight — already passed in
  window 1b, and Arm A's own `iree` group is the comparison baseline.
- commands: one foreground invocation per config, in run order —
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a`
  (each console tee'd to `results/armB/status_<id>.txt`; the runner owns `ulimit`/`flock`/`timeout` —
  one `flock -w 7200 /tmp/gpbench_critical.lock` per config with `timeout -s INT -k 30 3600` inside it),
  plus the read-only ETL import probes appended to `etl_import_probe.txt` at window start.
- cwd / interpreter: repository root of the Arm B `iree` matrix worktree,
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141`   (shadow re-verified at window start, see
  `etl_import_probe.txt` fourth block; patched-file hashes unchanged since the agreement window)
- CUDA_VISIBLE_DEVICES: 6   (the runner default, **kept**: an 8-sample 4-s utilization series of GPU 6
  taken at window start read `0 / 0 / 0 / 48 / 50 / 14 / 90 / 31 %` — i.e. the ≤30 GB `syfang` tenant
  oscillates and never held the sustained ≥90 % that triggers the switch rule, while GPU 0 was at
  71–96 % of its own foreign load and is therefore not a quieter alternative; GPU 7 — hardware ECC
  fault — is never used)
- nvidia-smi (ALL GPUs), 2026-09-12T23:17:50+08:00 (UTC 15:17:50):
  ```text
  0, NVIDIA RTX A6000, 97, 20003, 49140
  1, NVIDIA RTX A6000, 96, 19941, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 37, 15315, 49140
  4, NVIDIA RTX A6000, 28, 15387, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 100, 30443, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
  (the 23:17:30 pre-check snapshot read GPU 6 at `0, 30443` instead — the tenant's duty cycle, and the
  reason the decision below is based on the series rather than on one sample)
- compute apps (pid → user, MiB), same snapshot:
  - GPU 0 (`GPU-7ca10c94…`): 2394552 → hyqiu (pt_main_thread, up 53 m) 19996
  - GPU 1 (`GPU-38ba30b9…`): 2394555 → hyqiu (pt_main_thread, up 53 m) 19934
  - GPU 2 (`GPU-01343fc6…`): 1228200 → whhu (VLLM::EngineCore) 16454
  - GPU 3 (`GPU-25828e67…`): 1285079 → hyqiu (pt_main_thread, up 1 d 12 h) 15258
  - GPU 4 (`GPU-098b70f2…`): 1284680 → hyqiu (pt_main_thread, up 1 d 12 h) 15298
  - GPU 5 (`GPU-a4c47eb9…`): 1228191 → whhu (VLLM) 24274
  - GPU 6 (`GPU-b8253071…`): 2246336 → syfang (python, up 3 h 22 m) 30406 — the same ≈30 GB foreign
    tenant as windows 0/1/1b/2, **oscillating** (see the series above)
  - GPU 7 (`GPU-0f5f6bfd…`): no compute apps (1 MiB resident) — the known hardware-ECC-fault card,
    never used
- uptime / loadavg: `23:17:50 up 252 days, 21:33, 46 users, load average: 109.72, 100.65, 102.24`
  (`/proc/loadavg` 109.72 100.65 102.24 63/14386)
- ulimit -n (after the runner's raise): 65536
- contention notes: same tenant set as windows 1/1b/2 (hyqiu on GPUs 0/1/3/4, whhu's VLLM on 2/5,
  `syfang` on GPU 6) with the host at loadavg ≈100–110 / 46 users, and **GPU 6 kept under the
  documented rule** (oscillating, not sustained ≥90 %; GPU 0 was the busier card in the series). The
  `iree` group is the dense route, i.e. one graph per structure instead of per-tree specialization, so
  its first-call numbers inherit the documented IREE compile spread (±a factor of a few) and its warm
  numbers the shared-box ±10 % band; per-config evidence is the `gpu (before)/(after)` pair in each
  committed `results/armB/new/<id>.log`. The `sr` arm's fitness trajectory is contention-independent
  and is the cross-check that this window measured the same workload as Arm A / R2.

---
## Window 4 — 2026-09-12 15:26 UTC (23:26 HKT) — matrix group `numpy` (3 configs, GPU 6)
- purpose / phase: matrix group `numpy` — the three default-`numpy`-backend configs
  (`armB_np_c5r2`, `armB_np_fixture200_n64`, `armB_np_sr_a`; `--stage-probe` ON, no env overrides)
  against the patched ETL shadow. On `numpy` the engine keeps the SPECIALIZED route, so unlike the
  dense `iree` eval arms this group's `stages` blocks are expected to be meaningful — their contents
  are recorded verbatim in the runlog section.
- commands: one foreground invocation per config, in run order —
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_c5r2`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_fixture200_n64`,
  `bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_sr_a`
  (each console tee'd to `results/armB/status_<id>.txt`; the runner owns `ulimit`/`flock`/`timeout` —
  one `flock -w 7200 /tmp/gpbench_critical.lock` per config with `timeout -s INT -k 30 3600` inside
  it), plus the read-only ETL import probes appended to `etl_import_probe.txt` at window start and the
  full-manifest validator (`check_r3.py --manifest ... --require-stages`) after the last config.
- cwd / interpreter: repository root of the Arm B `numpy` matrix worktree,
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
- PYTHONPATH: `src:/tmp/evogp_etl_patched_51a7141`   (shadow re-verified at window start, see
  `etl_import_probe.txt` fifth block; patched-file hashes unchanged since the agreement window)
- CUDA_VISIBLE_DEVICES: 6   (the runner default, **kept**: two utilization series of GPU 6 taken at
  window start — 12 samples / 3 s: `0 0 0 0 0 0 0 0 85 42 98 97 %`, then 15 samples / 3 s:
  `0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 %` — show the ≈30 GB `syfang` tenant **bursting, not sustained
  ≥90 %**, so the switch rule does not trigger; the group is the CPU-side reference interpreter
  anyway, so the device only holds tensors. GPU 7 — hardware ECC fault — is never used.)
- nvidia-smi (ALL GPUs), 2026-09-12T23:25:59+08:00 (UTC 15:25:59):
  ```text
  0, NVIDIA RTX A6000, 0, 1, 49140
  1, NVIDIA RTX A6000, 0, 1, 49140
  2, NVIDIA RTX A6000, 0, 16583, 49140
  3, NVIDIA RTX A6000, 0, 15315, 49140
  4, NVIDIA RTX A6000, 0, 15387, 49140
  5, NVIDIA RTX A6000, 0, 24395, 49140
  6, NVIDIA RTX A6000, 0, 30439, 49140
  7, NVIDIA RTX A6000, 0, 1, 46068
  ```
- compute apps (pid → user, MiB), same snapshot:
  - GPU 0 (`GPU-7ca10c94…`): no compute apps (1 MiB resident) — hyqiu's window-3 job has exited
  - GPU 1 (`GPU-38ba30b9…`): no compute apps (1 MiB resident) — same
  - GPU 2 (`GPU-01343fc6…`): 1228200 → whhu (VLLM::EngineCore, up 35 d) 16454
  - GPU 3 (`GPU-25828e67…`): 1285079 → hyqiu (pt_main_thread, up 1 d 12 h) 15258
  - GPU 4 (`GPU-098b70f2…`): 1284680 → hyqiu (pt_main_thread, up 1 d 12 h) 15298
  - GPU 5 (`GPU-a4c47eb9…`): 1228191 → whhu (VLLM, up 35 d) 24274
  - GPU 6 (`GPU-b8253071…`): 2246336 → syfang (`python main.py --config exps/video_attention.json`,
    up 3 h 31 m) 30406 — the same ≈30 GB foreign tenant as windows 0/1/1b/2/3
  - GPU 7 (`GPU-0f5f6bfd…`): no compute apps (1 MiB resident) — the known hardware-ECC-fault card,
    never used
- uptime / loadavg: `23:26:01 up 252 days, 21:41, 46 users, load average: 80.59, 82.54, 92.51`
  (`/proc/loadavg` 80.59 82.54 92.51 5/13956)
- ulimit -n (after the runner's raise): 65536
- contention notes: GPUs 0 and 1 became tenant-free since window 3 (hyqiu's job that the `iree` group
  shared the card with has exited), but GPU 6 was kept under the documented rule — the `syfang` tenant
  bursts to 85–98 % for a few seconds and then idles, never sustained ≥90 % — so this window stays on
  the same device as windows 1b/2/3 and is directly comparable to them. Host loadavg at window start
  ≈81–93 / 46 users (lower than windows 2/3's ≈100–110). The three configs are the **numpy reference
  interpreter** route, i.e. host-side (CPU) graph work: the shared-host load is the first-order
  contention source, the GPU only holds the tensors, and each config's `gpu (before)/(after)` pair is
  recorded in its committed log. The `sr` arm's fitness trajectory is contention-independent and is
  the cross-check that this window measured the same workload as Arm A / R2.
