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
