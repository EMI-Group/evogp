# `test/benchmark/new/` — NEW-side (ETL "true jit" engine) harness notes + committed round-1 records

## Intent
This node owns the NEW side of the legacy-vs-ETL benchmark: the side's documentation (`NOTES.md`), its
round-1 config-matrix runner (`run_matrix.sh`), its validator (`check_smoke.py`) and the committed
result records (`results/`), including the R2 records produced with the round-2 harness extensions.
The side-agnostic harness itself is the sibling `../bench_gp.py`; methodology is frozen in `../SPEC.md`.
Nothing in this node is imported by `evogp`; everything is hand-run from the repository root with the
project venv and a CUDA-visible GPU (never GPU 7 — ECC fault; runners default to GPU 6 in later rounds).

## Contents
- `NOTES.md` (766 lines) — the side's documentation of record: §1–§9 Phase-1 (harness CLI, task wiring,
  timing conventions, full JSON schema, old-compat audit, smoke results, matrix results, observations,
  deviations), then `## R2` (harness extensions + stage probe + calibration) and `## R4` (`--task
  eval-pure`, measurement type 2, pure/compile/e2e_companion JSON blocks). Keep section titles stable.
- `run_matrix.sh` — round-1 SPEC runner; config table `C1 C5 C2b C4 C2 C0 C5b C7` (line 34), per-config
  timeouts and env (lines 53–64); internal `flock -w 7200 /tmp/gbench_critical.lock` with `timeout`
  inside the lock; auto-picks an A6000 excluding nonzero-ECC cards (lines 78–123).
- `check_smoke.py` — stdlib-only validator for `bench_gp.py` JSONs: schema keys (TRAIN_KEYS/EVAL_KEYS
  lines 36–107), rep-0 mirroring, `median_warm_ms == median(per_call_ms[1:])` (line 187), partial/
  projection/complete consistency, per-tag domain checks (TAG_EXPECTATIONS lines 111–114).
- `results/` — round-1 records: `C0 C1 C2 C2b C4 C5 C5b C7` JSON+log (+ `_console_*.log` batch logs);
  `results/smoke/` Phase-1 smoke JSONs; `results/r2/` R2 records (+ `results/r2/smoke/` h1/h2 probe
  smokes, `results/r2/backend_agreement/` IREE-vs-numpy pre-flight) — data only, cite paths in reports.

## API surface (JSON record schema; authoritative detail in NOTES.md §1 + R2 + R4)
- Shared header: `tag, side, task, evogp_file, evogp_commit(+_error), etl_version, problem, config,
  env, gpu, cwd, harness, wall_s, harness_wall_s, construct_s, partial, projection, complete, per_rep[],
  notes, started_at, finished_at`; R2 adds `fds{start,after_cold,end}, nofile_rlimit, cold_builds,
  skipped, skip_reason, stages`; R4 adds `pure, compile, e2e_companion`.
- Training (`sr`/`xor`): `generation_loop_s, per_gen_ms[], per_gen_best_fitness[], cold_first_gen_ms
  (= per_gen_ms[0]), best_fitness_final`; eval (`eval64`/`evalfresh`): `eval_kind, fixture,
  per_call_ms[], first_call_ms (= per_call_ms[0]), median_warm_ms (= median(per_call_ms[1:]), null
  when <1 warm call)`.
- Stage probe `stages` (null when off): `status{probe, installed, wrapped, missing, engine_module,
  etl_module, counters, nested_runs}` + `reps[].records[]` per call/gen with `total_ms, probe_span_ms,
  decode_ms, sr_fitness_ms, evaluate_ms, loss_ms, misc_ms, build_ms, build_hit_ms, build_miss_ms,
  run_ms, readback_ms, upload_ms` and call counts, + `totals` with `cold_builds/cold_build_ms`
  (first record only), `total_build_miss*`, `warm_misses/warm_build_miss_ms` (misses after the first
  record; `warm_misses == 0` proves a warm pass compiled nothing).
- Flag reference: R2 — `--backend {default,numpy,iree}`, `--iree-device 0`, `--iree-opt-level {O0,O1}`,
  `--data-rows N`, `--mem-guard-mb 8192`, `--max-cold-builds 300` (IREE+new only, counts cache misses at
  checkpoints), `--allow-cold-builds`, `--stage-probe`; R4 — `--task eval-pure`, `--mode {auto,dense,
  specialized,dynamic}`, `--invocations 50`, `--warmup 10`, `--e2e-companion K`, `--pure-runner
  {auto,frozen,fallback}`, `--reps 5` for eval-pure. Exit codes: 0 complete · 3 skip/guard · 130
  interrupt · 1 crash.

## Constraints
- `../SPEC.md` is frozen round-1 methodology — never edit it; deviations belong in `NOTES.md`.
- A config id lives in two places (runner table + manifest) and must stay in sync; keep `run_matrix.sh`
  config ids aligned with `../SPEC.md`'s matrix (C0, C1, C2, C2b, C4, C5, C5b, C7 — there are no C3/C6).
- Records are immutable evidence: never rewrite `results/**` JSONs/logs; a re-run writes fresh files.
- One `flock /tmp/gbench_critical.lock` per process (never nested), `timeout` inside the lock, medians
  treated as ±10 % on this shared box (contended windows inflate 2–3×).
- `results/C5.json` has a special provenance (Phase-1 artifact at `evogp_commit=7e6cd0f`, old worktree
  paths in `cwd`/`harness`, no `backend` env keys) — keep it, label it in tables (NOTES.md §9.5).

## Known Issues
- Probe overhead is real but small: +1.343 ms/warm call (22.880 vs 21.537 ms, ≈+6 %) on `eval64` numpy
  (NOTES.md lines 621–625; `results/r2/smoke/h2_np_eval64{,_noprobe}.json`); later A/B pairs measure
  +0.218 ms (numpy pop64), +1.590 ms (iree pop64) and +4.7 ms (iree fresh16, noisy). Probe-on/off
  first-call deltas are dominated by IREE compile-time variance, not the probe.
- All R2 records in this node used the **specialized** per-structure route (the dense engine post-dates
  R2): the probe's `build_hit_ms`/`run_ms`/`readback_ms` are per-tree aggregates, not dense-graph numbers.
- IREE compile time is not reproducible on this box (≈285–1729 ms per build across h2 scans; 888/1729/285
  for X/Y/Z) — only quote `cold_build_ms` per config, never a global s/build (NOTES.md 596–617).
- IREE leaks ≈2 FD per run and order 10–20 FD per cold build (stock ETL); `results/r2/iree_sr_b2.json` is
  the R2 SIGSEGV/crash skeleton (exit 139 after ~800–900 builds) and must never be read as a training
  result; C7's first attempt died at the FD limit and was replaced by the retry (NOTES.md §9.2).
- Cache warmth is process-local and evicts: a pop-200×100-gen run touches ~10–20k distinct structures
  against `etl.CompileCache` (default 4096), so `--reps 2` is only warm when the distinct-structure count
  fits; C4 (xor, pop 50) rep1 = 5.1× faster than rep0, C1 rep1 is not faster (NOTES.md 344–352, 361–382).
- `--max-cold-builds` samples cumulative cache misses at checkpoints and can overshoot within one
  generation; a cap of 300 ≈ ≈6.3k FDs at 21 FD/build — budget FDs separately (NOTES.md 626–633).
- `iree_sr_a.json` records `EVOGP_ETL_CACHE_SIZE=8192` (a non-default cache size for that record).

## Notes for Agents
- The stage-probe module itself is the sibling `../stage_probe.py` (self-test `python3
  test/benchmark/stage_probe.py`, no torch/GPU/etl); R4's pure-runner core is the sibling
  `../pure_forward.py`; both are exercised through the sibling harness `../bench_gp.py`.
- R2/R4 validators/runners are siblings too: `../r2/check_r2.py`, `../r2/run_all.sh`; `../r4/check_r4.py`,
  `../r4/run_pure.sh`, `../r4/run_e2e.sh` (+ `check_e2e_r4.py`, `check_agreement_r4.py`).
- In R2 records `evogp_commit` varies per record (6f773b4 / dc5c0f5 / cbe4237 / 3812724 …) — always read
  the engine identity from the JSON before comparing numbers across records.
- The strongest phase-level decomposition in this node is the R2 stage-probe smoke (h2) and the probed
  campaign records `results/r2/{np,iree}_fixture{64,200}_n64.json`, `{np,iree}_sr_a.json`,
  `{np,iree}_c1r2|c7r.json` — warm medians break down as decode ~8 % (numpy) / 2–3 % (iree), per-tree
  `build_cached` hit ~16–46 %, per-tree `run` ~40–51 %, D2H readback ~29 % (iree only), upload <2 %.

## Routing Table / See Also
- `results/` (incl. `results/smoke/`, `results/r2/`) is data only — no sub-nodes; cite file paths.
- `../SPEC.md` → frozen methodology (workloads W1–W3, matrix, rules); `../REPORT.md` → analysis of record
  (Round-4 head); `../r4/{README.md,SUMMARY.md}` → current verdict + type-2 numbers; `../r2/README.md` →
  R2 protocol; `../stage_probe.py` → probe module; `../bench_gp.py` → harness.
- `../../goldens/README.md` → the W3 fixture provenance (`forest_random.npz`); `../fixtures/README.md` →
  the 200-tree R2 fixture; `../../src/evogp/tree/warm_path_findings.md` → wave-1 stage record cited by
  the R3/R4 verdicts.
