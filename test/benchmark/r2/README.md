# `test/benchmark/r2/` — R2 GPU round: legacy CUDA vs ETL (numpy + IREE-cuda)

Runner + validator + expectations for the **second measurement round** of the old-vs-new comparison
(methodology and headline numbers of round 1: `../SPEC.md`, `../SUMMARY.md`, `../new/NOTES.md`,
`../legacy/NOTES.md`; the report written from this round's data is `../REPORT.md`).

Round 2 keeps the round-1 side-agnostic harness (`../bench_gp.py`, unchanged in its contract) and adds
what round 1 lacked: **data-row scaling** (`--data-rows`), a larger **200-tree / 64-node fixture**
(`fixtures/forest200_tree64_random.npz`), the **`stages` probe** (per-structure cold builds vs warm
cache misses), a **cold-build budget** for the opt-in IREE backend (`--max-cold-builds`), and **A/B
controls** that repeat a workload with the stage probe switched off.

| file | what it is |
|---|---|
| `run_all.sh` | the matrix runner: one `bench_gp.py` process per config, serialized on the shared lock; writes JSON + log per config and an id → exit-code table to `last_run.txt`. |
| `manifest.txt` | the validator's expectation list: `id <TAB> side(leg\|new) <TAB> json path <TAB> stages?` for all 35 configs. |
| `check_r2.py` | stdlib-only validator (no torch, no GPU): required keys per task kind, missing-id detection, side/`stages` expectations, one report table. |
| `last_run.txt` | **generated** by `run_all.sh` (last invocation's id → exit-code table); not in the manifest, safe to delete/ignore. |

## Usage

Everything runs from the **repository root** and needs the project venv (there is no `python` on
`PATH`) and a CUDA-visible device:

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python

test/benchmark/r2/run_all.sh --list                 # config table: id, group, side, task, tag, json
test/benchmark/r2/run_all.sh --dry-run              # print every exact command, execute nothing
test/benchmark/r2/run_all.sh                        # all 35 configs, light group first
test/benchmark/r2/run_all.sh --group light          #   23 configs: legacy + numpy mirrors  (~minutes)
test/benchmark/r2/run_all.sh --group iree           #   12 configs: IREE-cuda              (~tens of minutes)
test/benchmark/r2/run_all.sh --only iree_sr_a       # exactly one config (comma-separate for more)
test/benchmark/r2/run_all.sh --only np_c5r2 --timeout 7200   # raise this batch's per-config cap

# validate what landed (safe anywhere; no torch, no GPU):
python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages
python3 test/benchmark/r2/check_r2.py                # same files, no missing-id/stage expectations
python3 test/benchmark/r2/check_r2.py --verbose      # + dump the raw `stages` keys of probed runs
```

* `--only` overrides `--group`; unknown ids are a usage error (exit 2). Selection order is always the
  table order, so `iree_fixture64_n64` (the per-build-rate calibration) runs first in its group and
  `iree_sr_b2` (the heaviest) last.
* `--dry-run` prints, per config, the results paths, the note, the exact `flock … bash -c '<cmd>'`
  line and the `timeout` it would apply — nothing is executed, no directory is created.

## Rules baked into the runner (and why)

* **One process per config.** The engine's `etl.CompileCache` (and IREE's runtime) are per-process, so
  the first call of each config is a genuine cold measurement; a config can never inherit another's
  warm cache, and a crash cannot poison the next config.
* **`ulimit -n 65536` first.** The IREE backend opens hundreds of file descriptors; raising the soft
  limit is the first statement of the script.
* **Exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per invocation, never nested.** All heavy
  phases on this shared box serialize on that lock (other agents use it too); the wait is *outside*
  all timed regions, and `timeout` sits **inside** the lock so a config is bounded but never charged
  for lock waiting. `timeout` defaults to `TIMEOUT_S=3600` (env-overridable, or `--timeout SECONDS`).
* **`CUDA_VISIBLE_DEVICES` defaults to 6** (env-overridable). Never use GPU 7 — it has a hardware
  uncorrectable-ECC fault and any CUDA context on it dies; check `nvidia-smi` for contention first,
  and treat medians as ±10% (a contended window inflates runs 2–3×).
* **Side selection is `PYTHONPATH`** (the harness is side-agnostic): `src` for the new ETL engine,
  `/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils` for the legacy CUDA checkout. The legacy
  pair is overridable (`LEGACY_ROOT`, `LEGACY_SITEUTILS`, `LEG_PYTHONPATH`).
* **Per-config env overrides** are exported only for that config's process — today
  `EVOGP_ETL_CACHE_SIZE=8192` for `iree_sr_a` / `iree_sr_b2`, whose structures change on almost every
  generation and would otherwise thrash the default cache. Everything else keeps the harness defaults
  (`ETL_DISABLE_LOCATIONS=1`, backend `numpy`, `--seed 0`).
* **Guards never fake data.** A config whose input is absent — legacy checkout/siteutils missing, or
  the 200-tree fixture missing — is reported `GUARD-SKIP` with `exit 3`, which counts as *OK for the
  batch exit code* but leaves **no JSON behind**, so `check_r2.py --manifest` still lists that id as
  missing. (`exit 3` is the harness's own "guard-stopped" code and is treated the same way inside the
  batch.)
* A failing config never aborts the batch: its log/JSON stay on disk, the table says `FAIL`, the run
  continues, and the batch exits 1 at the end.

### Exit codes

| `run_all.sh` exit | meaning |
|---|---|
| 0 | every selected config exited 0 or 3 (`OK` / `SKIP`) |
| 1 | at least one config failed (1 crash, 124/137 timeout, 130 interrupt, …) — see the table |
| 2 | usage/precondition error (not run from the repo root, unknown id, missing interpreter) |

`run_all.sh` maps a per-config exit code to a status: `0 OK`, `3 SKIP`, `124/137 TIMEOUT`, `130
INTERRUPT`, anything else `FAIL`. `check_r2.py` exits 1 on malformed JSONs or missing manifest ids.

## Configs and expected wall time

| group | configs | what | rough wall time |
|---|---|---|---|
| `light` | 11 legacy + 12 new | legacy CUDA side, the same workloads on the new `numpy` backend (`--stage-probe`), plus probe-off controls (`np_c5r2_noprobe`, `np_sr_a_noprobe`) | **minutes total** — the legacy side is sub-second per run (round 1: C5 0.17 s, C5b 0.19 s, C1 1.5 s/rep) plus ~2 s process startup each; the numpy mirrors are the round-1 new-side numbers (C5 ≈ 2 s, C1 ≈ 25 s, C2b-scale ≈ 25 s, C5b ≈ 12 s) |
| `iree` | 12 new | `--backend iree` (opt-in, explicitly non-primary): 64- and 200-tree fixtures × 64 / 1 024 / 10 000 data rows, `evalfresh`, short SR runs, the old `C7` scale, and probe-off controls | **tens of minutes** — IREE is *compile*-bound, not data-bound: cost tracks the number of distinct tree structures (cold builds) per config, ~0.5 s each on round-1 evidence (C7: 50 builds ≈ 22 s first generation), capped by `--max-cold-builds` (200/400/3200). `iree_sr_b2` is the heaviest and may need `--timeout 7200` |

Data-row scaling is what the `fixture*_n1024` / `fixture*_n10000` configs measure: same forest,
1 024 or 10 000 rows instead of the fixture's native rows. `leg_c5b_r2` / `np_fresh16` /
`iree_fresh16` use `evalfresh` (a freshly generated population), so their cold-build counts scale with
the population, not the fixture.

**Prerequisite:** the 200-tree fixture `test/benchmark/fixtures/forest200_tree64_random.npz` must
exist (path is relative to the repo root; the six `*_fixture200_*` legacy+new configs and the three
`iree_fixture200_*` configs guard-skip without it).

## Where results land

```
test/benchmark/new/results/r2/<id>.json       test/benchmark/legacy/results/r2/<id>.json
test/benchmark/new/results/r2/<id>.log        test/benchmark/legacy/results/r2/<id>.log
test/benchmark/r2/last_run.txt                # id -> exit code of the last invocation
```

`<id>` is the manifest id (e.g. `np_c5r2`, `iree_sr_a`); the `bench_gp.py` `--tag` is the uppercase id
(`NP_C5R2`). The JSON is the harness's incremental snapshot (rewritten after every generation/call,
so a hard kill still leaves usable data); the log holds the GPU snapshot, the exact command and a
`tee` of the process output, preceded by the runner's header block. Logs are **appended**, so a re-run
adds a new dated block instead of erasing the previous attempt. `last_run.txt` is generated output —
delete it freely; it is not part of the manifest.

## Re-running the round against a redesigned engine

Nothing in this directory is engine-specific: the runner only fixes *what to run* (paths, flags,
env, order) and *how to serialize it*. To repeat R2 on top of an engine change (new ETL version,
another backend, another batching/memoization strategy):

1. check out / install the engine under test so that `PYTHONPATH=src` resolves it (that is the "new"
   side) — or point `PY`/`NEW_PYTHONPATH`/`LEG_PYTHONPATH` at other checkouts;
2. run `test/benchmark/r2/run_all.sh` (optionally `--group light` first) — no edit required;
3. `python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages`;
4. if a config must change, edit the config table in `run_all.sh` **and** its line in `manifest.txt`
   together (ids must match); the frozen round-1 methodology in `../SPEC.md` stays untouched, and
   round-2 numbers always state which engine revision they were taken on (each JSON records
   `evogp_commit`, `evogp_file`, `etl_version`, `env`, `gpu`).
