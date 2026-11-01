# `test/archive/` — the benchmark records of the deleted torch stack (read-only evidence)

## What this is
Every number below was measured on the **pre-2.0.0a2 tree**: the torch-native stack (`evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline`) and the four `EVOGP_EVAL_MODE` routes that engine layer offered (`fused`, `dense`, `specialized`, the opt-in `batch`) against plain-PyTorch baselines, plus the earlier rounds against the hand-written CUDA engine at `b613503`.
That stack was deleted in the 2.0.0a2 pre-release, so **nothing here is reproducible from the current tree** and **none of the code that produced it is here** — the harness's `*.py`, its shell drivers and its compiler dumps were deleted with the stack.
The producing code is recoverable at commit `6720723` (`git show 6720723:test/benchmark/evoxc/campaign.py`, or `git checkout 6720723 -- test/benchmark`); a worktree at that commit is the only way to re-run anything in this directory.
This is evidence, not tooling: do not re-run, re-derive or re-gate these records, and **do not edit a single number** in them.
Treat every `CONTEXT.md` under this directory as archived harness documentation — each one opens with an `ARCHIVED` banner, and its routing table points at code that no longer exists.
Every path in this README is relative to `test/archive/benchmark/` unless it starts with `src/`; the pre-move location of that file was `test/benchmark/<same path>`.

## What the current tree measures instead
The shipped stack is `evogp.gp` (torch-free, `evoxc` + numpy), the optional Triton accelerator `evogp.fused`, and `evogp.evox_ext`.
Its harnesses of record live in the library, not here:
- `src/evogp/gp/measure_popscale.py` — per-call evaluation along the population axis, one shared population fed to every lane (`numpy`, `gp`, `gp-fused`, `torch-batched`, `torch-per-tree`); report `src/evogp/gp/POPSCALE.md`.
- `src/evogp/gp/measure_generation.py` — one whole GP generation per cell, the device-resident loop against the numpy path, `--cells popscale` for the eight-population column; report `src/evogp/gp/DEVICE_LOOP.md`.
- `src/evogp/gp/measure_fused.py` — the Triton `fused` objective against the in-graph `evoxc` route, with `--check` for the parity acceptance run.
- `python -m evogp.fused.check --stage all` — the fused route's own battery (parity, multi-output, goldens, declines, wiring, timing, tuning); report `src/evogp/fused/FUSED.md`.
The independent numerics oracle those bodies compare against stays in `test/oracle/`, which is live dev tooling and is **not** part of this archive.

## Reports of record (read these, not the JSON, first)
| file | what it is |
|---|---|
| `REPORT.md` | the old-vs-new GPU rounds (R2 baseline + the R4 verdict): legacy CUDA engine vs the evoxc engine, warm/steady-state |
| `SPEC.md` | the protocol specification those rounds were run under |
| `SUMMARY.md` | the round-1 headline: the legacy CUDA engine was ~34×–7 670× faster per generation, at the cost of a compiled extension |
| `README.md` | the round-1…R4 overview as it stood when the evoxc engine became the only engine |
| `CONTEXT.md` | the archived harness's routing table, including the **two measurement types** the rounds use (type 1 = end-to-end public API, type 2 = pure compiled-graph forward) |
| `evoxc/FINAL.md` | the evoxc engine vs plain PyTorch vs the pre-optimization snapshot, the shipped `specialized`/`dense` routes |
| `evoxc/BATCH.md` | the opt-in `batch` (merged-graph) route: design, rounds 1–3, the closing §15 verdict — it stays opt-in and loses any GP loop |
| `evoxc/LARGE_SCALE.md`, `evoxc/LARGE_SCALE_ROUND2.md` | the large-scale column (pop 2 048–8 192, `N` 4 096–16 384, 123–201-node trees); round 2 is the current report for batch-vs-dense there |
| `evoxc/POPSCALE.md` | the population axis, pop 64 → 8 192 at one fixed shape, over the four routes |
| `evoxc/VALIDATION.md` | numeric parity and cross-arm digest agreement of every route against the plain-torch oracle |
| `evoxc/SUMMARY.md` | the evoxc harness's baseline summary |
| `evoxc/README.md` | the harness's methodology, cell/arm definitions and reproduce commands (code gone — history-only) |
| `evoxc/results/t1a1_auto_r3/README.md` | the SHIPPED default of that tree (`auto` → fused) campaign of record: `batch_forward` 0.0449–0.1152 ms = 80.2×–228.9× `torch-batched`, a whole generation 2.841–3.282 ms with 14.2–15.4 % in evaluation |
| `evoxc/results/t1a1_auto/README.md` | that default's one-screen verdict page (quotes r3; keeps the pre-change JSON as the A/B baseline) |
| `evoxc/results/t1a1_auto_r2/README.md` | the previous window of the same column |
| `evoxc/results/t1a1_fused/README.md` | the explicit `EVOGP_EVAL_MODE=fused` column (41×–157× `torch-batched`), with `ENVELOPE.md` and `campaign.md` beside it |
| `evoxc/results/t1a1_baseline/BASELINE.md` | the pre-fused default, the reference the three above are read against |
| `evoxc/results/minpop_r3/SUMMARY.md` | the `EVOGP_FUSED_MIN_POP` column re-taken: the kernel wins warm at pop 1, so the floor of 16 is a cold-start guard |
| `evoxc/results/batch_opt_round{1,2,3}/SUMMARY.md` | the three `batch`-route optimization rounds (chunked compile, plan key, band policy; the refreshed cold/warm/loop columns) |
| `r2/README.md`, `r2/RUNLOG.md`, `r2/round3/`, `r4/SUMMARY.md`, `r5/RCA.md`, `r6/SUMMARY.md` | rounds R2–R6: the legacy-engine comparison, the pure-forward round, the ≈9.5× gap root-cause analysis, and the Triton prototype that the shipped `fused` route superseded |
| `legacy/`, `new/`, `xval/`, `fixtures/` | the two harness sides of round 1 together with their cross-validation notes and records |

## What is inside, by kind
- `**/*.md` — the reports, run logs and archived context files listed above.
- `**/*.json` — the raw timing records (`campaign.json`, one record per cell × arm × workload with medians, samples, cold calls, digests and the resolved route), the round summaries and the gates' outputs. These are the primary evidence; the markdown renders them.
- `**/*.log`, `**/*.txt`, `**/*.csv`, `**/*.dot`, `**/*.npy`, `**/*.gz`, `**/*.cmd` — the runs' console logs, the profiler summaries, the compile-side dispatch graphs and the reduction traces those reports cite, plus the verbatim `.cmd` invocation records of the round-4 smoke cells.
- `**/*.npz` — the four round-4 eval-output captures under `r4/results/after/ab/{fallback,frozen}_save/` (a few kilobytes each), kept as that round's parity record.

## What was deliberately not carried over
Deleted with the code that used them, all still in git history: the harness's `*.py` (71 files, ~39 k lines, recoverable at `6720723`), its `*.sh` drivers, the legacy `setup_legacy.sh` build script, the 6 IREE CUDA-module `.ptx` dumps under `r5/results/q2_gpu/dumps/*/executables/` (compiler output, summarized by the sibling `dumps/*.json` + `dispatch_graph.dot` and by `r5/RCA.md` §5.1), and the R2 cross-side fixture `fixtures/forest200_tree64_random.npz` (an input of the deleted `bench_gp.py`; its sanity record `fixtures/sanity_eval.json` stays).
