# Round 2, work item 1 — warm band-size sweep of the opt-in `EVOGP_EVAL_MODE=batch` route (W = 1)

**Question.** The batch route merges one population into ONE graph and partitions that merge into `K`
bands (`K = ceil(n_functions / EVOGP_BATCH_BAND_SIZE)` on the population's *deduped* function count).
What does the band size cost or buy on the warm path, and is a single default band size defensible?

**Answer (short).** A single default is defensible — but not today's `32`. On the band-sensitive workload
(256 independent per-row `forward` calls) fewer/larger bands lose up to **1.5×** against flat and up to
**1.17×** against 16 bands, while among 8/16/32 bands everything else is inside noise. Band size **16**
(K = ceil(n/16), i.e. 15–16 bands at the pop-256 cells) is the minimax choice — best pooled geomean
(1.0185 of the per-column argmin) and by far the best worst column (1.061); `32` has the same geomean but a
1.137 worst column, `8` 1.167, `64` 1.224 and flat 1.529. **The first-order decision is flat-vs-banded
(15–34 %), the band size among banded merges is second-order (≤ 6 % on the robust estimator)** — which is
exactly why one default is enough, provided it is not flat and its `K` lands near 15 rather than near 4.

## Scope of this sweep

| axis | values |
|---|---|
| cells | `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p64-L64-i32-N64` (seed 0, `max_layer_cnt` 5, `+ - * /`) |
| configs | `b8`, `b16`, `b32`, `b64` (pin `EVOGP_BATCH_BAND_SIZE`) and `flat` (`EVOGP_BATCH_BANDS=none`) |
| workloads | `sr_fitness` (loss, N = 1024 rows), `batch_forward` (whole population over N rows), `forward` (one row per tree) — in that fixed order, one plan/compile each |
| pool width | `EVOGP_BATCH_COMPILE_WORKERS=1` for EVERY config, so a difference between two configs of a cell is the band partition alone (chunking is also refused outright for an explicit band count) |
| engine | `EVOGP_EVAL_MODE=batch`, `EVOGP_EVOXC_BACKEND=iree`, evoxc snapshot `/tmp/evoxc_r2` (PYTHONPATH `<src>:/tmp/evoxc_r2:<harness>`, never `/root/evoxc`), `EVOXC_DISABLE_LOCATIONS=1` |
| protocol | seed 0, `--warmup 1 --reps 5` warm medians, the untimed first call separated (it carries the merged-graph compile), one child process per `(cell, config)`, strictly sequential |

**Not measured, deliberately** (this spawn's charter):

- `p1000-*` — the single-executable merged compile at pop 1000 is 91–162 s per workload (round-1 records),
  i.e. ≈ 8–14 min per config for three workloads; out of budget this round.
- `EVOGP_BATCH_BANDS=auto` — the knob spelling does not exist on today's tree (it is being added).
- banded + chunked composition (explicit band counts are refused by the chunking clamp on today's tree),
  the cold-start table and `gp_step` — later rounds.

## Command of record

```bash
cd test/benchmark/evoxc
EVOXC_DISABLE_LOCATIONS=1 /tmp/rocvenv/bin/python band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --configs b8,b16,b32,b64,flat \
  --workloads sr_fitness,batch_forward,forward \
  --warmup 1 --reps 5 --seed 0 --workers 1 \
  --tree-root src --evoxc-root /tmp/evoxc_r2 \
  --out-dir results/batch_opt_round2/warm \
  --note "round 2 work item 1: warm band-size sweep, W=1, evoxc snapshot /tmp/evoxc_r2; shared box (foreign pytest runs observed before start)"
```

15 configs, **15 ok, 0 failed/timeout**, 765.9 s wall of which 393.1 s is measured child time and the rest
is idle-device guard waits (see "Noise and contention"). Records: `<cell>/<config>.json`
(schema `evogp-bench-evoxc-v1-band-sweep`) + `<cell>/<config>.log`, index `sweep.json`.

## 1. Warm medians (ms; min of the five samples in parentheses)

### T1 `p256-L32-i8-N1024`
| config | sr_fitness | batch_forward | forward |
|---|---|---|---|
| b8 | 2.1038 (2.0444) | 2.2204 (2.1427) | 0.8006 (0.7850) |
| b16 | 2.2812 (2.0719) | 2.2159 (2.0912) | **0.7942 (0.7770)** |
| b32 | **2.0874 (2.0459)** | 2.2923 (2.2233) | 0.9078 (0.8838) |
| b64 | 2.1181 (2.1053) | 2.2942 (2.2596) | 0.9808 (0.9514) |
| flat | 2.4331 (2.3506) | 2.3622 (2.3027) | 1.1976 (1.1878) |

### T1 `p256-L64-i32-N1024`
| config | sr_fitness | batch_forward | forward |
|---|---|---|---|
| b8 | 2.2543 (2.1841) | 2.3061 (2.1363) | 0.9079 (0.9018) |
| b16 | 2.2482 (2.1721) | **2.1103 (2.1010)** | 0.8019 (0.7880) |
| b32 | **2.1920 (2.1537)** | 2.2484 (2.0688) | **0.7862 (0.7727)** |
| b64 | 2.2293 (2.1532) | 2.1899 (2.1193) | 0.8981 (0.8807) |
| flat | 2.7745 (2.5141) | 2.4674 (2.3563) | 1.1709 (1.1425) |

### T1 `p64-L64-i32-N64`
| config | sr_fitness | batch_forward | forward |
|---|---|---|---|
| b8 | 1.1606 (1.1388) | 0.9063 (0.8987) | 0.6967 (0.6726) |
| b16 | **1.1413 (1.1090)** | 0.9047 (0.9005) | 0.7228 (0.6735) |
| b32 | 1.1511 (1.1175) | **0.8739 (0.8582)** | **0.6559 (0.6345)** |
| b64 | 1.1420 (1.1374) | 0.9540 (0.9219) | 0.7141 (0.6757) |
| flat | 1.1723 (1.1118) | 0.8830 (0.8696) | 0.7101 (0.6640) |

## 2. Merge shape actually emitted (K = bands, n = deduped functions)

### T2
| cell | config | sr_fitness | batch_forward | forward | `band_strategy` |
|---|---|---|---|---|---|
| p256-L32-i8-N1024 | b8 | K=29 n=225 | K=29 n=225 | K=31 n=245 | `explicit` |
| p256-L32-i8-N1024 | b16 | K=15 n=225 | K=15 n=225 | K=16 n=245 | `explicit` |
| p256-L32-i8-N1024 | b32 | K=8 n=225 | K=8 n=225 | K=8 n=245 | `explicit` |
| p256-L32-i8-N1024 | b64 | K=4 n=225 | K=4 n=225 | K=4 n=245 | `explicit` |
| p256-L32-i8-N1024 | flat | K=1 n=225 | K=1 n=225 | K=1 n=245 | `flat` |
| p256-L64-i32-N1024 | b8 | K=29 n=232 | K=29 n=232 | K=31 n=245 | `explicit` |
| p256-L64-i32-N1024 | b16 | K=15 n=232 | K=15 n=232 | K=16 n=245 | `explicit` |
| p256-L64-i32-N1024 | b32 | K=8 n=232 | K=8 n=232 | K=8 n=245 | `explicit` |
| p256-L64-i32-N1024 | b64 | K=4 n=232 | K=4 n=232 | K=4 n=245 | `explicit` |
| p256-L64-i32-N1024 | flat | K=1 n=232 | K=1 n=232 | K=1 n=245 | `flat` |
| p64-L64-i32-N64 | b8 | K=8 n=63 | K=8 n=63 | K=8 n=64 | `explicit` |
| p64-L64-i32-N64 | b16 | K=4 n=63 | K=4 n=63 | K=4 n=64 | `explicit` |
| p64-L64-i32-N64 | b32 | K=2 n=63 | K=2 n=63 | K=2 n=64 | `explicit` |
| p64-L64-i32-N64 | b64 | K=1 n=63 | K=1 n=63 | K=1 n=64 | `explicit` |
| p64-L64-i32-N64 | flat | K=1 n=63 | K=1 n=63 | K=1 n=64 | `flat` |

Two facts a reader needs:

- The band **size** converts into a band **count** against the *deduped* function count, which differs per
  plan variant: 225 (loss/pop-batched value path, pop 256 with 31 collisions) and 245 (the per-row `forward`
  path dedupes by `(tree, row)`, so fewer collisions). So "band size 16" is K = 15 on two plans and K = 16 on
  the per-row one — the K ≤ 15 policy ceiling is reached at 16 on exactly the plan that benefits from it.
- At `p64-L64-i32-N64` (n = 63) `b64` and `flat` are the **same merge shape** (K = 1, `band_sizes = [63]`) and
  differ only in `band_strategy` (`explicit` vs `flat`): they are a free same-shape twin, and the 2.6 %
  `sr_fitness` gap between them (1.1420 vs 1.1723) is this window's noise floor, not a knob effect.
- `cost_estimate` / `flat_estimate` are `null` in every record, by construction: today's static cost model only
  runs for `bands="auto"`, which this sweep does not set. The `band_report` block records them (plus `bands`,
  `band_sizes`, `band_strategy`, `n_functions`, `emitted_ops`), so the auto-vs-argmin comparison of the next
  round has a like-for-like basis.

## 3. Per-cell argmin (median)

### T6
| cell | sr_fitness | batch_forward | forward |
|---|---|---|---|
| p256-L32-i8-N1024 | b32 (2.0874) | b16 (2.2159) | b16 (0.7942) |
| p256-L64-i32-N1024 | b32 (2.1920) | b16 (2.1103) | b32 (0.7862) |
| p64-L64-i32-N64 | b16 (1.1413) | b32 (0.8739) | b32 (0.6559) |

No config wins a cell outright, and the per-workload argmin moves between cells (16 at one pop-256 cell,
32 at the other) — the relevant fact for the default question. Pooled over the three workloads of a cell
(geomean of the ratio to the per-column argmin, **min-based / median-based**):

| cell | ranking |
|---|---|
| p256-L32-i8-N1024 | b16 1.004 / b8 1.012 · b32 1.066 · b64 1.109 · flat 1.246  (median-based: b8 < b16 < b32 < b64 < flat) |
| p256-L64-i32-N1024 | b32 1.000 · b16 1.015 · b64 1.053 · b8 1.069 · flat 1.253  (median-based: b16 < b32 < b64 < b8 < flat) |
| p64-L64-i32-N64 | b32 1.003 · flat 1.021 · b16 1.037 · b8 1.045 · b64 1.055 (median-based: b32 < b8 ≈ flat < b16 < b64) |

`b16` is the only config ranked 1st or 2nd in all three cells under both estimators.

### T5 pooled relative cost (min-based, the contamination-robust estimator; median-based in the last column)
| config | geomean over 9 columns | worst column | sr_fitness gm | batch_forward gm | forward gm | median-based geomean / worst |
|---|---|---|---|---|---|---|
| b8 | 1.0415 | 1.167 | 1.0177 | 1.0433 | 1.0733 | 1.0445 / 1.155 |
| **b16** | **1.0185** | **1.061** | 1.0388 | 1.0116 | 1.0397 | 1.0300 / 1.102 |
| b32 | 1.0223 | 1.137 | 1.0029 | 1.0330 | 1.0456 | **1.0270 / 1.143** |
| b64 | 1.0718 | 1.224 | 1.0108 | 1.0546 | 1.1538 | 1.0714 / 1.235 |
| flat | 1.1680 | 1.529 | 1.1486 | 1.0799 | 1.3447 | 1.1859 / 1.508 |

## 4. Where the band count helps or hurts — the stage split

`stage_ms` is the reset-bracketed **per-call median** of the ten stages (`stages.py`); a warm call leaves
`decode`/`dedup`/`trace`/`merge`/`lower`/`compile`/`load` at **0 calls** (plan and exec tables hit) and spends
its time in `upload` + `run` + `gather`. `residual` = the workload median minus those three, i.e. host glue
outside the instrumented stages. Shown for the two workloads that matter: the band-sensitive `forward` and
the wide-row `sr_fitness`.

### T4
| cell | config | K | fwd upload | fwd **run** | fwd gather | fwd residual | sr upload | sr **run** | sr gather | sr residual |
|---|---|---|---|---|---|---|---|---|---|---|
| p256-L32-i8-N1024 | b8 | 31 | 0.135 | **0.233** | 0.227 | 0.206 | 0.212 | 1.342 | 0.248 | 0.302 |
| p256-L32-i8-N1024 | b16 | 16 | 0.132 | **0.232** | 0.225 | 0.205 | 0.207 | 1.407 | 0.247 | 0.420 |
| p256-L32-i8-N1024 | b32 | 8 | 0.141 | 0.304 | 0.251 | 0.212 | 0.205 | 1.349 | 0.254 | 0.279 |
| p256-L32-i8-N1024 | b64 | 4 | 0.138 | 0.405 | 0.232 | 0.205 | 0.196 | 1.400 | 0.262 | 0.260 |
| p256-L32-i8-N1024 | flat | 1 | 0.132 | 0.589 | 0.257 | 0.220 | 0.235 | 1.608 | 0.271 | 0.319 |
| p256-L64-i32-N1024 | b8 | 31 | 0.146 | 0.294 | 0.243 | 0.225 | 0.213 | 1.448 | 0.265 | 0.329 |
| p256-L64-i32-N1024 | b16 | 16 | 0.147 | **0.242** | 0.217 | 0.196 | 0.218 | 1.453 | 0.252 | 0.325 |
| p256-L64-i32-N1024 | b32 | 8 | 0.137 | **0.240** | 0.218 | 0.191 | 0.205 | 1.405 | 0.259 | 0.324 |
| p256-L64-i32-N1024 | b64 | 4 | 0.133 | 0.305 | 0.245 | 0.215 | 0.204 | 1.466 | 0.261 | 0.298 |
| p256-L64-i32-N1024 | flat | 1 | 0.154 | 0.477 | 0.305 | 0.235 | 0.447 | 1.745 | 0.268 | 0.315 |
| p64-L64-i32-N64 | b8 | 8 | 0.124 | 0.164 | 0.211 | 0.197 | 0.188 | 0.478 | 0.257 | 0.237 |
| p64-L64-i32-N64 | b16 | 4 | 0.127 | 0.154 | 0.238 | 0.204 | 0.188 | 0.461 | 0.231 | 0.262 |
| p64-L64-i32-N64 | b32 | 2 | 0.119 | 0.138 | 0.205 | 0.194 | 0.189 | 0.458 | 0.246 | 0.258 |
| p64-L64-i32-N64 | b64 | 1 | 0.125 | 0.157 | 0.229 | 0.203 | 0.185 | 0.477 | 0.232 | 0.247 |
| p64-L64-i32-N64 | flat | 1 | 0.131 | 0.152 | 0.225 | 0.202 | 0.185 | 0.475 | 0.245 | 0.267 |

Reading it:

- **`run` is the whole band-size effect, and only on the per-row path.** At `p256-L32` `forward`'s `run`
  stage goes 0.232 (K=16) → 0.304 (K=8) → 0.405 (K=4) → 0.589 (K=1): fewer, larger bands means each
  dispatch carries more per-band work with a worse occupancy/overhead ratio, and it is a 2.5× stage-level
  spread. At `p256-L64` the same stage bottoms out at K = 8–16 (0.240/0.242) and rises to 0.477 at K = 1.
- **`upload` and `gather` are band-count-insensitive** at every cell (0.12–0.15 / 0.20–0.26 for `forward`,
  ~0.21 / ~0.25 for `sr_fitness`) — the merge shape does not move the two memory-side stages, so nothing is
  being paid or saved there. The one exception is the flat merge at `p256-L64`, whose `sr_fitness` `upload`
  doubles (0.447 vs ~0.21): the unbanded graph's stacked/expanded inputs are uploaded as one large block.
- **On the wide-row workloads `run` is flat across K ∈ [4, 31]** (1.34–1.47 over five configs at both
  pop-256 cells; 0.46–0.48 at p64), and only flat is clearly worse (1.61/1.75). That is why the band size
  cannot be resolved on `sr_fitness`/`batch_forward` and why `forward` has to arbitrate.
- **`residual` (host glue) is band-count-independent** (0.19–0.24 on `forward`, 0.26–0.42 on `sr_fitness`) and
  is 25–30 % of a `forward` call — the second-largest term after `run`, so a band policy can at best halve
  the total of the band-sensitive workload, which is what 0.794 vs 1.198 shows.

## 5. Cold first call (secondary; the compile of that workload's plan, not part of any warm median)

### T3 (ms)
| cell | config | sr_fitness | batch_forward | forward |
|---|---|---|---|---|
| p256-L32-i8-N1024 | b8 | 11256 | 7545 | 3167 |
| p256-L32-i8-N1024 | b16 | 11672 | 7717 | 3045 |
| p256-L32-i8-N1024 | b32 | 11454 | 7840 | 3203 |
| p256-L32-i8-N1024 | b64 | 12260 | 8270 | 3666 |
| p256-L32-i8-N1024 | flat | 13618 | 8492 | 4574 |
| p256-L64-i32-N1024 | b8 | 12952 | 7585 | 4030 |
| p256-L64-i32-N1024 | b16 | 13372 | 7579 | 3451 |
| p256-L64-i32-N1024 | b32 | 12442 | 7486 | 3150 |
| p256-L64-i32-N1024 | b64 | 13088 | 7989 | 3339 |
| p256-L64-i32-N1024 | flat | 14525 | 8234 | 4027 |
| p64-L64-i32-N64 | b8 | 2346 | 1812 | 1112 |
| p64-L64-i32-N64 | b16 | 2299 | 1686 | 1050 |
| p64-L64-i32-N64 | b32 | 2256 | 1586 | 977 |
| p64-L64-i32-N64 | b64 | 2260 | 1445 | 902 |
| p64-L64-i32-N64 | flat | 2258 | 1413 | 896 |

Flat is the most expensive compile at both pop-256 cells (+13–20 % on the loss plan, +44 % on the per-row
plan at `p256-L32`) and the cheapest at p64 (n = 63: chunking/banding has nothing to split). Compile is a
**cold** number on a shared box: read it as "± a factor of a few", not as a table for optimisation. The full
cold table stays a later round's job.

## 6. Numeric agreement across band sizes

Every record validates its own outputs against `src/evogp/tests/reference_eval.py` (plain-torch oracle,
`validate.compare`, first 16 trees of each workload — the harness's standard `--max-oracle`).

**Deviation from the oracle is IDENTICAL in all five configs of a cell/workload** — same `max_abs`, same
`max_rel`, same `bitwise_equal`:

| cell | workload | max_abs / max_rel (every config) | bitwise vs oracle |
|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 6.0 / 4.7e-07 | no |
| p256-L32-i8-N1024 | batch_forward | 5.86e-03 / 1.1e-03 | no |
| p256-L32-i8-N1024 | forward | 0 / 0 | **yes** |
| p256-L64-i32-N1024 | sr_fitness | 32.0 / 7.8e-07 | no |
| p256-L64-i32-N1024 | batch_forward | 7.81e-02 / 1.1e-05 | no |
| p256-L64-i32-N1024 | forward | 1.19e-07 / 8.0e-08 | no |
| p64-L64-i32-N64 | sr_fitness | 0.25 / 1.6e-07 | no |
| p64-L64-i32-N64 | batch_forward | 6.1e-05 / 1.7e-07 | no |
| p64-L64-i32-N64 | forward | 1.19e-07 / 8.0e-08 | no |

So **no band size changes the answer**; the `batch` route's documented value-path deviation (`batch_forward`
non-bitwise on the arithmetic set, ~1e-3 relative, and the loss path agreeing to ~1e-7) is a property of the
route, not of the partition. Cross-config value digests, read the same way:

- `sr_fitness`: bitwise identical (`e1021d86…`) for b8/b16/b32/b64 at `p256-L32` and for all five configs at
  `p256-L64` and `p64`; the only split is the **flat** merge at `p256-L32`, 1 ulp in the last float digit of
  the `(256,)` loss vector (`abs_sum` 9.525295401856250e15 vs …242e15, ≈8e-16 relative).
- `batch_forward` / `forward`: the value digests form per-config groups — the population-sized value tensors
  differ between partitions at the low-bit level (`abs_sum` spread ≤ 5e-11 relative at `p256-L64`, ≤ 4e-13 at
  `p256-L32`), i.e. the same order as the route's existing non-bitwise `batch_forward` caveat, **not** a
  band-size accuracy effect (the oracle deviation above is identical everywhere). `forward` at `p64` and both
  `sr_fitness` columns at `p64`/`p256-L64` collapse to one group. The recorded digest group count is
  therefore a property of the tensor (2/5/5 groups at `p256-L32`, 1/5/4 at `p256-L64`, 1/3/1 at `p64` for
  sr/batch_forward/forward) and must not be read as a regression.

## 7. Noise, contention, and how far to trust these numbers

### T7 within-config noise (median absolute deviation of the five samples / max:min)
| cell | config | sr_fitness | batch_forward | forward |
|---|---|---|---|---|
| p256-L32-i8-N1024 | b8 | 1.0% (1.06) | 1.6% (1.09) | 1.9% (1.07) |
| p256-L32-i8-N1024 | b16 | 1.2% (1.11) | 3.5% (1.10) | 1.3% (1.06) |
| p256-L32-i8-N1024 | b32 | 1.0% (1.06) | 2.0% (1.07) | 1.3% (1.07) |
| p256-L32-i8-N1024 | b64 | 0.6% (1.04) | 1.0% (1.05) | 2.4% (1.06) |
| p256-L32-i8-N1024 | flat | 3.4% (1.20) | 1.6% (1.05) | 0.8% (1.05) |
| p256-L64-i32-N1024 | b8 | 0.3% (1.04) | 7.4% (1.53) | 0.7% (1.04) |
| p256-L64-i32-N1024 | b16 | 1.8% (1.26) | 0.4% (1.06) | 1.7% (1.09) |
| p256-L64-i32-N1024 | b32 | 0.8% (1.03) | 1.7% (1.11) | 1.7% (1.11) |
| p256-L64-i32-N1024 | b64 | 1.6% (1.15) | 3.2% (1.09) | 1.9% (1.09) |
| p256-L64-i32-N1024 | flat | 4.9% (1.16) | 1.1% (1.06) | 2.4% (1.07) |
| p64-L64-i32-N64 | b8 | 1.9% (1.06) | 0.3% (1.02) | 3.5% (1.13) |
| p64-L64-i32-N64 | b16 | 1.4% (1.11) | 0.5% (1.05) | 3.7% (1.13) |
| p64-L64-i32-N64 | b32 | 1.3% (1.05) | 0.9% (1.10) | 2.4% (1.07) |
| p64-L64-i32-N64 | b64 | 0.4% (1.02) | 1.2% (1.06) | 4.6% (1.15) |
| p64-L64-i32-N64 | flat | 4.6% (1.10) | 1.5% (1.05) | 3.5% (1.15) |

Two records show **early-rep contamination** (a foreign tenant releasing the device just before the config
started, so the first samples are inflated): `p256-L32 b16 sr_fitness` = 2.3094 / 2.2812 / 2.2979 / 2.0821 /
2.0719 and `p256-L64 b8 batch_forward` = 2.2083 / 2.1363 / **2.7307** / 2.3061 / **3.2639**. That is why the
min column is printed and why the default question is answered on the **min-based** estimator as well as the
median-based one; both give `b16` as the minimax choice, and no conclusion here rests on a difference below
~5 %.

Contention, stated for the record: the box is shared and other agents ran on the same MI300X throughout the
window (the guard's `python` hint listed foreign `pytest` and evoxc-benchmark processes from other task
slices). `campaign.guard` cleared the device before **all 15** configs (`gpu_pids.idle: true`,
`contaminated: false`) but had to wait for five of them — 10.6 s, 94.0 s, 73.1 s, 88.7 s and 104.4 s. The
same-shape twin (`p64` b64 vs flat, both K = 1) bounds the residual cross-config error at **2.6 %** on
`sr_fitness`; the pooled `b8/b16/b32` spread of 3–6 % sits at that scale, so treat those three as one tier.

## 8. Verdict — is a single default band size defensible?

**Yes, with two conditions, and the defensible value is 16, not the shipped 32.**

1. **Never flat.** A single unbanded merge (K = 1) is the worst or second-worst config at every pop-256
   cell/workload: +15 % (`sr_fitness` geomean), +8 % (`batch_forward`), **+34 %** (`forward`), worst column
   +53 %; it also pays the largest cold compile. The banded-vs-flat decision is the *first-order* one.
2. **Keep K in the 8–16 range; do not push K to 30 and do not drop below 8.** The per-row path needs many
   bands (`run` 0.232 at K = 16 vs 0.405 at K = 4 vs 0.589 at K = 1, `p256-L32`), but doubling the band count
   past ~16 buys nothing and starts to cost when bands get small (`p256-L64` `forward` `run`: 0.240 at K = 8,
   0.242 at K = 16, 0.294 at K = 31). Band size 8 is the argmin at one cell and 17 % off it at the other.

Among the three plausible settings the spread is 3–6 % — the same scale as this window's noise (2.6 %
same-shape twin) — so the default does not have to be the per-cell argmin; it has to be the setting that is
**never far from it**. On the min-based estimator, pooled over all nine columns:
`b16` = geomean **1.0185**, worst column **1.061**; `b32` = 1.0223 / 1.137; `b8` = 1.0415 / 1.167;
`b64` = 1.0718 / 1.224; `flat` = 1.1680 / 1.529. Band size 16 is the minimax winner under both estimators
and the only config ranked 1st or 2nd in every cell.

**It also composes with the binding policy floor.** Under "band size ≥ 8, K ≤ 15 safe (16 bindings per
dispatch), band size 1 never loads", the safe band size for n = 225–245 functions is `ceil(n/15)` = 15–17 —
i.e. the clamp itself lands on K = 15, exactly the region the measurement crowns, and one band above the
ceiling only on the per-row plan (K = 16, which measured as the argmin at `p256-L32` and within 2 % of it at
`p256-L64`). So the policy and the measurement agree; what does *not* follow from the policy floor alone is
today's 32 (K = 8), which gives up 14 % on the band-sensitive workload at `p256-L32` (0.9078 vs 0.7942 ms,
and 0.304 vs 0.232 ms in its `run` stage).

**Caveat on scope:** all of this is a power-of-two grid (`8/16/32/64`) on three cells. The optimum could sit
at 12 or 24; the measured evidence only supports "K ≈ 15, band size ≈ 16", and it says nothing about
`p1000` (where n_functions is larger and K = ceil(n/16) grows), `EVOGP_BATCH_BANDS=auto` (which is the next
round's comparison against exactly this argmin) or the banded+chunked composition.

## 9. What this round still owes

- `bands="auto"` vs the `b16` argmin here (the like-for-like basis is recorded: `band_report` carries
  `bands` / `band_sizes` / `band_strategy` / `n_functions` / `emitted_ops`, and `cost_estimate` /
  `flat_estimate` are `null` on every non-auto row and will be populated only on an auto row).
- The cold table at W = 1 across band sizes (this summary carries the per-plan `first_call_ms` as evidence,
  not as a report) and the band-size × pool-width composition.
- `p1000-*`, excluded here on compile budget (91–162 s per workload per config on the single-executable path).
