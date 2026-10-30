# `EVOGP_EVAL_MODE=fused` — independent verification campaign (Triton whole-population route)

Report of record for an independent re-measurement of the `fused` route (`fused_engine.py` + `fused_kernels.py`,
wired at the five call sites in `forest.py` / `tree.py` through `dense_engine.resolve_route`).
Measured **2026-10-30** on **AMD Instinct MI300X VF** (191.7 GB HBM), torch **2.13.0+rocm7.1**, triton 3.7.1,
iree-base 3.11.0, `/tmp/rocvenv/bin/python` 3.12.3, `evoxc` from `/root/evoxc`.
Every record lives next to this file under `fused_results/`; the raw evidence is JSON, nothing here is an estimate.

Protocol (identical to `test/benchmark/evoxc/results/t1a1_baseline/BASELINE.md`): one process per (cell, arm),
seed 0, `common.build_workload(cell)` (`Forest.random_generate`, `x = rand(N, input_len)`), `warmup 2` untimed,
`reps 5` sync-bracketed samples, **warm medians**, the cold first call reported separately and never inside the
medians, idle-device guard per config (all 20 configs report `gpu_idle=True`), `EVOXC_DISABLE_LOCATIONS=1`,
`TRITON_CACHE_DIR` pointed at a writable path (the box's `$HOME` is read-only, so the default `/root/.triton` dies).
Ratio convention: **`torch-batched median / arm median`**, so > 1.00 means the arm is faster than the torch baseline.
Cells: `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p256-L64-i32-N4096`, `p1000-L64-i32-N1024`,
`p4096-L64-i32-N1024`, `p2048-L128-i32-N1024-d7` (123-node trees, `max_layer_cnt 7`).

Campaign size: **20 timing configs, 0 failures** (77 cell×arm×workload entries in `fused_results/campaign.json`;
the merged `fused_results/main/campaign.json` indexes the four driver invocations), plus 6 per-cell probes,
6 cold-cache probes and one route-validation run (18 cases).
The harness's own gate over the records (`fused_gate.py` → `check.py --allow-missing --allow-digest-mismatch`, the
same two allowances the baseline campaign needs) reads **20 records / 574 checks / 0 failures, 20/20 configs on an
idle device, 0 contaminated**; its digest warnings are exactly the NaN-payload / float32-reduction splits described
in §6, never a value mismatch.

---

## 0. Verdict in one paragraph

The route is real and it is fast: **41×–157× the harness's own `torch-batched` arm on the value surface and
32×–131× on the loss surface**, 8.6×–663× the shipped `dense` default, and it costs **≈2.4–2.6 ms of operator time
out of a 2.6–2.9 ms GP generation** (i.e. evaluation is now 9 % of a generation). Value outputs are **bit-identical**
to both the harness arm and the plain-torch oracle on every cell (the only differing bits are NaN payloads;
finite-space `max_abs = 0.0` everywhere), losses agree to **≤ 3.8e-7 relative**. A GP loop pays **no per-generation
compile and no re-prep** (5/5 transforms hit, 5/5 scratch reuses, 0 compiles, 0 declines). Cold start is a one-time
**1.64–1.67 s (value) + 0.94–0.99 s (loss) + 0.32 s (per-row)** Triton JIT from an empty cache — 4× cheaper than a
dense compile and 25–500× cheaper than a `batch` merged compile. The one thing to fix is cross-node bookkeeping:
`src/evogp/tests/test_dense_policy.py::test_eval_mode_tables_stay_canonical` still pins the pre-fused
`EVAL_MODES`/`ENGINES` tuples and is the **single** failing test in the suite (1206 passed / 7 skipped).

---

## 1. The route really ran fused (not a silent fallback)

A fused decline is invisible through the public API: `Forest.*` catches `FusedUnsupported` and continues on
`route.fallback_backend`, which with an explicit `iree` backend is `specialized`+`iree` (compile-bound, minutes on a
fresh population, and it would still look plausible). Two independent proofs are recorded per cell:

- **Policy**: `dense_engine.resolve_route(pop, rows=…)` returns `engine="fused"`, `use_fused=True`,
  `backend="iree"`, `fallback_backend="iree"` for all three call shapes (`sr_fitness`, `batch_forward`, `forward`) on
  all six cells — `route` block, `route_all_fused: true` in every probe record.
- **Execution**: the 13-key `fused_engine.fused_cache_stats()` delta around each workload shows
  `fused_kernel_launches == 1 + warmup + reps`, `fused_declines == 0`, `fused_scratch_allocs + fused_scratch_reuses
  == launches`, and `fused_xt_hits + fused_xt_misses == launches` on the two `per_row=False` surfaces
  (`served: true` on every workload of every cell). A fallback moves `fused_calls` but leaves `fused_kernel_launches`
  frozen — which is why the probe asserts on launches, never on calls.

The `evoxc_engine.cache_stats()` block in these records is all-zero by construction (the route performs no evoxc
compile), so a fused record's `cache.compile_misses_first_call = 0` must **not** be read as "the cold call was free".

---

## 2. Four-way warm medians (ms) and ratios

`bf` = `Forest.batch_forward(x)` `(pop, N, 1)`; `sf` = `Forest.SR_fitness(x, labels)` `(pop,)`;
`fw` = `Forest.forward(x_per_row)`. All arms through the public API; the torch reference is
`arms_torch.BatchedEvaluator` (the harness's own implementation).

### `p256-L32-i8-N1024`
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 4.0547 | 1.00× | 4.1639 | 1.00× | 4.0248 | 1.00× |
| **`fused`** | **0.0995** | **40.8×** | **0.1299** | **32.1×** | **0.0871** | **46.2×** |
| `dense` (iree) | 1.9857 | 2.04× | 1.1185 | 3.72× | 0.6651 | 6.05× |
| `batch` (iree) | 3.3566 | 1.21× | 3.4803 | 1.20× | 2.0029 | 2.01× |

### `p256-L64-i32-N1024`
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 4.4274 | 1.00× | 4.5626 | 1.00× | 4.4368 | 1.00× |
| **`fused`** | **0.1016** | **43.6×** | **0.1399** | **32.6×** | **0.0911** | **48.7×** |
| `dense` (iree) | 2.8460 | 1.56× | 2.0800 | 2.19× | 0.8475 | 5.24× |
| `batch` (iree) | 3.4437 | 1.29× | 3.6764 | 1.24× | 1.8506 | 2.40× |

### `p256-L64-i32-N4096`
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 4.9177 | 1.00× | 4.9519 | 1.00× | 4.2940 | 1.00× |
| **`fused`** | **0.1095** | **44.9×** | **0.1282** | **38.6×** | **0.0857** | **50.1×** |
| `dense` (iree) | 7.8461 | 0.63× | 5.8490 | 0.85× | 0.8995 | 4.77× |
| `batch` (iree) | not scheduled | — | — | — | — | — |

### `p1000-L64-i32-N1024`
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 4.8867 | 1.00× | 4.9105 | 1.00× | 4.7694 | 1.00× |
| **`fused`** | **0.1053** | **46.4×** | **0.1435** | **34.2×** | **0.0956** | **49.9×** |
| `dense` (iree) | 7.3829 | 0.66× | 5.6936 | 0.86× | 0.8690 | 5.49× |
| `batch` (iree) | 6.9349 | 0.70× | 7.2061 | 0.68× | 2.4343 | 1.96× |

### `p4096-L64-i32-N1024`
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 10.6400 | 1.00× | 10.6602 | 1.00× | 4.5302 | 1.00× |
| **`fused`** | **0.1460** | **72.9×** | **0.1786** | **59.7×** | **0.1070** | **42.3×** |
| `dense` (iree) | not scheduled in this campaign — the four-way arm set is the three cells above plus the d7 cell; baseline of record 45.97 / 37.25 ms | | | | | |
| `batch` (iree) | not scheduled — baseline of record 21.98 / 21.41 ms warm at a 65.6 s / 154.9 s cold compile | | | | | |

### `p2048-L128-i32-N1024-d7` (123-node trees)
| arm | bf ms | vs torch | sf ms | vs torch | fw ms | vs torch |
|---|---|---|---|---|---|---|
| `torch-batched` | 26.2510 | 1.00× | 26.2540 | 1.00× | 16.1361 | 1.00× |
| **`fused`** | **0.1673** | **156.9×** | **0.2005** | **130.9×** | **0.1275** | **126.6×** |
| `dense` (`EVOGP_DENSE_MAX_TREE_LEN=128`, the `auto` cap raise) | 110.9945 | 0.24× | 105.7609 | 0.25× | 1.2759 | 12.6× |
| `batch` (iree) | not scheduled (baseline of record: 75.5 s / 164.1 s cold compile, 19.68 / 20.90 ms warm) | | | | | |

**fused against the shipped default (`dense`), warm:** 20.0× / 8.6× (`bf`/`sf`) at `p256-L32`, 28.0× / 14.9× at
`p256-L64`, 71.7× / 45.6× at `N=4096`, 70.1× / 39.7× at `p1000`, and **663× / 527×** on the d7 cell where dense runs
at the raised trace cap 128. **Against `batch`:** 33.7× / 26.8× (`p256-L32`), 33.9× / 26.3× (`p256-L64`),
65.9× / 50.2× (`p1000`).

**Row axis is nearly free:** `bf` goes 0.1016 → 0.1095 ms for 4× the rows (`p256`, N 1024 → 4096), and the
`N=4096` cell is still 44.9× the torch arm. **Pop axis costs the tree walk:** 0.0995 → 0.1460 ms from pop 256 to
4096 (same tree length), which is the documented "the tree walk repeats once per row block, and per-program work
scales with pop" limit.

---

## 3. Where a warm call goes (stage split)

Kernel = the raw `fused_kernels.launch_value` / `launch_loss` median; host = the full facade call minus that kernel
(route resolution + `_prepare` + scratch lookup + `contiguous` epilogue); transpose = the un-memoized
`x.t().contiguous()` the memo replaces (warmup 2, reps 10). This route has **no readback stage**: both the value
buffer and the `(pop, n_rb)` loss chart stay device-resident, so the honest split is three buckets, not four.

| cell (rows) | value: kernel ms | host ms | kernel share | loss: kernel ms | host ms | kernel share | transpose(cold) ms | n_rb |
|---|---|---|---|---|---|---|---|---|
| `p256-L32` (1024) | 0.0604 | 0.0359 | 62.7 % | 0.0615 | 0.0549 | 52.8 % | 0.0192 | 4 |
| `p256-L64` (1024) | 0.0626 | 0.0379 | 62.3 % | 0.0640 | 0.0619 | 50.8 % | 0.0196 | 4 |
| `p256-L64` (4096) | 0.0736 | 0.0328 | 69.2 % | 0.0751 | 0.0455 | 62.3 % | 0.0189 | 16 |
| `p1000-L64` (1024) | 0.0702 | 0.0374 | 65.2 % | 0.0715 | 0.0501 | 58.8 % | 0.0200 | 4 |
| `p4096-L64` (1024) | 0.1078 | 0.0314 | 77.5 % | 0.1039 | 0.0469 | 68.9 % | 0.0191 | 4 |
| `p2048-L128-d7` (1024) | 0.1314 | 0.0365 | 78.3 % | 0.1322 | 0.0477 | 73.5 % | 0.0191 | 4 |

The kernel is 51–78 % of a warm call and the host glue **22–49 %** — a much healthier ratio than the dense route
(host glue ≈ 15 %, but on a 1–7 ms call). Absolute host glue is flat at 0.031–0.062 ms across three orders of
population size, i.e. it is Python-level per-call overhead, not data movement; the remaining headroom in the route
is host-side. `transpose(cold)` 0.019 ms is what the identity memo removes from every warm call.

---

## 4. The GP loop (required in this campaign, absent from the previous report)

`arms.make_pipeline` over the cell descriptor (`DefaultCrossover`, `DefaultMutation(rate 0.2,
max_layer_cnt 3)`, `DefaultSelection(survival 0.3, elite 0.01)`), `gp_warmup 1`, `gp_reps 5`, one `step()` per sample,
sync-bracketed; the population is replaced every generation.

| cell | arm | generation total ms | eval ms | ops ms | vs torch | vs dense | best fitness |
|---|---|---|---|---|---|---|---|
| `p256-L32` | `torch-batched` | 23.358 | 20.636 | 2.723 | 1.00× | — | −0.084110 |
| | **`fused`** | **2.647** | **0.241** | 2.406 | **8.82×** | **2.03×** | −0.084110 |
| | `dense` | 5.374 | 2.919 | 2.455 | 4.35× | 1.00× | −0.084110 |
| `p256-L64` | `torch-batched` | 28.394 | 25.772 | 2.621 | 1.00× | — | −0.084197 |
| | **`fused`** | **2.637** | **0.236** | 2.401 | **10.77×** | **2.55×** | −0.084197 |
| | `dense` | 6.731 | 4.203 | 2.529 | 4.22× | 1.00× | −0.084197 |
| `p256-L64-N4096` | `torch-batched` | 25.100 | 22.569 | 2.531 | 1.00× | — | −0.084161 |
| | **`fused`** | **2.679** | **0.234** | 2.446 | **9.37×** | **3.99×** | −0.084161 |
| | `dense` | 10.685 | 8.071 | 2.614 | 2.35× | 1.00× | −0.084161 |
| `p1000-L64` | `torch-batched` | 68.019 | 65.437 | 2.582 | 1.00× | — | −0.084145 |
| | **`fused`** | **2.898** | **0.263** | 2.636 | **23.47×** | **4.37×** | −0.084145 |
| | `dense` | 12.677 | 10.140 | 2.537 | 5.37× | 1.00× | −0.084145 |
| `p4096-L64` | `torch-batched` | 296.399 | 293.719 | 2.680 | 1.00× | — | −0.084252 |
| | **`fused`** | **2.730** | **0.255** | 2.475 | **108.6×** | — | −0.084252 |
| `p2048-L128-d7` | `torch-batched` | 232.483 | 229.598 | 2.886 | 1.00× | — | −0.084304 |
| | **`fused`** | **2.747** | **0.248** | 2.499 | **84.6×** | **45.3×** | −0.084304 |
| | `dense` (cap 128) | 124.530 | 121.820 | 2.711 | 1.87× | 1.00× | −0.084304 |

Two structural facts matter more than the ratios:

1. **Evaluation is no longer the loop.** On the fused route a generation is **8.7–9.1 % evaluation and 90.9–91.3 %
   operators** (0.23–0.26 ms eval against 2.39–2.64 ms of crossover/mutation/selection); the generation cost is flat
   in pop and tree depth from 256×32 to 4096×64 to 2048×128. The route has moved GP's bottleneck to the operator
   layer exactly as the dense route never could.
2. **No per-generation compile and no re-prep.** Per-step counters (`probes/*.json` → `gp_step.per_generation`):
   `kernel_launches_per_step = [1,1,1,1,1]`, `kernel_compiles_total = 0` (all five warm steps),
   `xt_hits_total = 5`, `xt_misses_total = 0` (the data block is the same object, so the transpose memo hits),
   `scratch_allocs_total = 0`, `scratch_reuses_total = 5`, `declines_total = 0` — on every cell. A changed population
   changes only the node tensors the kernel reads; nothing is re-planned, re-traced or recompiled. This is the
   structural reason the route wins a turnover loop where the `batch` route loses 17.8–71.5× to dense.

---

## 5. Cold start

Two honest numbers, because they answer different questions.

**A. First call in a fresh process, warm Triton disk cache** (what a repeat user pays; `first_call_ms` of the
timing records). Because the specialization key is `(per_row, reduce, use_mse, block_rows, num_warps)` and does not
depend on the cell's shape, the first cell that builds a key pays the load and later cells reuse it:

| cell | bf first call | sf first call | fw first call |
|---|---|---|---|
| `p256-L32-i8-N1024` | 2.1 ms | 469.2 ms | 1.7 ms |
| `p256-L64-i32-N1024` | 953.2 ms | 1400.9 ms | 330.7 ms |
| `p256-L64-i32-N4096` | 2.2 ms | 461.8 ms | 1.7 ms |
| `p1000-L64-i32-N1024` | 2.1 ms | 473.0 ms | 1.7 ms |
| `p4096-L64-i32-N1024` | 2.1 ms | 478.1 ms | 1.8 ms |
| `p2048-L128-i32-N1024-d7` | 2.1 ms | 469.4 ms | 1.7 ms |

The 2.1 ms entries are a warm **disk** hit for that key (Triton's cache is process-independent) and the 953/1401 ms
entries are the one config in the campaign that built the `tree_len`-dependent value and loss specializations for
the first time. Within one process the second surface pays ~0.47 s (loss) or ~1.7 ms (per-row), never 1.4 s.

**B. Cold Triton cache** (fresh `TRITON_CACHE_DIR` per cell, one call per surface, one process per cell — the
worst case a first-ever user sees): **`batch_forward` 1 642–1 674 ms**, **`sr_fitness` 940–991 ms**,
**`forward` 317–330 ms**, i.e. **≈2.9 s of one-time JIT for all three surfaces** and 1.64 s for the value kernel
alone. For scale: the shipped dense route compiles in 3.2–6.6 s per (shape, id-set) and 14.3 s at trace cap 128,
and the `batch` route's merged compile is 2.7–164 s. The fused cold cost is paid **once per kernel specialization
per machine**, never per population and never per generation — the opposite of the `batch` route's economics.

End-to-end sanity check (the documented install-validation run, `python -m evogp.sr_test`):
**fused −0.2259 in 10.36 s wall** vs the default `auto` route **−0.2259 in 16.06 s** on the same box — same answer,
1.55× faster end to end, because the run skips the 3–6 s dense compiles entirely.

---

## 6. Numeric parity on the timed populations

Probe records compare the fused result against (i) the harness's `torch-batched` arm on the *same* workload tensors
(full population) and (ii) `src/evogp/tests/reference_eval.py` (the plain-torch oracle) on the first 32 trees.
`max_rel = max |cand − ref| / max(|ref|, 1)` over finite pairs. `nan/inf mismatch` counts placement disagreements.

| cell | surface | vs `torch-batched` max_abs / max_rel | nan/inf mismatch | vs oracle max_abs / max_rel |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 1.05e+06 / 3.44e-07 | 0 / 0 | 1.20e+01 / 2.88e-07 |
| `p256-L64-i32-N1024` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 3.28e+04 / 2.29e-07 | 0 / 0 | 2.56e+02 / 3.43e-07 |
| `p256-L64-i32-N4096` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 2.68e+08 / 2.55e-07 | 0 / 0 | 6.40e+01 / 2.26e-07 |
| `p1000-L64-i32-N1024` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 8.39e+06 / 3.19e-07 | 0 / 0 | 2.56e+02 / 3.43e-07 |
| `p4096-L64-i32-N1024` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 1.68e+07 / 3.78e-07 | 0 / 0 | 2.56e+02 / 3.43e-07 |
| `p2048-L128-i32-N1024-d7` | `batch_forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `forward` | 0.000e+00 / 0.000e+00 | 0 / 0 | 0.000e+00 / 0.000e+00 |
| | `sr_fitness` | 2.75e+11 / 3.13e-07 | 0 / 0 | 8.19e+03 / 2.42e-07 |

Reading the zero rows correctly: `validate.compare` reports `bitwise_equal: false` on several of them because it
calls `torch.equal` on tensors that contain NaN, not because a value differs. A direct bit-pattern check on the
timed populations confirms it — `p256-L32` `batch_forward`: **0 mismatching bit patterns of 262 144**;
`p1000` 1 024 of 1 024 000 and `p2048-d7` 16 384 of 2 097 152 mismatches, **all of them NaN payload bits**
(`torch.isnan` on every mismatched element, `nan_to_num` equal), with finite-space `max_abs = 0.0`. So the value
surfaces are bit-identical on every finite point and differ only in NaN payloads / signed zeros — the same class
`fused_check` documents.

**Route validation across function sets** (`validate.py`, fused arm, 6 cells × `arith`/`rich`/`ternary`, 32 oracle
trees; `fused_results/validation-fused.json`):

- `batch_forward` and `forward` vs the **oracle**: `max_abs = 0.000e+00` on **all 18 cases**, including the
  `rich` (transcendental) set. This is a genuinely new result: on `rich`, the *torch* arms deviate from the oracle
  by up to 4.8e-4 (rel 1.8e-3) while fused is exact — the f64-through-libdevice lowering is the correctly-rounded
  path, and it removes the documented `rich`-set caveat for this route (`dense`/`iree` deviates up to 4.4e-1).
- `sr_fitness` vs oracle: `max_rel ≤ 3.4e-7` on every case (worst absolute 8.19e+03 at d7, on losses of order
  1e+12), `nan_mismatch = 0`, `inf_mismatch = 0` everywhere.
- `arith`-set digests therefore differ from the other routes on `sr_fitness` only (float32 block-reduction order);
  a harness gate over a fused column needs `--allow-digest-mismatch`, exactly as the batch route does.

---

## 7. Honest residual gaps (with numbers)

1. **One stale test is the only suite failure.** `pytest src/evogp/tests -q` → **1 failed, 1206 passed, 7 skipped,
   0 errors in 118.0 s**; the failure is `test_dense_policy.py::test_eval_mode_tables_stay_canonical`, which pins
   `EVAL_MODES == ("dense","specialized","batch","auto")` / `ENGINES == ("specialized","dense","batch")` while
   `dense_engine` now ships `("dense","specialized","batch","fused","auto")` /
   `("specialized","dense","batch","fused")`. It is a cross-node edit (see `fused_sibling_patches.md` §1), not a
   library defect. Everything else passes, and `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -c "import
   evogp.tree"` works with `torch.cuda.is_available() == False`, `fused_available() == False` and
   `resolve_route(256, eval_mode="fused").engine == "specialized"` — the route degrades to the policy's CPU answer
   and the import needs no GPU.
2. **No pytest covers the fused route at all** (`rg "fused" src/evogp/tests/` is empty apart from the stale tuple
   pin). The gate is `fused_check.py`'s stages plus this campaign; a smoke test that drives one 64-population cell
   through `Forest.SR_fitness` with `EVOGP_EVAL_MODE=fused` and asserts `fused_kernel_launches` moved would be the
   cheapest regression net.
3. **Loss ratios in this campaign land 4–17 % below the previously reported ones** while the value surface matches:
   value 40.8/43.6/44.9/46.4/72.9/156.9× here against 38.9/45.8/42.8/45.5/71.0/153.5× reported (all within ±5 %);
   loss 32.1/32.6/38.6/34.2/59.7/130.9× here against 33.5/39.1/40.7/38.4/69.0/146.6×. The absolute loss medians here
   are 0.1299–0.2005 ms. Two candidate causes, both benign and neither excluded: a 0.1 ms-scale measurement with 5
   reps is sensitive to GPU clock state, and this campaign's own `torch-batched` reference also ran 2–3 % slower than
   the baseline of record (4.0547 vs 4.0044 ms `bf` at `p256-L32`). Do not quote the loss path tighter than
   "32–131× in this campaign, up to ~147× measured previously".
4. **Host glue is now 22–49 % of a warm call** (0.031–0.062 ms) — route resolution, `_prepare`, the scratch lookup
   and the `contiguous` epilogue. The kernel itself is 0.06–0.13 ms. Further gains need host-side work, not kernel
   work; and the value path still has no device-resident output (a `(pop, N, 1)` tensor stays allocated, 1 MB –
   8 MB, but is not read back).
5. **Single output only.** `output_len != 1` declines by construction, so the multi-output `CombinedForest` route
   (`CustomLoss`) cannot use fused today; the decline is clean (no kernel compiled, no scratch allocated).
6. **Pop-1 trees never use it under the default policy** (`pop_size < EVOGP_FUSED_MIN_POP` = 64 with no explicit
   backend resolves to `specialized`+`numpy`), and **`EVOGP_EVOXC_BACKEND=numpy` still reports `engine="fused"`
   while every call declines** — the engine name is not a served-call proof. This campaign therefore asserts on
   `fused_kernel_launches`; the harness's own route reporting cannot yet distinguish the two (`fused_sibling_patches.md` §3/§4).
7. **One process pays the JIT again** (`fused_kernel_compiles` counts a first launch per specialization even when
   Triton serves it from disk in 2.1 ms), and `clear_fused_cache()` resets the facade counters but not
   `fused_kernels._KERNELS["specializations"]`, so a post-reset repeat can under-report compiles.
8. **This box needs `TRITON_CACHE_DIR`.** `$HOME` is read-only; without the variable the first launch dies with
   `OSError: [Errno 30] Read-only file system: '/root/.triton'`. Nothing in `src/` sets or validates it.

---

## 8. Reproduce

```bash
cd <repo root>
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
export EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/tf-cache        # required on this box ($HOME is read-only)
PY=/tmp/rocvenv/bin/python

# 1. timing campaign — one process per (cell, arm), idle-device guard, warm medians
#    (fused_campaign.py registers evoxc-fused-iree / evoxc-fused into the harness registry, then calls campaign.py;
#     fused_gate.py does the same for check.py. The harness files themselves are untouched.)
$PY src/evogp/tree/fused_results/fused_campaign.py --out src/evogp/tree/fused_results/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms torch-batched,evoxc-fused-iree --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY src/evogp/tree/fused_results/fused_campaign.py --out src/evogp/tree/fused_results/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024 \
    --arms evoxc-dense-iree --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY src/evogp/tree/fused_results/fused_campaign.py --out src/evogp/tree/fused_results/main \
    --cells p2048-L128-i32-N1024-d7 --arms evoxc-dense-iree-raised \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY src/evogp/tree/fused_results/fused_campaign.py --out src/evogp/tree/fused_results/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p1000-L64-i32-N1024 \
    --arms evoxc-batch-iree --workloads sr_fitness,batch_forward,forward \
    --warmup 2 --reps 5 --seed 0 --guard-wait 60 --timeout 1800

# 2. per-cell probe: route proof, served-call counters, numerics (torch-batched + CPU oracle),
#    kernel/host/transpose split, cold first calls, and the GP generation loop
export EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree
for c in p256-L32-i8-N1024 p256-L64-i32-N1024 p256-L64-i32-N4096 p1000-L64-i32-N1024 p4096-L64-i32-N1024 p2048-L128-i32-N1024-d7; do
  $PY src/evogp/tree/fused_results/fused_probe.py --cell $c --gp --oracle-trees 32 \
      --out src/evogp/tree/fused_results/probes/$c.json
done

# 3. cold Triton cache: a fresh TRITON_CACHE_DIR per cell, one call per surface
i=0; for c in p256-L32-i8-N1024 p256-L64-i32-N1024 p256-L64-i32-N4096 p1000-L64-i32-N1024 p4096-L64-i32-N1024 p2048-L128-i32-N1024-d7; do
  i=$((i+1)); rm -rf $TMPDIR/tf-cold-$i; TRITON_CACHE_DIR=$TMPDIR/tf-cold-$i \
  $PY src/evogp/tree/fused_results/fused_probe.py --cell $c --cold-cache \
      --out src/evogp/tree/fused_results/cold/$c.json
done

# 4. numeric validation against the plain-torch oracle, three function sets
$PY test/benchmark/evoxc/validate.py \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --func-sets arith,rich,ternary --max-oracle 32 \
    --out src/evogp/tree/fused_results/validation-fused.json

# 5. index (one campaign.json entry per cell × arm × workload) + the markdown summary
$PY src/evogp/tree/fused_results/campaign_index.py

# 6. end-to-end and the hermetic suite (run the suite AFTER all timing — parts of it touch the GPU)
EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree $PY -m evogp.sr_test     # best fitness -0.2259 in ~10.4 s
$PY -m evogp.sr_test                                                    # default auto: -0.2259 in ~16.1 s
PYTHONPATH=src:/root/evoxc $PY -m pytest src/evogp/tests -q             # 1 failed / 1206 passed / 7 skipped
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONPATH=src:/root/evoxc $PY -c "import evogp.tree"
```

`.log` worker files are not committed; only the JSON records, the probes and this report are.

---

## 9. Where these records belong

The campaign is benchmark evidence for the `evoxc` harness, so its home is
**`test/benchmark/evoxc/results/t1a1_fused/`** (the sibling of `results/t1a1_baseline/`), with `main/` as the
gateable directory and `probes/`, `cold/`, `validation-fused.json` as sibling columns.
It is parked here, under `src/evogp/tree/fused_results/`, **only because `test/` is outside this node's write
scope**; the parent agent should move (or copy) the directory there and add the fused arm to the harness registry
(`fused_sibling_patches.md` §2–§4). Nothing in the numbers depends on the location.

## 10. Files

| path | content |
|---|---|
| `fused_results/campaign.json` | index: one entry per cell × arm × workload (77 entries, 0 failed, 0 missing) |
| `fused_results/campaign.md` | the same index rendered as tables |
| `fused_results/main/<cell>/<arm>.json` | harness-schema timing records (workloads, medians, samples, cold calls, digests, routes, `gp_step`) |
| `fused_results/main/campaign.A.json`, `.B1.json`, `.B2.json`, `.C.json` | the four driver invocations' own `campaign.json` files |
| `fused_results/probes/<cell>.json` | route proof, served-call counter deltas, numerics vs arm + oracle, stage split, cold calls, GP loop |
| `fused_results/cold/<cell>.json` | cold-Triton-cache first calls (one fresh cache dir per cell) |
| `fused_results/validation-fused.json` | `validate.py` output, 18 cases × 15 checks |
| `fused_results/fused_campaign.py`, `fused_gate.py`, `fused_probe.py`, `campaign_index.py` | the tooling that produced the above |
| `fused_sibling_patches.md` | the exact cross-node edits (test pin, harness arm registry, route reporting, worker counters, doc drift) |

## Measured detail (moved from tree/CONTEXT.md)

Facade bookkeeping and route-shape facts that this campaign's sections above rely on (no number below replaces one
stated above; where they overlap, the campaign's own tables win):

- `fused_cache_stats()` returns 13 keys: `fused_calls` / `fused_declines` (facade attempts / declines),
  `fused_xt_hits` / `fused_xt_misses` / `fused_xt_entries` / `fused_xt_maxsize` (the transpose table, touched ONLY
  by `per_row=False` and the loss path), `fused_kernel_launches` / `fused_kernel_compiles` / `fused_scratch_allocs`
  / `fused_scratch_reuses` / `fused_scratch_entries` (the launch side), plus the informational `fused_block_rows` /
  `fused_num_warps`.
- `fused_calls` counts ATTEMPTS (bumped before the try, `fused_engine.py:608` / `:747`), so a decline moves BOTH
  counters and a shape-contract `ValueError` moves `fused_calls` alone: proving a call was served means
  `Δcalls − Δdeclines == n` AND `Δfused_kernel_launches == n`, while `fused_kernel_compiles` is "first launch of
  this `(per_row, reduce, use_mse, block, warps)` key" — the key set lives in `fused_kernels._KERNELS` and is NOT
  reset by `clear_caches()`, so a counter reset can under-report it.
- `EVOGP_FUSED_HOST_MEMO` has NO per-call reader: `_default_host_memo` reads it once at import to size `XT_MEMO`
  (`fused_engine.py:296-305`), so a change needs a fresh process.
- There is NO per-stage timer surface on this route (unlike `batch_stages.py`): the split must be reconstructed by
  timing `_prepare` / `_transposed` / `fused_kernels.launch_value` / `launch_loss` separately, which is what
  `fused_check._stage_split` does for the value/loss kernel-vs-host columns; the route has no host readback, so a
  "readback" stage does not exist.
- An explicit `EVOGP_EVOXC_BACKEND` changes the route's shape: `iree` makes the FALLBACK `iree` as well
  (`Route(False, 'iree', 'iree', engine='fused')`, so a decline lands on per-tree iree compiles) and `numpy` still
  returns `engine='fused'` with `backend='numpy'`, which `_decline_if_numpy` refuses on every call — a probe
  printing only `resolve_route(pop, eval_mode='fused').engine` therefore cannot distinguish a served call from a
  guaranteed decline.
- The facade declines (never guesses) on no triton, no device, a host tensor, `backend != "iree"`,
  `output_len != 1`, `rows < 1`, `pop < 1` and the `EVOGP_FUSED_MAX_SCRATCH_MB` envelope, and maps
  `fused_kernels.FusedUnavailable` onto that decline so an optional route never raises `ImportError`.
- Prep is device-side only: `per_row=False` transposes `x` once into the identity-memoized `XT_MEMO` (a
  `DenseMemo`), `per_row=True` needs nothing.
- `fused_check.py` is ~1.9k lines on purpose: like `dense_adversarial_check.py` and `device_resident_check.py` it is
  one self-contained dev battery (parity sets, the golden cross-check, the decline matrix, the wiring gate, the
  six-cell timing and the geometry sweep share one workload builder and one comparison helper), so do not split it
  into modules.
- `--stage wired` is the wiring gate: it drives the three `Forest` call sites with `EVOGP_EVAL_MODE=fused` set
  in-process and asserts they equal the `specialized` reference (value bitwise or ≤1e-6 relative, loss ≤1e-5), that
  the facade SERVED each call (its counters move), that `resolve_route` names fused on `iree` with a `numpy`
  fallback while `pop_size == 1` / a population below `EVOGP_FUSED_MIN_POP` / a box with no device stay
  `specialized`+`numpy`, that `auto` never selects it, and that a 64-row `output_len=2` population declines into
  the fallback with no kernel compiled and no scratch buffer allocated; `--stage timing --wired` re-times the six
  cells through the same public calls. Recorded: 56/56 oracle comparisons, 10/10 goldens at 1e-4, 21/21
  decline/contract checks, 30/30 wiring checks.

The earlier six-cell sweep that preceded this campaign (kept for the record; `BLOCK_R` 256 / 1 warp): **48.6x-167.6x**
faster than the `torch-batched` arm on the value path, 43.4x-156.4x on the loss path, 56x-150x `per_row`; the kernel
was 72-82 % of a warm call (0.060-0.128 of 0.083-0.157 ms), the transpose 0.018-0.020 ms, and the cold first call
375 ms (warm Triton disk cache) to 1.38 s (cold) per specialization, ~1.9 ms for the next one.

### Facade surface and route wiring (relocated from tree/CONTEXT.md)
`fused_engine` is the facade in the dense shape: `FusedUnsupported(reason, device_fallback=False)`, `evaluate_fused`, `sr_fitness_fused`, `fused_available`, `fused_cache_stats()` (13 all-`fused_*` keys), `clear_fused_cache` (registered with `evoxc_engine._register_cache_reset`) and the knob readers `resolve_block_rows` / `resolve_num_warps` / `resolve_max_scratch_mb` / `resolve_min_pop`.
It declines (never guesses) on no triton, no device, a host tensor, `backend != "iree"`, `output_len != 1`, `rows < 1`, `pop < 1` and the `EVOGP_FUSED_MAX_SCRATCH_MB` envelope, and it maps `fused_kernels.FusedUnavailable` onto that decline so an optional route never raises `ImportError`.
Prep is device-side only: `per_row=False` transposes `x` once into the identity-memoized `XT_MEMO` (a `DenseMemo`, `EVOGP_FUSED_HOST_MEMO` 16 entries), `per_row=True` needs nothing.
Route wiring: `EVOGP_EVAL_MODE=fused` is named by `EVAL_MODES` and `ENGINES`, `Route.use_fused` reports it and `resolve_route`'s `fused` branch decides — on an explicitly named backend, or with a device usable from torch AND `pop_size >= EVOGP_FUSED_MIN_POP` (64); otherwise (and on a CPU box) the route IS specialized+`numpy` with `engine="specialized"`.
Its backend SPELLING is `iree` and its fallback is always specialized+`numpy`; an explicit `EVOGP_EVOXC_BACKEND` changes the shape (`iree` makes the fallback `iree` too, `numpy` still reports `engine='fused'` while every call declines — the engine name is never a served-call proof).
`Route.__post_init__` refuses `use_dense=True` with `engine="fused"`; `should_use_dense` is `False` for it; `resolve_fused_min_pop` is the single parser of `EVOGP_FUSED_MIN_POP` and delegates to `fused_engine.resolve_min_pop` through a LOCAL import so the module DAG stays one-way.
All five call sites carry the fused attempt between the dense one and the specialized tail, catching exactly `fused_engine.FusedUnsupported` and continuing on `route.fallback_backend`.
There is NO per-stage timer surface on this route (the split must be reconstructed by timing `_prepare` / `_transposed` / `fused_kernels.launch_value` / `launch_loss` separately, which is what `fused_check._stage_split` does), and there is no readback stage at all.
`import triton` happens only inside `fused_kernels.kernels()`, so `import evogp.tree` stays GPU/triton-free; `TRITON_CACHE_DIR` must be writable or the first launch dies with `OSError: [Errno 30] Read-only file system: '/root/.triton'` — on this box `$HOME` is read-only, so every fused run needs `TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache`.
`fused_check.py` is ~1.9k lines on purpose: like `dense_adversarial_check.py` and `device_resident_check.py` it is one self-contained dev battery (parity sets, the golden cross-check, the decline matrix, the wiring gate, the six-cell timing and the geometry sweep share one workload builder and one comparison helper), so do not split it into modules.
