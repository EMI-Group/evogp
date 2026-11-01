> **Record note (post-2.0.0a2).** This document is the fused route's independent verification campaign and design record of record.
> It was measured on the pre-2.0.0a2 tree, whose torch-native stack (`evogp.tree`, `evogp.algorithm`, `evogp.problem`, `evogp.pipeline`) is now gone.
> Module names in its prose (`fused_engine.py`, `forest.py`, `tree.py`, `dense_engine.resolve_route`, `EVOGP_EVAL_MODE`) refer to that deleted wiring.
> The route itself now lives in `src/evogp/fused/` (`engine.py` + `kernels.py`) and is called directly by `evogp.gp.fused_eval`.
> The in-library battery `fused_check.py` became `python -m evogp.fused.check` (`fused/check.py` plus the `cases` / `contract` / `timing` libraries), and the independent oracle `reference_eval.py` moved to the dev-only `test/oracle/reference_eval.py`.
> The `fused_envelope.py` harness (§11–§15) was deleted rather than ported, so its commands no longer exist; its stored records stay under `test/benchmark/evoxc/results/t1a1_fused/` and were re-checked against today's rule at the sizes quoted below.
> The numbers below are unchanged and are no longer reproducible from this tree (the comparison harness is in `test/benchmark/evoxc/results/t1a1_auto_r3/`).

# `EVOGP_EVAL_MODE=fused` — independent verification campaign (Triton whole-population route)

Report of record for an independent re-measurement of the `fused` route (`fused_engine.py` + `fused_kernels.py`,
wired at the five call sites in `forest.py` / `tree.py` through `dense_engine.resolve_route`).
The route serves `output_len` 1..`fused_engine.MAX_OUT_LEN` (8): `out_len == 1` runs `_fused_kernel`, `out_len >= 2`
runs `_fused_multi_kernel` (§11).
Measured **2026-10-30** on **AMD Instinct MI300X VF** (191.7 GB HBM), torch **2.13.0+rocm7.1**, triton 3.7.1,
iree-base 3.11.0, `/tmp/rocvenv/bin/python` 3.12.3, `evoxc` from `/root/evoxc`.
Every record lives under `test/benchmark/evoxc/results/t1a1_fused/`, the harness's results tree; the raw evidence is JSON, nothing here is an estimate.

Protocol (identical to `test/benchmark/evoxc/results/t1a1_baseline/BASELINE.md`): one process per (cell, arm),
seed 0, `common.build_workload(cell)` (`Forest.random_generate`, `x = rand(N, input_len)`), `warmup 2` untimed,
`reps 5` sync-bracketed samples, **warm medians**, the cold first call reported separately and never inside the
medians, idle-device guard per config (all 20 configs report `gpu_idle=True`), `EVOXC_DISABLE_LOCATIONS=1`,
`TRITON_CACHE_DIR` pointed at a writable path (the box's `$HOME` is read-only, so the default `/root/.triton` dies).
Ratio convention: **`torch-batched median / arm median`**, so > 1.00 means the arm is faster than the torch baseline.
Cells: `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p256-L64-i32-N4096`, `p1000-L64-i32-N1024`,
`p4096-L64-i32-N1024`, `p2048-L128-i32-N1024-d7` (123-node trees, `max_layer_cnt 7`).

Campaign size: **20 timing configs, 0 failures** (77 cell×arm×workload entries in `test/benchmark/evoxc/results/t1a1_fused/campaign.json`;
the merged `test/benchmark/evoxc/results/t1a1_fused/main/campaign.json` indexes the four driver invocations), plus 6 per-cell probes,
6 cold-cache probes and one route-validation run (18 cases).
The harness's own gate over the records (`fused_gate.py` → `check.py --allow-missing --allow-digest-mismatch`, the
same two allowances the baseline campaign needs) reads **20 records / 580 checks / 0 failures, 20/20 configs on an
idle device, 0 contaminated**; its digest warnings are exactly the NaN-payload / float32-reduction splits described
in §6, never a value mismatch.

The route's second campaign (`fused_envelope.py`, §11–§15, table set `test/benchmark/evoxc/results/t1a1_fused/ENVELOPE.md`) covers what this
one does not: the multi-output (`OUT_NODE`) protocol, the memory/population envelope over 12 cells, the measured
basis of `EVOGP_FUSED_MIN_POP`, the block/warp geometry sweep and a public-API re-verification of the six headline
cells.

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
`src/evogp/tests/test_dense_policy.py::test_eval_mode_tables_stay_canonical` used to pin the pre-fused
`EVAL_MODES`/`ENGINES` tuples; it has since been updated to `("dense","specialized","batch","fused","auto")` /
`("specialized","dense","batch","fused")` and the suite is green (1 308 passed / 7 skipped / 0 failed).

Kernel round (§17): the walk now keeps its top three stack slots in registers (`STACK_WINDOW` 3, gated to
`BLOCK_R >= 32`), which takes the DEVICE kernel **−10.9 % on `p4096-L64` / −12.4 % on its loss path and −3.8…−4.5 % on
the 123-node `d7` cell**, leaves the `p256` cells at −1…−8 % and the per-row surface bit-identically unchanged, with
every value digest identical to the pre-change tree. The per-call metric cannot resolve that (its null spread is
±5–12 %, §17.5) and the remaining headroom is the Triton LAUNCH path, not the kernel body (§17.7).
Scope and envelope, re-measured (§11–§15): the route serves `output_len` 1 up to `fused_engine.MAX_OUT_LEN` (8) with a
**bitwise** multi-output value path, and over the 12-cell envelope it is **35.8×–206×** the `torch-batched` arm,
**32.2×–155 847×** `specialized` and **16.3×–495×** the shipped `dense` default, i.e. it loses no measured cell on any
axis. Its device cost is the envelope `fused_engine._check_envelope` charges — the scratch block `pop * n_rb * tree_len *
block * 4` (row axis rounded UP to the block) plus the `(pop, rows, out_len)` value buffer and the loss chart, 2 GiB of
scratch at the largest measured cell, which allocates it and does not OOM, 4× inside the 8192 MiB default — and
`EVOGP_FUSED_MIN_POP` is 16 because the kernel is
**flat in `pop_size`** (0.033–0.036 ms `batch_forward` and 0.040–0.043 ms `SR_fitness` from pop 1 to 128, re-measured for §13 at 5 warmups + 20 reps) while the specialized fallback scales with the tree count (0.466 → 23.225 ms over pop 1 → 128).
That floor is a cold-start guard, not a warm crossover: the kernel already wins warm at pop 1, and 16 keeps a one-shot single `Tree` call off the ~1.6 s Triton JIT (§13).

---

## 1. The route really ran fused (not a silent fallback)

A fused decline is invisible through the public API: `Forest.*` catches `FusedUnsupported` and continues on
`route.fallback_backend`, which with an explicit `iree` backend is `specialized`+`iree` (compile-bound, minutes on a
fresh population, and it would still look plausible). Two independent proofs are recorded per cell:

- **Policy**: `dense_engine.resolve_route(pop, rows=…)` returns `engine="fused"`, `use_fused=True`,
  `backend="iree"`, `fallback_backend="iree"` for all three call shapes (`sr_fitness`, `batch_forward`, `forward`) on
  all six cells — `route` block, `route_all_fused: true` in every probe record.
- **Execution**: the 25-key `fused_engine.fused_cache_stats()` delta around each workload shows
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
| **`fused`** | **0.0762** | **53.8×** | **0.0858** | **48.4×** | **0.0680** | **59.8×** |
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
`N=4096` cell is still 44.9× the torch arm. **Pop axis costs the tree walk:** 0.1016 → 0.1460 ms from pop 256 to
4096 at the same tree length (`L=64`: `p256-L64-i32-N1024` → `p4096-L64-i32-N1024`), which is the documented
"the tree walk repeats once per row block, and per-program work scales with pop" limit.

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

A host-glue round cut the per-call prep out of the warm path (see §7): the whole prepared call — the normalized
population tables, the transposed data operand, the envelope verdict, the geometry and the **bound launch** — is
memoized in `fused_engine.PLAN_MEMO` on `(id, _version)` of the operand tensors plus the resolved knobs, so a warm
call is one key build, one dict lookup and `call.run(destination)`.  Same-session A/B on the six harness cells
(`fused_check --stage timing --wired`): **0.108/0.108/0.122/0.117/0.154/0.181 → 0.076/0.075/0.089/0.082/0.124/0.149 ms**
(−17.7 % to −30.6 %; −19.5 % on `p4096`, whose call is 54 % kernel).  Host-only (fake-kernel) per call:
`batch_forward` 19.5 → 14.3 µs, `SR_fitness` 37.6 → 30.0 µs.  The kernel is now the larger half again and the
host glue is **≈12 %** of a warm `batch_forward` at ``p256`` (14 µs of 76 µs) — a much healthier ratio than the dense route
(host glue ≈ 15 %, but on a 1–7 ms call). Absolute host glue is flat at 0.031–0.062 ms across three orders of
population size, i.e. it is Python-level per-call overhead, not data movement; the remaining headroom in the route
is host-side. `transpose(cold)` 0.019 ms is what the identity memo removes from every warm call.

**What "kernel" means in this table, and where the device time really is.** The `kernel` column times a *call* of
`fused_kernels.launch_value`, i.e. Triton's `JITFunction.run` — argument binding, cache lookup, launch metadata,
`cuLaunchKernel` — not the executed kernel. Torch-profiler device durations on the same populations
(`ProfilerActivity.CUDA`, 40 launches per sample) read **16.5 µs at `p256-L64` / 56.7 at `p4096-L64` / 86.0 on the
123-node `d7` cell** against the 0.063 / 0.111 / 0.137 ms "kernel" column, so a warm `batch_forward` is
**≈22 % device time at `p256`, ≈25 % at `p1000`, ≈51 % at `p4096` and ≈63 % at `d7`**, and the rest of the
"kernel" bucket is the Python launcher plus the facade's host glue. The two statements are consistent (the table
measures the launch, the profiler the kernel) but they bracket the headroom differently: a kernel-level win of
*k* % moves the shipped per-call metric by only `k × device share` (22–63 %), while a launch-path win moves it
one-for-one. §17 is the kernel round and works inside that bound; the launcher is what §7's remaining-gap note now
points at.


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
trees; `test/benchmark/evoxc/results/t1a1_fused/validation-fused.json`):

- `batch_forward` and `forward` vs the **oracle**: `max_abs = 0.000e+00` on **all 18 cases**, including the
  `rich` (transcendental) set. This is a genuinely new result: on `rich`, the *torch* arms deviate from the oracle
  by up to 4.8e-4 (rel 1.8e-3) while fused is exact — the f64-through-libdevice lowering is the correctly-rounded
  path, and it removes the documented `rich`-set caveat for this route (`dense`/`iree` deviates up to 4.4e-1).
- `sr_fitness` vs oracle: `max_rel ≤ 3.4e-7` on every case (worst absolute 8.19e+03 at d7, on losses of order
  1e+12), `nan_mismatch = 0`, `inf_mismatch = 0` everywhere.
- `arith`-set digests therefore differ from the other routes on `sr_fitness` only (float32 block-reduction order);
  a harness gate over a fused column needs `--allow-digest-mismatch`, exactly as the batch route does.
- The **multi-output** (`OUT_NODE`) value path is bitwise against the oracle and its loss `≤ 1.2e-07` relative on
  `output_len` 2, 3 and 8 — the protocol, its adversarial cases and the `--stage multi` counts are §11.

---

## 7. Honest residual gaps (with numbers)

1. **The suite is green** (the two stale assertions the campaign opened with have since been fixed, in
   `tests/test_dense_policy.py::test_eval_mode_tables_stay_canonical` and in `fused/tests/test_fused_engine.py`).
   `pytest src/evogp/tests -q` → **1 308 passed, 7 skipped, 0 failed in ~121 s** on the device reference box (53 of
   them in `fused/tests/test_fused_engine.py`, which compiles and launches the real kernel).
   `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -c "import evogp.fused"` works with
   `torch.cuda.is_available() == False` and `fused_available() == False` — the import needs no GPU, because
   `import evogp.fused` imports torch at module scope while `import triton` happens only inside `kernels.kernels()`
   (and there is no policy left to degrade: `resolve_route` went with the deleted torch stack, so a declined call
   raises `FusedUnsupported` for the caller to handle).
2. **The pytest net is `fused/tests/test_fused_engine.py`** (53 cases: value/loss parity against `reference_eval` on the
   three function sets, single-node and constant populations, a deep chain, protected div and NaN/Inf placement,
   multi-output value + loss, the route policy above and below `EVOGP_FUSED_MIN_POP`, `fused_kernel_launches` as the
   served-call proof, the decline matrix including `output_len > MAX_OUT_LEN`, non-contiguous inputs and the
   `subtree_size` clamp). It is not complete: there is no case for `out=` (shape / aliasing / an identity `out`), no
   MAE (`use_MSE=False`) loss case, no multi-output `per_row=True` case with weighted outputs, and no rows axis that
   lands exactly ON a `BLOCK_R = 256` boundary (the widest tested cell is 160 rows).
   `fused_check.py`'s stages plus this campaign remain the measurement gate.
3. **Loss ratios in this campaign land 4–17 % below the previously reported ones** while the value surface matches:
   value 40.8/43.6/44.9/46.4/72.9/156.9× here against 38.9/45.8/42.8/45.5/71.0/153.5× reported (all within ±5 %);
   loss 32.1/32.6/38.6/34.2/59.7/130.9× here against 33.5/39.1/40.7/38.4/69.0/146.6×. The absolute loss medians here
   are 0.1299–0.2005 ms. Two candidate causes, both benign and neither excluded: a 0.1 ms-scale measurement with 5
   reps is sensitive to GPU clock state, and this campaign's own `torch-batched` reference also ran 2–3 % slower than
   the baseline of record (4.0547 vs 4.0044 ms `bf` at `p256-L32`). Do not quote the loss path tighter than
   "32–131× in this campaign, up to ~147× measured previously".
4. **The host-glue share no longer decomposes.** A warm call is a memoized prepared call (`fused_engine.PLAN_MEMO`), a pre-bound launch (`EVOGP_FUSED_DISPATCH=bound` serving Triton 3.7.1's `CompiledKernel.run`, with `jit` restoring `JITFunction.run`), a single-slot plan fast path for a repeated population and the memoized route decision (`dense_engine.route_cache_stats()`), so the whole call reads cheaper than one independently timed launch and only the totals are quotable: the shipped default is 0.0449–0.1152 ms `batch_forward` on the six cells (`test/benchmark/evoxc/results/t1a1_auto_r3/`).
   The DEVICE kernel inside it is **16.5–86 µs** (§3's profiler column, §17.1), i.e. 22 % of a `p256` call, 51 % of `p4096` and 63 % of `d7` before the pre-bound launch, and ≈31 % at `p256` after it (§18); the remaining headroom on the small cells is the per-call Python of the facade, not the kernel body.
5. **The multi-output loss path materializes the predictions.** `output_len >= 2` writes the `(pop, rows, out_len)`
   prediction buffer to device memory (zeroed, then read-modify-written at every tagged node — the accumulator is
   that buffer, see §11) and reduces it in the same kernel, where the single-output path folds in registers and moves
   only `1/rows` of the traffic; the multi-output call's only device output is still the `(pop, n_rb)` partial chart,
   and the single-output path keeps its original numbers. `MAX_OUT_LEN` 8 is a register/tile budget per out-node, not
   a measured cliff: nothing above 8 outputs has been measured, and a wider call declines cleanly.
6. **The default policy's population floor is 16** (`pop_size < EVOGP_FUSED_MIN_POP` with no explicit backend
   resolves to `specialized`+`numpy`), so a single `Tree.forward` / `Tree.SR_fitness` never reaches the kernel and
   `EVOGP_FUSED_MIN_POP=0` is refused; the measured basis of the number is §13. **`EVOGP_EVOXC_BACKEND=numpy` still
   reports `engine="fused"` while every call declines** — the engine name is not a served-call proof. Both campaigns
   therefore assert on `fused_kernel_launches`; the harness's own route reporting cannot yet distinguish the two
   (`fused_sibling_patches.md` §5).
7. **One process pays the JIT again** (`fused_kernel_compiles` counts a first launch per specialization even when
   Triton serves it from disk in 2.1 ms), and `clear_fused_cache()` resets the facade counters but not
   `fused_kernels._KERNELS["specializations"]`, so a post-reset repeat can under-report compiles.
8. **This box needs `TRITON_CACHE_DIR`.** `$HOME` is read-only; without the variable the first launch dies with
   `OSError: [Errno 30] Read-only file system: '/root/.triton'`. Nothing in `src/` sets or validates it.
9. **The unmeasured class is tiny rows at tiny trees**: `rows <= 64` with `tree_len` 8–32 and pops 1–8, where the
   kernel's ~0.09 ms launch floor is the whole call and a single per-tree numpy run could undercut it. Every measured
   cell is at `tree_len >= 64` (rows 1–16 384, pops 1–8 192), so nothing here prices that corner.
10. **The per-call metric has a ±5–12 % noise floor** and cannot resolve kernel work: the null experiment
   (A against A, same tree, three order-alternated passes, the campaign's 2 warmup + 10 sync-bracketed reps) spreads
   −10.3 … +1.0 % on `value`, −12.3 … +0.0 % on `loss` and −5.7 … +1.4 % on `per_row`, and at reps 100 a systematic
   ±2–4 % first-arm advantage remains (§17.5). Any future round that quotes a < 5 % per-call delta as a result is
   quoting the noise, so kernel work is gated on the torch-profiler device durations and the per-call table is a
   cross-check only.
11. **Route mechanics that are not gaps but bound what can be measured**: prep is device-side only (no host work to
    attribute), there is NO per-stage timer surface (the split is reconstructed by `fused_check._stage_split`), the
    value path has no zero-copy device export (a `(pop, rows, 1)` tensor stays allocated, 1 MB – 8 GB, and is never
    read back), `--stage goldens` still skips `src/evogp/tests/goldens/06_multi_output.npz` (§11 is its coverage),
    and every number above was taken under the 12 GB sandbox `MemoryMax`, which is why the envelope's 2 GiB cell is
    the largest shape measured.

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
$PY test/benchmark/evoxc/results/t1a1_fused/fused_campaign.py --out test/benchmark/evoxc/results/t1a1_fused/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms torch-batched,evoxc-fused-iree --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY test/benchmark/evoxc/results/t1a1_fused/fused_campaign.py --out test/benchmark/evoxc/results/t1a1_fused/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024 \
    --arms evoxc-dense-iree --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY test/benchmark/evoxc/results/t1a1_fused/fused_campaign.py --out test/benchmark/evoxc/results/t1a1_fused/main \
    --cells p2048-L128-i32-N1024-d7 --arms evoxc-dense-iree-raised \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 --seed 0 --guard-wait 60 --timeout 1800

$PY test/benchmark/evoxc/results/t1a1_fused/fused_campaign.py --out test/benchmark/evoxc/results/t1a1_fused/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p1000-L64-i32-N1024 \
    --arms evoxc-batch-iree --workloads sr_fitness,batch_forward,forward \
    --warmup 2 --reps 5 --seed 0 --guard-wait 60 --timeout 1800

# 2. per-cell probe: route proof, served-call counters, numerics (torch-batched + CPU oracle),
#    kernel/host/transpose split, cold first calls, and the GP generation loop
export EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree
for c in p256-L32-i8-N1024 p256-L64-i32-N1024 p256-L64-i32-N4096 p1000-L64-i32-N1024 p4096-L64-i32-N1024 p2048-L128-i32-N1024-d7; do
  $PY test/benchmark/evoxc/results/t1a1_fused/fused_probe.py --cell $c --gp --oracle-trees 32 \
      --out test/benchmark/evoxc/results/t1a1_fused/probes/$c.json
done

# 3. cold Triton cache: a fresh TRITON_CACHE_DIR per cell, one call per surface
i=0; for c in p256-L32-i8-N1024 p256-L64-i32-N1024 p256-L64-i32-N4096 p1000-L64-i32-N1024 p4096-L64-i32-N1024 p2048-L128-i32-N1024-d7; do
  i=$((i+1)); rm -rf $TMPDIR/tf-cold-$i; TRITON_CACHE_DIR=$TMPDIR/tf-cold-$i \
  $PY test/benchmark/evoxc/results/t1a1_fused/fused_probe.py --cell $c --cold-cache \
      --out test/benchmark/evoxc/results/t1a1_fused/cold/$c.json
done

# 4. numeric validation against the plain-torch oracle, three function sets
$PY test/benchmark/evoxc/validate.py \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --func-sets arith,rich,ternary --max-oracle 32 \
    --out test/benchmark/evoxc/results/t1a1_fused/validation-fused.json

# 5. index (one campaign.json entry per cell × arm × workload) + the markdown summary
$PY test/benchmark/evoxc/results/t1a1_fused/campaign_index.py

# 6. the route's second campaign, as far as the library battery still carries it: the
#    OUT_NODE protocol and the decline matrix (one process per config; each stage writes
#    its own JSON record, `--json PATH` overrides the file)
$PY -m evogp.fused.check --stage multi      # the OUT_NODE protocol vs the oracle
$PY -m evogp.fused.check --stage declines   # the decline/contract matrix
#    (The `fused_envelope.py` harness behind §12–§15 — `--stage headline|envelope|minpop|
#    sweep|degenerate` and the read-only `records` audit — was deleted with the pre-2.0.0a2
#    tree and NOT ported; its own records stay beside this document.)

# 7. end-to-end and the hermetic suite (run the suite AFTER all timing — parts of it touch the GPU)
EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree $PY -m evogp.sr_test     # best fitness -0.2259 in ~10.4 s
$PY -m evogp.sr_test                                                    # default auto: -0.2259 in ~16.1 s
PYTHONPATH=src:/root/evoxc $PY -m pytest src/evogp/tests -q             # 1 308 passed / 7 skipped / 0 failed
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONPATH=src:/root/evoxc $PY -c "import evogp.fused"
```

`.log` worker files are not committed; only the JSON records, the probes and this report are.

---

## 9. Where these records live

The campaign is benchmark evidence for the `evoxc` harness, so it lives at
**`test/benchmark/evoxc/results/t1a1_fused/`** (the sibling of `results/t1a1_baseline/`), with `main/` as the
gateable directory and `probes/`, `cold/`, `validation-fused.json` as sibling columns.
The harness registry carries both arms (`evoxc-fused-iree`, `evoxc-fused` in `common.EVOXC_ARMS`), so a re-run is a
plain `campaign.py` invocation and the gate is `check.py --results test/benchmark/evoxc/results/t1a1_fused --allow-snapshot --allow-missing --allow-digest-mismatch`.
This document stays in the library as the route's design/report; §10 lists the records beside it.

## 10. Files

| path | content |
|---|---|
| `test/benchmark/evoxc/results/t1a1_fused/campaign.json` | index: one entry per cell × arm × workload (77 entries, 0 failed, 0 missing) |
| `test/benchmark/evoxc/results/t1a1_fused/campaign.md` | the same index rendered as tables |
| `test/benchmark/evoxc/results/t1a1_fused/main/<cell>/<arm>.json` | harness-schema timing records (workloads, medians, samples, cold calls, digests, routes, `gp_step`) |
| `test/benchmark/evoxc/results/t1a1_fused/main/campaign.A.json`, `.B1.json`, `.B2.json`, `.C.json` | the four driver invocations' own `campaign.json` files |
| `test/benchmark/evoxc/results/t1a1_fused/probes/<cell>.json` | route proof, served-call counter deltas, numerics vs arm + oracle, stage split, cold calls, GP loop |
| `test/benchmark/evoxc/results/t1a1_fused/cold/<cell>.json` | cold-Triton-cache first calls (one fresh cache dir per cell) |
| `test/benchmark/evoxc/results/t1a1_fused/validation-fused.json` | `validate.py` output, 18 cases × 15 checks |
| `test/benchmark/evoxc/results/t1a1_fused/fused_campaign.py`, `fused_gate.py`, `fused_probe.py`, `campaign_index.py` | the tooling that produced the above |
| `test/benchmark/evoxc/results/t1a1_fused/main/headline_<UTC>.json` | the six-cell public-API headline re-verification of §15 (value/loss/per-row medians, the `fused_direct` kernel column and the policy's route per cell) |
| `test/benchmark/evoxc/results/t1a1_fused/envelope.json` | the 12 envelope cells of §12 (arms `fused` / `torch-batched` / `specialized` / `dense`, scratch bytes, the device peak over the fused window, parity) |
| `test/benchmark/evoxc/results/t1a1_fused/minpop.json` | the pop 1–128 column of §13 (`fused_direct` + the public API + `specialized` + `torch-batched` + the policy's engine) |
| `test/benchmark/evoxc/results/t1a1_fused/sweep/sweep.json` | the 12-config block/warp geometry sweep of §14 (four cells per config) |
| `test/benchmark/evoxc/results/t1a1_fused/degenerate.json` | the degenerate shapes (rows 1 / rows 1000 / `input_len` 1 / all-constant population / single-node population) |
| `test/benchmark/evoxc/results/t1a1_fused/ENVELOPE.md` | the table set of that campaign: protocol, headline, envelope, min-pop, sweep, degenerate shapes, reader notes |
| `fused_envelope.py` | the driver behind §12–§15 (`--stage headline\|envelope\|minpop\|sweep\|degenerate`, plus the `*-one` single-process legs the drivers spawn) and the read-only `--stage records` audit of the committed files against the engine's envelope rule |
| `fused_sibling_patches.md` | the exact cross-node edits (test pin, harness arm registry, route reporting, worker counters, doc drift) |

---

## 11. The multi-output (`OUT_NODE`) protocol — served for `output_len` 1..8

`EVOGP_EVAL_MODE=fused` serves every `output_len` from 1 up to `fused_engine.MAX_OUT_LEN` (8).
`out_len == 1` runs `_fused_kernel` (whose six headline cells of §2 keep their numbers), and `out_len >= 2`
runs `_fused_multi_kernel` in `fused_kernels.py` — the same reverse-prefix walk plus the multi-output rules of the
legacy `treeGPEvalKernel<true>`:

- the base kind is `node_type & 0x7F` and `node_type & 0x80` is the `OUT_NODE` tag;
- a TAGGED FUNCTION NODE resolves its operator from the packed value bits (`function = bits & 0xFFFF`,
  `outIndex = (bits >> 16) & 0xFFFF`, an index `>= out_len` dropped — the oracle's unsigned compare) and adds its OWN
  value into `out[outIndex]`;
- every node passes its RIGHT-MOST operand up to its parent (`pass_up`: the second operand of a binary node, the third
  of a ternary one, the operand of a unary one, its own value for a leaf), so the root's own value is ignored;
- tagged LEAVES accumulate nothing and outputs no node ever tags stay exactly `0.0`.

Semantics of record for the oracle side: `src/evogp/tests/reference_eval.py::forward_tree(multi_output=True)`.

**Implementation note that cost a debugging round**: the accumulator is the prediction buffer itself
(`(pop, rows, out_len)`, zeroed in-kernel and then read-modify-written at every tagged node), never a
`[out_len, BLOCK_R]` register tile — a tile carried across the loop phis makes triton unify two layouts for the value
registers and the LLVM translation dies on `builtin.unrealized_conversion_cast`. `pass_up` must likewise be a FRESH
zero tile, never an alias of `val`. Consequence for the loss path: `output_len >= 2` materializes the
`(pop, rows, out_len)` predictions in device memory and reduces them in the same kernel (its only device output is
still the `(pop, n_rb)` partial chart), where the single-output path keeps its `1/rows` traffic and its original
numbers.

### Parity (`fused_check.py --stage multi`, one process)

**65/65 comparisons**: 13 case×`out_len` populations (`out_len` 2, 3 and one at 8), each on the four surfaces.

| surface | tolerance | recorded |
|---|---|---|
| `batch_forward` (`out_len >= 2`) | bitwise vs the oracle | `max_abs = max_rel = 0.0` |
| `forward` (per-tree, one row per tree) | bitwise vs the oracle | `max_abs = max_rel = 0.0` |
| `SR_fitness` (MSE) | `<= 1e-5` relative | `max_rel <= 1.2e-07` |
| `SR_fitness` (MAE) | `<= 1e-5` relative | `max_rel <= 1.2e-07` |

Three STRUCTURAL expectations are pinned next to the numeric ones: outputs no node tags (and an `outIndex` out of
range) read exactly `0.0`; an untagged population's loss is exactly `sum(labels^2)/rows` for MSE; and two rows whose
packed operator differs (`MUL` beside `ADD` on identical operands) must produce different numbers.

Adversarial coverage of that set: two outputs deep + shallow; the tag on the ROOT (a tagged ternary `IF`); only one
output tagged; two nodes tagging the SAME output (accumulation); tagged `VAR` and `CONST` leaves; an out-of-range
`outIndex` (dropped); one tagged node per arity (unary / binary / ternary); no tagged node at all; a packed `MUL`
beside a packed `ADD` on identical operands (function extraction); one population read at `out_len` 2 and again at 3
(the tag on index 2 changes meaning); `out_len = 8` (the widest the route serves); and a garbage-padded multi-output
row (NaN/nonsense cells behind the tree).

### Declines and the gate counts

- `output_len > MAX_OUT_LEN` declines with a reason naming `MAX_OUT_LEN` (clean: no kernel compiled, no scratch
  allocated), and `output_len = 0` raises `ValueError` (a shape contract, not a route decline).
- The battery's matrices read: decline/contract **DECLINES 57/57** (two-sided on `output_len`: 2 served, above
  `MAX_OUT_LEN` declined), wiring **WIRED 32/32** (the public-API leg drives a multi-output population and asserts
  one `fused_calls`, no `fused_declines` and the specialized reference's numbers), parity **PARITY 56/56** (the
  single-output set of §6), multi-output **MULTI 65/65** and goldens **GOLDENS 10/10**.
- `--stage goldens` still SKIPS `src/evogp/tests/goldens/06_multi_output.npz` — it is the single-output golden gate
  and serving the fixture would move its recorded count. `--stage multi` is the multi-output coverage; the fixture
  is not read by any fused stage.

---

## 12. The envelope: memory, rows and population (12 cells)

Records `test/benchmark/evoxc/results/t1a1_fused/envelope.json`, table set `test/benchmark/evoxc/results/t1a1_fused/ENVELOPE.md`. Shipped geometry
(`BLOCK_R` 256 / 1 warp), warm medians in ms, one process per config, `TREE_LEN = 64` unless the cell names `d7`
(`max_layer_cnt` 7, 123-node trees) or `d8`.

| cell | fused value | fused loss | arm | specialized | dense | arm/fused | spec/fused | scratch B | n_rb | peak alloc / reserved MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `p16-L64-i32-N1024` | 0.088 | 0.099 | 3.162 | 2.845 | 1.474 | 35.8 | 32.2 | 4 194 304 | 4 | 4.4 / 22 |
| `p64-L64-i32-N1024` | 0.107 | 0.138 | 4.046 | 12.088 | 1.753 | 37.7 | 112.6 | 16 777 216 | 4 | 16.8 / 18 |
| `p256-L64-i32-N1024` | 0.102 | 0.123 | 4.809 | 45.587 | 2.889 | 47.3 | 448.3 | 67 108 864 | 4 | 66.4 / 68 |
| `p1000-L64-i32-N1024` | 0.136 | 0.122 | 5.367 | 179.966 | 7.730 | 39.5 | 1323 | 262 144 000 | 4 | 258.7 / 274 |
| `p256-L64-i32-N4096` | 0.126 | 0.121 | 5.203 | 49.752 | 7.855 | 41.4 | 396.2 | 268 435 456 | 16 | 265.2 / 278 |
| `p256-L64-i32-N16384` | 0.164 | 0.171 | 10.257 | 110.135 | 80.996 | 62.7 | 673.0 | 1 073 741 824 | 64 | 1060.2 / 1062 |
| `p1000-L64-i32-N4096` | 0.188 | 0.193 | 10.221 | 197.589 | 49.170 | 54.5 | 1052.7 | 1 048 576 000 | 16 | 1033.6 / 1036 |
| `p2048-L64-i32-N1024` | 0.121 | 0.138 | 7.122 | 369.162 | 14.561 | 59.0 | 3058 | 536 870 912 | 4 | 529.5 / 538 |
| `p4096-L64-i32-N1024` | 0.156 | 0.157 | 10.531 | 720.569 | 46.933 | 67.6 | 4627 | 1 073 741 824 | 4 | 1058.8 / 1072 |
| `p8192-L64-i32-N1024` | 0.218 | 0.201 | 17.673 | 33904.4 | 95.973 | 81.2 | 155847 | 2 147 483 648 | 4 | 2117.3 / 2142 |
| `p2048-L128-i32-N1024-d7` | 0.191 | 0.189 | 27.520 | 888.204 | DE | 143.9 | 4645 | 1 073 741 824 | 4 | 1042.5 / 1056 |
| `p2048-L256-i32-N1024-d8` | 0.238 | 0.233 | 49.013 | 1398.001 | DE | 206.0 | 5876 | 2 147 483 648 | 4 | 2068.5 / 2096 |

`arm` = `torch-batched`; the ratio convention stays `reference_ms / fused_ms`, so higher means fused wins.
`DE` = `dense_expected_decline`: the tree is above `EVOGP_DENSE_MAX_TREE_LEN` (64), where an explicit `dense` request
falls back to per-structure `specialized` and timing it would only report that fallback.

- Value parity against the arm is **bitwise on every cell**; loss agreement is **1.6e-07 … 4.1e-07** relative
  (in-graph reduction order).
- The recorded `scratch B` column is the scratch term `pop * n_rb * tree_len * block * 4`; it equals
  `pop_size * tree_len * rows * 4` on every cell here only because `rows` is a multiple of the 256-lane block.
  The shipped envelope (`fused_engine._check_envelope`, §16.4) charges that term with the row axis ROUNDED UP to the
  block PLUS the `(pop, rows, out_len)` value buffer (8–32 MiB on these cells at `out_len` 1) and, on the loss path,
  the `(pop, n_rb)` chart; the default `EVOGP_FUSED_MAX_SCRATCH_MB` (8192) is 4× clear of the largest measured cell
  under either rule, and `fused_envelope.py --stage records` re-checks every stored `envelope_ok` read-only (79
  stored records, 0 disagreements).
- The node-count cell really allocates its 2 GiB stack scratch — device peak over the fused window Δ 2064 MiB
  allocated / 2048 MiB reserved — and does **not** OOM.
- No cell loses: the narrowest margin over `specialized` is 32.2× (`p16`), over `dense` 16.3× (`p64`, whose cost is
  shape-bound rather than population-bound).

---

## 13. `EVOGP_FUSED_MIN_POP` is 16

Re-measured **2026-10-31** on the current tree — pre-bound dispatch and plan fast path shipped — with
`test/benchmark/evoxc/minpop_probe.py` (records and `SUMMARY.md` in `test/benchmark/evoxc/results/minpop_r3/`):
5 untimed warmups + 20 sync-bracketed reps, warm medians, one process per (pop, arm), idle-device guard per config
(40/40 configs on an idle device), `L=64` / `i32` / `N=1024`.
`kernel` is `fused_engine.evaluate_fused` / `sr_fitness_fused` (policy-free, so it is measurable below the floor);
the `fused` column is `Forest.batch_forward` / `SR_fitness` with `EVOGP_EVAL_MODE=fused`, i.e. what the policy serves.

The kernel is FLAT in `pop_size` — **0.033–0.036 ms** on the value surface and **0.040–0.043 ms** on the loss surface
from pop 1 to 128 — because its cost is the row-block sweep, not the trees, while `specialized` scales linearly with
the tree count (0.466 → 23.225 ms over the same pops).
| pop | kernel value | kernel loss | `fused` (public) value | `fused` (public) loss | specialized value | torch-per-tree value | torch-batched value | spec/kernel | torch-1/kernel | policy engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.035 | 0.040 | 0.517 | 0.542 | 0.466 | 0.233 | 1.359 | 13.2 | 6.6 | specialized |
| 2 | 0.035 | 0.043 | 0.666 | 0.680 | 0.628 | 0.254 | 1.587 | 17.9 | 7.3 | specialized |
| 4 | 0.035 | 0.040 | 0.918 | 0.952 | 0.928 | 0.433 | 1.897 | 26.4 | 12.3 | specialized |
| 8 | 0.033 | 0.043 | 1.599 | 1.571 | 1.600 | 0.788 | 2.503 | 47.8 | 23.5 | specialized |
| 16 | 0.034 | 0.041 | 0.040 | 0.049 | 2.814 | 1.410 | 3.195 | 82.1 | 41.1 | **fused** |
| 32 | 0.034 | 0.041 | 0.039 | 0.049 | 5.547 | 2.736 | 3.782 | 161.6 | 79.7 | fused |
| 64 | 0.035 | 0.042 | 0.040 | 0.048 | 13.413 | 6.088 | 4.126 | 388.3 | 176.2 | fused |
| 128 | 0.036 | 0.041 | 0.038 | 0.048 | 23.225 | 11.243 | 4.103 | 650.2 | 314.8 | fused |
Below the floor the `fused` column IS that fallback — same route spelling (`specialized`+`numpy`), 0 kernel launches,
`served: false` — so the two labels read the same number within box noise at pops 1–8 (0.517 / 0.466 at pop 1,
1.599 / 1.600 at pop 8).
The warm crossover is therefore **pop 1, not 16**: at pop 1 the kernel already beats `specialized` by 13.2× (13.3× on
loss), the per-tree torch walk by 6.6× (6.2×) and the batched torch arm by 38.6× (35.0×); at pop 16 those margins are
82.1× / 41.1× / 93.2× (71.2× / 45.6× / 77.5×).
The floor is NOT a throughput crossover — it is a COLD-START guard on the one call shape that is one-shot by
construction: the kernel's first call pays the Triton JIT (1619 ms measured at pop 1; the empty-cache figures are
~1.6 s for the value kernel and ~0.94–0.99 s for the loss kernel) against a 0.431 ms/call warm saving over
`specialized` at that shape, so a one-shot single-tree call needs ≈3 760 calls to repay it, and the whole warm win
below 16 is real but uncollected on purpose (`fused_engine.resolve_min_pop`, `EVOGP_FUSED_MIN_POP`;
`dense_engine.resolve_fused_min_pop` delegates to it through a local import).
16 stands as a CONSERVATIVE floor: lowering it to the measured crossover (pop 1) would put every single
`Tree.forward` / `Tree.SR_fitness` on the JIT — ~3 400× a warm specialized call on its first call — while no GP
population is generated below 16, so nothing on the throughput axis is left on the table.
The second reason a slide to 1–8 is not taken is unchanged and still unmeasured: the whole column is `N=1024`, and a
1–8-tree population on a tiny row plane is the class where a per-tree numpy run could undercut a ~0.035 ms launch
floor (§7's open corner).
`EVOGP_FUSED_MIN_POP=0` stays refused (a pop-size threshold must be `>= 1`), an explicitly named backend still
serves below the threshold, and `auto` selects fused from the threshold up (the default on a device box,
`dense_engine.auto_fused_decline`).

---

## 14. Block/warp sweep — the default sits on a flat minimax top at 256 / 1

The table below is the earlier window's 12-config × 4-cell sweep (`test/benchmark/evoxc/results/t1a1_fused/sweep/sweep.json`), value medians in ms, one process per (config, cell).
`worst` there = the config's slowest cell relative to that cell's best config.

| cfg | `p256-L64-N1024` | `p4096-L64-N1024` | `p256-L64-N4096` | `p2048-L128-d7` | worst |
|---|---|---|---|---|---|
| B128-W1 | 0.088 | 0.161 | 0.109 | 0.204 | 1.27 |
| B128-W2 | 0.098 | 0.240 | 0.125 | 0.309 | 1.94 |
| B128-W4 | 0.101 | 0.370 | 0.168 | 0.469 | 2.93 |
| **B256-W1 (shipped)** | **0.096** | **0.137** | **0.113** | **0.160** | **1.20** |
| B256-W2 | 0.090 | 0.149 | 0.108 | 0.190 | 1.19 |
| B256-W4 | 0.092 | 0.231 | 0.121 | 0.284 | 1.78 |
| B512-W1 | 0.102 | 0.154 | 0.110 | 0.161 | 1.26 |
| B512-W2 | 0.090 | 0.139 | 0.107 | 0.160 | 1.12 |
| B512-W4 | 0.099 | 0.159 | 0.112 | 0.181 | 1.24 |
| B1024-W1 | 0.094 | 0.170 | 0.101 | 0.174 | 1.24 |
| B1024-W2 | 0.096 | 0.157 | 0.114 | 0.165 | 1.19 |
| B1024-W4 | 0.081 | 0.140 | 0.108 | 0.164 | 1.07 |

The verdict is `fused_check --stage tuning` (12 configs × 6 cells through the public API, one process per config+cell, `worst value ratio` = the config's worst cell's fused/torch-batched ratio): across the two measured windows the minimax alternates between the shipped `EVOGP_FUSED_BLOCK_ROWS=256` / `EVOGP_FUSED_NUM_WARPS=1` (0.009) and `B512-W2` (0.009), so the shipped default is inside **1.00–1.05×** of the best cell (worst value ratio 0.009–0.010 on `p256-L64-i32-N4096`, worst loss 0.011; every other configuration sits at 0.010–0.037, with `B128-W1` 0.010 and `B512-W1` / `B512-W4` 0.010–0.011 inside the same noise band).
The latest re-run of the stage reads minimax `B512-W2` at 0.009 against the shipped `256/1` at 0.010, i.e. 1.05× — grid noise on a flat surface, not a policy change, exactly as the previous window's opposite reading (`256/1` on top at 1.00×) was.
Neither sweep moves the default: both tuning windows sit on the same flat top, and in the earlier table the lower-minimax `B1024-W4` (1.07) wins solely on `p256-L64-i32-N1024`, whose own within-record spread (52 %) exceeds the 18 % gap.
The clearly-bad region is narrow blocks with many warps (128 lanes × 4 warps: 1.67–3.04×), consistent with the walk being a scalar loop-carried dependency chain.

---

## 15. Headline re-verification — no regression

Six cells of record through the PUBLIC API at the shipped geometry (`test/benchmark/evoxc/results/t1a1_fused/main/headline_<UTC>.json`), ms,
warm medians, one process per config:

| cell | value | loss | per_row | arm | ratio | kernel-only | recorded kernel-only |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.101 | 0.116 | 0.082 | 3.937 | 38.9 | 0.080 | 0.083 |
| `p256-L64-i32-N1024` | 0.106 | 0.115 | 0.090 | 4.441 | 42.0 | 0.082 | 0.082 |
| `p256-L64-i32-N4096` | 0.121 | 0.126 | 0.092 | 5.247 | 43.4 | 0.105 | 0.097 |
| `p1000-L64-i32-N1024` | 0.117 | 0.122 | 0.086 | 5.375 | 45.9 | 0.101 | 0.096 |
| `p4096-L64-i32-N1024` | 0.143 | 0.154 | 0.102 | 10.628 | 74.2 | 0.127 | 0.134 |
| `p2048-L128-i32-N1024-d7` | 0.176 | 0.191 | 0.125 | 27.831 | 157.8 | 0.155 | 0.161 |

`ratio` = `arm / value`; `kernel-only` = the facade without the route policy (`fused_direct`, i.e. `evaluate_fused`
called straight) and `recorded kernel-only` = the per-cell kernel medians of §3.

- **No cell regressed by more than 10 %** (worst +8.4 % on the kernel-only column, inside the protocol's spread).
- The two columns are not the same measurement: the PUBLIC-API column sits **+7 … +29 %** above the kernel-only
  numbers because it pays the facade, the route decision and the `(pop, rows, 1)` staging (20–24 µs). The kernel
  itself is unchanged.

---

## 16. Correctness hardening — an independent review's findings, fixed with a case each

An independent review of the shipped route found eleven defects in three classes: an **out-of-bounds device
read** (B1), **silent wrong numbers** (B2), a **fallback that contradicted its own docstring** (B3), and six
robustness/hygiene items (S1, S2, S3, S5, S7, S9) plus a gate that was weaker than its claims (S6).
All are fixed at HEAD; every one carries a regression case in `fused_check.py --stage declines`
(**24 → 54** contract checks; `--stage parity` **56 → 60** comparisons), and the six headline cells were
re-timed against an extracted pre-fix tree: **+0.4 … +3.6 % per cell** (§16.6), i.e. inside the §15 protocol
spread and inside the ≤10 % bar.

### 16.1 Out-of-bounds device read (B1) and the previous call's bytes (S8)

`subtree_size[0]` was never clamped: it drove `range(0, n)`, the `j = n-1` first index, every
`node_value_ptr + row_base + j` / `node_type_ptr + row_base + j` load and the child lookahead, so a row whose
`subtree_size[0]` exceeded `max_tree_len` read the NEXT row and — on the last row — past the allocation.

- **Fix (B1):** `n = tl.minimum(tl.maximum(tl.load(subtree_size_ptr + row_base).to(tl.int32), 0), tree_len)`
  in BOTH kernels; the loop bound, the first index and the child lookahead all derive from the clamped value,
  so a malformed row walks its own row only, for at most `tree_len` iterations.
- **Proof:** `p256-L64-i32-N1024` with `subtree_size[0] = 32 767` costs **13.7010 ms** before the clamp and
  **0.0526 ms** after it, against **0.0561 ms** for the honest row — the old kernel really did run 32 767
  stray iterations. The battery (`_check_row_length_clamp`) asserts the oversized row now produces EXACTLY
  the honest row's value (`max|delta| = 0.000e+00`) for `subtree_size[0]` = 30 000 / 32 767 / −5, including a
  case whose FOLLOWING row holds a different tree (a spill would show up as a delta).
- **S8 (deterministic underflow):** a binary/ternary pop read `stack[sp-1]`/`stack[sp-2]` through a clamped
  slot index, so a malformed tree read whatever the shared scratch block held from an earlier call (measured:
  the same call returned `6.0e6` or `-4.5e9` depending on what dirtied the block). Both pops are now masked
  (`mask = rmask & (sp >= 1)` / `(sp >= 2)`, `other=0.0`), so the operand is a deterministic `0.0`
  (`_check_underflow_determinism` calls the malformed row twice with different junk fills and asserts
  `torch.equal`).
- **Residual:** a malformed tree still yields a defined-but-arbitrary number instead of the specialized
  engine's `ValueError`, because the route declines nothing on STRUCTURE (that contract stays with
  `evoxc_engine.decode_population`). What changed is that the value is bounded, deterministic, and cannot read
  out of bounds. The scratch block remains shared per `(device, n_prog, stack_cap, BLOCK_R)` shape class, so
  concurrent calls of one shape class need one stream — stated in `fused_kernels`' docstring.

### 16.2 Silent wrong numbers on non-contiguous inputs (B2)

The kernel does raw pointer arithmetic (`x_ptr + t*input_len + vi`, `node_value_ptr + row_base + j`,
`labels_ptr + r*out_len + k`) while `_as_device_tensor` only cast device/dtype, and `utils.check_tensor` passes
an already-cuda tensor through untouched — so any non-contiguous `x`, label block or population table was read
with the wrong strides and returned plausible garbage:

| case (`EVOGP_EVAL_MODE=fused`, `p256-L64-i32-N1024`) | pre-fix max&#124;delta&#124; vs the contiguous copy | now |
|---|---|---|
| `Forest.forward(x_padded[:, :32])` (per_row) | **1.150381e+04** | 0.0 (bitwise) |
| `Forest.SR_fitness(x_nc, labels_padded[:, :1])` | **1.024000e+03** | 0.0 (bitwise) |
| `Forest.batch_forward`, stride-2 population tables | **1.182486e+08** | 0.0 (bitwise) |
| `Forest.SR_fitness`, stride-2 population tables | **1.394022e+13** | 0.0 (bitwise) |
| multi-output (`out_len=2`) value + loss, strided labels/tables | (same class) | 0.0 (bitwise) |

**Fix:** `_as_device_tensor` makes the tensor contiguous after the device/dtype casts (`if not
value.is_contiguous(): value = value.contiguous()`), and it is the one place all five operands pass (the three
population tables in `_prepare`, `x` on both value paths, `inputs`, `labels`).
**Cost:** a cached flag test, no copy — measured 0.084 µs per tensor, 0.35–1.00 µs per call across the five
operands (`is_contiguous()` fired zero times in every measured cell). The battery (`_check_noncontiguous`) and
the new `out_device` checks pin it bitwise.

### 16.3 The decline now lands where the docstring says (B3)

`resolve_route`'s fused branch used to build `Route(False, name, name, engine="fused")` for an explicitly named
backend, so `EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree` fell back to per-structure `iree` compiles — the
compile-bound, artifact-unbounded path `DENSE_EVAL.md` warns about — while both `resolve_route`'s and
`dense_engine`'s docstrings promised specialized+`numpy`.

**Resolution:** the fallback is ALWAYS `"numpy"` on every arm (the reasoning used for the batch route: a
per-tree iree compile cannot amortize), and the explicit-iree arm keeps `backend="iree"` so the served path is
unchanged. Both fields stay inside `evoxc_engine.SUPPORTED_BACKENDS`.
**Proof (`_check_decline_fallback`):** with `EVOGP_EVAL_MODE=fused` + `EVOGP_EVOXC_BACKEND=iree` +
`EVOGP_FUSED_MAX_SCRATCH_MB=1` on a pop-32/L64/N1024 forest the call really declines (`fused_declines` +1) and
is still served; a spy on `evoxc_engine.evaluate` records `backend='numpy'`; and
`resolve_route(256, backend="iree", eval_mode="fused").fallback_backend == "numpy"`.

### 16.4 What the envelope and the scratch table actually hold (S1, S2)

Exact bytes one launch allocates, and what the pre-fix formula compared (value path unless noted):

| shape | lanes (`n_rb` × block) | scratch | value buffer | chart | total | old formula |
|---|---|---|---|---|---|---|
| `p2048-L128-N1024`, `out_len=8` | 4 × 256 (1 024) | 1 024.0 MiB | 64.0 | 0 | **1 088.0** | 1 024.0 → the old check SERVED this at a 1 050 MiB budget |
| `p2048-L128-i32-N1024-d7` | 4 × 256 | 1 024.0 | 8.0 | 0 | **1 032.0** | 1 024.0 (served at 8 192) |
| `p256-L64-i32-N1024` (loss) | 4 × 256 | 64.0 | 0 | 0.004 | 64.0 | 64.0 |
| `p1000-L64-i32-N1000` (loss, rows not a block multiple) | 4 × 256 (1 024 lanes) | 250.0 | 0 | 0.015 | **250.02** | 244.14 |

- **S2:** `_check_envelope(pop, tree_len, rows, max_scratch_mb, *, block_rows=None, num_warps=None,
  out_len=1, holds_values=True) -> int` charges `pop*n_rb*tree_len*block*4` (with the row-block ROUNDING, so
  1 000 rows at 256 lanes is 1 024 lanes) + `pop*rows*out_len*4` (the value path, and the multi-output loss
  path) + `pop*n_rb*4` (the loss chart), declines with a message naming all three components, and returns the
  resolved budget so the scratch TABLE is budgeted by the number the envelope was checked against. The campaign's
  measured peak-allocated tracks the scratch term within 3.5 %, the value buffer on top (`test/benchmark/evoxc/results/t1a1_fused/ENVELOPE.md`).
  Battery `_check_envelope_bytes`: the 1 050 MiB call now declines naming `value buffer`; `d7` at 8 192 returns `8192`.
- **S1:** the scratch table is an LRU bounded by total BYTES, `n_prog*stack_cap*block_rows*4` against the call's
  `EVOGP_FUSED_MAX_SCRATCH_MB` (it used to be bounded only by an entry count of 8, i.e. up to 8× the budget).
  Counters `fused_scratch_bytes` / `fused_scratch_evictions` (`_check_scratch_budget`: 4.00 MiB in 6 entries
  ≤ budget; a 1 MiB budget on a fresh shape class evicts, +6 evictions, 4.00 → 0.00 MiB).
- **The verdict is memoized** on its eight resolved inputs (`_ENVELOPE_MEMO`, 32 entries, FIFO, 0.54 µs against
  2.34 µs): a pure-function memo, so an in-process knob flip (`EVOGP_FUSED_MAX_SCRATCH_MB=1`,
  `EVOGP_FUSED_BLOCK_ROWS=100`, `EVOGP_FUSED_NUM_WARPS=0`) is a miss on the next call and needs no invalidation.

### 16.5 Knobs, tables and a gate that measures its claims (S3, S5, S6, S7, S9)

- **S3:** `EVOGP_FUSED_BLOCK_ROWS` is a CAP rounded DOWN to a power of two (`100` → `64`, the rule
  `tl.arange` needs and the one `plan` applies), reported as the effective value by
  `fused_cache_stats()["fused_block_rows"]`; a value below 1 still raises naming the variable. Battery
  `_check_geometry_knobs` serves a real call at `=100`, checks `plan(...)["block_rows"] == 64`, and checks
  `=abc` → stats `-1` with the resolver still raising.
- **S5:** the op table is data, not a fifth hard-coded id list — `emitted_function_ids()` derives the ids from
  `utils.Func` and a module-import cross-check compares them against `utils.FUNCS` (by arity) AND against the
  ids the `if/elif` chains actually compare (an AST scan, so a removed branch fails at import); `DELTA` is
  single-sourced from `utils.DELTA` as a `tl.constexpr`. Battery `_check_operator_table`: binary 13/13
  (ids 1–13), unary 15/15 (ids 14–28).
- **S7:** `fused_cache_stats()` never raises — an unusable ambient knob reports `-1` (informational), while the
  resolver still raises `ValueError`.
- **S9:** `num_warps` default 4 → 1 in the geometry note; `_prepare`'s documented order is the code's order
  (cheap shape probes → `_decline_if_numpy` → `_require_available`); `fused_available()` is now the real gate
  inside `_require_available`; dead `import math` / `_TRITON_IMPORT_ERROR` removed; `fused_check.py --warps`
  defaults to `fused_kernels.DEFAULT_NUM_WARPS` and its `sys.path` mutation moved from import time into `main()`.
- **S6 (the gate):** every parity set now carries a MEASURED tolerance instead of one global `VALUE_RTOL`, and
  the four sets became five (the new `compare_unary` covers `GT` / `LE` / `NEG` / `ABS` / `SINH` / `COSH`, the
  six operators the gate never exercised — `sinh`/`cosh` are the hand-written `(exp(x) ∓ exp(-x))·0.5`
  compositions, exactly where divergence from `torch.sinh`/`torch.cosh` could hide):

  | set | `value_tol` | measured `max_rel` (value) | value bitwise | loss `max_rel` (MSE / MAE) |
  |---|---|---|---|---|
  | `arithmetic` | 0.0 | 0.0 | yes | 1.46e-07 / 1.96e-07 |
  | `compare_unary` (new) | 0.0 | 0.0 | **yes** (`sinh`/`cosh` included) | 1.03e-07 / 1.77e-07 |
  | `ternary` | 1e-7 | 0.0 | yes | 1.65e-07 / 1.59e-07 |
  | `rich` | 1e-7 | 0.0 | yes | 1.06e-07 / 1.77e-07 |
  | `loose` | 1e-7 | 0.0 | no — bytes differ, `max_abs` 0.0 | 1.11e-07 / 1.77e-07 |

  The `1e-7` margins are "measured 0 with a margin", not observed deviations. The handmade adversarial rows now
  carry a measured `bitwise` flag and are enforced at `value_limit=0.0` (8 of 10; `mixed_deep` and `mixed_bag`
  stay at `VALUE_RTOL` because their DIV-by-a-literal-zero row writes a different NaN payload — values agree,
  `max_abs` 0.0). Loss keeps `LOSS_RTOL` 1e-5 on every set (accumulation order only).
- **The bitwise claim is therefore per surface:** the value path is bit-for-bit against
  `src/evogp/tests/reference_eval.py` on the arithmetic, comparison, ternary and transcendental surfaces
  (`sinh`/`cosh` included); the loose (`loose_*`) surface can differ in the NaN payload bit while the value
  agrees — so the kernel docstring's "bit-for-bit" line names the sets instead of claiming all of them.

### 16.6 What the hardening costs (six cells, public API, vs the pre-fix tree)

One process per cell, `EVOGP_EVAL_MODE=fused`, 3 warmup + 15 reps × 5 rounds, four order-alternated rounds,
medians; the pre-fix arm is `git archive 47cd344^` extracted to a scratch dir and driven by the same script:

| cell | value pre → now | loss pre → now | per_row pre → now | value speedup pre → now |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.1024 → 0.1053 (+2.9 %) | 0.1240 → 0.1275 (+2.9 %) | 0.0947 → 0.0977 (+3.2 %) | 40.5× → 39.4× |
| `p256-L64-i32-N1024` | 0.1021 → 0.1055 (+3.4 %) | 0.1264 → 0.1273 (+0.8 %) | 0.0964 → 0.0980 (+1.7 %) | 43.9× → 42.5× |
| `p256-L64-i32-N4096` | 0.1104 → 0.1125 (+1.9 %) | 0.1241 → 0.1268 (+2.2 %) | 0.0948 → 0.0977 (+3.1 %) | 45.1× → 44.2× |
| `p1000-L64-i32-N1024` | 0.1093 → 0.1127 (+3.1 %) | 0.1240 → 0.1280 (+3.2 %) | 0.0949 → 0.0983 (+3.6 %) | 44.6× → 43.2× |
| `p4096-L64-i32-N1024` | 0.1436 → 0.1455 (+1.3 %) | 0.1575 → 0.1582 (+0.4 %) | 0.0995 → 0.1013 (+1.8 %) | 73.0× → 72.1× |
| `p2048-L128-i32-N1024-d7` | 0.1729 → 0.1765 (+2.1 %) | 0.1860 → 0.1889 (+1.6 %) | 0.1265 → 0.1299 (+2.6 %) | 152.1× → 149.0× |

Where it goes (paired same-process stubs, `p256-L64`, host-only): the **kernel is unchanged** (kernel-only A/B
±3 %, as often faster as slower — the clamps and the masked pops cost nothing measurable), the exact-bytes
envelope is **+0.54 µs** once memoized (it was +2.1…3.9 µs unmemoized, which is what the first re-timing caught),
and the contiguity guard is **+0.4…1.0 µs** of flag tests. Measured public-API delta: **+2.8 µs mean**, ~+2.4 %;
the residual ~1.5 µs is inside the ±3 % inter-process band of this shared box.
A first (2-round, not order-alternated) protocol read +8…+11 % on the p256 cells and −9 % on `p4096`; that spread
was arm-order drift, which is why §15's protocol keeps the order alternation.

### 16.7 Residual gaps after the round

- The route still declines nothing on STRUCTURE: a malformed tree produces a bounded, deterministic,
  arbitrary number instead of the specialized engine's `ValueError` (by design, §16.1).
- One scratch block is shared per `(device, n_prog, stack_cap, BLOCK_R)` shape class — single-stream use, or
  the pop masks' `0.0` and the block's previous contents can interleave (documented, not enforced).
- `EVOGP_FUSED_BLOCK_ROWS` values between two powers of two silently become the lower one (reported in the
  stats, not refused).
- `MAX_OUT_LEN` is 8: a wider multi-output population declines by name.
- A single-output request over an `OUT_NODE`-tagged population is REFUSED host-side in `fused_engine._prepare` (`evogp.tree.multi_output.check_multi_output`, a `ValueError` naming row and node) before any launch — the kernel's type mapping is unchanged and its "any other type is a ternary `if`" fall-through is now UNREACHABLE for validated input, so a mis-wired `output_len` cannot produce a plausible wrong number on this route (the tag is not a decline: `fused_declines` / `fused_kernel_launches` stay flat, pinned by `fused/tests/test_fused_engine.py` and `fused_check --stage declines`).
- The records under `test/benchmark/evoxc/results/t1a1_fused/` predate §16.4, so their `geometry.scratch_bytes` is the older round figure and
  their geometric keys carry no value-buffer/chart terms. Their `envelope_ok` is unaffected: all 79 stored records
  still pass today's rule, which `fused_envelope.py --stage records` re-checks read-only (it never rewrites them).
- Module sizes are deliberate single units: `fused_engine.py` ~1.1 k lines and `fused_kernels.py` ~1.2 k lines
  (a compilation unit each, like `evoxc_engine.py`, and each says so at the top of the file); `fused_check.py`
  3 151 lines (one self-contained battery).

---

## 17. The kernel-performance round — the register-window stack
The host-glue round (§3) left the route with a warm call that is 22–63 % DEVICE time (§3, profiler column), so this
round attacked the kernel body in `tree/fused_kernels.py` and nothing else. Its verdict: one candidate cleared the
±3 % noise bar and ships (a three-slot register window over the walk's memory stack, gated to wide blocks);
everything else was measured and reverted. **The shipped per-call metric cannot resolve kernel work below
5–7 %** — see "the metric" below — so every number here is a torch-profiler device duration on 40 launches per
sample, and the six-cell public-API table is quoted with its noise floor.

### 17.1 Profile of record (device µs, `BLOCK_R` 256 / 1 warp, HEAD `e391cda`)
| cell | value | loss (MSE) | per_row |
|---|---|---|---|
| `p256-L32-i8-N1024` | 17.03 | 17.07 | 12.27 |
| `p256-L64-i32-N1024` | 16.54 | 16.90 | 12.34 |
| `p256-L64-i32-N4096` | 24.40 | 25.46 | 12.24 |
| `p1000-L64-i32-N1024` | 24.06 | 25.27 | 13.60 |
| `p4096-L64-i32-N1024` | 56.70 | 56.99 | 16.73 |
| `p2048-L128-i32-N1024-d7` | 86.20 | 86.01 | 45.49 |
Ablations at the same geometries (`p256-L64` / `p4096-L64` / `d7`), each a stripped-down variant of the same kernel:
metadata skeleton with no vector work **6.8 / 36.7 / –**; no stack LOADS 15.4 / 43.2 / 71.7; no stack STORES
13.1 / 38.3 / 65.4; neither **10.6 / 28.1 / 49.5**; no `x` loads 14.5 / 46.8 / 71.4; the op ladder replaced by a tiny
op (diagnostic only) 14.8 / 51.4 / 75.3.
Three cost centres, in order: (1) the **scalar metadata chain** — one `node_type` and one `node_value` load plus the
per-node branch resolution per node, ~40 % of the iterations' work; (2) **vector stack traffic** — 6.6 / 28.6 /
36.2 µs, i.e. 40 % / 50 % / 42 % of the kernel, because all `BLOCK_R` lanes execute both operand selects on every
pop and store on every push; (3) the **inlined op ladder** — 2–11 µs, and only when the function mix is unlucky.
Branch divergence is **not** a cost centre: the 4 node kinds are predicated selects, both operands are always
computed, and a per-kind reordering therefore cannot pay (measured: reordering the arithmetic path branch-free,
i.e. `pre_unmasked_*`, is −1…−6 %, all of it recovered by the window below).

### 17.2 Candidate matrix (measured, `p4096` / `p256-L64` device µs unless noted)
| candidate | result | verdict |
|---|---|---|
| **register window over the top K stack slots** (`K = 3`) | 56.7 → 50.5 (−10.9 %) / 16.5 → 15.8 (−4.0 %) at K = 3, and the same win on the loss path | **SHIPS**, gated (§17.3) |
| same, K = 1 / 2 / 4 / 6 / 8 | K = 4 → 55.6, K = 6 → 71.2, K = 8 → 97.4 at `p4096` (select pressure) | K = 3 is the optimum |
| same, ungated (every geometry) | per_row **+7 % / +7 % / +11 %** (`p256-L64-N4096` / `p4096` / `d7`) | rejected → gated to `BLOCK_R >= 32` |
| pre-load BOTH operands before the branch (`pre_masked_*`) | +2…+14 % | reverted |
| pre-load, unmasked (`pre_unmasked_*`) | −1…−6 %, fully recovered by the window | reverted as redundant |
| row block as the OUTER grid dimension (node loop inner) | flat (±1 %) | reverted |
| `num_stages` 1 / 4 | flat | reverted |
| packed / int32 metadata conversion at prepare time | not pursued: the ablation puts the chain's cost in its DEPENDENCE (one load per node feeding the branch), not in the load's width, and narrowing it would change the channel schema the facade caches on | dropped |
| geometry: `BLOCK_R`/warps re-run of §14 | minimax alternates between the shipped `256/1` and `B512-W2` (0.009 each) across the two measured windows, the shipped default 1.00–1.05× of the best cell; `B128-W1` 0.010 and `B512-W1` / `B512-W4` 0.010–0.011 in the same band | surface flat within ~5 %; default kept |

### 17.3 What ships
`STACK_WINDOW = 3` keeps the walk's top three stack slots in registers (`w0/w1/w2`), the memory stack keeps the deep
tail, and both halves use the SAME clamped slot index, so the values are bit-identical by construction — a
population deeper than 3 is served exactly as before (the `d7` cell's depth histogram reaches past the window).
The window is compiled out at `BLOCK_R < 32` (`_WINDOW_MIN_BLOCK`): a register tile pays only when a lane carries a
vector of rows, and the per-row surface (`BLOCK_R == 1`, one data point per tree) is the measured regression above,
so that geometry keeps the old code **verbatim** (a constexpr branch, not a mask). It is a Python constant of the
trace, so it adds no launch parameter and no specialization key, and `_fused_multi_kernel` is untouched (its cost is
the prediction accumulator, §7.5).

### 17.4 The A/B (kernel-only, 3 alternating passes, medians)
| cell | value A → B | loss A → B | per_row A → B |
|---|---|---|---|
| `p256-L32-i8-N1024` | 17.06 → 16.58 (−2.8 %) | 17.07 → 16.87 (−1.2 %) | 12.27 → 12.29 (0.0 %) |
| `p256-L64-i32-N1024` | 16.54 → 15.26 (−7.7 %) | 16.90 → 15.74 (−6.9 %) | 12.34 → 13.28 (+7.6 %, ungated) |
| `p256-L64-i32-N4096` | 24.40 → 23.61 (−3.2 %) | 25.46 → 24.62 (−3.3 %) | 12.24 → 12.29 (0.0 %) |
| `p1000-L64-i32-N1024` | 24.06 → 22.15 (−7.9 %) | 25.27 → 22.81 (−9.7 %) | 13.60 → 14.53 (+6.9 %, ungated) |
| `p4096-L64-i32-N1024` | **56.70 → 50.52 (−10.9 %)** | **56.99 → 49.90 (−12.4 %)** | 16.73 → 16.75 (0.0 %) |
| `p2048-L128-i32-N1024-d7` | 86.20 → 82.93 (−3.8 %) | 86.01 → 82.10 (−4.5 %) | 45.49 → 45.21 (−0.6 %) |
The `per_row` column's two "+" entries are the UNGATED window (what it cost before the gate); gated, the surface
reproduced its baseline to 0.0 % on all six cells. The `p256-L64` row's B column comes from the ungated build, which
is the same code at `BLOCK_R` 256 (the gate only changes the `BLOCK_R == 1` specialization). Value and loss digests are identical between the arms on every
cell (`batch_forward` and `SR_fitness` outputs, sha256), i.e. bitwise.

### 17.5 The metric — why this round reports kernel µs
The shipped per-call metric (one process per cell, sync-bracketed warm medians, reps 10) is too noisy at these call
lengths to resolve the change, and the null experiment says so: **A against A** — the same tree, two arms,
three order-alternated passes each, reps 10 — spreads **−10.3 … +1.0 %** on `value`, **−12.3 … +0.0 %** on `loss`
and **−5.7 … +1.4 %** on `per_row`; at reps 100 it still carries a systematic ±2–4 % first-arm advantage (the
`per_row` surface, whose compiled code is IDENTICAL, read +2.4 … +4.2 % in one campaign and +1.6 … +7.4 % in
another). The A/B above the fold is therefore the profiler, and the per-call table below is quoted as the record's
own before/after with that floor attached:
| cell | value A → B | loss A → B | per_row A → B |
|---|---|---|---|
| `p256-L32-i8-N1024` | 0.0736 → 0.0769 | 0.0857 → 0.0904 | 0.0696 → 0.0735 |
| `p256-L64-i32-N1024` | 0.0725 → 0.0753 | 0.0855 → 0.0909 | 0.0689 → 0.0728 |
| `p256-L64-i32-N4096` | 0.0812 → 0.0836 | 0.0912 → 0.0937 | 0.0687 → 0.0725 |
| `p1000-L64-i32-N1024` | 0.0811 → 0.0823 | 0.0912 → 0.0919 | 0.0702 → 0.0744 |
| `p4096-L64-i32-N1024` | 0.1142 → 0.1108 (−3.0 %) | 0.1232 → 0.1197 (−2.8 %) | 0.0735 → 0.0747 |
| `p2048-L128-i32-N1024-d7` | 0.1403 → 0.1449 | 0.1540 → 0.1551 | 0.0996 → 0.1069 |
Only `p4096` reads in the expected direction at this resolution; everything else is inside the null spread (the
`per_row` column is expected to be EXACTLY zero and is not, which is the floor's own proof).

### 17.6 Gates after the change
`fused_check --stage all` → **PARITY 56/56** comparisons inside tolerance (value `max_abs = 0.0` on all three
function sets, loss `max_rel` ≤ 3.8e-07), **MULTI 65/65**, **GOLDENS 10/10** inside 1e-4, **DECLINES 57/57**,
**WIRED 32/32**, **TIMING** worst-cell 102.59× `value` / 90.99× `loss` against `torch-batched`, **TUNING** minimax
alternates between the shipped `EVOGP_FUSED_BLOCK_ROWS=256` / `EVOGP_FUSED_NUM_WARPS=1` and `B512-W2` (0.009 each) across the two measured windows, with the shipped default at 0.010 on `p256-L64-i32-N4096`, i.e. 1.00–1.05× (default kept).
`pytest src/evogp/tests -q` → **1 308 passed, 7 skipped, 0 failed**.

### 17.7 What this leaves
- **The launch path, not the kernel, is the p256 headroom.** A warm `batch_forward` at `p256` is 0.065 ms of which
  ~0.060 ms is the `launch_value` CALL and only 0.0165 ms the kernel, so the Triton launcher's own Python
  (`JITFunction.run` → binding, cache lookup, `cuLaunchKernel`) is now the largest single item. A pre-bound
  `CompiledKernel` dispatch is the next candidate and would move the per-call metric one-for-one; nothing in this
  round touched it.
- **The metadata chain** (one `node_type` + one `node_value` load and a 4-way resolution per node, ~40 % of the
  iterations' work) is the largest remaining IN-kernel item and the hardest: it is the walk itself, and every
  variant that pre-loaded the operands outside the branch lost more in registers than it saved in loads.
- **The floor is measured, not assumed**: `STACK_WINDOW` 4 and above is worse than 3, `num_stages` and the grid
  order are flat, and the geometry minimax alternates between the shipped `256/1` and `512/2` across the two measured
  windows with the default inside 1.00–1.05× (§14: worst value ratio 0.009–0.010 on `p256-L64-i32-N4096`; every other
  config sits at 0.010–0.037, so no alternative clears the noise floor).

## Measured detail (moved from tree/CONTEXT.md)

Facade bookkeeping and route-shape facts that this campaign's sections above rely on (no number below replaces one
stated above; where they overlap, the campaign's own tables win):

- `fused_cache_stats()` returns 25 keys: `fused_calls` / `fused_declines` (facade attempts / declines),
  `fused_xt_hits` / `fused_xt_misses` / `fused_xt_entries` / `fused_xt_maxsize` (the transpose table, touched ONLY
  by `per_row=False` and the loss path), `fused_tag_hits` / `fused_tag_misses` / `fused_tag_entries` /
  `fused_tag_maxsize` (the `OUT_NODE` verdict table), `fused_plan_hits` / `fused_plan_misses` / `fused_plan_entries` /
  `fused_plan_maxsize` (the prepared-call table), `fused_kernel_launches` / `fused_kernel_compiles` / `fused_scratch_allocs`
  / `fused_scratch_reuses` / `fused_scratch_entries` / `fused_scratch_evictions` / `fused_scratch_bytes` (the launch
  side, byte-bounded per §16.4), `fused_bound_launches` / `fused_dispatch_declines` (the launch PATH: pre-bound
  `CompiledKernel.run` launches and the specializations that declined back to `JITFunction.run`, echoed from
  `fused_kernels.dispatch_stats()` alongside its `jit_launches`), plus the informational `fused_block_rows` /
  `fused_num_warps` (which report `-1` instead of raising when the ambient knob is unusable).
- `fused_calls` counts ATTEMPTS (bumped before the try in `evaluate_fused` / `sr_fitness_fused`), so a decline moves BOTH
  counters and a shape-contract `ValueError` moves `fused_calls` alone: proving a call was served means
  `Δcalls − Δdeclines == n` AND `Δfused_kernel_launches == n`, while `fused_kernel_compiles` is "first launch of
  this `(per_row, reduce, use_mse, block, warps)` key" — the key set lives in `fused_kernels._KERNELS` and is NOT
  reset by `clear_caches()`, so a counter reset can under-report it.
- `EVOGP_FUSED_HOST_MEMO` has NO per-call reader: `_default_host_memo` reads it once at import to size `XT_MEMO`
  (`fused_engine._default_host_memo`, called once when `XT_MEMO` is built), so a change needs a fresh process.
- There is NO per-stage timer surface on this route (unlike `batch_stages.py`): the split must be reconstructed by
  timing `_prepare` / `_transposed` / `fused_kernels.launch_value` / `launch_loss` separately, which is what
  `fused_check._stage_split` does for the value/loss kernel-vs-host columns; the route has no host readback, so a
  "readback" stage does not exist.
- An explicit `EVOGP_EVOXC_BACKEND` changes only the SERVED path, never the fallback: `iree` keeps
  `backend='iree'` with `fallback_backend='numpy'` (`Route(False, 'iree', 'numpy', engine='fused')`, so a decline
  lands on specialized+`numpy`, never on a per-tree iree compile) and `numpy` still returns `engine='fused'` with
  `backend='numpy'`, which `_decline_if_numpy` refuses on every call — a probe printing only
  `resolve_route(pop, eval_mode='fused').engine` therefore cannot distinguish a served call from a guaranteed
  decline.
- The facade declines (never guesses) on no triton, no device, a host tensor, `backend != "iree"`,
  `output_len > MAX_OUT_LEN`, `rows < 1`, `pop < 1` and the `EVOGP_FUSED_MAX_SCRATCH_MB` envelope, and maps
  `fused_kernels.FusedUnavailable` onto that decline so an optional route never raises `ImportError`.
  `output_len = 0` and `tree_len < 1` are shape-contract `ValueError`s, not declines.
- Prep is device-side only: `per_row=False` transposes `x` once into the identity-memoized `XT_MEMO` (a
  `DenseMemo`), `per_row=True` needs nothing.
- `fused_check.py` is ~3.1k lines on purpose: like `dense_adversarial_check.py` and `device_resident_check.py` it is
  one self-contained dev battery (parity sets, the golden cross-check, the decline matrix, the wiring gate, the
  six-cell timing and the geometry sweep share one workload builder and one comparison helper), so do not split it
  into modules.
- `--stage wired` is the wiring gate: it drives the three `Forest` call sites with `EVOGP_EVAL_MODE=fused` set
  in-process and asserts they equal the `specialized` reference (value bitwise or ≤1e-6 relative, loss ≤1e-5), that
  the facade SERVED each call (its counters move), that `resolve_route` names fused on `iree` with a `numpy`
  fallback while `pop_size == 1` / a population below `EVOGP_FUSED_MIN_POP` / a box with no device stay
  `specialized`+`numpy`, that `auto` selects it from `EVOGP_FUSED_MIN_POP` up on a device box and never below it, and that a multi-output population (`output_len=2`) is SERVED
  by `_fused_multi_kernel` with the specialized reference's numbers, one `fused_calls` and no `fused_declines`;
  `--stage timing --wired` re-times the six cells through the same public calls. Recorded: PARITY 56/56 oracle comparisons,
  GOLDENS 10/10 at 1e-4, DECLINES 57/57 decline/contract checks, WIRED 32/32 wiring checks (the `--stage multi` set of §11 adds
  MULTI 65/65 comparisons on top).

The earlier six-cell sweep that preceded this campaign (kept for the record; `BLOCK_R` 256 / 1 warp): **48.6x-167.6x**
faster than the `torch-batched` arm on the value path, 43.4x-156.4x on the loss path, 56x-150x `per_row`; the kernel
was 72-82 % of a warm call (0.060-0.128 of 0.083-0.157 ms), the transpose 0.018-0.020 ms, and the cold first call
375 ms (warm Triton disk cache) to 1.38 s (cold) per specialization, ~1.9 ms for the next one.

### Facade surface and route wiring (relocated from tree/CONTEXT.md)
`fused_engine` is the facade in the dense shape: `FusedUnsupported(reason, device_fallback=False)`, `evaluate_fused`, `sr_fitness_fused`, `fused_available`, `fused_cache_stats()` (25 all-`fused_*` keys, `fused_bound_launches` / `fused_dispatch_declines` included), `clear_fused_cache` (registered with `evoxc_engine._register_cache_reset`) and the knob readers `resolve_block_rows` / `resolve_num_warps` / `resolve_max_scratch_mb` / `resolve_min_pop`.
It declines (never guesses) on no triton, no device, a host tensor, `backend != "iree"`, `output_len > MAX_OUT_LEN`, `rows < 1`, `pop < 1` and the `EVOGP_FUSED_MAX_SCRATCH_MB` envelope, and it maps `fused_kernels.FusedUnavailable` onto that decline so an optional route never raises `ImportError`.
Both entry points take an optional keyword-only `out=None` destination: `sr_fitness_fused(..., out=buf)` writes the `(pop_size,)` loss into `buf` (`torch.sum(partials, dim=1, out=buf)` then `div_(rows)`) and returns that same tensor, and `evaluate_fused(..., out=buf)` hands the kernel the `(pop, rows, out_len)` value buffer (`launch_value(..., out=...)`, reused as-is, `out.squeeze(1)` for `per_row=True`) instead of allocating one.
`out` is validated before the launch (`_out_destination` / `_same_device`: a torch tensor, exactly the kernel's shape, float32, contiguous, `requires_grad=False`, on the input's device, and never together with a contradicting `out_device`) and a bad one raises `ValueError` naming `out`, so the destination is a caller contract rather than a silent reinterpretation.
With `out=None` both entry points allocate exactly as before, which is why the public wrappers (`Forest` / `Tree` / `CombinedForest`) and the `fused_*` counters are untouched by it; the one shipped caller that passes a destination is `evogp.gp.fused_eval.FusedDeviceEvaluator`, which points the kernel at an exported `evoxc` device buffer so no per-call copy or import step is needed.
Prep is device-side only: `per_row=False` transposes `x` once into the identity-memoized `XT_MEMO` (a `DenseMemo`, `EVOGP_FUSED_HOST_MEMO` 16 entries), `per_row=True` needs nothing.
Route wiring: `EVOGP_EVAL_MODE=fused` is named by `EVAL_MODES` and `ENGINES`, `Route.use_fused` reports it and `resolve_route`'s `fused` branch decides — on an explicitly named backend, or with a device usable from torch AND `pop_size >= EVOGP_FUSED_MIN_POP` (16); otherwise (and on a CPU box) the route IS specialized+`numpy` with `engine="specialized"`.
Its backend SPELLING is `iree` and its fallback is ALWAYS specialized+`numpy`, on every arm: an explicit `EVOGP_EVOXC_BACKEND=iree` builds `Route(False, 'iree', 'numpy', engine='fused')`, so a decline lands on specialized+`numpy`, never on a per-tree iree compile, while `EVOGP_EVOXC_BACKEND=numpy` still reports `engine='fused'` and declines on every call — the engine name is never a served-call proof.
`Route.__post_init__` refuses `use_dense=True` with `engine="fused"`; `should_use_dense` is `False` for it; `resolve_fused_min_pop` is the single parser of `EVOGP_FUSED_MIN_POP` and delegates to `fused_engine.resolve_min_pop` through a LOCAL import so the module DAG stays one-way.
All five call sites carry the fused attempt between the dense one and the specialized tail, catching exactly `fused_engine.FusedUnsupported` and continuing on `route.fallback_backend`.
There is NO per-stage timer surface on this route (the split must be reconstructed by timing `_prepare` / `_transposed` / `fused_kernels.launch_value` / `launch_loss` separately, which is what `fused_check._stage_split` does), and there is no readback stage at all.
`import evogp.fused` imports torch at module scope, and `import triton` happens only inside `kernels.kernels()`, so the package stays GPU/triton-free (a triton-less box gets `FusedUnsupported` / `FusedUnavailable` from `kernels()`, never an ImportError); `TRITON_CACHE_DIR` must be writable or the first launch dies with `OSError: [Errno 30] Read-only file system: '/root/.triton'` — on this box `$HOME` is read-only, so every fused run needs `TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache`.
`fused_check.py` is ~3.1k lines on purpose: like `dense_adversarial_check.py` and `device_resident_check.py` it is one self-contained dev battery (parity sets, the golden cross-check, the decline matrix, the wiring gate, the six-cell timing and the geometry sweep share one workload builder and one comparison helper), so do not split it into modules.

---
## 18. The launch path — a pre-bound `CompiledKernel` dispatch (§17.7's remaining item)

**Why this was the lever.** §3's profiler row says the device kernel is 16.5 µs at `p256-L64` while the shipped
`batch_forward` used to read 0.078–0.086 ms: the other ~60 µs were Python. A **CPU-only** decomposition of one warm
launch (no per-sample sync, so the numbers are launch cost and not device time) put **22.9 µs in
`JITFunction.run`** — signature binding, option resolution, `compute_cache_key`, the kernel-cache lookup, the
`launch_metadata` construction and the hook plumbing — against **5.1 µs for the backend launcher itself**
(`HIPLauncher.__call__`: 10 positional params plus `*args`). Everything else in the call was small and already
memoized (`torch.empty` of the value buffer 2.4 µs, `_value_key` 3.2, `PLAN_MEMO` lookup 0.7, `_bump` 0.3,
`resolve_route` 3.5). So the fix is not kernel work: call the compiled kernel directly.

**What ships** (`fused_kernels._BoundLaunch`, `EVOGP_FUSED_DISPATCH`). A specialization's first prepared call still
goes through `JITFunction.run(..., warmup=False)` — that call is what produces/loads the compiled kernel — and it
*arms* a launch descriptor from the resulting `CompiledKernel`: `kernel.run` (the backend launcher),
`kernel.function` (the raw `CUfunction`), `kernel.packed_metadata`, the grid, the launch stream getter
(`triton.runtime.driver.active.get_current_stream`, 0.1 µs), the launch options and the device index. Every later
warm call calls the launcher with the already-bound argument tuple and **skips** binder, cache lookup, options and
metadata construction. The descriptor is stored on the memoized plan, so `PLAN_MEMO` invalidation and
`clear_fused_cache()` re-arm it (pinned by `test_clear_fused_cache_rearms_the_dispatch`). Nothing about the numerics
changes: the same compiled kernel, the same arguments, the same grid and stream — `test_bound_and_jit_agree_bitwise`
asserts `torch.equal` on `batch_forward` / `SR_fitness` / `forward` output.

**The Triton API it binds to: 3.7.1** (`/tmp/rocvenv/lib/python3.12/site-packages/triton/`), i.e.
`JITFunction.run(jit.py:708)` returning a `CompiledKernel` whose `.run` property is `HIPLauncher` on ROCm (and
`CudaLauncher` on CUDA — the same shape). `CompiledKernel.__getitem__(grid, stream)` is the *documented*
`(grid, stream)` runner, but on this Triton it takes a 3-tuple grid (`IndexError: tuple index out of range` on a
2-tuple), so the code calls `ck.run` directly — the same ~5 µs, one level lower. Every step is **feature-detected**
(`_launcher_takes_launch_args` inspects `type(launcher).__call__` for 9 positional parameters plus varargs; the launch
hooks are `HookChain`s and are skipped only when their `.calls` list is empty — `_hook_active`), and anything
unusable increments `_STATS["declines"]` and serves the call through `JITFunction.run` as before, so an older or
newer Triton silently degrades to today's path rather than breaking. `import triton` stays lazy;
`fused_kernel_launches`, the memo invalidation, the `EVOGP_FUSED_*` knobs and the sync-free warm path are untouched;
no new `.item()` anywhere, and `dispatch_stats()` (`bound_launches` / `jit_launches` / `declines`, surfaced as
`fused_bound_launches` / `fused_dispatch_declines` in `scratch_stats()`) makes the served path observable.

**The A/B (ABBA, one process per cell, idle-GPU guard, warm medians; `EVOGP_FUSED_DISPATCH` is the only difference).**
`jit` = today's path, `bound` = the shipped default; µs, six harness cells:

| cell | `bf` jit → bound | Δ | `SR_fitness` jit → bound | Δ | `forward` jit → bound | Δ |
|---|---|---|---|---|---|---|
| `p256-L32-i8` | 79.3 → **52.4** | −33.9 % | 114.0 → **71.1** | −37.6 % | 78.7 → **48.3** | −38.6 % |
| `p256-L64` | 85.0 → **52.6** | −38.1 % | 104.7 → **62.7** | −40.1 % | 77.2 → **48.3** | −37.4 % |
| `p256-L64-N4096` | 89.8 → **66.5** | −26.0 % | 104.9 → **78.0** | −25.6 % | 78.2 → **49.9** | −36.2 % |
| `p1000-L64` | 84.7 → **64.0** | −24.4 % | 109.9 → **75.9** | −30.9 % | 76.2 → **52.8** | −30.7 % |
| `p4096-L64` | 113.9 → **94.9** | −16.7 % | 138.9 → **102.6** | −26.1 % | 85.8 → **50.1** | −41.6 % |
| `p2048-L128-d7` | 157.2 → **121.5** | −22.7 % | 172.1 → **136.6** | −20.6 % | 118.9 → **81.9** | −31.1 % |

The floors were ≥30 % on `p256-L32` / `p256-L64` / `p1000`, ≥10 % on `p4096` / `d7`: the two `p256` cells clear 30 %
(−34 % / −38 %), `p1000` lands at −24 % (short of its 30 % because 1 000 trees cost more device time per launch,
§17.1), and `p4096` / `d7` clear 10 % with room. `SR_fitness` and `forward` are never regressed and improve more
than `batch_forward` (their larger host share). A same-session single-process spot check at `p256-L64` read
value 0.0624 → 0.0369 ms and loss 0.0760 → 0.0431 ms with the device at 16.4–16.8 µs, consistent with the campaign.
Against the harness's `torch-batched` arm in the same sessions, the bound column is **75–217×** on `batch_forward`
(4 017 / 4 586 / 5 019 / 5 263 / 10 494 / 26 320 µs on the six cells).

**The new floor.** A warm `p256` `batch_forward` is now ≈52 µs of which ≈16.5 µs is the kernel (**≈31 %**, up from
≈22 %), ≈5 µs is the backend launcher (≈10 %, was ≈27 % for `JITFunction.run`'s 23 µs) and ≈30 µs is the facade's
host glue (route resolution, `_prepare`-memo lookup, key build, the `contiguous` epilogue). At `p4096` the split is
≈57 µs device / 5 launcher / ≈33 host; on `d7` ≈86 µs device of 122 µs. So the launch path is no longer the
bottleneck: the small-cell headroom left is the ≈30 µs of per-call Python in the facade, and the device share is
what bounds any further kernel-level win.

**Not pursued, with the measurement that closed it.** A HIP/CUDA graph capture of the launch (`torch.cuda.CUDAGraph`,
replayed on the same static buffers) measures **5.8 µs CPU and 14.5 µs device per replay** — no better than the
pre-bound path (≈5.1 µs) and it would need a stale-buffer invalidation protocol on top, so it was dropped. The
`batch_forward` surface also cannot repay more per-call work: `gp_step` is unchanged (2.84–3.24 ms against
2.86–3.39 ms on the `jit` column) because a generation is 91 % operators (§4), which is why this round is priced
per call and not per loop.

## 19. The route-resolution memo — a host-glue win of 6–17 %, not 30 %

`dense_engine.resolve_route` used to re-read `EVOGP_EVAL_MODE`, `EVOGP_EVOXC_BACKEND` and the whole
`EVOGP_DENSE_*` / `EVOGP_FUSED_*` knob family and re-derive the policy on **every** call, which cost
5.85 µs per call (11.88 µs on the envelope arm that passes `rows`/`tree_len`). Since a warm fused
call is only 38–90 µs end to end, that is a measurable fraction of its host glue.

The memo (`_ROUTE_MEMO`, 32 entries, `clear_route_memo()`, `route_cache_stats()` →
`route_hits` / `route_misses` / `route_bypasses` / `route_entries` / `route_maxsize`) keys on the raw
arguments, the resolved mode and backend spelling, the **values of the availability probes that were
already known**, and every environment name the arm can read. It takes the fast reader of
`os.environ._data` (bounds-checked: a non-`dict` falls back to `os.environ.get`) and decodes exactly
what `os.environ.get` answers.

The laziness law is what makes it sound (the previous attempt broke 36 cases by *resolving* a probe
to build its key): `_route_probe_needs(mode, explicit)` says which of `_DEVICE_AVAILABLE` /
`_FUSED_AVAILABLE` the requested pair consults, and **a needed probe that is still `None` is a
bypass** — derived fresh, nothing stored, counter `route_bypasses` — so an explicit mode or backend
still probes nothing (`test_route_policy`'s isolation cases) and keeps its memo entry. A raise is
never stored; a `DenseUnsupported`/`FusedUnsupported` decline is an ordinary `Route` and is served
identically from the memo. `test_route_memo.py` pins all of it (21 cases), including a per-knob A/B
table that proves every knob moves or re-keys the answer.

`resolve_route(256)` **5.845 → 1.273 µs**, the `rows`/`tree_len` arm **11.884 → 1.558 µs**,
`should_use_dense(256)` **5.750 → 1.378 µs**.

### Warm medians, same-session ABBA (A = pre-round tree `091a6a8`, B = this tree, 10 reps)

`torch-batched` ratio is B against the harness's own whole-population torch arm; digests are
bitwise-identical between A and B on every cell.

| cell | surface | A (ms) | B (ms) | Δ | null A / B |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | `batch_forward` | 0.0495 | 0.0420 | **−15.0 %** | −1.0 / +3.7 % |
| | `sr_fitness` | 0.0598 | 0.0516 | **−13.6 %** | −12.7 / +2.6 % |
| | `forward` | 0.0444 | 0.0386 | **−13.0 %** | −5.1 / +1.8 % |
| `p256-L64-i32-N1024` | `batch_forward` | 0.0476 | 0.0425 | **−10.7 %** | −1.8 / −0.9 % |
| | `sr_fitness` | 0.0580 | 0.0524 | **−9.7 %** | −4.2 / +17.4 % |
| | `forward` | 0.0445 | 0.0396 | **−11.0 %** | +5.2 / +9.3 % |
| `p256-L64-i32-N4096` | `batch_forward` | 0.0599 | 0.0522 | **−12.9 %** | −4.5 / +2.6 % |
| | `sr_fitness` | 0.0694 | 0.0601 | **−13.4 %** | −10.2 / +4.4 % |
| | `forward` | 0.0444 | 0.0369 | **−16.9 %** | −1.6 / +1.0 % |
| `p1000-L64-i32-N1024` | `batch_forward` | 0.0557 | 0.0500 | **−10.3 %** | +1.1 / −2.1 % |
| | `sr_fitness` | 0.0644 | 0.0582 | **−9.6 %** | −4.5 / +1.7 % |
| | `forward` | 0.0468 | 0.0390 | **−16.7 %** | +3.2 / −4.9 % |
| `p4096-L64-i32-N1024` | `batch_forward` | 0.0856 | 0.0805 | **−5.9 %** | −1.3 / −2.6 % |
| | `sr_fitness` | 0.0903 | 0.0844 | **−6.5 %** | −3.4 / −3.4 % |
| | `forward` | 0.0471 | 0.0425 | **−9.8 %** | −7.5 / −1.4 % |

**Honest verdict: 6–17 % off, not the 30 %/15 % target.** The memo is the biggest single host-glue
item that was left, and it is now gone, but 1.3 µs of a 1.5–3 µs glue budget cannot be worth a third
of a 38–90 µs call: the kernel is 88–96 % of it (§3). The remaining glue is the plan-memo lookup,
the `_version`/`_geometry` checks and the launch dispatch, none of which is a policy re-derivation —
so a further ≥15 % needs the two items §18 lists (a wider plan-table hit rate on a *fresh*
population, and the pre-bound dispatch), not another host-side table.
The p4096 cell gains least (−6 %) because its 81 µs call is dominated by the per-program walk, which
the memo never touched. The `d7` cell did not finish inside this round's budget and is **not
measured** — the pre-round column there was 0.1460 ms `batch_forward` and no claim is made about it.
