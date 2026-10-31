# The device-resident `evogp.gp` generation loop — measurement report

Report of record for `python -m evogp.gp.measure_generation` (harness `measure_generation.py`, raw numbers in `evidence/device_loop.json`).
One "generation" below is exactly what a GP loop pays: evaluation + selection/crossover/mutation (+ the elite splice), i.e. `workflow.step` on the `evogp.gp` sides and `StandardPipeline.step` on the torch sides.
All numbers are warm medians of 15 timed repetitions after a 5-repetition warmup, one process per (side, cell), a device-wide sync around every timed repetition, behind an `rocm-smi` idle-device guard, on an AMD Instinct MI300X VF (ROCm 7.1, torch 2.13.0+rocm7.1, triton 3.7.1).
The workload is the acceptance config: `max_nodes=32, input_len=8, using_funcs=("+","-","*","/"), max_layer_cnt=4, const_prob=0.5, layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8`, `GPConfig(elite_count=2, tournament_size=4, crossover_prob=0.9, mutation_prob=0.2)`, `N` rows of `measure_fused.make_data`, seed 0, over FIVE cells: pop 64 / 256 / 1000 / 4096 at `N=1024` rows plus pop 256 at `N=4096` rows.

Sides (six, each in its own process per cell):

| side | what it runs |
|---|---|
| `gp` | `evogp.gp` driven DEVICE-RESIDENT: `init_gp(backend="iree", device="rocm")` + `graph_eval.GraphEvaluator` + `workflow.step` — the in-graph `evoxc` evaluation, two graph runs per generation. |
| `gp-fused` | the SAME device loop, same config, same `workflow.step`, same variation graph — only the objective changes: `fused_eval.FusedDeviceEvaluator`, i.e. the Triton kernel fed through zero-copy DLPack views of the device channels, whose `(pop_size,)` loss comes back as an `evoxc` DEVICE tensor. |
| `numpy` | the same stack on the numpy backend with the host `SymbolicRegression.evaluate` (the CPU reference and parity oracle of this stack). |
| `torch` | the torch stack's `StandardPipeline.step` — `Forest` / `GeneticProgramming` / `SymbolicRegression` built exactly the way `test/benchmark/evoxc/arms.py` builds them, on its default `auto` route (dense+iree). |
| `torch-fused` | the same torch pipeline with `EVOGP_EVAL_MODE=fused` (the torch stack's own Triton route, `Forest.batch_forward`). |
| `fused` | NOT a generation: the evaluation head-to-head of §7 (in-graph `evoxc` vs the fused kernel, from host channels and from device channels with the round trip included), which is why its records carry `parity`/`verdict` instead of `structure`/`trajectory`. |

`gp` and `gp-fused` differ in ONE argument, so the gap between those two columns is the evaluation lane and nothing else — which is what makes the comparison in §2 and §7 a measurement of the fused evaluator rather than of two wirings.
The torch stack selects by truncation (`survival_rate=0.3`, `elite_rate=0.01`) while `evogp.gp` runs a tournament of 4, so the two families are deliberately not the same operator mix; their trajectories are compared in §4, never their fitness values.

## 1. One generation, per cell

`evaluation` and `operator` (the variation graph) are two separately timed lanes on the `evogp.gp` sides, each inside its own sync bracket, so they sum to `total` within a few percent rather than exactly; `total` is the one-bracket number to quote.
On the torch sides `operator` is `total - evaluation`, both taken inside one step through a timed problem wrapper.

| cell | side | evaluation ms | operator ms | total ms | total min–max | cold s |
|---|---|---:|---:|---:|---:|---:|
| p64-N1024 | **gp (in-graph)** | 0.585 | 0.525 | **1.106** | 1.065–1.165 | 4.88 |
| p64-N1024 | gp-fused (device-native) | 0.600 | 0.578 | 1.161 | 1.136–1.193 | 3.38 |
| p64-N1024 | numpy | 5.796 | 4.335 | 10.105 | 9.705–11.320 | 0.08 |
| p64-N1024 | torch | 2.364 | 2.438 | 4.802 | 4.727–4.913 | 3.53 |
| p64-N1024 | torch-fused | 0.216 | 2.340 | 2.556 | 2.518–2.694 | 0.62 |
| p256-N1024 | gp (in-graph) | 0.712 | 0.538 | 1.255 | 1.241–1.452 | 4.85 |
| p256-N1024 | **gp-fused (device-native)** | 0.608 | 0.633 | **1.237** | 1.183–1.277 | 3.29 |
| p256-N1024 | numpy | 17.424 | 5.000 | 27.371 | 25.091–32.599 | 0.09 |
| p256-N1024 | torch | 2.954 | 2.522 | 5.476 | 5.308–5.657 | 3.52 |
| p256-N1024 | torch-fused | 0.226 | 2.433 | 2.659 | 2.566–2.797 | 0.63 |
| p1000-N1024 | gp (in-graph) | 1.497 | 0.767 | 2.269 | 2.245–2.428 | 5.06 |
| p1000-N1024 | **gp-fused (device-native)** | 0.608 | 0.810 | **1.400** | 1.370–1.460 | 3.44 |
| p1000-N1024 | numpy | 63.033 | 7.584 | 67.878 | 64.818–72.751 | 0.15 |
| p1000-N1024 | torch | 4.852 | 2.589 | 7.441 | 7.067–8.055 | 3.56 |
| p1000-N1024 | torch-fused | 0.241 | 2.453 | 2.693 | 2.636–2.755 | 0.61 |
| p4096-N1024 | gp (in-graph) | 5.398 | 0.654 | 6.023 | 5.993–6.228 | 4.99 |
| p4096-N1024 | **gp-fused (device-native)** | 0.617 | 0.646 | **1.248** | 1.218–1.327 | 3.45 |
| p4096-N1024 | numpy | 326.690 | 20.236 | 458.825 | 449.081–465.189 | 0.38 |
| p4096-N1024 | torch | 18.372 | 2.700 | 21.072 | 18.135–22.205 | 3.50 |
| p4096-N1024 | torch-fused | 0.234 | 2.482 | 2.716 | 2.649–2.814 | 0.61 |
| p256-N4096 | gp (in-graph) | 1.563 | 0.610 | 2.134 | 2.111–2.317 | 4.84 |
| p256-N4096 | **gp-fused (device-native)** | 0.597 | 0.612 | **1.207** | 1.168–1.236 | 3.28 |
| p256-N4096 | numpy | 60.935 | 5.122 | 67.518 | 65.998–74.364 | 0.13 |
| p256-N4096 | torch | 3.716 | 2.625 | 6.340 | 6.260–6.522 | 3.54 |
| p256-N4096 | torch-fused | 0.217 | 2.414 | 2.630 | 2.494–2.665 | 0.62 |

`cold` is the first `step` of the process: two IREE graph compiles for `gp` (variation + evaluation), ONE for `gp-fused` (variation only — its objective is a Triton JIT, also inside that figure), one dense compile for `torch`, the Triton JIT for `torch-fused`, nothing for `numpy`.
The `gp-fused` evaluation lane is FLAT (0.597–0.617 ms at every cell) while the in-graph lane tracks `pop × rows` (0.585 → 5.398 ms), because the kernel is flat in population and the row axis while the in-graph route reads `2 × width` child values per column in traced tensors.
The numpy split lane does not decompose its own total and is not meant to be read as a sum: its evaluation lane at pop 4096 spans 220.7–409.1 ms (host allocation noise) while the one-bracket total is 449.1–465.2 ms, so quote `total` for numpy and treat its segments as an indication only.

## 2. Speedups (the `gp-fused` total is the denominator)

| cell | torch / gp-fused | torch-fused / gp-fused | torch-operator / gp-fused | numpy / gp-fused | gp in-graph / gp-fused |
|---|---:|---:|---:|---:|---:|
| p64-N1024 | 4.14× | 2.20× | 2.10× | 8.7× | **0.95×** |
| p256-N1024 | 4.43× | 2.15× | 2.04× | 22.1× | 1.01× |
| p1000-N1024 | 5.32× | 1.92× | 1.85× | 48.5× | 1.62× |
| p4096-N1024 | **16.88×** | 2.18× | 2.16× | **367.6×** | 4.83× |
| p256-N4096 | 5.25× | 2.18× | 2.17× | 55.9× | 1.77× |

**The headline: the device-native fused loop beats the torch stack on its own FUSED route at every cell (2.20× / 2.15× / 1.92× / 2.18× / 2.18×) and the plain torch stack by 4.1×–16.9×, with no cell lost to either.**
The third column is the strictest form of the same statement: the torch stack's *operator layer alone* (2.438 / 2.522 / 2.589 / 2.700 / 2.625 ms) is 1.85–2.17× a WHOLE `gp-fused` generation, so even a torch stack with a free evaluator would lose this grid.
The fourth column is why the numpy route is a correctness backend and not a loop backend: a generation of the same stack on the host evaluator is 8.7–368× dearer.
The last column is the only remaining loss anywhere in the grid, and it is inside our own stack rather than against torch: at pop 64 the in-graph route's evaluation (0.585 ms) is cheaper than the fused route's (0.600 ms), so `gp` wins that cell by 5 % (1.106 vs 1.161 ms) — see §8(a).
From pop 256 up the fused evaluation is the cheaper one (0.608 vs 0.712 ms at p256, 0.608 vs 1.497 at p1000, 0.617 vs 5.398 at p4096), and the generation follows.

## 3. What one generation costs structurally

| side | graph runs per generation | host syncs | host readbacks | host↔device transfers | kernel launches |
|---|---:|---:|---:|---:|---:|
| `gp` (device) | 2 (`variation` incl. the elite splice, then `evaluation`) | 0 | 0 | 0 | 0 |
| `gp-fused` (device-native fused) | 1 (`variation` only) | 0 | 0 | 0 | 1 |
| `numpy` | 2 (the same two graphs, host tensors) | — | 1 fitness readback | 1 up + 1 down per call | 0 |
| `torch` | 1 engine call + operator kernels | the engine's own | the engine's readback | per engine call | 0 (1 on the `fused` route) |

The device counts are measured, not asserted (`count_runs` wraps `evoxc.run`, `count_fused_launches` reads the evaluator's own route counter), and every `gp` record of the JSON reads `generation_runs=2, variation_runs=1, evaluation_runs=1, syncs=0, host_readbacks=0, transfers=0` at all five cells.
Every `gp-fused` record reads `generation_runs=1, variation_runs=1, evaluation_runs=0, syncs=0, host_readbacks=0, transfers=0, fused_launches=1, host_route_launches=0, route_counts={"device": 74, "host": 0}, last_fallback=null`.
That is one fewer graph run per generation than the in-graph route, and it is not a trick: the objective is no longer a graph at all.
The three channel tensors cross into torch as DLPack views of the same memory, the legacy-triple conversion runs in torch on the device, the kernel reduces the loss in-kernel, and the negated/sanitized `(pop_size,)` buffer is adopted back as an `evoxc` device tensor that `workflow.step` hands straight to the variation graph (§6).
So nothing is uploaded, nothing is read back, and the driver's zero-sync property is inherited unchanged from the in-graph wiring.
The first call of every cell was served on the device route (`evaluator.route == "device"`, `fallback == null`, `available` and `device_available` both true), i.e. no `gp-fused` row of §1 contains a silent fallback to the host path.
The ONE remaining host touch is stated rather than glossed over, in the JSON's own words:

> `route_note`: device route: the channels never leave the device (zero-copy DLPack views) and the driver consumes the device fitness tensor, so a generation is one variation graph run with no channel transfer and no fitness readback; the remaining host touch is the evaluator's OUT_NODE guard (a single boolean readback) inside the evaluation lane.

That boolean — the `OUT_NODE` rejection the route cannot make without looking at the host — is the ~0.04 ms `fused_eval.py` documents for it, and it is included in the `evaluation` column of §1.

Cold start, and when it pays back: `gp-fused` compiles ONE IREE graph (the variation graph) plus the Triton kernels — 3.28–3.45 s — where `gp` compiles two — 4.84–5.06 s.
That is why the fused route is the cheaper start as well as the cheaper generation; against `gp` itself the only cell where it is not ahead is pop 64 (§8(a)).
`payback = (gp-fused cold − side cold) / (side warm − gp-fused warm)`, i.e. the generation after which both routes have spent the same wall time:

| cell | vs numpy | vs torch | vs torch-fused |
|---|---:|---:|---:|
| p64-N1024 | 369 gens | 0 (ahead cold AND warm) | 1 979 gens |
| p256-N1024 | 122 gens | 0 (ahead cold AND warm) | 1 876 gens |
| p1000-N1024 | 50 gens | 0 (ahead cold AND warm) | 2 191 gens |
| p4096-N1024 | 7 gens | 0 (ahead cold AND warm) | 1 931 gens |
| p256-N4096 | 48 gens | 0 (ahead cold AND warm) | 1 868 gens |

Against the default torch route there is nothing to pay back: `gp-fused` is cheaper cold (3.28–3.45 s vs 3.50–3.56 s) and 4.1–16.9× cheaper warm at every measured cell.
Against numpy the payback is 7–369 generations, and against the torch stack's own fused route it is ~1 870–2 190 generations, i.e. a real but long run — which is the honest price of a 2.2× warm margin behind a 2.7 s compile.

## 4. Convergence: does the device loop do the same work?

Best fitness of the 20-generation run (same seed, same target, same data), and the population median at generation 19.

| cell | side | g0 | g4 | g9 | g14 | g19 | median g19 |
|---|---|---:|---:|---:|---:|---:|---:|
| p64-N1024 | gp | -7.692 | -2.896 | -2.611 | -1.540 | -1.377 | -7.098 |
| p64-N1024 | gp-fused | -7.692 | -2.896 | -2.611 | -1.540 | -1.377 | -7.098 |
| p64-N1024 | numpy | -7.692 | -2.896 | -2.611 | -1.540 | -1.377 | -7.098 |
| p64-N1024 | torch | -11.703 | -4.798 | -2.364 | -1.489 | -1.356 | -3.164 |
| p64-N1024 | torch-fused | -11.703 | -4.798 | -2.364 | -1.489 | -1.356 | -3.164 |
| p256-N1024 | gp | -7.692 | -5.708 | -1.574 | -0.725 | -0.460 | -2.467 |
| p256-N1024 | gp-fused | -7.692 | -5.708 | -1.574 | -0.725 | -0.460 | -2.467 |
| p256-N1024 | numpy | -7.692 | -5.708 | -1.574 | -0.725 | -0.460 | -2.467 |
| p256-N1024 | torch | -6.517 | -5.399 | -0.476 | -0.476 | -0.443 | -2.452 |
| p256-N1024 | torch-fused | -6.517 | -5.399 | -0.476 | -0.476 | -0.476 | -2.465 |
| p1000-N1024 | gp | -5.989 | -2.556 | -0.476 | -0.476 | **-0.476** | -3.938 |
| p1000-N1024 | gp-fused | -5.989 | -2.556 | -0.476 | -0.476 | **-0.430** | -3.551 |
| p1000-N1024 | numpy | -5.989 | -2.556 | -0.476 | -0.476 | -0.476 | -3.551 |
| p1000-N1024 | torch | -6.517 | -2.630 | -1.416 | -0.484 | -0.484 | -3.429 |
| p1000-N1024 | torch-fused | -6.517 | -2.630 | -1.416 | -1.416 | -1.416 | -3.551 |
| p4096-N1024 | gp | -5.907 | -2.440 | -0.476 | -0.476 | -0.249 | -2.457 |
| p4096-N1024 | gp-fused | -5.907 | -2.440 | -0.476 | -0.476 | -0.249 | -2.440 |
| p4096-N1024 | numpy | -5.907 | -2.440 | -0.476 | -0.476 | -0.249 | -2.457 |
| p4096-N1024 | torch | -2.551 | -0.788 | -0.476 | -0.249 | -0.249 | -2.212 |
| p4096-N1024 | torch-fused | -2.551 | -0.788 | -0.476 | -0.405 | -0.249 | -2.293 |
| p256-N4096 | gp | -7.829 | -4.475 | -2.338 | -0.547 | -0.547 | -5.140 |
| p256-N4096 | gp-fused | -7.829 | -4.475 | -2.338 | -0.547 | -0.547 | -5.140 |
| p256-N4096 | numpy | -7.829 | -4.475 | -2.338 | -0.547 | -0.547 | -5.140 |
| p256-N4096 | torch | -6.536 | -3.469 | -0.466 | -0.466 | -0.466 | -3.678 |
| p256-N4096 | torch-fused | -6.536 | -2.405 | -0.466 | -0.466 | -0.273 | -3.654 |

The three `evogp.gp` sides are the same search: `gp` and `numpy` agree on the best-fitness series to ≤1.0e-06 at every checkpoint of all five cells (exactly equal at p1000 / p4096 / p256-N4096), so the in-graph device path makes the same selection decisions as the numpy reference, and `gp-fused` matches both to ≤1.0e-06 everywhere except the single flip below.
Every 20-generation trajectory in this record — all five generation sides, all five cells — is BIT-IDENTICAL to the previous record's, i.e. the operator-layer optimization of §8 changed the cost of a generation and not the search it performs.
`gp-fused` differs from `gp` in exactly one checkpoint of one cell: the p1000 generation-19 best (-0.430 vs -0.476, a tie flip inside one elite slot), plus last-bit float32 differences (≤1.0e-06) elsewhere.
That flip is a selection-tie consequence of the route's arithmetic, not a numeric defect: the fused kernel and the in-graph evaluation disagree by at most 4.2e-07 relative (§7), and two trees whose fitness is equal to within one float32 ulp can swap places in a tournament or an elite slot and then diverge.
The medians tell the same story with the same cause — they drift once such a tie has been broken (first divergence at generation 9 at p1000, 11 at p4096, 14 at p256-N4096, never at p64 / p256; max |Δmedian| 0.77 at p1000, 0.08–0.09 on the other two, against generation-19 medians of −3.94 / −2.46 / −5.14 on those cells).
The torch side is a different trajectory by construction, not by a defect: different RNG stream (torch vs `splitmix64`), different selection (truncation vs tournament-4) and different mutation wiring, so its generation-0 population is already a different draw and only the endpoint is comparable — and the two torch routes are not even each other's twin (dense vs fused evaluation break their ties differently: g19 -0.443 vs -0.476 at p256, -0.484 vs -1.416 at p1000, -0.466 vs -0.273 at p256-N4096), which is the same tie mechanism at a coarser margin.
At this 20-generation budget the two families trade the endpoint: torch reaches the better best at p64 (-1.356 vs -1.377), p1000 (-0.484 vs -0.476) and p256-N4096 (-0.466 vs -0.547) on its default route, `evogp.gp` at p256 (-0.460 vs -0.443) and at p4096 (-0.249, equal); that is operator pressure at a short horizon, not evaluation, and `evogp.gp`'s own CPU evidence reaches -0.107 at 100 generations with tournament 4 (`CONTEXT.md`).

## 5. Evaluation cost vs tree growth (and what the fused route does differently)

The timed lane runs on an evolved population, so its trees are longer than generation 0. Re-evaluating a *freshly generated* population shows which side of the cost model that matters on.

| cell | gp fresh | gp evolved | gp-fused fresh | gp-fused evolved | numpy fresh | numpy evolved |
|---|---:|---:|---:|---:|---:|---:|
| p64-N1024 | 0.562 | 0.585 | 0.497 | 0.600 | 2.943 | 5.796 |
| p256-N1024 | 0.725 | 0.712 | 0.513 | 0.608 | 11.476 | 17.424 |
| p1000-N1024 | 1.494 | 1.497 | 0.499 | 0.608 | 37.088 | 63.033 |
| p4096-N1024 | 5.484 | 5.398 | 0.518 | 0.617 | 224.349 | 326.690 |
| p256-N4096 | 1.565 | 1.563 | 0.496 | 0.597 | 32.543 | 60.935 |

The JSON does carry the fresh/evolved split for `gp-fused` (`fresh_evaluation` is measured for every `evogp.gp` side), and it is the one lane on which the fused route is not perfectly flat: it costs 1.18–1.22× more on an evolved population than on a fresh one, because the kernel walks each tree's own nodes, so a deeper population is more work.
The in-graph route is flat in tree length by construction (its cost is the `max_nodes`-static child-read chain — 0.96–1.02× fresh → evolved) and the host numpy evaluator grows 1.46–1.97×, so a numpy evaluation of a fresh population is 31–49 % cheaper than its §1 lane.
Both device routes stay 5–630× cheaper than numpy on either population; what the fresh/evolved split decides is only how much of the fused route's advantage survives an evolved population (it does: 0.600 ms at p64 evolved against the in-graph 0.585 fresh).

## 6. How the device-native fused route works

The route exists because the Triton kernel is the evaluator whose cost is flat in population and rows — the one shape the in-graph route cannot match — and the only thing standing between a device-resident loop and it was the round trip: `evoxc`'s documented interop is host-only (`from_numpy` is a zero-copy host wrap; a device tensor is read back through an explicit `to(Device("cpu", 0))`), while the kernel needs torch device tensors.
`_device_interop.py` is that bridge, and it is deliberately LOCAL and TEMPORARY — it should be deleted the day `evoxc` grows a first-class device-payload DLPack export/import.

How a call goes (`fused_eval.FusedDeviceEvaluator`):

| step | mechanism |
|---|---|
| channels in | three `evoxc` DEVICE tensors (metadata-only dispatch: `is_device_tensor`, one shared device, 2-D population — anything else takes the host path) |
| export | `payload.device_array._device.create_dlpack_capsule(payload.device_array._buffer_view, code, device_id)` → `torch.utils.dlpack.from_dlpack(capsule)` — a SHARING view, no bytes move (proven by mutating the torch view and seeing it through the `evoxc` tensor) |
| conversion | `_legacy_triple_device`: `to_legacy_triple`'s `where` chain rewritten in torch on the device (function node ⇒ function id + `FUNC_TYPE` marker, leaf ⇒ payload + opcode, padding ⇒ zeros), bitwise-equal to the host bridge |
| kernel | `evogp.tree.fused_engine.sr_fitness_fused` on the legacy triple; the SR loss is reduced in-kernel, so the only device output is `(pop_size,)` |
| post | negate + non-finite → `-inf` in torch (the fitness contract), then `hal.from_dlpack_capsule(tensor.__dlpack__())` → an `IreeDevicePayload` → an `evoxc.Tensor` on the same device |
| consume | `workflow.step` hands that tensor straight to the variation graph — no readback, so the generation stays ONE graph run |

DLPack type codes are single-sourced (`DEVICE_TYPE_CODES`: `rocm`/`hip` → 10, `cuda` → 2) and every other kind declines, `cpu` included, because the host path already covers it.
Validation is hard rather than silent: a non-`evoxc` or host payload, an unsupported dtype, a device mismatch, a shape/dtype disagreement between the payload metadata and the exported tensor, or a non-contiguous import without `copy=True` raises `InteropUnavailable` instead of handing back a reinterpreted buffer.
Synchronization is an assumption with a tripwire rather than a theorem: iree's HIP driver and torch share the process' default stream, so pointer sharing needs no fence — validated by the chained-generation parity check in `tests/test_gp_stack_device_fused.py` (five `workflow.step` generations whose channels must match the in-graph route bit for bit), which is what breaks if that ever stops being true.
`x`/`y` are uploaded once at construction, the `FUNC_TYPE` table is memoized per device, and `TRITON_CACHE_DIR` is defaulted to `$TMPDIR/evogp-triton-cache` when the caller has not set it (`/tmp/evogp-triton-cache` in this record) because this box's `$HOME` is read-only.

Fallback, never a wrong number: when the interop or the kernel refuses (a cpu device, an unsupported dtype, a torch device mismatch, a population outside the kernel's scratch envelope) the call RETRIES on the host route and returns its numpy vector.
The route actually taken is visible in `.last_route` (`"device"` / `"host"`), counted in `.route_counts`, and the reason in `.last_fallback`; the harness records all three per cell (`evaluator` in the JSON), which is how "no silent fallback" in §3 is a measurement and not a claim.
The one thing not caught is the host route's own `FusedUnavailable`: when that raises, no fused route exists on this box at all, and `.available()` already said so.

Upstream plan: the shim borrows two private surfaces (`IreeDevicePayload.device_array`, `HalDevice.create_dlpack_capsule` / `from_dlpack_capsule`) that `evoxc` does not export yet.
When it does, `_device_interop.export_to_torch` / `import_from_torch` become one-line `evoxc` calls and the module disappears; until then it stays a validated, declining, single-purpose bridge and not a second code path.

## 7. Fused (Triton) vs in-graph: the head-to-head, and where the crossover went

The `fused` side times the two evaluators on the same channels, the same `x`/`y`, medians of 15, transfers included: `in_graph` is `GraphEvaluator.evaluate` on device channels (no readback — the route `gp` ships), `transfer` is those three tensors downloaded to numpy (idle, and queued right behind an in-graph run), `fused_host` is the kernel on channels already in host memory (host decode included), and `fused_device` is the round trip + kernel composed in ONE repetition.

| cell | in-graph | transfer (idle) | transfer (queued) | fused from host | fused from device | fused from device (queued) | in-graph / fused (queued) |
|---|---:|---:|---:|---:|---:|---:|---:|
| p64-N1024 | 0.616 | 0.323 | 0.305 | 0.306 | 0.824 | 0.611 | 1.01× |
| p256-N1024 | 0.713 | 0.310 | 0.418 | 0.415 | 1.059 | 0.833 | **0.86×** |
| p1000-N1024 | 1.491 | 0.429 | 0.528 | 0.701 | 1.381 | 1.229 | 1.21× |
| p4096-N1024 | 5.489 | 0.922 | 1.152 | 1.889 | 3.015 | 3.041 | 1.80× |
| p256-N4096 | 1.537 | 0.324 | 0.377 | 0.379 | 0.991 | 0.755 | 2.03× |

**The crossover moved out of this grid.** With the fifth cell added, the in-graph evaluation costs 5.489 ms at pop 4096 while the kernel stays flat, so the round-trip lane now wins four of five cells (1.01× / 1.21× / 1.80× / 2.03×) and loses only pop 256 (0.86×); the earlier record's crossover — in-graph ahead at pop ≤ 256, fused ahead at pop 1000 and on the wide-row cell — is now a single losing cell and a wash.
Read the columns as bounds, not as routes: `fused from device (queued)` is the SUM of two separately measured medians (the transfer behind a device run plus the host-channels kernel), so it excludes the per-call compose overhead; it reads 0.15–0.24 ms BELOW the composed lane at p64 / p256 / p1000 / p256-N4096 and 0.03 ms above it at p4096, i.e. it is an optimistic bound everywhere except where the transfer lane itself stops being additive.
The composed lane (`in-graph / fused from device`) is the honest round trip and wins three of five cells (1.08× at p1000, 1.82× at p4096, 1.55× on the wide-row cell) while losing p64 (0.75×) and p256 (0.67×).
What decides it is the toll: 0.32–0.92 ms idle / 0.31–1.15 ms queued, i.e. 17–52 % of the in-graph call, which the round trip has to recover before the kernel's flatness pays anything.

**This section is no longer what the device loop pays**, and that is the point of §6: `FusedDeviceEvaluator` does not download anything, so there is no toll to recover.
Measured inside the loop (the `evaluation` lanes of §1) it is 0.597–0.617 ms at every cell against the in-graph 0.585 / 0.712 / 1.497 / 5.398 / 1.563 ms — 0.98× / 1.17× / 2.46× / 8.75× / 2.62×, the first of which is the pop-64 loss of §8(a) — which is why §2's headline is written against the fused loop and not against these lanes.
The head-to-head keeps measuring the round-trip lane because it is the lane every earlier round measured, so the two records stay comparable; the device-native lane is the one that shipped.
Parity of the fused lanes against in-graph: `max_rel` 1.166e-07 / 1.673e-07 / 3.362e-07 / 4.158e-07 / 2.214e-07 depending on the cell, `max_rel_scaled` ≤9.1e-08, and ZERO finiteness mismatches on all five cells — about two and a half orders of magnitude inside the 1e-4 acceptance box.

## 8. Where we lose, and why

- **(a) pop 64: the in-graph route is 5 % faster than the fused one (1.106 vs 1.161 ms), and that is the only cell anywhere in §2 that `evogp.gp` loses.** The evaluation lanes are 0.585 (in-graph) against 0.600 (fused), the variation lanes 0.525 against 0.578, so the fused route loses on both segments at that size: a `variation`-graph run is fixed graph-run overhead (~0.53 ms floor), and at 64 trees the kernel's ~0.22 ms launch plus the conversion glue cannot undercut a graph whose whole population is 64 × 32 cells.
  It also cannot be paid back: `gp-fused` starts 1.5 s cheaper cold, so `gp` only overtakes it after ~27 000 generations of the 0.055 ms warm margin.
- **(b) The residual fused evaluation lane: a flat ~0.60 ms, of which the kernel is the smaller part.** The torch stack's own fused lane — the same kernel with the populations' parameters already resident as torch tensors — is 0.216–0.241 ms across the grid, so ~0.38 ms of our 0.597–0.617 ms is the crossing (three DLPack capsule exports, the legacy-triple `where` chain in torch, the negate/sanitize, the import back) plus the one mandatory boolean readback (~0.04 ms).
  That is a constant, hence a win wherever the in-graph route grows, and a loss at pop 64 where the whole generation is 1.1 ms; the way to shrink it is an `evoxc` device export (§6's upstream plan), not a different kernel.
- **(c) The operator layer's floor (~0.53–0.81 ms per generation).** The variation lane is 0.525 / 0.538 / 0.767 / 0.654 / 0.610 ms at p64 / p256 / p1000 / p4096 / p256-N4096 on the in-graph side (0.578 / 0.633 / 0.810 / 0.646 / 0.612 on the fused side), i.e. roughly half of a pop-64 generation and 50–58 % of the fused ones. It is a graph-run floor — a fixed number of traced ops plus the two population-scaled costs (the generation placement and the elite ranking) — and it is now the largest single item in every `gp-fused` cell.
- **The operator-layer optimization that produced this record (commit `3b7df8f`; the variation column of §1, before → after)**: `encoding._generator_fn` places the padded growth grid with ONE axis-1 `argsort` of the `[pop_size, padded_total]` destination map plus a `take_per_row` gather per channel, replacing a flat `evoxc.scatter` of the inverse map that lowered to a sequential stablehlo update loop (~45 ns per index, 2.77 ms of the generation graph's 2.35–2.77 ms at pop 4096, against ~0.16 ms for the argsort); `variation._elite_order` takes the top `k` rows with `k` repeated `argmax` passes over a running mask instead of one population-sized stable `argsort` (1.31–1.46 ms at pop 4096), keeping the stable-sort tie rule exactly (and as the fallback above 50 elites).

  | pop | gp variation before | gp after | gp-fused before | gp-fused after |
  |---|---:|---:|---:|---:|
  | 64 | 0.538 | 0.525 | 0.558 | 0.578 |
  | 256 | 0.672 | 0.538 | 0.690 | 0.633 |
  | 1000 | 1.273 | 0.767 | 1.226 | 0.810 |
  | 4096 | 4.970 | 0.654 | 4.511 | 0.646 |
  | 256×4096 | 0.678 | 0.610 | 0.718 | 0.612 |

  Both changes are bitwise-equivalent — the parity capture that landed with that commit found every recorded array identical (2 556 of 2 556, pre/post, over 3 configs × pops {5, 17, 64, 257, 1000} × 3 seeds on numpy plus 168 numpy and 140 iree/rocm elite-order cells), and §4's trajectories are bit-identical to the previous record's — so the 7.6× variation-lane win at pop 4096 is free.
  It moved the fused route's totals from 1.209 / 1.286 / 1.842 / 5.130 / 1.331 ms (the previous record's own JSON) to 1.161 / 1.237 / 1.400 / 1.248 / 1.207 ms.
- **The numpy route is not competitive for a loop** (8.7× / 22.1× / 48.5× / 368× / 55.9× behind `gp-fused`): its evaluation is the host evaluator, which is 1.46–1.97× dearer again on evolved populations and swings 220.7–409.1 ms at pop 4096. It is the correctness oracle, not a loop backend.
- **Cold start**: `gp-fused` 3.28–3.45 s, `gp` 4.84–5.06 s, `torch` 3.50–3.56 s, `torch-fused` 0.61–0.63 s, `numpy` 0.08–0.38 s. Irrelevant for a long run (§3's payback rows: 7–369 generations against numpy, 1 868–2 191 against torch-fused, nothing against torch) but it dominates any short benchmark, and it is the one column where the fused loop does not beat `torch-fused`.
- **Run-to-run spread decides the close cells**: measured against the previous record's medians on the sides whose code did not change in between, the torch total moved -2.6 % … +1.9 %, `torch-fused` -1.0 % … +8.1 % (the wide-row cell is the outlier), and numpy -17.8 % … +3.7 %; the device lanes were re-measured after the operator change, so their only comparable column is the trajectory (§4), which is bit-identical. A cell decided by less than ~10 % is called parity here.
- **Variance**: `gp-fused` totals span 1.136–1.193 (p64), 1.183–1.277 (p256), 1.370–1.460 (p1000), 1.218–1.327 (p4096) and 1.168–1.236 ms (p256-N4096) — a few percent, no bimodality; the in-graph side is wider where the evaluation scales (5.993–6.228 at p4096) and the torch side widest (18.135–22.205 at p4096) because its engine call allocates per generation.

## 9. Reproduce

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --cells 64,256,1000,4096,256x4096 \
  --json src/evogp/gp/evidence/device_loop.json
```

One cell of one side, in a single process (what the grid above does per row of the tables):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --side gp-fused --pop 256 --rows 4096 --warmup 5 --reps 15
```

`--side` is one of `gp`, `gp-fused`, `numpy`, `torch`, `torch-fused`, `fused`; the default (`--cells`) grid runs all six sides per cell, each in its own process (the parent re-executes itself, so no cell's compile, allocator or RNG state can leak into another), and merges the children's JSON into one file (30 records, 378 s of child wall in this campaign).
`--side fused` is the §7 head-to-head, `--sides` narrows the grid, `--no-guard` skips the `rocm-smi` idle probe (`--guard-wait` is its timeout), and `--json <path>` writes the record — the parent merges its children's into that one file.
A `gp-fused` run additionally needs a writable `TRITON_CACHE_DIR` (the harness defaults it) and torch+triton importable; `_device_interop.interop_available` is the probe that decides whether the device route exists at all.

Environment of record: python 3.12.3, torch 2.13.0+rocm7.1 (HIP 7.1.52802), triton 3.7.1, numpy 2.5.4, iree-base-compiler / iree-base-runtime 3.11.0, evoxc `9f66fec` (checkout `/root/evoxc`), `evogp` `d00d95b` (the tree that produced these numbers), GPU `AMD Instinct MI300X VF`.
