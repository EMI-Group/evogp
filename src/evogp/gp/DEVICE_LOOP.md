# The device-resident `evogp.gp` generation loop — measurement report
> Measured on the pre-2.0.0a2 tree, where `evogp.tree` still existed: the torch comparison code (`Forest`-based lanes, `evogp.algorithm`/`evogp.problem`/`evogp.pipeline` sides) was removed in 2.0.0a2 — recover it at commit 6720723. The numbers below are unchanged and are not reproducible from this tree.

Report of record for `python -m evogp.gp.measure_generation` (harness `measure_generation.py`, raw numbers in `evidence/device_loop.json`).
See also `POPSCALE.md` — the population-scaling report of record for the SAME driver: eight populations 64 → 8 192 at one fixed shape, the shared-population per-call column (`measure_popscale.py`) and the generation column (`measure_generation.py --cells popscale`).
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
| p64-N1024 | gp-fused (device-native) | 0.549 | 0.582 | 1.149 | 1.093–1.221 | 3.32 |
| p64-N1024 | numpy | 5.796 | 4.335 | 10.105 | 9.705–11.320 | 0.08 |
| p64-N1024 | torch | 2.364 | 2.438 | 4.802 | 4.727–4.913 | 3.53 |
| p64-N1024 | torch-fused | 0.216 | 2.340 | 2.556 | 2.518–2.694 | 0.62 |
| p256-N1024 | gp (in-graph) | 0.712 | 0.538 | 1.255 | 1.241–1.452 | 4.85 |
| p256-N1024 | **gp-fused (device-native)** | 0.547 | 0.638 | **1.189** | 1.150–1.253 | 3.41 |
| p256-N1024 | numpy | 17.424 | 5.000 | 27.371 | 25.091–32.599 | 0.09 |
| p256-N1024 | torch | 2.954 | 2.522 | 5.476 | 5.308–5.657 | 3.52 |
| p256-N1024 | torch-fused | 0.226 | 2.433 | 2.659 | 2.566–2.797 | 0.63 |
| p1000-N1024 | gp (in-graph) | 1.497 | 0.767 | 2.269 | 2.245–2.428 | 5.06 |
| p1000-N1024 | **gp-fused (device-native)** | 0.550 | 0.810 | **1.372** | 1.332–1.631 | 3.58 |
| p1000-N1024 | numpy | 63.033 | 7.584 | 67.878 | 64.818–72.751 | 0.15 |
| p1000-N1024 | torch | 4.852 | 2.589 | 7.441 | 7.067–8.055 | 3.56 |
| p1000-N1024 | torch-fused | 0.241 | 2.453 | 2.693 | 2.636–2.755 | 0.61 |
| p4096-N1024 | gp (in-graph) | 5.398 | 0.654 | 6.023 | 5.993–6.228 | 4.99 |
| p4096-N1024 | **gp-fused (device-native)** | 0.566 | 0.651 | **1.231** | 1.197–1.405 | 3.61 |
| p4096-N1024 | numpy | 326.690 | 20.236 | 458.825 | 449.081–465.189 | 0.38 |
| p4096-N1024 | torch | 18.372 | 2.700 | 21.072 | 18.135–22.205 | 3.50 |
| p4096-N1024 | torch-fused | 0.234 | 2.482 | 2.716 | 2.649–2.814 | 0.61 |
| p256-N4096 | gp (in-graph) | 1.563 | 0.610 | 2.134 | 2.111–2.317 | 4.84 |
| p256-N4096 | **gp-fused (device-native)** | 0.553 | 0.627 | **1.179** | 1.141–1.207 | 3.42 |
| p256-N4096 | numpy | 60.935 | 5.122 | 67.518 | 65.998–74.364 | 0.13 |
| p256-N4096 | torch | 3.716 | 2.625 | 6.340 | 6.260–6.522 | 3.54 |
| p256-N4096 | torch-fused | 0.217 | 2.414 | 2.630 | 2.494–2.665 | 0.62 |

`cold` is the first `step` of the process: two IREE graph compiles for `gp` (variation + evaluation), ONE for `gp-fused` (variation only — its objective is a Triton JIT, also inside that figure), one dense compile for `torch`, the Triton JIT for `torch-fused`, nothing for `numpy`; `gp-fused` cold is 3.32–3.61 s because that one variation graph grows with the population.
The `gp-fused` evaluation lane is FLAT (0.547–0.566 ms at every cell) while the in-graph lane tracks `pop × rows` (0.585 → 5.398 ms), because the kernel is flat in population and the row axis while the in-graph route reads `2 × width` child values per column in traced tensors.
The numpy split lane does not decompose its own total and is not meant to be read as a sum: its evaluation lane at pop 4096 spans 220.7–409.1 ms (host allocation noise) while the one-bracket total is 449.1–465.2 ms, so quote `total` for numpy and treat its segments as an indication only.

## 2. Speedups (the `gp-fused` total is the denominator)

| cell | torch / gp-fused | torch-fused / gp-fused | torch-operator / gp-fused | numpy / gp-fused | gp in-graph / gp-fused |
|---|---:|---:|---:|---:|---:|
| p64-N1024 | 4.18× | 2.22× | 2.12× | 8.8× | **0.96×** |
| p256-N1024 | 4.61× | 2.24× | 2.12× | 23.0× | 1.06× |
| p1000-N1024 | 5.42× | 1.96× | 1.89× | 49.5× | 1.65× |
| p4096-N1024 | **17.12×** | 2.21× | 2.19× | **372.7×** | 4.89× |
| p256-N4096 | 5.38× | 2.23× | 2.23× | 57.3× | 1.81× |

**The headline: the device-native fused loop beats the torch stack on its own FUSED route at every cell (2.22× / 2.24× / 1.96× / 2.21× / 2.23×) and the plain torch stack by 4.2×–17.1×, with no cell lost to either.**
The third column is the strictest form of the same statement: the torch stack's *operator layer alone* (2.438 / 2.522 / 2.589 / 2.700 / 2.625 ms) is 1.89–2.23× a WHOLE `gp-fused` generation, so even a torch stack with a free evaluator would lose this grid.
The fourth column is why the numpy route is a correctness backend and not a loop backend: a generation of the same stack on the host evaluator is 8.8–373× dearer.
The last column is the only remaining loss anywhere in the grid, and it is inside our own stack rather than against torch: `gp` wins pop 64 by 4 % (1.106 vs 1.149 ms) — see §8(a) — and loses the other four cells by 6–389 %.
The fused evaluation is the cheaper one at EVERY cell (0.549 vs 0.585 ms at p64, 0.547 vs 0.712 at p256, 0.550 vs 1.497 at p1000, 0.566 vs 5.398 at p4096, 0.553 vs 1.563 on the wide-row cell), so the in-graph route only ever wins a generation through its variation lane.

## 3. What one generation costs structurally

| side | graph runs per generation | host syncs | host readbacks | host↔device transfers | kernel launches |
|---|---:|---:|---:|---:|---:|
| `gp` (device) | 2 (`variation` incl. the elite splice, then `evaluation`) | 0 | 0 | 0 | 0 |
| `gp-fused` (device-native fused) | 1 (`variation` only) | 1 `torch.cuda.synchronize()` fence, no readback (§8(b)) | 0 | 0 | 1 |
| `numpy` | 2 (the same two graphs, host tensors) | — | 1 fitness readback | 1 up + 1 down per call | 0 |
| `torch` | 1 engine call + operator kernels | the engine's own | the engine's readback | per engine call | 0 (1 on the `fused` route) |

The device counts are measured, not asserted (`count_runs` wraps `evoxc.run`, `count_fused_launches` reads the evaluator's own route counter), and every `gp` record of the JSON reads `generation_runs=2, variation_runs=1, evaluation_runs=1, syncs=0, host_readbacks=0, transfers=0` at all five cells.
Every `gp-fused` record reads `generation_runs=1, variation_runs=1, evaluation_runs=0, syncs=0, host_readbacks=0, transfers=0, fused_launches=1, host_route_launches=0, route_counts={"device": 74, "host": 0}, last_fallback=null`.
That is one fewer graph run per generation than the in-graph route, and it is not a trick: the objective is no longer a graph at all.
The three channel tensors cross into torch as DLPack views of the same memory, the legacy-triple conversion runs in torch on the device, the kernel reduces the loss in-kernel and writes the `(pop_size,)` result straight into the exported view of the evaluator's persistent `evoxc` fitness buffer (`tree.fused_engine.sr_fitness_fused(..., out=view)`), and `workflow.step` hands that SAME `evoxc` tensor to the variation graph (§6).
So nothing is uploaded, nothing is read back, there is no per-call copy and no import step, and the driver's zero-sync property is inherited unchanged from the in-graph wiring.
The first call of every cell was served on the device route (`evaluator.route == "device"`, `fallback == null`, `available` and `device_available` both true), i.e. no `gp-fused` row of §1 contains a silent fallback to the host path.
`syncs` counts the driver's own host round trips (it stays 0); the ONE cross-runtime fence per device call is a `torch.cuda.synchronize()`, which is not a readback and is quantified in §8(b).
The ONE remaining host touch is stated rather than glossed over, in the JSON's own words:

> `route_note`: device route: the channels never leave the device (zero-copy DLPack views) and the driver consumes the device fitness tensor, so a generation is one variation graph run with no channel transfer and no fitness readback; the remaining host touch is the evaluator's OUT_NODE guard (a single boolean readback) inside the evaluation lane.

That boolean — the `OUT_NODE` rejection the route cannot make without looking at the host — is the ~0.04 ms `fused_eval.py` documents for it, and it is included in the `evaluation` column of §1.

Cold start, and when it pays back: `gp-fused` compiles ONE IREE graph (the variation graph) plus the Triton kernels — 3.32–3.61 s — where `gp` compiles two — 4.84–5.06 s.
That is why the fused route is the cheaper start as well as the cheaper generation; against `gp` itself the only cell where it is not ahead is pop 64 (§8(a)).
`payback = (gp-fused cold − side cold) / (side warm − gp-fused warm)`, i.e. the generation after which both routes have spent the same wall time:

| cell | vs numpy | vs torch | vs torch-fused |
|---|---:|---:|---:|
| p64-N1024 | 362 gens | 0 (ahead cold AND warm) | 1 921 gens |
| p256-N1024 | 127 gens | 0 (ahead cold AND warm) | 1 890 gens |
| p1000-N1024 | 52 gens | 3 gens | 2 246 gens |
| p4096-N1024 | 7 gens | 5 gens | 2 018 gens |
| p256-N4096 | 50 gens | 0 (ahead cold AND warm) | 1 927 gens |

Against the default torch route there is almost nothing to pay back: `gp-fused` is cheaper cold at three cells (3.32 / 3.41 / 3.42 s against 3.53 / 3.52 / 3.54 s), at parity at p1000 (3.58 vs 3.56 s) and 0.11 s dearer at p4096, and 4.2–17.1× cheaper warm at every measured cell — hence 0 generations at three cells and 3 / 5 at the other two.
Against numpy the payback is 7–362 generations, and against the torch stack's own fused route it is ~1 890–2 250 generations, i.e. a real but long run — which is the honest price of a 2.2× warm margin behind a 2.7 s compile.

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
| p64-N1024 | 0.562 | 0.585 | 0.483 | 0.549 | 2.943 | 5.796 |
| p256-N1024 | 0.725 | 0.712 | 0.448 | 0.547 | 11.476 | 17.424 |
| p1000-N1024 | 1.494 | 1.497 | 0.458 | 0.550 | 37.088 | 63.033 |
| p4096-N1024 | 5.484 | 5.398 | 0.463 | 0.566 | 224.349 | 326.690 |
| p256-N4096 | 1.565 | 1.563 | 0.453 | 0.553 | 32.543 | 60.935 |

The JSON does carry the fresh/evolved split for `gp-fused` (`fresh_evaluation` is measured for every `evogp.gp` side), and it is the one lane on which the fused route is not perfectly flat: it costs 1.14–1.22× more on an evolved population than on a fresh one, because the kernel walks each tree's own nodes, so a deeper population is more work.
The in-graph route is flat in tree length by construction (its cost is the `max_nodes`-static child-read chain — 0.96–1.02× fresh → evolved) and the host numpy evaluator grows 1.46–1.97×, so a numpy evaluation of a fresh population is 31–49 % cheaper than its §1 lane.
Both device routes stay 6–578× cheaper than numpy on either population, and the fused route is cheaper than the in-graph lane on BOTH populations of every cell (0.549 ms at p64 evolved against the in-graph 0.585 evolved).

## 6. How the device-native fused route works

The route exists because the Triton kernel is the evaluator whose cost is flat in population and rows — the one shape the in-graph route cannot match — and the only thing standing between a device-resident loop and it was the round trip: `evoxc`'s documented interop is host-only (`from_numpy` is a zero-copy host wrap; a device tensor is read back through an explicit `to(Device("cpu", 0))`), while the kernel needs torch device tensors.
`evoxc` has since grown a first-class device-payload export (`Tensor.__dlpack__` / `__dlpack_device__`), so the bridge is the standard DLPack protocol itself: `fused_eval.export_to_torch` is `torch.from_dlpack` plus the validation, and there is no local interop module at all — no payload handle is touched, and no shim exists to be removed later.

How a call goes (`fused_eval.FusedDeviceEvaluator`):

| step | mechanism |
|---|---|
| channels in | three `evoxc` DEVICE tensors (metadata-only dispatch: `is_device_tensor`, one shared device, 2-D population — anything else takes the host path) |
| export | `torch.from_dlpack(evoxc_tensor)` after checking `__dlpack_device__()` against the tensor's own device — a SHARING view, no bytes move (proven by mutating the torch view and seeing it through the `evoxc` tensor) |
| conversion | `_legacy_triple_device`: `to_legacy_triple`'s `where` chain rewritten in torch on the device (function node ⇒ function id + `FUNC_TYPE` marker, leaf ⇒ payload + opcode, padding ⇒ zeros), bitwise-equal to the host bridge |
| buffer | the evaluator's persistent `evoxc` fitness tensor (`evoxc.zeros((pop,), float32).to(device)`, exported once per `(pop, device)`), whose torch view is passed as `out=` |
| kernel | `evogp.fused.engine.sr_fitness_fused(..., out=view)` on the legacy triple: the SR loss is reduced in-kernel and the `(pop_size,)` result is SUMMED AND DIVIDED straight into that exported buffer (`torch.sum(partials, dim=1, out=...)`, then `div_(rows)`), so the kernel's own `(pop, n_rb)` chart is the only temporary |
| post | negate + non-finite → `-inf` IN PLACE on the same view (the fitness contract), plus ONE `torch.cuda.synchronize()` |
| consume | `workflow.step` hands that same `evoxc` tensor straight to the variation graph — no copy, no readback, no import step, so the generation stays ONE graph run |

DLPack type codes are single-sourced (`DEVICE_TYPE_CODES`: `rocm`/`hip` → 10, `cuda` → 2) and every other kind declines, `cpu` included, because the host path already covers it.
Validation is hard rather than silent: a non-`evoxc` or host payload, an unsupported dtype, a device mismatch, a shape/dtype/contiguity disagreement between the `evoxc` tensor and what `torch.from_dlpack` handed back, or a device-kind contradiction raises `InteropUnavailable` instead of handing back a reinterpreted buffer.
Synchronization is an assumption with a tripwire rather than a theorem: iree's HIP driver and torch share the process' default stream, but the WRITE direction (torch → iree) takes one explicit `torch.cuda.synchronize()` per device call because iree's HAL adds no fence of its own; the tripwire is the chained-generation parity check in `tests/test_gp_stack_device_fused.py` (five `workflow.step` generations whose channels must match the in-graph route bit for bit), which is what breaks if the assumption ever stops holding.
`x`/`y` are uploaded once at construction, the `FUNC_TYPE` table is memoized per device, and `TRITON_CACHE_DIR` is defaulted to `$TMPDIR/evogp-triton-cache` when the caller has not set it (`/tmp/evogp-triton-cache` in this record) because this box's `$HOME` is read-only.
`out=` validates hard before the launch (`tree/fused_engine._out_destination`: a torch tensor of exactly the kernel's shape, float32, contiguous, `requires_grad=False`, on the same device) and the three channel views plus the fitness view are the ONLY per-call host objects; `out=None` keeps the allocating signature every other caller uses.

Fallback, never a wrong number: when the interop or the kernel refuses (a cpu device, an unsupported dtype, a torch device mismatch, a population outside the kernel's scratch envelope) the call RETRIES on the host route and returns its numpy vector.
The route actually taken is visible in `.last_route` (`"device"` / `"host"`), counted in `.route_counts`, and the reason in `.last_fallback`; the harness records all three per cell (`evaluator` in the JSON), which is how "no silent fallback" in §3 is a measurement and not a claim.
The one thing not caught is the host route's own `FusedUnavailable`: when that raises, no fused route exists on this box at all, and `.available()` already said so.

The export direction is a one-line `evoxc` call; the import direction has no native equivalent (`external_call` refuses a device-resident return and evoxc's `from_dlpack` is host-only), which is why the return leg is a persistent `evoxc` buffer the kernel WRITES INTO (`out=`) rather than a per-call import — the buffer is the destination, not a staging area, so neither a `.copy_()` nor a DLPack import appears in the call.
`register_external_kernel(..., device_resident=True)` imports a kernel-returned `torch.Tensor` zero-copy instead, but it costs one graph run around the call (+0.20 ms measured, 0.56 vs 0.36 ms) and forces the sync inside the kernel, so the buffer route stays the shipped one.

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
Measured inside the loop (the `evaluation` lanes of §1) it is 0.547–0.566 ms at every cell against the in-graph 0.585 / 0.712 / 1.497 / 5.398 / 1.563 ms — 1.07× / 1.30× / 2.72× / 9.54× / 2.83×, i.e. the fused lane is cheaper at EVERY cell of this grid, pop 64 included — which is why §2's headline is written against the fused loop and not against these lanes.
The head-to-head keeps measuring the round-trip lane because it is the lane every earlier round measured, so the two records stay comparable; the device-native lane is the one that shipped.
Parity of the fused lanes against in-graph: `max_rel` 1.166e-07 / 1.673e-07 / 3.362e-07 / 4.158e-07 / 2.214e-07 depending on the cell, `max_rel_scaled` ≤9.1e-08, and ZERO finiteness mismatches on all five cells — about two and a half orders of magnitude inside the 1e-4 acceptance box.

## 8. Where we lose, and why

- **(a) pop 64: the in-graph route is 4 % faster than the fused one (1.106 vs 1.149 ms), and that is the only cell anywhere in §2 that `evogp.gp` loses.** It no longer loses on both segments: the evaluation lanes are 0.585 (in-graph) against 0.549 (fused), so the fused objective is now the cheaper evaluator even at 64 trees; what keeps the cell is the variation lane (0.525 against 0.582), i.e. run-to-run noise on one fixed graph-run floor rather than a property of the objective.
  It also cannot be paid back: `gp-fused` starts 1.56 s cheaper cold, so `gp` only overtakes it after ~36 000 generations of the 0.043 ms warm margin.
- **(b) The residual fused evaluation lane: a flat ~0.55 ms, of which the kernel is the smaller part.** The torch stack's own fused lane — the same kernel with the populations' parameters already resident as torch tensors — is 0.216–0.241 ms across the grid, so ~0.31–0.33 ms of our 0.547–0.566 ms is the crossing (three DLPack capsule exports, the legacy-triple `where` chain in torch, the negate/sanitize in place) plus the one mandatory boolean readback (~0.04 ms) and the fence below.
  **The mandatory fence is 0.030–0.035 ms of it** (+6.3 % of the evaluation lane, +2.5–3.0 % of a whole generation at pop 256): measured by re-running this cell with `torch.cuda.synchronize` patched out at the harness level, the shipped call reads 0.5514 ms against 0.5170 ms without it. It stays: iree's HAL adds no fence of its own, so the ordering that makes the variation graph read the fitness the kernel wrote is exactly this synchronize — a correctness requirement, not a tunable.
  The rest is a constant, hence a win wherever the in-graph route grows; what remains of the crossing is the three exports and the legacy-triple conversion, i.e. the host glue the kernel cannot absorb.
- **(c) The operator layer's floor (~0.53–0.81 ms per generation).** The variation lane is 0.525 / 0.538 / 0.767 / 0.654 / 0.610 ms at p64 / p256 / p1000 / p4096 / p256-N4096 on the in-graph side (0.582 / 0.638 / 0.810 / 0.651 / 0.627 on the fused side), i.e. 51–59 % of a `gp-fused` generation and more than half of it at every cell. It is a graph-run floor — a fixed number of traced ops plus the two population-scaled costs (the generation placement and the elite ranking) — and it is now the largest single item in every `gp-fused` cell.
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
- **The hand-off, in three eras (same cells, same protocol, warm medians).** The evaluation lane is the only thing that changed between them, and the last one writes into the buffer variation reads instead of copying into it:

  | cell | shim era (private payload handles) | native export + `copy_` | native export + `out=` (shipped) |
  |---|---:|---:|---:|
  | p64-N1024 | 1.161 | 1.223 | **1.149** |
  | p256-N1024 | 1.237 | 1.401 | **1.189** |
  | p1000-N1024 | 1.400 | 1.452 | **1.372** |
  | p4096-N1024 | 1.248 | 1.368 | **1.231** |
  | p256-N4096 | 1.207 | 1.227 | **1.179** |

  The middle column is what replacing the private handles with the official DLPack export cost when the return leg still had to `copy_` a torch result into the `evoxc` buffer (+1.7 % … +13.3 %, the widest at pop 256); the `out=` destination removes that copy AND keeps the zero-copy export, so the shipped column is below the shim era at every cell (-1.0 % … -3.9 %): the 0.030–0.035 ms fence plus the 0.05–0.08 ms of per-call copy and hand-off glue the buffer destination saves.
- **The numpy route is not competitive for a loop** (8.7× / 22.1× / 48.5× / 368× / 55.9× behind `gp-fused`): its evaluation is the host evaluator, which is 1.46–1.97× dearer again on evolved populations and swings 220.7–409.1 ms at pop 4096. It is the correctness oracle, not a loop backend.
- **Cold start**: `gp-fused` 3.32–3.61 s, `gp` 4.84–5.06 s, `torch` 3.50–3.56 s, `torch-fused` 0.61–0.63 s, `numpy` 0.08–0.38 s. Irrelevant for a long run (§3's payback rows: 7–362 generations against numpy, 1 890–2 246 against torch-fused, 0–5 against torch) but it dominates any short benchmark, and it is the one column where the fused loop does not beat `torch-fused`.
- **Run-to-run spread decides the close cells**: measured against the previous record's medians on the sides whose code did not change in between, the torch total moved -2.6 % … +1.9 %, `torch-fused` -1.0 % … +8.1 % (the wide-row cell is the outlier), and numpy -17.8 % … +3.7 %; the device lanes were re-measured after the operator change, so their only comparable column is the trajectory (§4), which is bit-identical. A cell decided by less than ~10 % is called parity here.
- **Variance**: `gp-fused` totals span 1.093–1.221 (p64), 1.150–1.253 (p256), 1.332–1.631 (p1000), 1.197–1.405 (p4096) and 1.141–1.207 ms (p256-N4096) — a few percent with one 1.63 outlier at p1000, no bimodality; the in-graph side is wider where the evaluation scales (5.993–6.228 at p4096) and the torch side widest (18.135–22.205 at p4096) because its engine call allocates per generation.

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
A `gp-fused` run additionally needs a writable `TRITON_CACHE_DIR` (the harness defaults it) and torch+triton importable; `fused_eval.interop_available` is the probe that decides whether the device route exists at all (it round-trips one element through the native DLPack export, so an evoxc without it answers `False` instead of failing mid-call).

Environment of record: python 3.12.3, torch 2.13.0+rocm7.1 (HIP 7.1.52802), triton 3.7.1, numpy 2.5.4, iree-base-compiler / iree-base-runtime 3.11.0, `evogp` `a2053ee` (the tree that produced the `gp-fused` column, with `out=`), GPU `AMD Instinct MI300X VF`.
The `gp`/`numpy`/`torch`/`torch-fused`/`fused` columns are the ones of the record at `d00d95b` and were not re-measured; the `gp-fused` column was re-measured against `evoxc` `474bf72` — the commit that adds the native DLPack device export, reached as a checkout through the task's evoxc worktree — because the `/root/evoxc` HEAD `9f66fec` still refuses `Tensor.__dlpack__` on a device payload (`DeviceError`).
The run's idle-device guard read `ok: true` on all five cells; the `guards` block records one co-resident idle python holding VRAM at 0–1 % GPU use throughout, which is why the close cells (pop 64) are read as parity.

## §9 Round 2 of the variation lane: the two lane divergences are gone, the lane is not faster

Two divergences the audit found are fixed in this round (see `gp/CONTEXT.md`):

* **non-finite fitness**: `variation._sanitize_fitness` now maps EVERY non-finite value (NaN, `+inf`, `-inf`) to `-inf` in-graph — the rule the host paths already applied — so a `+inf` can no longer win a tournament slot or an elite row on the device lane (`workflow._fitness`'s numpy branch got the same `~np.isfinite` fix);
* **the tie rule**: `selection`'s `rank` mode now ranks with a stable ASCENDING `argsort` of `-fitness` (first-maximal row wins), matching `variation._argmax_order` and the torch stack, instead of `argsort(..., descending=True)` whose flip resolved ties to the LAST row.

Both are pinned by new tests on both lanes (`test_gp_stack_operators_variation.py::{test_a_non_finite_fitness_can_never_become_an_elite, test_a_non_finite_fitness_cannot_win_a_tournament, test_the_elite_tie_rule_prefers_the_lower_row_index}`, `test_gp_stack_operators_selection.py::test_rank_mode_breaks_ties_towards_the_lower_row_index`, `test_gp_stack_device_driver.py::test_a_non_finite_fitness_can_never_become_a_device_elite[inf]`).

### Re-measured per-generation table (`--sides gp-fused`, same protocol: one process per cell, warm medians of 15, idle-device guard, raw numbers `evidence/device_loop_gp_fused_round2.json`)

| cell | evaluation | operator | total | previous total |
|------|-----------:|---------:|------:|---------------:|
| p64      | 0.549 | 0.574 | 1.117 | 1.149 |
| p256     | 0.600 | 0.663 | 1.221 | 1.189 |
| p1000    | 0.544 | 0.794 | 1.353 | 1.372 |
| p4096    | 0.571 | 0.656 | 1.230 | 1.231 |
| p256×4096| 0.549 | 0.626 | 1.163 | 1.179 |

**Verdict: no measurable gain.** Every cell is inside the ±0.03–0.05 ms box noise of the previous round (p256 reads 0.663 against 0.638, p1000 0.794 against 0.810), so the two correctness fixes plus the elite-slice simplification (`variation._assemble_elites` takes the kept offspring rows with a static `slice` instead of an index `gather`) are neutral on the clock. The targets of this round (operator ≤ 0.45 ms at p256/p4096, ≤ 0.55 ms at p1000) were **not met**.

Where the operator lane's ~0.6 ms actually goes (component cost on the device, acceptance config, standalone graphs at p256, medians of 30, ~0.09 ms of per-run floor included):

| component | p256 |
|-----------|-----:|
| floor (one dispatch, five device inputs) | ~0.09 |
| `split_n(key, 5)` alone | 0.095 |
| two `select_indices` (tournament 4) | 0.137 |
| `draw_positions` | 0.093 |
| one `_splice_traced` (3 channels) | 0.156 |
| `apply_mutation("subtree")` (generator + splice) | 0.345 |
| `_elite_order` / `_assemble_elites` | ~0.10 |

The two facts that explain the plateau:

* **op count is not the clock.** 1 526 lowered stablehlo ops (`456` constants, `242` reshapes, `140` gathers), but chained elementwise ops are free: 200 / 400 / 800 chained `add`s on `[256, 32]` all read ~0.077 ms. The cost sits in the non-fusable ops — the counter-based RNG (each `uniform`/`randint` hashes its whole tensor at ~4–5 ns per element), the gathers, the one sort and the per-run dispatch floor.
* **the generator is the whale.** `mutation_kind="subtree"` re-runs the whole padded generator as its donor: 3 × `[pop, 15]` uniform draws plus the arity-cdf chain. Folding those three draws into one `[pop, 45]` uniform and reading the two index draws off it (measured, then reverted) shortened the graph by ~200 ops and bought 0.06 ms of the mutation stage and nothing end to end — the RNG cost is per ELEMENT, not per draw, so the element count has to fall to pay, and every reduction there changes the offspring distribution (which the round's rule forbids).

The next lever is therefore the generator's element count, not its op count: e.g. generating the subtree-mutation donor on a smaller static layout (a shorter `max_layer_cnt` for the donor only) changes the mutation distribution and would have to be re-validated against the torch reference before it can ship.
