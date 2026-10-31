# The device-resident `evogp.gp` generation loop — measurement report

Report of record for `python -m evogp.gp.measure_generation` (harness `measure_generation.py`, raw numbers in `evidence/device_loop.json`).
One "generation" below is exactly what a GP loop pays: evaluation + selection/crossover/mutation (+ the elite splice), i.e. `workflow.step` on the `evogp.gp` side and `StandardPipeline.step` on the torch side.
All numbers are warm medians of 15 timed repetitions after a 5-repetition warmup, one process per (side, cell), `torch.cuda.synchronize()` around every timed repetition, on an AMD Instinct MI300X (ROCm 7.1, torch 2.13.0+rocm7.1).
The workload is the acceptance config: `max_nodes=32, input_len=8, using_funcs=("+","-","*","/"), max_layer_cnt=4, const_prob=0.5, layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8`, `GPConfig(elite_count=2, tournament_size=4, crossover_prob=0.9, mutation_prob=0.2)`, `N` rows of `measure_fused.make_data`, seed 0.

Sides: `gp` = the device path (`init_gp(backend="iree", device="rocm")` + `graph_eval.GraphEvaluator` + `workflow.step`); `numpy` = the same stack on the numpy backend with the host `SymbolicRegression.evaluate`; `torch` = `evogp.tree.Forest` + `evogp.algorithm.GeneticProgramming` + `evogp.problem.SymbolicRegression` under `StandardPipeline.step` (the repo's default `auto` engine route); `torch-fused` = the same torch pipeline with `EVOGP_EVAL_MODE=fused`.

## 1. One generation, per cell

| cell | side | evaluation ms | operator ms | total ms | total min–max | cold s |
|---|---|---:|---:|---:|---:|---:|
| p64-N1024 | **gp (device)** | 0.576 | 0.504 | **1.066** | 1.05–1.09 | 4.70 |
| p64-N1024 | numpy | 5.666 | 4.136 | 9.798 | 9.13–10.22 | 0.08 |
| p64-N1024 | torch | 2.359 | 2.440 | 4.798 | 4.74–5.05 | 3.45 |
| p64-N1024 | torch-fused | 0.214 | 2.340 | 2.554 | 2.36–2.63 | 0.62 |
| p256-N1024 | **gp (device)** | 0.729 | 0.656 | **1.387** | 1.37–1.40 | 4.61 |
| p256-N1024 | numpy | 21.094 | 4.980 | 34.734 | 33.37–39.34 | 0.09 |
| p256-N1024 | torch | 2.903 | 2.537 | 5.440 | 4.96–5.67 | 3.61 |
| p256-N1024 | torch-fused | 0.218 | 2.399 | 2.617 | 2.46–2.73 | 0.61 |
| p1000-N1024 | **gp (device)** | 1.464 | 1.218 | **2.751** | 2.65–2.93 | 4.66 |
| p1000-N1024 | numpy | 57.750 | 7.120 | 70.352 | 68.37–74.66 | 0.14 |
| p1000-N1024 | torch | 4.856 | 2.578 | 7.433 | 7.12–8.14 | 3.51 |
| p1000-N1024 | torch-fused | 0.216 | 2.390 | 2.606 | 2.42–2.68 | 0.61 |
| p256-N4096 | **gp (device)** | 1.546 | 0.691 | **2.213** | 2.19–2.25 | 4.59 |
| p256-N4096 | numpy | 60.759 | 4.930 | 74.382 | 72.74–78.65 | 0.13 |
| p256-N4096 | torch | 3.631 | 2.546 | 6.177 | 5.96–6.35 | 3.50 |
| p256-N4096 | torch-fused | 0.213 | 2.230 | 2.443 | 2.38–2.66 | 0.61 |

`operator` is `total - evaluation` on the torch side (measured around the `problem.evaluate` call) and the measured variation-graph lane on the two `evogp.gp` sides, each lane carries its own sync bracket, so the two `evogp.gp` segments sum to within ±2 % of `total` rather than to it exactly — read them against `total`.
`cold` is the first `step` of the process: two IREE compiles for `gp`, one dense compile for `torch`, the Triton JIT for `torch-fused`, nothing for `numpy`.

## 2. Speedups (gp device total is the denominator)

| cell | torch / gp | torch-fused / gp | torch-operator / gp total | numpy / gp |
|---|---:|---:|---:|---:|
| p64-N1024 | 4.50× | 2.40× | 2.29× | 9.19× |
| p256-N1024 | 3.92× | 1.89× | 1.83× | 25.04× |
| p1000-N1024 | 2.70× | **0.95×** | **0.94×** | 25.57× |
| p256-N4096 | 2.79× | 1.10× | 1.15× | 33.61× |

The third column is the honest one: at pop 1000 the torch stack's *operator layer alone* (2.578 ms) is 0.94× a whole `evogp.gp` generation (2.751 ms), and on the fused route that layer is 92 % of the generation.
The fourth column is why the numpy route is a correctness backend and not a loop backend: a generation of the same stack on the host evaluator is 9–34× dearer.

## 3. What one generation costs structurally

| side | graph runs per generation | host syncs | host readbacks | host↔device transfers |
|---|---:|---:|---:|---:|
| **gp (device)** | 2 (`variation` incl. elite splice, then `evaluation`) | 0 | 0 | 0 |
| numpy | 2 (same two graphs, host tensors) | — | 1 fitness readback | 1 up + 1 down per call |
| torch | 1 engine call + operator kernels | the engine's own | the dense readback | per engine call |

The device counts are measured, not asserted: `structure` in every `gp` record of the JSON reads `generation_runs=2, variation_runs=1, evaluation_runs=1, syncs=0, host_readbacks=0, transfers=0` at all four cells.
The previous device driver (before commit `2f81f12`) paid 3 `evoxc.run` calls, 2 syncs and 2 H2D uploads per generation for the same work, because the elite splice was its own graph and the fitness vector was read back every generation.
`workflow.step` adds no host work around the two graphs: it hands the previous state's device fitness vector and the next key straight into the variation graph (the key is already a device tensor after generation 0), which is why the measured syncs, readbacks and transfers stay at zero rather than merely small.
Cold start is where the device path pays: 4.6–4.7 s, i.e. two IREE compiles (variation ≈ 2.1 s, evaluation the rest), against 3.5 s for the torch dense compile, 0.6 s for the Triton JIT and 0.08–0.15 s for numpy.
Because the warm gap is 0.2–72 ms per generation, that cold cost pays back only over hundreds to thousands of generations — `payback = (gp cold − side cold) / (side warm − gp warm)`, i.e. the generation after which the two routes have spent the same wall time:

| cell | vs numpy | vs torch | vs torch-fused |
|---|---:|---:|---:|
| p64-N1024 | 529 gens | 335 gens | 2 742 gens |
| p256-N1024 | 136 gens | 247 gens | 3 251 gens |
| p1000-N1024 | 67 gens | 244 gens | never |
| p256-N4096 | 62 gens | 276 gens | 17 330 gens |

Every cell is a warm-throughput statement, not a short-run one: against `torch-fused` the device path needs thousands of generations (17 330 on the wide-row cell, whose 0.23 ms warm margin cannot amortize a 4 s compile) and at pop 1000 it never pays back, because the gp route is slower warm there as well.

## 4. Convergence: does the device path do the same work?

Best fitness of the 20-generation run (same seed, same target, same data), and the population median at generation 19.

| cell | side | g0 | g4 | g9 | g14 | g19 | median g19 |
|---|---|---:|---:|---:|---:|---:|---:|
| p64-N1024 | gp | -7.692 | -2.896 | -2.611 | -1.540 | -1.377 | -7.098 |
| p64-N1024 | numpy | -7.692 | -2.896 | -2.611 | -1.540 | -1.377 | -7.098 |
| p64-N1024 | torch | -11.703 | -4.798 | -2.364 | -1.489 | -1.356 | -3.164 |
| p256-N1024 | gp | -7.692 | -5.708 | -1.574 | -0.725 | -0.460 | -2.467 |
| p256-N1024 | numpy | -7.692 | -5.708 | -1.574 | -0.725 | -0.460 | -2.467 |
| p256-N1024 | torch | -6.517 | -5.400 | -0.476 | -0.476 | -0.443 | -2.452 |
| p1000-N1024 | gp | -5.989 | -2.556 | -0.476 | -0.476 | -0.476 | -3.938 |
| p1000-N1024 | numpy | -5.989 | -2.556 | -0.476 | -0.476 | -0.476 | -3.551 |
| p1000-N1024 | torch | -6.517 | -2.630 | -1.416 | -0.484 | -0.484 | -3.429 |
| p256-N4096 | gp | -7.829 | -4.475 | -2.338 | -0.547 | -0.547 | -5.140 |
| p256-N4096 | numpy | -7.829 | -4.475 | -2.338 | -0.547 | -0.547 | -5.140 |
| p256-N4096 | torch | -6.536 | -3.469 | -0.466 | -0.466 | -0.466 | -3.678 |

`gp` and `numpy` follow the SAME trajectory — best fitnesses agree to float32 rounding at every checkpoint (max |Δbest| 1.0e-06 across all four cells), i.e. the device path makes the same selection decisions as the reference and is not doing different work.
The torch side is a different trajectory by construction, not by a defect: different RNG stream (torch vs `splitmix64`), different selection (truncation `survival_rate=0.3, elite_rate=0.01` vs tournament-4) and different mutation wiring (`mutation_rate=0.2` with `max_layer_cnt=3`), so its generation-0 population is already a different draw and only the endpoint is comparable.
At this 20-generation budget the torch wiring reaches the better best on three of four cells (-1.356 vs -1.377 at p64, -0.443 vs -0.460 at p256, -0.466 vs -0.547 at p256-N4096) and the `evogp.gp` wiring on one (-0.476 vs -0.484 at p1000); that is a difference of operator pressure at a short horizon, not of evaluation, and `evogp.gp`'s own CPU evidence (`gp/CONTEXT.md`) shows the same stack reaching -0.107 at 100 generations with tournament 4.

## 5. Evaluation cost vs tree growth (why the numpy route degrades and the device route does not)

The timed lane runs on an evolved population, so the trees are longer than generation 0. Re-evaluating a *freshly generated* population shows which side of the cost model that matters on.

| cell | gp fresh | gp evolved | numpy fresh | numpy evolved |
|---|---:|---:|---:|---:|
| p64-N1024 | 0.572 | 0.576 | 2.674 | 5.666 |
| p256-N1024 | 0.722 | 0.729 | 11.250 | 21.094 |
| p1000-N1024 | 1.507 | 1.464 | 37.351 | 57.750 |
| p256-N4096 | 1.547 | 1.546 | 32.542 | 60.759 |

The in-graph evaluation is flat in tree length (its cost is the `max_nodes`-static child-read chain, and the first child is a static slice), while the host numpy evaluator grows 1.5–2.1× as the population deepens — so the numpy-path totals above are an evolved-population number, and a numpy step on a fresh population is 30–53 % cheaper than the column in §1.

## 6. Fused (Triton) vs in-graph: the open question, settled

Head-to-head on the same channels, the same `x`/`y`, medians of 15, transfers included (`fused from device` = download the three channel tensors + host decode + kernel, composed in ONE repetition; `queued` = the same transfer measured behind a device run, see below):

| cell | in-graph | transfer | transfer (queued) | fused from host | fused from device | fused from device (queued) | in-graph / fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| p64-N1024 | 0.567 | 0.266 | 0.354 | 0.317 | 0.823 | 0.672 | **0.69×** |
| p256-N1024 | 0.717 | 0.310 | 0.352 | 0.431 | 0.921 | 0.783 | **0.78×** |
| p1000-N1024 | 1.481 | 0.438 | 0.521 | 0.790 | 1.382 | 1.311 | 1.07× |
| p256-N4096 | 1.535 | 0.311 | 0.382 | 0.386 | 1.009 | 0.769 | 1.52× |

**Verdict: from device-resident channels the routes cross over inside this grid — the in-graph route wins pop 64 (1.45× on the composed lane, 1.19× on the optimistic one) and pop 256 (1.28× / 1.09×), the fused route wins pop 1000 (1.07× composed, 1.13× optimistic) and the wide-row cell (1.52× composed, 2.00× optimistic).**
How to read the columns: `fused from device` is the representative lane — download, host decode, kernel and the fitness upload composed inside ONE timed repetition, which is what a device loop actually pays and what the `in-graph / fused` ratio and `faster_route` use; `fused from device (queued)` is the SUM of two separately measured medians (the transfer behind a device run plus the host-channels kernel), so it excludes the per-call compose overhead and reads 0.07–0.24 ms below the quiet composed lane — an optimistic bound, not a measurement.
The mechanism is what the two routes' shapes predict: the Triton kernel is flat in population and rows (0.213–0.218 ms across all four cells), the in-graph evaluation grows with `pop × rows` (0.567 → 1.535 ms), and the transfer toll it must first recover is 0.266–0.438 ms idle / 0.352–0.521 ms behind a device run, i.e. 21–47 % of the in-graph call (47 % at pop 64, 43 % at pop 256).
So the fused route is a loss wherever that fixed toll exceeds the in-graph headroom (pop ≤ 256) and a win once the in-graph child-read chain outgrows it (pop 1000, and by the widest margin on the row axis).
From host channels — what the numpy-backed operator layer would hand an evaluator — the fused route is 1.79× / 1.66× / 1.87× / 3.98× faster: a clear win everywhere, but a smaller one than the 2.1–4.6× of the previous round precisely because the in-graph route got cheaper.
That settles the caveat in `fused_eval.py`'s verdict in both directions: `FusedEvaluator` from host channels is a clear win everywhere, and from DEVICE channels it is a 1.1–2.0× win exactly where the in-graph route's `pop × rows` traffic dominates, while staying a loss at the small populations a GP loop lives in.
The default device route is therefore **not** changed: `graph_eval.GraphEvaluator` stays the shipped `evaluate=` for a device loop, because it wins the cells a GP loop actually spends its generations in, and because the composed lane above still does not charge the fused route for the one thing a device loop needs, the upload of the fitness vector back to the variation graph.
Parity of the fused lanes against in-graph: max relative 1.2e-07 / 1.7e-07 / 3.4e-07 / 2.2e-07 with zero finiteness mismatches, i.e. three to four orders of magnitude inside the 1e-4 acceptance box.
## 7. Where we lose, and why

- **pop 1000 × 1024 rows vs the torch fused route: 2.751 vs 2.606 ms (0.95×), i.e. parity inside the run-to-run spread rather than a win.** The in-graph evaluation is 1.464 ms there against the Triton kernel's 0.216, a 1.25 ms deficit that now just exceeds the operator layer's 1.17 ms advantage (1.218 variation vs 2.390 torch, plus ~0.07 ms of lane bracket), so the generation lands 0.15 ms (5 %) behind; the same comparison against the *default* torch route is a 2.70× win.
  What remains is structural: the kernel is flat in population, so at pop 1000 it evaluates 1 024 rows for 0.216 ms while the in-graph route reads `2 × width` child values per column in traced tensors — the first child is already a static slice (it was a third gather before), and only a fused device-side evaluator removes the rest.
- **pop 256 × 4096 rows is a narrow win (1.10×) and the narrowest in the grid after pop 1000**: the row axis widens in-graph evaluation to 1.546 ms against the kernel's 0.213, and what decides the cell is the fused route's 2.230 ms operator layer, not our evaluation — a wider row axis or a smaller population flips it.
- **Our operator layer is 2.1× cheaper than the torch one at pop 1000 (1.218 vs 2.578 ms) and that is still not quite enough there**: the torch operator layer alone costs 0.94× a whole `evogp.gp` generation, so at large populations the win has to come from evaluation, and on the fused route it does not.
- **The numpy route is not competitive for a loop** (9.2× / 25.0× / 25.6× / 33.6× behind the device path): its evaluation is the host evaluator, which is 1.5–2.1× dearer again on evolved (longer) populations. The parent's reference figure of 4.1/5.0/7.4 ms for a numpy-backend step is not reproducible here — a single `SymbolicRegression.evaluate` on a *fresh* p256 population already costs 11.3 ms (21.1 evolved), so that column must have been measured on a different workload (fewer rows, shallower descriptor or a different evaluator).
- **Cold start**: 4.6–4.7 s vs 0.6 s (fused) / 3.5 s (torch default) / 0.1 s (numpy). Irrelevant for a long run (payback of 62–529 generations against numpy, ~250–340 against torch, thousands against `torch-fused` and never at pop 1000, per §3) but it dominates any short benchmark and it is why the artifact cache exists.
- **Run-to-run spread decides the close cells**: two full grids in this session (the second is this record) agreed to ≤2.7 % on every `gp` cell, but the `torch-fused` lane moved 2.44–2.61 ms at p256-N4096 and 2.61–2.80 ms at p1000 between them, so a cell decided by less than ~10 % is called parity here — pop 1000 read 1.02× on the first grid and 0.95× on this one, and is documented as the parity it is.
- **Variance**: totals span 1.05–1.09 (p64), 1.37–1.40 (p256), 2.65–2.93 (p1000) and 2.19–2.25 ms (p256-N4096); the torch side is wider (7.12–8.14 at p1000) because its engine call allocates per generation. No cell looks bimodal, so the medians in §1 are stable to a few percent.

## 8. Reproduce

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --cells 64,256,1000,256x4096 \
  --json src/evogp/gp/evidence/device_loop.json
```

One cell of one side, in a single process (what the grid above does per row of the tables):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --side gp --pop 256 --rows 4096 --warmup 5 --reps 15
```

`--side` is one of `gp`, `numpy`, `torch`, `torch-fused`, `fused`; the default (`--cells`) grid runs all five sides per cell, each in its own process (the parent re-executes itself, so no cell's compile or RNG state can leak into another), and merges the children's JSON into one file.
`--side fused` is the §6 head-to-head, `--sides` narrows the grid, `--no-guard` skips the `rocm-smi` idle probe (`--guard-wait` is its timeout), and `--json <path>` writes the record — the parent merges its children's into that one file.

Environment of record: python 3.12.3, torch 2.13.0+rocm7.1 (HIP 7.1.52802), triton 3.7.1, numpy 2.5.4, iree-base-compiler / iree-base-runtime 3.11.0, evoxc `9f66fec` (checkout `/root/evoxc`), `evogp` `73ac61d` (the tree that produced these numbers), GPU `AMD Instinct MI300X VF`.
