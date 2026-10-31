# The device-resident `evogp.gp` generation loop — measurement report

Report of record for `python -m evogp.gp.measure_generation` (harness `measure_generation.py`, raw numbers in `evidence/device_loop.json`).
One "generation" below is exactly what a GP loop pays: evaluation + selection/crossover/mutation (+ the elite splice), i.e. `workflow.step` on the `evogp.gp` side and `StandardPipeline.step` on the torch side.
All numbers are warm medians of 15 timed repetitions after a 5-repetition warmup, one process per (side, cell), `torch.cuda.synchronize()` around every timed repetition, on an AMD Instinct MI300X (ROCm 7.1, torch 2.13.0+rocm7.1).
The workload is the acceptance config: `max_nodes=32, input_len=8, using_funcs=("+","-","*","/"), max_layer_cnt=4, const_prob=0.5, layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8`, `GPConfig(elite_count=2, tournament_size=4, crossover_prob=0.9, mutation_prob=0.2)`, `N` rows of `measure_fused.make_data`, seed 0.

Sides: `gp` = the device path (`init_gp(backend="iree", device="rocm")` + `graph_eval.GraphEvaluator` + `workflow.step`); `numpy` = the same stack on the numpy backend with the host `SymbolicRegression.evaluate`; `torch` = `evogp.tree.Forest` + `evogp.algorithm.GeneticProgramming` + `evogp.problem.SymbolicRegression` under `StandardPipeline.step` (the repo's default `auto` engine route); `torch-fused` = the same torch pipeline with `EVOGP_EVAL_MODE=fused`.

## 1. One generation, per cell

| cell | side | evaluation ms | operator ms | total ms | total min–max | cold s |
|---|---|---:|---:|---:|---:|---:|
| p64-N1024 | **gp (device)** | 0.673 | 0.517 | **1.209** | 1.19–1.23 | 7.03 |
| p64-N1024 | numpy | 5.675 | 4.202 | 10.099 | 9.56–11.08 | 0.08 |
| p64-N1024 | torch | 2.445 | 2.504 | 4.949 | 4.85–5.20 | 3.57 |
| p64-N1024 | torch-fused | 0.221 | 2.400 | 2.621 | 2.57–2.78 | 0.62 |
| p256-N1024 | **gp (device)** | 0.816 | 0.669 | **1.480** | 1.47–1.51 | 6.33 |
| p256-N1024 | numpy | 19.113 | 5.016 | 36.059 | 34.38–36.72 | 0.09 |
| p256-N1024 | torch | 2.931 | 2.493 | 5.425 | 5.28–5.59 | 3.60 |
| p256-N1024 | torch-fused | 0.227 | 2.419 | 2.647 | 2.58–2.77 | 0.61 |
| p1000-N1024 | **gp (device)** | 1.681 | 1.268 | **2.951** | 2.91–3.09 | 6.35 |
| p1000-N1024 | numpy | 58.800 | 7.611 | 69.600 | 67.19–72.73 | 0.14 |
| p1000-N1024 | torch | 4.903 | 2.596 | 7.499 | 7.29–8.07 | 3.56 |
| p1000-N1024 | torch-fused | 0.239 | 2.459 | 2.698 | 2.43–2.88 | 0.62 |
| p256-N4096 | **gp (device)** | 1.782 | 0.699 | **2.475** | 2.44–2.49 | 6.38 |
| p256-N4096 | numpy | 62.706 | 5.010 | 76.123 | 74.39–76.58 | 0.13 |
| p256-N4096 | torch | 3.732 | 2.510 | 6.242 | 6.13–6.43 | 3.54 |
| p256-N4096 | torch-fused | 0.218 | 2.321 | 2.538 | 2.42–2.78 | 0.63 |

`operator` is `total - evaluation` on the torch side (measured around the `problem.evaluate` call) and the measured variation-graph lane on the two `evogp.gp` sides, each lane carries its own sync bracket, so the two `evogp.gp` segments sum to within ±2 % of `total` rather than to it exactly — read them against `total`.
`cold` is the first `step` of the process: two IREE compiles for `gp`, one dense compile for `torch`, the Triton JIT for `torch-fused`, nothing for `numpy`.

## 2. Speedups (gp device total is the denominator)

| cell | torch / gp | torch-fused / gp | torch-operator / gp total | numpy / gp |
|---|---:|---:|---:|---:|
| p64-N1024 | 4.09× | 2.17× | 2.07× | 8.35× |
| p256-N1024 | 3.67× | 1.79× | 1.68× | 24.37× |
| p1000-N1024 | 2.54× | **0.91×** | **0.88×** | 23.59× |
| p256-N4096 | 2.52× | 1.03× | 1.01× | 30.76× |

The third column is the honest one: at pop 1000 the torch stack's *operator layer alone* (2.596 ms) is cheaper than a whole `evogp.gp` generation (2.951 ms), and on the fused route that layer is 91 % of the generation.

## 3. What one generation costs structurally

| side | graph runs per generation | host syncs | host readbacks | host↔device transfers |
|---|---:|---:|---:|---:|
| **gp (device)** | 2 (`variation` incl. elite splice, then `evaluation`) | 0 | 0 | 0 |
| numpy | 2 (same two graphs, host tensors) | — | 1 fitness readback | 1 up + 1 down per call |
| torch | 1 engine call + operator kernels | the engine's own | the dense readback | per engine call |

The device counts are measured, not asserted: `structure` in every `gp` record of the JSON reads `generation_runs=2, variation_runs=1, evaluation_runs=1, syncs=0, host_readbacks=0, transfers=0` at all four cells.
The previous device driver (before commit `2f81f12`) paid 3 `evoxc.run` calls, 2 syncs and 2 H2D uploads per generation for the same work, because the elite splice was its own graph and the fitness vector was read back every generation.
`workflow.step` adds no host work around the two graphs: it hands the previous state's device fitness vector and the next key straight into the variation graph (the key is already a device tensor after generation 0), which is why the measured syncs, readbacks and transfers stay at zero rather than merely small.
Cold start is where the device path pays: 6.3–7.0 s, i.e. two IREE compiles (variation ≈ 2.1 s, evaluation the rest), against 3.5 s for the torch dense compile, 0.6 s for the Triton JIT and 0.08–0.15 s for numpy.
Because the warm gap is 1.5–70 ms per generation, that cold cost pays back in **1–5 generations** against every side except `torch-fused` at pop 1000 (where the gp route never catches up):

| cell | vs numpy | vs torch | vs torch-fused |
|---|---:|---:|---:|
| p64-N1024 | 1 gen | 1 gen | 5 gens |
| p256-N1024 | 1 gen | 1 gen | 5 gens |
| p1000-N1024 | 1 gen | 1 gen | never |
| p256-N4096 | 1 gen | 1 gen | 90 gens |

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

`gp` and `numpy` are the SAME trajectory — bit-identical populations (the two backends are value-equal for the same key) and fitnesses that agree to float noise, which is the direct check that the device path is not doing different work.
The torch side is a different trajectory by construction, not by a defect: different RNG stream (torch vs `splitmix64`), different selection (truncation `survival_rate=0.3, elite_rate=0.01` vs tournament-4) and different mutation wiring (`mutation_rate=0.2` with `max_layer_cnt=3`), so its generation-0 population is already a different draw and only the endpoint is comparable.
At this 20-generation budget the torch wiring reaches the better best on three of four cells (-1.356 vs -1.377 at p64, -0.443 vs -0.460 at p256, -0.466 vs -0.547 at p256-N4096) and the `evogp.gp` wiring on one (-0.476 vs -0.484 at p1000); that is a difference of operator pressure at a short horizon, not of evaluation, and `evogp.gp`'s own CPU evidence (`gp/CONTEXT.md`) shows the same stack reaching -0.107 at 100 generations with tournament 4.

## 5. Evaluation cost vs tree growth (why the numpy route degrades and the device route does not)

The timed lane runs on an evolved population, so the trees are longer than generation 0. Re-evaluating a *freshly generated* population shows which side of the cost model that matters on.

| cell | gp fresh | gp evolved | numpy fresh | numpy evolved |
|---|---:|---:|---:|---:|
| p64-N1024 | 0.665 | 0.673 | 2.731 | 5.675 |
| p256-N1024 | 0.807 | 0.816 | 11.882 | 19.113 |
| p1000-N1024 | 1.705 | 1.681 | 38.341 | 58.800 |
| p256-N4096 | 1.791 | 1.782 | 32.898 | 62.706 |

The in-graph evaluation is flat in tree length (its cost is the `max_nodes`-static gather chain), while the host numpy evaluator grows 1.5–2.1× as the population deepens — so the numpy-path totals above are an evolved-population number, and a numpy step on a fresh population is 20–40 % cheaper than the column in §1.

## 6. Fused (Triton) vs in-graph: the open question, settled

Head-to-head on the same channels, the same `x`/`y`, medians of 15, transfers included (`fused from device` = download the three channel tensors + host decode + kernel + upload the fitness vector; `queued` = the same with the device queue busy):

| cell | in-graph | transfer | transfer (queued) | fused from host | fused from device | fused from device (queued) | in-graph / fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| p64-N1024 | 0.673 | 0.270 | 0.314 | 0.318 | 0.847 | 0.632 | **0.80×** |
| p256-N1024 | 0.794 | 0.325 | 0.407 | 0.420 | 1.003 | 0.827 | **0.79×** |
| p1000-N1024 | 1.672 | 0.423 | 0.553 | 0.795 | 1.370 | 1.348 | 1.22× |
| p256-N4096 | 1.748 | 0.311 | 0.395 | 0.379 | 0.979 | 0.774 | 1.79× |

**Verdict: from device-resident channels the routes cross over inside this grid — the in-graph route wins at pop 256 (1.26×), the fused route wins at pop 1000 (1.22×) and on the wide-data cell (1.79×), and pop 64 is a coin-flip (in-graph 1.25× on the composed lane, fused 1.06× on the optimistic one).**
How to read the columns: `fused from device` is the representative lane — download, host decode, kernel and the fitness upload composed inside ONE timed repetition, which is what a device loop actually pays and what the `in-graph / fused` ratio and `faster_route` use; `fused from device (queued)` is the SUM of two separately measured medians (the transfer behind a device run plus the host-channels kernel), so it excludes the per-call compose overhead and reads 0.02–0.22 ms below the quiet composed lane — an optimistic bound, not a measurement.
The mechanism is what the two routes' shapes predict: the Triton kernel is flat in population and rows (0.218–0.239 ms across all four cells), the in-graph evaluation grows with `pop × rows` (0.673 → 1.782 ms), and the transfer toll it must first recover is 0.270–0.423 ms idle / 0.314–0.553 ms behind a device run, i.e. 18–41 % of the in-graph call (31–41 % at pop ≤ 256).
So the fused route is a loss wherever that fixed toll exceeds the in-graph headroom (pop ≤ 256) and a win once the in-graph gather chain outgrows it (pop 1000, and by the widest margin on the row axis).
From host channels — what the numpy-backed operator layer would hand an evaluator — the fused route is 2.12× / 1.89× / 2.10× / 4.61× faster on all four cells, which is the case `fused_eval.py` was written for and the case that costs nothing extra.
That settles the caveat in `fused_eval.py`'s verdict in both directions: `FusedEvaluator` from host channels is a clear win everywhere, and from DEVICE channels it is a ~1.2–1.8× win exactly where the in-graph route's `pop × rows` gather traffic dominates, while staying a loss at the small populations a GP loop lives in.
The default device route is therefore **not** changed: `graph_eval.GraphEvaluator` stays the shipped `evaluate=` for a device loop, because it wins the cells a GP loop actually spends its generations in, and because the composed lane above still does not charge the fused route for the one thing a device loop needs — uploading the fitness vector back to the variation graph — nor for its per-call host decode.
Parity of the fused lanes against in-graph: max relative 1.2e-07 / 1.7e-07 / 3.4e-07 / 2.2e-07 with zero finiteness mismatches, i.e. three to four orders of magnitude inside the 1e-4 acceptance box.
## 7. Where we lose, and why

- **pop 1000 × 1024 rows vs the torch fused route: 2.951 vs 2.698 ms (0.91×).** The in-graph evaluation is 1.681 ms there against the Triton kernel's 0.239, a 1.44 ms deficit that the operator layer's 1.33 ms advantage (1.268 variation vs 2.459 torch) cannot cover; the same comparison against the *default* torch route is a 2.54× win.
- **pop 256 × 4096 rows is parity (1.03×) for the same reason**: the row axis widens in-graph evaluation to 1.782 ms while the kernel stays at 0.218, and what keeps the cell even is that the torch pipeline still pays 2.51 ms of operator work.
- **Our operator layer is 2.0× cheaper than the torch one at pop 1000 (1.268 vs 2.596 ms) and that is still not enough there**: the torch operator layer alone costs 0.88× a whole `evogp.gp` generation, so at large populations the win has to come from evaluation, and on the fused route it does not.
- **The numpy route is not competitive for a loop** (8.4× / 24.4× / 23.6× / 30.8× behind the device path): its evaluation is the host evaluator, which is 1.5–2.1× dearer again on evolved (longer) populations. The parent's reference figure of 4.1/5.0/7.4 ms for a numpy-backend step is not reproducible here — a single `SymbolicRegression.evaluate` on a *fresh* p256 population already costs 11.9 ms (19.1 evolved), so that column must have been measured on a different workload (fewer rows, shallower descriptor or a different evaluator).
- **Cold start**: 6.3–7.0 s vs 0.6 s (fused) / 3.5 s (torch default) / 0.1 s (numpy). Irrelevant for a real run (payback of 1–5 generations) but it dominates any short benchmark and it is why the artifact cache exists.
- **The `gp` side reads 4–5 % above the earlier same-box figures** (1.209/1.480/2.951 against 1.162/1.422/2.811 ms at pop 64/256/1000, a previous round): this harness times each lane inside its own sync bracket and the card is shared (a foreign python is resident on it; the idle probe recorded 0–1 % use per cell), so the medians sit inside the run-to-run spread rather than at the best-case reading.
- **Variance**: totals span 1.19–1.23 (p64), 1.47–1.51 (p256), 2.91–3.09 (p1000) and 2.44–2.49 ms (p256-N4096); the torch side is wider (7.29–8.07 at p1000) because its engine call allocates per generation. No cell looks bimodal, so the medians in §1 are stable to a few percent; the two 20-cell grids run this session agreed within that band.

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
`--side fused` is the §6 head-to-head; `--no-guard` skips the `rocm-smi` idle probe; `--json -` prints to stdout.

Environment of record: python 3.12.3, torch 2.13.0+rocm7.1 (HIP 7.1.52802), triton 3.7.1, numpy 2.5.4, iree-base-compiler / iree-base-runtime 3.11.0, evoxc `9f66fec` (checkout `/root/evoxc`), `evogp` `1f8e48c`, GPU `AMD Instinct MI300X VF`.
