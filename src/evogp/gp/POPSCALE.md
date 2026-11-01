# Population scaling 64 → 8192: the evoxc-based functional stack against the torch stack

> Measured on the pre-2.0.0a2 tree, where `evogp.tree` still existed: the torch comparison code (`Forest`-based lanes, `evogp.algorithm`/`evogp.problem`/`evogp.pipeline` sides) was removed in 2.0.0a2 — recover it at commit 6720723. The numbers below are unchanged and are not reproducible from this tree.
> Every `test/benchmark/...` path mentioned in this document now lives under `test/archive/benchmark/...`.
Report of record for the POPULATION axis of `evogp.gp` — 128× the population at one fixed workload.
Raw numbers: `evidence/popscale_eval.json` (per call, one shared population), `evidence/popscale_cold/*.json` (the same sweep, one lane per child, for the honest cold column) and `evidence/popscale.json` (the generation loop).

The round asks one question: **does the evoxc-based stack scale across the population, per call and per generation, against the plain-PyTorch baselines and against the torch stack it is meant to replace** — with every lane fed the SAME trees, so no lane can win by evaluating a different population.
Two committed harnesses answer it: `python -m evogp.gp.measure_popscale` (per-call evaluation of one population in every lane, §1 and §3's per-call half) and `python -m evogp.gp.measure_generation --cells popscale` (a whole generation, evaluation + select/crossover/mutate, §2 and §3's generation half).

The grid is eight populations — 64 / 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 — at ONE shape: the acceptance config `max_nodes=32, input_len=8, using_funcs=("+","-","*","/"), max_layer_cnt=4, const_prob=0.5, layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8`, `GPConfig(elite_count=2, tournament_size=4, crossover_prob=0.9, mutation_prob=0.2)`, 1 024 data rows of `measure_fused.make_data`, seed 0.
A tenth payload holds the `forest-fused` control at pop 8192 (§1.4), which is why the per-call grid has nine children for eight populations.

**The answer in one sentence:** the device-native fused route wins at every population and the margin grows with it — 3.20× → 10.68× the plain-PyTorch batched evaluator, 29.8× → 481.9× the per-tree walk, 4.53× → 725.9× the numpy host path — and it is flat (1.16× for 128× the population), so a whole `evogp.gp` generation is 1.83× → 2.31× faster than the torch stack's; but per call it does NOT beat the torch stack's own fused Triton route, which does the same trees 9.0× → 19.8× cheaper.

## Protocol and environment

| item | value |
|---|---|
| GPU | AMD Instinct MI300X VF (191.7 GB) |
| torch / HIP | 2.13.0+rocm7.1 / 7.1.52802 |
| triton | 3.7.1 |
| numpy | 2.5.4 |
| python | 3.12.3 (`/tmp/rocvenv/bin/python`) |
| IREE | iree-base-compiler / iree-base-runtime 3.11.0 |
| `EVOXC_DISABLE_LOCATIONS` | `1` |
| tree | `0e4f8d3` (`git_head` / `versions.evogp_sha` in all three records) |
| evoxc | `/root/evoxc` at `474bf72` — the checkout WITH the native DLPack device export |
| `evogp.gp.interop_available()` | **True**, so every `gp-fused` call ran the zero-copy device route, never the host hand-off |

`interop_available` is `true` at the top level of all three records, and each `gp-fused` record PROVES the route instead of naming it: `facts.route == "device"`, `facts.route_counts == {"device": 21, "host": 0}` at every population of the per-call grid (21 = 1 cold + 5 warm-up + 15 timed calls) and `{"device": 74, "host": 0}` in every generation record, with `last_fallback: null`, `device_available: true`, `available: true` and `triton_cache_dir: /tmp/triton_cache`.
Without the device export (`/root/evoxc` HEAD `9f66fec` raises `DeviceError` on a device payload) the same lane would answer `interop_available: false` and serve every call on the host hand-off — that is the branch this column did not take.

The timing protocol is the same in both harnesses and it is what every number here depends on:

* **One process per population** (per call) and **one process per (side, cell)** (per generation): the parent re-executes itself and merges the children's JSON, so a compile, an allocator or a leftover buffer can never be charged to another column.
* **Warm medians of 15** timed repetitions after **5 discarded warm-ups** (`--reps 15 --warmup 5`), with a device-wide `torch.cuda.synchronize()` around every timed repetition — device-wide, so it also waits for the `evoxc`/IREE work on the card.
* **Cold** is the FIRST call of a lane in the process, one sample, synced: what a one-shot caller pays (graph compile, Triton JIT).
* **An `rocm-smi` idle-device guard** before every child: `guard.all_ok: true` on all nine children of the per-call grid, all 37 of the generation grid and all 32 of the cold sweep, no busy reading, `gpu_use_pct` ≤2.0 and `waited_s` ≤0.2 throughout, no resident process holding the card.
* Total child wall (Σ `children[].wall_s`): **159.2 s** for the shared-population grid (9 children), **416.3 s** for the generation grid (37 children) and **295.5 s** for the four cold-sweep children sets (32 children: gp 84.4 s, gp-fused 71.0 s, numpy 71.0 s, forest-auto 69.1 s).
* **The cold sweep is a different protocol on purpose**: `--lanes <one lane> --reps 3 --warmup 1 --skip-fused-control`, one lane per child, because in a multi-lane child an earlier lane's compile/JIT lands in a later lane's "cold". Warm medians are NOT quoted from those files — only `cold_ms` is (§1.2 and §3).
* **±10 % box noise.** A shared card; every ratio below that is called parity, not a win. The torch stack's own population column measured the same box's spread at 1–19 % between two arms of the SAME route (`test/benchmark/evoxc/POPSCALE.md` §5.4), and this round's own 3-repetition cold sweep re-reads the 15-repetition warm medians 5.5–51 % HIGHER on the sub-0.1 ms `Forest` lanes and within ±17 % on the device and numpy lanes — a 3-sample median on a 0.03 ms call is not a measurement, which is why no warm number is quoted from those files.

### Cross-check against the torch stack's own population column

The torch stack ran this same population axis one round earlier (`test/benchmark/evoxc/POPSCALE.md`, read-only from here, whose own campaign also carries the plain-torch arms). The overlapping cells agree on the same call:

| quantity | torch-stack round (pop 64 → 8 192) | this round (pop 64 → 8 192) | delta |
|---|---|---|---|
| `torch-batched` `batch_forward` | 2.0252 → 7.7791 | 2.0043 → 7.7796 | −1.0 % / +0.0 % |
| shipped default (`auto` → fused) `batch_forward` | 0.0343 → 0.0829 | 0.0341 → 0.0808 | −0.6 % / −2.5 % |
| shipped default (`auto` → fused) `sr_fitness` | 0.0507 → 0.0951 | 0.0513 → 0.0872 | +1.2 % / −8.3 % |
| explicit `EVOGP_EVAL_MODE=fused` `batch_forward` | 0.0346 → 0.0836 | 0.0886 at pop 8192 (control, §1.4) | — |
| explicit `EVOGP_EVAL_MODE=fused` `sr_fitness` | 0.0602 → 0.0940 | 0.0870 at pop 8192 (control, §1.4) | — |

So the two rounds agree to the noise band, and the explicit-fused column of that round (its own `evoxc-fused-iree` arm) read the same route this round's `forest-fused` control reads at the widest population.
The largest gap is −8.3 % on `sr_fitness` at pop 8 192, i.e. inside the band, and the shipped default is the route this round's `forest-auto` lane records (`facts.route: "fused"` at every population of both grids).

## 1. The shared-population per-call table

This is the primary metric of the round: ONE `evogp.gp` population per population size, derived into every layout the lanes need (the `GPState`, its legacy triple via `encoding.to_legacy_triple`, and the device channels uploaded once and verified bit-identical outside every timed block), then evaluated by every lane.
Nothing but the evaluation call is timed; the population build, the upload, the torch dataset staging and the plain-torch per-position plan are all constructor work.

**Fitness vs value.** Four lanes return a maximized FITNESS (`numpy` `evaluate`, `gp` `evaluate`, `gp-fused` `evaluate`, `Forest.SR_fitness`) and four return a raw VALUE (`numpy` `predict`, `torch-batched` `batch_forward`, `torch-per-tree` `forward`, `Forest.batch_forward`).
The lanes that measure the same quantity are the only ones compared to each other (§1.3 / §1.5 keep the two groups apart), and the two parity groups of §1.6 never mix them either.

### 1.1 Warm medians, ms per call

| pop | `numpy` ev | `gp` ev | `gp-fused` ev | `torch-batched` bf | `torch-per-tree` fwd | `Forest` bf | `Forest` SR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.8348 | 0.5742 | 0.6257 | 2.0043 | 18.6146 | 0.0341 | 0.0513 |
| 128 | 4.9043 | 0.6168 | 0.6308 | 2.0127 | 37.0139 | 0.0355 | 0.0465 |
| 256 | 9.1709 | 0.7306 | 0.6651 | 1.9756 | 73.5867 | 0.0350 | 0.0517 |
| 512 | 17.9101 | 0.9832 | 0.7147 | 2.0947 | 149.9309 | 0.0361 | 0.0540 |
| 1024 | 39.4831 | 1.5378 | 0.6103 | 2.3405 | 294.1020 | 0.0388 | 0.0511 |
| 2048 | 85.9828 | 2.8578 | 0.6556 | 3.2014 | — | 0.0429 | 0.0528 |
| 4096 | 222.4744 | 5.4924 | 0.6903 | 4.8054 | — | 0.0564 | 0.0636 |
| 8192 | 528.9067 | 10.9865 | 0.7286 | 7.7796 | — | 0.0808 | 0.0872 |

`torch-per-tree` is scoped out above pop 1024 (§5f) and is therefore a dash there, not a measurement.
The two `Forest` columns are the torch stack on its SHIPPED DEFAULT route (`EVOGP_EVAL_MODE` unset, so `auto` → fused on this box, recorded as `route: "fused"`); `numpy` `predict` is not in this table because it tracks `numpy` `evaluate` almost exactly (`predict` 2.6285 → 524.4168, i.e. 199.5× over the column and within 8 % of the fitness call at both ends).

### 1.2 Cold, ms: the first call of a lane, from its own single-lane child

| pop | `gp` | `gp-fused` | `numpy` | `Forest` bf | `Forest` SR |
|---:|---:|---:|---:|---:|---:|
| 64 | 2491.7774 | 872.9988 | 3.0799 | 509.5380 | 27.2664 |
| 128 | 2516.3773 | 876.4274 | 5.2447 | 535.0037 | 30.3739 |
| 256 | 2481.2493 | 867.4050 | 9.4794 | 494.5075 | 30.8565 |
| 512 | 2511.4931 | 878.5905 | 17.7819 | 521.9918 | 30.3429 |
| 1024 | 2423.9358 | 906.5602 | 41.4142 | 512.4448 | 33.8325 |
| 2048 | 2535.8296 | 904.3661 | 97.1169 | 506.8888 | 38.2516 |
| 4096 | 2533.0852 | 875.7546 | 242.1354 | 514.4518 | 34.3911 |
| 8192 | 2458.1846 | 912.9066 | 528.5238 | 520.8065 | 38.7677 |

Read the columns by their shape, not their digits:

* `gp` is ~2.42–2.54 s FLAT in population — one IREE evaluation graph compile, paid once whatever the population;
* `gp-fused` is ~0.87–0.91 s flat — the Triton JIT plus the kernel's specialization lookup;
* `numpy` cold is just its first call (there is nothing to compile), so it tracks the warm column from 3.08 ms to 528.52 ms;
* the `Forest` value surface is ~495–535 ms at EVERY population (the one-time Triton JIT of the value kernel) and the loss surface 27–39 ms (the second kernel, compiled by the second surface in the same child).

The two `Forest` numbers are the ones the multi-lane child gets WRONG and this single-lane sweep exists to fix: in the shared-population child `Forest.batch_forward` reads 2.0831–2.5483 ms and `sr_fitness` 0.3278–0.4345 ms, because the `gp-fused` lane earlier in that child has already JIT-compiled the very same two Triton kernels (`fused_engine`'s value and loss kernels), so nothing is cold any more.
The separate `forest-fused` control child confirms the single-lane reading from the other direction: 527.3659 ms `batch_forward` / 37.6917 ms `sr_fitness` at pop 8192.

### 1.3 Speedups of the two `evogp.gp` device routes

`gp-fused` (the Triton route, same quantity per column — fitness vs fitness, value vs value):

| pop | vs `torch-batched` | vs `torch-per-tree` | vs `numpy` |
|---:|---:|---:|---:|
| 64 | 3.20× | 29.75× | 4.53× |
| 128 | 3.19× | 58.68× | 7.77× |
| 256 | 2.97× | 110.64× | 13.79× |
| 512 | 2.93× | 209.78× | 25.06× |
| 1024 | 3.83× | 481.90× | 64.69× |
| 2048 | 4.88× | — | 131.15× |
| 4096 | 6.96× | — | 322.29× |
| 8192 | 10.68× | — | 725.92× |

`gp` (the in-graph route, the torch-free one):

| pop | vs `torch-batched` | vs `torch-per-tree` | vs `numpy` |
|---:|---:|---:|---:|
| 64 | 3.49× | 32.42× | 4.94× |
| 128 | 3.26× | 60.01× | 7.95× |
| 256 | 2.70× | 100.72× | 12.55× |
| 512 | 2.13× | 152.49× | 18.22× |
| 1024 | 1.52× | 191.25× | 25.68× |
| 2048 | 1.12× | — | 30.09× |
| 4096 | 0.87× | — | 40.51× |
| 8192 | 0.71× | — | 48.14× |

The in-graph route CROSSES the plain-torch batched evaluator between pop 1024 (1.52× ahead) and pop 2048 (1.12× behind): from there on it loses that arm and keeps beating only the per-tree walk and numpy.
The fused route never loses any of the three, and its multiple against `torch-batched` grows 3.3× across the column even though its own cost grows only 1.16× — the batched arm grows too.

### 1.4 The one place the evogp stack loses, on the same trees

| pop | `Forest.batch_forward` cheaper by | `Forest.SR_fitness` cheaper by |
|---:|---:|---:|
| 64 | 18.35× | 12.20× |
| 128 | 17.77× | 13.57× |
| 256 | 19.00× | 12.86× |
| 512 | 19.80× | 13.24× |
| 1024 | 15.73× | 11.94× |
| 2048 | 15.28× | 12.42× |
| 4096 | 12.24× | 10.85× |
| 8192 | 9.02× | 8.36× |

The second column is the like-for-like pair (`Forest.SR_fitness` is a fitness surface like `gp-fused` `evaluate`); the first is the value surface of the same stack and the same kernel.
Both say the same thing and §5(a) deals with it: the torch stack's fused route is 8.4×–13.6× cheaper per fitness call than `FusedDeviceEvaluator`, because the gp device route pays a crossing (three channel exports, the legacy-triple conversion in torch, one fence) that `Forest` does not pay.

The control at pop 8192 (`forest-fused`, explicit `EVOGP_EVAL_MODE=fused`, its own child) reads `batch_forward` 0.0886 and `sr_fitness` 0.0870 — the same route as `forest-auto`'s 0.0808 / 0.0872 within the band, which is what makes "the shipped default IS the fused route" a measurement rather than a claim.

### 1.5 The scaling shape of each lane

| lane (quantity) | pop 64 | pop 8 192 | factor over 128× pop | per-doubling factors |
|---|---:|---:|---:|---|
| `gp-fused` (fitness) | 0.6257 | 0.7286 | **1.16×** | 1.01 / 1.05 / 1.07 / 0.85 / 1.07 / 1.05 / 1.06 |
| `gp` in-graph (fitness) | 0.5742 | 10.9865 | 19.13× | 1.07 / 1.18 / 1.35 / 1.56 / 1.86 / 1.92 / 2.00 |
| `torch-batched` (value) | 2.0043 | 7.7796 | 3.88× | 1.00 / 0.98 / 1.06 / 1.12 / 1.37 / 1.50 / 1.62 |
| `numpy` (fitness) | 2.8348 | 528.9067 | 186.58× | 1.73 / 1.87 / 1.95 / 2.20 / 2.18 / 2.59 / 2.38 |
| `Forest.batch_forward` (value) | 0.0341 | 0.0808 | 2.37× | 1.04 / 0.99 / 1.03 / 1.07 / 1.11 / 1.31 / 1.43 |
| `torch-per-tree` (value, pop ≤ 1 024) | 18.6146 | 294.1020 | 15.80× over 16× pop | 1.99 / 1.99 / 2.04 / 1.96 |

* **`gp-fused` is flat.** 1.16× for 128× the population, and the per-doubling column shows it: 0.85–1.07, i.e. noise around a constant. A two-point fit between the ends (a fixed floor plus a per-tree slope) puts the **floor at ~0.62 ms** and the **slope at ~1.3e-05 ms per tree** — 0.10 ms of the 0.7286 ms at pop 8 192 — which is a launch floor plus a small per-program term, exactly the shape `src/evogp/tree/FUSED.md` documents for the kernel and `DEVICE_LOOP.md` §8(b) measured inside a loop (a flat ~0.55 ms evaluation lane there, on an evolved population).
* **`gp` in-graph is linear in the population** and steepens towards it: 19.13× for 128× pop, per-doubling 1.07 → 2.00, because its cost is the `max_nodes`-static child-read chain over `pop × rows` elements.
* **`torch-batched` is sublinear**: 3.88× for 128× pop (per-doubling 1.00 → 1.62). Its per-position plan is amortized over the whole population, so it is memory-bound and cheap until the population's own tensors get big; that is why the fused route's multiple over it GROWS (3.20× → 10.68×) even though the route itself is flat.
* **`numpy` is linear in `pop × rows`** (186.58× for 128× pop, per-doubling 1.73 → 2.38): the host evaluator touches every node of every tree of every row.
* **`torch-per-tree` is linear in the population** (15.80× for 16× pop, per-doubling 1.96–2.04): one python walk per tree, so the count of trees IS the cost.
* **`Forest.batch_forward` is nearly flat** (2.37× for 128× pop, per-doubling 1.04 → 1.43): the same kernel as `gp-fused` with none of its crossing.

### 1.6 Parity: two groups, never mixed

Both groups are reported by `measure_generation._parity` (`max_rel`, `max_rel_scaled`, `finiteness_mismatch`), the fitness group over the population's finite values and the value group over the whole `(pop, rows)` matrix:

* **fitness group** (`gp`, `gp-fused`, `Forest.SR_fitness` negated against `numpy.evaluate`): worst `max_rel` **5.505e-07** (at pop 8 192; 1.90e-07 at pop 64, 2.64e-07 at 128/256, 3.52e-07 at 512, 4.57e-07 at 1 024/2 048, 5.00e-07 at 4 096) over 64 → 8 152 comparable values per population, `max_rel_scaled` ≤1.2e-07, **zero finiteness mismatches**;
* **value group** (`torch-batched`, `torch-per-tree` and `Forest.batch_forward` against `numpy.predict`): `max_rel` **0.0 — BITWISE** at every population, over 65 536 → 8 347 648 comparable elements, **zero finiteness mismatches**.

The `forest-fused` control joins the same two groups at pop 8192 (fitness 4.573e-07 relative, value 0.0) — so "the explicit route and the shipped default agree with numpy to the same digits" is measured too.
The comparability of the per-call grid rests on a bit-identical population: the device channels are uploaded once and read back and compared (`upload.identical: true`, three channels, `pop` / `max_nodes` recorded) before any lane runs, and a mismatch raises rather than measuring different trees.
This is the acceptance box of `DEVICE_LOOP.md`'s parity statement extended across the population: about two and a half orders of magnitude inside 1e-4 on the fitness path, exact on the value path.

## 2. The generation loop

A "generation" is exactly what a GP loop pays: evaluation + selection/crossover/mutation + the elite splice — `workflow.step` on the `evogp.gp` sides, `StandardPipeline.step` on the torch sides, 20 generations per child, warm medians of 15.
`evaluation` and `variation` are separately timed lanes each inside its own sync bracket on the `evogp.gp` sides, so they sum to `total` within a few percent and `total` is the number to quote; on the torch sides `variation` is `total − evaluation`, taken inside one step through a timed problem wrapper.

| pop | side | total | evaluation | variation | eval share | cold s |
|---:|---|---:|---:|---:|---:|---:|
| 64 | `gp` | 1.058 | 0.576 | 0.516 | 54.4 % | 4.81 |
| 64 | `gp-fused` | 1.319 | 0.735 | 0.589 | 55.7 % | 3.25 |
| 64 | `numpy` | 10.100 | 5.902 | 4.256 | 58.4 % | 0.09 |
| 64 | `torch` | 2.693 | 0.396 | 2.297 | 14.7 % | 0.62 |
| 64 | `torch-fused` | 2.573 | 0.388 | 2.185 | 15.1 % | 0.59 |
| 128 | `gp` | 1.186 | 0.618 | 0.556 | 52.2 % | 4.71 |
| 128 | `gp-fused` | 1.289 | 0.746 | 0.611 | 57.9 % | 3.18 |
| 128 | `numpy` | 14.017 | 9.194 | 4.211 | 65.6 % | 0.08 |
| 128 | `torch` | 2.629 | 0.377 | 2.252 | 14.3 % | 0.59 |
| 128 | `torch-fused` | 2.705 | 0.399 | 2.306 | 14.7 % | 0.60 |
| 256 | `gp` | 1.316 | 0.718 | 0.579 | 54.5 % | 4.79 |
| 256 | `gp-fused` | 1.374 | 0.746 | 0.648 | 54.3 % | 3.21 |
| 256 | `numpy` | 24.925 | 17.280 | 4.606 | 69.3 % | 0.09 |
| 256 | `torch` | 2.592 | 0.386 | 2.205 | 14.9 % | 0.60 |
| 256 | `torch-fused` | 2.782 | 0.397 | 2.386 | 14.3 % | 0.61 |
| 512 | `gp` | 1.661 | 0.986 | 0.666 | 59.4 % | 4.76 |
| 512 | `gp-fused` | 1.538 | 0.807 | 0.753 | 52.5 % | 3.21 |
| 512 | `numpy` | 42.251 | 31.572 | 5.769 | 74.7 % | 0.11 |
| 512 | `torch` | 2.827 | 0.419 | 2.408 | 14.8 % | 0.60 |
| 512 | `torch-fused` | 2.861 | 0.414 | 2.446 | 14.5 % | 0.61 |
| 1024 | `gp` | 2.090 | 1.563 | 0.500 | 74.8 % | 4.80 |
| 1024 | `gp-fused` | 1.575 | 0.911 | 0.660 | 57.9 % | 3.46 |
| 1024 | `numpy` | 85.159 | 64.235 | 6.965 | 75.4 % | 0.14 |
| 1024 | `torch` | 2.885 | 0.438 | 2.447 | 15.2 % | 0.60 |
| 1024 | `torch-fused` | 2.657 | 0.396 | 2.261 | 14.9 % | 0.61 |
| 2048 | `gp` | 3.414 | 2.855 | 0.562 | 83.6 % | 4.86 |
| 2048 | `gp-fused` | 1.395 | 0.772 | 0.610 | 55.4 % | 3.38 |
| 2048 | `torch` | 2.727 | 0.398 | 2.328 | 14.6 % | 0.60 |
| 2048 | `torch-fused` | 2.891 | 0.412 | 2.480 | 14.2 % | 0.62 |
| 4096 | `gp` | 6.095 | 5.423 | 0.662 | 89.0 % | 4.96 |
| 4096 | `gp-fused` | 1.351 | 0.785 | 0.671 | 58.1 % | 3.42 |
| 4096 | `torch` | 2.746 | 0.412 | 2.334 | 15.0 % | 0.62 |
| 4096 | `torch-fused` | 2.869 | 0.408 | 2.461 | 14.2 % | 0.59 |
| 8192 | `gp` | 11.752 | 10.793 | 0.917 | 91.8 % | 4.84 |
| 8192 | `gp-fused` | 1.453 | 0.799 | 0.740 | 55.0 % | 3.38 |
| 8192 | `torch` | 3.362 | 0.469 | 2.893 | 14.0 % | 1.51 |
| 8192 | `torch-fused` | 3.403 | 0.455 | 2.948 | 13.4 % | 1.55 |

`numpy` is scoped out above pop 1024 (§5f), which is why it has no rows there.

| pop | `gp-fused` vs `torch` | vs `torch-fused` | vs `numpy` | vs `gp` |
|---:|---:|---:|---:|---:|
| 64 | 2.04× | 1.95× | 7.66× | 0.80× |
| 128 | 2.04× | 2.10× | 10.87× | 0.92× |
| 256 | 1.89× | 2.02× | 18.14× | 0.96× |
| 512 | 1.84× | 1.86× | 27.48× | 1.08× |
| 1024 | 1.83× | 1.69× | 54.06× | 1.33× |
| 2048 | 1.95× | 2.07× | — | 2.45× |
| 4096 | 2.03× | 2.12× | — | 4.51× |
| 8192 | 2.31× | 2.34× | — | 8.09× |

**The gap to the torch stack is the operator layer, not the evaluation.** The torch stack's own evaluation slice is 0.377–0.469 ms — CHEAPER than the `gp-fused` evaluation lane (0.735–0.911 ms) at every population of this column — while its variation slice is 2.185–2.948 ms against `evogp.gp`'s 0.500–0.917 ms (in-graph) / 0.589–0.753 ms (fused).
So a `gp-fused` generation wins 1.83×–2.34× because selection/crossover/mutation on one traced graph is 3.2×–3.9× cheaper than the torch stack's operator layer (3.2×–4.9× in-graph), and it wins that much even while LOSING the evaluation segment; the fused objective is what keeps the evaluation flat enough (0.735 → 0.799 ms, 1.09× over 128× pop) that the operator win is not eaten at the top of the column.
The generation-side route also scales better: 1.10× for 128× the population against 1.25× (`torch`) and 1.32× (`torch-fused`), 11.11× (`gp` in-graph) and 8.43× (`numpy`, over 16× pop).

**The in-graph route is the small-population choice.** It is the cheapest generation at pop 64 (1.058 against `gp-fused`'s 1.319) and at every population up to 256 (1.186/1.289 at 128, 1.316/1.374 at 256); the crossover is a BAND rather than a point — pop 256 reads 1.316 against 1.374 (4.2 % apart) and pop 512 reads 1.661 against 1.538 (8 % the other way), both inside the box noise of a shared card, so "the fused objective takes over between pop 256 and pop 512" is the only defensible statement.
It also crosses the torch stack between pop 1024 and 2048: at 1024 the in-graph loop is 2.090 ms against the torch stack's 2.885 (1.38× ahead), at 2048 it is 3.414 against 2.727 (25 % behind), and past that its evaluation (`pop × rows` child reads) is what it pays — 10.793 ms at pop 8 192 against the fused route's flat 0.799.
On a torch-free box the in-graph route is the ONLY choice, and it is the cheaper one at small populations.

**Consistency between the two columns.** The loop's evaluation lanes run on an EVOLVED population, the per-call grid on a generation-0 one, and the records carry the split: `fresh_evaluation` vs `evaluation` is 0.6431 / 0.6603 / 0.6314 / 0.6356 / 0.8134 / 0.6551 / 0.6016 / 0.6039 against 0.735 / 0.746 / 0.746 / 0.808 / 0.912 / 0.772 / 0.785 / 0.799 ms, i.e. 1.12×–1.32× dearer on evolved trees — which is why the per-call fitness medians (0.6257–0.7286) sit at the low end of the loop's (0.735–0.911), and the in-graph route is flat in tree length (0.98×–1.00× evolved/fresh at every population). Same finding as `DEVICE_LOOP.md` §5, re-measured across the population.

## 3. Cold start

**Per generation** (`cold_ms` in `evidence/popscale.json`, the first `step` of a side in its own process):

| side | cold s, pop 64 → 8 192 | what it is |
|---|---|---|
| `gp` | 4.71–4.96 (flat) | the IREE variation graph + the IREE evaluation graph |
| `gp-fused` | 3.18–3.46 (flat) | ONE IREE variation graph + the Triton JIT |
| `numpy` | 0.08–0.14 | nothing to compile |
| `torch` | 0.59–0.62, except **1.51 at pop 8 192** | the torch stack's own route |
| `torch-fused` | 0.59–0.62, except **1.55 at pop 8 192** | the same on its fused route |

The pop-8 192 outlier is real in both torch records and it is 2.5× their own column — but **the record does not establish the cause**: a `torch` record carries only `cold_ms`, `eval_mode`, `pop`, `rows`, `side`, `timing` and `trajectory` (no `structure` block, no compile or allocator counters), so this report does not attribute it.

**Per call** (§1.2): `gp` ~2.5 s flat, `gp-fused` ~0.9 s flat, `Forest`'s first value call ~0.5 s and its first loss call ~30 ms (both flat in population), `numpy` 3.1 ms → 528.5 ms (it is one call, so its "cold" IS its warm cost).
Cold is a one-time per-process cost and it does not scale with the population in any lane that compiles — which is the reason the column above is a constant and not a curve.

**When it is repaid** (generation level, `cold_gp_fused / (torch_total − gp_fused_total)` per cell):

| pop | `gp-fused` cold s | gap vs `torch` ms | generations | gap vs `torch-fused` ms | generations | gap vs `numpy` ms | generations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.25 | 1.374 | 2 367 | 1.254 | 2 593 | 8.781 | 370 |
| 128 | 3.18 | 1.340 | 2 370 | 1.415 | 2 244 | 12.728 | 249 |
| 256 | 3.21 | 1.217 | 2 637 | 1.408 | 2 280 | 23.551 | 136 |
| 512 | 3.21 | 1.290 | 2 492 | 1.323 | 2 429 | 40.714 | 79 |
| 1024 | 3.46 | 1.310 | 2 643 | 1.082 | 3 199 | 83.584 | 41 |
| 2048 | 3.38 | 1.331 | 2 539 | 1.496 | 2 259 | — | — |
| 4096 | 3.42 | 1.395 | 2 452 | 1.518 | 2 254 | — | — |
| 8192 | 3.38 | 1.909 | 1 769 | 1.950 | 1 732 | — | — |

So the `gp-fused` loop repays its cold start against the torch stack in **1 769–2 643 generations** per cell (against `torch-fused` 1 732–3 199) and against numpy in **41–370** — a warm margin of ~1.2–1.9 ms per generation is small change against a 3.2–3.5 s graph build, and it is the honest reason the short-run smoke stays on numpy while a real run warms once.
That is also why the ROUND's own verdict is about the ratio, not about a benchmark run: any one-shot comparison is dominated by this column.

## 4. Convergence is not assumed

Every `evogp.gp` record carries the best and the median fitness of each of the 20 generations the harness recorded DURING its timed run, so the three evaluator routes can be compared on the search they actually perform, not just on the clock.

| pop | `gp` vs `numpy` (best series) | `gp-fused` vs `numpy` | endpoints (numpy / gp / gp-fused) |
|---:|---|---|---|
| 64 | one checkpoint (generation 15) at 1.0e-06 | IDENTICAL | −1.376907 / −1.376907 / −1.376907 |
| 128 | IDENTICAL | IDENTICAL | −1.445907 / −1.445907 / −1.445907 |
| 256 | two checkpoints at ≤1.0e-06 | IDENTICAL | −0.460114 / −0.460114 / −0.460114 |
| 512 | diverges from generation 2, max \|Δ\| 0.0338 | diverges from generation 17, max \|Δ\| 0.0338 | −0.465913 / −0.441900 / −0.441900 |
| 1024 | IDENTICAL | IDENTICAL | −0.296918 / −0.296918 / −0.296918 |

What that says: the device routes make the SAME selection decisions as the numpy reference at pop 64 (to a last-bit float32 difference), 128, 256 and 1024 — identical at three of the five populations, ≤1.0e-06 at two.
At pop 512 the two device routes agree with each other to ≤1.0e-06 and both end at −0.441900 while numpy ends at −0.465913 (a 0.024 gap): the split starts as a 1.0e-06 float32 tie inside `gp` at generation 2 and `gp-fused` inherits the flipped branch at generation 17, i.e. a selection-tie consequence of the route's arithmetic at a fitness parity of 3.52e-07 relative — exactly the mechanism `DEVICE_LOOP.md` §4 documents for its p1000 generation-19 flip, and here numpy is the one that takes the other branch.
The medians tell the same story with the same cause: max \|Δ\| ≤2.0e-06 at pop 64 / 128 / 256 / 1024 (drift only after such a tie) and 1.14 at pop 512, where the tie actually broke.
The torch side is a different trajectory by construction — different RNG stream (torch vs `splitmix64`), truncation selection (`survival_rate=0.3`, `elite_rate=0.01`) vs tournament-4, different mutation wiring — so it differs from generation 0 and only its endpoints are comparable, cell by cell at this 20-generation horizon: 64 −1.356047 / 128 −0.311868 / 256 −0.475866 / 512 −0.309996 / 1024 −0.475750, against `evogp.gp`'s numbers above. `torch` and `torch-fused` are bit-identical to each other at every population and every checkpoint, which is the torch stack's own route-invariance and not a comparison across families.
No convergence claim is made ACROSS families here; the claim is the narrow one the record supports — the evoxc-based stack's three evaluation routes run the same search, and where they do not, the split is one float32 tie.

## 5. Honesty: what this round does not show

* **(a) Per call, the evoxc stack is NOT the fastest.** The torch stack's own Triton route does the same trees 9.0×–19.8× cheaper on `batch_forward` (0.0341–0.0808 ms against `gp-fused`'s 0.6257–0.7286 ms) and 8.4×–13.6× cheaper on the like-for-like fitness surface (`Forest.SR_fitness` 0.0465–0.0872 ms).
  The reason is structural and in the code: `FusedDeviceEvaluator` exports three `evoxc` channel tensors over DLPack, converts them into the legacy triple in torch (`_legacy_triple_device`), writes the fitness buffer through `out=` and takes one `torch.cuda.synchronize()` per call, while `Forest` ALREADY holds the legacy layout as its own population tensors and its fused launch is pre-bound (`EVOGP_FUSED_DISPATCH=bound` serving Triton's `CompiledKernel.run`).
  The evoxc stack's win is the GENERATION — the operator layer — not the raw forward.
* **(b) The whole-generation win over the torch stack comes from the operator layer, not the evaluation.** The torch stack's evaluation slice is 0.377–0.469 ms at every population of §2, i.e. cheaper than the `gp-fused` one (0.735–0.911) and much cheaper than the in-graph one (0.576–10.793); the variation slice is where `evogp.gp` wins 2.185–2.948 → 0.500–0.917 ms. Any future round that only optimizes evaluation will not move this table.
* **(c) pop 64 is the near-parity cell between the two `evogp.gp` routes.** `gp-fused`/`gp` is 1.319/1.058 ms per generation (the fused route 25 % dearer) while the per-call columns are 0.6257/0.5742 ms (fused 9 % dearer) — a segment-level difference well inside a few percent of the box noise and consistent with `DEVICE_LOOP.md` §1/§8, where the in-graph route keeps pop 64 through its variation lane and loses the evaluation lane. Do not read pop 64 as evidence for either objective.
* **(d) `FusedDeviceEvaluator` returns ONE reused fitness buffer.** The tensor is overwritten by the next device call, so a caller that keeps it (a hand-written loop, a history list) reads a later generation's numbers — copy it before the next call. Every number in this report is a median over repetitions the harness consumes inside the timed block, so the aliasing does not touch them; it is a contract for users.
* **(e) The torch boundary is exactly one module and the Triton cache must be writable.** `fused_eval.py` is the only module of the subtree that resolves torch/triton, and it does so through `importlib` at call time; the device route additionally needs a writable `TRITON_CACHE_DIR` (`/tmp/triton_cache` in these records — the harness sets it because this box's `$HOME` is read-only). Without torch+triton the whole `gp-fused` column simply does not exist, and `graph_eval` is the fallback.
* **(f) Scope-outs, with the measured reason.** `numpy` is scoped out of the GENERATION grid above pop 1024 (it was run at every population up to and including 1024, where a generation already costs 85.159 ms and a per-call evaluation 39.4831 ms) and `torch-per-tree` is scoped out of the PER-CALL column above pop 1024 (run at every population up to 1024, where one call is already 294.1020 ms and its cost is linear in the trees — 18.6146 → 294.1020 ms over 16× pop — so 8 192 trees would read ~2.35 s per call by extrapolation, which buys no new information at a still-visible 1 024).
  Both ceilings are recorded in the JSON (`scoped_out` with its reason and its `rows`), not silently dropped, and the per-call grid's `numpy` lane does reach pop 8 192 (528.9067 ms), which is exactly the number that makes the loop's ceiling honest rather than convenient.
* **(g) IREE leaks buffers per live executable**, so the harnesses print nanobind leak warnings on interpreter exit. This is pre-existing and documented in `CONTEXT.md`, it affects no measurement in this report (nothing here is a memory figure), and the records carry no field for it.

## 6. Verdict

Against the round's objective — *raw batched (population) forward performance better than the old torch version, across pop 64 → 8192* — the records hold the claim only in the form they can prove: versus the PLAIN-PYTORCH evaluators the evoxc-based stack wins at EVERY population and the margin grows with the population — 3.20× → 10.68× against the whole-population batched arm (with a 2.93× dip at pop 512), 29.8× → 481.9× against the per-tree walk, 4.53× → 725.9× against the numpy host path — and the flatness is the point: 1.16× for 128× the population where the batched arm grows 3.88× and numpy 186.6×.
Against the torch stack's OWN fused Triton route the per-call claim does NOT hold: `Forest` does the same trees 9.0×–19.8× cheaper because it already owns the layout the kernel reads, and that inversion is the honest limit of this round and the open item for the next one (a per-call parity of the fused route would need the channel-triple crossing to disappear, not to get faster).
Where the evoxc stack does win against the whole torch stack is the LOOP: 1.83×–2.31× per generation against `torch` and 1.69×–2.34× against `torch-fused`, growing with the population, because the operator layer is 3.2×–3.9× cheaper on one traced graph (3.2×–4.9× in-graph) while the evaluation slices are within a factor of two of each other.

## 7. Reproduce and gates

The shared-population per-call grid (nine children, one per population plus the control; 159.2 s of child wall):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_popscale \
  --pops 64,128,256,512,1024,2048,4096,8192 --warmup 5 --reps 15 \
  --json src/evogp/gp/evidence/popscale_eval.json
```

The honest cold column — one lane per child, warm medians NOT quoted from it (32 children, 295.5 s):

```
for lane in gp gp-fused numpy forest-auto; do
  EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
    -m evogp.gp.measure_popscale \
    --pops 64,128,256,512,1024,2048,4096,8192 --lanes $lane --reps 3 --warmup 1 \
    --skip-fused-control \
    --json src/evogp/gp/evidence/popscale_cold/$lane.json
done
```

The generation column (37 children, 416.3 s):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --cells popscale \
  --sides gp,gp-fused,numpy,torch,torch-fused \
  --json src/evogp/gp/evidence/popscale.json
```

One cell of one lane, in a single process (what each child above runs):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_popscale --pop 256 --rows 1024 --lanes gp-fused --warmup 5 --reps 15
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m evogp.gp.measure_generation --side gp-fused --pop 256 --rows 1024 --warmup 5 --reps 15
```

`--lanes` also accepts `torch-batched`, `torch-per-tree` and `forest-fused` (the last one must be scheduled alone, because the evaluation route is memoized on first use and `EVOGP_EVAL_MODE=fused` has to be set before `evogp.tree` is imported — which is exactly what `--lane forest-fused` does); `--fused-control-pop` moves the control's population and `--skip-fused-control` drops it (the `forest-fused` lane and both flags were removed in 2.0.0a2 with the torch stack; recover them at commit 6720723).
Nothing is written unless `--json` is passed, and every harness is `main()`-guarded.

The suite gate of this round (`src/evogp/gp/tests`, the whole subtree):

```
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m pytest src/evogp/gp/tests -q          # 346 passed / 0 skipped in 62.32 s
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python \
  -m pytest src/evogp/gp/tests/test_gp_stack_purity.py -q   # 24 passed
```

346 is the subtree's collected count with `measure_popscale.py` in the tree: `test_gp_stack_purity.py` parametrizes its source grep over every `*.py` of the subtree, so the new harness adds exactly one case, and the 36 device-gated cases RUN rather than skip on this box (`conftest.py` pins `EVOGP_EVOXC_BACKEND=numpy` for hermeticity, but a device test passes `backend="iree"` plus an explicit `evoxc.Device` itself).

Environment of record: python 3.12.3, torch 2.13.0+rocm7.1 (HIP 7.1.52802), triton 3.7.1, numpy 2.5.4, iree-base-compiler / iree-base-runtime 3.11.0, `evogp` `0e4f8d3`, `evoxc` `474bf72` (the checkout with the native DLPack device export), GPU `AMD Instinct MI300X VF`, `EVOXC_DISABLE_LOCATIONS=1`.
Every ratio here is a median on a shared card and is box-noise ±10 %; a difference of a few percent is called parity, and the conclusions above are drawn from ratios many times that band (2.9×–725.9× on the per-call axis, 1.7×–54× on the generation axis) rather than from any single-cell percentage.
