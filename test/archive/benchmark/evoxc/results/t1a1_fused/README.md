# `t1a1_fused/` — the fused (Triton) route's campaign of record

Evidence for `EVOGP_EVAL_MODE=fused` (`src/evogp/tree/fused_engine.py`), measured through this harness's own
public-API workloads on the same box, protocol and arms as `../t1a1_baseline/` — the pre-fused reference
baseline this round is read against. The route's design and report document is `src/evogp/tree/FUSED.md`;
its arms are now entries of the harness registry (`common.EVOXC_ARMS`: `evoxc-fused-iree`, `evoxc-fused`).

## What was measured
One process per `(cell, arm)`, `seed 0`, `common.build_workload(cell)`, `warmup 2` untimed, `reps 5`
sync-bracketed warm samples (medians), the cold first call reported separately, idle-device guard per config
(20/20 idle), `EVOXC_DISABLE_LOCATIONS=1`. Workloads: `sr_fitness`, `batch_forward`, `forward`, `gp_step`.

Arms per cell: `evoxc-fused-iree` (the route), `torch-batched` (the baseline the ratios are taken against),
plus `evoxc-dense-iree` (4 cells), `evoxc-dense-iree-raised` (`d7` only, the plain dense arm is not scheduled
there — it OOM-kills) and `evoxc-batch-iree` (3 cells) as the shipped alternatives.

| cell | pop | tree len | input len | rows | trees |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 256 | 32 | 8 | 1 024 | depth 5 |
| `p256-L64-i32-N1024` | 256 | 64 | 32 | 1 024 | depth 5 |
| `p256-L64-i32-N4096` | 256 | 64 | 32 | 4 096 | depth 5 |
| `p1000-L64-i32-N1024` | 1 000 | 64 | 32 | 1 024 | depth 5 |
| `p4096-L64-i32-N1024` | 4 096 | 64 | 32 | 1 024 | depth 5 |
| `p2048-L128-i32-N1024-d7` | 2 048 | 128 | 32 | 1 024 | depth 7 (123-node trees) |

## Verdict in one line
The fused route is **41×–157× faster than `torch-batched` on `batch_forward`** (32×–131× on `sr_fitness`)
across the six cells, and its value paths are **bitwise identical** to `torch-batched` and to the plain-torch
oracle (`max_abs = 0.0` on the finite space; the only differing bits are NaN payloads, which is why this
directory needs `--allow-digest-mismatch`). Ratios are `torch-batched median / fused median` (baked from the
per-cell records above, not quoted from the report).

## Reproduce
Harness registration is in place, so the round is a plain campaign (one process per config, GPU box):

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache     # the box's $HOME is read-only
cd test/benchmark/evoxc
$PY campaign.py --out results/t1a1_fused/main \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-fused-iree,torch-batched,evoxc-dense-iree,evoxc-batch-iree \
    --workloads sr_fitness,batch_forward,forward,gp_step
```
`fused_campaign.py` is the driver this round was actually run with (it registered the two arms before they
lived in `common.py`); it is a no-op today and still works against a registry that lacks them.
Gate of record (**20 records / 580 checks / 0 failures**):

```bash
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_fused \
    --allow-snapshot --allow-missing --allow-digest-mismatch
```

## Layout
| path | what it is |
|---|---|
| `main/<cell>/<arm>.json` | harness-schema timing records (workloads, medians, samples, cold calls, digests, routes, `gp_step`) |
| `main/campaign.json` | the merged index of the four driver invocations (`campaign.{A,B1,B2,C}.json`) |
| `campaign.json`, `campaign.md` | the one-row-per cell × arm × workload index (77 entries, 0 failed) and its tables |
| `probes/<cell>.json` | route proof, served-call counter deltas, numerics vs arm and oracle, stage split, cold calls, GP loop |
| `cold/<cell>.json` | cold-Triton-cache first calls (one fresh cache dir per cell) |
| `validation-fused.json` | `validate.py` output (18 cases) |
| `envelope.json`, `minpop.json`, `degenerate.json`, `ENVELOPE.md` | the second campaign: memory/population envelope, the `EVOGP_FUSED_MIN_POP` basis, geometry sweep, multi-output path |
| `sweep/sweep.json` | the block/warp geometry sweep |
| `fused_campaign.py`, `fused_gate.py`, `fused_probe.py`, `campaign_index.py` | the tooling that produced the above |

## Caveat
These records predate the harness's fused counter surface, so each `evoxc-fused-iree` record carries
`routes.sr_fitness.eval_mode = "fused"` but no `engine` and no `fused_cache` block: the gate WARNs that they
cannot PROVE a Triton launch (the route compiles no evoxc graph, so `compile_misses_first_call` is 0 by
construction). They are valid timing records — the launch proof for a re-run comes from
`workloads[*].cache.fused_served` / `fused_cache.served_every_workload`.
