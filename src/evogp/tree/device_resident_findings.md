# Device-resident dense hot path — findings

Measurement + evidence record for the **device-resident** work on
`dense_engine.py`: the SR loss is folded into the compiled dense graph and the
per-call host round trips of the dense route are counted and reduced.

The reproducible battery is `device_resident_check.py` (in this directory; a dev
CLI, NOT pytest-collected).  Every number below was produced from the repository
root with the project venv and

```bash
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
  <venv>/bin/python src/evogp/tree/device_resident_check.py
```

## 1. Round 1 — the pre-refactor capture (FAILING)

Command (same as above, plus `--json /tmp/device_resident_before.json`):

```
== FAILED: 39/60 checks passed ==
```

The red checks are exactly the round trips this work removes:

| check | pre-refactor observation |
| --- | --- |
| `inputs converted to host exactly ONCE` | **2** conversions (`inputs` + `x`: `sr_fitness_dense` converts the inputs, then `evaluate_dense` converts them again) |
| `every readback is (P,)-scale` (unchunked) | readback `[8, 37]` — the whole prediction, 296 elements for a population of 8 |
| `no host->torch tensor larger than pop_size` | `torch.from_numpy` on `[8, 37, 1]` (296 elements) |
| `no labels round trip through a torch tensor` | `torch.from_numpy` on `[37, 1]` after a D2H, re-uploaded with `.to(device)` |
| `x prepared ONCE per n-chunk` (both-chunked) | **296** preparations for 8x37 chunks (= one per `(p0, n0)` chunk; `_prep_columns` is memoized per `p0`, the x slice is not) |
| `labels prepared ONCE per n-chunk` | **0** — `labels` never becomes a graph payload at all (torch-side reduction) |
| `resolve_eval_mode('dynamic') -> 'dense'` (+5 more policy checks) | `ValueError: unsupported eval mode 'dynamic'` |
| `the readback-free run helper exists` | `dense_engine._iter_plan_chunks` missing: every chunk result is read back inside `run_plan` |

`stage parity` was already green (13/13): the dense *value* route matches the
specialized engine within `rtol=atol=1e-4` with exact NaN/±inf masks
(`non_bit_exact_finite` 0–7, `max_rel` <= 2e-7), and that stays the bar for the
in-graph loss.

## 2. What still crosses the host (per call)

See §4 (after) for the finalised list per backend.

## 3. Measurement protocol

Warm per-call medians from `device_resident_check.py --measure`: each case is
called once untimed (build/compile + cache warm), then `--repeats` timed calls
with a `torch.cuda.synchronize()` around each sample, and finally one *counted*
call whose `Traffic` summary is recorded next to the timing.  The before/after
numbers of §4 were taken from the same script at the same grid on the same box
(see §5 for the exact before run).

## 4. Round 2 — after (see §5)
