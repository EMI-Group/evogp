# Numeric validation and cross-arm digest agreement (final campaign)

Companion to [`FINAL.md`](FINAL.md) §6: this file holds the per-checksum evidence behind that section.
Source: `results/final/validation.json`, one sweep on the same session/box as the campaign (`env.tree_root_sha` `761ef05c…`, `evoxc_sha` `855050d7…`, AMD Instinct MI300X VF).

## Oracle validation (`validate.py`)

Oracle: `../src/evogp/tests/reference_eval.py` (plain torch, CPU).
12 cases = `arith` / `rich` / `ternary` × the four scheduled cells `p16-L32-i8-N1024`, `p256-L64-i32-N1024`, `p1000-L64-i32-N1024`, `p256-L64-i32-N64`; 16 / 64 / 64 / 64 trees compared per case (the CPU oracle sees the sliced population the record reports).
The `engine` column is the `auto` route as it resolves per cell: specialized+numpy at `p16-L32-i8-N1024`, dense above the `pop >= 64` threshold on the other three.

Worst deviation over the four cells, per function set and workload:

| function set | workload | engine vs oracle | `torch-per-tree` vs oracle | `torch-batched` vs oracle |
|---|---|---|---|---|
| `arith` (`+ - * /`) | `batch_forward` | max abs `0.0`, `bitwise_equal true` | `0.0`, bitwise | `0.0`, bitwise |
| `arith` | `forward` | `0.0`, bitwise | `0.0`, bitwise | `0.0`, bitwise |
| `arith` | `sr_fitness` | abs `2.56e+02` / rel `3.4e-07`, not bitwise | abs `2.56e+02` / rel `2.3e-07` | abs `2.56e+02` / rel `2.2e-07` |
| `rich` (14 ops incl. transcendentals) | `batch_forward` | abs `1.61e+00` / rel `1.56e+01`, not bitwise | abs `1.42e+00` / rel `2.30e+00` | same as `torch-per-tree` |
| `rich` | `forward` | `0.0`, not bitwise | abs `2.98e-08` / rel `8.10e-08` | same as `torch-per-tree` |
| `rich` | `sr_fitness` | abs `2.50e-01` / rel `2.95e-07` | abs `2.50e-01` / rel `1.65e-05` | same as `torch-per-tree` |
| `ternary` (`if + - * /`) | `batch_forward` / `forward` | `0.0`, bitwise | `0.0`, bitwise | `0.0`, bitwise |
| `ternary` | `sr_fitness` | abs `1.56e-02` / rel `2.08e-07`, not bitwise | abs `1.56e-02` / rel `2.08e-07` | same as `torch-per-tree` |

Read of record:

* **The value path (`batch_forward`, `forward`) is bitwise exact** for `arith` and `ternary` in every cell: `max_abs 0.0` with `bitwise_equal true` for all three arms.
* **`sr_fitness`** agrees to float32 reduction order only: relative ≤ `~3.5e-07`, absolute up to `2.56e+02` where the reference losses themselves are large (`6.400e+01` on `p256-L64-i32-N64`). NaN/Inf placement mismatched nowhere in the sweep (0 mismatches in every check).
* **`rich` is the only set with real spread.** At `p16-L32-i8-N1024` the torch arms deviate by `9.351e-02` abs / `1.493e-04` rel on `batch_forward` while the engine column (specialized+numpy at that cell) reads `0.000e+00`; at `p256-L64-i32-N64` the engine (dense at that cell) reads `4.369e-01` abs / `6.398e-01` rel against the oracle on `batch_forward` where both torch arms read `5.245e-06`, and its own `forward` on the same cell is `0.000e+00`.
  The attribution is the dense/IREE lowering of transcendentals at single points of a population whose reference magnitudes reach `1.0e+09`, not a wrong tree evaluation: the plain-arithmetic set is bitwise exact everywhere, and the spread never appears on two workloads at once.
  Practical reading: on the default route a transcendental-heavy tree can differ from plain torch by `~1e-1` absolute in the tail; the default SR set (`+ - * /`) stays exact, so this does not touch the SR training path.
* **Bit-pattern-only differences sit outside the bitwise claim.** At `p16-L32-i8-N1024` the engine's `rich` rows report `bitwise_equal false` at `max_abs 0.000e+00` (a bit-pattern-level difference with matching NaN/Inf placement), which is why the bitwise statement is scoped to `arith` and `ternary`.

## Cross-arm digest agreement (final campaign)

`check.py` compares the `forward` / `batch_forward` sha256 across arms per cell (`sr_fitness` digests legitimately differ by arm, so they are not compared).
On the final records it reports 28 cross-arm spreads, and they form two distinct findings:

* **The per-tree-structure specialized IREE graph is the bit-level outlier.** At `pop 16` and `pop 64` `evoxc-specialized-iree` hashes alone (`p64-L64-i8-N1024/batch_forward`: `d1fd2ddb…` for that arm, `960a58af…` for the other six), and at `pop 16` `evoxc-auto-iree` joins it because `auto` + iree resolves to the specialized route below the threshold. The candidate causes stay as recorded earlier: a different reduction/dispatch order inside one compiled per-structure graph (float32 reassociation) or a per-structure elementwise lowering detail.
* **At `pop 1000` the dense route forms its own group.** `evoxc-auto`, `evoxc-auto-iree` and `evoxc-dense-iree` share one sha256 and `evoxc-specialized-numpy` + both torch arms share another, on both `forward` and `batch_forward`, in all four `pop 1000` cells (e.g. `p1000-L64-i32-N1024/batch_forward`: `31fa45e2…` vs `bf68ce50…`). At `pop 256` all six scheduled arms still hash together, and at `pop 64` everything except `evoxc-specialized-iree` agrees.

Practical relevance of the specialized-iree divergence is small: the default policy routes `pop >= 64` to dense+iree, so the divergent arm is off the default path and only matters if someone forces `EVOGP_EVAL_MODE=specialized` together with `EVOGP_EVOXC_BACKEND=iree`.
Coverage limits: `evoxc-specialized-iree` is measured on the `pop ∈ {16, 64}` cells only (272–284 ms of compile per tree structure makes a 256/1000-tree population unaffordable — the same compile-cost rule that keeps it off the device default), and the sweep covers neither `CustomLoss`/multi-output models nor the robotics problems.

Two practical notes for anyone re-running the sweep: `validate.py` compares the engine against the oracle *and* against both torch arms in the same record (so a torch-side deviation cannot masquerade as an engine bug), and the `p16` case is the only one that exercises the specialized+numpy route — the three `pop >= 64` cells exercise the dense route, which is the route the transcendental-spread finding belongs to.
