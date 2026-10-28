# Numeric validation and cross-arm digest agreement (baseline campaign)

## Oracle validation (`validate.py`, run before the campaign)
Oracle: `src/evogp/tests/reference_eval.py` (plain torch, CPU). Cell `p16-L32-i8-N1024`,
16 trees compared per function set, arms `engine` (route `auto`/numpy), `torch-per-tree`,
`torch-batched`.

| function set | workload | engine vs oracle | torch-per-tree vs oracle | torch-batched vs oracle |
|---|---|---|---|---|
| `arith` (`+ - * /`) | `batch_forward` | max abs `0.000e+00`, bitwise | max abs `0.000e+00`, bitwise | max abs `0.000e+00`, bitwise |
| `arith` | `forward` | `0.000e+00`, bitwise | `0.000e+00`, bitwise | `0.000e+00`, bitwise |
| `arith` | `sr_fitness` | max abs `1.25e-01` / rel `1.6e-07` | max abs `2.05e+03` / rel `2.5e-07` | max abs `2.05e+03` / rel `2.5e-07` |
| `rich` (14 ops incl. transcendentals) | `batch_forward` | `0.000e+00` | abs `1.5e-05` / rel `5.8e-05` | abs `1.5e-05` / rel `5.8e-05` |
| `rich` | `forward` | `0.000e+00` | abs `1.2e-07` / rel `1.7e-07` | `1.2e-07` / `1.7e-07` |
| `ternary` (`if + - * /`) | `batch_forward` / `forward` | `0.000e+00`, bitwise | `0.000e+00`, bitwise | `0.000e+00`, bitwise |
| `ternary` | `sr_fitness` | rel `1.2e-07` | max abs `6.1e-05` / rel `1.2e-07` | abs `6.0e-08` / rel `1.2e-07` |

So: the value path is **bit-exact** across the oracle, both torch baselines and the
engine on the arithmetic (campaign) function set; the `sr_fitness` residual is
float32 reduction order (engine reduces on the host in numpy, the arms in torch), and
`rich` shows the expected ~1e-5 transcendental-kernel spread. NaN/Inf placement never
mismatched (0 mismatches in every check).

## Cross-arm digest agreement in the campaign (`check.py`)
`check.py` reports 39 failures out of 3472 checks; all of them are one thing: on the
cells where it ran, the **`evoxc-specialized-iree` route returns different bits** than
every other arm (`evoxc-auto`, `evoxc-auto-iree`, `evoxc-specialized-numpy`,
`evoxc-dense-iree`, `torch-per-tree`, `torch-batched` — these six always agree
bit-for-bit). Example: `p64-L64-i8-N1024/batch_forward` has sha `6e8fa35c…` for those
six arms and `e12861e3…` for `evoxc-specialized-iree`.
`evoxc-auto-iree` and `evoxc-dense-iree` are numerically identical to the torch arms, so
this is specific to the per-tree-structure specialized IREE graph, not to IREE: the
candidate causes are a different reduction/dispatch order inside one compiled
per-structure graph (float32 reassociation), or a per-structure elementwise lowering
detail. It is a real finding for the next round, recorded verbatim rather than
tolerated: it is the one place where the "specialized" route is not bit-comparable with
the rest of the engine.
