# test/benchmark/r2/round3/results/armB/gate_outputs.md — verbatim Arm B gate outputs
Append-only evidence file: one section per gate invocation, in execution order. Each section records
the date, the exact command, the process exit code and the **verbatim stdout** of that run (nothing
reformatted, nothing trimmed). The machine-readable evidence the checker reads stays where it was
produced — `results/armB/backend_agreement/<stem>.{json,_values.npy,log}` plus
`run_agreement_armB.last_run.txt` for the arms, `results/backend_agreement/` (Arm A, frozen) for the
identity anchors; this file only archives the gate's own report.
Environment for every run below: repository root as cwd, system `python3` (stdlib + numpy only — the
checker never imports torch/evogp or touches a GPU), interpreter of the arms is the project venv
`/mnt/local-ssd/bchuang/evox/.venv/bin/python`.

## Gate run 1 — 2026-09-12T23:05+08:00 (UTC 15:05) — Arm B agreement gate (patched-ETL shadow, six arms)
- date/time: 2026-09-12 23:05 HKT (2026-09-12 15:05 UTC); records produced 2026-09-12T23:01:01–23:04:05 HKT
- command: `python3 test/benchmark/r2/round3/check_agreement_armB.py`
- exit code: **0** (`gate: PASS (38 check(s), 0 failure(s))`)
- gates exercised: (a) npz/array identity, (b) dense `numpy`↔`iree` within `rtol=atol=1e-4,
  equal_nan=True` + identical NaN/±inf counts, (c) the HARD Arm A→Arm B `values_sha256` identity gate
  (6 same-stem pairs)
- verbatim output (stdout, copied unedited):
```text
check_agreement_armB.py -- Arm B backend-agreement gate (patched-ETL shadow; numpy vs iree[+spec])
  agreement dir : /mnt/local-ssd/bchuang/evogp/.genesis/workers/worker_T3_A255/test/benchmark/r2/round3/results/armB/backend_agreement
  Arm A dir     : /mnt/local-ssd/bchuang/evogp/.genesis/workers/worker_T3_A255/test/benchmark/r2/round3/results/backend_agreement (HARD Arm A/B identity gate)
  tolerance     : rtol=0.0001 atol=0.0001 equal_nan=True

== fixture64  (test/goldens/forest_random.npz) ==
arm          backend   iree                count   nan   inf npz_sha256     values_sha256  npy==json
numpy        numpy     -                      64     3     0 0bfb0bbf48cd…  2e5c0fa1fdbe…  ok       
iree         iree      O1 dev=0               64     3     0 0bfb0bbf48cd…  b8192cb1770b…  ok       
iree_spec    iree      O1 dev=0 spec          64     3     0 0bfb0bbf48cd…  2e5c0fa1fdbe…  ok       
gate: NaN/±inf counts identical across numpy/iree/iree_spec (nan=3, inf=0)
gate fixture64: numpy vs iree              PASS  max|diff|=7.62939e-06 max rel diff=1.08241e-07  bit-identical=False
gate fixture64: numpy vs iree_spec         PASS  max|diff|=0 max rel diff=0  bit-identical=True
gate fixture64: iree vs iree_spec          PASS  max|diff|=7.62939e-06 max rel diff=1.08241e-07  bit-identical=False
gate fixture64: fixture64_numpy vs Arm A   PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture64: fixture64_iree vs Arm A    PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture64: fixture64_iree_spec vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0

== fixture200  (test/benchmark/fixtures/forest200_tree64_random.npz) ==
arm          backend   iree                count   nan   inf npz_sha256     values_sha256  npy==json
numpy        numpy     -                     200    12     0 6789ef61ecf8…  16d5bbe98481…  ok       
iree         iree      O1 dev=0              200    12     0 6789ef61ecf8…  07948efededc…  ok       
iree_spec    iree      O1 dev=0 spec         200    12     0 6789ef61ecf8…  16d5bbe98481…  ok       
gate: NaN/±inf counts identical across numpy/iree/iree_spec (nan=12, inf=0)
gate fixture200: numpy vs iree             PASS  max|diff|=0.00390625 max rel diff=2.02762e-07  bit-identical=False
gate fixture200: numpy vs iree_spec        PASS  max|diff|=0 max rel diff=0  bit-identical=True
gate fixture200: iree vs iree_spec         PASS  max|diff|=0.00390625 max rel diff=2.02762e-07  bit-identical=False
gate fixture200: fixture200_numpy vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture200: fixture200_iree vs Arm A  PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture200: fixture200_iree_spec vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0

== Arm A vs Arm B identity (HARD gate) ==
  INFO fixture64/numpy: Arm B 2e5c0fa1fdbe… vs Arm A 2e5c0fa1fdbe… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture64/iree: Arm B b8192cb1770b… vs Arm A b8192cb1770b… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture64/iree_spec: Arm B 2e5c0fa1fdbe… vs Arm A 2e5c0fa1fdbe… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/numpy: Arm B 16d5bbe98481… vs Arm A 16d5bbe98481… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/iree: Arm B 07948efededc… vs Arm A 07948efededc… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/iree_spec: Arm B 16d5bbe98481… vs Arm A 16d5bbe98481… -- bit-identical (max|diff|=0 max rel diff=0)

gate: PASS  (38 check(s), 0 failure(s))
  Arm B IREE timings are quotable for every fixture above (within the recorded tolerance), and every Arm B arm reproduces its Arm A fitness vector bitwise.
CHECKER_EXIT=0
```
