#!/usr/bin/env python3
"""check_agreement_armB.py -- Arm B numeric gate over the patched-ETL backend-agreement records.

Arm B re-runs the six R3 agreement arms with a PATCHED ETL checkout shadowed onto
``PYTHONPATH`` (``src:/tmp/evogp_etl_patched_51a7141``).  This checker reads what
``test/benchmark/r2/round3/run_agreement_armB.sh`` wrote -- six arms in
``<this dir>/results/armB/backend_agreement/`` (it drives the READ-ONLY R2 tool
``test/benchmark/r2/check_backend_agreement.py``) -- and decides, per fixture, whether any IREE
timing from the Arm B round may be quoted:

* **(a) identity** -- every ``<stem>.json`` parses and reports the expected fixture sha256
  (the npz the arm actually loaded), and the ``<stem>_values.npy`` beside it is the array the
  JSON's ``values_sha256`` was taken from (recomputed here, plus length == ``count``).
* **(b) PRIMARY GATE** -- per fixture, the NaN/±inf counts of numpy / iree / iree_spec are
  identical and ``np.allclose(numpy, iree, rtol=1e-4, atol=1e-4, equal_nan=True)`` and
  ``np.allclose(numpy, iree_spec, ...)`` both hold.  Bit-identical ``values_sha256`` for a
  pair is *reported*, never required (the IREE default route is the dense evaluator, whose
  reduction order differs from the numpy interpreter's).
* **(c) ARM A vs ARM B -- HARD GATE** -- each Arm B arm must reproduce the ``values_sha256``
  of its SAME-STEM Arm A counterpart in ``<this dir>/results/backend_agreement/``
  (3 arms x 2 fixtures = 6 pairs).  Any mismatch fails this checker, a missing Arm A record
  fails it too (an unverifiable pair is not a pass); the ``max |diff|`` vs the Arm A array is
  reported as evidence only.  This is the Arm B-specific replacement for the R3 checker's
  INFO-only R2-anchor continuity: Arm B's whole point is that the patched ETL checkout changes
  *nothing* numerically, so its anchor is Arm A, not R2.

Python 3 + numpy only: no torch, no evogp, no GPU, no network.  Missing inputs are reported
per stem and make the run exit 1 (the round may still be mid-flight).

Usage (from the repository root, or anywhere -- paths are resolved from this file)::

    python3 test/benchmark/r2/round3/check_agreement_armB.py

Exit code: 0 = the (b) and (c) gates pass for every fixture present; 1 = any gate failure,
missing or malformed input.
"""

import hashlib
import json
import os
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a data problem
    print("check_agreement_armB.py: numpy is required (stdlib + numpy only, no torch/GPU)",
          file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
AGREE_DIR = os.path.join(HERE, "results", "armB", "backend_agreement")
ARM_A_AGREE_DIR = os.path.join(HERE, "results", "backend_agreement")

RTOL = 1e-4
ATOL = 1e-4

#: per-fixture expectation table (npz identity hash; the Arm A counterpart has the same stem)
FIXTURES = (
    {
        "fixture": "fixture64",
        "npz": "test/goldens/forest_random.npz",
        "npz_sha256": "0bfb0bbf48cd65832b2c024480bc572de5121675946d3611a2c7a5b234ff9ca0",
        "pop": 64,
        "stems": ("fixture64_numpy", "fixture64_iree", "fixture64_iree_spec"),
    },
    {
        "fixture": "fixture200",
        "npz": "test/benchmark/fixtures/forest200_tree64_random.npz",
        "npz_sha256": "6789ef61ecf8898b9b2a00774ee655e20b0b5be7c72eb861afc8d6b8799741b9",
        "pop": 200,
        "stems": ("fixture200_numpy", "fixture200_iree", "fixture200_iree_spec"),
    },
)

ARMS = ("numpy", "iree", "iree_spec")            # the three roles of every fixture
PAIR_ROLES = (("numpy", "iree"), ("numpy", "iree_spec"), ("iree", "iree_spec"))

# ---------------------------------------------------------------------------
# tiny check accumulator
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return bool(condition)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stem_for(fixture, role):
    """(Arm B JSON stem, Arm A counterpart stem) of a fixture/role pair.

    Arm B re-runs the very same six arms, so the Arm A counterpart has the same stem.
    """
    stem = f"{fixture['fixture']}_{role}"
    return stem, stem


def sha256_values(values):
    """sha256 of the float64 little-endian bytes -- the driver's hash convention."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def load_json(path):
    """(payload, error) -- never raises."""
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(payload, dict):
        return None, "not a JSON object"
    return payload, None


def load_values(path):
    """(array, error) -- never raises."""
    try:
        values = np.load(path)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    values = np.asarray(values)
    if values.ndim != 1:
        return None, f"expected a 1-D vector, got shape {tuple(values.shape)}"
    return values.astype(np.float64, copy=False), None


def counts(values):
    values = np.asarray(values, dtype=np.float64)
    return int(np.count_nonzero(np.isnan(values))), int(np.count_nonzero(np.isinf(values)))


def values_hash(record):
    """The ``values_sha256`` a record's JSON reports (None when the JSON is unreadable)."""
    payload = record.get("payload")
    return payload.get("values_sha256") if isinstance(payload, dict) else None

def max_abs_rel(a, b):
    """(max |a-b|, max |a-b|/|b|) over entries finite in both; (None, None) when none."""
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return None, None
    diff = np.abs(a[mask] - b[mask])
    max_abs = float(diff.max())
    denom = np.abs(b[mask])
    nonzero = denom > 0
    max_rel = float((diff[nonzero] / denom[nonzero]).max()) if nonzero.any() else None
    return max_abs, max_rel

def fmt(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def short(digest):
    return digest[:12] + "…" if isinstance(digest, str) else "-"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    checker = Checker()
    info = []

    print("check_agreement_armB.py -- Arm B backend-agreement gate "
          "(patched-ETL shadow; numpy vs iree[+spec])")
    print(f"  agreement dir : {AGREE_DIR}")
    print(f"  Arm A dir     : {ARM_A_AGREE_DIR} (HARD Arm A/B identity gate)")
    print(f"  tolerance     : rtol={RTOL:g} atol={ATOL:g} equal_nan=True")

    for fixture in FIXTURES:
        name = fixture["fixture"]
        print()
        print(f"== {name}  ({fixture['npz']}) ==")

        records = {}
        for role in ARMS:
            stem, arm_a_stem = stem_for(fixture, role)
            json_path = os.path.join(AGREE_DIR, f"{stem}.json")
            values_path = os.path.join(AGREE_DIR, f"{stem}_values.npy")
            payload, json_err = load_json(json_path)
            values, values_err = load_values(values_path)
            records[role] = {
                "stem": stem, "arm_a_stem": arm_a_stem, "payload": payload, "values": values,
                "json_path": json_path, "values_path": values_path,
            }
            if json_err:
                checker.check(False, f"{name}/{stem}: JSON {json_err} ({json_path})")
            if values_err:
                checker.check(False, f"{name}/{stem}: values array {values_err} ({values_path})")

        # ---- (a) identity ---------------------------------------------------
        print(f"{'arm':<12} {'backend':<9} {'iree':<18} {'count':>6} {'nan':>5} {'inf':>5} "
              f"{'npz_sha256':<14} {'values_sha256':<14} {'npy==json':<9}")
        for role in ARMS:
            rec = records[role]
            payload, values = rec["payload"], rec["values"]
            if payload is None or values is None:
                print(f"{role:<12} {'-':<9} {'-':<18} {'-':>6} {'-':>5} {'-':>5} "
                      f"{'-':<14} {'-':<14} {'-':<9}")
                continue
            nan, inf = counts(values)
            recomputed = sha256_values(values)
            recorded = payload.get("values_sha256")
            json_sha = payload.get("fixture_sha256")
            iree_note = "-"
            if payload.get("backend") == "iree":
                opt = payload.get("iree_opt_level") or "O?"
                dev = payload.get("iree_device")
                iree_note = f"{opt} dev={dev}"
                if role == "iree_spec":
                    iree_note += " spec"

            checker.check(json_sha == fixture["npz_sha256"],
                          f"{name}/{rec['stem']}: fixture_sha256 {json_sha!r} != the {name} npz hash "
                          f"{fixture['npz_sha256']!r}")
            checker.check(recorded == recomputed,
                          f"{name}/{rec['stem']}: JSON values_sha256 {recorded!r} != the hash of "
                          f"{os.path.basename(rec['values_path'])} {recomputed!r}")
            checker.check(values.size == payload.get("count") == fixture["pop"],
                          f"{name}/{rec['stem']}: length mismatch -- array {values.size}, JSON "
                          f"count {payload.get('count')!r}, fixture pop {fixture['pop']}")
            checker.check(payload.get("nan_count") == nan and payload.get("inf_count") == inf,
                          f"{name}/{rec['stem']}: JSON nan/inf counts "
                          f"({payload.get('nan_count')!r}/{payload.get('inf_count')!r}) != recomputed "
                          f"({nan}/{inf})")

            print(f"{role:<12} {str(payload.get('backend') or '-'):<9} {iree_note:<18} "
                  f"{values.size:>6} {nan:>5} {inf:>5} {short(json_sha):<14} {short(recorded):<14} "
                  f"{('ok' if recorded == recomputed else 'MISMATCH'):<9}")
        # ---- (b) PRIMARY GATE ----------------------------------------------
        reference = records["numpy"]
        ref_values = reference["values"]
        if ref_values is not None:
            nan, inf = counts(ref_values)
            same_counts = True
            for role in ("iree", "iree_spec"):
                other = records[role]["values"]
                if other is None:
                    same_counts = False
                    continue
                other_nan, other_inf = counts(other)
                if (other_nan, other_inf) != (nan, inf):
                    same_counts = False
                    checker.check(False, f"{name}: NaN/±inf counts differ -- numpy {nan}/{inf} vs "
                                         f"{role} {other_nan}/{other_inf}")
                else:
                    checker.check(True, f"{name}: NaN/±inf counts identical (numpy vs {role})")
            if same_counts:
                print(f"gate: NaN/±inf counts identical across numpy/iree/iree_spec "
                      f"(nan={nan}, inf={inf})")

        for left, right in PAIR_ROLES:
            a, b = records[left]["values"], records[right]["values"]
            label = f"gate {name}: {left} vs {right}"
            if a is None or b is None:
                print(f"{label:<42} SKIPPED (missing side)")
                continue
            ok = bool(np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True))
            max_abs, max_rel = max_abs_rel(a, b)
            hash_left, hash_right = values_hash(records[left]), values_hash(records[right])
            if hash_left is None or hash_right is None:
                bit_identical = "?"          # one of the JSONs is unreadable: see the (a) table
            else:
                bit_identical = hash_left == hash_right
            if left == "numpy":
                checker.check(ok, f"{name}: np.allclose(numpy, {right}, rtol={RTOL:g}, "
                                  f"atol={ATOL:g}, equal_nan=True) is False")
            print(f"{label:<42} {'PASS' if ok else 'FAIL'}  max|diff|={fmt(max_abs)} "
                  f"max rel diff={fmt(max_rel)}  bit-identical={bit_identical}")

        # ---- (c) ARM A vs ARM B -- HARD GATE (bit-identical values_sha256) --
        for role in ARMS:
            rec = records[role]
            stem = rec["stem"]
            arm_a_json = os.path.join(ARM_A_AGREE_DIR, f"{rec['arm_a_stem']}.json")
            label = f"gate {name}: {stem} vs Arm A"
            arm_b_hash = values_hash(rec)
            arm_a_payload, arm_a_err = load_json(arm_a_json)
            if arm_a_err:
                checker.check(False, f"{name}/{stem}: Arm A record {arm_a_err} ({arm_a_json}) -- "
                                     f"the Arm A/B identity cannot be verified")
                print(f"{label:<42} FAIL (Arm A record {arm_a_err})")
                continue
            arm_a_hash = arm_a_payload.get("values_sha256")
            if arm_a_hash is None:
                checker.check(False, f"{name}/{stem}: Arm A record {arm_a_json} carries no "
                                     f"values_sha256")
                print(f"{label:<42} FAIL (Arm A record has no values_sha256)")
                continue
            if arm_b_hash is None:
                print(f"{label:<42} SKIPPED (Arm B JSON unreadable: see the (a) table)")
                continue

            identical = arm_a_hash == arm_b_hash
            checker.check(identical, f"{name}/{stem}: Arm B values_sha256 {arm_b_hash!r} != Arm A "
                                     f"{arm_a_hash!r} ({arm_a_json}) -- the patched ETL checkout "
                                     f"must not change the fitness vector")

            # evidence only (never a gate): the max |diff| against the Arm A array
            arm_a_values, arm_a_values_err = load_values(
                os.path.join(ARM_A_AGREE_DIR, f"{rec['arm_a_stem']}_values.npy"))
            if arm_a_values_err or rec["values"] is None:
                diff_note = f"Arm A array {arm_a_values_err or 'unavailable'}"
            elif arm_a_values.size != rec["values"].size:
                diff_note = (f"Arm A array length {arm_a_values.size} != Arm B "
                             f"{rec['values'].size}")
            else:
                max_abs, max_rel = max_abs_rel(rec["values"], arm_a_values)
                diff_note = f"max|diff|={fmt(max_abs)} max rel diff={fmt(max_rel)}"
            info.append(f"{name}/{role}: Arm B {short(arm_b_hash)} vs Arm A {short(arm_a_hash)} "
                        f"-- {'bit-identical' if identical else 'MISMATCH'} ({diff_note})")
            print(f"{label:<42} {'PASS' if identical else 'FAIL'}  "
                  f"bit-identical={identical}  {diff_note}")

    # ---- report -------------------------------------------------------------
    print()
    print("== Arm A vs Arm B identity (HARD gate) ==")
    if info:
        for line in info:
            print(f"  INFO {line}")
    else:
        print("  (no Arm A/B pairs could be compared)")

    failed = checker.failures
    print()
    print(f"gate: {'PASS' if not failed else 'FAIL'}  ({checker.checks} check(s), "
          f"{len(failed)} failure(s))")
    for failure in failed:
        print(f"  FAIL {failure}")
    if failed:
        print("  (a missing stem usually means the Arm B pre-flight has not run yet: run "
              "test/benchmark/r2/round3/run_agreement_armB.sh)")
        return 1
    print("  Arm B IREE timings are quotable for every fixture above (within the recorded "
          "tolerance), and every Arm B arm reproduces its Arm A fitness vector bitwise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
