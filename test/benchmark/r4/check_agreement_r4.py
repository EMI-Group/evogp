#!/usr/bin/env python3
"""check_agreement_r4.py -- numeric gate over the R4 backend-agreement pre-flight records.

Reads the six records written by ``test/benchmark/r4/run_agreement_r4.sh`` (which drives the
READ-ONLY R2 tool ``test/benchmark/r2/check_backend_agreement.py``) from
``<this dir>/results/backend_agreement/`` and decides, per fixture, whether any IREE timing from
the R4 round may be quoted.

The R4 pure-forward matrix runs on the same box window as this pre-flight (unlike R3, whose
matrix ran days after its pre-flight), so the gate is deliberately strict -- and matches the
objective's wording:

* **(a) identity** -- every ``<stem>.json`` parses, reports the sha256 of the npz the arm
  actually loaded (recomputed here from the fixture file itself), the expected population size,
  and the ``<stem>_values.npy`` beside it hashes back to the JSON's ``values_sha256``.
* **(b) PRIMARY GATE** -- per fixture, the three roles (numpy / iree under ``EVOGP_EVAL_MODE``
  auto -> dense / iree forced specialized) report the SAME ``values_sha256``.  Bit-identity is
  the expected outcome (R2/R3 precedent: identical fitness vectors, not merely close).
* **(c) fallback (only when a sha differs)** -- load the ``_values.npy`` vectors and report
  max|diff| / max rel diff vs numpy: ``np.allclose(..., rtol=1e-4, atol=1e-4, equal_nan=True)``
  on every pair upgrades the fixture to **PASS-with-note** (quotable, with the deviation
  recorded); a pair beyond 1e-4 is a **hard FAIL** and no R4 IREE number may be quoted until it
  is understood.

Python 3 + numpy only: no torch, no evogp, no GPU, no network.  Missing or malformed inputs are
failures (the round may still be mid-flight -- then simply re-run the pre-flight runner).

Usage (from the repository root, or anywhere -- paths are resolved from this file)::

    python3 test/benchmark/r4/check_agreement_r4.py

Exit code: 0 = the (b) gate passes (or (c) finds every pair within 1e-4) for every fixture;
1 = any hard failure, missing or malformed input.
"""

import hashlib
import json
import os
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a data problem
    print("check_agreement_r4.py: numpy is required (stdlib + numpy only, no torch/GPU)",
          file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
#: the fixtures are referred to by their repository-root-relative paths (the runner's convention)
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
AGREE_DIR = os.path.join(HERE, "results", "backend_agreement")

RTOL = 1e-4
ATOL = 1e-4

#: per-fixture expectation table (the npz the three roles must have loaded, and its population)
FIXTURES = (
    {
        "fixture": "fixture64",
        "npz": "test/goldens/forest_random.npz",
        "pop": 64,
        "stems": ("fixture64_numpy", "fixture64_iree", "fixture64_iree_spec"),
    },
    {
        "fixture": "fixture200",
        "npz": "test/benchmark/fixtures/forest200_tree64_random.npz",
        "pop": 200,
        "stems": ("fixture200_numpy", "fixture200_iree", "fixture200_iree_spec"),
    },
)

ROLES = ("numpy", "iree", "iree_spec")           # the three roles of every fixture
PAIR_ROLES = (("numpy", "iree"), ("numpy", "iree_spec"), ("iree", "iree_spec"))


# ---------------------------------------------------------------------------
# tiny check accumulator (same shape as the R3 checker)
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
def sha256_file(path):
    """sha256 of a file's bytes -- the driver's fixture-hash convention."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values):
    """sha256 of the float64 little-endian bytes -- the driver's values-hash convention."""
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


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def short(digest):
    return digest[:12] + "…" if isinstance(digest, (str, bytes)) else "-"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    checker = Checker()
    verdicts = {}          # fixture -> "PASS" | "PASS-with-note" | "FAIL" | "MISSING"

    print("check_agreement_r4.py -- R4 backend-agreement gate (numpy vs iree[auto] vs iree[spec])")
    print(f"  agreement dir : {AGREE_DIR}")
    print(f"  fixtures      : {', '.join(f['npz'] for f in FIXTURES)} (relative to the repo root)")
    print(f"  primary gate  : identical values_sha256 across the three roles")
    print(f"  fallback gate : every pair within np.allclose(rtol={RTOL:g}, atol={ATOL:g}, equal_nan=True)")

    for fixture in FIXTURES:
        name = fixture["fixture"]
        npz_path = os.path.join(REPO_ROOT, fixture["npz"])
        npz_sha = None
        npz_err = None
        try:
            npz_sha = sha256_file(npz_path)
        except OSError as exc:
            npz_err = f"{type(exc).__name__}: {exc}"
        print()
        print(f"== {name}  ({fixture['npz']}, pop {fixture['pop']}) ==")
        if npz_err:
            print(f"  fixture unreadable: {npz_err}")
        else:
            print(f"  fixture sha256: {npz_sha}")

        records = {}
        malformed = False
        for role in ROLES:
            stem = f"{name}_{role}"
            json_path = os.path.join(AGREE_DIR, f"{stem}.json")
            values_path = os.path.join(AGREE_DIR, f"{stem}_values.npy")
            payload, json_err = load_json(json_path)
            records[role] = {
                "stem": stem, "payload": payload, "values": None,
                "json_path": json_path, "values_path": values_path,
                "values_sha256": payload.get("values_sha256") if payload else None,
            }
            if json_err:
                malformed = True
                checker.check(False, f"{name}/{stem}: JSON {json_err} ({json_path})")

        # ---- (a) identity ---------------------------------------------------
        print(f"{'role':<10} {'backend':<8} {'values_sha256':<14} {'count':>6} {'nan':>5} {'inf':>5} "
              f"{'npy==json':<9} {'fixture_sha256==npz':<20}")
        for role in ROLES:
            rec = records[role]
            payload = rec["payload"]
            if payload is None:
                print(f"{role:<10} {'-':<8} {'-':<14} {'-':>6} {'-':>5} {'-':>5} "
                      f"{'-':<9} {'-':<20}")
                continue
            values, values_err = load_values(rec["values_path"])
            rec["values"] = values
            if values_err:
                malformed = True
                checker.check(False,
                              f"{name}/{rec['stem']}: values array {values_err} "
                              f"({rec['values_path']})")
            # the three roles must all be the same fixture, rebuilt by the same code path
            if npz_sha is not None:
                checker.check(payload.get("fixture_sha256") == npz_sha,
                              f"{name}/{rec['stem']}: fixture_sha256 "
                              f"{payload.get('fixture_sha256')!r} != sha256({fixture['npz']}) "
                              f"{npz_sha!r}")
            if values is not None:
                nan = int(np.count_nonzero(np.isnan(values)))
                inf = int(np.count_nonzero(np.isinf(values)))
                recomputed = sha256_values(values)
                npy_agrees = recomputed == rec["values_sha256"]
                checker.check(npy_agrees,
                              f"{name}/{rec['stem']}: JSON values_sha256 "
                              f"{rec['values_sha256']!r} != hash of "
                              f"{os.path.basename(rec['values_path'])} {recomputed!r}")
                checker.check(values.size == payload.get("count") == fixture["pop"],
                              f"{name}/{rec['stem']}: length mismatch -- array {values.size}, "
                              f"JSON count {payload.get('count')!r}, fixture pop {fixture['pop']}")
                checker.check(nan == payload.get("nan_count") and inf == payload.get("inf_count"),
                              f"{name}/{rec['stem']}: JSON nan/inf counts "
                              f"({payload.get('nan_count')!r}/{payload.get('inf_count')!r}) != "
                              f"recomputed ({nan}/{inf})")
                counts = f"{values.size:>6} {nan:>5} {inf:>5}"
                hash_col = short(rec["values_sha256"])
                npy_col = "ok" if npy_agrees else "MISMATCH"
            else:
                counts, hash_col, npy_col = f"{'-':>6} {'-':>5} {'-':>5}", short(rec["values_sha256"]), "-"
            fixture_col = "ok" if (npz_sha is not None
                                   and payload.get("fixture_sha256") == npz_sha) else "?"
            print(f"{role:<10} {str(payload.get('backend') or '-'):<8} {hash_col:<14} {counts} "
                  f"{npy_col:<9} {fixture_col:<20}")

        # ---- (b) PRIMARY GATE: all three hashes equal -----------------------
        hashes = [records[role]["values_sha256"] for role in ROLES]
        if any(h is None for h in hashes):
            verdicts[name] = "MISSING"
            print("gate: MISSING (at least one role has no readable JSON -- see (a))")
            continue
        identical = len(set(hashes)) == 1
        checker.check(identical,
                      f"{name}: values_sha256 differ across roles -- "
                      + ", ".join(f"{role}={short(records[role]['values_sha256'])}"
                                  for role in ROLES))
        if identical:
            verdicts[name] = "PASS"
            print(f"gate: PASS -- all three roles bit-identical ({short(hashes[0])})")
            continue

        # ---- (c) fallback: same values within 1e-4? -------------------------
        print("gate: hashes differ -> fallback comparison against numpy "
              f"(rtol={RTOL:g}, atol={ATOL:g}, equal_nan=True)")
        note = False
        hard_fail = malformed
        for left, right in PAIR_ROLES:
            a, b = records[left]["values"], records[right]["values"]
            if a is None or b is None:
                hard_fail = True
                print(f"  {left} vs {right:<10} UNKNOWN (a values array is missing)")
                continue
            if a.size != b.size:
                hard_fail = True
                print(f"  {left} vs {right:<10} FAIL  length mismatch ({a.size} vs {b.size})")
                continue
            ok = bool(np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True))
            max_abs, max_rel = max_abs_rel(a, b)
            if not ok:
                hard_fail = True
            else:
                note = True
            print(f"  {left} vs {right:<10} {'within 1e-4' if ok else 'BEYOND 1e-4'}  "
                  f"max|diff|={fmt(max_abs)}  max rel diff={fmt(max_rel)}")
        if hard_fail and malformed:
            verdicts[name] = "FAIL"
        elif hard_fail:
            # register the failure explicitly (the allclose check below already did per pair)
            for left, right in PAIR_ROLES:
                a, b = records[left]["values"], records[right]["values"]
                if a is not None and b is not None and a.size == b.size:
                    checker.check(bool(np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)),
                                  f"{name}: {left} vs {right} differ beyond rtol={RTOL:g}, "
                                  f"atol={ATOL:g}")
            verdicts[name] = "FAIL"
        else:
            verdicts[name] = "PASS-with-note" if note else "FAIL"

    # ---- report -------------------------------------------------------------
    print()
    print("== verdict ==")
    for fixture in FIXTURES:
        name = fixture["fixture"]
        print(f"  {name:<12} {verdicts.get(name, 'FAIL')}")
    failed = checker.failures
    print()
    print(f"gate: {'PASS' if not failed else 'FAIL'}  ({checker.checks} check(s), "
          f"{len(failed)} failure(s))")
    for failure in failed:
        print(f"  FAIL {failure}")
    if failed:
        print("  (a missing stem usually means the pre-flight is still mid-flight: run "
              "test/benchmark/r4/run_agreement_r4.sh)")
        print("  R4 IREE numbers are NOT quotable until this gate passes.")
        return 1
    if all(verdicts.get(f["fixture"]) == "PASS" for f in FIXTURES):
        print("  numpy / iree(auto -> dense) / iree(specialized) are bit-identical on both "
              "fixtures: R4 IREE timings are quotable.")
    else:
        print("  every role pair is within the recorded tolerance (PASS-with-note): R4 IREE "
              "timings are quotable, with the noted deviation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
