"""Dev-only self-test for check_agreement_armB.py (no GPU, no repo writes -- fake data in a tempdir).

Builds a fake Arm A dir (results/backend_agreement analogue) and a fake Arm B dir
(results/armB/backend_agreement analogue) from the real R2 fitness arrays, then exercises the
module's gates end to end, including the Arm B-specific HARD Arm A/Arm B hash gate.
"""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir, os.pardir))
R2_VALUES = os.path.join(REPO, "test", "benchmark", "new", "results", "r2", "backend_agreement")

FIXTURES = {
    "fixture64": dict(npz_sha="0bfb0bbf48cd65832b2c024480bc572de5121675946d3611a2c7a5b234ff9ca0",
                      pop=64, r2="fixture64"),
    "fixture200": dict(npz_sha="6789ef61ecf8898b9b2a00774ee655e20b0b5be7c72eb861afc8d6b8799741b9",
                       pop=200, r2="fixture200"),
}


def spec(name):
    path = os.path.join(HERE, "check_agreement_armB.py")
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def write_arm(out_dir, stem, values, fixture, npz_sha, backend, opt=None, dev=None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{stem}_values.npy"), values)
    payload = {
        "fixture": f"<{fixture}>",
        "fixture_sha256": npz_sha,
        "backend": backend,
        "iree_opt_level": opt,
        "iree_device": dev,
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "inf_count": int(np.count_nonzero(np.isinf(values))),
        "count": int(values.size),
        "finite_mean": float(values[np.isfinite(values)].mean()),
        "per_call_ms": [1.0, 2.0],
        "values_sha256": hashlib.sha256(
            np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest(),
    }
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build(arm_a_dir, arm_b_dir, mutate_a=None, mutate_b=None):
    """(Re)write both fake record sets; base arrays come from the frozen R2 numpy records."""
    for dir_path in (arm_a_dir, arm_b_dir):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    for fixture, meta in FIXTURES.items():
        base = np.load(os.path.join(R2_VALUES, f"{meta['r2']}_numpy_values.npy")).astype(np.float64)
        arms_a = {"numpy": base.copy(), "iree": base.copy(), "iree_spec": base.copy()}
        arms_b = {"numpy": base.copy(), "iree": base.copy(), "iree_spec": base.copy()}
        if mutate_a:
            mutate_a(fixture, arms_a, arms_b)
        if mutate_b:
            mutate_b(fixture, arms_a, arms_b)
        for role, values in arms_a.items():
            if role == "numpy":
                backend, opt, dev = "numpy", None, None
            else:
                backend, opt, dev = "iree", "O1", 0
            write_arm(arm_a_dir, f"{fixture}_{role}", values, fixture, meta["npz_sha"],
                      backend, opt, dev)
        for role, values in arms_b.items():
            if role == "numpy":
                backend, opt, dev = "numpy", None, None
            else:
                backend, opt, dev = "iree", "O1", 0
            write_arm(arm_b_dir, f"{fixture}_{role}", values, fixture, meta["npz_sha"],
                      backend, opt, dev)


def run(module, arm_b_dir, arm_a_dir):
    module.AGREE_DIR = arm_b_dir
    module.ARM_A_AGREE_DIR = arm_a_dir
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = module.main()
    return code, buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix="armB-agree-selftest-")
    arm_a = os.path.join(tmp, "backend_agreement")
    arm_b = os.path.join(tmp, "armB", "backend_agreement")
    failures = []

    # 1. Arm B == Arm A bitwise -> both gates PASS
    build(arm_a, arm_b)
    code, out = run(spec("m_pass"), arm_b, arm_a)
    print("case pass              : exit", code)
    for needle in ("gate: PASS", "max|diff|=0", "bit-identical=True",
                   "bit-identical=True  max|diff|=0", "Arm A vs Arm B identity (HARD gate)"):
        if needle not in out:
            failures.append(f"PASS case missing {needle!r}")
    if code != 0:
        failures.append(f"PASS case returned {code}")

    # 2. Arm B within tolerance but NOT bit-identical to Arm A -> HARD A/B gate fails
    def tol(fixture, arms_a, arms_b):
        arms_b["iree"] = arms_b["iree"] + 1e-7
    build(arm_a, arm_b, mutate_b=tol)
    code, out = run(spec("m_abtol"), arm_b, arm_a)
    print("case A/B tiny drift    : exit", code)
    if code != 1 or "must not change the fitness vector" not in out:
        failures.append(f"A/B-drift case: exit {code}, A/B failure present="
                        f"{'must not change the fitness vector' in out}")

    # 3. Arm A alone perturbed (Arm B still internally consistent) -> HARD A/B gate fails
    def tol_a(fixture, arms_a, arms_b):
        arms_a["numpy"] = arms_a["numpy"] + 1e-7
    build(arm_a, arm_b, mutate_a=tol_a)
    code, out = run(spec("m_atol"), arm_b, arm_a)
    print("case Arm A drift       : exit", code)
    if code != 1 or "Arm B values_sha256" not in out:
        failures.append(f"Arm-A-drift case: exit {code}")

    # 4. Arm B iree outside tolerance vs its own numpy -> primary gate fails
    def big(fixture, arms_a, arms_b):
        arms_b["iree"] = arms_b["iree"] + 1.0
        arms_a["iree"] = arms_a["iree"] + 1.0     # keep A/B identity intact
    build(arm_a, arm_b, mutate_a=big, mutate_b=big)
    code, out = run(spec("m_fail"), arm_b, arm_a)
    print("case out of tolerance : exit", code)
    if code != 1 or "gate: FAIL" not in out or "np.allclose(numpy, iree" not in out:
        failures.append(f"fail case: exit {code}, gate FAIL present={'gate: FAIL' in out}")

    # 5. different NaN mask -> FAIL
    def nanmask(fixture, arms_a, arms_b):
        arms_a["iree"][0] = np.nan
        arms_b["iree"][0] = np.nan
    build(arm_a, arm_b, mutate_a=nanmask, mutate_b=nanmask)
    code, out = run(spec("m_nan"), arm_b, arm_a)
    print("case NaN-mask differ  : exit", code)
    if code != 1 or "NaN/±inf counts differ" not in out:
        failures.append(f"nan case: exit {code}, 'counts differ' present="
                        f"{'NaN/±inf counts differ' in out}")

    # 6. missing Arm B fixture200 arms -> FAIL, clean message, fixture64 still gates
    build(arm_a, arm_b)
    for role in ("numpy", "iree", "iree_spec"):
        os.remove(os.path.join(arm_b, f"fixture200_{role}.json"))
    code, out = run(spec("m_missing"), arm_b, arm_a)
    print("case missing stem     : exit", code)
    if code != 1 or "fixture200/fixture200_numpy: JSON missing" not in out:
        failures.append(f"missing case: exit {code}")

    # 7. missing Arm A record -> FAIL (an unverifiable pair is not a pass)
    build(arm_a, arm_b)
    os.remove(os.path.join(arm_a, "fixture64_iree_spec.json"))
    code, out = run(spec("m_noA"), arm_b, arm_a)
    print("case missing Arm A    : exit", code)
    if code != 1 or "Arm A record missing" not in out:
        failures.append(f"missing-Arm-A case: exit {code}, message present="
                        f"{'Arm A record missing' in out}")

    # 8. malformed Arm B JSON -> FAIL, no traceback
    build(arm_a, arm_b)
    with open(os.path.join(arm_b, "fixture64_iree.json"), "w") as handle:
        handle.write("{not json")
    code, out = run(spec("m_bad"), arm_b, arm_a)
    print("case malformed JSON   : exit", code)
    if code != 1 or "unreadable" not in out:
        failures.append(f"malformed case: exit {code}")

    # 9. no records at all -> FAIL gracefully, no traceback
    shutil.rmtree(arm_b, ignore_errors=True)
    code, out = run(spec("m_empty"), arm_b, arm_a)
    print("case no records       : exit", code)
    if code != 1 or "gate: FAIL" not in out or "Traceback" in out:
        failures.append(f"empty case: exit {code}")

    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print("\nSELFTEST FAIL")
        for item in failures:
            print("  -", item)
        return 1
    print("\nSELFTEST PASS (9 cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
