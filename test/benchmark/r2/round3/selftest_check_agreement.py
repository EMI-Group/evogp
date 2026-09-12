"""Dev-only self-test for check_agreement_r3.py (no GPU, no repo writes -- fake data in a tempdir)."""
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
    path = os.path.join(HERE, "check_agreement_r3.py")
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


def build(dir_path, mutate=None):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    for fixture, meta in FIXTURES.items():
        base = np.load(os.path.join(R2_VALUES, f"{meta['r2']}_numpy_values.npy")).astype(np.float64)
        arms = {"numpy": base, "iree": base.copy(), "iree_spec": base.copy()}
        if mutate:
            mutate(fixture, arms)
        for role, values in arms.items():
            if role == "numpy":
                backend, opt, dev = "numpy", None, None
            else:
                backend, opt, dev = "iree", "O1", 0
            write_arm(dir_path, f"{fixture}_{role}", values, fixture, meta["npz_sha"],
                      backend, opt, dev)


def run(module, agree_dir, r2_dir):
    module.AGREE_DIR = agree_dir
    module.R2_AGREE_DIR = r2_dir
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = module.main()
    return code, buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix="r3-agree-selftest-")
    agree = os.path.join(tmp, "backend_agreement")
    failures = []

    # 1. all arms bit-identical -> gate PASS, continuity INFO from the real R2 arrays
    build(agree)
    code, out = run(spec("m_pass"), agree, R2_VALUES)
    print("case pass            : exit", code)
    for needle in ("gate: PASS", "max|diff|=0", "bit-identical=True",
                   "matches R2 anchor (bit-identical)", "max rel diff=0"):
        if needle not in out:
            failures.append(f"PASS case missing {needle!r}")
    if code != 0:
        failures.append(f"PASS case returned {code}")

    # 2. iree within tolerance but not bit-identical -> still PASS, non-zero max|diff|
    def perturb(fixture, arms):
        arms["iree"] = arms["iree"] + 1e-7
        arms["iree_spec"] = arms["iree_spec"] - 1e-7
    build(agree, perturb)
    code, out = run(spec("m_tol"), agree, R2_VALUES)
    print("case within tolerance: exit", code)
    if code != 0 or "bit-identical=False" not in out:
        failures.append(f"tolerance case: exit {code}, bit-identical=False present="
                        f"{'bit-identical=False' in out}")

    # 3. iree outside tolerance -> gate FAIL (exit 1)
    def big(fixture, arms):
        arms["iree"] = arms["iree"] + 1.0
    build(agree, big)
    code, out = run(spec("m_fail"), agree, R2_VALUES)
    print("case out of tolerance: exit", code)
    if code != 1 or "gate: FAIL" not in out or "np.allclose(numpy, iree" not in out:
        failures.append(f"fail case: exit {code}, gate FAIL present={'gate: FAIL' in out}")

    # 4. different NaN mask -> FAIL
    def nanmask(fixture, arms):
        side = arms["iree"]
        side[0] = np.nan
    build(agree, nanmask)
    code, out = run(spec("m_nan"), agree, R2_VALUES)
    print("case NaN-mask differ : exit", code)
    if code != 1 or "NaN/±inf counts differ" not in out:
        failures.append(f"nan case: exit {code}, 'counts differ' present="
                        f"{'NaN/±inf counts differ' in out}")

    # 5. missing fixture200 arms -> FAIL, clean message, PASS lines for fixture64 only
    build(agree)
    for role in ("numpy", "iree", "iree_spec"):
        os.remove(os.path.join(agree, f"fixture200_{role}.json"))
    code, out = run(spec("m_missing"), agree, R2_VALUES)
    print("case missing stem    : exit", code)
    if code != 1 or "fixture200/fixture200_numpy: JSON missing" not in out:
        failures.append(f"missing case: exit {code}")

    # 6. malformed JSON -> FAIL, no traceback
    build(agree)
    with open(os.path.join(agree, "fixture64_iree.json"), "w") as handle:
        handle.write("{not json")
    code, out = run(spec("m_bad"), agree, R2_VALUES)
    print("case malformed JSON  : exit", code)
    if code != 1 or "unreadable" not in out:
        failures.append(f"malformed case: exit {code}")

    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print("\nSELFTEST FAIL")
        for item in failures:
            print("  -", item)
        return 1
    print("\nSELFTEST PASS (6 cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
