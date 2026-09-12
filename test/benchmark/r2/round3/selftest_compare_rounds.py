"""Dev-only self-test for compare_rounds.py (fake R3 records in a tempdir; no GPU, no repo writes)."""
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir, os.pardir))
R2_NEW = os.path.join(REPO, "test", "benchmark", "new", "results", "r2")
R2_LEG = os.path.join(REPO, "test", "benchmark", "legacy", "results", "r2")

EVAL_IDS = ["leg_c5r2", "leg_fixture200_n64", "np_c5r2", "np_fixture200_n64",
            "iree_fixture64_n64", "iree_fixture200_n64"]
TRAIN_IDS = ["leg_sr_a", "np_sr_a", "iree_sr_a"]
SPEC_IDS = ["iree_fixture64_n64_spec", "iree_fixture200_n64_spec", "iree_sr_a_spec"]


def module():
    spec = importlib.util.spec_from_file_location("cmp", os.path.join(HERE, "compare_rounds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_eval(src, dst, factor):
    payload = json.load(open(src))
    payload["tag"] = payload["tag"] + "_FAKE"
    payload["median_warm_ms"] = payload["median_warm_ms"] / factor
    payload["first_call_ms"] = payload["first_call_ms"] / factor
    json.dump(payload, open(dst, "w"))


def make_train(src, dst, factor):
    payload = json.load(open(src))
    payload["tag"] = payload["tag"] + "_FAKE"
    for rep in payload.get("per_rep") or []:
        rep["per_gen_ms"] = [g / factor for g in rep["per_gen_ms"]]
        if isinstance(rep.get("cold_first_gen_ms"), (int, float)):
            rep["cold_first_gen_ms"] = rep["cold_first_gen_ms"] / factor
    payload["per_gen_ms"] = [g / factor for g in payload["per_gen_ms"]]
    if isinstance(payload.get("cold_first_gen_ms"), (int, float)):
        payload["cold_first_gen_ms"] = payload["cold_first_gen_ms"] / factor
    json.dump(payload, open(dst, "w"))


def main():
    tmp = tempfile.mkdtemp(prefix="r3-compare-selftest-")
    out_new = os.path.join(tmp, "new")
    os.makedirs(out_new)
    failures = []

    for id_ in EVAL_IDS + TRAIN_IDS:
        src_dir = R2_LEG if id_.startswith("leg_") else R2_NEW
        src = os.path.join(src_dir, f"{id_}.json")
        dst = os.path.join(out_new, f"{id_}.json")
        (make_train if id_ in TRAIN_IDS else make_eval)(src, dst, 2.0)
    for spec_id, base in (("iree_fixture64_n64_spec", "iree_fixture64_n64"),                          ("iree_fixture200_n64_spec", "iree_fixture200_n64")):
        make_eval(os.path.join(R2_NEW, base + ".json"), os.path.join(out_new, spec_id + ".json"), 4.0)
    make_train(os.path.join(R2_NEW, "iree_sr_a.json"), os.path.join(out_new, "iree_sr_a_spec.json"), 3.0)

    mod = module()
    mod.ROUND_DIRS[("r3", "new")] = out_new          # legacy R3 side stays empty on purpose
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main([])
    text = buf.getvalue()
    print(text)

    # R2 warm 21.06 -> fake R3 10.53 (factor 2) => -50.0%; spec arms are /4 => 0.50x
    needles = ("-50.0%", "0.50x", "churning", "fully warm", "spec/auto R3", "678.15",
               "R3 spec (specialized)")
    for needle in needles:
        if needle not in text:
            failures.append(f"missing {needle!r} in the report")
    if "-5.0e+01%" in text:
        failures.append("delta not formatted")
    if code != 0:
        failures.append(f"exit {code}")
    shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("SELFTEST FAIL")
        for item in failures:
            print("  -", item)
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
