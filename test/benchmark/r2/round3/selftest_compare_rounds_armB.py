"""Dev-only self-test for compare_rounds_armB.py (fake records in a tempdir; no GPU, no repo writes).

The Arm B generator reads four record sources, so the test materialises all four as scaled copies of
the committed Arm B W2 records in a temp directory (the same trick ``selftest_compare_rounds.py`` uses
for the Arm A generator) and checks the report against values the scaling makes predictable:
Arm A = template x4, W1 = x2, W2 = x1 (R2 = x8) for the warm metrics, i.e. A->B = -75.0% and R2->B = -87.5%.
The W1 git reader is patched out (no ``git show`` in a temp world); everything else -- the loaders,
the metric rules delegated to ``compare_rounds.py``, the renderers, the robustness cross-checks and
the never-abort-on-missing contract -- is exercised for real.
"""
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
ARM_B_NEW = os.path.join(HERE, "results", "armB", "new")
LEGACY_DIR = os.path.join(HERE, "results", "legacy")

EVAL_KEYS = ("median_warm_ms", "first_call_ms")
TRAIN_KEYS = ("cold_first_gen_ms",)


def module():
    spec = importlib.util.spec_from_file_location(
        "cmp_armB", os.path.join(HERE, "compare_rounds_armB.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scaled(payload, factor):
    """A copy of a record with every warm/first-call timing multiplied by ``factor``."""
    out = json.loads(json.dumps(payload))
    for key in EVAL_KEYS + TRAIN_KEYS:
        if isinstance(out.get(key), (int, float)):
            out[key] = out[key] * factor
    out["per_call_ms"] = [value * factor for value in out.get("per_call_ms") or []]
    out["per_gen_ms"] = [value * factor for value in out.get("per_gen_ms") or []]
    for rep in out.get("per_rep") or []:
        for key in EVAL_KEYS + TRAIN_KEYS:
            if isinstance(rep.get(key), (int, float)):
                rep[key] = rep[key] * factor
        rep["per_call_ms"] = [value * factor for value in rep.get("per_call_ms") or []]
        rep["per_gen_ms"] = [value * factor for value in rep.get("per_gen_ms") or []]
    return out


def write_sources(tmp, mod):
    """Materialise r2 / Arm A / Arm B W1 / Arm B W2 fake records as scaled templates.

    One directory per source; the fake R2 and Arm A files are named with the ids the generator
    actually asks for (``r2_id`` / ``a_id`` of each workbook row, template = that row's Arm B record;
    the legacy rows have no Arm B row, so their template is the committed Arm A legacy record).
    """
    dirs = {name: os.path.join(tmp, name) for name in ("r2", "a", "w1", "w2")}
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    mod.ARM_B_DIR = dirs["w2"]

    def dump(directory, name, payload):
        with open(os.path.join(directory, f"{name}.json"), "w") as handle:
            json.dump(payload, handle)

    seen = {}
    for workload in mod.WORKLOADS:
        for _label, _side, r2_id, a_id, b_id in workload["rows"]:
            if b_id is None:                      # legacy: the Arm A legacy record is the template
                with open(os.path.join(LEGACY_DIR, f"{a_id}.json")) as handle:
                    template = json.load(handle)
                dump(dirs["a"], a_id, scaled(template, 4.0))
                dump(dirs["r2"], r2_id, scaled(template, 8.0))
                continue
            with open(os.path.join(ARM_B_NEW, f"{b_id}.json")) as handle:
                template = json.load(handle)
            seen[b_id] = template
            dump(dirs["a"], a_id, scaled(template, 4.0))
            dump(dirs["r2"], r2_id, scaled(template, 8.0))
            dump(dirs["w1"], b_id, scaled(template, 2.0))
            dump(dirs["w2"], b_id, scaled(template, 1.0))
    for id_ in mod.B2_IDS:                        # the non-manifest b2 re-attempt records
        if id_ in seen:
            continue
        with open(os.path.join(ARM_B_NEW, f"{id_}.json")) as handle:
            dump(dirs["w2"], id_, scaled(json.load(handle), 1.0))

    mod.load_arm_b_w1 = lambda id_: (json.load(open(os.path.join(dirs["w1"], f"{id_}.json"))), "ok")
    mod.cr.ROUND_DIRS[("r2", "new")] = dirs["r2"]
    mod.cr.ROUND_DIRS[("r2", "leg")] = dirs["r2"]
    mod.cr.ROUND_DIRS[("r3", "new")] = dirs["a"]
    mod.cr.ROUND_DIRS[("r3", "leg")] = dirs["a"]
    return seen


def run(mod, argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main(argv)
    return buf.getvalue(), code


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="armB-compare-selftest-")
    try:
        mod = module()
        template = write_sources(tmp, mod)
        text, code = run(mod, ["--markdown"])

        # Arm A = x4, W2 = x1 => (1-4)/4 = -75.0%; R2 = x8 => (1-8)/8 = -87.5%
        needles = ["-75.0%", "-87.5%", "churning", "fully warm", "dense (probe-degenerate)",
                   "route taken (stage fingerprint)", "skeleton (crash)", "IDENTICAL",
                   "per_gen_best_fitness", "cold+warm traffic", "fds start->cold->end",
                   "fully warm rep of the training ids", "json cold misses A / W1 / W2"]
        needles.append(f"{template['armB_np_c5r2']['median_warm_ms']:.2f}")       # the W2 warm cell
        needles.append(f"{template['armB_np_c5r2']['median_warm_ms'] * 4.0:.2f}")  # the Arm A warm cell
        for needle in needles:
            if needle not in text:
                failures.append(f"missing {needle!r} in the report")
        if "DIFFER" in text:
            failures.append("a cross-window identity check reported DIFFER on identical fixtures")
        if "MISSING" in text or "NOGIT" in text:
            failures.append("a readable record was reported MISSING/NOGIT")
        if code != 0:
            failures.append(f"exit {code} on a fully readable fake round")

        # the never-abort contract: an empty record tree must still print a report and exit 0
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty, exist_ok=True)
        mod.ARM_B_DIR = empty
        mod.load_arm_b_w1 = lambda id_: (None, "NOGIT")
        mod.cr.ROUND_DIRS[("r2", "new")] = empty
        mod.cr.ROUND_DIRS[("r2", "leg")] = empty
        mod.cr.ROUND_DIRS[("r3", "new")] = empty
        mod.cr.ROUND_DIRS[("r3", "leg")] = empty
        mod._MEMO.clear()
        text, code = run(mod, [])
        if "MISSING" not in text:
            failures.append("an empty record tree did not report MISSING")
        if "== robustness cross-checks" not in text:
            failures.append("an empty record tree aborted before the cross-checks")
        if code != 0:
            failures.append(f"exit {code} on an empty record tree (must never abort)")
    finally:
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
