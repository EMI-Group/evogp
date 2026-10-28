"""Dense-vs-specialized warm CROSSOVER probe (dev CLI, not pytest-collected).

Staged, one eval-mode side and one process per run: it times
``Forest.SR_fitness`` warm (median of ``AB_REPEATS``, ``torch.cuda.synchronize``
around every sample) on a rows sweep for ONE population, so the two sides of the
`EVOGP_DENSE_AUTO_MAX_ROWS` question (the dense route and the specialized
engine) can be compared on the same fixture on a device backend.

    cd /tmp/dr_before_wt            # or the repo root, for the other side
    MODE=dense        EVOGP_EVOXC_DEVICE=0 AB_ROWS=64,1024,4096,8192,16384 \
      PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      <venv>/bin/python dense_crossover_probe.py dense-after /tmp/dense.json
    MODE=specialized  ...  dense_crossover_probe.py specialized /tmp/spec.json

The first call of a case pays the compile (one dense graph, or one graph per
distinct tree structure on the specialized side) and is untimed; the recorded
grid and the gate decision live in ``device_resident_findings.md`` §5.
"""

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "src")
from evogp.tree.device_resident_check import fixture  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else "?"
out_path = sys.argv[2] if len(sys.argv) > 2 else None
repeats = int(os.environ.get("AB_REPEATS", "7"))
mode = os.environ.get("MODE", "dense")
os.environ["EVOGP_EVAL_MODE"] = mode
os.environ["EVOGP_EVOXC_BACKEND"] = "iree"

rows = [int(item) for item in os.environ.get("AB_ROWS", "64,1024,4096,16384").split(",")]
pop = int(os.environ.get("AB_POP", "16"))

forest, inputs, labels = fixture(pop, max(rows), seed=41)
out = {"label": label, "mode": mode, "repeats": repeats, "pop": pop, "rows": []}
for n in rows:
    x = inputs[:n].contiguous()
    y = labels[:n].contiguous()
    call = lambda x=x, y=y: forest.SR_fitness(x, y, True, "auto")
    call()  # compile/build + warm
    call()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    row = {
        "rows": n,
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
    }
    out["rows"].append(row)
    print(
        f"{label:>12} {mode:>11} P={pop} N={n:>6} "
        f"median={row['median_ms']:9.3f} ms min={row['min_ms']:9.3f} ms"
    )
    sys.stdout.flush()

if out_path:
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"[json] {out_path}")
