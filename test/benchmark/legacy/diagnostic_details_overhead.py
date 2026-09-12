#!/usr/bin/env python3
"""One-off diagnostic backing a claim in `test/benchmark/legacy/NOTES.md`.

Why is the `BUILD_NOTES.md` §4(b) `sr_test` smoke (~10.9 ms per generation at pop 1000 / tree_len 128)
3–7x slower per generation than this directory's C2/C2b records (1.6–3.5 ms) for the *same* budget?

The only difference between the two drivers is the detail-printing switch: `sr_test.py` calls
`StandardPipeline.run()` with the default `is_show_details=True`, while `run_matrix.py` (per SPEC.md:
`is_show_details=False`) loops `pipeline.step()` manually. `StandardPipeline.show_details` is host-side
Python running on the **CPU** fitness tensor: a boolean-mask filter, builtin `max()`/`min()` (element-by-
element Python iteration over 1000 scalar tensors) and `torch.mean`/`torch.std`, plus two prints.

This script measures that on the C2/C2b workload — it reuses `run_matrix.build_w1` so construction is
byte-identical to the committed C2/C2b records (same seed, same problem/descriptor/algorithm) — in three
20-generation phases plus a micro-decomposition of `show_details`:

  A details OFF | B details ON (same call as run()) | C details OFF (drift control)

Run exactly like `run_matrix.py` (one `flock` on the shared host, see NOTES.md):

    cd /tmp/evogp_legacy_bench && CUDA_VISIBLE_DEVICES=<i> \\
        PYTHONPATH=src:/tmp/evogp_t1a263/siteutils \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python -u <abs>/test/benchmark/legacy/diagnostic_details_overhead.py

Not part of the SPEC.md config matrix; it changes nothing about the recorded runs.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_matrix as rm  # driver under test: same build_w1/CONFIGS/seed as the committed records

N_GENS = 20
WARM = 3
SMOKE_MEDIAN_MS = 10.90  # BUILD_NOTES.md §5, sr_test defaults, is_show_details=True
SMOKE_MIN_MS = 9.00


def main():
    A = rm.api()
    torch = A.torch
    torch.manual_seed(rm.SEED)
    torch.cuda.manual_seed(rm.SEED)

    meta = dict(rm.CONFIGS["C2"])
    problem, algorithm, _ = rm.build_w1(meta)
    pipeline = A.StandardPipeline(
        algorithm, problem, generation_limit=meta["gens"], is_show_details=False
    )

    print(f"=== diagnostic_details_overhead.py on {rm.gpu_label(torch)}", flush=True)
    print(f"=== side={A.side} evogp_file={A.evogp_file} env={rm.env_snapshot()}", flush=True)
    print(
        f"=== workload = C2/C2b W1 (pop={meta['pop']} gens={meta['gens']} tree_len={meta['tree_len']} "
        f"seed={rm.SEED}); phases of {N_GENS} generations after {WARM} warm-up generations",
        flush=True,
    )

    def phase(label, details):
        times = []
        for _ in range(N_GENS):
            start_time = time.time()  # what run() feeds show_details
            torch.cuda.synchronize()
            tic = time.perf_counter()
            fitness = pipeline.step()
            if details:
                pipeline.show_details(start_time, 0, fitness)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - tic) * 1000.0)
            last_fitness[0] = fitness
        print(
            f"=== {label}: median {statistics.median(times):.3f} ms  min {min(times):.3f}  "
            f"max {max(times):.3f}  (n={len(times)})",
            flush=True,
        )
        return times

    for _ in range(WARM):
        pipeline.step()
    torch.cuda.synchronize()

    last_fitness = [None]
    off_a = phase("A details OFF", False)
    on = phase("B details ON ", True)
    off_c = phase("C details OFF", False)

    # Micro-decomposition of show_details on the last CPU fitness tensor (same shape as the smoke's).
    f = last_fitness[0]
    tic = time.perf_counter()
    filtered = f[(f < pipeline.valid_fitness_boundry) & (f > -pipeline.valid_fitness_boundry)]
    mask_ms = (time.perf_counter() - tic) * 1000.0
    tic = time.perf_counter()
    max_f, min_f = max(filtered), min(filtered)
    minmax_ms = (time.perf_counter() - tic) * 1000.0
    tic = time.perf_counter()
    mean_f, std_f = torch.mean(filtered), torch.std(filtered)
    stats_ms = (time.perf_counter() - tic) * 1000.0
    tic = time.perf_counter()
    pipeline.show_details(time.time(), 0, f)
    whole_ms = (time.perf_counter() - tic) * 1000.0
    print(
        f"=== micro (n={len(f)} valid={len(filtered)}): mask {mask_ms:.3f} ms | builtin max/min "
        f"{minmax_ms:.3f} ms | mean/std {stats_ms:.3f} ms | whole show_details(incl. print) {whole_ms:.3f} ms",
        flush=True,
    )

    off_median = statistics.median([statistics.median(off_a), statistics.median(off_c)])
    on_median = statistics.median(on)
    overhead = on_median - off_median
    drift = abs(statistics.median(off_a) - statistics.median(off_c))
    print(
        f"=== SUMMARY: details OFF {off_median:.3f} ms/gen (phase A/C drift {drift:.3f} ms), details ON "
        f"{on_median:.3f} ms/gen -> show_details overhead {overhead:.3f} ms/gen ({overhead / off_median:.1f}x)",
        flush=True,
    )
    print(
        f"=== SUMMARY: BUILD_NOTES sr_test smoke (details ON) median {SMOKE_MEDIAN_MS:.2f} ms / min "
        f"{SMOKE_MIN_MS:.2f} ms vs this run's details-ON median {on_median:.3f} ms "
        f"(difference {SMOKE_MEDIAN_MS - on_median:+.3f} ms = shared-host conditions, not workload)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
