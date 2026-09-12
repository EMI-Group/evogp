"""xval helper: where does the wave-1 C5 driver's wall_s go?

``legacy/run_matrix.py`` reports ``wall_s`` = "construction + all 25 calls", but the
25 calls only sum to ~37 ms while the recorded wall is ~0.18 s.  This probe imports
the *driver itself* (``importlib``; its module body is ``__main__``-guarded) and times
its ``build_c5`` construction piece by piece plus a faithful replication of the
driver's timed scope, so the residual can be attributed instead of guessed.

Modes::

    # construction breakdown + driver-style scope, CUDA already warm (default)
    --fixture test/goldens/forest_random.npz

    # fresh-process mimic: nothing but the seeds before the scope, and no
    # synchronize before the wall clock starts (exactly run_matrix's eval path)
    --fixture test/goldens/forest_random.npz --scope-only --no-presync

Run with the old-side PYTHONPATH::

    PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils \
        python -u test/benchmark/xval/legacy_driver_scope_probe.py [flags]
"""

import argparse
import importlib.util
import os
import sys
import time


def load_driver():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "legacy", "run_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("run_matrix", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="test/goldens/forest_random.npz")
    parser.add_argument(
        "--no-presync",
        action="store_true",
        help="skip the probe's own pre-scope synchronize (run_matrix's eval path has "
        "no synchronize before its wall clock starts)",
    )
    parser.add_argument(
        "--scope-only",
        action="store_true",
        help="skip the construction-piece diagnostics (fresh-process mimic)",
    )
    return parser.parse_args()


def construction_breakdown(drv, A, torch, fixture):
    t0 = time.perf_counter()
    data = drv._fixture_arrays(fixture)
    t_load = ms(t0)

    t0 = time.perf_counter()
    tensors = [
        torch.tensor(data[k], device="cuda")
        for k in ("forest__node_value", "forest__node_type", "forest__subtree_size")
    ]
    torch.cuda.synchronize()
    t_make = ms(t0)

    t0 = time.perf_counter()
    forest = A.Forest(
        input_len=2,
        output_len=1,
        batch_node_value=tensors[0],
        batch_node_type=tensors[1],
        batch_subtree_size=tensors[2],
    )
    torch.cuda.synchronize()
    t_forest = ms(t0)

    t0 = time.perf_counter()
    problem, num_data = drv._fixture_problem(fixture)
    torch.cuda.synchronize()
    t_problem = ms(t0)

    # the harness-style construction (numpy -> .to("cuda")) for comparison
    t0 = time.perf_counter()
    for key in ("forest__node_value", "forest__node_type", "forest__subtree_size"):
        torch.from_numpy(data[key].copy()).to("cuda")
    torch.cuda.synchronize()
    t_alt = ms(t0)

    t0 = time.perf_counter()
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    t_eval = ms(t0)

    print(
        f"np.load={t_load:.2f} ms | torch.tensor(device=cuda) x3={t_make:.2f} ms | "
        f"Forest(...)={t_forest:.2f} ms | _fixture_problem={t_problem:.2f} ms | "
        f"from_numpy().to(cuda) x3={t_alt:.2f} ms | first evaluate={t_eval:.2f} ms "
        f"| num_data={num_data} fitness_shape={tuple(fitness.shape)}",
        flush=True,
    )


def driver_style_scope(drv, torch, fixture, presync):
    """Replicate ``run_matrix.run_eval_config``'s window: wall starts before build_c5."""
    if presync:
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    problem, forest, _ = drv.build_c5(fixture)
    per_call = []
    first_presync_ms = None
    for _ in range(25):
        t_sync = time.perf_counter()
        torch.cuda.synchronize()
        if first_presync_ms is None:
            first_presync_ms = ms(t_sync)
        tic = time.perf_counter()
        problem.evaluate(forest)
        torch.cuda.synchronize()
        per_call.append(ms(tic))
    wall = (time.perf_counter() - wall_start) * 1000.0
    warm = sorted(per_call[1:])
    print(
        f"driver-style scope (build + 25 calls): wall={wall:.1f} ms, "
        f"sum(per_call)={sum(per_call):.1f} ms, residual={wall - sum(per_call):.1f} ms, "
        f"first untimed pre-call synchronize()={first_presync_ms:.1f} ms, "
        f"first call={per_call[0]:.2f} ms, warm-median={warm[len(warm) // 2]:.3f} ms",
        flush=True,
    )


def main():
    args = parse_args()
    drv = load_driver()
    A = drv.api()
    torch = A.torch
    # mimic the driver's logging header, which touches the device before its wall clock
    torch.cuda.get_device_name(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print(
        f"side={A.side} evogp_file={A.evogp_file} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"scope_only={args.scope_only} presync={not args.no_presync}",
        flush=True,
    )
    if not args.scope_only:
        construction_breakdown(drv, A, torch, args.fixture)
    driver_style_scope(drv, torch, args.fixture, presync=not args.no_presync)
    return 0


if __name__ == "__main__":
    sys.exit(main())
