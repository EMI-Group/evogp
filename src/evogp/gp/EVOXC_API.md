# evoxc API contract for the functional GP stack (verified against `/root/evoxc`)

Verified by reading the evoxc sources (`evoxc/__init__.py`, `ops/*.py`, `random.py`, `trace/control_flow.py`, `pipeline.py`, `build_cache.py`, `core/*`).
Read this before writing any traced code here; every claim below is a source fact, not a guess.

## Build / run ladder
- `evoxc.build(fn, *specs, backend=..., device=..., **options) -> Executable`; `*specs` are `evoxc.TensorSpec(shape, dtype)` or **static Python values** (flattened as one positional pytree of `fn`).
- `evoxc.build_cached(fn, *specs, backend=..., device=..., cache=CompileCache(...), **options)` — keys on the static leaves, one cache entry per distinct program.
- `evoxc.lower(graph)` → `evoxc.compile(lowered)` → `evoxc.load(artifact)` is the only path that accepts a `Graph` (`build*` rejects one).
- **Executables are not callable** — no `__call__`, no `.run()`. Invoke only with `evoxc.run(exe_or_bound, *args)`.
- `evoxc.run` accepts raw `np.ndarray` leaves and returns `evoxc.Tensor` (multi-output = tuple); read back with `.numpy()`.
- `evoxc.evaluate(fn, *concrete_args)` is the example-args shorthand; it **cannot** take static args.
- Force a backend with `backend="numpy"` (default) / `backend="iree"`; device defaults to `cpu:0`.
- `evoxc.from_numpy(arr)` is a zero-copy host wrap; `evoxc.tensor(...)`, `evoxc.zeros/ones/full(...)` construct; `evoxc.constant(concrete_tensor)` embeds a graph constant.

## Tensor value model
- `evoxc.Tensor` exposes `dtype`, `shape`, `device`, `numpy()`, `to(device)` — and **no** `__getitem__`/`__setitem__`/`__len__`/`__array__`, and **no in-place write**.
- Inside a trace everything is a `SymbolicTensor`: `x[static_int_or_slice]` works; symbolic / boolean-mask / newaxis indexing raises `TraceError`.
- dtypes at the top level: `float16/32/64`, `int8/16/32/64`, `uint8/16/32/64`, `bool_`, `complex64/128`. Cast with `evoxc.cast(x, dtype)`.

## Ops (all top-level re-exports, all trace-time)
- Movement: `gather(x, indices, axis=0)`, `scatter(x, indices, updates, axis=0)` (**replacement-only, returns a copy**), `select(pred, on_true, on_false)`, `slice(x, start, lengths, strides=1)` (**static** lengths), `concatenate`, `stack`, `reshape`, `broadcast`, `transpose`, `tile`, `flip`, `roll`, `pad` (constant mode), `arange(start, stop, step)` (static bounds), `eye`, `linspace`.
- Sort/reduce-ish: `sort`, `argsort` (int64), `topk(x, k, axis=-1)`, `argmax`, `argmin`, `cumsum(x, axes=..., reverse=False)` — **`cumsum`/`cumprod` default `axis=0`**.
- Arithmetic: `add/subtract/multiply/divide/power/negate/remainder` and `+ - * / **`. **`//` and `%` do not exist** (use `evoxc.remainder`).
- Comparisons return **bool**: `equal, not_equal, less, less_equal, greater, greater_equal`, dunders `< <= > >= ==`. **`!=` is not hooked** — use `evoxc.not_equal`.
- Logic: `logical_and/or/not` (bool operands). **No `any`/`all`** (compose from a bool→int cast + `sum`).
- Math: `abs, sqrt, exp, log, log1p, sin, cos, tan, acos, tanh, sigmoid, floor, ceil, round, sign, relu, gelu, erf, square, maximum, minimum, clamp(x, min, max)`.
- Reductions take **`axes=`**, not `axis=`: `sum/max/min/mean/prod/var/std/nansum(x, axes=None, keepdims=False)`, `median(x, axis=...)`, `norm`.
- `where` is **not** top-level: use `evoxc.select` or `evoxc.numpy.where`. `unsqueeze`/`expand`/`expand_dims` only as `evoxc.numpy.expand_dims`.

## Random (stateless, key-based)
`evoxc.random` exports exactly: `key, split, split_n, uniform, normal, randint, permutation, multinomial, Algorithm, ALGORITHMS`.
- `key(seed, algorithm="splitmix64")` — rank-0 int64, usable both inside a trace (inlined constant) and outside (concrete `Tensor` you can pass to `run`).
- `split(key) -> (a, b)`; `split_n(key, n) -> n keys` — **only inside a trace** (`TraceError` outside; you cannot split at Python level).
- `uniform(key, shape, low=0.0, high=1.0, dtype=float32)`, `normal(...)`, `randint(key, shape, low, high, dtype=int32)` (**high exclusive, low/high required**), `permutation(key, n)`, `multinomial(key, input, num_samples)`.
- Backend support: `random_key_mix/uniform/normal/randint/permutation` all have StableHLO special emitters (`evoxc/backends/stablehlo/ops.py`, `random_export.py`), so they work on `backend="iree"` too — verified here by building one graph on numpy and on iree from the same key and getting identical values. **Only `multinomial` is deferred (numpy-only)** — never depend on it.
- Ops never consume a key: same key + operands ⇒ bit-identical output. No global state, no `choice`/`fold_in`/`shuffle`/`Generator`.
- `low`/`high` may be broadcastable SymbolicTensors but not `np.float32` scalars.

## Control flow
- `evoxc.cond(pred, true_fn, false_fn, *operands)`: `pred` is a 0-d bool tensor, `*operands` are passed positionally to both branches; branches are plain callables (a `Defn` object raises).
- `evoxc.while_loop(cond_fn, body_fn, init)`: `init` **is** the carry; shapes may be symbolic but must be compatible per iteration.
- `evoxc.scan(f, init, xs, length=None) -> (carry, stacked)`: scanned axis is the leading axis of `xs`; **`length` must be static**.
- There is **no `jit`**; `@evoxc.defn` only marks a graph definition.

## Device pointers: the DLPack hand-off (verified on this box, MI300X/ROCm, iree 3.11)
`evoxc` now implements the standard DLPack protocol on a device payload, so no private handle is needed to share a buffer with torch:
- `Tensor.__dlpack__(stream=None, max_version=None, copy=None)` returns a DLPack capsule for a DEVICE tensor, and `Tensor.__dlpack_device__()` returns `(device_type_code, device_id)` with codes kDLCUDA 2 and kDLROCM 10 and the 0-based ordinal as the id (torch calls a ROCm device `cuda`, so an evoxc `rocm` tensor arrives as `cuda:<index>`).
- `torch.from_dlpack(evoxc_device_tensor)` consumes it zero-copy (shape/dtype/values identical, a view at a byte offset included); a HOST payload keeps the old host semantics, and `numpy.from_dlpack` on a device tensor refuses by design.
- Refusals are explicit `DeviceError`s, not reinterpretations: `copy=True`, a foreign `dl_device` override, and an unsupported override dtype are rejected — so a caller that cannot share the pointer finds out instead of reading the wrong bytes.
- The RETURN direction has no native import (evoxc's `from_dlpack` is host-only via `np.from_dlpack`, and `external_call` refuses a device-resident return). The working idiom is a persistent `evoxc` device buffer: `buffer = evoxc.zeros((pop,), evoxc.float32).to(device)`, exported ONCE with `torch.from_dlpack`, handed to the Triton kernel as its `out=` destination (`fused_engine.sr_fitness_fused(..., out=view)`), and then passed to `variation` as the `fitness` operand — same storage, one write, no copy and no import step.
- `register_external_kernel(name, fn, backend="iree", device_resident=True)` additionally imports a `torch.Tensor` RETURNED by the kernel zero-copy; it costs one graph run around the call (+0.20 ms measured, 0.56 vs 0.36 ms direct) and a kernel that writes asynchronously must `torch.cuda.synchronize()` before returning, because iree's HAL uses its own stream and adds no fence.
- Any torch↔iree hand-off needs that fence in the direction torch writes: the model is "iree's driver and torch share the process' default stream", which the chained-generation parity test validates rather than the API text. `evogp.gp.fused_eval` therefore syncs once per device call, right after the kernel's write (it measured +0.030-0.035 ms per generation, 2.5-3.0 % of a generation at pop 256).
- A non-cpu payload's `.numpy()` raises `DeviceError` (no implicit D2H); the host copy is `t.to(Device("cpu", 0)).numpy()`.
## Confirmed gaps (design around them)
`bitcast` — no. `searchsorted` — no. `nonzero` / boolean masking — no. `any`/`all` — no. In-place writes — no. `unsqueeze`/`expand` — no (only `evoxc.numpy.expand_dims`). Dynamic shapes — yes, via symbolic `Dim`/`None` in `TensorSpec`, but numpy-backend only in practice.

## Minimal working numpy-backend idiom
```python
import numpy as np, evoxc

@evoxc.defn
def f(x, w):
    return evoxc.sum(evoxc.tanh(evoxc.dot(x, w)), axes=1)

cache = evoxc.CompileCache(maxsize=32)
exe = evoxc.build_cached(f, evoxc.TensorSpec((4, evoxc.dim("k")), evoxc.float32),
                         evoxc.TensorSpec((evoxc.dim("k"), 3), evoxc.float32),
                         backend="numpy", device="cpu", cache=cache)
out = evoxc.run(exe, np.zeros((4, 8), np.float32), np.ones((8, 3), np.float32)).numpy()
```

## Idioms to copy in evoxc's own tests
`tests/ops/test_structural.py:111-118` (shortest build + numpy in + `.numpy()` out), `tests/test_meta_programming.py:192-201` (GP-shaped `build_cached` loop over static trees with repeated `run`), `tests/ops/test_random.py:401-411` (concrete key as an explicit input), `tests/ops/conftest.py::run_numpy`.
