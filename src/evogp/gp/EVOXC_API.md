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
- `uniform(key, shape, low=0.0, high=1.0, dtype=float32)`, `normal(...)`, `randint(key, shape, low, high, dtype=int32)` (**high exclusive, low/high required**), `permutation(key, n)`, `multinomial(key, input, num_samples)` (**numpy backend only**).
- Ops never consume a key: same key + operands ⇒ bit-identical output. No global state, no `choice`/`fold_in`/`shuffle`/`Generator`.
- `low`/`high` may be broadcastable SymbolicTensors but not `np.float32` scalars.

## Control flow
- `evoxc.cond(pred, true_fn, false_fn, *operands)`: `pred` is a 0-d bool tensor, `*operands` are passed positionally to both branches; branches are plain callables (a `Defn` object raises).
- `evoxc.while_loop(cond_fn, body_fn, init)`: `init` **is** the carry; shapes may be symbolic but must be compatible per iteration.
- `evoxc.scan(f, init, xs, length=None) -> (carry, stacked)`: scanned axis is the leading axis of `xs`; **`length` must be static**.
- There is **no `jit`**; `@evoxc.defn` only marks a graph definition.

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
