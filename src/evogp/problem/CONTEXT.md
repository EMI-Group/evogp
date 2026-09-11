# `src/evogp/problem` — benchmark / task layer

## Intent
Defines the objective functions ("problems") that EvoGP optimizes: each class turns a population of GP trees into a per-individual fitness vector.
Three benchmark families live here: machine-learning tasks (symbolic regression, classification, feature transformation) and robotics-control tasks (Brax / MuJoCo-Playground / Genesis simulators).
This is the only layer that knows the numerical meaning of a task — `pipeline/` and `algorithm/` only ever see the fitness numbers.
Data tensors are created with `device="cuda"` throughout; there is no CPU code path in this directory (the same "cuda" strings also select the GPU under ROCm/HIP builds, see README).

## Fitness contract (applies to every class here)
`evaluate(forest) -> torch.Tensor` of shape `(pop_size,)`, dtype float32, on CUDA; **higher is better** (maximization).
Every class in this directory already returns a *maximized* score: `-MSE`, accuracy, `|correlation|`, or accumulated episode return.
The consumer is `StandardPipeline.step` (`src/evogp/pipeline/standard.py:40-52`), which does `fitnesses[torch.isnan(fitnesses)] = -torch.inf` **in place**, takes `argmax` for the best tree, then forwards the tensor to `GeneticProgramming.step`.
`GeneticProgramming.step` asserts `fitness.shape == (forest.pop_size,)` and all selection operators sort **descending** (higher = better), so a problem that returns a plain error value (lower-is-better) will silently optimize the wrong direction.
NaN/`inf` are tolerated (NaN is remapped to `-inf` by the pipeline); returning `None` or a non-tensor breaks the pipeline immediately.
Absolute scale is irrelevant, only monotonicity matters — `StandardPipeline(fitness_target=...)` and its printed stats must be interpreted in the problem's own units (e.g. `0.0` for MSE, `1.0` for correlation/accuracy).

## API Surface
`__init__.py` re-exports exactly: `BaseProblem`, `SymbolicRegression`, `Classification`, `Transformation`, `BraxProblem`, `GenesisProblem`, `MujocoProblem` (in that order).
`CustomLoss` and `Go2Env` are **not** re-exported (see module map below).

### `base.py` — `BaseProblem` (abstract contract)
Plain class, no ABC/metaclass; declares `evaluate(self, forest)`, `problem_dim` (property) and `solution_dim` (property), all raising `NotImplementedError`.
`problem_dim`/`solution_dim` are the numbers fed into `GenerateDescriptor(input_len=..., output_len=...)` in every example, so they must be `int`s that match the forest the user builds.
`BaseProblem` is used only as a type hint in `StandardPipeline.__init__`; the dimension cross-checks in the pipeline are commented out (`standard.py:22-23`), so a mismatch surfaces later inside `Forest.batch_forward`/`SR_fitness` asserts.

### `symbolic_regression.py` — `SymbolicRegression`
`__init__(datapoints=None, labels=None, func=None, num_inputs=None, num_data=100, lower_bounds=-1, upper_bounds=1, execute_mode="auto")`.
If `datapoints` **and** `labels` are both given they are stored verbatim (checked only for `is not None`); otherwise `func`+`num_inputs` are required and `generate_data` synthesizes data on the fly.
`generate_data` builds `inputs` as `(num_data, num_inputs)` uniform samples on cuda and `outputs = torch.vmap(func)(inputs)` — `func` receives a **1-D** tensor of length `num_inputs` and must return something reshapeable to a 1-D vector (README's example ends with `val.reshape(-1)`), giving `labels` shape `(num_data, 1)`.
`datapoints` must be 2-D `(num_data, num_inputs)` and `labels` must be 2-D `(num_data, output_len)`; `problem_dim = datapoints.shape[1]`, `solution_dim = labels.shape[1]`.
`evaluate(forest, use_MSE=True)` returns `-forest.SR_fitness(self.datapoints, self.labels, use_MSE, self.execute_mode)` (`symbolic_regression.py:83`) — ONE engine call for every `execute_mode` string. The engine (`etl_engine.sr_fitness`, `../tree/etl_engine.py:1200`) returns the *positive* loss `sum_N sum_out err / N`: errors are summed over all outputs and divided only by the number of data points, never by `output_len` (a perfect fit scores exactly 0; NaN propagates, and the pipeline maps it to `-inf`), and it is negated here because fitness is always maximized.
`execute_mode` is kept for backwards compatibility only: the constructor validates the five strings and all five now produce bit-identical fitness, so the former `kernel_type`/parallelism distinction is gone.
`use_MSE` is honored in every mode — `True` squared error, `False` absolute error.
A `CombinedForest` argument raises `NotImplementedError` (`symbolic_regression.py:70-77`) explaining that the multi-output combined path is driven by the `data_info` dict API (see `custom_loss.py`) and that the removed `formula=`/`share_input=` API no longer exists; only a plain `Forest` is evaluable here.

### `classification.py` — `Classification`
`__init__(datapoints=None, labels=None, dataset=None, multi_output=True)`.
Built-in datasets via scikit-learn: `"iris"`, `"wine"`, `"breast_cancer"`, `"digits"` (anything else raises `ValueError("Invalid dataset")`); `generate_data` converts them to float32 cuda tensors and casts labels to float (so `labels` is dtype float32 even though it holds class indices).
`labels` must be a **1-D** tensor of contiguous class indices starting at 0; `self.maximum = int(labels.max())` and the number of classes is `maximum + 1`.
`problem_dim = datapoints.shape[1]`, `solution_dim = maximum + 1` when `multi_output=True` else `1`.
Multi-output is *not* handled by `CombinedForest`: it is a single `Forest` whose `output_len` equals the class count, scored as accuracy — `softmax(outputs, dim=2)`, `argmax(dim=2)`, `sum(y_pred == labels, dim=1) / labels.shape[0]`, in `[0, 1]`.
With `multi_output=False` a single scalar output is mapped to a class index by `transform`: `clamp(round(x + maximum/2), 0, maximum)`, then the same accuracy formula.
`self.onehot_labels` is built in `__init__` but only used by the commented-out cross-entropy loss, i.e. it is currently dead weight.
`evaluate(forest)` takes no extra arguments and ignores `problem_dim` checks; it requires a plain `Forest` (a `CombinedForest` would fail because `batch_forward` would receive a tensor instead of a dict).

### `transformation.py` — `Transformation`
`__init__(datapoints=None, labels=None, dataset=None)`; only dataset `"diabetes"` exists (10 features, 1-D regression target); labels are 1-D.
`evaluate(forest)` measures how strongly each tree's output correlates with the target: `forest.batch_forward(datapoints).squeeze()`, demean outputs and labels by their global means, then `|Pearson correlation|` per tree.
Fitness is in `[0, 1]` (higher is better); constant/zero-variance tree outputs make the denominator zero and produce NaN, which the pipeline maps to `-inf` by design.
`new_feature(forest, n_best, n_features)` is the second half of the API: it keeps the `n_best` trees by fitness, computes pairwise `|corrcoef|` between their outputs, greedily drops the most correlated ones until at most `n_features` remain, and returns those outputs transposed as `(num_data, kept_features)` — ready for `torch.cat((problem.datapoints, problem.new_feature(...)), dim=1)` (the intended use, see `example/transformer.py`).
`problem_dim`/`solution_dim` are **not implemented** on this class (inherited `NotImplementedError`), even though examples index `transformation_problem[name]["input_len"]` by hand instead.

### Robotics-control family
`brax_problem.py` — `BraxProblem(env_name, max_episode_length, seed=42, pop_size=None, backend=None, output_transform=torch.tanh, jax_pre_allocate_memory=0.5)`.
`import_jax_based_package(frac)` sets `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]`, then declares `global jax, brax, jnp` and imports jax + `brax.envs`; it is called from `__init__`, so importing this module never requires jax.
`__init__` loads the env (`brax.envs.get_environment`), jit-vmaps `reset`/`step`, and stores a `jax.random.PRNGKey(seed)` (`self.batch_setp` is a typo for `batch_step`, cosmetic).
`evaluate(forest)` rolls out `max_episode_length` steps: jax obs → torch (`from_jax_array`), `forest.forward(obs)` (obs shape `(pop_size, observation_size)`), `output_transform`, torch action → jax (`to_jax_array`), `step`, accumulate `reward * ~done`, break early when all envs are done.
Returns the per-tree accumulated (undiscounted) episodic reward as a torch tensor via DLPack, higher is better; `pop_size` is latched from the first `evaluate` call (`len(forest)`).
`problem_dim = env.observation_size`, `solution_dim = env.action_size`.
`mujoco_playground.py` — `MujocoProblem(env_name, max_episode_length, seed=42, pop_size=None, output_transform=torch.tanh, jax_pre_allocate_memory=0.5)`: same rollout logic, env from `mujoco_playground.registry.load(env_name)`, plus a pytree-observation unwrap (`obs["state"]`, otherwise `ImportError("This Pytree observation space is not supported yet"))`.
It is a near-copy of `brax_problem.py` (`to_jax_array`/`from_jax_array` are byte-identical duplicates) and is currently non-functional — see Known Issues.
`genesis_problem.py` — `GenesisProblem(env_name, max_episode_length, seed=42, pop_size=None, backend=None, output_transform=torch.tanh, jax_pre_allocate_memory=0.5)`: a **stub**. `import_genesis_based_package` lazily `import genesis` (local scope, argument ignored), `evaluate` is `pass` (returns `None`), no `self.env` is ever assigned, and every constructor argument except being accepted is unused.
`get_cfgs()` in the same file returns the Go2 quadruped `(env_cfg, obs_cfg, reward_cfg, command_cfg)` dicts consumed by `go2_env.Go2Env`.
`go2_env.py` — `Go2Env`, a Genesis simulator wrapper for the Go2 robot (PD control, observation assembly, command resampling, termination checks, `_reward_*` functions). Not imported by `__init__.py` and not referenced anywhere else in the repo; it is scaffolding for the unfinished `GenesisProblem`.

### `custom_loss.py` — `CustomLoss` (not exported)
`CustomLoss(existing_data: dict, loss_func: Callable)` wraps a user-written loss so that arbitrary named variables can be optimized.
`inspect_function` (from `evogp.tree.utils`) extracts the loss function's parameter names; `torch.vmap(loss_func, in_dims=(None * len(existing_data)) + (0 * (len(parameters) - len(existing_data))))` maps over the parameters that are *not* keys of `existing_data`.
`evaluate(forest)` calls `forest.batch_forward(existing_data)` (so it requires a `CombinedForest`), squeezes each output's last dim, binds the parameters in signature order (`existing_data[name]` if present, otherwise the forest output), and returns `-loss` (higher is better).
Reachable only via `from evogp.problem.custom_loss import CustomLoss`; there is currently **no** example, notebook, README section or test using it.

## Module map / routing
- `base.py` → `BaseProblem` abstract contract every class here inherits.
- `symbolic_regression.py` → regression benchmark, the only problem that also accepts `CombinedForest`.
- `classification.py` → accuracy benchmark over sklearn datasets.
- `transformation.py` → feature-construction benchmark (correlation fitness + `new_feature`).
- `brax_problem.py` → Brax locomotion control (torch↔jax loop inside `evaluate`).
- `mujoco_playground.py` → MuJoCo Playground control (`MujocoProblem`); duplicated from brax module.
- `genesis_problem.py` → Genesis/Go2 control stub + `get_cfgs()` configs.
- `go2_env.py` → Genesis `Go2Env` simulator wrapper (not exported, top-level `import genesis`).
- `custom_loss.py` → user-defined loss adapter (`CustomLoss`, not exported).

Sibling directories (read-only for this node — escalate writes to the parent `./src/evogp`):
- `../tree/` → `Forest`/`CombinedForest`/`Tree`, `GenerateDescriptor`, `check_tensor`, `inspect_function`; owns `batch_forward`, `forward`, `SR_fitness` and the CUDA kernel bindings this layer calls.
- `../pipeline/standard.py` → the only in-repo consumer of `BaseProblem.evaluate`; defines the NaN→`-inf` and best-tree bookkeeping.
- `../algorithm/genetic_programming.py` → asserts `fitness.shape == (forest.pop_size,)`, consumes the tensor returned here.
- `../../example/` → de-facto usage documentation/smoke tests (`basic.py`, `regressor.py`, `classifier.py`, `transformer.py`, `brax_task.py`, `combined_sr.py`, `mjx_task.py`, `custom_sr.py`).
- `../../../tutorial/` → notebooks (`evogp_intro.ipynb`, `combined_gp.ipynb`) showing `SymbolicRegression`/`Classification` wiring.

## Constraints
The `evaluate` return value must be a 1-D torch tensor of length `pop_size` (asserted downstream) on any device, and must be writable (the pipeline assigns `-inf` into it in place).
Higher fitness must mean "better" for every problem class — negate error-style metrics as `SymbolicRegression` and `CustomLoss` do.
`problem_dim`/`solution_dim` must be ints consistent with the forest the user generates, because the pipeline's consistency asserts are disabled.
Input data (`datapoints`, `labels`) is expected to live on `cuda` and be float32; `Forest.batch_forward`/`SR_fitness` call `check_tensor`, which silently moves tensors to cuda, detaches and casts non-tensors, so a CPU tensor will not raise here but will silently be moved.
No problem in this layer is ever pure-torch: `Classification`, `Transformation`, `CustomLoss`, the robotics problems and even `SymbolicRegression(execute_mode="torch")` all forward through `Forest.batch_forward`/`Forest.forward`, which dispatch `torch.ops.evogp_cuda.tree_evaluate` (`src/evogp/tree/forest.py:163`, `src/evogp/tree/tree.py:61`); only the metric reduction (MSE/MAE, softmax+argmax, correlation) runs in Python/torch.
Do not reorder the imports in `__init__.py`: sibling modules use `from . import BaseProblem` (which requires `.base` to have been imported first by `__init__`), so `.base` must stay first.
Submodules must keep heavy optional dependencies **inside functions** (established pattern); a new top-level `import jax/genesis/...` breaks `import evogp.problem` for everyone.

## Optional-dependency matrix
| Module | External packages needed to import the module | Needed to actually run the problem |
|---|---|---|
| `base.py`, `custom_loss.py` | torch (+ evogp/tree) | — |
| `symbolic_regression.py` | torch | — |
| `classification.py`, `transformation.py` | torch only — scikit-learn is imported lazily inside `generate_data`, so a venv without sklearn can still `import evogp.problem` and use the `datapoints=`/`labels=` path | `scikit-learn` for the built-in `dataset=` loaders only |
| `brax_problem.py` | torch only (lazy bodies) | `jax`, `brax` (`setup.py` extra `brax = ["jax", "brax"]`) |
| `mujoco_playground.py` | torch only (lazy bodies) | `jax`, `mujoco_playground` (**no setup.py extra exists**) |
| `genesis_problem.py` | torch only (lazy body) | `genesis` (**no setup.py extra exists**) |
| `go2_env.py` | **`genesis` at module import time** | `genesis` + Go2 URDF assets (absent from the repo) |
`import_requires` is torch/numpy/scikit_learn/networkx/sympy, so scikit-learn is always available; jax/brax/genesis/mujoco_playground are not pinned anywhere, and README only documents the jax+brax installation note.

## Design Decisions
`BaseProblem` is deliberately minimal and interface-only, keeping `problem/` swappable and free of pipeline knowledge.
The problem layer owns data generation (`SymbolicRegression.generate_data`, sklearn loaders) so tests/examples are one-liners.
Fitness is *always* maximized (errors negated at the problem boundary) so selection operators can uniformly sort descending.
Robotics problems keep the simulator on the jax side and the GP trees on the torch side, bridging per step with zero-copy DLPack (`to_jax_array`/`from_jax_array`); `output_transform=torch.tanh` squashes raw tree outputs into the nominal action range.
Optional heavy packages are imported lazily inside functions so that `import evogp.problem` works on a plain torch install (only `brax_problem.py` does this correctly with `global jax, brax, jnp`; the mujoco copy is missing the `global` statement).
Classification solves multi-class by giving one tree multiple outputs (softmax over `output_len = num_classes`) rather than by composing several trees, which keeps a single `Forest` and the CUDA batch path usable.

## Known Issues
`MujocoProblem` (and `mujoco_playground.py`'s `to_jax_array`) is broken: `import_jax_based_package` imports `jax`/`jnp`/`MjxEnv`/`registry` as *locals* without a `global` declaration and the module never binds them globally, so `__init__` raises `NameError: name 'registry' is not defined` at `registry.load(...)` (`mujoco_playground.py:37`; the `jax.jit` at `:39` and `jnp.zeros` at `:56` would fail next), and module-level `to_jax_array` references the undefined global `jax` as well.
`SymbolicRegression.evaluate` rejects a `CombinedForest` with `NotImplementedError` by design (see API surface); `example/combined_sr.py`, `tutorial/combined_gp.ipynb` and `test/combined_test.ipynb` still call the removed `CombinedForest.random_generate(formula=..., share_input=...)` API and remain unrunnable — the supported route for combined problems is `evogp.problem.custom_loss.CustomLoss`.
`SymbolicRegression.generate_data` has a copy/paste bug in its scalar-bound checks (`if isinstance(upper_bounds, int) or isinstance(lower_bounds, float)`), so e.g. `lower_bounds=-1` (int) with `upper_bounds=1.0` (float) skips conversion and then fails on `1.0[None, :]`; also the `func`/`num_inputs` assertion is reached whenever only one of `datapoints`/`labels` is supplied, producing a confusing error.
`GenesisProblem` is a stub: `evaluate` returns `None`, `self.env` is never assigned (so `problem_dim`/`solution_dim` raise `AttributeError`), and all rollout arguments are ignored — do not advertise it as a working benchmark.
`go2_env.py` has a top-level `import genesis as gs`, so importing it fails without genesis (it is excluded from `__init__.py`, which is the only reason `import evogp.problem` still works); it also loads URDFs from relative paths (`urdf/plane/plane.urdf`, `urdf/go2/urdf/go2.urdf`) that do not exist in this repository, and its `_reward_tracking_lin_vel`/`_reward_base_height` read `reward_cfg["tracking_sigma"]`/`["base_height_target"]`, keys absent from `get_cfgs()`.
`Transformation` uses `.squeeze()` with no dimension argument, so a population of size 1 (or one data point) collapses the population dimension and the correlation math produces a scalar/shape error; `Transformation` also never implements `problem_dim`/`solution_dim`. Its `evaluate` demeans with the population-wide mean (`torch.mean(outputs)`), so one NaN-producing or constant-output tree turns the correlation of EVERY row into NaN — the pipeline then maps them all to `-inf` and selection degenerates for that generation.
`BraxProblem.evaluate` latches `self.pop_size` on first use, so reusing one problem instance with a different population size silently produces wrong-shaped rollouts; the problem also mutates global JAX state (`XLA_PYTHON_CLIENT_MEM_FRACTION`), which only takes effect before JAX initializes the GPU backend.
Robotics rollout results come back through DLPack from an immutable jax buffer while the pipeline writes `-inf` into the same tensor for NaN entries — treat the returned tensor as short-lived and do not rely on aliasing semantics.
`problem_dim`/`solution_dim` on `MujocoProblem`/`BraxProblem` return `env.observation_size`/`action_size`, which in current MuJoCo-Playground envs are pytrees/dicts for many environments, so the values are not always usable as `GenerateDescriptor` ints.
`example/mjx_task.py` imports `from problem import MujocoProblem` (wrong module path); it must be `from evogp.problem import MujocoProblem`.

## Notes for Agents
`problem/` depends on `tree/` and nothing inside `evogp` depends on `problem/` except `pipeline/standard.py` — so changes here cannot ripple into `tree/` or `algorithm/` (do not "fix" the interface to fit an example; fix the caller).
`problem/__init__.py` is the single source of truth for the public API; `CustomLoss` and `Go2Env` are intentionally (or accidentally) outside it.
`Classification(multi_output, dataset="iris")` in `example/classifier.py` passes `multi_output` **positionally**, where it lands in `datapoints`; it only works because `labels` is then `None` and the sklearn branch runs, and `multi_output` stays at its default `True`. Use keyword arguments.
`Classification` expects integer class indices in a 1-D float tensor (see `example/uci_classifier.py`, which slices `y_torch[:, 0]`); `SymbolicRegression` expects 2-D labels. Getting this wrong shows up as a broadcast error inside `evaluate`, not at construction time.
To add a benchmark, subclass `BaseProblem`, return a `(pop_size,)` maximized-cuda tensor from `evaluate`, expose int `problem_dim`/`solution_dim`, keep any new heavy import inside a function, and add the class to `__init__.py`.

## Test Strategy
There is no automated test for this directory: `test/` only contains `test_bind_success.py` (CUDA binding smoke test), `test_torch_env.py`, two notebooks and scratch scripts, none of which exercise `evogp.problem`.
The de-facto smoke tests are `example/basic.py` (torch-mode SR, XOR-3d), `example/regressor.py`, `example/classifier.py`, `example/transformer.py`, `example/brax_task.py`, plus `python -m evogp.sr_test` documented in the README.
Running any of them requires a CUDA (or ROCm/HIP) GPU, the compiled `evogp.evogp_cuda` extension, and float32 data on device; the robotics examples additionally require a jax build that can share the GPU with the GP kernels.
Verification without a GPU is limited to static inspection — the constructors touch `device="cuda"` unconditionally, so even `Classification(dataset="iris")` will not instantiate on CPU-only machines.
Known coverage gaps: no test for `CustomLoss`, for `execute_mode` equivalence (`"torch"` vs CUDA), for `Classification(multi_output=False)`, for `Transformation.new_feature`, for `use_MSE=False`, or for the robotics problems (all require external simulators).

## See Also
`../tree/CONTEXT.md` for `Forest.batch_forward`/`SR_fitness` shapes and the CUDA kernel entry points (`src/evogp/cuda/forward.cu`).
`../pipeline/CONTEXT.md` for the best-tree/NaN handling around `problem.evaluate`.
`README.md` "Supported Benchmarks" section for the user-facing description of `SymbolicRegression`, `Classification`, `Transformation` and `BraxProblem`.
