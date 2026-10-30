# `src/evogp/gp/` — the evoxc-based functional GP stack (torch-free)
## Intent
This subtree is the new Evox-2.0-style EvoGP: a functional, torch-free genetic-programming stack built only on `evoxc` (EvoX Compiler) tensors and numpy host glue.
It exists to replace the Torch-based stack in `../tree/`, `../algorithm/`, `../problem/` and `../pipeline/` once parity is established; the old stack stays in place, unchanged, as the reference implementation and the migration source until then.
Design idiom is copied from Evox 2.0 (`/root/evox/src/evox/DESIGN.md`): frozen config dataclasses holding static leaves only, state as a dataclass whose leaves are `evoxc` tensors, and plain module-level functions in the `init(config, key) -> state` / `step(config, state, evaluate) -> state` step protocol.
`evox` 2.0 has no tree/GP/symbolic-regression component (verified), so this is EvoGP's contribution to the Evox family rather than a port of something existing.
## API Surface
- `GPTreeConfig` / `GPConfig` — frozen dataclasses: population size, `max_nodes` (static tree-buffer width), the function/terminal set, operator rates, selection mode. Static leaves only (Python scalars, tuples); never mutated, never traced.
- `GPState` — dataclass of `evoxc` tensors: the population encoding channels, the fitness vector and the RNG key. Every leaf shape is constant for a whole run (a shape change forces a re-trace/re-compile).
- `init(config, key) -> GPState`, `step(config, state, evaluate) -> GPState` — one full generation: select parents, crossover, mutate, `evaluate(offspring)`, elitist replacement, key advance.
- `evaluate(population) -> (pop_size,) float32` — caller-supplied, same maximized/negated-loss convention as the rest of the repository; the driver wires it to `evogp.tree`'s evoxc engines.
- `run(config, evaluate, generations=...)` — the plain driver loop (`workflow.py`); deliberately NOT `evox.StdWorkflow`, see Constraints.
## Constraints
- **No torch import anywhere in this subtree** (not even lazily). Host-side glue is numpy; device tensors are `evoxc` tensors. Torch may appear only in tests as an oracle.
- Tree encoding is static-shape: `[pop_size, max_nodes]` channels, prefix (pre-order) order, node 0 = root, padded roster past the row's live node count. Runtime variation lives in tensor VALUES only — never in shapes.
- EvoX Compiler has no eager mode: every `evoxc` op must run inside a trace (`evoxc.build`/`evoxc.build_cached`/`evoxc.run` of a built function). Module-level helpers that look eager must actually be trace-time functions.
- Randomness is stateless and key-based (`evoxc.random.key/split/split_n`); the key lives in `GPState` and is advanced there. There is no global seed.
- Known evoxc gaps to design around (verify before relying on the opposite): no `bitcast`, no `searchsorted`, no `nonzero`/runtime-shaped tensors, no `any`/`all` reducers, no in-place tensor write (`scatter` replaces but copies), `multinomial` is single-axis only.
- Numerics must match the semantics of record in `../tree/ENGINE_SPEC.md` (protected `DIV`, the `LOOSE_*` variants, `IF` short-circuit, ±1 comparison clamps).
- Fitness contract: `(pop_size,)` float32, MAXIMIZED (losses negated). Callers may not mutate it in place.
## Design Decisions
- **Driver is a plain loop, not `evox.StdWorkflow`.** `StdWorkflow` requires `evaluate` to be traced in-graph, but EvoGP evaluation is per-generation host-side engine work (per-structure `build_cached` specialization, or the shape-static dense/fused graph). `run()` therefore alternates one built operator graph per generation with one engine call, keeping one host sync per generation (the fitness readback), which is the existing hot-path budget.
- **Sibling subtree, not an in-place rewrite**: the torch stack keeps working (it is the oracle and the benchmark baseline) while this stack reaches parity; `tree/ENGINE_SPEC.md` + `tree/CONTEXT.md` are the shared semantics.
- **Evaluation is pluggable**: the operator layer owns population evolution, the engine owns evaluation. Any backend implementing `evaluate(population) -> fitness` (the fused Triton route, dense, specialized, numpy oracle) can drive it.
## Routing Table
(children are added as the stack grows: `encoding`, `operators`, `problem`, `workflow`, `evaluator`)
## Notes for Agents
- Read `../tree/CONTEXT.md` first for the tree representation and engine semantics, and `/root/evox/src/evox/DESIGN.md` for the functional style this subtree follows.
- Environment: `PYTHONPATH=src:/root/evoxc`, the torch-carrying venv `/tmp/rocvenv/bin/python` (needed only for the torch oracle/baseline), `EVOXC_DISABLE_LOCATIONS=1` for speed.
- The numpy evoxc backend is the correctness backend; `iree` is the device backend. Parity is checked on numpy first, then on the device.
