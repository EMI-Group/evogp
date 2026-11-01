# `src/` — install root for the `evogp` Python package
## Intent
This directory exists only to make `src/evogp/` the distribution root (the `src` layout).
`setup.py` at the repo root declares `packages=find_packages(where="src")` and `package_dir={"": "src"}`, so the import root is `evogp` (never `src.evogp`) and every import in the repo is written as `from evogp.<subpackage> import ...`.
It contains a single child and no other files: there is no code, no `__init__.py` and no packaging metadata at this level.
## Routing Table
- `src/evogp/` → the whole library: the torch-free functional GP stack `gp/`, the optional Triton route `fused/`, the EvoX 2.0 adapter `evox_ext/`, and the two shared modules `_tables.py` / `_multi_output.py`. See `src/evogp/CONTEXT.md`.
- `src/evogp/gp/` → the functional, torch-free GP stack (config/state dataclasses, encoding, the operator layer, the numpy evaluator + in-graph problem layer, the device-resident `graph_eval` route, the one `fused_eval` torch boundary, the `init_gp`/`step`/`run` driver, the smoke run and the measurement harnesses), built on `evoxc` and numpy. See `src/evogp/gp/CONTEXT.md`.
- `src/evogp/fused/` → the optional device-native Triton evaluation route (ONE fused kernel launch per population, JIT-compiled at the first call, the `fused` extra = torch + triton), its pytest net and its `python -m evogp.fused.check` battery. See `src/evogp/fused/CONTEXT.md`.
- `src/evogp/evox_ext/` → the EvoX 2.0 component adapter: the GP algorithm and the SR problem as `evox.core.StdWorkflow` components over `evogp.gp`, and the only subtree that imports `evox`. See `src/evogp/evox_ext/CONTEXT.md`.
- `src/evogp/_tables.py` → the node-type / function-id tables (`DELTA`, `MAX_STACK`, `MAX_FULL_DEPTH`, `NType`, `Func`, `FUNCS`, `FUNCS_NAMES`, `FUNC_IDS`, `FUNC_ARITY`, `FUNC_TYPE`, `MAX_OUT_LEN`, `OUT_NODE`), the single source of truth, importing nothing at all.
- `src/evogp/_multi_output.py` → the one single-output `OUT_NODE` refusal rule (`MultiOutputError`, `check_multi_output`, `find_tagged_node`, `out_node_message`), numpy only.
- Environment knobs live in the module that reads them (`EVOGP_FUSED_*` in `fused/`, `EVOGP_GP_*` in `gp/`), together with `EVOXC_DISABLE_LOCATIONS` and `TRITON_CACHE_DIR`; the authoritative list is in `src/evogp/CONTEXT.md`.
- Sibling paths (read-only from here — escalate writes to this node's parent): `../setup.py` (packaging metadata, the `torch` optional extra and the `fused` extra = torch + triton), `../requirements.txt`, `../README.md` (installation + API docs), `../example/` (runnable end-to-end scripts), `../tutorial/` (notebooks), `../test/` (dev tooling, including the numeric oracle at `test/oracle/`).
