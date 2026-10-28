# `src/` — install root for the `evogp` Python package
## Intent
This directory exists only to make `src/evogp/` the distribution root (`src` layout).
`setup.py` at the repo root declares `packages=find_packages(where="src")` and `package_dir={"": "src"}`, so the import root is `evogp` (never `src.evogp`) and every import in the repo is written as `from evogp.<subpackage> import ...`.
It contains a single child and no other files; there is no code, no `__init__.py` and no packaging metadata at this level.
## Routing Table
- `src/evogp/` → The entire EvoGP library (tree representation + evoxc engine — `tree/evoxc_engine.py` plus the dense all-ops route in `tree/dense_engine.py` — GP operators, problems, pipeline, packaging-free checks, and the pytest suite). See `src/evogp/CONTEXT.md`.
- Sibling paths (read-only from here — escalate writes to this node's parent): `../setup.py` (packaging metadata / extras), `../requirements.txt`, `../README.md` (install + API docs), `../example/`, `../tutorial/`, `../test/`.
