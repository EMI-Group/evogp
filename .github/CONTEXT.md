# `.github/` — the repository's only automation: the release pipeline
## Intent
This directory holds the one piece of automation the repository has: `.github/workflows/release.yml`, which turns a published GitHub Release into a tested wheel/sdist on PyPI plus the same files attached to that release.
Nothing here runs on a pull request or a push, and there is no CI gate on `main`: the test gate inside the workflow is the only automated run of the suite, and it is a release-time gate, not a per-commit one.
The workflow targets the `uv_build` backend declared in `pyproject.toml` (`uv build`), so `pyproject.toml` carries the version and `setup.py` is not consulted by the pipeline.
## API Surface / trigger surface
- Trigger `release: types: [published]` — publishing a GitHub Release (not drafting it, not pushing the tag) starts `build`, then `publish` and `attach`.
- Trigger `workflow_dispatch` with a required `tag` string input — a DRY RUN: `build` checks out that tag, verifies it against `[project].version`, runs the gate, builds and smoke-tests the wheel; `publish` and `attach` are skipped (`if: github.event_name == 'release'`).
- Job `build` (permissions `contents: read`, timeout 30 min): checkout at `${{ github.event.release.tag_name || inputs.tag }}` with `fetch-depth: 0`; `astral-sh/setup-uv@v6` (`python-version: "3.12"`, `enable-cache: true`); tag/version check; `uv run --with pytest --with evoxc --with numpy python -m pytest src/evogp/gp/tests -q` with `EVOXC_DISABLE_LOCATIONS=1`; `uv build`; install `dist/*.whl` into a fresh `uv venv` under `$RUNNER_TEMP`; `import evogp.gp` + `python -m evogp.gp.smoke` executed from `$RUNNER_TEMP` (never the checkout) with `EVOXC_DISABLE_LOCATIONS=1`; `actions/upload-artifact@v4` of `dist/*` as artifact `dist` (7-day retention).
- Job `publish` (needs `build`, `environment: pypi`, permissions `id-token: write` + `contents: read`): downloads artifact `dist` and runs `pypa/gh-action-pypi-publish@release/v1` with `packages-dir: dist/`; no API token, no `skip-existing` — a version already on PyPI fails the job rather than being silently ignored, because a duplicate upload means the tag was already released.
- Job `attach` (needs `build`, permissions `contents: write`): downloads artifact `dist` and runs `gh release upload "${RELEASE_TAG}" dist/* --clobber` with `GH_TOKEN: ${{ github.token }}` and `GH_REPO` set, so the step needs no checkout.
- `concurrency.group: release-<tag>` with `cancel-in-progress: false`: runs for one tag queue instead of racing, so one version is built and published once.
- `permissions: contents: read` at the top level is the minimal baseline; `id-token: write` and `contents: write` are granted only by the single job that needs each.
## Constraints
- PyPI Trusted Publisher must exist for project `evogp` on pypi.org with owner `EMI-Group`, repository `evogp`, workflow `release.yml` and environment `pypi`; without it `publish` fails at the OIDC exchange, and no API-token secret is configured by design.
- The `fused` and `evox_ext` suites do not run here: `src/evogp/fused/tests` needs torch + triton and a GPU, which no GitHub-hosted runner provides; only `src/evogp/gp/tests` (torch-free, numpy + evoxc) is wired into the gate.
- The tag must equal `v<version>` where `<version>` is `[project].version` in `pyproject.toml`; the build job strips the leading `v`, compares with `tomllib`, and fails with both values printed on any mismatch or a tag that does not start with `v`.
- A pre-release version such as `1.0.0a1` is uploaded to PyPI as a pre-release by PEP 440 semantics alone — no extra flag or index hack is needed — and the matching GitHub Release should be checked "Pre-release" so `releases/latest` and the badge do not point at an alpha.
- The wheel is smoke-tested from `$RUNNER_TEMP` on purpose: a src-layout checkout at the repo root would otherwise be a hazard for any future `pip install -e`, and the step asserts `evogp.gp.__file__` resolves inside `site-packages`.
- The workflow is validated by YAML parsing only (no `actionlint` in this repo): edit it carefully and re-parse after every change.
- Nothing else may be added to this directory without a reason — no PR CI, no lint, no docs deploy, no nightly.
## How to cut 1.0-alpha
1. Bump `[project].version` in `pyproject.toml` to `1.0.0a1` (the other documents of record quote the same string) and commit.
2. Tag and push: `git tag v1.0.0a1 && git push origin v1.0.0a1`.
3. Publish a GitHub Release for that tag, titled `EvoGP 1.0-alpha`, tick **Set as a pre-release**, and press "Publish release".
4. The workflow then verifies tag against version, gates on `src/evogp/gp/tests`, runs `uv build`, smoke-tests the wheel installed in a clean venv, uploads the `dist` artifact, publishes the wheel + sdist to PyPI (pre-release), and attaches the same files to the GitHub Release.
5. To rehearse without publishing, run the workflow manually (`workflow_dispatch`) with `tag: v1.0.0a1` and read the `build` job log.
## Routing Table
- `./workflows/` → the workflow files; today a single file, `release.yml`, documented here rather than in a nested CONTEXT.md.
## Notes for Agents
- `release.yml` is the whole automation surface: changing job names, the environment name or the workflow filename breaks the Trusted Publisher binding on PyPI.
- Shell steps are `bash` with `-eo pipefail` (workflow-level `defaults.run.shell`) and quote every variable; keep that style.
- Keep the dispatch dry run side-effect free: any new publishing step must carry `if: github.event_name == 'release'`.
## See Also
Parent `../CONTEXT.md` (repository layout and packaging metadata) and `../pyproject.toml` (the version the tag must match).
The test suites referenced here are described in `../src/evogp/gp/CONTEXT.md`, `../src/evogp/fused/CONTEXT.md` and `../src/evogp/evox_ext/CONTEXT.md`.
