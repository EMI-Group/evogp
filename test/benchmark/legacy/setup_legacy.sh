#!/usr/bin/env bash
# =============================================================================
# setup_legacy.sh — materialize + build + smoke-import the LEGACY EvoGP engine
# =============================================================================
# Side "OLD" of the benchmark described in ../SPEC.md: the pre-ETL revision
# `b613503`, whose evaluation engine is the hand-written CUDA extension
# `evogp.evogp_cuda` built from src/evogp/cuda/*.cu via setup.py's CUDAExtension.
# `src/evogp/tree/__init__.py` does `import evogp.evogp_cuda`, so the built
# extension is a hard prerequisite — there is no CPU fallback and no useful
# pure-python import path without it.
#
# Phases:
#   (a) materialize $EVOGP_LEGACY_DIR from `git archive <ref>` if missing
#   (b) export the known-good CUDA build env (see below)
#   (c) build the extension ONLY if the .so is missing
#   (d) smoke-import evogp.tree + evogp.evogp_cuda
#
# The build and the smoke phases are each wrapped in exactly ONE
# `flock -w 7200 /tmp/gpbench_critical.lock bash -c '<cmd>'`: the NEW-engine
# side of the benchmark is measured on the same shared box and takes the same
# lock, so waiting here is expected and fine. The flocks are never nested.
#
# Idempotent: once the .so exists, a re-run re-materializes nothing and re-runs
# only the smoke import (one lock acquisition).
#
# Env overrides (defaults are the values from ../SPEC.md):
#   EVOGP_LEGACY_DIR   checkout root          (default /tmp/evogp_legacy_bench)
#   EVOGP_LEGACY_REF   commit/tag to archive  (default b613503)
#   EVOGP_BENCH_LOGDIR log dir                (default <this dir>/results/logs)
#
# Usage:  bash test/benchmark/legacy/setup_legacy.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

BENCH_DIR="${EVOGP_LEGACY_DIR:-/tmp/evogp_legacy_bench}"
LEGACY_REF="${EVOGP_LEGACY_REF:-b613503}"
LOGDIR="${EVOGP_BENCH_LOGDIR:-$SCRIPT_DIR/results/logs}"
LOCK=/tmp/gpbench_critical.lock
LOCK_WAIT=7200

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
SITEUTILS=/tmp/evogp_t1a263/siteutils          # sklearn 1.7.2 + joblib/threadpoolctl/cloudpickle (no internet)
PYDEV=/tmp/evogp_t1a263/pydev                  # recovered CPython 3.11 headers (system python3.11-dev is absent)
SO_NAME=evogp_cuda.cpython-311-x86_64-linux-gnu.so

phase() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*"; }
die()   { printf '\n!! FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- (a) checkout
phase "materialize legacy checkout at $BENCH_DIR (ref $LEGACY_REF)"
if [[ -f "$BENCH_DIR/src/evogp/cuda/forward.cu" && -f "$BENCH_DIR/setup.py" ]]; then
    echo "SKIP materialize: checkout already present ($BENCH_DIR/src/evogp/cuda/forward.cu)"
else
    git -C "$REPO_ROOT" rev-parse --verify --quiet "${LEGACY_REF}^{commit}" >/dev/null \
        || die "commit '$LEGACY_REF' not found in $REPO_ROOT"
    mkdir -p "$BENCH_DIR" || die "cannot create $BENCH_DIR"
    # NOT a git checkout afterwards: the tarball carries no .git (see BUILD_NOTES.md).
    git -C "$REPO_ROOT" archive "$LEGACY_REF" | tar -x -C "$BENCH_DIR" \
        || die "git archive $LEGACY_REF | tar -x failed"
    echo "materialized $(git -C "$REPO_ROOT" rev-parse "$LEGACY_REF") -> $BENCH_DIR"
fi
[[ -f "$BENCH_DIR/src/evogp/cuda/forward.cu" ]] || die "missing src/evogp/cuda/forward.cu"
grep -q 'CUDAExtension' "$BENCH_DIR/setup.py"      || die "setup.py has no CUDAExtension"

# ---------------------------------------------------------------- (b) build env
# CUDA 12.6 toolchain from the wenet_tiktoken prefix. The system nvcc 11.8 is
# rejected by torch cu124, so CUDA_HOME/PATH must point here. cicc (and ptxas on
# other prefixes) live in targets/x86_64-linux/nvvm/bin — builds fail without it.
CUDA_ENV="${CUDA_ENV:-/opt/miniforge3/envs/wenet_tiktoken}"
export CUDA_HOME="$CUDA_ENV/targets/x86_64-linux"
export PATH="$CUDA_ENV/bin:$CUDA_ENV/targets/x86_64-linux/nvvm/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"          # /usr/bin/g++ -> g++-12 (CUDA 12.6 supports <= 13)
export CPATH="$PYDEV/usr/include/python3.11:$PYDEV/usr/include:$PYDEV/usr/include/x86_64-linux-gnu"
export CPLUS_INCLUDE_PATH="$CPATH"                       # Python.h is not in /usr/include/python3.11 on this box
export TORCH_CUDA_ARCH_LIST=8.6                          # RTX A6000
export MAX_JOBS="${MAX_JOBS:-8}"                         # NOTE: setup.py overrides this with os.cpu_count() (128)

phase "environment"
echo "REPO_ROOT   = $REPO_ROOT"
echo "LEGACY_REF  = $LEGACY_REF ($(git -C "$REPO_ROOT" rev-parse "$LEGACY_REF"))"
echo "BENCH_DIR   = $BENCH_DIR"
echo "PY          = $PY ($("$PY" -V 2>&1))"
echo "CUDA_HOME   = $CUDA_HOME"
echo "NVCC        = $(command -v nvcc) — $(nvcc --version | tail -1)"
echo "ccbin       = /usr/bin/g++ -> $(/usr/bin/g++ -dumpversion 2>/dev/null)"
echo "CPATH       = $CPATH"
echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST   MAX_JOBS(requested) = $MAX_JOBS"
echo "LOCK        = $LOCK (waits up to ${LOCK_WAIT}s per phase)"

# ---------------------------------------------------------------- (c) build
SO_PATH="$BENCH_DIR/src/evogp/$SO_NAME"
if [[ -f "$SO_PATH" ]]; then
    phase "build SKIPPED — extension already built"
    echo "$SO_PATH"
    ls -l "$SO_PATH"
else
    phase "build evogp.evogp_cuda (this is the only heavy step besides the smoke)"
    mkdir -p "$LOGDIR"
    BUILD_LOG="$LOGDIR/build_$(date +%Y%m%d_%H%M).log"
    echo "compiler output -> $BUILD_LOG"
    # ONE lock acquisition for the whole build. ${PIPESTATUS[0]} is expanded by
    # the inner bash so tee's status does not mask a compile failure.
    BUILD_CMD="cd '$BENCH_DIR' && MAX_JOBS=$MAX_JOBS '$PY' setup.py build_ext --inplace 2>&1 | tee '$BUILD_LOG'; exit \${PIPESTATUS[0]}"
    flock -w "$LOCK_WAIT" "$LOCK" bash -c "$BUILD_CMD"
    build_rc=$?
    [[ $build_rc -eq 0 ]] || die "build failed (rc=$build_rc); see $BUILD_LOG"
    [[ -f "$SO_PATH" ]] || die "build reported success but $SO_PATH is missing (see $BUILD_LOG)"
    echo "built $SO_PATH"
    ls -l "$SO_PATH"
fi

# ---------------------------------------------------------------- (d) smoke import
phase "smoke import (one lock acquisition)"
SMOKE_CMP="import evogp.tree; import evogp.evogp_cuda as m; print('ok')"
SMOKE_CMD="cd '$BENCH_DIR' && PYTHONPATH='$BENCH_DIR/src:$SITEUTILS' '$PY' -c \"$SMOKE_CMP\"; exit \$?"
flock -w "$LOCK_WAIT" "$LOCK" bash -c "$SMOKE_CMD"
smoke_rc=$?
[[ $smoke_rc -eq 0 ]] || die "smoke import failed (rc=$smoke_rc)"

phase "DONE"
echo "legacy checkout : $BENCH_DIR  (materialized via 'git archive $LEGACY_REF' — not a git repo)"
echo "extension       : $SO_PATH"
echo "smoke           : import evogp.tree + evogp.evogp_cuda -> ok"
exit 0
