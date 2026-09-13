#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r5/gpu_probe/run_rows.sh -- R5 Q2(c): the --data-rows
#   row-scaling series on the dense-iree PURE route.
#
# Q2 asks what the compiled dense graph puts on the GPU.  The dispatch dumps
# (gpu_probe/dispatch_dump.py) give its static shape; this runner gives the
# *row* dependence of the same pure forward, i.e. the slope/intercept of
# "warm per-invocation ms vs data rows" for a fixed (population, trace) pair.
# Every cell is the frozen r4 pure-dense-iree invocation
# (test/benchmark/r4/run_pure.sh, config pure64_dense_iree) plus `--data-rows N
# --output-checksum`; changing rows changes the x shape => a NEW compile, so
# one fresh process per row count and the construct/compile time is recorded
# separately (never mixed into the warm number).
#
# Run from the REPOSITORY ROOT with the project venv and a CUDA-visible GPU:
#
#     bash test/benchmark/r5/gpu_probe/run_rows.sh              # the full series
#     bash test/benchmark/r5/gpu_probe/run_rows.sh --only f64_r8,f64_r256
#     bash test/benchmark/r5/gpu_probe/run_rows.sh --list
#     bash test/benchmark/r5/gpu_probe/run_rows.sh --dry-run
#
# Rules baked in (same as every r2/r4/r5 runner):
#   * `ulimit -n 65536` first (IREE opens hundreds of fds).
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per cell, never
#     nested; the `timeout` sits INSIDE the lock so lock wait is never charged.
#   * one bench_gp.py process per cell (per-process ETL CompileCache).
#   * GPU 6 by default; GPU 7 has a hardware ECC fault and is never used.
#   * a failed cell never aborts the batch: log + FAILED + keep going.
#
# Env overrides: PY, CUDA_VISIBLE_DEVICES, TIMEOUT_S (default 900), LOCK,
# FLOCK_WAIT, NEW_PYTHONPATH (default `src`), ROWS_RESULT_DIR
# (default test/benchmark/r5/results/q2_gpu/rows).
# =============================================================================
set -u

R5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
TIMEOUT_S="${TIMEOUT_S:-900}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"
ROWS_RESULT_DIR="${ROWS_RESULT_DIR:-test/benchmark/r5/results/q2_gpu/rows}"
FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz
ETL_REPO="${ETL_REPO:-/mnt/local-ssd/bchuang/etl}"

# id -> fixture + rows
CELLS=(f64_r8 f64_r16 f64_r32 f64_r64 f64_r128 f64_r256 f200_r64 f200_r128)
cell_spec() {
  case "$1" in
    f64_r8)   CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=8 ;;
    f64_r16)  CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=16 ;;
    f64_r32)  CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=32 ;;
    f64_r64)  CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=64 ;;
    f64_r128) CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=128 ;;
    f64_r256) CELL_FIXTURE="$FIXTURE64";  CELL_ROWS=256 ;;
    f200_r64) CELL_FIXTURE="$FIXTURE200"; CELL_ROWS=64 ;;
    f200_r128) CELL_FIXTURE="$FIXTURE200"; CELL_ROWS=128 ;;
    *) return 1 ;;
  esac
  return 0
}

usage() {
  echo "usage: run_rows.sh [--only id1,id2] [--list] [--dry-run]"
  echo "       ids: ${CELLS[*]}"
}
SELECTED=()
ONLY=""
DRY_RUN=0
while (( $# > 0 )); do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --list) SELECTED=("${CELLS[@]}"); ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a SELECTED <<< "$ONLY"
  for id in "${SELECTED[@]}"; do
    cell_spec "$id" || { echo "run_rows.sh: unknown cell id '$id'" >&2; exit 2; }
  done
fi
if [[ ! -f "$BENCH" ]]; then echo "run_rows.sh: run me from the repository root ($BENCH not found)" >&2; exit 2; fi
if [[ ! -x "$PY" ]]; then echo "run_rows.sh: interpreter '$PY' not executable" >&2; exit 2; fi
if [[ ${#SELECTED[@]} -eq 0 ]]; then SELECTED=("${CELLS[@]}"); fi

if (( DRY_RUN == 0 )); then
  mkdir -p "$ROWS_RESULT_DIR"
  touch "$ROWS_RESULT_DIR/rows_manifest.txt"
  if [[ ! -f "$ROWS_RESULT_DIR/rows_meta.txt" ]]; then
    {
      echo "=== r5 Q2(c) row-scaling series -- $(date '+%Y-%m-%dT%H:%M:%S%z') ==="
      echo "evogp:   $(git rev-parse HEAD 2>/dev/null) ($(git status --porcelain 2>/dev/null | wc -l) dirty files)"
      echo "etl:     $(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null) ($ETL_REPO)"
      echo "python:  $PY"
      echo "gpu:     CUDA_VISIBLE_DEVICES=$GPU"
    } | tee "$ROWS_RESULT_DIR/rows_meta.txt"
  fi
fi

run_one() {
  local id="$1"
  cell_spec "$id" || { echo "run_rows.sh: unknown cell id '$id'" >&2; return 1; }
  local json="$ROWS_RESULT_DIR/$id.json"
  local log="$ROWS_RESULT_DIR/$id.log"
  local args="--task eval-pure --fixture $CELL_FIXTURE --mode dense --backend iree --reps 5 \
--warmup 10 --invocations 100 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400 \
--data-rows $CELL_ROWS --output-checksum --tag $id --json-out $json"
  if (( DRY_RUN == 1 )); then
    echo "DRY: flock -w $FLOCK_WAIT $LOCK bash -c 'ulimit -n 65536; timeout -s INT -k 30 $TIMEOUT_S env PYTHONPATH=$NEW_PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU ETL_DISABLE_LOCATIONS=1 EVOGP_EVAL_MODE=dense EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 $PY $BENCH $args'"
    return 0
  fi
  echo "--- $id (rows=$CELL_ROWS, $CELL_FIXTURE) ---"
  local rc=0
  flock -w "$FLOCK_WAIT" "$LOCK" bash -c "ulimit -n 65536; \
    timeout -s INT -k 30 $TIMEOUT_S env PYTHONPATH=$NEW_PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU \
    ETL_DISABLE_LOCATIONS=1 EVOGP_EVAL_MODE=dense EVOGP_ETL_BACKEND=iree \
    EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 $PY $BENCH $args" \
    > "$log" 2>&1 || rc=$?
  printf '%-10s exit=%d  %s\n' "$id" "$rc" "$json" | tee -a "$ROWS_RESULT_DIR/rows_manifest.txt"
  if (( rc != 0 )); then echo "run_rows.sh: cell '$id' FAILED (exit $rc), see $log" >&2; fi
  return 0
}

FAILED=0
for id in "${SELECTED[@]}"; do
  run_one "$id" || FAILED=1
done
exit "$FAILED"
