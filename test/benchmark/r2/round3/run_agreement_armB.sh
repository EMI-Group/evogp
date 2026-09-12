#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r2/round3/run_agreement_armB.sh -- Arm B backend-agreement pre-flight
#   numpy  vs  IREE-cuda O1  vs  IREE-cuda O1 + EVOGP_EVAL_MODE=specialized
#   (all six arms against the PATCHED ETL checkout shadowed on PYTHONPATH)
# =============================================================================
#
# The R2 gate that makes any IREE timing quotable (numpy == IREE within
# rtol=atol=1e-4 on the same fixture) re-run as Arm B: the identical six arms as
# run_agreement_r3.sh (Arm A), but with the patched ETL checkout in front of the
# stock editable install --
#
#     NEW_PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141
#
# -- so the Arm B numeric gate is established for the patched engine, and
# `check_agreement_armB.py` can additionally require bit-identical fitness
# vectors against the same-stem Arm A records (results/backend_agreement/).
# The patched checkout is READ-ONLY: nothing here writes to it.
#
# One `check_backend_agreement.py` PROCESS per arm, serialized on the shared
# benchmark lock.  Run it from the REPOSITORY ROOT:
#
#     test/benchmark/r2/round3/run_agreement_armB.sh            # all six arms, sequentially
#     test/benchmark/r2/round3/run_agreement_armB.sh --only fixture64_numpy
#     test/benchmark/r2/round3/run_agreement_armB.sh --only fixture200_iree,fixture200_iree_spec
#     test/benchmark/r2/round3/run_agreement_armB.sh --dry-run  # print every command, run nothing
#     test/benchmark/r2/round3/run_agreement_armB.sh --list     # print the arm table
#
# `check_backend_agreement.py` is READ-ONLY R2 tooling: this runner drives it, never
# edits it.  The backend is chosen by the ENVIRONMENT (EVOGP_ETL_BACKEND), not by a
# flag, so each arm is one fresh process with its own env block.
#
# Rules baked in here (same as the matrix runner):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors.
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per arm, never nested
#     (other agents share that lock).
#   * no `timeout` wrapper inside the lock: the driver does exactly two
#     `problem.evaluate` calls and is bounded by its own (much shorter) work.
#   * a failing arm never aborts the batch: log + FAIL + keep going.
#   * an arm whose fixture is absent is GUARD-SKIPPED (exit 3, counts as OK) and
#     leaves no JSON behind, so check_agreement_armB.py reports it as missing.
#
# Environment overrides (all optional):
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault (never use it)
#   LOCK / FLOCK_WAIT     shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   NEW_PYTHONPATH        PYTHONPATH for the new side (default: `src` + the patched ETL checkout)
#
# Exit codes:
#   0  every selected arm exited 0 (or was guard-skipped, exit 3)
#   1  at least one arm failed (crash 1, timeout 124/137, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown arm, missing interpreter)
# Per-arm evidence lands in
#   test/benchmark/r2/round3/results/armB/backend_agreement/<stem>.json        (driver output)
#   test/benchmark/r2/round3/results/armB/backend_agreement/<stem>_values.npy  (raw fitness vector)
#   test/benchmark/r2/round3/results/armB/backend_agreement/<stem>.log         (this runner's log)
# and the report column for the id -> exit-code table is
#   test/benchmark/r2/round3/results/armB/backend_agreement/run_agreement_armB.last_run.txt
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_agreement_armB.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
DRIVER=test/benchmark/r2/check_backend_agreement.py
AGREE_DIR=test/benchmark/r2/round3/results/armB/backend_agreement
SUMMARY_FILE="$R3_DIR/results/armB/backend_agreement/run_agreement_armB.last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

# ARM B: the patched ETL checkout is shadowed onto PYTHONPATH ahead of the stock
# editable install; override NEW_PYTHONPATH to point at another checkout.
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src:/tmp/evogp_etl_patched_51a7141}"

FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r2/round3/run_agreement_armB.sh [--only stem1,stem2] [--list] [--dry-run] [--help]

  Arm B: the six R3 agreement arms re-run with a patched ETL checkout shadowed
  onto PYTHONPATH (default `src:/tmp/evogp_etl_patched_51a7141`); records go to
  results/armB/backend_agreement/.

  (no arguments)  run all six arms sequentially, in table order
  --only STEMS    run exactly these arm stems (comma-separated); overrides the default set
  --list          print the arm table (stem, fixture, envs, json) and exit
  --dry-run       print every command without executing anything
  --help          this text

Arms: fixture64_numpy, fixture64_iree, fixture64_iree_spec,
      fixture200_numpy, fixture200_iree, fixture200_iree_spec

Run from the repository root.  Arm B's numeric gate is
`python3 test/benchmark/r2/round3/check_agreement_armB.py`.
EOF
}

ONLY=""
DRY_RUN=0
LIST=0
while (( $# > 0 )); do
  case "$1" in
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "run_agreement_armB.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

# --- 4. arm table -----------------------------------------------------------
# Order is the run order (do not sort): fixture64 (cheap) before fixture200.
ARM_IDS=(
  fixture64_numpy fixture64_iree fixture64_iree_spec
  fixture200_numpy fixture200_iree fixture200_iree_spec
)

# arm_spec STEM -> ARM_FIXTURE, ARM_ENVS (per-arm env overrides), ARM_NOTE
arm_spec() {
  ARM_FIXTURE="" ; ARM_ENVS=() ; ARM_NOTE=""
  case "$1" in
    fixture64_numpy)      ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=numpy)
                          ARM_NOTE="64-tree fixture, numpy reference backend" ;;
    fixture64_iree)       ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)
                          ARM_NOTE="64-tree fixture, IREE-cuda O1, default auto policy (dense route)" ;;
    fixture64_iree_spec)  ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 EVOGP_EVAL_MODE=specialized)
                          ARM_NOTE="64-tree fixture, IREE-cuda O1, forced specialized (R2-era route)" ;;
    fixture200_numpy)     ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=numpy)
                          ARM_NOTE="200-tree fixture, numpy reference backend (needs r2 fixture)" ;;
    fixture200_iree)      ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)
                          ARM_NOTE="200-tree fixture, IREE-cuda O1, default auto policy (dense route)" ;;
    fixture200_iree_spec) ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 EVOGP_EVAL_MODE=specialized)
                          ARM_NOTE="200-tree fixture, IREE-cuda O1, forced specialized (R2-era route)" ;;
    *) return 1 ;;
  esac
  return 0
}

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$DRIVER" ]]; then
  echo "run_agreement_armB.sh: must be run from the REPOSITORY ROOT (need src/evogp and $DRIVER); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_agreement_armB.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi

# selection
SELECTED=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_raw <<< "$ONLY"
  _only=()
  for _id in "${_only_raw[@]}"; do
    _id="$(printf '%s' "$_id" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    [[ -z "$_id" ]] && continue
    if ! arm_spec "$_id"; then
      echo "run_agreement_armB.sh: unknown arm stem '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_agreement_armB.sh: --only is empty" >&2; exit 2; }
  for _id in "${ARM_IDS[@]}"; do      # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done
else
  SELECTED=("${ARM_IDS[@]}")
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_agreement_armB.sh: nothing selected" >&2; exit 2; }

# --- 6. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
}

status_of_rc() {
  case "$1" in
    0) echo "OK" ;;
    3) echo "SKIP" ;;          # guard-stop: counts as OK for the batch exit code
    124|137) echo "TIMEOUT" ;;
    130) echo "INTERRUPT" ;;
    *) echo "FAIL" ;;
  esac
}

# guard_reason STEM -> prints a skip reason (empty when the arm is runnable)
guard_reason() {
  local stem="$1"
  arm_spec "$stem" || return 0
  if [[ ! -f "$ARM_FIXTURE" ]]; then
    echo "fixture missing ($ARM_FIXTURE; relative to the repo root)"
    return 0
  fi
  return 0
}

# --- 7. --list --------------------------------------------------------------
if (( LIST )); then
  printf '%-22s %-52s %s\n' STEM FIXTURE JSON
  for stem in "${SELECTED[@]}"; do
    arm_spec "$stem"
    printf '%-22s %-52s %s\n' "$stem" "$ARM_FIXTURE" "$AGREE_DIR/$stem.json"
  done
  echo
  echo "-- per-arm env overrides (+ pre-flight guard state) --"
  for stem in "${SELECTED[@]}"; do
    arm_spec "$stem"
    printf '%-22s env: %s\n' "$stem" "${ARM_ENVS[*]}"
    reason="$(guard_reason "$stem")"
    if [[ -n "$reason" ]]; then
      printf '%-22s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one arm ---------------------------------------------------------
run_one() {
  local stem="$1"
  arm_spec "$stem" || { echo "run_agreement_armB.sh: unknown arm stem '$stem'" >&2; return 1; }

  local json log
  json="$AGREE_DIR/$stem.json"
  log="$AGREE_DIR/$stem.log"

  # pre-flight guard: an arm whose fixture is absent is "skipped by guard" (exit 3 == OK),
  # never a fake failure and never a silent omission.
  local reason; reason="$(guard_reason "$stem")"

  local -a child=(env "PYTHONPATH=$NEW_PYTHONPATH" ETL_DISABLE_LOCATIONS=1 "CUDA_VISIBLE_DEVICES=$GPU")
  local e
  for e in "${ARM_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$DRIVER" "--fixture" "$ARM_FIXTURE" "--json-out" "$json")
  local inner; inner="$(printf '%q ' "${child[@]}")"; inner="${inner% }"

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $stem"
    if [[ -n "$reason" ]]; then
      echo "        GUARD-SKIP: $reason   -> recorded as exit 3 (OK)"
    fi
    echo "        json: $json"
    echo "        log : $log"
    if [[ -n "$ARM_NOTE" ]]; then echo "        note: $ARM_NOTE"; fi
    # the inner command contains no single quotes by construction, so the printed
    # single-quoted form is exactly what `bash -c` receives at run time.
    echo "        flock -w $FLOCK_WAIT $LOCK bash -c '$inner'"
    echo "        (no timeout wrapper: the driver does two evaluate() calls)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "arm $stem: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-22s %-6s %-9s %s  # guard: %s' "$stem" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$stem")
    return 0
  fi

  mkdir -p "$AGREE_DIR"
  {
    echo "==============================================================================="
    echo "Arm B agreement arm $stem  started $(date -Is)"
    echo "note: ${ARM_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "env: PYTHONPATH=$NEW_PYTHONPATH ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=$GPU ${ARM_ENVS[*]}"
    echo "etl: $(env PYTHONPATH="$NEW_PYTHONPATH" "$PY" -c 'import etl; print(etl.__file__)' 2>&1 | tail -1)"
    echo "lock: exactly one flock -w $FLOCK_WAIT $LOCK, no timeout wrapper (driver does two evaluate() calls)"
    echo "gpu (before): $(gpu_line)"
    echo "cmd: $inner"
    echo "json: $json"
    echo "--- run output ---"
  } >> "$log"

  local rc
  flock -w "$FLOCK_WAIT" "$LOCK" bash -c "$inner" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}

  {
    echo "--- run output end ---"
    echo "exit code: $rc ($(status_of_rc "$rc"))"
    echo "gpu (after): $(gpu_line)"
    echo "Arm B agreement arm $stem  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "arm $stem: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-22s %-6s %-9s %s' "$stem" "$rc" "$status" "$json")")
  if (( rc == 0 || rc == 3 )); then
    OK+=("$stem")
  else
    FAILED+=("$stem")
  fi
  return 0
}

# --- 9. main loop -----------------------------------------------------------
SUMMARY=() ; OK=() ; SKIPPED=() ; FAILED=()
TOTAL=${#SELECTED[@]}
INDEX=0

echo "r3/run_agreement_armB.sh: py=$PY"
echo "r3/run_agreement_armB.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r3/run_agreement_armB.sh: only=${ONLY:-<all>} dry_run=$DRY_RUN arms=$TOTAL lock=$LOCK"
echo "r3/run_agreement_armB.sh: gpu snapshot: $(gpu_line)"
if (( DRY_RUN )); then
  echo "r3/run_agreement_armB.sh: DRY RUN -- nothing is executed, no results dirs are created"
fi
echo

for stem in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$stem"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r3 run_agreement_armB.sh summary (CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-22s %-6s %-9s %s\n' ARM EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} arm(s) planned (dry run; $SUMMARY_FILE NOT written)"
  exit 0
fi

{
  echo "# test/benchmark/r2/round3/results/armB/backend_agreement/run_agreement_armB.last_run.txt -- arm -> exit-code table"
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: only=${ONLY:-<all>} arms=$TOTAL"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-22s %-6s %-9s %s\n' ARM EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r2/round3/check_agreement_armB.py"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
