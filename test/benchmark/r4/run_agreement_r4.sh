#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r4/run_agreement_r4.sh -- R4 backend-agreement pre-flight
#   numpy  vs  IREE-cuda O1  vs  IREE-cuda O1 + EVOGP_EVAL_MODE=specialized
# =============================================================================
#
# The R2/R3 gate that makes any IREE timing quotable (numpy == IREE within
# rtol=atol=1e-4 on the same fixture) re-run for R4, on the two fixtures the
# pure-forward matrix measures: the 64-tree golden fixture and the 200-tree R2
# fixture.  The `_spec` arms exist because the R4 default iree route is the DENSE
# evaluator while `*_specialized_*` configs pin the specialized route.
#
# One `check_backend_agreement.py` PROCESS per record, serialized on the shared
# benchmark lock.  Run it from the REPOSITORY ROOT:
#
#     test/benchmark/r4/run_agreement_r4.sh              # all six records
#     test/benchmark/r4/run_agreement_r4.sh --only fixture64_numpy
#     test/benchmark/r4/run_agreement_r4.sh --only fixture200_iree,fixture200_iree_spec
#     test/benchmark/r4/run_agreement_r4.sh --only fixture200_iree_spec --timeout 3600
#     test/benchmark/r4/run_agreement_r4.sh --dry-run    # print every command, run nothing
#     test/benchmark/r4/run_agreement_r4.sh --list       # print the record table
#
# `check_backend_agreement.py` is READ-ONLY R2 tooling: this runner drives it, never
# edits it.  The backend is chosen by the ENVIRONMENT (EVOGP_ETL_BACKEND), not by a
# flag, so each record is one fresh process with its own env block.
#
# Then the gate itself:
#
#     python3 test/benchmark/r4/check_agreement_r4.py
#
# Rules baked in here (same as the R2/R3 runners):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors.
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per record, never nested
#     (other agents share that lock), with the `timeout` INSIDE the lock so lock wait
#     is never charged to a record.
#   * a failing record never aborts the batch: log + FAIL + keep going.
#   * a record whose fixture is absent is GUARD-SKIPPED (exit 3, counts as OK) and
#     leaves no JSON behind, so check_agreement_r4.py reports it as missing.
#
# Environment overrides (all optional):
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault (never use it)
#   TIMEOUT_S             per-record wall cap, default 1800 (or pass --timeout SECONDS);
#                         `fixture200_iree_spec` compiles ~170 structures and is the
#                         slow record -- give it 1800s+ and never a short cap.
#   LOCK / FLOCK_WAIT     shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   NEW_PYTHONPATH        PYTHONPATH for the new side (default: src; prefix a patched
#                         ETL checkout to shadow it, exactly as in R3 Arm B)
#
# Exit codes:
#   0  every selected record exited 0 (or was guard-skipped, exit 3)
#   1  at least one record failed (crash 1, timeout 124/137, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown record, missing interpreter)
# Per-record evidence lands in
#   test/benchmark/r4/results/backend_agreement/<stem>.json          (driver output)
#   test/benchmark/r4/results/backend_agreement/<stem>_values.npy    (raw fitness vector)
#   test/benchmark/r4/results/backend_agreement/<stem>.log           (this runner's evidence log)
# and the record -> exit-code table is
#   test/benchmark/r4/results/backend_agreement/run_agreement_r4.last_run.txt  (disposable)
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_agreement_r4.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
DRIVER=test/benchmark/r2/check_backend_agreement.py
AGREE_DIR=test/benchmark/r4/results/backend_agreement
SUMMARY_FILE="$AGREE_DIR/run_agreement_r4.last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"

FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r4/run_agreement_r4.sh [--only stem1,stem2] [--timeout SECONDS]
                                            [--list] [--dry-run] [--help]

  (no arguments)  run all six records sequentially, in table order
  --only STEMS    run exactly these record stems (comma-separated); overrides the default set
  --timeout S     per-record wall cap fed to `timeout` INSIDE the lock (default: $TIMEOUT_S)
  --list          print the record table (stem, fixture, envs, json) and exit
  --dry-run       print every command without executing anything
  --help          this text

Records: fixture64_numpy, fixture64_iree, fixture64_iree_spec,
         fixture200_numpy, fixture200_iree, fixture200_iree_spec

Run from the repository root.  Gate: python3 test/benchmark/r4/check_agreement_r4.py
See test/benchmark/r4/README.md ("IREE agreement pre-flight") for the protocol.
EOF
}

ONLY=""
DRY_RUN=0
LIST=0
while (( $# > 0 )); do
  case "$1" in
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --timeout) TIMEOUT_S="${2:?--timeout needs a value}"; shift 2 ;;
    --timeout=*) TIMEOUT_S="${1#--timeout=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "run_agreement_r4.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

# --- 4. record table --------------------------------------------------------
# Order is the run order (do not sort): fixture64 (cheap) before fixture200.
ARM_IDS=(
  fixture64_numpy fixture64_iree fixture64_iree_spec
  fixture200_numpy fixture200_iree fixture200_iree_spec
)

# arm_spec STEM -> ARM_FIXTURE, ARM_ENVS (per-record backend/mode env), ARM_NOTE
# The common env of every record (PYTHONPATH, CUDA_VISIBLE_DEVICES,
# ETL_DISABLE_LOCATIONS=1, EVOGP_ETL_IREE_DEVICE=0, EVOGP_ETL_IREE_OPT_LEVEL=O1) is
# added in run_one(); only the backend (and the pinned route) differ per record.
arm_spec() {
  ARM_FIXTURE="" ; ARM_ENVS=() ; ARM_NOTE=""
  case "$1" in
    fixture64_numpy)      ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=numpy)
                          ARM_NOTE="64-tree fixture, numpy reference backend" ;;
    fixture64_iree)       ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree)
                          ARM_NOTE="64-tree fixture, IREE-cuda O1, default auto policy (dense route)" ;;
    fixture64_iree_spec)  ARM_FIXTURE="$FIXTURE64"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=specialized)
                          ARM_NOTE="64-tree fixture, IREE-cuda O1, forced specialized (R2-era route)" ;;
    fixture200_numpy)     ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=numpy)
                          ARM_NOTE="200-tree fixture, numpy reference backend (needs the r2 fixture)" ;;
    fixture200_iree)      ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree)
                          ARM_NOTE="200-tree fixture, IREE-cuda O1, default auto policy (dense route)" ;;
    fixture200_iree_spec) ARM_FIXTURE="$FIXTURE200"
                          ARM_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=specialized)
                          ARM_NOTE="200-tree fixture, IREE-cuda O1, forced specialized: ~170 cold builds, slow record" ;;
    *) return 1 ;;
  esac
  return 0
}

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$DRIVER" ]]; then
  echo "run_agreement_r4.sh: must be run from the REPOSITORY ROOT (need src/evogp and $DRIVER); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_agreement_r4.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
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
      echo "run_agreement_r4.sh: unknown record stem '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_agreement_r4.sh: --only is empty" >&2; exit 2; }
  for _id in "${ARM_IDS[@]}"; do      # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done
else
  SELECTED=("${ARM_IDS[@]}")
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_agreement_r4.sh: nothing selected" >&2; exit 2; }

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

# guard_reason STEM -> prints a skip reason (empty when the record is runnable)
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
  echo "-- per-record env (common + overrides) and pre-flight guard state --"
  for stem in "${SELECTED[@]}"; do
    arm_spec "$stem"
    printf '%-22s env: PYTHONPATH=%s ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=%s ' "$stem" "$NEW_PYTHONPATH" "$GPU"
    printf 'EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 %s\n' "${ARM_ENVS[*]}"
    reason="$(guard_reason "$stem")"
    if [[ -n "$reason" ]]; then
      printf '%-22s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one record ------------------------------------------------------
run_one() {
  local stem="$1"
  arm_spec "$stem" || { echo "run_agreement_r4.sh: unknown record stem '$stem'" >&2; return 1; }

  local json log
  json="$AGREE_DIR/$stem.json"
  log="$AGREE_DIR/$stem.log"

  # pre-flight guard: a record whose fixture is absent is "skipped by guard" (exit 3 == OK),
  # never a fake failure and never a silent omission.
  local reason; reason="$(guard_reason "$stem")"

  local -a child=(env "PYTHONPATH=$NEW_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$GPU"
                  ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)
  local e
  for e in "${ARM_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$DRIVER" "--fixture" "$ARM_FIXTURE" "--json-out" "$json")
  local -a timed=(timeout -s INT -k 30 "$TIMEOUT_S" "${child[@]}")
  local inner; inner="$(printf '%q ' "${timed[@]}")"; inner="${inner% }"

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
    echo "        (timeout -s INT -k 30 ${TIMEOUT_S}s runs INSIDE the lock)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "record $stem: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-22s %-6s %-9s %s  # guard: %s' "$stem" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$stem")
    return 0
  fi

  mkdir -p "$AGREE_DIR"
  {
    echo "==============================================================================="
    echo "R4 agreement record $stem  started $(date -Is)"
    echo "note: ${ARM_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "env: PYTHONPATH=$NEW_PYTHONPATH ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=$GPU EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 ${ARM_ENVS[*]}"
    echo "lock: exactly one flock -w $FLOCK_WAIT $LOCK, timeout -s INT -k 30 ${TIMEOUT_S}s INSIDE it"
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
    echo "R4 agreement record $stem  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "record $stem: $status (exit $rc) -> $json ; log $log"
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

echo "r4/run_agreement_r4.sh: py=$PY"
echo "r4/run_agreement_r4.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r4/run_agreement_r4.sh: only=${ONLY:-<all>} dry_run=$DRY_RUN records=$TOTAL timeout=${TIMEOUT_S}s lock=$LOCK"
echo "r4/run_agreement_r4.sh: gpu snapshot: $(gpu_line)"
if (( DRY_RUN )); then
  echo "r4/run_agreement_r4.sh: DRY RUN -- nothing is executed, no results dirs are created"
fi
echo

for stem in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$stem"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r4 run_agreement_r4.sh summary (CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-22s %-6s %-9s %s\n' RECORD EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} record(s) planned (dry run; $SUMMARY_FILE NOT written)"
  exit 0
fi

{
  echo "# test/benchmark/r4/results/backend_agreement/run_agreement_r4.last_run.txt -- record -> exit-code table"
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: only=${ONLY:-<all>} records=$TOTAL timeout=${TIMEOUT_S}s"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-22s %-6s %-9s %s\n' RECORD EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r4/check_agreement_r4.py"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
