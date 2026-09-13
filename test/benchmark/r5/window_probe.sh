#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r5/window_probe.sh -- R5 Q4 same-window anchor window (GPU)
#   legacy CUDA engine  vs  ETL "true jit" engine (dense iree)  vs  pure forward
#
# R5 asks one question this script answers: does the r4 headline old-vs-new gap
# (~9.5-10.5x) survive when every side is measured in ONE serialized window?
# The r4 quotes (6.667 / 6.050 ms dense-iree vs 0.6323 / 0.6357 ms legacy) are
# cross-window; here the three measurement kinds sit next to each other in time:
#
#   q4_leg64 / q4_leg200        legacy CUDA end-to-end  problem.evaluate (type 1)
#   q4_eval64_iree / _200       ETL dense-iree end-to-end  problem.evaluate (type 1)
#   q4_pure64_iree / _200       ETL dense-iree PURE internal forward (type 2)
#   q4_eval64_iree_probe        q4_eval64_iree + --stage-probe (probe-tax bridge)
#   q4_leg64_bracket            q4_leg64 again at window end (drift bracket)
#   q4_eval64_iree_shadow87c7562 q4_eval64_iree on the r4 87c7562 ETL shadow (bridge)
#
# plus a 3-cell pre-window SMOKE group (short reps, into results/smoke) whose only
# job is to prove the invocations work before the window clock starts.
#
# One `bench_gp.py` PROCESS per cell, serialized on the shared benchmark lock.
# Run it from the REPOSITORY ROOT (the checkout that owns test/benchmark/):
#
#     test/benchmark/r5/window_probe.sh --group smoke      # 3 short sanity cells
#     test/benchmark/r5/window_probe.sh --group window     # the 9 window cells
#     test/benchmark/r5/window_probe.sh --only q4_leg64,q4_leg64_bracket
#     test/benchmark/r5/window_probe.sh --result-dir test/benchmark/r5/results/q4_window/rerun_1
#     test/benchmark/r5/window_probe.sh --list             # print the cell table
#     test/benchmark/r5/window_probe.sh --dry-run          # print every command, run nothing
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2).
#
# Rules baked in here (the R2/R3/R4 runner rules; do not "improve" them per-run):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors.
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per cell, never nested
#     (other agents share that lock), with the `timeout` INSIDE the lock so lock-wait
#     time is never charged to a cell.
#   * one process per cell => the engine's `etl.CompileCache` and the dense
#     evaluator's graph cache are per-process, so `first_call_ms` / cold-build counts
#     belong to that cell alone and `median_warm_ms` is steady state by construction.
#   * GPU 6 only (GPU 7 has a hardware ECC fault; GPU 0-5 carry foreign tenants).
#   * a failing cell never aborts the window: log + FAIL + record the deviation +
#     keep going (the bracket cell still brackets the cells that DID run).
#   * a cell's fixture / shadow checkout that is absent is GUARD-SKIPPED (exit 3).
#   * results are immutable: a re-run goes to a fresh --result-dir (rerun_<n>/), never
#     in place; the log is rewritten per cell, so never re-run a cell in a used dir.
#
# Environment overrides (all optional):
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 is refused here
#   LOCK / FLOCK_WAIT     shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   NEW_PYTHONPATH        new-side PYTHONPATH (default `src`; the shadow cell uses
#                         SHADOW_PYTHONPATH instead)
#   LEG_PYTHONPATH        legacy PYTHONPATH (default /tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils)
#   SHADOW_DIR / SHADOW_PYTHONPATH   the r4 87c7562 ETL shadow (bridge cell only)
#   ETL_REPO              ETL checkout whose HEAD is recorded (default /mnt/local-ssd/bchuang/etl, read-only)
#   WINDOW_RESULT_DIR     results dir (default: <r5>/results/q4_window, or <r5>/results/smoke for --group smoke)
#   CAP / CAP_LEGACY      per-cell wall caps (default 1800 s / 600 s), fed to `timeout` INSIDE the lock
#   DEVIATION_NOTE        free text appended verbatim to window_meta.txt's deviations section
#                         (e.g. what the smoke phase had to fix)
#
# Exit codes:
#   0  every selected cell was OK or guard-skipped (bench_gp.py exit 3 = skipped, OK)
#   1  at least one cell exited non-zero (timeout 124/137, crash 1, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown id, missing interpreter, GPU 7)
# Records land in
#   <results>/<id>.json + <results>/<id>.log   (log = run evidence incl. `etl:` identity + gpu samples)
#   <results>/window_meta.txt                  (the committed window record: start/end, per-cell
#                                               timestamps + exit codes, co-tenant samples, identity,
#                                               lock path, deviations)
# and the disposable id -> exit-code table in
#   <r5>/last_run.txt  (generated; .gitignore'd)
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "window_probe.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
SUMMARY_FILE="$R5_DIR/last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

LEGACY_ROOT="${LEGACY_ROOT:-/tmp/evogp_legacy_bench}"
LEGACY_SITEUTILS="${LEGACY_SITEUTILS:-/tmp/evogp_t1a263/siteutils}"
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"
LEG_PYTHONPATH="${LEG_PYTHONPATH:-$LEGACY_ROOT/src:$LEGACY_SITEUTILS}"
SHADOW_DIR="${SHADOW_DIR:-/tmp/evogp_etl_patched_87c7562}"
SHADOW_PYTHONPATH="${SHADOW_PYTHONPATH:-src:$SHADOW_DIR}"
ETL_REPO="${ETL_REPO:-/mnt/local-ssd/bchuang/etl}"

CAP="${CAP:-1800}"
CAP_LEGACY="${CAP_LEGACY:-600}"

# absolute fixture paths (resolved from the repo root; the records then carry them verbatim)
FIXTURE64="$REPO_ROOT/test/goldens/forest_random.npz"
FIXTURE200="$REPO_ROOT/test/benchmark/fixtures/forest200_tree64_random.npz"

# the R5 primary new-side env block (default `auto` route on iree => dense evaluator)
NEW_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r5/window_probe.sh [--group smoke|window|all] [--only id1,id2]
                                        [--result-dir DIR] [--list] [--dry-run] [--help]

  --group smoke   the 3 short pre-window sanity cells (results/smoke/)
  --group window  the 9 Q4 window cells (results/q4_window/), table order (default)
  --group all     smoke then window (they use DIFFERENT result dirs)
  --only IDS      run exactly these cell ids (comma-separated); overrides --group
  --result-dir D  write records into D instead of the group default (use rerun_<n>/ for a re-run)
  --list          print the cell table (id, side, group, cap, json, args) and exit
  --dry-run       print every command without executing anything
  --help          this text

Run from the repository root.  See test/benchmark/r5/README.md for the round protocol.
EOF
}

GROUP=window
ONLY=""
RESULT_DIR=""
DRY_RUN=0
LIST=0
while (( $# > 0 )); do
  case "$1" in
    --group) GROUP="${2:?--group needs a value}"; shift 2 ;;
    --group=*) GROUP="${1#--group=}"; shift ;;
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --result-dir) RESULT_DIR="${2:?--result-dir needs a value}"; shift 2 ;;
    --result-dir=*) RESULT_DIR="${1#--result-dir=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "window_probe.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  smoke|window|all) ;;
  *) echo "window_probe.sh: --group must be one of smoke|window|all (got '$GROUP')" >&2; exit 2 ;;
esac
if [[ "$GPU" == "7" ]]; then
  echo "window_probe.sh: refusing CUDA_VISIBLE_DEVICES=7 (hardware ECC fault on this box)" >&2
  exit 2
fi

# --- 4. cell table ----------------------------------------------------------
# Order is the run order (do not sort): the legacy / iree / pure triple per fixture first
# (so the same-window comparison sits in adjacent time), then the probe-tax bridge pair
# (probe cell next to its plain twin is impossible -- one process per cell -- so it runs
# right after the pure cells, still inside the window), then the drift bracket LAST,
# then the optional shadow bridge.
SMOKE_IDS=(
  smoke_leg64 smoke_eval64_iree smoke_pure64_iree
)
WINDOW_IDS=(
  q4_leg64 q4_leg200
  q4_eval64_iree q4_eval200_iree
  q4_pure64_iree q4_pure200_iree
  q4_eval64_iree_probe
  q4_leg64_bracket
  q4_eval64_iree_shadow87c7562
)

# cell_spec ID -> CONFIG_SIDE (leg|new), CONFIG_GROUP, CONFIG_ARGS, CONFIG_ENVS,
#                CONFIG_FIXTURE (existence-guarded; "" = nothing to guard),
#                CONFIG_PYTHONPATH ("" = side default), CONFIG_CAP, CONFIG_GUARD_DIR
#                (extra directory that must exist), CONFIG_NOTE
cell_spec() {
  CONFIG_SIDE="" ; CONFIG_GROUP="" ; CONFIG_ARGS=() ; CONFIG_ENVS=() ; CONFIG_FIXTURE=""
  CONFIG_PYTHONPATH="" ; CONFIG_CAP="$CAP" ; CONFIG_GUARD_DIR="" ; CONFIG_NOTE=""
  case "$1" in
    # ---- smoke group: short reps, results/smoke/ ---------------------------
    smoke_leg64)      CONFIG_SIDE=leg; CONFIG_GROUP=smoke; CONFIG_FIXTURE="$FIXTURE64"; CONFIG_CAP="$CAP_LEGACY"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 3)
                      CONFIG_NOTE="smoke (a): legacy end-to-end, 3 evaluates" ;;
    smoke_eval64_iree) CONFIG_SIDE=new; CONFIG_GROUP=smoke; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 3 --mem-guard-mb 8192)
                      CONFIG_ENVS=("${NEW_ENVS[@]}")
                      CONFIG_NOTE="smoke (b): ETL dense-iree end-to-end, 3 evaluates" ;;
    smoke_pure64_iree) CONFIG_SIDE=new; CONFIG_GROUP=smoke; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode dense
                        --backend iree --reps 1 --warmup 5 --invocations 20 --e2e-companion 3
                        --mem-guard-mb 8192 --max-cold-builds 400 --output-checksum)
                      CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                      CONFIG_NOTE="smoke (c): pure dense-iree, 20 invocations / 1 rep / 3 e2e companions" ;;

    # ---- window group: the Q4 anchor cells, results/q4_window/ -------------
    q4_leg64)         CONFIG_SIDE=leg; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"; CONFIG_CAP="$CAP_LEGACY"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 25)
                      CONFIG_NOTE="Q4 legacy end-to-end anchor, 64-tree fixture, 25 evaluates" ;;
    q4_leg200)        CONFIG_SIDE=leg; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE200"; CONFIG_CAP="$CAP_LEGACY"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25)
                      CONFIG_NOTE="Q4 legacy end-to-end anchor, 200-tree fixture, 25 evaluates" ;;
    q4_eval64_iree)   CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 25 --mem-guard-mb 8192)
                      CONFIG_ENVS=("${NEW_ENVS[@]}")
                      CONFIG_NOTE="Q4 ETL dense-iree end-to-end (auto route), 64-tree fixture" ;;
    q4_eval200_iree)  CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE200"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --mem-guard-mb 8192)
                      CONFIG_ENVS=("${NEW_ENVS[@]}")
                      CONFIG_NOTE="Q4 ETL dense-iree end-to-end (auto route), 200-tree fixture" ;;
    q4_pure64_iree)   CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode dense
                        --backend iree --reps 5 --warmup 10 --invocations 100 --e2e-companion 11
                        --mem-guard-mb 8192 --max-cold-builds 400 --output-checksum)
                      CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                      CONFIG_NOTE="Q4 pure internal dense forward, 64 trees (r4 pure64_dense_iree flags)" ;;
    q4_pure200_iree)  CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE200"
                      CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE200" --mode dense
                        --backend iree --reps 5 --warmup 10 --invocations 100 --e2e-companion 11
                        --mem-guard-mb 8192 --max-cold-builds 400 --output-checksum)
                      CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                      CONFIG_NOTE="Q4 pure internal dense forward, 200 trees (r4 pure200_dense_iree flags)" ;;
    q4_eval64_iree_probe) CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 25
                        --mem-guard-mb 8192 --stage-probe)
                      CONFIG_ENVS=("${NEW_ENVS[@]}")
                      CONFIG_NOTE="probe-tax bridge: q4_eval64_iree + --stage-probe" ;;
    q4_leg64_bracket) CONFIG_SIDE=leg; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"; CONFIG_CAP="$CAP_LEGACY"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 25)
                      CONFIG_NOTE="drift bracket = q4_leg64 repeated at the end of the window" ;;
    q4_eval64_iree_shadow87c7562) CONFIG_SIDE=new; CONFIG_GROUP=window; CONFIG_FIXTURE="$FIXTURE64"
                      CONFIG_PYTHONPATH="$SHADOW_PYTHONPATH"; CONFIG_GUARD_DIR="$SHADOW_DIR"
                      CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE64" --eval-reps 25 --mem-guard-mb 8192)
                      CONFIG_ENVS=("${NEW_ENVS[@]}")
                      CONFIG_NOTE="OPTIONAL bridge to the r4 headline: q4_eval64_iree on the 87c7562 ETL shadow" ;;
    *) return 1 ;;
  esac
  return 0
}

all_ids() { printf '%s\n' "${SMOKE_IDS[@]}" "${WINDOW_IDS[@]}"; }

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "window_probe.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "window_probe.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
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
    if ! cell_spec "$_id"; then
      echo "window_probe.sh: unknown cell id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "window_probe.sh: --only is empty" >&2; exit 2; }
  while IFS= read -r _id; do            # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done < <(all_ids)
else
  case "$GROUP" in
    smoke)  SELECTED=("${SMOKE_IDS[@]}") ;;
    window) SELECTED=("${WINDOW_IDS[@]}") ;;
    all)    SELECTED=("${SMOKE_IDS[@]}" "${WINDOW_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "window_probe.sh: nothing selected" >&2; exit 2; }

# group order matters for --group all: smoke cells write into results/smoke, the window
# cells into results/q4_window, so the result dir is derived PER CELL when --result-dir
# is not given explicitly.
cell_result_dir() {
  if [[ -n "$RESULT_DIR" ]]; then
    printf '%s' "$RESULT_DIR"
  elif [[ "${CONFIG_GROUP:-smoke}" == "smoke" ]]; then
    printf '%s' "test/benchmark/r5/results/smoke"
  else
    printf '%s' "test/benchmark/r5/results/q4_window"
  fi
}

# --- 6. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
}

gpu_tenants() {  # who else is on the device (co-tenancy evidence for the window record)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    -i "$GPU" 2>/dev/null | tr '\n' ';' | sed 's/;$//'
}

# etl_identity_line PYTHONPATH -> one line of evidence about WHICH etl checkout the cell imports.
# Cheap and failure-tolerant: a missing checkout / a broken import must never fail a cell.
etl_identity_line() {
  local pyp="$1" rev path
  rev="$(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"
  path="$(env "PYTHONPATH=$pyp" "$PY" -c 'import etl; print(etl.__file__)' 2>/dev/null \
          || echo '<import etl failed>')"
  echo "etl: repo=$ETL_REPO rev=$rev file=$path"
}

status_of_rc() {
  case "$1" in
    0) echo "OK" ;;
    3) echo "SKIP" ;;          # harness guard-stop: counts as OK for the batch exit code
    124|137) echo "TIMEOUT" ;;
    130) echo "INTERRUPT" ;;
    *) echo "FAIL" ;;
  esac
}

# guard_reason ID -> prints a skip reason (empty when the cell is runnable)
guard_reason() {
  local id="$1"
  cell_spec "$id" || return 0
  if [[ "$CONFIG_SIDE" == "leg" ]]; then
    if [[ ! -d "$LEGACY_ROOT/src" ]]; then
      echo "legacy checkout missing ($LEGACY_ROOT/src): run test/benchmark/legacy/setup_legacy.sh"
      return 0
    fi
    if [[ ! -d "$LEGACY_SITEUTILS" ]]; then
      echo "legacy offline siteutils missing ($LEGACY_SITEUTILS): run test/benchmark/legacy/setup_legacy.sh"
      return 0
    fi
  fi
  if [[ -n "$CONFIG_GUARD_DIR" && ! -d "$CONFIG_GUARD_DIR" ]]; then
    echo "ETL shadow checkout missing ($CONFIG_GUARD_DIR): materialize it read-only from the foreign repo"
    return 0
  fi
  if [[ -n "$CONFIG_FIXTURE" && ! -f "$CONFIG_FIXTURE" ]]; then
    echo "fixture missing ($CONFIG_FIXTURE)"
    return 0
  fi
  return 0
}

# --- 7. --list --------------------------------------------------------------
if (( LIST )); then
  printf '%-32s %-5s %-7s %-6s %-22s %s\n' ID SIDE GROUP CAP PYTHONPATH JSON
  for id in "${SELECTED[@]}"; do
    cell_spec "$id"
    case "$CONFIG_SIDE" in
      leg) pyp="$LEG_PYTHONPATH" ;;
      *)   pyp="${CONFIG_PYTHONPATH:-$NEW_PYTHONPATH}" ;;
    esac
    tmp_resdir="$(cell_result_dir)"   # cell_result_dir reads CONFIG_GROUP, so call after cell_spec
    printf '%-32s %-5s %-7s %-6s %-22s %s\n' "$id" "$CONFIG_SIDE" "$CONFIG_GROUP" "$CONFIG_CAP" \
      "${pyp:0:22}" "$tmp_resdir/$id.json"
  done
  echo
  echo "-- exact args per cell (+ per-cell env overrides, + pre-flight guard state) --"
  for id in "${SELECTED[@]}"; do
    cell_spec "$id"
    envs=""
    if (( ${#CONFIG_ENVS[@]} > 0 )); then envs="  env: ${CONFIG_ENVS[*]}"; fi
    printf '%-32s %s%s\n' "$id" "${CONFIG_ARGS[*]}" "$envs"
    reason="$(guard_reason "$id")"
    if [[ -n "$reason" ]]; then
      printf '%-32s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one cell --------------------------------------------------------
run_one() {
  local id="$1"
  cell_spec "$id" || { echo "window_probe.sh: unknown cell id '$id'" >&2; return 1; }

  local tag dir json log pyp
  tag="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  if [[ "$CONFIG_SIDE" == "leg" ]]; then
    pyp="$LEG_PYTHONPATH"
  else
    pyp="${CONFIG_PYTHONPATH:-$NEW_PYTHONPATH}"
  fi
  dir="$(cell_result_dir)"
  json="$dir/$id.json"
  log="$dir/$id.log"

  local reason; reason="$(guard_reason "$id")"

  local -a child=(env "PYTHONPATH=$pyp" "CUDA_VISIBLE_DEVICES=$GPU" ETL_DISABLE_LOCATIONS=1)
  local e
  for e in "${CONFIG_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$BENCH" "${CONFIG_ARGS[@]}" "--tag" "$tag" "--json-out" "$json")
  local -a timed=(timeout -s INT -k 30 "$CONFIG_CAP" "${child[@]}")
  local inner; inner="$(printf '%q ' "${timed[@]}")"; inner="${inner% }"

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $id  (side=$CONFIG_SIDE group=$CONFIG_GROUP cap=${CONFIG_CAP}s)"
    if [[ -n "$reason" ]]; then
      echo "        GUARD-SKIP: $reason   -> recorded as exit 3 (OK)"
    fi
    echo "        json: $json"
    echo "        log : $log"
    if [[ -n "$CONFIG_NOTE" ]]; then echo "        note: $CONFIG_NOTE"; fi
    echo "        flock -w $FLOCK_WAIT $LOCK bash -c '$inner'"
    echo "        (timeout -s INT -k 30 ${CONFIG_CAP}s runs INSIDE the lock; stdout+stderr -> tee -a $log)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "cell $id: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-32s %-5s %-6s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_SIDE" "$CONFIG_GROUP" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$id")
    DEVIATIONS+=("cell $id GUARD-SKIPPED: $reason (exit 3, counts as OK)")
    return 0
  fi

  mkdir -p "$dir"
  local t_start t_end
  t_start="$(date -Is)"
  {
    echo "==============================================================================="
    echo "R5 window cell $id  (side=$CONFIG_SIDE group=$CONFIG_GROUP cap=${CONFIG_CAP}s)  started $t_start"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "timeout: ${CONFIG_CAP}s (SIGINT, SIGKILL after 30s) INSIDE flock -w $FLOCK_WAIT $LOCK"
    echo "env: PYTHONPATH=$pyp CUDA_VISIBLE_DEVICES=$GPU ETL_DISABLE_LOCATIONS=1 ${CONFIG_ENVS[*]:-}"
    echo "$(etl_identity_line "$pyp")"
    echo "gpu (before): $(gpu_line)"
    echo "gpu tenants (before): $(gpu_tenants)"
    echo "cmd: $inner"
    echo "json: $json"
    echo "--- run output ---"
  } >> "$log"

  local rc
  flock -w "$FLOCK_WAIT" "$LOCK" bash -c "$inner" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  t_end="$(date -Is)"

  {
    echo "--- run output end ---"
    echo "exit code: $rc ($(status_of_rc "$rc"))"
    echo "gpu (after): $(gpu_line)"
    echo "R5 window cell $id  finished $t_end"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "cell $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-32s %-5s %-6s %-6s %-9s %s' "$id" "$CONFIG_SIDE" "$CONFIG_GROUP" "$rc" "$status" "$json")")
  CELL_ROWS+=("$(printf '%-32s %-5s %-6s %s  %s  %s  %s  %s' \
      "$id" "$CONFIG_SIDE" "$CONFIG_GROUP" "$t_start" "$t_end" "$rc" "$status" "$json")")
  if (( rc == 0 || rc == 3 )); then
    OK+=("$id")
  else
    FAILED+=("$id")
    DEVIATIONS+=("cell $id exited $rc ($status) -- kept the log ($log), continued the window")
  fi
  return 0
}

# --- 9. main loop -----------------------------------------------------------
SUMMARY=() ; OK=() ; SKIPPED=() ; FAILED=() ; DEVIATIONS=() ; CELL_ROWS=()
TOTAL=${#SELECTED[@]}
INDEX=0

# the meta file lives in the group's default result dir when --result-dir is absent;
# with --group all (two dirs) it goes to the window dir and names both.
if [[ -n "$RESULT_DIR" ]]; then
  META_DIR="$RESULT_DIR"
elif [[ "$GROUP" == "smoke" ]]; then
  META_DIR="test/benchmark/r5/results/smoke"
else
  META_DIR="test/benchmark/r5/results/q4_window"
fi
META_FILE="$META_DIR/window_meta.txt"

echo "r5/window_probe.sh: py=$PY"
echo "r5/window_probe.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r5/window_probe.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN cells=$TOTAL"
echo "r5/window_probe.sh: caps: default=${CAP}s legacy=${CAP_LEGACY}s  lock=$LOCK (one flock per cell, never nested)"
echo "r5/window_probe.sh: gpu snapshot: $(gpu_line)"
echo "r5/window_probe.sh: gpu tenants: $(gpu_tenants)"
if (( DRY_RUN )); then
  echo "r5/window_probe.sh: DRY RUN -- nothing is executed, no result dirs are created"
fi
echo

WINDOW_START="$(date -Is)"
if (( ! DRY_RUN )); then
  mkdir -p "$META_DIR"
  {
    echo "# $META_FILE -- R5 Q4 window record (generated by test/benchmark/r5/window_probe.sh)"
    echo "# committed evidence; immutable once the window is recorded -- a re-run writes rerun_<n>/window_meta.txt"
    echo "# invocation: $WINDOW_START   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')   host=$(hostname)"
    echo "# args: group=$GROUP only=${ONLY:-<none>} cells=$TOTAL dir=$META_DIR caps=default/${CAP}s,legacy/${CAP_LEGACY}s"
    echo "# lock: $LOCK   (exactly ONE flock -w $FLOCK_WAIT per cell, timeout -s INT -k 30 INSIDE the lock)"
    echo "# gpu: CUDA_VISIBLE_DEVICES=$GPU (GPU 7 refused -- hardware ECC fault)"
    echo "# ident (evogp): repo=$REPO_ROOT commit=$(git rev-parse HEAD 2>/dev/null || echo '<n/a>')"
    echo "#                file=$REPO_ROOT/src/evogp/__init__.py"
    echo "# ident (etl primary): $(etl_identity_line "$NEW_PYTHONPATH")"
    echo "# ident (etl shadow, bridge cell only): dir=$SHADOW_DIR present=$([[ -d "$SHADOW_DIR" ]] && echo yes || echo no)"
    echo "# ident (legacy): $LEG_PYTHONPATH   so=$(ls "$LEGACY_ROOT"/src/evogp/evogp_cuda*.so 2>/dev/null || echo '<absent>')"
    echo "# env (new side cells): ETL_DISABLE_LOCATIONS=1 PYTHONPATH=$NEW_PYTHONPATH ${NEW_ENVS[*]}"
    echo "# env (legacy side cells): ETL_DISABLE_LOCATIONS=1 PYTHONPATH=$LEG_PYTHONPATH"
    echo "# fixtures: $FIXTURE64"
    echo "#           $FIXTURE200"
    echo
    echo "## window"
    echo "window_start = $WINDOW_START"
    echo "gpu sample (start)   = $(gpu_line)"
    echo "gpu tenants (start)  = $(gpu_tenants)"
    echo
    echo "## cells (run order; timestamps are the cell's own start/end, not the lock wait)"
    echo "ID                               SIDE  GROUP  START                            END                              EXIT  STATUS  JSON"
  } > "$META_FILE"
fi

for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
  if (( ! DRY_RUN )); then
    {
      echo "gpu sample (between) = $(gpu_line)"
      echo "gpu tenants (between) = $(gpu_tenants)"
    } >> "$META_FILE"
  fi
done

if (( ! DRY_RUN )); then
  WINDOW_END="$(date -Is)"
  {
    echo
    echo "## cells table"
    for row in "${CELL_ROWS[@]}"; do printf '%s\n' "$row"; done
    echo
    echo "## window end"
    echo "window_end = $WINDOW_END"
    echo "gpu sample (end)     = $(gpu_line)"
    echo "gpu tenants (end)    = $(gpu_tenants)"
    echo "exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
    echo "ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
    echo
    echo "## deviations and protocol notes"
    if [[ -n "${DEVIATION_NOTE:-}" ]]; then printf '%s\n' "$DEVIATION_NOTE"; fi
    if (( ${#DEVIATIONS[@]} > 0 )); then
      for d in "${DEVIATIONS[@]}"; do echo "- $d"; done
    else
      echo "- none: every selected cell ran under the protocol as written"
    fi
  } >> "$META_FILE"
fi

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r5 window_probe.sh summary (group=$GROUP, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-32s %-5s %-6s %-6s %-9s %s\n' ID SIDE GROUP EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} cell(s) planned (dry run; $SUMMARY_FILE NOT written)"
  exit 0
fi

{
  echo "# $SUMMARY_FILE -- id -> exit-code table of the last r5 window_probe.sh invocation"
  echo "# generated output: disposable, never committed."
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} cells=$TOTAL dir=$META_DIR"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-32s %-5s %-6s %-6s %-9s %s\n' ID SIDE GROUP EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "window record: $META_FILE"
echo "id table:      $SUMMARY_FILE (disposable)"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
