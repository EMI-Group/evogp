#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r4/run_pure.sh -- R4 round runner: the PURE forward measurement
#   (measurement type 2) on the ETL "true jit" engine, numpy + IREE-cuda
#
# R4 measures what the existing eval tasks cannot separate: the internal graph
# forward pass, i.e. one "invocation" = one full-population forward on a fixed x
# with per_row=False, with the readback / host row assembly / compile-key
# derivation excluded.  The existing eval tasks (eval64 / evalfresh) measure
# measurement type 1, END-TO-END problem.evaluate(forest); this round's
# --e2e-companion K option runs that end-to-end path in the SAME process as the
# pure numbers, so "E2E - pure" prices the per-call host glue.
#
# The measurement itself is implemented in test/benchmark/bench_gp.py
# (--task eval-pure) + test/benchmark/pure_forward.py; this script only
# enumerates the 8 configs, guards the environment and serializes the processes.
# =============================================================================
#
# One `bench_gp.py` PROCESS per config, serialized on the shared benchmark lock.
# Run it from the REPOSITORY ROOT (the checkout that owns test/benchmark/):
#
#     test/benchmark/r4/run_pure.sh                  # all 8 configs (numpy then iree)
#     test/benchmark/r4/run_pure.sh --group numpy    # 4 configs: numpy backend
#     test/benchmark/r4/run_pure.sh --group iree     # 4 configs: IREE-cuda backend
#     test/benchmark/r4/run_pure.sh --only pure64_dense_numpy
#     test/benchmark/r4/run_pure.sh --only pure64_dense_numpy,pure64_dense_iree --timeout 3600
#     test/benchmark/r4/run_pure.sh --list           # print the config table
#     test/benchmark/r4/run_pure.sh --dry-run        # print every command, execute nothing
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2).
#
# Rules baked in here (do not "improve" them per-run):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors.
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per invocation, never
#     nested (other agents share that lock); the `timeout` sits INSIDE the lock, so
#     lock wait time is never charged to a config.
#   * one process per config => the engine's `etl.CompileCache` is per-process, so a
#     config's compile block is never polluted by an earlier one (and the pure numbers
#     are steady-state by construction: construct once, --warmup 10, then 5 blocks).
#   * `EVOGP_EVAL_MODE=<mode>` is set per config so the pure route (--mode) and the
#     --e2e-companion route agree: BOTH knobs are set, never just one.
#   * the runner always passes `--tag` = uppercase id and `--json-out <results>/<id>.json`;
#     `--seed 0` is the harness default and is left implicit; `--fixture` paths are
#     relative to the repo root.
#   * a failed config never aborts the batch: log + FAILED + keep going.
#
# Environment overrides (all optional):
#   PY               interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault (never use it)
#   TIMEOUT_S        per-config wall cap, default 1800 (or pass --timeout SECONDS)
#   LOCK / FLOCK_WAIT    shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   NEW_PYTHONPATH   PYTHONPATH of the new side (default `src`; prefix another checkout
#                    (e.g. a patched ETL) to shadow it, exactly as in R3 Arm B)
#   NEW_RESULT_DIR   results dir (default test/benchmark/r4/results)
#   PURE_EXTRA_ARGS  extra flags appended verbatim to every config's bench_gp.py command
#                    (default empty; e.g. `PURE_EXTRA_ARGS=--output-checksum` for the
#                    after-round records -- post-timing, so timed parameters are unchanged)
#   ETL_REPO         the ETL checkout whose HEAD is recorded in every log (default
#                    /mnt/local-ssd/bchuang/etl; read-only, never written)
#
# Exit codes:
#   0  every selected config was OK or guard-skipped (bench_gp.py exit 3 = skipped, OK)
#   1  at least one config exited non-zero (timeout 124/137, crash 1, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown id, missing interpreter)
# Per-config results land in
#   test/benchmark/r4/results/<id>.{json,log}
# and the id -> exit-code table is written to test/benchmark/r4/last_run.txt
# (generated output, .gitignore'd, never committed).
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_pure.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
NEW_RESULT_DIR="${NEW_RESULT_DIR:-test/benchmark/r4/results}"
SUMMARY_FILE="$R4_DIR/last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"
ETL_REPO="${ETL_REPO:-/mnt/local-ssd/bchuang/etl}"

FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r4/run_pure.sh [--group numpy|iree|all] [--only id1,id2]
                                    [--timeout SECONDS] [--list] [--dry-run] [--help]

  --group numpy   the four numpy-backend configs (dense + specialized x 64/200 trees)
  --group iree    the four IREE-cuda configs (dense + specialized x 64/200 trees)
  --group all     numpy then iree (default)
  --only IDS      run exactly these config ids (comma-separated); overrides --group
  --timeout S     per-config wall cap, fed to `timeout` INSIDE the lock (default: $TIMEOUT_S)
  --list          print the config table (id, group, mode, backend, tag, json, args) and exit
  --dry-run       print every command without executing anything
  --help          this text

Run from the repository root.  See test/benchmark/r4/README.md for the full protocol.
EOF
}

GROUP=all
ONLY=""
DRY_RUN=0
LIST=0
while (( $# > 0 )); do
  case "$1" in
    --group) GROUP="${2:?--group needs a value}"; shift 2 ;;
    --group=*) GROUP="${1#--group=}"; shift ;;
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --timeout) TIMEOUT_S="${2:?--timeout needs a value}"; shift 2 ;;
    --timeout=*) TIMEOUT_S="${1#--timeout=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "run_pure.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  numpy|iree|all) ;;
  *) echo "run_pure.sh: --group must be one of numpy|iree|all (got '$GROUP')" >&2; exit 2 ;;
esac

# --- 4. config table --------------------------------------------------------
# Order inside each group is the run order (do not sort).
# The group order is numpy THEN iree (so an IREE-window problem can never invalidate the
# numpy rows); within a group: fixture64 before fixture200, dense before specialized.
NUMPY_IDS=(
  pure64_dense_numpy pure64_specialized_numpy
  pure200_dense_numpy pure200_specialized_numpy
)
IREE_IDS=(
  pure64_dense_iree pure64_specialized_iree
  pure200_dense_iree pure200_specialized_iree
)

# config_spec ID -> CONFIG_GROUP, CONFIG_MODE (dense|specialized), CONFIG_BACKEND
#                  (numpy|iree), CONFIG_ARGS, CONFIG_ENVS, CONFIG_FIXTURE
#                  (existence-guarded), CONFIG_NOTE
config_spec() {
  CONFIG_GROUP="" ; CONFIG_MODE="" ; CONFIG_BACKEND="" ; CONFIG_ARGS=()
  CONFIG_ENVS=() ; CONFIG_FIXTURE="" ; CONFIG_NOTE=""
  case "$1" in
    # ---- group numpy: the default backend, both routes, both fixtures -------
    pure64_dense_numpy)        CONFIG_GROUP=numpy; CONFIG_MODE=dense;       CONFIG_BACKEND=numpy
                               CONFIG_FIXTURE="$FIXTURE64"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode dense
                                 --backend numpy --reps 5 --warmup 10 --invocations 100
                                 --e2e-companion 11 --mem-guard-mb 8192)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                               CONFIG_NOTE="numpy dense route: ONE graph for 64 trees, 100 invocations/block" ;;
    pure64_specialized_numpy)  CONFIG_GROUP=numpy; CONFIG_MODE=specialized; CONFIG_BACKEND=numpy
                               CONFIG_FIXTURE="$FIXTURE64"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode specialized
                                 --backend numpy --reps 5 --warmup 10 --invocations 50
                                 --e2e-companion 11 --mem-guard-mb 8192)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                               CONFIG_NOTE="numpy specialized route: one etl.run per non-constant tree, 50 invocations/block" ;;
    pure200_dense_numpy)       CONFIG_GROUP=numpy; CONFIG_MODE=dense;       CONFIG_BACKEND=numpy
                               CONFIG_FIXTURE="$FIXTURE200"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE200" --mode dense
                                 --backend numpy --reps 5 --warmup 10 --invocations 100
                                 --e2e-companion 11 --mem-guard-mb 8192)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                               CONFIG_NOTE="numpy dense route, 200 trees (needs the r2 fixture)" ;;
    pure200_specialized_numpy) CONFIG_GROUP=numpy; CONFIG_MODE=specialized; CONFIG_BACKEND=numpy
                               CONFIG_FIXTURE="$FIXTURE200"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE200" --mode specialized
                                 --backend numpy --reps 5 --warmup 10 --invocations 50
                                 --e2e-companion 11 --mem-guard-mb 8192)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                               CONFIG_NOTE="numpy specialized route, 200 trees (needs the r2 fixture)" ;;

    # ---- group iree: the opt-in IREE-cuda backend, cold-build budget 400 ----
    pure64_dense_iree)         CONFIG_GROUP=iree; CONFIG_MODE=dense;        CONFIG_BACKEND=iree
                               CONFIG_FIXTURE="$FIXTURE64"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode dense
                                 --backend iree --reps 5 --warmup 10 --invocations 100
                                 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                               CONFIG_NOTE="iree dense route: ONE compiled graph for 64 trees (ulimit -n 65536 needed)" ;;
    pure64_specialized_iree)   CONFIG_GROUP=iree; CONFIG_MODE=specialized;  CONFIG_BACKEND=iree
                               CONFIG_FIXTURE="$FIXTURE64"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE64" --mode specialized
                                 --backend iree --reps 5 --warmup 10 --invocations 50
                                 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                               CONFIG_NOTE="iree specialized route: one compiled program per non-constant tree (cold-build cap 400)" ;;
    pure200_dense_iree)        CONFIG_GROUP=iree; CONFIG_MODE=dense;        CONFIG_BACKEND=iree
                               CONFIG_FIXTURE="$FIXTURE200"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE200" --mode dense
                                 --backend iree --reps 5 --warmup 10 --invocations 100
                                 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=dense)
                               CONFIG_NOTE="iree dense route, 200 trees (needs the r2 fixture)" ;;
    pure200_specialized_iree)  CONFIG_GROUP=iree; CONFIG_MODE=specialized;  CONFIG_BACKEND=iree
                               CONFIG_FIXTURE="$FIXTURE200"
                               CONFIG_ARGS=(--task eval-pure --fixture "$FIXTURE200" --mode specialized
                                 --backend iree --reps 5 --warmup 10 --invocations 50
                                 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400)
                               CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                               CONFIG_NOTE="iree specialized route, 200 trees: up to ~200 cold builds (cap 400)" ;;
    *) return 1 ;;
  esac
  return 0
}

# NOTE on the per-config environments: EVOGP_EVAL_MODE=<mode> pins the route for BOTH the
# pure runner (--mode) and the --e2e-companion problem.evaluate() calls, so the pure number
# and the E2E number below it are measured on the same route.  It is exported per config via
# `env VAR=...` below (NOT globally), and the harness records every ETL_*/EVOGP_* value in the
# JSON's `env` block.
# NOTE on the invocation counts: 100 per block on the dense route (one graph call per chunk,
# so a block is cheap) and 50 on the specialized route (one etl.run per non-constant tree, so
# a block is ~64/200 calls per invocation).

all_ids() { printf '%s\n' "${NUMPY_IDS[@]}" "${IREE_IDS[@]}"; }

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "run_pure.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_pure.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi

# advisory capability probe: --task eval-pure (+ --mode / --invocations / --e2e-companion) is
# a frozen CLI contract implemented in bench_gp.py.  Warn loudly when the harness on disk does
# not have it yet, so the resulting argparse failures are not misread as engine problems.
# (Advisory only -- never blocks a run.)
if ! grep -q -e '--e2e-companion' "$BENCH" 2>/dev/null; then
  echo "run_pure.sh: WARNING: $BENCH does not advertise --e2e-companion (--task eval-pure missing?);" >&2
  echo "             every config would fail with an argparse error -- materialize the R4 harness" >&2
  echo "             change first (the CLI contract is frozen)." >&2
fi

# selection
SELECTED=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_raw <<< "$ONLY"
  _only=()
  for _id in "${_only_raw[@]}"; do
    _id="$(printf '%s' "$_id" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    [[ -z "$_id" ]] && continue
    if ! config_spec "$_id"; then
      echo "run_pure.sh: unknown config id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_pure.sh: --only is empty" >&2; exit 2; }
  while IFS= read -r _id; do            # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done < <(all_ids)
else
  case "$GROUP" in
    numpy) SELECTED=("${NUMPY_IDS[@]}") ;;
    iree)  SELECTED=("${IREE_IDS[@]}") ;;
    all)   SELECTED=("${NUMPY_IDS[@]}" "${IREE_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_pure.sh: nothing selected" >&2; exit 2; }

# --- 6. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
}

# etl_identity_line -> one line of evidence about WHICH etl checkout the config will import.
# Cheap and failure-tolerant: a missing checkout / a broken import must never fail a config.
etl_identity_line() {
  local rev path
  rev="$(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"
  path="$(env "PYTHONPATH=$NEW_PYTHONPATH" "$PY" -c 'import etl; print(etl.__file__)' 2>/dev/null \
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

# guard_reason ID -> prints a skip reason (empty when the config is runnable)
guard_reason() {
  local id="$1"
  config_spec "$id" || return 0
  if [[ -n "$CONFIG_FIXTURE" && ! -f "$CONFIG_FIXTURE" ]]; then
    echo "fixture missing ($CONFIG_FIXTURE; relative to the repo root)"
    return 0
  fi
  return 0
}

# --- 7. --list --------------------------------------------------------------
if (( LIST )); then
  printf '%-26s %-6s %-11s %-8s %-26s %s\n' ID GROUP MODE BACKEND TAG JSON
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    printf '%-26s %-6s %-11s %-8s %-26s %s\n' "$id" "$CONFIG_GROUP" "$CONFIG_MODE" \
      "$CONFIG_BACKEND" "$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')" \
      "$NEW_RESULT_DIR/$id.json"
  done
  echo
  echo "-- exact args per config (+ per-config env overrides, + pre-flight guard state) --"
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    args="${CONFIG_ARGS[*]}${PURE_EXTRA_ARGS:+ $PURE_EXTRA_ARGS}"
    envs=""
    if (( ${#CONFIG_ENVS[@]} > 0 )); then envs="  env: ${CONFIG_ENVS[*]}"; fi
    printf '%-26s %s%s\n' "$id" "$args" "$envs"
    reason="$(guard_reason "$id")"
    if [[ -n "$reason" ]]; then
      printf '%-26s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one config ------------------------------------------------------
run_one() {
  local id="$1"
  config_spec "$id" || { echo "run_pure.sh: unknown config id '$id'" >&2; return 1; }

  local tag json log
  tag="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  json="$NEW_RESULT_DIR/$id.json"
  log="$NEW_RESULT_DIR/$id.log"

  # pre-flight guards: a config whose input is absent is "skipped by guard" (exit code 3 ==
  # OK), never a fake failure and never a silent omission.
  local reason; reason="$(guard_reason "$id")"

  local -a child=(env "PYTHONPATH=$NEW_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$GPU")
  local e
  for e in "${CONFIG_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$BENCH" "${CONFIG_ARGS[@]}")
  if [[ -n "${PURE_EXTRA_ARGS:-}" ]]; then
    local -a _extra; read -r -a _extra <<< "$PURE_EXTRA_ARGS"
    child+=("${_extra[@]}")
  fi
  child+=("--tag" "$tag" "--json-out" "$json")
  local -a timed=(timeout -s INT -k 30 "$TIMEOUT_S" "${child[@]}")
  local inner; inner="$(printf '%q ' "${timed[@]}")"; inner="${inner% }"

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $id  (group=$CONFIG_GROUP mode=$CONFIG_MODE backend=$CONFIG_BACKEND)"
    if [[ -n "$reason" ]]; then
      echo "        GUARD-SKIP: $reason   -> recorded as exit 3 (OK)"
    fi
    echo "        json: $json"
    echo "        log : $log"
    if [[ -n "$CONFIG_NOTE" ]]; then echo "        note: $CONFIG_NOTE"; fi
    # the inner command contains no single quotes by construction, so the printed
    # single-quoted form is exactly what `bash -c` receives at run time.
    echo "        flock -w $FLOCK_WAIT $LOCK bash -c '$inner'"
    echo "        (timeout -s INT -k 30 ${TIMEOUT_S}s runs INSIDE the lock; stdout+stderr -> tee -a $log)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "config $id: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-26s %-6s %-11s %-8s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_GROUP" "$CONFIG_MODE" "$CONFIG_BACKEND" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$id")
    return 0
  fi

  mkdir -p "$NEW_RESULT_DIR"
  {
    echo "==============================================================================="
    echo "R4 config $id  (group=$CONFIG_GROUP mode=$CONFIG_MODE backend=$CONFIG_BACKEND)  started $(date -Is)"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "timeout: ${TIMEOUT_S}s (SIGINT, SIGKILL after 30s) INSIDE flock -w $FLOCK_WAIT $LOCK"
    echo "gpu (before): $(gpu_line)"
    echo "$(etl_identity_line)"
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
    echo "R4 config $id  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "config $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-26s %-6s %-11s %-8s %-6s %-9s %s' "$id" "$CONFIG_GROUP" "$CONFIG_MODE" "$CONFIG_BACKEND" "$rc" "$status" "$json")")
  if (( rc == 0 || rc == 3 )); then
    OK+=("$id")
  else
    FAILED+=("$id")
  fi
  return 0
}

# --- 9. main loop -----------------------------------------------------------
SUMMARY=() ; OK=() ; SKIPPED=() ; FAILED=()
TOTAL=${#SELECTED[@]}
INDEX=0

echo "r4/run_pure.sh: py=$PY"
echo "r4/run_pure.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r4/run_pure.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN configs=$TOTAL timeout=${TIMEOUT_S}s lock=$LOCK"
echo "r4/run_pure.sh: results=$NEW_RESULT_DIR  new_pythonpath=$NEW_PYTHONPATH"
echo "r4/run_pure.sh: gpu snapshot: $(gpu_line)"
echo "r4/run_pure.sh: $(etl_identity_line)"
if (( DRY_RUN )); then
  echo "r4/run_pure.sh: DRY RUN -- nothing is executed, no results dirs are created"
fi
echo

for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r4 run_pure.sh summary (group=$GROUP, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-26s %-6s %-11s %-8s %-6s %-9s %s\n' ID GROUP MODE BACKEND EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} config(s) planned (dry run; test/benchmark/r4/last_run.txt NOT written)"
  exit 0
fi

{
  echo "# test/benchmark/r4/last_run.txt -- id -> exit-code table of the last r4 run_pure.sh invocation"
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} configs=$TOTAL timeout=${TIMEOUT_S}s"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-26s %-6s %-11s %-8s %-6s %-9s %s\n' ID GROUP MODE BACKEND EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
