#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r2/round3/run_all.sh -- R3 GPU-round matrix runner
#   legacy CUDA engine  vs  ETL "true jit" engine (numpy + IREE-cuda)
#
# Round 3 re-measures the R2 headline configs with the R2 methodology after two
# engine optimization waves: (1) host-side cuts in src/evogp/tree/etl_engine.py
# (decode / build / sr_fitness host cost) and (2) the dense one-graph evaluator
# src/evogp/tree/dense_engine.py, now the DEFAULT route for the iree backend under
# the default `auto` policy (pop >= 16, and on per_row=False calls <= 2048 data
# rows).  The `iree-spec` group pins the R2-era route (EVOGP_EVAL_MODE=specialized)
# on the same three workloads so the dense contribution stays attributable.
# =============================================================================
#
# One `bench_gp.py` PROCESS per config, serialized on the shared benchmark lock.
# Run it from the REPOSITORY ROOT (the checkout that owns test/benchmark/):
#
#     test/benchmark/r2/round3/run_all.sh                    # all 12 configs
#     test/benchmark/r2/round3/run_all.sh --group light      # 6 configs: legacy + numpy mirrors
#     test/benchmark/r2/round3/run_all.sh --group iree       # 3 configs: IREE, default auto route
#     test/benchmark/r2/round3/run_all.sh --group iree-spec  # 3 configs: IREE, forced specialized
#     test/benchmark/r2/round3/run_all.sh --only np_c5r2     # exactly these ids, comma-separated
#     test/benchmark/r2/round3/run_all.sh --only np_c5r2,leg_c5r2 --timeout 7200
#     test/benchmark/r2/round3/run_all.sh --dry-run          # print every command, execute nothing
#     test/benchmark/r2/round3/run_all.sh --list             # print the config table
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2).
#
# Nothing about the R2 methodology changes here: same harness (bench_gp.py), same
# flags, same fixtures, same wiring, same implicit `--seed 0`, same lock/`ulimit`
# order, one process per config -- and the ids intentionally match their R2
# counterparts, so every R2/R3 pair can be diffed directly (the R2 records stay
# untouched: they were measured on the pre-optimization engine).
#
# Rules baked in here (do not "improve" them per-run):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors.
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per invocation, never
#     nested (other agents share that lock); the `timeout` sits INSIDE the lock, so
#     lock wait time is never charged to a config.
#   * one process per config => the engine's `etl.CompileCache` (and the dense
#     evaluator's own compiled-graph cache) are per-process, so a config's
#     `first_call_ms` / cold-build counts are never polluted by an earlier one.
#   * the runner always passes `--tag` = uppercase id and `--json-out <results>/<id>.json`;
#     `--seed 0` is the harness default and is left implicit; `--fixture` paths are
#     relative to the repo root.
#   * a failed config never aborts the batch: log + FAILED + keep going.
#
# Environment overrides (all optional):
#   PY               interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault (never use it)
#   TIMEOUT_S        per-config wall cap, default 3600 (or pass --timeout SECONDS)
#   LOCK / FLOCK_WAIT    shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   LEGACY_ROOT      default /tmp/evogp_legacy_bench (materialized b613503 checkout)
#   LEGACY_SITEUTILS default /tmp/evogp_t1a263/siteutils (offline sklearn for the old side)
#   NEW_PYTHONPATH / LEG_PYTHONPATH   PYTHONPATH per side (defaults: `src`, legacy pair above)
#
# Exit codes:
#   0  every selected config was OK or guard-skipped (bench_gp.py exit 3 = skipped, OK)
#   1  at least one config exited non-zero (timeout 124/137, crash 1, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown id, missing interpreter)
# Per-config results land in
#   test/benchmark/r2/round3/results/new/<id>.{json,log}      (new side: numpy + iree)
#   test/benchmark/r2/round3/results/legacy/<id>.{json,log}   (legacy CUDA side)
# and the id -> exit-code table is written to test/benchmark/r2/round3/last_run.txt
# (generated output, .gitignore'd, never committed).
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_all.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
NEW_RESULT_DIR=test/benchmark/r2/round3/results/new
LEG_RESULT_DIR=test/benchmark/r2/round3/results/legacy
SUMMARY_FILE="$R3_DIR/last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

LEGACY_ROOT="${LEGACY_ROOT:-/tmp/evogp_legacy_bench}"
LEGACY_SITEUTILS="${LEGACY_SITEUTILS:-/tmp/evogp_t1a263/siteutils}"
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"
LEG_PYTHONPATH="${LEG_PYTHONPATH:-$LEGACY_ROOT/src:$LEGACY_SITEUTILS}"

DEFAULT_FIXTURE=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r2/round3/run_all.sh [--group light|iree|iree-spec|all] [--only id1,id2]
                                          [--timeout SECONDS] [--list] [--dry-run] [--help]

  --group light      legacy-side configs + their numpy mirrors (fast, ~minutes)
  --group iree       new-side IREE-cuda configs on the default `auto` eval policy
  --group iree-spec  the same three IREE workloads with EVOGP_EVAL_MODE=specialized
  --group all        all three, light then iree then iree-spec (default)
  --only IDS         run exactly these config ids (comma-separated); overrides --group
  --timeout S        per-config wall cap, fed to `timeout` INSIDE the lock (default: $TIMEOUT_S)
  --list             print the config table (id, group, side, task, tag, json, args) and exit
  --dry-run          print every command without executing anything
  --help             this text

Run from the repository root.  See test/benchmark/r2/round3/README.md for the full rules.
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
    *) echo "run_all.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  light|iree|iree-spec|all) ;;
  *) echo "run_all.sh: --group must be one of light|iree|iree-spec|all (got '$GROUP')" >&2; exit 2 ;;
esac

# --- 4. config table --------------------------------------------------------
# Order inside each group is the run order (do not sort).
LIGHT_IDS=(
  # legacy CUDA side (results in test/benchmark/r2/round3/results/legacy/)
  leg_c5r2 leg_fixture200_n64 leg_sr_a
  # new side, default "numpy" backend, --stage-probe ON (results in round3/results/new/)
  np_c5r2 np_fixture200_n64 np_sr_a
)
IREE_IDS=(
  # new side, --backend iree --stage-probe; the default `auto` eval policy selects the
  # dense one-graph evaluator here, so these arms measure the post-wave-2 default route
  iree_fixture64_n64 iree_fixture200_n64 iree_sr_a
)
IREE_SPEC_IDS=(
  # new side, --backend iree, EVOGP_EVAL_MODE=specialized: the R2-era route pinned on the
  # same workloads, so a wave-1 (host-side) effect is separable from the wave-2 (dense) one
  iree_fixture64_n64_spec iree_fixture200_n64_spec iree_sr_a_spec
)

# config_spec ID -> CONFIG_SIDE (leg|new), CONFIG_GROUP, CONFIG_ARGS, CONFIG_ENVS,
#                  CONFIG_FIXTURE (existence-guarded; "" = workload reads no fixture), CONFIG_NOTE
config_spec() {
  CONFIG_SIDE="" ; CONFIG_GROUP="" ; CONFIG_ARGS=() ; CONFIG_ENVS=() ; CONFIG_FIXTURE="" ; CONFIG_NOTE=""
  case "$1" in
    # ---- light group: legacy CUDA side ------------------------------------
    leg_c5r2)            CONFIG_SIDE=leg; CONFIG_GROUP=light; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25)
                         CONFIG_NOTE="legacy C5 re-run: 64-tree fixture, 25 evaluates" ;;
    leg_fixture200_n64)  CONFIG_SIDE=leg; CONFIG_GROUP=light; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25)
                         CONFIG_NOTE="legacy: 200-tree fixture (needs r2 fixture)" ;;
    leg_sr_a)            CONFIG_SIDE=leg; CONFIG_GROUP=light
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2)
                         CONFIG_NOTE="legacy short W1 SR training (pop 200 x 10 gens x 2 reps)" ;;

    # ---- light group: new side, numpy backend + stage probe ---------------
    np_c5r2)             CONFIG_SIDE=new; CONFIG_GROUP=light; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_c5r2" ;;
    np_fixture200_n64)   CONFIG_SIDE=new; CONFIG_GROUP=light; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_fixture200_n64" ;;
    np_sr_a)             CONFIG_SIDE=new; CONFIG_GROUP=light
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_sr_a" ;;

    # ---- iree group: new side, --backend iree, default `auto` eval policy --
    # `auto` keeps the specialized engine on numpy and picks the dense evaluator on iree
    # (pop >= 16, and <= 2048 rows on per_row=False calls), so these arms are the
    # post-wave-2 default route.
    iree_fixture64_n64)  CONFIG_SIDE=new; CONFIG_GROUP=iree; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --backend iree --stage-probe)
                         CONFIG_NOTE="IREE auto route (dense is the default for iree under the auto policy)" ;;
    iree_fixture200_n64) CONFIG_SIDE=new; CONFIG_GROUP=iree; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --backend iree --stage-probe --max-cold-builds 400)
                         CONFIG_NOTE="IREE auto route (needs r2 fixture); R2 cold-build budget 400 kept" ;;
    iree_sr_a)           CONFIG_SIDE=new; CONFIG_GROUP=iree
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --backend iree --stage-probe --max-cold-builds 1500)
                         CONFIG_ENVS=(EVOGP_ETL_CACHE_SIZE=8192)
                         CONFIG_NOTE="IREE auto route: short W1 SR, cache 8192; R2 flags kept (cold-build budget 1500)" ;;

    # ---- iree-spec group: the same three arms with EVOGP_EVAL_MODE=specialized
    # (the R2-era route: no dense evaluator -> wave-1 attribution only)
    iree_fixture64_n64_spec)  CONFIG_SIDE=new; CONFIG_GROUP=iree-spec; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --backend iree --stage-probe)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                         CONFIG_NOTE="forced specialized (R2-era route) for wave attribution" ;;
    iree_fixture200_n64_spec) CONFIG_SIDE=new; CONFIG_GROUP=iree-spec; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --backend iree --stage-probe --max-cold-builds 400)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                         CONFIG_NOTE="forced specialized for wave attribution (needs r2 fixture); budget 400 kept" ;;
    iree_sr_a_spec)      CONFIG_SIDE=new; CONFIG_GROUP=iree-spec
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --backend iree --stage-probe --max-cold-builds 1500)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized EVOGP_ETL_CACHE_SIZE=8192)
                         CONFIG_NOTE="forced specialized for wave attribution; cache 8192, budget 1500 kept" ;;
    *) return 1 ;;
  esac
  return 0
}

# NOTE on per-config environments: EVOGP_ETL_CACHE_SIZE=8192 for the `*_sr_a*` arms (the engine's
# per-process per-structure build cache is small by default and these runs change structure on
# almost every generation) and EVOGP_EVAL_MODE=specialized for the `iree-spec` arms (pin the R2-era
# route).  Such overrides are exported in the run env via `env VAR=...` below -- NOT set globally --
# so the other configs keep the harness defaults, and every value is recorded in each config's JSON
# `env` block.

all_ids() { printf '%s\n' "${LIGHT_IDS[@]}" "${IREE_IDS[@]}" "${IREE_SPEC_IDS[@]}"; }

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "run_all.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_all.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi

# advisory capability probe: the R2 CLI (--backend / --data-rows / --stage-probe /
# --max-cold-builds), reused unchanged by R3, is a frozen contract implemented in bench_gp.py.
# Warn loudly when the harness on disk does not have it yet, so the resulting argparse
# failures are not misread as engine problems.  (Advisory only -- never blocks a run.)
if ! grep -q -e '--stage-probe' "$BENCH" 2>/dev/null; then
  echo "run_all.sh: WARNING: $BENCH does not advertise --stage-probe (R2 CLI not present?);" >&2
  echo "             every probed/IREE config would fail with an argparse error -- materialize" >&2
  echo "             the R2 harness change first (the CLI contract is frozen)." >&2
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
      echo "run_all.sh: unknown config id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_all.sh: --only is empty" >&2; exit 2; }
  while IFS= read -r _id; do            # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done < <(all_ids)
else
  case "$GROUP" in
    light)     SELECTED=("${LIGHT_IDS[@]}") ;;
    iree)      SELECTED=("${IREE_IDS[@]}") ;;
    iree-spec) SELECTED=("${IREE_SPEC_IDS[@]}") ;;
    all)       SELECTED=("${LIGHT_IDS[@]}" "${IREE_IDS[@]}" "${IREE_SPEC_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_all.sh: nothing selected" >&2; exit 2; }

# --- 6. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
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
  if [[ -n "$CONFIG_FIXTURE" && ! -f "$CONFIG_FIXTURE" ]]; then
    echo "fixture missing ($CONFIG_FIXTURE; relative to the repo root)"
    return 0
  fi
  return 0
}

# --- 7. --list --------------------------------------------------------------
if (( LIST )); then
  printf '%-28s %-5s %-5s %-9s %-27s %s\n' ID GROUP SIDE TASK TAG JSON
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    task="?"
    for (( i=0; i<${#CONFIG_ARGS[@]}; i++ )); do
      [[ "${CONFIG_ARGS[$i]}" == "--task" ]] && task="${CONFIG_ARGS[$((i+1))]}"
    done
    if [[ "$CONFIG_SIDE" == leg ]]; then json="$LEG_RESULT_DIR/$id.json"; else json="$NEW_RESULT_DIR/$id.json"; fi
    printf '%-28s %-5s %-5s %-9s %-27s %s\n' "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" "$task" \
      "$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')" "$json"
  done
  echo
  echo "-- exact args per config (+ per-config env overrides, + pre-flight guard state) --"
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    args="${CONFIG_ARGS[*]}"
    envs=""
    if (( ${#CONFIG_ENVS[@]} > 0 )); then envs="  env: ${CONFIG_ENVS[*]}"; fi
    printf '%-28s %s%s\n' "$id" "$args" "$envs"
    reason="$(guard_reason "$id")"
    if [[ -n "$reason" ]]; then
      printf '%-28s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one config ------------------------------------------------------
run_one() {
  local id="$1"
  config_spec "$id" || { echo "run_all.sh: unknown config id '$id'" >&2; return 1; }

  local tag dir json log pyp
  tag="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  if [[ "$CONFIG_SIDE" == "leg" ]]; then
    dir="$LEG_RESULT_DIR" ; pyp="$LEG_PYTHONPATH"
  else
    dir="$NEW_RESULT_DIR" ; pyp="$NEW_PYTHONPATH"
  fi
  json="$dir/$id.json"
  log="$dir/$id.log"

  # pre-flight guards: a side whose environment/input is absent is "skipped by guard"
  # (exit code 3 == OK), never a fake failure and never a silent omission.
  local reason; reason="$(guard_reason "$id")"

  local -a child=(env "PYTHONPATH=$pyp" "CUDA_VISIBLE_DEVICES=$GPU")
  local e
  for e in "${CONFIG_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$BENCH" "${CONFIG_ARGS[@]}" "--tag" "$tag" "--json-out" "$json")
  local -a timed=(timeout -s INT -k 30 "$TIMEOUT_S" "${child[@]}")
  local inner; inner="$(printf '%q ' "${timed[@]}")"; inner="${inner% }"

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $id  (side=$CONFIG_SIDE group=$CONFIG_GROUP)"
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
    SUMMARY+=("$(printf '%-28s %-5s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_SIDE" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$id")
    return 0
  fi

  mkdir -p "$dir"
  {
    echo "==============================================================================="
    echo "R3 config $id  (side=$CONFIG_SIDE group=$CONFIG_GROUP)  started $(date -Is)"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "timeout: ${TIMEOUT_S}s (SIGINT, SIGKILL after 30s) INSIDE flock -w $FLOCK_WAIT $LOCK"
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
    echo "R3 config $id  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "config $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-28s %-5s %-6s %-9s %s' "$id" "$CONFIG_SIDE" "$rc" "$status" "$json")")
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

echo "r3/run_all.sh: py=$PY"
echo "r3/run_all.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r3/run_all.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN configs=$TOTAL timeout=${TIMEOUT_S}s lock=$LOCK"
echo "r3/run_all.sh: gpu snapshot: $(gpu_line)"
if (( DRY_RUN )); then
  echo "r3/run_all.sh: DRY RUN -- nothing is executed, no results dirs are created"
fi
echo

for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r3 run_all.sh summary (group=$GROUP, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-28s %-5s %-6s %-9s %s\n' ID SIDE EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} config(s) planned (dry run; test/benchmark/r2/round3/last_run.txt NOT written)"
  exit 0
fi

{
  echo "# test/benchmark/r2/round3/last_run.txt -- id -> exit-code table of the last r3 run_all.sh invocation"
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} configs=$TOTAL timeout=${TIMEOUT_S}s"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH leg_pythonpath=$LEG_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-28s %-5s %-6s %-9s %s\n' ID SIDE EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
