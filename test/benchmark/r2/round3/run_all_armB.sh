#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r2/round3/run_all_armB.sh -- "Arm B" matrix runner
#   the nine R3 new-side configs re-measured against a PATCHED ETL checkout
#
# Arm A (test/benchmark/r2/round3/run_all.sh) measured the R3 configs against
# the stock editable `etl` install (/mnt/local-ssd/bchuang/etl, `etl 0.1.0`).
# Arm B repeats the same nine new-side workloads with one extra PYTHONPATH entry
# shadowing that install:
#
#     NEW_PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141
#
# so every Arm B record is produced by the patched checkout, not by the stock
# one.  The shadow is proven per config: each log's header carries an
#
#     etl: <module file imported under this config's PYTHONPATH>
#
# line, which must read /tmp/evogp_etl_patched_51a7141/etl/__init__.py.  The
# patched checkout is READ-ONLY -- nothing here writes to it (or to the sibling
# ETL source checkout), and nothing here edits the frozen Arm A tooling.
#
# Only two things change relative to Arm A: (1) the PYTHONPATH shadow above, and
# (2) `EVOGP_EVAL_MODE` is passed EXPLICITLY on all six IREE configs -- `auto`
# on the three plain `iree` arms (Arm A relied on the implicit `auto` default
# there) and `specialized` on the three `_spec` arms -- so an Arm A/Arm B diff
# isolates the ETL checkout, never an eval-mode default.
#
# Arm B has NO legacy configs: the legacy CUDA side is frozen and absent by
# design (its R2/R3 records stay the reference).  The LEG_PYTHONPATH machinery
# below is kept only so the env/summary line format stays identical to Arm A.
#
# All Arm B artifacts live under test/benchmark/r2/round3/results/armB/ -- no
# Arm A result directory, no R2 record and no round-3 root file is touched.
# =============================================================================
#
# One `bench_gp.py` PROCESS per config, serialized on the shared benchmark lock.
# Run it from the REPOSITORY ROOT (the checkout that owns test/benchmark/):
#
#     test/benchmark/r2/round3/run_all_armB.sh                    # all 9 configs
#     test/benchmark/r2/round3/run_all_armB.sh --group iree       # 3 configs: IREE, explicit auto
#     test/benchmark/r2/round3/run_all_armB.sh --group iree-spec  # 3 configs: IREE, forced specialized
#     test/benchmark/r2/round3/run_all_armB.sh --group numpy      # 3 configs: numpy reference backend
#     test/benchmark/r2/round3/run_all_armB.sh --only armB_np_sr_a  # exactly these ids
#     test/benchmark/r2/round3/run_all_armB.sh --only armB_np_c5r2,armB_iree_sr_a --timeout 7200
#     test/benchmark/r2/round3/run_all_armB.sh --dry-run          # print every command, execute nothing
#     test/benchmark/r2/round3/run_all_armB.sh --list             # print the config table
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2); ids are matched
# case-insensitively, so `--only armB_np_sr_a` and `--only armb_np_sr_a` are the same id.
#
# Nothing about the R2/R3 methodology changes here: same harness (bench_gp.py),
# same flags, same fixtures, same wiring, same implicit `--seed 0`, same
# lock/`ulimit` order, one process per config -- and every id is its Arm A
# counterpart's id with an `armB_` prefix, so an Arm A/Arm B pair diffs directly.
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
#   NEW_PYTHONPATH   PYTHONPATH for the new side (default: `src` + the patched ETL checkout)
#   LEGACY_ROOT / LEGACY_SITEUTILS / LEG_PYTHONPATH   unused in Arm B (no legacy config);
#                    kept so the env/summary line format matches run_all.sh
#
# Exit codes:
#   0  every selected config was OK or guard-skipped (bench_gp.py exit 3 = skipped, OK)
#   1  at least one config exited non-zero (timeout 124/137, crash 1, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown id, missing interpreter)
# Per-config results land in
#   test/benchmark/r2/round3/results/armB/new/<id>.{json,log}   (all Arm B configs are new-side)
# (the unused results/armB/legacy/ path is kept for layout parity with Arm A)
# and the id -> exit-code table is written to test/benchmark/r2/round3/results/armB/last_run.txt
# (generated output, .gitignore'd, never committed).
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_all_armB.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
NEW_RESULT_DIR=test/benchmark/r2/round3/results/armB/new
LEG_RESULT_DIR=test/benchmark/r2/round3/results/armB/legacy
SUMMARY_FILE="$R3_DIR/results/armB/last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

# ARM B: the patched ETL checkout is shadowed onto PYTHONPATH ahead of the stock
# editable install; override NEW_PYTHONPATH to point at another checkout.
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src:/tmp/evogp_etl_patched_51a7141}"
# unused in Arm B (no legacy config) -- kept so the env/summary line format matches run_all.sh
LEGACY_ROOT="${LEGACY_ROOT:-/tmp/evogp_legacy_bench}"
LEGACY_SITEUTILS="${LEGACY_SITEUTILS:-/tmp/evogp_t1a263/siteutils}"
LEG_PYTHONPATH="${LEG_PYTHONPATH:-$LEGACY_ROOT/src:$LEGACY_SITEUTILS}"

DEFAULT_FIXTURE=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r2/round3/run_all_armB.sh [--group iree-spec|iree|numpy|all]
                                               [--only id1,id2] [--timeout SECONDS]
                                               [--list] [--dry-run] [--help]

  Arm B: the nine R3 new-side configs re-measured with a patched ETL checkout
  shadowed onto PYTHONPATH (default `src:/tmp/evogp_etl_patched_51a7141`); the
  legacy CUDA side is absent by design.  Results go to results/armB/.

  --group iree-spec  the three IREE workloads with EVOGP_EVAL_MODE=specialized
  --group iree       the same three with EVOGP_EVAL_MODE=auto (explicit)
  --group numpy      the three numpy-backend configs (no env overrides)
  --group all        all three, iree-spec then iree then numpy (default)
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
    *) echo "run_all_armB.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  iree-spec|iree|numpy|all) ;;
  *) echo "run_all_armB.sh: --group must be one of iree-spec|iree|numpy|all (got '$GROUP')" >&2; exit 2 ;;
esac

# --- 4. config table --------------------------------------------------------
# Order inside each group is the run order (do not sort).  Every id is the
# uppercase-prefixed id of its Arm A / R2 counterpart, with the identical flags.
IREE_SPEC_IDS=(
  # new side, --backend iree, EVOGP_EVAL_MODE=specialized: the R2-era route pinned on the
  # same workloads (Arm A passed the same env explicitly)
  armB_iree_fixture64_n64_spec armB_iree_fixture200_n64_spec armB_iree_sr_a_spec
)
IREE_IDS=(
  # new side, --backend iree, EVOGP_EVAL_MODE=auto passed EXPLICITLY (Arm A relied on the
  # implicit `auto` default here): the dense one-graph evaluator is the route for these arms
  armB_iree_fixture64_n64 armB_iree_fixture200_n64 armB_iree_sr_a
)
NUMPY_IDS=(
  # new side, default "numpy" backend, --stage-probe ON (results in results/armB/new/)
  armB_np_c5r2 armB_np_fixture200_n64 armB_np_sr_a
)

# config_spec ID -> CONFIG_SIDE (always new here), CONFIG_GROUP, CONFIG_ARGS, CONFIG_ENVS,
#                  CONFIG_FIXTURE (existence-guarded; "" = workload reads no fixture), CONFIG_NOTE
config_spec() {
  CONFIG_SIDE="" ; CONFIG_GROUP="" ; CONFIG_ARGS=() ; CONFIG_ENVS=() ; CONFIG_FIXTURE="" ; CONFIG_NOTE=""
  case "$1" in
    # ---- iree-spec group: --backend iree, EVOGP_EVAL_MODE=specialized -----------------
    armB_iree_fixture64_n64_spec)  CONFIG_SIDE=new; CONFIG_GROUP=iree-spec; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --backend iree --stage-probe)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                         CONFIG_NOTE="forced specialized (R2-era route) for wave attribution; Arm A twin iree_fixture64_n64_spec" ;;
    armB_iree_fixture200_n64_spec) CONFIG_SIDE=new; CONFIG_GROUP=iree-spec; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --backend iree --stage-probe --max-cold-builds 400)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized)
                         CONFIG_NOTE="forced specialized (needs r2 fixture); budget 400 kept; Arm A twin iree_fixture200_n64_spec" ;;
    armB_iree_sr_a_spec)  CONFIG_SIDE=new; CONFIG_GROUP=iree-spec
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --backend iree --stage-probe --max-cold-builds 1500)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=specialized EVOGP_ETL_CACHE_SIZE=8192)
                         CONFIG_NOTE="forced specialized; cache 8192, budget 1500 kept; Arm A twin iree_sr_a_spec" ;;

    # ---- iree group: --backend iree, EVOGP_EVAL_MODE=auto passed explicitly ----------
    # `auto` keeps the specialized engine on numpy and picks the dense evaluator on iree
    # (pop >= 16, and <= 2048 rows on per_row=False calls), so these arms are the
    # post-wave-2 default route.
    armB_iree_fixture64_n64)  CONFIG_SIDE=new; CONFIG_GROUP=iree; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --backend iree --stage-probe)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=auto)
                         CONFIG_NOTE="IREE auto route (dense), eval mode explicit; Arm A twin iree_fixture64_n64" ;;
    armB_iree_fixture200_n64) CONFIG_SIDE=new; CONFIG_GROUP=iree; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --backend iree --stage-probe --max-cold-builds 400)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=auto)
                         CONFIG_NOTE="IREE auto route (needs r2 fixture); cold-build budget 400 kept; Arm A twin iree_fixture200_n64" ;;
    armB_iree_sr_a)           CONFIG_SIDE=new; CONFIG_GROUP=iree
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --backend iree --stage-probe --max-cold-builds 1500)
                         CONFIG_ENVS=(EVOGP_EVAL_MODE=auto EVOGP_ETL_CACHE_SIZE=8192)
                         CONFIG_NOTE="IREE auto route: short W1 SR, cache 8192; cold-build budget 1500 kept; Arm A twin iree_sr_a" ;;

    # ---- numpy group: default backend, no env overrides -------------------------------
    armB_np_c5r2)             CONFIG_SIDE=new; CONFIG_GROUP=numpy; CONFIG_FIXTURE="$DEFAULT_FIXTURE"
                         CONFIG_ARGS=(--task eval64 --eval-reps 25 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_c5r2; Arm A twin np_c5r2" ;;
    armB_np_fixture200_n64)   CONFIG_SIDE=new; CONFIG_GROUP=numpy; CONFIG_FIXTURE="$FIXTURE200"
                         CONFIG_ARGS=(--task eval64 --fixture "$FIXTURE200" --eval-reps 25 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_fixture200_n64; Arm A twin np_fixture200_n64" ;;
    armB_np_sr_a)             CONFIG_SIDE=new; CONFIG_GROUP=numpy
                         CONFIG_ARGS=(--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --stage-probe)
                         CONFIG_NOTE="numpy mirror of leg_sr_a; Arm A twin np_sr_a" ;;
    *) return 1 ;;
  esac
  return 0
}

# NOTE on per-config environments: EVOGP_ETL_CACHE_SIZE=8192 for the `*_sr_a*` arms (the engine's
# per-process per-structure build cache is small by default and these runs change structure on
# almost every generation); EVOGP_EVAL_MODE is passed explicitly on all six IREE arms
# (`specialized` for the `iree-spec` group, `auto` for the `iree` group -- Arm A left the latter
# implicit).  Such overrides are exported in the run env via `env VAR=...` below -- NOT set
# globally -- so the other configs keep the harness defaults, and every value is recorded in each
# config's JSON `env` block.

all_ids() { printf '%s\n' "${IREE_SPEC_IDS[@]}" "${IREE_IDS[@]}" "${NUMPY_IDS[@]}"; }

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# canonical_id ID -> prints the config-table spelling of ID (matched case-insensitively,
# because every Arm B id carries a capital `B`: `--only armB_np_sr_a` == `--only armb_np_sr_a`);
# returns 1 when the id is not in the table.
canonical_id() {
  local want="$1" cand
  if config_spec "$want"; then printf '%s' "$want"; return 0; fi
  while IFS= read -r cand; do
    if [[ "$(lower "$cand")" == "$(lower "$want")" ]]; then printf '%s' "$cand"; return 0; fi
  done < <(all_ids)
  return 1
}

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "run_all_armB.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_all_armB.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi

# advisory capability probe: the R2 CLI (--backend / --data-rows / --stage-probe /
# --max-cold-builds), reused unchanged by R3, is a frozen contract implemented in bench_gp.py.
# Warn loudly when the harness on disk does not have it yet, so the resulting argparse
# failures are not misread as engine problems.  (Advisory only -- never blocks a run.)
if ! grep -q -e '--stage-probe' "$BENCH" 2>/dev/null; then
  echo "run_all_armB.sh: WARNING: $BENCH does not advertise --stage-probe (R2 CLI not present?);" >&2
  echo "             every probed/IREE config would fail with an argparse error -- materialize" >&2
  echo "             the R2 harness change first (the CLI contract is frozen)." >&2
fi

# selection
SELECTED=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_raw <<< "$ONLY"
  _only=()
  for _id in "${_only_raw[@]}"; do
    _id="$(printf '%s' "$_id" | tr -d '[:space:]')"
    [[ -z "$_id" ]] && continue
    if ! _canon="$(canonical_id "$_id")"; then
      echo "run_all_armB.sh: unknown config id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_canon")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_all_armB.sh: --only is empty" >&2; exit 2; }
  while IFS= read -r _id; do            # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done < <(all_ids)
else
  case "$GROUP" in
    iree-spec) SELECTED=("${IREE_SPEC_IDS[@]}") ;;
    iree)      SELECTED=("${IREE_IDS[@]}") ;;
    numpy)     SELECTED=("${NUMPY_IDS[@]}") ;;
    all)       SELECTED=("${IREE_SPEC_IDS[@]}" "${IREE_IDS[@]}" "${NUMPY_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_all_armB.sh: nothing selected" >&2; exit 2; }

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
# Arm B has no legacy side, so only the fixture guard remains.
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
  printf '%-32s %-10s %-5s %-9s %-34s %s\n' ID GROUP SIDE TASK TAG JSON
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    task="?"
    for (( i=0; i<${#CONFIG_ARGS[@]}; i++ )); do
      [[ "${CONFIG_ARGS[$i]}" == "--task" ]] && task="${CONFIG_ARGS[$((i+1))]}"
    done
    if [[ "$CONFIG_SIDE" == leg ]]; then json="$LEG_RESULT_DIR/$id.json"; else json="$NEW_RESULT_DIR/$id.json"; fi
    printf '%-32s %-10s %-5s %-9s %-34s %s\n' "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" "$task" \
      "$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')" "$json"
  done
  echo
  echo "-- exact args per config (+ per-config env overrides, + pre-flight guard state) --"
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    args="${CONFIG_ARGS[*]}"
    envs=""
    if (( ${#CONFIG_ENVS[@]} > 0 )); then envs="  env: ${CONFIG_ENVS[*]}"; fi
    printf '%-32s %s%s\n' "$id" "$args" "$envs"
    reason="$(guard_reason "$id")"
    if [[ -n "$reason" ]]; then
      printf '%-32s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one config ------------------------------------------------------
run_one() {
  local id="$1"
  config_spec "$id" || { echo "run_all_armB.sh: unknown config id '$id'" >&2; return 1; }

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
    SUMMARY+=("$(printf '%-32s %-5s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_SIDE" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$id")
    return 0
  fi

  mkdir -p "$dir"
  {
    echo "==============================================================================="
    echo "Arm B config $id  (side=$CONFIG_SIDE group=$CONFIG_GROUP)  started $(date -Is)"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "etl: $(env PYTHONPATH="$pyp" "$PY" -c 'import etl; print(etl.__file__)' 2>&1 | tail -1)"
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
    echo "Arm B config $id  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "config $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-32s %-5s %-6s %-9s %s' "$id" "$CONFIG_SIDE" "$rc" "$status" "$json")")
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

echo "r3/run_all_armB.sh: py=$PY"
echo "r3/run_all_armB.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r3/run_all_armB.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN configs=$TOTAL timeout=${TIMEOUT_S}s lock=$LOCK"
echo "r3/run_all_armB.sh: etl shadow: PYTHONPATH=$NEW_PYTHONPATH"
echo "r3/run_all_armB.sh: gpu snapshot: $(gpu_line)"
if (( DRY_RUN )); then
  echo "r3/run_all_armB.sh: DRY RUN -- nothing is executed, no results dirs are created"
fi
echo

for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r3 run_all_armB.sh summary (group=$GROUP, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-32s %-5s %-6s %-9s %s\n' ID SIDE EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} config(s) planned (dry run; test/benchmark/r2/round3/results/armB/last_run.txt NOT written)"
  exit 0
fi

{
  echo "# test/benchmark/r2/round3/results/armB/last_run.txt -- id -> exit-code table of the last r3 run_all_armB.sh invocation"
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} configs=$TOTAL timeout=${TIMEOUT_S}s"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK new_pythonpath=$NEW_PYTHONPATH leg_pythonpath=$LEG_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-32s %-5s %-6s %-9s %s\n' ID SIDE EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
