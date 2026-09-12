#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r4/run_e2e.sh -- R4 after-round END-TO-END (measurement type 1)
#   matrix runner: the public-API problem.evaluate(forest) measurement on the ETL
#   "true jit" engine (numpy + IREE-cuda), 12 patch-shadow cells + 2 stock cells
# =============================================================================
#
# Measurement type 1 is what R1-R3 quote: `bench_gp.py --task eval64` drives the
# real public API on a FIXED population (the 64-tree golden fixture, or the
# 200-tree R2 fixture), warms it and times 25 synchronized `problem.evaluate()`
# calls -- one process per config, `median_warm_ms` = the warm/steady-state number
# of record.  R4's node re-runs exactly that route after the code round, on the
# same 14-record matrix the Round-3 Arm A/Arm B comparison was built from:
#
#   12 SHADOW cells  {fixture64, fixture200} x {numpy, iree} x EVOGP_EVAL_MODE
#                    {auto, dynamic, specialized}, measured against the PATCHED
#                    etl checkout (the R3 Arm B wiring, so this is the after-state);
#    2 STOCK cells   the headline `iree` + `auto` cells (fixture64 / fixture200)
#                    measured in the SAME window against the STOCK etl checkout,
#                    so "patched vs stock" is a same-window delta and never a
#                    cross-window one.  In the full run (`--group all`) each stock
#                    cell follows its shadow twin, so the pair sits next to each
#                    other in time.
#
# R3 anchor (for the after-round diff): iree auto = 10.64 ms (fixture64) /
# 10.50 ms (fixture200) warm median per evaluate; legacy CUDA 0.6323 / 0.6357 ms.
#
# The record table lives HERE (ids + order); the validator's expectation list is
# test/benchmark/r4/e2e_manifest.txt (6 TAB-separated fields) -- keep the two in
# sync (ids, fixture/backend/mode/kind columns, and the id order).
# =============================================================================
#
# One `bench_gp.py` PROCESS per record, serialized on the shared benchmark lock.
# Run it from the REPOSITORY ROOT (the checkout that owns test/benchmark/):
#
#     test/benchmark/r4/run_e2e.sh                    # all 14 records, table order
#     test/benchmark/r4/run_e2e.sh --group shadow     # 12 records: the patched-ETL cells
#     test/benchmark/r4/run_e2e.sh --group stock      # 2 records: the stock counterparts
#     test/benchmark/r4/run_e2e.sh --only e2e_fixture64_numpy_auto
#     test/benchmark/r4/run_e2e.sh --only e2e_fixture64_iree_auto,e2e_stock_fixture64_iree_auto
#     test/benchmark/r4/run_e2e.sh --only e2e_fixture64_iree_auto --timeout 3600
#     test/benchmark/r4/run_e2e.sh --list             # print the config table
#     test/benchmark/r4/run_e2e.sh --dry-run          # print every command, run nothing
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2).
#
# Rules baked in here (the R2/R3/R4 runner rules; do not "improve" them per-run):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file descriptors
#     (the R3 stock-ETL records ended at ~6.5k fds).
#   * exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per record, never nested
#     (other agents share that lock), with the `timeout` INSIDE the lock so lock wait
#     time is never charged to a record.
#   * one process per record => the engine's per-process `etl.CompileCache` and the
#     dense evaluator's own graph cache are never carried over, so `first_call_ms` /
#     the cold-build counters belong to that record alone and `median_warm_ms` is
#     steady-state by construction.
#   * every record carries ETL_DISABLE_LOCATIONS=1 on top of the selected backend /
#     route: ETL_DISABLE_LOCATIONS=1  EVOGP_ETL_BACKEND=<numpy|iree>
#     EVOGP_EVAL_MODE=<auto|dynamic|specialized>  and, on the iree cells,
#     EVOGP_ETL_IREE_DEVICE=0  EVOGP_ETL_IREE_OPT_LEVEL=O1
#     (O2/O3 miscompile CUDA silently -- never used here).
#   * kind is expressed ONLY by PYTHONPATH (which `etl` gets imported): the shadow
#     cells get `src:<shadow checkout>`, the stock cells exactly `src`.  That fact is
#     written into the record itself (`--notes ... PYTHONPATH=...`) AND into the
#     per-record log's `etl:` identity line, which is what check_e2e_r4.py verifies.
#   * the runner always passes `--tag` = UPPER(id) and `--json-out <results>/<id>.json`;
#     `--seed 0` is the harness default and is left implicit; `--fixture` paths are
#     relative to the repo root.
#   * `--max-cold-builds 400` is raised ONLY on the fixture200 iree cells (R3 mirror);
#     every other record keeps the harness default 300.
#   * a failed record never aborts the batch: log + FAIL + keep going.
#   * a re-run APPENDS to the record's log (R2/R3/R4 house rule), and check_e2e_r4.py
#     reads every `etl:` line it finds: a log that carries two different kinds (a stock
#     attempt followed by a shadow one) is ambiguous evidence and fails the kind check --
#     delete the stale log before re-running a record under the other kind.#
# Environment overrides (all optional):
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault and is REFUSED
#   TIMEOUT_S             per-record wall cap, default 1800 (or pass --timeout SECONDS);
#                         the slow record is the fixture200 iree specialized cell
#                         (R3: 143 s wall), so the default is generous.
#   LOCK / FLOCK_WAIT     shared lock path / flock wait (default /tmp/gpbench_critical.lock, 7200)
#   SHADOW_DIR            patched-ETL checkout materialized outside the repo
#                         (default /tmp/evogp_etl_patched_87c7562); missing => the
#                         shadow cells are GUARD-SKIPPED (exit 3, counts as OK)
#   SHADOW_PYTHONPATH     PYTHONPATH of the shadow cells (default `src:$SHADOW_DIR`)
#   STOCK_PYTHONPATH      PYTHONPATH of the stock cells (default `src` -- the editable
#                         etl install every other round used)
#   ETL_REPO              the ETL checkout whose HEAD is recorded in every log
#                         (default /mnt/local-ssd/bchuang/etl; read-only, never written)
#   E2E_RESULT_DIR        results dir (default test/benchmark/r4/results/after/e2e)
#
# Exit codes:
#   0  every selected record was OK or guard-skipped (bench_gp.py exit 3 = skipped, OK)
#   1  at least one record exited non-zero (timeout 124/137, crash 1, interrupt 130, ...)
#   2  usage / preconditions (not run from the repo root, unknown id, missing interpreter,
#      an id the config table cannot derive, GPU 7)
# Per-record results land in
#   test/benchmark/r4/results/after/e2e/<id>.{json,log}
# and the id -> exit-code table is written to
#   test/benchmark/r4/results/after/e2e/run_e2e_r4.last_run.txt
# (generated output, disposable, never committed; the logs ARE evidence -- the repo-root
# `*.log` ignore is re-opened for `results/**/*.log` by test/benchmark/r4/.gitignore).
#
# Validate afterwards with:
#   python3 test/benchmark/r4/check_e2e_r4.py
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_e2e.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE needs it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
RESULT_DIR="${E2E_RESULT_DIR:-test/benchmark/r4/results/after/e2e}"
SUMMARY_FILE="$RESULT_DIR/run_e2e_r4.last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"
FLOCK_WAIT="${FLOCK_WAIT:-7200}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

SHADOW_DIR="${SHADOW_DIR:-/tmp/evogp_etl_patched_87c7562}"
SHADOW_PYTHONPATH="${SHADOW_PYTHONPATH:-src:$SHADOW_DIR}"
STOCK_PYTHONPATH="${STOCK_PYTHONPATH:-src}"
ETL_REPO="${ETL_REPO:-/mnt/local-ssd/bchuang/etl}"

FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz
EVAL_REPS=25

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r4/run_e2e.sh [--group shadow|stock|all] [--only id1,id2]
                                   [--timeout SECONDS] [--list] [--dry-run] [--help]

  --group shadow  the 12 patched-ETL cells (fixture64/200 x numpy/iree x auto/dynamic/specialized)
  --group stock   the 2 stock-ETL counterparts of the headline iree-auto cells
  --group all     all 14 in table order (default): shadow iree-auto + its stock twin per
                  fixture first, then the remaining shadow cells
  --only IDS      run exactly these config ids (comma-separated); overrides --group
  --timeout S     per-record wall cap, fed to `timeout` INSIDE the lock (default: $TIMEOUT_S)
  --list          print the config table (id, kind, fixture, backend, mode, tag, json) and exit
  --dry-run       print every command without executing anything (creates no results dir)
  --help          this text

Run from the repository root.  Gate: python3 test/benchmark/r4/check_e2e_r4.py
Keep the id/order in sync with test/benchmark/r4/e2e_manifest.txt.
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
    *) echo "run_e2e.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  shadow|stock|all) ;;
  *) echo "run_e2e.sh: --group must be one of shadow|stock|all (got '$GROUP')" >&2; exit 2 ;;
esac

# --- 4. record table --------------------------------------------------------
# ALL_IDS is the canonical order (--only selects in this order, --group all runs it as is).
# The headline pairs come first so that a stock cell and its shadow twin are adjacent in
# time; the remaining shadow cells follow, numpy first (an IREE-window problem can then
# never invalidate the numpy rows), fixture64 before fixture200, auto -> dynamic ->
# specialized.  The ids are the ONLY place the matrix is written down: config_spec()
# derives kind/fixture/backend/mode from the id text (and rejects anything that is not in
# ALL_IDS), so a typo can never run a mis-derived config -- the startup table self-check
# below proves it.
ALL_IDS=(
  # headline pairs (shadow iree-auto immediately followed by its stock counterpart)
  e2e_fixture64_iree_auto
  e2e_stock_fixture64_iree_auto
  e2e_fixture200_iree_auto
  e2e_stock_fixture200_iree_auto
  # remaining shadow cells: numpy backend, both fixtures, auto -> dynamic -> specialized
  e2e_fixture64_numpy_auto
  e2e_fixture64_numpy_dynamic
  e2e_fixture64_numpy_specialized
  e2e_fixture200_numpy_auto
  e2e_fixture200_numpy_dynamic
  e2e_fixture200_numpy_specialized
  # remaining shadow cells: iree backend (auto already measured above)
  e2e_fixture64_iree_dynamic
  e2e_fixture64_iree_specialized
  e2e_fixture200_iree_dynamic
  e2e_fixture200_iree_specialized
)
SHADOW_IDS=(
  e2e_fixture64_iree_auto
  e2e_fixture200_iree_auto
  e2e_fixture64_numpy_auto
  e2e_fixture64_numpy_dynamic
  e2e_fixture64_numpy_specialized
  e2e_fixture200_numpy_auto
  e2e_fixture200_numpy_dynamic
  e2e_fixture200_numpy_specialized
  e2e_fixture64_iree_dynamic
  e2e_fixture64_iree_specialized
  e2e_fixture200_iree_dynamic
  e2e_fixture200_iree_specialized
)
STOCK_IDS=(
  e2e_stock_fixture64_iree_auto
  e2e_stock_fixture200_iree_auto
)

# config_spec ID -> CONFIG_KIND (shadow|stock), CONFIG_FIXTURE (fixture64|fixture200),
#                  CONFIG_POP (64|200), CONFIG_FIXTURE_FILE (root-relative npz),
#                  CONFIG_BACKEND (numpy|iree), CONFIG_MODE (auto|dynamic|specialized),
#                  CONFIG_PYTHONPATH, CONFIG_ARGS, CONFIG_ENVS, CONFIG_NOTE, CONFIG_NOTES
# Id shape: e2e_[stock_]<fixture>_<backend>_<mode>; anything else is rejected.
config_spec() {
  CONFIG_KIND="" ; CONFIG_FIXTURE="" ; CONFIG_POP="" ; CONFIG_FIXTURE_FILE=""
  CONFIG_BACKEND="" ; CONFIG_MODE="" ; CONFIG_PYTHONPATH=""
  CONFIG_ARGS=() ; CONFIG_ENVS=() ; CONFIG_NOTE="" ; CONFIG_NOTES=""

  local id="$1" rest="$1" candidate known=0
  case "$rest" in
    e2e_stock_*) CONFIG_KIND=stock  ; rest="${rest#e2e_stock_}" ;;
    e2e_*)       CONFIG_KIND=shadow ; rest="${rest#e2e_}" ;;
    *) return 1 ;;
  esac
  CONFIG_FIXTURE="${rest%%_*}"          # fixture64 | fixture200
  rest="${rest#*_}"
  CONFIG_BACKEND="${rest%%_*}"          # numpy | iree
  CONFIG_MODE="${rest#*_}"              # auto | dynamic | specialized

  # the id must be one of the 14 canonical ids: a typo in a table (or in --only) can never
  # run a config that the rest of this script (and the manifest) does not know about.
  for candidate in "${ALL_IDS[@]}"; do
    if [[ "$candidate" == "$id" ]]; then known=1; break; fi
  done
  (( known )) || return 1

  case "$CONFIG_FIXTURE" in
    fixture64)  CONFIG_FIXTURE_FILE="$FIXTURE64"  ; CONFIG_POP=64
                CONFIG_NOTE="64-tree golden fixture ($FIXTURE64)" ;;
    fixture200) CONFIG_FIXTURE_FILE="$FIXTURE200" ; CONFIG_POP=200
                CONFIG_NOTE="200-tree R2 fixture ($FIXTURE200; needs test/benchmark/fixtures/)" ;;
    *) return 1 ;;
  esac
  case "$CONFIG_BACKEND" in numpy|iree) ;; *) return 1 ;; esac
  case "$CONFIG_MODE" in auto|dynamic|specialized) ;; *) return 1 ;; esac

  # kind == which `etl` checkout is imported (recorded in the log's `etl:` line + --notes)
  CONFIG_PYTHONPATH="$STOCK_PYTHONPATH"
  if [[ "$CONFIG_KIND" == shadow ]]; then CONFIG_PYTHONPATH="$SHADOW_PYTHONPATH"; fi

  # --- per-record flags (R3 mirror) ---------------------------------------
  CONFIG_ARGS=(--task eval64)
  if [[ "$CONFIG_FIXTURE" == fixture200 ]]; then CONFIG_ARGS+=(--fixture "$FIXTURE200"); fi
  CONFIG_ARGS+=(--eval-reps "$EVAL_REPS" --mem-guard-mb 8192)
  # --- per-record env (exported via `env VAR=...`, never globally) ---------
  CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=1 "EVOGP_ETL_BACKEND=$CONFIG_BACKEND" "EVOGP_EVAL_MODE=$CONFIG_MODE")
  if [[ "$CONFIG_BACKEND" == iree ]]; then
    CONFIG_ARGS+=(--backend iree)
    CONFIG_ENVS+=(EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)
    # R3 mirror: the cold-build budget is raised to 400 ONLY on the fixture200 iree cells.
    if [[ "$CONFIG_FIXTURE" == fixture200 ]]; then CONFIG_ARGS+=(--max-cold-builds 400); fi
  fi
  CONFIG_ARGS+=(--stage-probe)

  # the record must carry its own kind identity: check_e2e_r4.py reads this PYTHONPATH out
  # of `notes` (and out of the log's `etl:` line) to prove shadow != stock.
  CONFIG_NOTES="after-round E2E type 1: kind=$CONFIG_KIND fixture=$CONFIG_FIXTURE backend=$CONFIG_BACKEND mode=$CONFIG_MODE PYTHONPATH=$CONFIG_PYTHONPATH"
  return 0
}

# NOTE on the per-config environments: EVOGP_ETL_BACKEND pins the backend the record claims
# (the harness re-exports the same value for `--backend iree`), EVOGP_EVAL_MODE pins the route
# (`auto` = the engine's measured policy -> dense on iree for these fixtures, `dynamic` = the
# `dense` alias, `specialized` = the R2-era route).  Every value lands in the record's `env`
# block, and check_e2e_r4.py verifies it against the manifest's backend/mode columns.
# NOTE on `--notes`: it is the only harness channel that records a non-ETL_* env var, so the
# effective PYTHONPATH (i.e. the kind) is written there in addition to the log header.

all_ids() { printf '%s\n' "${ALL_IDS[@]}"; }

# --- 5. preconditions -------------------------------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "run_e2e.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_e2e.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi
if [[ "$GPU" == "7" || ",$GPU," == *,7,* ]]; then
  echo "run_e2e.sh: CUDA_VISIBLE_DEVICES='$GPU' names GPU 7, which has a hardware ECC fault" >&2
  echo "             (any CUDA context on it dies) -- use the default 6" >&2
  exit 2
fi

# advisory capability probe: --task eval64 / --eval-reps / --stage-probe are the frozen CLI
# contract implemented in bench_gp.py.  Warn loudly when the harness on disk does not have it,
# so the resulting argparse failures are not misread as engine problems.  (Advisory only.)
for flag in --eval-reps --stage-probe --mem-guard-mb; do
  if ! grep -q -e "$flag" "$BENCH" 2>/dev/null; then
    echo "run_e2e.sh: WARNING: $BENCH does not advertise $flag (the eval64 CLI contract is" >&2
    echo "             missing?); every record would fail with an argparse error." >&2
  fi
done

# --- 5b. table self-check ---------------------------------------------------
# The id tables, the id parser and the group arrays must agree BEFORE anything runs.
TABLE_OK=1
_seen_ids=""
for id in "${ALL_IDS[@]}"; do
  if ! config_spec "$id"; then
    echo "run_e2e.sh: config_spec() cannot derive '$id' (id tables / parser out of sync)" >&2
    TABLE_OK=0 ; continue
  fi
  case " $_seen_ids " in
    *" $id "*) echo "run_e2e.sh: duplicate id '$id' in ALL_IDS" >&2 ; TABLE_OK=0 ;;
  esac
  _seen_ids="$_seen_ids $id"
done
for id in "${SHADOW_IDS[@]}"; do
  config_spec "$id" || { echo "run_e2e.sh: SHADOW_IDS lists unknown id '$id'" >&2 ; TABLE_OK=0 ; continue ; }
  if [[ "$CONFIG_KIND" != shadow ]]; then
    echo "run_e2e.sh: SHADOW_IDS lists '$id' but its id text says kind=$CONFIG_KIND" >&2 ; TABLE_OK=0
  fi
done
for id in "${STOCK_IDS[@]}"; do
  config_spec "$id" || { echo "run_e2e.sh: STOCK_IDS lists unknown id '$id'" >&2 ; TABLE_OK=0 ; continue ; }
  if [[ "$CONFIG_KIND" != stock ]]; then
    echo "run_e2e.sh: STOCK_IDS lists '$id' but its id text says kind=$CONFIG_KIND" >&2 ; TABLE_OK=0
  fi
done
if (( ${#SHADOW_IDS[@]} + ${#STOCK_IDS[@]} != ${#ALL_IDS[@]} )); then
  echo "run_e2e.sh: the shadow/stock groups do not partition ALL_IDS" >&2 ; TABLE_OK=0
fi
(( TABLE_OK )) || { echo "run_e2e.sh: refusing to run with an inconsistent record table (see above)" >&2; exit 2; }

# selection
SELECTED=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_raw <<< "$ONLY"
  _only=()
  for _id in "${_only_raw[@]}"; do
    _id="$(printf '%s' "$_id" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    [[ -z "$_id" ]] && continue
    if ! config_spec "$_id"; then
      echo "run_e2e.sh: unknown config id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_e2e.sh: --only is empty" >&2; exit 2; }
  for _id in "${ALL_IDS[@]}"; do         # canonical (table) order
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done
else
  case "$GROUP" in
    shadow) SELECTED=("${SHADOW_IDS[@]}") ;;
    stock)  SELECTED=("${STOCK_IDS[@]}") ;;
    all)    SELECTED=("${ALL_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_e2e.sh: nothing selected" >&2; exit 2; }

# --- 6. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
}

# etl_identity_line PYTHONPATH -> one line of evidence about WHICH etl checkout that record
# imports.  Cheap and failure-tolerant: a missing checkout / a broken import must never fail a
# config.  NB `rev` is the HEAD of $ETL_REPO (a git checkout); the patched shadow is normally
# materialized with `git archive` (no .git), so the discriminating fields are `file=`
# (etl.__file__) and `pythonpath=` -- exactly what check_e2e_r4.py's kind check reads.
etl_identity_line() {
  local pyp="$1" rev path
  rev="$(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"
  path="$(env "PYTHONPATH=$pyp" "$PY" -c 'import etl; print(etl.__file__)' 2>/dev/null \
          || echo '<import etl failed>')"
  echo "etl: repo=$ETL_REPO rev=$rev file=$path pythonpath=$pyp"
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

# guard_reason ID -> prints a skip reason (empty when the record is runnable)
guard_reason() {
  local id="$1"
  config_spec "$id" || return 0
  if [[ "$CONFIG_KIND" == shadow && ! -d "$SHADOW_DIR" ]]; then
    echo "shadow checkout missing ($SHADOW_DIR; materialized outside the repo, SHADOW_DIR overrides)"
    return 0
  fi
  if [[ ! -f "$CONFIG_FIXTURE_FILE" ]]; then
    echo "fixture missing ($CONFIG_FIXTURE_FILE; relative to the repo root)"
    return 0
  fi
  return 0
}

# --- 7. --list --------------------------------------------------------------
if (( LIST )); then
  printf '%-31s %-6s %-11s %-8s %-12s %-31s %s\n' ID KIND FIXTURE BACKEND MODE TAG JSON
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    printf '%-31s %-6s %-11s %-8s %-12s %-31s %s\n' "$id" "$CONFIG_KIND" "$CONFIG_FIXTURE" \
      "$CONFIG_BACKEND" "$CONFIG_MODE" "$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')" \
      "$RESULT_DIR/$id.json"
  done
  echo
  echo "-- exact args per record (+ per-record env, PYTHONPATH = kind, + pre-flight guard state) --"
  for id in "${SELECTED[@]}"; do
    config_spec "$id"
    printf '%-31s %s\n' "$id" "${CONFIG_ARGS[*]}"
    printf '%-31s env: PYTHONPATH=%s %s\n' "" "$CONFIG_PYTHONPATH" "${CONFIG_ENVS[*]}"
    if [[ -n "$CONFIG_NOTE" ]]; then printf '%-31s note: %s (pop=%s)\n' "" "$CONFIG_NOTE" "$CONFIG_POP"; fi
    reason="$(guard_reason "$id")"
    if [[ -n "$reason" ]]; then
      printf '%-31s GUARD-SKIP: %s\n' "" "$reason"
    fi
  done
  exit 0
fi

# --- 8. run one record ------------------------------------------------------
run_one() {
  local id="$1"
  config_spec "$id" || { echo "run_e2e.sh: unknown config id '$id'" >&2; return 1; }

  local tag json log
  tag="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  json="$RESULT_DIR/$id.json"
  log="$RESULT_DIR/$id.log"

  # pre-flight guards: a record whose shadow checkout / fixture is absent is "skipped by
  # guard" (exit code 3 == OK), never a fake failure and never a silent omission.
  local reason; reason="$(guard_reason "$id")"

  local -a child=(env "PYTHONPATH=$CONFIG_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$GPU")
  local e
  for e in "${CONFIG_ENVS[@]}"; do child+=("$e"); done
  child+=("$PY" "$BENCH" "${CONFIG_ARGS[@]}")
  child+=("--tag" "$tag" "--notes" "$CONFIG_NOTES" "--json-out" "$json")
  local -a timed=(timeout -s INT -k 30 "$TIMEOUT_S" "${child[@]}")
  local inner; inner="$(printf '%q ' "${timed[@]}")"; inner="${inner% }"

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $id  (kind=$CONFIG_KIND fixture=$CONFIG_FIXTURE backend=$CONFIG_BACKEND mode=$CONFIG_MODE pop=$CONFIG_POP)"
    if [[ -n "$reason" ]]; then
      echo "        GUARD-SKIP: $reason   -> recorded as exit 3 (OK)"
    fi
    echo "        json: $json"
    echo "        log : $log"
    echo "        pythonpath: $CONFIG_PYTHONPATH"
    if [[ -n "$CONFIG_NOTE" ]]; then echo "        note: $CONFIG_NOTE"; fi
    # the inner command contains no single quotes by construction, so the printed
    # single-quoted form is exactly what `bash -c` receives at run time.
    echo "        flock -w $FLOCK_WAIT $LOCK bash -c '$inner'"
    echo "        (timeout -s INT -k 30 ${TIMEOUT_S}s runs INSIDE the lock; stdout+stderr -> tee -a $log)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "record $id: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-31s %-6s %-11s %-8s %-12s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_KIND" "$CONFIG_FIXTURE" "$CONFIG_BACKEND" "$CONFIG_MODE" 3 "SKIP" "$json" "$reason")")
    SKIPPED+=("$id")
    return 0
  fi

  mkdir -p "$RESULT_DIR"
  {
    echo "==============================================================================="
    echo "R4 after-round E2E record $id  (kind=$CONFIG_KIND fixture=$CONFIG_FIXTURE backend=$CONFIG_BACKEND mode=$CONFIG_MODE pop=$CONFIG_POP)  started $(date -Is)"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "timeout: ${TIMEOUT_S}s (SIGINT, SIGKILL after 30s) INSIDE flock -w $FLOCK_WAIT $LOCK"
    echo "gpu (before): $(gpu_line)"
    echo "$(etl_identity_line "$CONFIG_PYTHONPATH")"
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
    echo "R4 after-round E2E record $id  finished $(date -Is)"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "record $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-31s %-6s %-11s %-8s %-12s %-6s %-9s %s' "$id" "$CONFIG_KIND" "$CONFIG_FIXTURE" "$CONFIG_BACKEND" "$CONFIG_MODE" "$rc" "$status" "$json")")
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

echo "r4/run_e2e.sh: py=$PY"
echo "r4/run_e2e.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r4/run_e2e.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN records=$TOTAL timeout=${TIMEOUT_S}s lock=$LOCK"
echo "r4/run_e2e.sh: results=$RESULT_DIR  stock_pythonpath=$STOCK_PYTHONPATH  shadow_pythonpath=$SHADOW_PYTHONPATH"
echo "r4/run_e2e.sh: gpu snapshot: $(gpu_line)"
if [[ -d "$SHADOW_DIR" ]]; then
  echo "r4/run_e2e.sh: shadow checkout present: $SHADOW_DIR"
else
  echo "r4/run_e2e.sh: shadow checkout MISSING: $SHADOW_DIR -- shadow cells will be GUARD-SKIPPED (exit 3)"
fi
if (( DRY_RUN )); then
  echo "r4/run_e2e.sh: DRY RUN -- nothing is executed, no results dirs are created, no etl probe runs"
fi
echo

for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
done

# --- 10. summary ------------------------------------------------------------
echo
echo "=== r4 run_e2e.sh summary (group=$GROUP, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-31s %-6s %-11s %-8s %-12s %-6s %-9s %s\n' ID KIND FIXTURE BACKEND MODE EXIT STATUS JSON
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo "ok: 0 executed, ${#SELECTED[@]} record(s) planned (dry run; $SUMMARY_FILE NOT written)"
  exit 0
fi

{
  echo "# $SUMMARY_FILE -- id -> exit-code table of the last r4 run_e2e.sh invocation"
  echo "# generated output: disposable, never committed."
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} records=$TOTAL timeout=${TIMEOUT_S}s"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY lock=$LOCK stock_pythonpath=$STOCK_PYTHONPATH shadow_pythonpath=$SHADOW_PYTHONPATH"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-31s %-6s %-11s %-8s %-12s %-6s %-9s %s\n' ID KIND FIXTURE BACKEND MODE EXIT STATUS JSON
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$SUMMARY_FILE"

echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "summary table: $SUMMARY_FILE"
echo "validate with: python3 test/benchmark/r4/check_e2e_r4.py"
echo "               (add --dir $RESULT_DIR --manifest test/benchmark/r4/e2e_manifest.txt only if the paths were overridden)"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
