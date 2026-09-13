#!/usr/bin/env bash
# =============================================================================
# test/benchmark/r6/run_cells.sh -- R6 SAME-WINDOW CAMPAIGN RUNNER (GPU)
#   legacy CUDA engine  vs  ETL "true jit" dense-iree engine  vs  triton fused
#   prototype -- all three sides measured in ONE continuous window
# =============================================================================
#
# R6 tests the r5 RCA hypothesis (../r5/RCA.md SS9): a fused custom kernel that
# replicates the legacy CUDA engine's design restores legacy-class performance
# for EvoGP's single-output SR case.  This runner produces the timing evidence:
# one fresh PROCESS per cell, every cell serialized on the shared benchmark
# lock, every cell writing its own JSON + log into ONE fresh window directory.
#
# Two measurement types (the r4/r5 protocol, unchanged):
#   TYPE 1  end-to-end public API: `problem.evaluate(forest)` on a FIXED
#           population -- `bench_gp.py --task eval64 --fixture <F> --eval-reps 25`;
#           call 0 is cold, `median_warm_ms` = median(calls 1..24).
#   TYPE 2  pure internal compiled forward -- `bench_gp.py --task eval-pure`
#           (dense-iree) or `run_triton_eval64.py --mode pure` (triton): times
#           only the graph/kernel call(s) after a separately reported
#           build/warmup, with an `--e2e-companion` price for the excluded glue.
#
# THE CELL TABLE (ids are stable and greppable; `--list` prints it):
#
#   group   id               side    type  fixture     command
#   ------  ---------------  ------  ----  ----------  ----------------------------
#   check   check_deps       local   -     -           dependency pre-flight (below)
#   legacy  leg_f64          old     1     fixture64   bench_gp.py --task eval64
#   legacy  leg_f200         old     1     fixture200  bench_gp.py --task eval64
#   dense   dense_f64        new     1     fixture64   bench_gp.py --task eval64
#   dense   dense_f200       new     1     fixture200  bench_gp.py --task eval64
#   triton  triton_f64       triton  1     fixture64   run_triton_eval64.py
#   triton  triton_f200      triton  1     fixture200  run_triton_eval64.py
#   pure    densepure_f64    new     2     fixture64   bench_gp.py --task eval-pure
#   pure    densepure_f200   new     2     fixture200  bench_gp.py --task eval-pure
#   pure    tritonpure_f64   triton  2     fixture64   run_triton_eval64.py --mode pure
#   pure    tritonpure_f200  triton  2     fixture200  run_triton_eval64.py --mode pure
#
# `--group all` runs the ten MEASUREMENT cells in exactly that order (each
# kind's two fixtures adjacent, so the same-window comparison sits in adjacent
# time).  `--group check` is a separate, tiny dependency pre-flight whose report
# lands in the same window dir; run it first with the same --window label.
#
# THE CAMPAIGN INVOCATION (one continuous window, back to back):
#
#   bash test/benchmark/r6/run_cells.sh --group check --window <label>   # pre-flight
#   bash test/benchmark/r6/run_cells.sh --group all   --window <label>   # 10 cells
#
# and the inspection / partial forms:
#
#   bash test/benchmark/r6/run_cells.sh --list
#   bash test/benchmark/r6/run_cells.sh --dry-run --group all
#   bash test/benchmark/r6/run_cells.sh --only triton_f64,densepure_f64
#   bash test/benchmark/r6/run_cells.sh --group legacy --timeout 900
#
# `--only` OVERRIDES `--group` (unknown ids are a usage error, exit 2).  Run
# everything from the REPOSITORY ROOT (fixture paths are cwd-relative and the
# scripts assume this checkout owns test/benchmark/).
#
# LOCK RULE (differs from r4/r5): the `flock(1)` BINARY hangs in this sandbox,
# so every cell is serialized by the coordinator-provided Python wrapper
#
#     "$PY" test/benchmark/r6/run_under_lock.py          # fcntl.flock(LOCK_EX)
#
# on the SAME lock file r4/r5 use (/tmp/gpbench_critical.lock).  Exactly ONE
# lock acquisition per cell, NEVER nested, and `timeout -s INT -k 30 <caps>` is
# passed INSIDE the lock, so lock-wait time is never charged to a measurement.
#
# Rules baked in here (the R2/R3/R4/R5 runner rules; do not "improve" per run):
#   * `ulimit -n 65536` first -- the IREE backend opens hundreds of file
#     descriptors (stock-ETL fd leak; see ../CONTEXT.md).
#   * one process per cell => the ETL per-process `etl.CompileCache` and the
#     dense evaluator's graph cache are never carried over, so first-call /
#     cold-build numbers belong to that cell alone and `median_warm_ms` is
#     steady-state by construction.
#   * every new-side (dense/pure-dense) cell runs with ETL_DISABLE_LOCATIONS=1
#     on top of  EVOGP_ETL_BACKEND=iree  EVOGP_EVAL_MODE=<auto|dense>
#     EVOGP_ETL_IREE_DEVICE=0  EVOGP_ETL_IREE_OPT_LEVEL=O1  (O2/O3 miscompile
#     CUDA silently -- never used here).
#   * legacy cells get the legacy PYTHONPATH and NO ETL_*/EVOGP_* variable at
#     all (they are inert on the old side; the objective's rule is stricter
#     than r5's "export ETL_DISABLE_LOCATIONS=1 everywhere").
#   * triton cells get PYTHONPATH=src + ETL_DISABLE_LOCATIONS=1 (harmless) +
#     TRITON_CACHE_DIR (reproducible compile caching, kept out of the repo) and
#     CPATH when the session's Python.h header tree is present (triton's
#     runtime compile needs Python.h; see ./CONTEXT.md "Environment").  Their
#     driver CLI is PINNED by the round brief -- no extra flags are invented.
#   * a cell that fails never aborts the batch: log + FAIL + keep going; the
#     batch exits 1 at the end.  Exit 3 = guard-skip (counts as OK).
#   * a failed/incomplete cell APPENDS to its log; a re-run of a recorded
#     window must use a NEW --window label (the freshness gate below refuses to
#     write into a window dir that already holds records).
#   * every timed region is inside the harness's own `torch.cuda.synchronize()`
#     brackets -- this runner only orders and records the processes.
#
# RECORDS (fresh dir per run; the window label is the dir name):
#   test/benchmark/r6/results/<window>/<cellid>.json    the bench_gp/triton record
#   test/benchmark/r6/results/<window>/<cellid>.log     run evidence (identity
#                                                      headers, gpu samples, argv)
#   test/benchmark/r6/results/<window>/check_deps.txt   the `check` cell's report
#   test/benchmark/r6/results/<window>/window_meta.txt  the window record: start/
#                                                      end, per-cell timestamps +
#                                                      exit codes, gpu samples
#   test/benchmark/r6/last_run.txt                      disposable id -> exit table
# `results/<window>/` must be FRESH: the runner refuses (exit 2) to write into a
# dir that holds anything except the whitelisted pre-flight artifacts
# (check_deps.txt/.log, window_meta.txt), because r6 records are immutable
# evidence.  A re-run therefore needs a new label.
#
# Environment overrides (all optional):
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  default 6; GPU 7 has a hardware ECC fault and is REFUSED
#   R6_WINDOW             window label when --window is not given
#                         (default: the UTC timestamp `date -u +%Y%m%dT%H%M%SZ`)
#   R6_RESULTS_ROOT       results root (default test/benchmark/r6/results)
#   TIMEOUT_S             per-cell cap for new-side + triton cells (default 900)
#   TIMEOUT_LEGACY        per-cell cap for legacy cells (default 600)
#   TIMEOUT_CHECK         cap for the `check` cell (default 180)
#   LEGACY_ROOT / LEGACY_SITEUTILS / LEG_PYTHONPATH
#                         legacy side (defaults /tmp/evogp_legacy_bench,
#                         /tmp/evogp_t1a263/siteutils, `<root>/src:<siteutils>`)
#   NEW_PYTHONPATH        new-side PYTHONPATH (default `src`)
#   ETL_REPO              ETL checkout whose HEAD the log header records
#                         (default /mnt/local-ssd/bchuang/etl; read-only)
#   TRITON_CACHE_DIR      triton compile cache (default /tmp/evogp_r6_triton_cache)
#   R6_TRITON_CPATH       CPATH for triton cells; `auto` (default) probes
#                         /tmp/triton_feas/py311hdr, `""` exports nothing
#
# Exit codes:
#   0  every selected cell was OK or guard-skipped
#   1  at least one cell exited non-zero (timeout 124/137, crash 1, interrupt 130,
#      or the `check` cell found a MISSING dependency)
#   2  usage / preconditions (not run from the repo root, unknown id, missing
#      interpreter or lock wrapper, GPU 7, non-fresh results dir, bad --window)
#
# Validate the window afterwards with test/benchmark/r6/check_r6.py (GPU-free).
# =============================================================================
set -u

# --- 1. file-descriptor limit FIRST -----------------------------------------
ulimit -n 65536 2>/dev/null || echo "run_cells.sh: WARNING: could not raise 'ulimit -n' to 65536 (IREE cells need it)" >&2

# --- 2. paths and knobs -----------------------------------------------------
R6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BENCH=test/benchmark/bench_gp.py
TRITON_DRIVER="$R6_DIR/run_triton_eval64.py"
LOCK_WRAPPER="$R6_DIR/run_under_lock.py"
RESULTS_ROOT="${R6_RESULTS_ROOT:-test/benchmark/r6/results}"
LAST_RUN_FILE="$R6_DIR/last_run.txt"

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOCK="${LOCK:-/tmp/gpbench_critical.lock}"

LEGACY_ROOT="${LEGACY_ROOT:-/tmp/evogp_legacy_bench}"
LEGACY_SITEUTILS="${LEGACY_SITEUTILS:-/tmp/evogp_t1a263/siteutils}"
LEG_PYTHONPATH="${LEG_PYTHONPATH:-$LEGACY_ROOT/src:$LEGACY_SITEUTILS}"
NEW_PYTHONPATH="${NEW_PYTHONPATH:-src}"
ETL_REPO="${ETL_REPO:-/mnt/local-ssd/bchuang/etl}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/evogp_r6_triton_cache}"
R6_TRITON_CPATH="${R6_TRITON_CPATH:-auto}"

CAP="${TIMEOUT_S:-900}"
CAP_LEGACY="${TIMEOUT_LEGACY:-600}"
CAP_CHECK="${TIMEOUT_CHECK:-180}"

EVAL_REPS=25

# repo-root-relative fixtures (the harness and the triton driver both resolve
# them against the cwd, which must therefore be the repository root)
FIXTURE64=test/goldens/forest_random.npz
FIXTURE200=test/benchmark/fixtures/forest200_tree64_random.npz

# the r5 Q4 / r6 dense-cell env block (default `auto` route on iree => dense evaluator)
NEW_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)

# triton runtime compile needs Python.h; the session keeps a ready header tree
# outside the repo (see ./CONTEXT.md).  `auto` probes it, an explicit value wins.
resolve_triton_cpath() {
  if [[ "$R6_TRITON_CPATH" != "auto" ]]; then printf '%s' "$R6_TRITON_CPATH"; return 0; fi
  local root dirs=()
  for root in /tmp/triton_feas/py311hdr; do
    [[ -d "$root/usr/include/python3.11" ]] && dirs+=("$root/usr/include/python3.11")
    [[ -d "$root/usr/include" ]] && dirs+=("$root/usr/include")
  done
  if (( ${#dirs[@]} > 0 )); then (IFS=:; printf '%s' "${dirs[*]}"); fi
}
TRITON_CPATH_RESOLVED="$(resolve_triton_cpath)"

# the lock path the wrapper actually uses (informational: the wrapper owns it)
WRAPPER_LOCK="$(grep -m1 -oE '/tmp/[A-Za-z0-9_.-]+\.lock' "$LOCK_WRAPPER" 2>/dev/null || true)"
WRAPPER_LOCK="${WRAPPER_LOCK:-<unreadable>}"

# --- 3. usage / argument parsing -------------------------------------------
usage() {
  cat <<'EOF'
usage: test/benchmark/r6/run_cells.sh [--group check|legacy|dense|triton|pure|all]
                                     [--only id1,id2] [--window LABEL] [--timeout SECONDS]
                                     [--list] [--dry-run] [--help]

  --group check   the dependency pre-flight cell (check_deps), same window dir; NOT part of
                  --group all, run it first with the same --window label
  --group legacy  the 2 legacy-CUDA type-1 cells (leg_f64, leg_f200)
  --group dense   the 2 dense-iree type-1 cells (dense_f64, dense_f200)
  --group triton  the 2 triton type-1 cells (triton_f64, triton_f200)
  --group pure    the 4 type-2 pure cells (densepure_f64/200, tritonpure_f64/200)
  --group all     the 10 measurement cells in table order (default)
  --only IDS      run exactly these cell ids (comma-separated); overrides --group
  --window LABEL  results dir name: results/<LABEL>/ (default: $R6_WINDOW or a UTC
                  timestamp); must match [A-Za-z0-9._-]+ and must be FRESH
  --timeout S     per-cell wall cap for every measurement cell (fed to `timeout`
                  INSIDE the lock); the `check` cell keeps its own 180s cap
  --list          print the cell table (id, group, side, type, fixture, cap, out) and exit
  --dry-run       print every command without executing anything (creates nothing)
  --help          this text

Run from the repository root.  Validate with: python3 test/benchmark/r6/check_r6.py
EOF
}

GROUP=all
ONLY=""
WINDOW_ARG=""
DRY_RUN=0
LIST=0
INTERNAL_DEP_CHECK=0
TIMEOUT_OVERRIDE=""
while (( $# > 0 )); do
  case "$1" in
    --group) GROUP="${2:?--group needs a value}"; shift 2 ;;
    --group=*) GROUP="${1#--group=}"; shift ;;
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --window) WINDOW_ARG="${2:?--window needs a value}"; shift 2 ;;
    --window=*) WINDOW_ARG="${1#--window=}"; shift ;;
    --timeout) TIMEOUT_OVERRIDE="${2:?--timeout needs a value}"; shift 2 ;;
    --timeout=*) TIMEOUT_OVERRIDE="${1#--timeout=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --internal-dep-check) INTERNAL_DEP_CHECK=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "run_cells.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
case "$GROUP" in
  check|legacy|dense|triton|pure|all) ;;
  *) echo "run_cells.sh: --group must be one of check|legacy|dense|triton|pure|all (got '$GROUP')" >&2; exit 2 ;;
esac
if [[ -n "$TIMEOUT_OVERRIDE" ]]; then
  CAP="$TIMEOUT_OVERRIDE"
  CAP_LEGACY="$TIMEOUT_OVERRIDE"
fi

# --- 4. window label and result dir ----------------------------------------
WINDOW="$WINDOW_ARG"
[[ -z "$WINDOW" ]] && WINDOW="${R6_WINDOW:-}"
[[ -z "$WINDOW" ]] && WINDOW="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ ! "$WINDOW" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run_cells.sh: invalid window label '$WINDOW' (allowed: [A-Za-z0-9._-]+)" >&2
  exit 2
fi
RESULT_DIR="$RESULTS_ROOT/$WINDOW"

# --- 5. record table --------------------------------------------------------
# ALL_IDS is the canonical MEASUREMENT order (--only selects in this order).
# Table order == run order: each kind's two fixtures adjacent, the type-1
# triple (legacy -> dense -> triton) first and the type-2 pairs after it, so the
# same-window comparisons sit in adjacent time.  The `check` cell sits in its own
# group and is NOT part of --group all (it writes a report, not a record).
CHECK_IDS=(check_deps)
LEGACY_IDS=(leg_f64 leg_f200)
DENSE_IDS=(dense_f64 dense_f200)
TRITON_IDS=(triton_f64 triton_f200)
PURE_IDS=(densepure_f64 densepure_f200 tritonpure_f64 tritonpure_f200)
ALL_IDS=("${LEGACY_IDS[@]}" "${DENSE_IDS[@]}" "${TRITON_IDS[@]}" "${PURE_IDS[@]}")
CANONICAL_IDS=("${CHECK_IDS[@]}" "${ALL_IDS[@]}")

# cell_spec ID -> CONFIG_GROUP, CONFIG_SIDE (local|leg|new|triton), CONFIG_TYPE
#                (-|1|2), CONFIG_FIXTURE (fixture64|fixture200|""),
#                CONFIG_FIXTURE_FILE, CONFIG_ARGS, CONFIG_ENVS, CONFIG_PYTHONPATH,
#                CONFIG_CAP, CONFIG_NOTE, CONFIG_NOTES
cell_spec() {
  CONFIG_GROUP="" ; CONFIG_SIDE="" ; CONFIG_TYPE="" ; CONFIG_FIXTURE=""
  CONFIG_FIXTURE_FILE="" ; CONFIG_ARGS=() ; CONFIG_ENVS=() ; CONFIG_PYTHONPATH=""
  CONFIG_CAP="$CAP" ; CONFIG_NOTE="" ; CONFIG_NOTES=""

  local id="$1"
  case "$id" in
    # ---- group check: the dependency pre-flight ----------------------------
    check_deps)
      CONFIG_GROUP=check ; CONFIG_SIDE=local ; CONFIG_TYPE="-" ; CONFIG_CAP="$CAP_CHECK"
      CONFIG_NOTE="dependency pre-flight: GPU snapshot, legacy .so + siteutils, torch/triton/etl imports, fixtures, driver" ;;

    # ---- group legacy: type 1 on the old CUDA engine -----------------------
    leg_f64|leg_f200)
      CONFIG_GROUP=legacy ; CONFIG_SIDE=leg ; CONFIG_TYPE=1 ; CONFIG_CAP="$CAP_LEGACY"
      if [[ "$id" == leg_f64 ]]; then CONFIG_FIXTURE=fixture64 ; CONFIG_FIXTURE_FILE="$FIXTURE64"
      else CONFIG_FIXTURE=fixture200 ; CONFIG_FIXTURE_FILE="$FIXTURE200" ; fi
      CONFIG_ARGS=(--task eval64 --fixture "$CONFIG_FIXTURE_FILE" --eval-reps "$EVAL_REPS")
      CONFIG_NOTE="type 1: legacy CUDA end-to-end problem.evaluate(forest), $CONFIG_FIXTURE, $EVAL_REPS evaluates (call 0 cold)" ;;

    # ---- group dense: type 1 on the dense-iree ETL evaluator ---------------
    dense_f64|dense_f200)
      CONFIG_GROUP=dense ; CONFIG_SIDE=new ; CONFIG_TYPE=1
      if [[ "$id" == dense_f64 ]]; then CONFIG_FIXTURE=fixture64 ; CONFIG_FIXTURE_FILE="$FIXTURE64"
      else CONFIG_FIXTURE=fixture200 ; CONFIG_FIXTURE_FILE="$FIXTURE200" ; fi
      CONFIG_ARGS=(--task eval64 --fixture "$CONFIG_FIXTURE_FILE" --eval-reps "$EVAL_REPS" --mem-guard-mb 8192)
      CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=1 "${NEW_ENVS[@]}")
      CONFIG_NOTE="type 1: ETL dense-iree end-to-end problem.evaluate (EVOGP_EVAL_MODE=auto => dense), $CONFIG_FIXTURE" ;;

    # ---- group triton: type 1 on the triton fused prototype ----------------
    triton_f64|triton_f200)
      CONFIG_GROUP=triton ; CONFIG_SIDE=triton ; CONFIG_TYPE=1
      if [[ "$id" == triton_f64 ]]; then CONFIG_FIXTURE=fixture64 ; CONFIG_FIXTURE_FILE="$FIXTURE64"
      else CONFIG_FIXTURE=fixture200 ; CONFIG_FIXTURE_FILE="$FIXTURE200" ; fi
      CONFIG_ARGS=(--fixture "$CONFIG_FIXTURE_FILE" --reps "$EVAL_REPS")
      CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=1 "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" "R6_WINDOW=$WINDOW" "R6_CELL_ID=$id")
      [[ -n "$TRITON_CPATH_RESOLVED" ]] && CONFIG_ENVS+=("CPATH=$TRITON_CPATH_RESOLVED")
      CONFIG_NOTE="type 1: triton fused evaluator (dense-seam shadow), $CONFIG_FIXTURE, $EVAL_REPS evaluates; driver CLI pinned by the round brief" ;;

    # ---- group pure: type 2, dense-iree ------------------------------------
    densepure_f64|densepure_f200)
      CONFIG_GROUP=pure ; CONFIG_SIDE=new ; CONFIG_TYPE=2
      if [[ "$id" == densepure_f64 ]]; then CONFIG_FIXTURE=fixture64 ; CONFIG_FIXTURE_FILE="$FIXTURE64"
      else CONFIG_FIXTURE=fixture200 ; CONFIG_FIXTURE_FILE="$FIXTURE200" ; fi
      CONFIG_ARGS=(--task eval-pure --fixture "$CONFIG_FIXTURE_FILE" --mode dense
        --backend iree --reps 5 --warmup 10 --invocations 100
        --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400 --output-checksum)
      CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=1 EVOGP_EVAL_MODE=dense)
      CONFIG_NOTE="type 2: pure dense-iree forward, 100 invocations/block x 5 blocks + 11 e2e companions (r4/run_pure.sh pure64_dense_iree + the r5 window flags)" ;;

    # ---- group pure: type 2, triton ----------------------------------------
    tritonpure_f64|tritonpure_f200)
      CONFIG_GROUP=pure ; CONFIG_SIDE=triton ; CONFIG_TYPE=2
      if [[ "$id" == tritonpure_f64 ]]; then CONFIG_FIXTURE=fixture64 ; CONFIG_FIXTURE_FILE="$FIXTURE64"
      else CONFIG_FIXTURE=fixture200 ; CONFIG_FIXTURE_FILE="$FIXTURE200" ; fi
      CONFIG_ARGS=(--mode pure --fixture "$CONFIG_FIXTURE_FILE" --warmup 10 --invocations 100
        --reps 5 --e2e-companion 11)
      CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=1 "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" "R6_WINDOW=$WINDOW" "R6_CELL_ID=$id")
      [[ -n "$TRITON_CPATH_RESOLVED" ]] && CONFIG_ENVS+=("CPATH=$TRITON_CPATH_RESOLVED")
      CONFIG_NOTE="type 2: triton pure forward, 100 invocations/block x 5 blocks + 11 e2e companions; driver CLI pinned by the round brief" ;;

    *) return 1 ;;
  esac

  # which evogp/etl checkout the cell imports (recorded in notes + log header)
  CONFIG_PYTHONPATH="$NEW_PYTHONPATH"
  [[ "$CONFIG_SIDE" == leg ]] && CONFIG_PYTHONPATH="$LEG_PYTHONPATH"
  # the only harness channel for a non-ETL_* env var is --notes: the window, the
  # cell identity and the effective PYTHONPATH (i.e. which side ran) go there.
  CONFIG_NOTES="r6 same-window campaign: window=$WINDOW cell=$id group=$CONFIG_GROUP side=$CONFIG_SIDE type=$CONFIG_TYPE fixture=${CONFIG_FIXTURE:-none} PYTHONPATH=$CONFIG_PYTHONPATH"
  return 0
}

# NOTE on the per-cell environments: denseness/route/backend are pinned by
# EVOGP_* vars (exported per cell via `env VAR=...` below, never globally) so the
# harness records them in the JSON's `env` block; the legacy side gets no such
# variable; the triton side gets only cache/header plumbing plus the window
# identity (best-effort provenance -- the pinned driver CLI has no --notes).
# NOTE on the type-2 flags: the dense-pure line mirrors r4/run_pure.sh's
# `pure64_dense_iree` cell verbatim (same task/mode/backend/reps/warmup/
# invocations/e2e-companion/mem-guard/max-cold-builds) plus the two flags the r6
# brief pins for this round (`--output-checksum`, and `--notes` for provenance).

# --- 6. preconditions and table self-check ---------------------------------
if [[ ! -d src/evogp || ! -f "$BENCH" ]]; then
  echo "run_cells.sh: must be run from the REPOSITORY ROOT (need src/evogp and $BENCH); cwd=$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "run_cells.sh: interpreter '$PY' is not executable (override with PY=/path/to/python)" >&2
  exit 2
fi
if [[ ! -f "$LOCK_WRAPPER" ]]; then
  echo "run_cells.sh: lock wrapper '$LOCK_WRAPPER' is missing (it is the round's serialization primitive)" >&2
  exit 2
fi
if [[ "$GPU" == "7" || ",$GPU," == *,7,* ]]; then
  echo "run_cells.sh: CUDA_VISIBLE_DEVICES='$GPU' names GPU 7, which has a hardware ECC fault" >&2
  echo "             (any CUDA context on it dies) -- use the default 6" >&2
  exit 2
fi

# advisory capability probe: the bench_gp CLI contract used by the eval cells.
# Warn loudly when the harness on disk does not advertise it, so argparse
# failures are not misread as engine problems.  (Advisory only.)
for flag in --task --eval-reps --mem-guard-mb; do
  if ! grep -q -e "$flag" "$BENCH" 2>/dev/null; then
    echo "run_cells.sh: WARNING: $BENCH does not advertise $flag (the eval64 CLI contract" >&2
    echo "             is missing?); every bench_gp cell would fail with an argparse error." >&2
  fi
done
if ! grep -q -e '--e2e-companion' "$BENCH" 2>/dev/null; then
  echo "run_cells.sh: WARNING: $BENCH does not advertise --e2e-companion (--task eval-pure" >&2
  echo "             missing?); the densepure_* cells would fail with an argparse error." >&2
fi

# table self-check: the id arrays, the parser and the groups must agree BEFORE
# anything runs.  A typo in a table (or in --only) can never run a config the
# rest of this script does not know about.
TABLE_OK=1
_seen=""
for id in "${CANONICAL_IDS[@]}"; do
  if ! cell_spec "$id"; then
    echo "run_cells.sh: cell_spec() cannot derive '$id' (id table / parser out of sync)" >&2
    TABLE_OK=0 ; continue
  fi
  case " $_seen " in
    *" $id "*) echo "run_cells.sh: duplicate id '$id' in the table arrays" >&2 ; TABLE_OK=0 ;;
  esac
  _seen="$_seen $id"
done
if (( ${#CHECK_IDS[@]} + ${#LEGACY_IDS[@]} + ${#DENSE_IDS[@]} + ${#TRITON_IDS[@]} + ${#PURE_IDS[@]} != ${#CANONICAL_IDS[@]} )); then
  echo "run_cells.sh: the group arrays do not partition the canonical id list" >&2 ; TABLE_OK=0
fi
for id in "${ALL_IDS[@]}"; do
  case " ${CHECK_IDS[*]} " in
    *" $id "*) echo "run_cells.sh: '$id' is in ALL_IDS and in CHECK_IDS (--group all must not run the check cell)" >&2 ; TABLE_OK=0 ;;
  esac
done
[[ -f "$TRITON_DRIVER" ]] || echo "run_cells.sh: note: triton driver $TRITON_DRIVER is not present (its cells are guard-skipped until it lands)" >&2
(( TABLE_OK )) || { echo "run_cells.sh: refusing to run with an inconsistent cell table (see above)" >&2; exit 2; }

# selection (--only overrides --group; canonical order)
SELECTED=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_raw <<< "$ONLY"
  _only=()
  for _id in "${_only_raw[@]}"; do
    _id="$(printf '%s' "$_id" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    [[ -z "$_id" ]] && continue
    if ! cell_spec "$_id"; then
      echo "run_cells.sh: unknown cell id '$_id' in --only (see --list)" >&2
      exit 2
    fi
    _only+=("$_id")
  done
  (( ${#_only[@]} > 0 )) || { echo "run_cells.sh: --only is empty" >&2; exit 2; }
  for _id in "${CANONICAL_IDS[@]}"; do
    for _want in "${_only[@]}"; do
      if [[ "$_id" == "$_want" ]]; then SELECTED+=("$_id"); break; fi
    done
  done
else
  case "$GROUP" in
    check)  SELECTED=("${CHECK_IDS[@]}") ;;
    legacy) SELECTED=("${LEGACY_IDS[@]}") ;;
    dense)  SELECTED=("${DENSE_IDS[@]}") ;;
    triton) SELECTED=("${TRITON_IDS[@]}") ;;
    pure)   SELECTED=("${PURE_IDS[@]}") ;;
    all)    SELECTED=("${ALL_IDS[@]}") ;;
  esac
fi
(( ${#SELECTED[@]} > 0 )) || { echo "run_cells.sh: nothing selected" >&2; exit 2; }

# --- 7. helpers -------------------------------------------------------------
gpu_line() {  # one-line GPU snapshot (physical index = the CUDA_VISIBLE_DEVICES value)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -1 | tr -d '\n'
}

gpu_tenants() {  # who else is on the device (co-tenancy evidence for the window record)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    -i "$GPU" 2>/dev/null | tr '\n' ';' | sed 's/;$//'
}

etl_identity_line() {  # which `etl` checkout this PYTHONPATH imports (failure-tolerant)
  local pyp="$1" rev path
  rev="$(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"
  path="$(env "PYTHONPATH=$pyp" "$PY" -c 'import etl; print(etl.__file__)' 2>/dev/null \
          || echo '<import etl failed>')"
  echo "etl: repo=$ETL_REPO rev=$rev file=$path pythonpath=$pyp"
}

evogp_identity_line() {
  local so rev
  rev="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '<no git>')"
  so="$(ls -1 "$LEGACY_ROOT"/src/evogp/evogp_cuda*.so 2>/dev/null | head -1)"
  echo "evogp: repo=$REPO_ROOT rev=$rev new_pythonpath=$NEW_PYTHONPATH legacy_root=$LEGACY_ROOT legacy_so=${so:-<missing>} legacy_pythonpath=$LEG_PYTHONPATH"
}

identity_lines() {  # per-cell identity evidence for the log header
  case "$CONFIG_SIDE" in
    leg)    echo "side: legacy CUDA engine ($LEGACY_ROOT) -- no ETL_*/EVOGP_* variable" ;;
    new)    echo "side: ETL engine ($NEW_PYTHONPATH) -- $(etl_identity_line "$CONFIG_PYTHONPATH")" ;;
    triton) echo "side: triton prototype -- driver=$TRITON_DRIVER cache=$TRITON_CACHE_DIR cpath=${TRITON_CPATH_RESOLVED:-<unset>}" ;;
    local)  echo "side: local dependency pre-flight (no engine)" ;;
  esac
  echo "$(evogp_identity_line)"
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

quote_argv() { local s; s="$(printf '%q ' "$@")"; printf '%s' "${s% }"; }

# guard_reason ID -> prints a skip reason (empty when the cell is runnable)
guard_reason() {
  local id="$1"
  cell_spec "$id" || return 0
  case "$CONFIG_SIDE" in
    leg)
      if [[ ! -d "$LEGACY_ROOT/src" ]]; then
        echo "legacy checkout missing ($LEGACY_ROOT/src): rebuild with test/benchmark/legacy/setup_legacy.sh"
        return 0
      fi
      if [[ ! -d "$LEGACY_SITEUTILS" ]]; then
        echo "legacy offline siteutils missing ($LEGACY_SITEUTILS): rebuild with test/benchmark/legacy/setup_legacy.sh"
        return 0
      fi
      ;;
    triton)
      if [[ ! -f "$TRITON_DRIVER" ]]; then
        echo "triton driver missing ($TRITON_DRIVER): built by the r6 driver owner (CLI is pinned)"
        return 0
      fi
      ;;
  esac
  if [[ -n "$CONFIG_FIXTURE_FILE" && ! -f "$CONFIG_FIXTURE_FILE" ]]; then
    echo "fixture missing ($CONFIG_FIXTURE_FILE; relative to the repo root)"
    return 0
  fi
  return 0
}

# fresh_results_check DIR -> 0 when DIR is absent/empty or holds only the
# pre-flight artifacts (r6 records are immutable evidence: a re-run needs a new
# window label).  Prints the offending entries and returns 1 otherwise.
FRESH_WHITELIST=(check_deps.txt check_deps.log window_meta.txt)
fresh_results_check() {
  local dir="$1"
  if [[ -e "$dir" && ! -d "$dir" ]]; then
    echo "run_cells.sh: result target '$dir' exists and is not a directory" >&2
    return 1
  fi
  [[ -d "$dir" ]] || return 0
  local entry base w ok offenders=()
  while IFS= read -r entry; do
    base="${entry##*/}"
    ok=0
    for w in "${FRESH_WHITELIST[@]}"; do [[ "$base" == "$w" ]] && ok=1; done
    (( ok )) || offenders+=("$entry")
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -print 2>/dev/null | sort)
  if (( ${#offenders[@]} > 0 )); then
    echo "run_cells.sh: refusing to write into the non-empty results dir '$dir'" >&2
    echo "             (fresh window dir per run -- r6 records are immutable evidence):" >&2
    for entry in "${offenders[@]}"; do echo "               $entry" >&2; done
    echo "             use a NEW --window label (or R6_WINDOW=...) for a fresh window" >&2
    return 1
  fi
  return 0
}

# --- 8. dependency pre-flight body (the `check` cell) -----------------------
# Reached only via `--internal-dep-check`, i.e. from a parent run_cells.sh
# invocation that already holds the lock (see run_one).  Prints the report on
# stdout (the parent tees it into results/<window>/check_deps.log) and writes the
# same report to results/<window>/check_deps.txt.  Exit 0 = every required
# dependency present (WARN items are advisory), 1 = at least one MISSING item.
DEP_LINES=() ; DEP_MISSING=() ; DEP_WARN=() ; DEP_DETAIL=()

dep_note() {  # dep_note NAME STATUS DETAIL   (OK|WARN|MISSING)
  local name="$1" status="$2" detail="${3:-}"
  DEP_LINES+=("$(printf '%-22s %-7s %s' "$name" "$status" "$detail")")
  case "$status" in
    OK) ;;
    WARN) DEP_WARN+=("$name ($detail)") ;;
    *)    DEP_MISSING+=("$name ($detail)") ;;
  esac
}

dep_detail() {  # dep_detail NAME BODY  (multiline raw probe output for the report)
  DEP_DETAIL+=("--- $1" "$2")
}

dep_run() {  # dep_run STATUS_ON_FAIL NAME <cmd...>
  local fail_status="$1" name="$2"; shift 2
  local out rc one
  out="$("$@" 2>&1)"; rc=$?
  one="$(printf '%s' "$out" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-220)"
  dep_detail "$name" "${out:-<no output>}"
  if (( rc == 0 )) && [[ -n "$one" ]]; then
    dep_note "$name" OK "$one"
  else
    dep_note "$name" "$fail_status" "rc=$rc out=${one:-<empty>}"
  fi
}

run_dep_check() {
  DEP_LINES=() ; DEP_MISSING=() ; DEP_WARN=() ; DEP_DETAIL=()
  local gpu_probe legacy_import so_list

  dep_note repo-root "$([[ -d src/evogp && -f $BENCH ]] && echo OK || echo MISSING)" \
    "cwd=$REPO_ROOT src/evogp=$([[ -d src/evogp ]] && echo yes || echo NO) bench_gp=$([[ -f $BENCH ]] && echo yes || echo NO)"
  dep_note lock-wrapper "$([[ -f $LOCK_WRAPPER ]] && echo OK || echo MISSING)" \
    "$LOCK_WRAPPER (lock=$WRAPPER_LOCK; shared with r4/r5)"
  dep_note window-dir OK "$RESULT_DIR (fresh gate: whitelist ${FRESH_WHITELIST[*]})"

  # --- interpreter + GPU + python deps (the new side's stack) ---------------
  dep_run MISSING python "$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'
  gpu_probe="$(env "CUDA_VISIBLE_DEVICES=$GPU" "$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))' 2>&1)"
  dep_detail gpu-torch "$gpu_probe"
  if [[ "$gpu_probe" == *"available True"* ]]; then
    dep_note gpu-torch OK "$(printf '%s' "$gpu_probe" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-220)"
  else
    dep_note gpu-torch MISSING "$(printf '%s' "${gpu_probe:-<no output>}" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-220)"
  fi
  dep_run MISSING triton "$PY" -c 'import triton; print("triton", triton.__version__)'

  # --- triton runtime-compile prerequisites --------------------------------
  # CPATH is a search path: Python.h must be found in ONE of its entries, not
  # all of them (the session's header tree has usr/include/python3.11 first).
  if [[ -n "$TRITON_CPATH_RESOLVED" ]]; then
    local hdr_found="" d hdr_status=WARN
    IFS=':' read -r -a _cpath_dirs <<< "$TRITON_CPATH_RESOLVED"
    for d in "${_cpath_dirs[@]}"; do [[ -f "$d/Python.h" ]] && hdr_found="$d/Python.h"; done
    [[ -n "$hdr_found" ]] && hdr_status=OK
    dep_note python-h "$hdr_status" "CPATH=$TRITON_CPATH_RESOLVED (Python.h: ${hdr_found:-not found})"
  else
    dep_note python-h WARN "no CPATH detected (triton runtime compile needs Python.h; set R6_TRITON_CPATH)"
  fi
  if [[ -d "$TRITON_CACHE_DIR" ]]; then
    dep_note triton-cache OK "$TRITON_CACHE_DIR (present)"
  elif mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null; then
    dep_note triton-cache OK "$TRITON_CACHE_DIR (created)"
  else
    dep_note triton-cache WARN "$TRITON_CACHE_DIR not creatable (triton will use its default cache)"
  fi

  # --- legacy side (old CUDA engine) --------------------------------------
  dep_note legacy-root "$([[ -d $LEGACY_ROOT/src ]] && echo OK || echo MISSING)" "$LEGACY_ROOT/src"
  so_list="$(ls -1 "$LEGACY_ROOT"/src/evogp/evogp_cuda*.so 2>/dev/null || true)"
  dep_note legacy-so "$([[ -n $so_list ]] && echo OK || echo MISSING)" "${so_list:-no evogp_cuda*.so under $LEGACY_ROOT/src/evogp (rebuild: legacy/setup_legacy.sh)}"
  dep_note legacy-siteutils "$([[ -d $LEGACY_SITEUTILS && -d $LEGACY_SITEUTILS/sklearn ]] && echo OK || echo MISSING)" "$LEGACY_SITEUTILS"
  legacy_import="$(env "PYTHONPATH=$LEG_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$GPU" "$PY" -c 'import torch; from evogp import evogp_cuda; import evogp.tree.forest as f; print("legacy import ok", evogp_cuda.__file__, f.__file__)' 2>&1)"
  dep_detail legacy-import "$legacy_import"
  if [[ "$legacy_import" == *"legacy import ok"* ]]; then
    dep_note legacy-import OK "$(printf '%s' "$legacy_import" | tr '\n' ' ' | cut -c1-220)"
  else
    dep_note legacy-import MISSING "$(printf '%s' "${legacy_import:-<no output>}" | tail -1 | cut -c1-220)"
  fi

  # --- new side (ETL engine) ----------------------------------------------
  dep_run MISSING etl "$PY" -c 'import etl; print("etl", getattr(etl, "__version__", "?"), etl.__file__)'
  dep_run MISSING evogp-tree env "PYTHONPATH=$NEW_PYTHONPATH" "$PY" -c 'import evogp.tree; print("evogp.tree ok", evogp.tree.__file__)'
  dep_note evogp-commit "$(git -C "$REPO_ROOT" rev-parse --short HEAD >/dev/null 2>&1 && echo OK || echo WARN)" \
    "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"
  dep_note etl-commit "$(git -C "$ETL_REPO" rev-parse --short HEAD >/dev/null 2>&1 && echo OK || echo WARN)" \
    "$(git -C "$ETL_REPO" rev-parse HEAD 2>/dev/null || echo '<git rev-parse failed>')"

  # --- fixtures + driver + gpu device -------------------------------------
  local f
  for f in "$FIXTURE64:fixture64" "$FIXTURE200:fixture200"; do
    local p="${f%%:*}" n="${f##*:}"
    if [[ -f "$p" ]]; then
      dep_note "$n" OK "$p ($(stat -c '%s bytes' "$p" 2>/dev/null || echo 'size?'))"
    else
      dep_note "$n" MISSING "$p (repo-root-relative; generate fixture200 with test/benchmark/make_r2_fixtures.py)"
    fi
  done
  dep_note triton-driver "$([[ -f $TRITON_DRIVER ]] && echo OK || echo MISSING)" "$TRITON_DRIVER"
  dep_note gpu-smi "$(command -v nvidia-smi >/dev/null 2>&1 && echo OK || echo WARN)" \
    "index=$GPU $(gpu_line) tenants=[$(gpu_tenants)]"

  # --- report -------------------------------------------------------------
  local verdict=READY n_ok
  n_ok=$(( ${#DEP_LINES[@]} - ${#DEP_MISSING[@]} - ${#DEP_WARN[@]} ))
  (( ${#DEP_MISSING[@]} > 0 )) && verdict="NOT-READY"
  mkdir -p "$RESULT_DIR"
  {
    echo "# check_deps.txt -- R6 dependency pre-flight (generated by test/benchmark/r6/run_cells.sh --group check)"
    echo "# window:            $WINDOW"
    echo "# result dir:        $RESULT_DIR"
    echo "# invocation:        $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')   host=$(hostname 2>/dev/null || echo '?')"
    echo "# env:               PY=$PY CUDA_VISIBLE_DEVICES=$GPU legacy_pythonpath=$LEG_PYTHONPATH new_pythonpath=$NEW_PYTHONPATH"
    echo "# lock:              $LOCK_WRAPPER (wrapper lock $WRAPPER_LOCK) -- one acquisition per cell, timeout inside"
    echo "# caps:              new/triton=${CAP}s legacy=${CAP_LEGACY}s check=${CAP_CHECK}s"
    echo "# verdict:           $verdict   ($n_ok ok / ${#DEP_WARN[@]} warn / ${#DEP_MISSING[@]} missing)"
    echo
    printf '%s\n' "${DEP_LINES[@]}"
    if (( ${#DEP_MISSING[@]} > 0 )); then
      echo
      echo "## MISSING (the campaign is not runnable until these are fixed)"
      printf '  * %s\n' "${DEP_MISSING[@]}"
    fi
    if (( ${#DEP_WARN[@]} > 0 )); then
      echo
      echo "## WARN (advisory)"
      printf '  * %s\n' "${DEP_WARN[@]}"
    fi
    echo
    echo "## raw probe output"
    printf '%s\n' "${DEP_DETAIL[@]}"
    echo
    echo "## gpu snapshot (all devices, at pre-flight time)"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || echo "<nvidia-smi failed>"
    echo "## gpu compute tenants (all devices)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>&1 || echo "<nvidia-smi failed>"
  } > "$RESULT_DIR/check_deps.txt"

  cat "$RESULT_DIR/check_deps.txt"
  echo
  echo "check_deps: $verdict (${#DEP_LINES[@]} items: $((${#DEP_LINES[@]} - ${#DEP_MISSING[@]} - ${#DEP_WARN[@]})) ok / ${#DEP_WARN[@]} warn / ${#DEP_MISSING[@]} missing) -> $RESULT_DIR/check_deps.txt"
  if (( ${#DEP_MISSING[@]} > 0 )); then
    return 1
  fi
  return 0
}

# the internal path: already inside the lock (the parent wrapped this very
# invocation), so it only runs the checks and reports; nothing else happens.
if (( INTERNAL_DEP_CHECK )); then
  run_dep_check
  exit $?
fi

# --- 9. --list (read-only: never gated on the window dir) -------------------
if (( LIST )); then
  printf '%-17s %-7s %-7s %-4s %-11s %-6s %-34s %s\n' ID GROUP SIDE TYPE FIXTURE CAP PYTHONPATH OUT
  for id in "${SELECTED[@]}"; do
    cell_spec "$id"
    [[ "$CONFIG_FIXTURE" == "" ]] && CONFIG_FIXTURE="-"
    [[ "$CONFIG_TYPE" == "" ]] && CONFIG_TYPE="-"
    printf '%-17s %-7s %-7s %-4s %-11s %-6s %-34s %s\n' "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" \
      "$CONFIG_TYPE" "$CONFIG_FIXTURE" "${CONFIG_CAP}s" "${CONFIG_PYTHONPATH:0:34}" \
      "$RESULT_DIR/$id.$([[ "$CONFIG_SIDE" == local ]] && echo txt || echo json)"
  done
  echo
  echo "-- exact args + per-cell env per cell, and the pre-flight guard state --"
  for id in "${SELECTED[@]}"; do
    cell_spec "$id"
    printf '%-17s %s\n' "$id" "${CONFIG_ARGS[*]:-<built-in: dependency pre-flight>}"
    printf '%-17s env: PYTHONPATH=%s CUDA_VISIBLE_DEVICES=%s %s\n' "" "$CONFIG_PYTHONPATH" "$GPU" "${CONFIG_ENVS[*]:-}"
    [[ -n "$CONFIG_NOTE" ]] && printf '%-17s note: %s\n' "" "$CONFIG_NOTE"
    printf '%-17s lock: %s timeout -s INT -k 30 %ss <cmd>\n' "" "$(basename "$LOCK_WRAPPER")" "$CONFIG_CAP"
    reason="$(guard_reason "$id")"
    [[ -n "$reason" ]] && printf '%-17s GUARD-SKIP: %s\n' "" "$reason"
  done
  echo
  echo "window: $WINDOW   results: $RESULT_DIR   (fresh gate: only ${FRESH_WHITELIST[*]} may pre-exist)"
  echo "triton driver: $TRITON_DRIVER $([[ -f $TRITON_DRIVER ]] && echo '(present)' || echo '(MISSING -> triton cells guard-skip)')"
  exit 0
fi

# --- 10. freshness gate (skipped by --dry-run: it creates nothing) ----------
if (( ! DRY_RUN )); then
  fresh_results_check "$RESULT_DIR" || exit 2
fi

# --- 11. run one cell -------------------------------------------------------
run_one() {
  local id="$1"
  cell_spec "$id" || { echo "run_cells.sh: unknown cell id '$id'" >&2; return 1; }

  local tag json log reason
  tag="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  json="$RESULT_DIR/$id.json"
  log="$RESULT_DIR/$id.log"
  [[ "$CONFIG_SIDE" == local ]] && json="$RESULT_DIR/check_deps.txt"   # the check cell writes a report

  # pre-flight guard: a cell whose input is absent is "skipped by guard"
  # (exit code 3 == OK), never a fake failure and never a silent omission.
  reason="$(guard_reason "$id")"

  local -a inner
  if [[ "$CONFIG_SIDE" == local ]]; then
    # re-enter this script inside the lock; the child writes check_deps.txt and
    # prints the report (teed into check_deps.log by the parent)
    inner=(bash "$R6_DIR/run_cells.sh" --internal-dep-check --window "$WINDOW")
  else
    inner=(env "PYTHONPATH=$CONFIG_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$GPU")
    local e
    for e in "${CONFIG_ENVS[@]}"; do inner+=("$e"); done
    if [[ "$CONFIG_SIDE" == triton ]]; then
      inner+=("$PY" "$TRITON_DRIVER" "${CONFIG_ARGS[@]}" --tag "$tag" --json-out "$json")
    else
      inner+=("$PY" "$BENCH" "${CONFIG_ARGS[@]}" --tag "$tag" --notes "$CONFIG_NOTES" --json-out "$json")
    fi
  fi
  # EXACTLY ONE lock acquisition per cell (never nested), `timeout` INSIDE it,
  # and no shell involved: run_under_lock.py execs this argv directly.
  local -a cmd=("$PY" "$LOCK_WRAPPER" timeout -s INT -k 30 "$CONFIG_CAP" "${inner[@]}")

  # ---- dry run -------------------------------------------------------------
  if (( DRY_RUN )); then
    echo "[$INDEX/$TOTAL] $id  (group=$CONFIG_GROUP side=$CONFIG_SIDE type=$CONFIG_TYPE fixture=${CONFIG_FIXTURE:--} cap=${CONFIG_CAP}s)"
    if [[ -n "$reason" ]]; then
      echo "        GUARD-SKIP: $reason   -> recorded as exit 3 (OK)"
    fi
    echo "        out : $json"
    echo "        log : $log"
    [[ -n "$CONFIG_NOTE" ]] && echo "        note: $CONFIG_NOTE"
    echo "        cmd : $(quote_argv "${cmd[@]}")"
    echo "        (argv form, no shell; timeout -s INT -k 30 ${CONFIG_CAP}s runs INSIDE the lock wrapper on $WRAPPER_LOCK; stdout+stderr -> tee -a $log)"
    return 0
  fi

  if [[ -n "$reason" ]]; then
    echo "cell $id: GUARD-SKIP ($reason)"
    SUMMARY+=("$(printf '%-17s %-7s %-7s %-4s %-6s %-9s %s  # guard: %s' "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" "$CONFIG_TYPE" 3 "SKIP" "$json" "${reason:0:70}")")
    SKIPPED+=("$id")
    DEVIATIONS+=("cell $id GUARD-SKIPPED: $reason (exit 3, counts as OK)")
    return 0
  fi

  mkdir -p "$RESULT_DIR"
  local t_start t_end
  t_start="$(date -Is)"
  {
    echo "==============================================================================="
    echo "R6 window cell $id  (group=$CONFIG_GROUP side=$CONFIG_SIDE type=$CONFIG_TYPE fixture=${CONFIG_FIXTURE:--} cap=${CONFIG_CAP}s)  started $t_start"
    echo "window: $WINDOW   result dir: $RESULT_DIR"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $REPO_ROOT   py: $PY"
    echo "timeout: ${CONFIG_CAP}s (SIGINT, SIGKILL after 30s) INSIDE the lock wrapper ($LOCK_WRAPPER -> fcntl.flock $WRAPPER_LOCK)"
    echo "env: PYTHONPATH=$CONFIG_PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU ${CONFIG_ENVS[*]:-}"
    echo "$(identity_lines)"
    echo "gpu (before): $(gpu_line)"
    echo "gpu tenants (before): $(gpu_tenants)"
    echo "cmd: $(quote_argv "${cmd[@]}")"
    echo "out: $json"
    echo "--- run output ---"
  } >> "$log"

  local rc
  "${cmd[@]}" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  t_end="$(date -Is)"

  {
    echo "--- run output end ---"
    echo "exit code: $rc ($(status_of_rc "$rc"))"
    echo "gpu (after): $(gpu_line)"
    echo "gpu tenants (after): $(gpu_tenants)"
    echo "R6 window cell $id  finished $t_end"
  } >> "$log"

  local status; status="$(status_of_rc "$rc")"
  echo "cell $id: $status (exit $rc) -> $json ; log $log"
  SUMMARY+=("$(printf '%-17s %-7s %-7s %-4s %-6s %-9s %s' "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" "$CONFIG_TYPE" "$rc" "$status" "$json")")
  CELL_ROWS+=("$(printf '%-17s %-7s %-7s %-4s %-11s %-24s %-24s %-5s %-9s %s' \
      "$id" "$CONFIG_GROUP" "$CONFIG_SIDE" "$CONFIG_TYPE" "${CONFIG_FIXTURE:--}" \
      "$t_start" "$t_end" "$rc" "$status" "$json")")
  if (( rc == 0 || rc == 3 )); then
    OK+=("$id")
  else
    FAILED+=("$id")
    DEVIATIONS+=("cell $id exited $rc ($status) -- kept the log ($log), continued the window")
  fi
  return 0
}

# --- 12. main loop ---------------------------------------------------------
SUMMARY=() ; OK=() ; SKIPPED=() ; FAILED=() ; DEVIATIONS=() ; CELL_ROWS=()
TOTAL=${#SELECTED[@]}
INDEX=0

echo "r6/run_cells.sh: py=$PY"
echo "r6/run_cells.sh: repo=$REPO_ROOT  gpu(CUDA_VISIBLE_DEVICES)=$GPU"
echo "r6/run_cells.sh: group=$GROUP only=${ONLY:-<none>} dry_run=$DRY_RUN cells=$TOTAL"
echo "r6/run_cells.sh: window=$WINDOW  results=$RESULT_DIR"
echo "r6/run_cells.sh: caps: new/triton=${CAP}s legacy=${CAP_LEGACY}s check=${CAP_CHECK}s"
echo "r6/run_cells.sh: lock: exactly ONE '$PY $(basename "$LOCK_WRAPPER")' per cell -> fcntl.flock $WRAPPER_LOCK (timeout inside)"
echo "r6/run_cells.sh: legacy=$LEGACY_ROOT (siteutils $LEGACY_SITEUTILS)  new=$NEW_PYTHONPATH"
echo "r6/run_cells.sh: triton driver=$TRITON_DRIVER $([[ -f $TRITON_DRIVER ]] && echo '(present)' || echo '(MISSING -> guard-skip)')  cache=$TRITON_CACHE_DIR cpath=${TRITON_CPATH_RESOLVED:-<unset>}"
echo "r6/run_cells.sh: gpu snapshot: $(gpu_line)"
echo "r6/run_cells.sh: gpu tenants: $(gpu_tenants)"
if (( DRY_RUN )); then
  echo "r6/run_cells.sh: DRY RUN -- nothing is executed, no result dirs are created"
else
  if [[ -e "$RESULT_DIR" ]]; then
    echo "r6/run_cells.sh: results dir $RESULT_DIR exists and passed the fresh gate (only pre-flight artifacts)"
  else
    echo "r6/run_cells.sh: results dir $RESULT_DIR is fresh (will be created)"
  fi
  if [[ "$GROUP" != "check" && ! -f "$RESULT_DIR/check_deps.txt" ]]; then
    echo "r6/run_cells.sh: NOTE: no check_deps.txt in this window yet -- consider running"
    echo "               'bash test/benchmark/r6/run_cells.sh --group check --window $WINDOW' first"
  fi
fi
echo

WINDOW_START="$(date -Is)"
for id in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  run_one "$id"
done
WINDOW_END="$(date -Is)"

# --- 13. summary, window record, disposable exit table ---------------------
echo
echo "=== r6 run_cells.sh summary (group=$GROUP, window=$WINDOW, CUDA_VISIBLE_DEVICES=$GPU, dry_run=$DRY_RUN) ==="
printf '%-17s %-7s %-7s %-4s %-6s %-9s %s\n' ID GROUP SIDE TYPE EXIT STATUS OUT
for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done

if (( DRY_RUN )); then
  echo
  echo "ok: 0 executed, ${#SELECTED[@]} cell(s) planned (dry run; nothing written, no lock taken)"
  exit 0
fi

{
  echo "==============================================================================="
  echo "# window_meta.txt -- R6 same-window campaign record (generated by test/benchmark/r6/run_cells.sh)"
  echo "# committed evidence: immutable once the window is recorded -- a re-run uses a NEW window label"
  echo "# window:     $WINDOW"
  echo "# result dir: $RESULT_DIR"
  echo "# invocation: $WINDOW_START   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')   host=$(hostname 2>/dev/null || echo '?')"
  echo "# args:       group=$GROUP only=${ONLY:-<none>} cells=$TOTAL caps: new/triton=${CAP}s legacy=${CAP_LEGACY}s check=${CAP_CHECK}s"
  echo "# env:        PY=$PY CUDA_VISIBLE_DEVICES=$GPU new_pythonpath=$NEW_PYTHONPATH"
  echo "#             legacy_root=$LEGACY_ROOT legacy_siteutils=$LEGACY_SITEUTILS legacy_pythonpath=$LEG_PYTHONPATH"
  echo "#             etl_repo=$ETL_REPO triton_driver=$TRITON_DRIVER triton_cache=$TRITON_CACHE_DIR cpath=${TRITON_CPATH_RESOLVED:-<unset>}"
  echo "# lock:       $PY $LOCK_WRAPPER (fcntl.flock on $WRAPPER_LOCK) -- exactly ONE acquisition per cell, timeout inside"
  echo "#             never nested; shared with r4/r5 jobs"
  echo "# cell order: table order (see --list); one fresh process per cell, 25 sync-bracketed"
  echo "#             problem.evaluate() calls on the type-1 cells (call 0 cold)"
  echo "# deps:       $RESULT_DIR/check_deps.txt when the check group ran in this window"
  echo "# gpu (start): $(gpu_line)"
  echo "# gpu tenants (start): $(gpu_tenants)"
  echo "# start: $WINDOW_START"
  printf '# %-17s %-7s %-7s %-4s %-11s %-24s %-24s %-5s %-9s %s\n' ID GROUP SIDE TYPE FIXTURE START END EXIT STATUS OUT
  for row in "${CELL_ROWS[@]}"; do printf '# %s\n' "$row"; done
  echo "# end: $WINDOW_END"
  echo "# gpu (end): $(gpu_line)"
  echo "# gpu tenants (end): $(gpu_tenants)"
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
  if (( ${#DEVIATIONS[@]} > 0 )); then
    echo "# deviations:"
    for d in "${DEVIATIONS[@]}"; do echo "#   * $d"; done
  fi
} > "$RESULT_DIR/window_meta.txt"

{
  echo "# $LAST_RUN_FILE -- id -> exit-code table of the last r6 run_cells.sh invocation"
  echo "# generated output: disposable, never committed."
  echo "# invocation: $(date -Is)   cwd=$REPO_ROOT   user=$(id -un 2>/dev/null || echo '?')"
  echo "# args: group=$GROUP only=${ONLY:-<none>} window=$WINDOW cells=$TOTAL"
  echo "# env: CUDA_VISIBLE_DEVICES=$GPU py=$PY results=$RESULT_DIR"
  echo "# exit codes: 0 ok | 3 skipped by guard (counts as ok) | 124/137 timeout | 130 interrupt | 1 crash"
  printf '%-17s %-7s %-7s %-4s %-6s %-9s %s\n' ID GROUP SIDE TYPE EXIT STATUS OUT
  for row in "${SUMMARY[@]}"; do printf '%s\n' "$row"; done
  echo "# ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
} > "$LAST_RUN_FILE"

echo
echo "ok: ${#OK[@]}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"
echo "window record: $RESULT_DIR/window_meta.txt"
echo "exit table:    $LAST_RUN_FILE"
echo "validate with: python3 test/benchmark/r6/check_r6.py"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
