#!/usr/bin/env bash
# test/benchmark/new/run_matrix.sh -- new-side (ETL "true jit" engine) config-matrix runner.
#
# Runs the SPEC.md config matrix through test/benchmark/bench_gp.py and keeps one
# JSON + one raw log per config under test/benchmark/new/results/.
#
# Usage (from the REPOSITORY ROOT -- the checkout that owns test/benchmark/):
#
#     ./run_matrix.sh                 # all configs, in SPEC priority order
#     ./run_matrix.sh C1 C5           # only these configs
#
# The caller must NOT wrap this script in its own flock: every run below is
# serialized internally with `flock -w 7200 /tmp/gpbench_critical.lock` (the same
# lock the legacy-side agent uses), and the `timeout` sits INSIDE the lock so only
# the benchmark itself is bounded, never the wait for the lock.
#
# Environment:
#   PY                    interpreter (default: the project venv below)
#   CUDA_VISIBLE_DEVICES  if preset, it is used as-is (the run then uses its first
#                         visible device); otherwise a GPU is picked from
#                         `nvidia-smi` (prefers "A6000", then least utilization,
#                         then least memory; cards with uncorrectable ECC errors
#                         are excluded, see pick_gpu()).  EVOGP_ETL_IREE_DEVICE=0
#                         always matches "cuda" -> first visible device (C7).
#
# Exit code: 0 if every selected config succeeded, 1 if any failed (the batch never
# aborts early -- a failing config is logged and the next one runs).
set -u

PY="${PY:-/mnt/local-ssd/bchuang/evox/.venv/bin/python}"
LOCK=/tmp/gpbench_critical.lock
RESULT_DIR=test/benchmark/new/results
BENCH=test/benchmark/bench_gp.py
ALL_CONFIGS=(C1 C5 C2b C4 C2 C0 C5b C7)

# --- sanity: repo root? -----------------------------------------------------
if [[ ! -f "$BENCH" || ! -d src/evogp ]]; then
  echo "run_matrix.sh: must be run from the repository root (need $BENCH and src/evogp); cwd=$(pwd)" >&2
  exit 2
fi

# --- config table (exact SPEC.md parameters) --------------------------------
CONFIG_TIMEOUT=0
CONFIG_ARGS=()
CONFIG_ENVS=()
CONFIG_NOTE=""

config_spec() {
  CONFIG_ARGS=()
  CONFIG_ENVS=()
  CONFIG_NOTE=""
  case "$1" in
    C1)  CONFIG_TIMEOUT=1500; CONFIG_ARGS=(--task sr --pop 200 --gens 100 --tree-len 64 --tag C1 --reps 2) ;;
    C5)  CONFIG_TIMEOUT=900;  CONFIG_ARGS=(--task eval64 --eval-reps 25 --tag C5) ;;
    C2b) CONFIG_TIMEOUT=2700; CONFIG_ARGS=(--task sr --pop 1000 --gens 20 --tree-len 128 --tag C2b) ;;
    C4)  CONFIG_TIMEOUT=1500; CONFIG_ARGS=(--task xor --pop 50 --gens 100 --tag C4 --reps 2) ;;
    C2)  CONFIG_TIMEOUT=3600; CONFIG_ARGS=(--task sr --pop 1000 --gens 100 --tree-len 128 --tag C2 --max-wall-s 1800 --min-gens 20) ;;
    C0)  CONFIG_TIMEOUT=2700; CONFIG_ARGS=(--task sr --pop 200 --gens 100 --tree-len 64 --tag C0 --no-disable-locations)
                              CONFIG_ENVS=(ETL_DISABLE_LOCATIONS=0)
                              CONFIG_NOTE="control run: ETL call-site capture left ON (ETL_DISABLE_LOCATIONS=0)" ;;
    C5b) CONFIG_TIMEOUT=2700; CONFIG_ARGS=(--task evalfresh --pop 1000 --tree-len 64 --eval-reps 10 --tag C5b) ;;
    C7)  CONFIG_TIMEOUT=960;  CONFIG_ARGS=(--task sr --pop 50 --gens 3 --tree-len 32 --tag C7 --max-wall-s 900 --notes "opt-in IREE backend, cap 15min")
                              CONFIG_ENVS=(EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1)
                              CONFIG_NOTE="opt-in IREE backend (EVOGP_ETL_BACKEND=iree, O1), cap 15 min" ;;
    *) return 1 ;;
  esac
  return 0
}

# --- GPU pick ----------------------------------------------------------------
# Prints either "preset:<CUDA_VISIBLE_DEVICES>" or the chosen physical index,
# or nothing when nvidia-smi cannot be parsed.
# Rule (SPEC.md): prefer "A6000" in the name, then least utilization, then least
# memory.  One added health guard, logged loudly: a card reporting a nonzero
# `ecc.errors.uncorrected.volatile.total` is excluded ([N/A] = no counter = ok) --
# gpu 7 on this host has a permanent uncorrectable-ECC fault and any CUDA job on it
# dies with "uncorrectable ECC error encountered".
pick_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "preset:${CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  local idx name util mem ecc key best_key="" pick=""
  while IFS=',' read -r idx name util mem ecc; do
    [[ -z "${idx// /}" ]] && continue
    idx="${idx// /}"
    name="${name#"${name%%[![:space:]]*}"}"
    name="${name%"${name##*[![:space:]]}"}"
    util="${util// /}"
    mem="${mem// /}"
    ecc="${ecc// /}"
    case "$util" in (*[!0-9]*|"") util=999999 ;; esac
    case "$mem"  in (*[!0-9]*|"") mem=999999999 ;; esac
    local pref=1
    [[ "$name" == *A6000* ]] && pref=0
    case "$ecc" in
      ([!0-9]*|"") ;;  # "[N/A]"/empty: counter not reported -> healthy as far as we know
      (*) if (( ecc > 0 )); then
            pref=2
            echo "GPU: EXCLUDING index ${idx} (${name}): ${ecc} uncorrectable volatile ECC errors" >&2
          fi ;;
    esac
    printf -v key '%d %09d %010d' "$pref" "$util" "$mem"
    if [[ -z "$best_key" || "$key" < "$best_key" ]]; then
      best_key="$key"
      pick="$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null)
  echo "$pick"
}

PICK=$(pick_gpu)
if [[ "$PICK" == preset:* ]]; then
  VISIBLE="${PICK#preset:}"
  echo "GPU: CUDA_VISIBLE_DEVICES already preset -> keeping it (${VISIBLE}); the run uses its first visible device."
elif [[ -n "$PICK" ]]; then
  VISIBLE="$PICK"
  echo "GPU: picked physical index ${VISIBLE} (A6000-preferred, least utilization, then least memory)."
else
  VISIBLE=0
  echo "GPU: WARNING could not parse 'nvidia-smi --query-gpu=...' -> FALLING BACK to index 0." >&2
fi
export CUDA_VISIBLE_DEVICES="$VISIBLE"

mkdir -p "$RESULT_DIR"
echo "run_matrix.sh: PY=$PY lock=$LOCK results=$RESULT_DIR configs=[$*] CUDA_VISIBLE_DEVICES=$VISIBLE"

# --- per-config runner -------------------------------------------------------
FAILED=()
OK=()
SUMMARY=()

run_one() {
  local id="$1"
  if ! config_spec "$id"; then
    echo "run_matrix.sh: unknown config id '$id' (known: ${ALL_CONFIGS[*]})" >&2
    FAILED+=("$id")
    SUMMARY+=("$id|UNKNOWN|unknown config id|-")
    return 1
  fi

  local log="$RESULT_DIR/${id}.log"
  local json="$RESULT_DIR/${id}.json"

  # inner command string: `timeout` INSIDE the lock, then the env, then the harness.
  local inner="timeout -s INT -k 30 ${CONFIG_TIMEOUT} env PYTHONPATH=src"
  inner+=" $(printf '%q' "CUDA_VISIBLE_DEVICES=$VISIBLE")"
  if (( ${#CONFIG_ENVS[@]} > 0 )); then
    local e
    for e in "${CONFIG_ENVS[@]}"; do
      inner+=" $(printf '%q' "$e")"
    done
  fi
  inner+=" $(printf '%q' "$PY") $(printf '%q' "$BENCH")"
  local a
  for a in "${CONFIG_ARGS[@]}"; do
    inner+=" $(printf '%q' "$a")"
  done
  inner+=" --json-out $(printf '%q' "$json")"

  {
    echo "==================================================================="
    echo "config $id  started $(date -Is)"
    echo "note: ${CONFIG_NOTE:-none}"
    echo "cwd: $(pwd)"
    echo "PY: $PY"
    echo "timeout: ${CONFIG_TIMEOUT}s (SIGINT first, SIGKILL after 30s), inside flock $LOCK"
    echo "env: PYTHONPATH=src CUDA_VISIBLE_DEVICES=$VISIBLE ${CONFIG_ENVS[*]:-}"
    echo "flock wrapper: flock -w 7200 $LOCK bash -c '<cmd>'"
    echo "cmd: $inner"
    echo "--- nvidia-smi (before) ---"
    nvidia-smi
    echo "--- run output ---"
  } > "$log"

  flock -w 7200 "$LOCK" bash -c "$inner" >> "$log" 2>&1
  local rc=$?

  {
    echo "--- run output end ---"
    echo "exit code: $rc"
    echo "config $id  finished $(date -Is)"
    echo "--- nvidia-smi (after) ---"
    nvidia-smi
    echo "json: $json"
  } >> "$log"

  if (( rc == 0 )); then
    OK+=("$id")
    echo "config $id: OK (exit 0) -> $json ; log $log"
  else
    FAILED+=("$id")
    echo "config $id: FAILED (exit $rc) -> log $log"
  fi
  SUMMARY+=("$id|$([[ $rc -eq 0 ]] && echo OK || echo FAIL)|exit=$rc|$log")
  return 0
}

SELECTED=("$@")
if (( ${#SELECTED[@]} == 0 )); then
  SELECTED=("${ALL_CONFIGS[@]}")
fi

for id in "${SELECTED[@]}"; do
  run_one "$id"
done

echo
echo "=== run_matrix.sh summary (CUDA_VISIBLE_DEVICES=$VISIBLE) ==="
printf '%-6s %-6s %-8s %s\n' CONFIG STATUS EXIT LOG
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r id status rc log <<< "$row"
  printf '%-6s %-6s %-8s %s\n' "$id" "$status" "$rc" "$log"
done
echo "ok: ${#OK[@]}  failed: ${#FAILED[@]}${FAILED[*]:+ (${FAILED[*]})}"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
