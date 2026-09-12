set -u
ulimit -n 65536
S=test/benchmark/r4/results/smoke
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
run() { echo "### $*"; "$@"; echo "### EXIT=$?"; }
run env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 $PY test/benchmark/bench_gp.py --task eval64 \
  --fixture test/goldens/forest_random.npz --backend numpy --reps 1 --eval-reps 3 \
  --tag SMOKE_EVAL64_NUMPY --json-out $S/eval64_regression.json
run env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 $PY test/benchmark/bench_gp.py --task eval64 \
  --fixture test/goldens/forest_random.npz --backend iree --reps 1 --eval-reps 5 \
  --max-cold-builds 400 --tag SMOKE_EVAL64_IREE --json-out $S/eval64_iree_crosscheck.json
