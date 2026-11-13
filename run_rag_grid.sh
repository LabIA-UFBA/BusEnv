#!/bin/bash
# ---------------------------------------------------------------------------------------
# Trains BusEnv once for every combination of retrieval method x algorithm x seed.
# All runs use the same training days (2024) and the same reward weights, so the only
# difference between them is the retrieval method.
#
#   ./run_rag_grid.sh                                    # everything below
#   METHODS="none hybrid rl-chunk ma-chunk" ALGOS="mappo" SEEDS="1 2 3" ./run_rag_grid.sh
#
# METHODS uses the names of --retrieval (see src/rag/methods.py); "none" = original BusEnv.
# Tables must exist first (offline build, see docs/research/llm_rag_rl_busenv/RUNBOOK.md).
# Output: runs_rag/logs/<algo>-<method>-s<seed>.log, runs_rag/metrics, runs_rag/ray_results
# A run whose log already ends with RUN_DONE is skipped, so the script can be restarted.
# ---------------------------------------------------------------------------------------
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=${PY:-/mnt/ssd1/conda/envs/marllib/bin/python}
METHODS=${METHODS:-"none zeros lookahead bm25 dense hybrid rl-chunk ma-chunk random"}
ALGOS=${ALGOS:-"mappo ippo"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
MAX_PARALLEL=${MAX_PARALLEL:-3}      # how many trainings run at the same time
TIMESTEPS=${TIMESTEPS:-200000}
NUM_WORKERS=${NUM_WORKERS:-10}       # Ray workers per training
OUT=${OUT:-$ROOT_DIR/runs_rag}

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export RAG_REWARD_WEIGHTS=${RAG_REWARD_WEIGHTS:-'{"occ_penalty":0.6,"uptime_bonus":0.4,"sync_score":0.5,"energy_efficiency":0.5}'}
export SUNT_METRICS_DIR=$OUT/metrics
export RAY_RESULTS_DIR=$OUT/ray_results
export TMPDIR=${TMPDIR:-/mnt/ssd1/tmp}
mkdir -p "$OUT/logs" "$OUT/metrics" "$OUT/ray_results" "$OUT/codecarbon"

run_one() {
  local algo=$1 method=$2 seed=$3
  local tag="${algo}-${method}-s${seed}"
  local log="$OUT/logs/${tag}.log"
  if grep -q "RUN_DONE" "$log" 2>/dev/null; then echo "skip $tag (already done)"; return; fi
  echo "start $tag"
  RAY_TMPDIR="$TMPDIR/ray_${tag}" \
    "$PY" src/pipelines/train_marllib.py --algo "$algo" --retrieval "$method" --day-split train \
      --seed "$seed" --timesteps "$TIMESTEPS" --num-workers "$NUM_WORKERS" \
      --cc-run-id "$tag" --cc-output-dir "$OUT/codecarbon" > "$log" 2>&1 \
    && echo "RUN_DONE" >> "$log"
  echo "end $tag (exit $?)"
}

jobs_list=()
for seed in $SEEDS; do for algo in $ALGOS; do for method in $METHODS; do jobs_list+=("$algo $method $seed"); done; done; done
echo "${#jobs_list[@]} runs, $MAX_PARALLEL at a time"
for j in "${jobs_list[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 10; done
  run_one $j &
done
wait
echo "all runs finished"
