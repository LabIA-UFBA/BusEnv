#!/bin/bash
# -------------------------
# Nohup launcher for MARLlib training
# -------------------------

# ✅ All available MARLlib algorithms
algos=("iql" "ipg" "ia2c" "iddpg" "itrpo" "ippo" \
       "maa2c" "coma" "maddpg" "matrpo" "mappo" \
       "hatrpo" "happo" "vdn" "qmix" "facmac" \
       "vda2c" "vdppo")

# 🔄 Number of runs per algorithm
runs_per_algo=50

# Python entrypoint (your parametrized trainer)
train_script="./src/pipelines/train_marllib.py"

# Logs directory
logdir="logs"
mkdir -p "$logdir"

# -------------------------
# Main loop
# -------------------------
for algo in "${algos[@]}"; do
  for run in $(seq 1 $runs_per_algo); do
    run_id="run${run}"
    log_file="${logdir}/${algo}_${run_id}.log"

    echo "Launching: $algo (Run $run) -> $log_file"

    nohup python "$train_script" \
      --algo "$algo" \
      --cc-run-id "$algo-$run_id" \
      > "$log_file" 2>&1 &

    # Small delay so jobs don’t all start at the same time
    sleep 2
  done
done

echo "✅ All jobs launched in background. Logs available in $logdir/"
