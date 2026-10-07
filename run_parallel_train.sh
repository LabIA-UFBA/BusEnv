#!/bin/bash
# ---------------------------------------------------------
# Nohup launcher for MARLlib (CodeCarbon enabled by default)
# Expected structure:
#   ./run_all.sh
#   ./src/pipelines/train_marllib.py
#   ./codecarbon  (created if it doesn't exist)
#   ./logs        (created if it doesn't exist)
#
# CHANGE (ALGORITHM-based queue): there are fixed MAX_PARALLEL
# "slots" (workers). Each worker picks the NEXT algorithm not yet
# started from the `algos` list, runs its `runs_per_algo` executions
# SEQUENTIALLY (one at a time), and only then picks up the next
# available algorithm from the list. Thus, with 5 algorithms and
# MAX_PARALLEL=3: the first 3 start immediately, each occupying
# a slot; as soon as any one finishes its 5 runs, the slot is
# freed and the worker automatically picks up the 4th (then the 5th)
# — without needing to edit `algos` and run the script again.
# ---------------------------------------------------------

# Modify if necessary before running; if you do not wish to select an output folder, you can comment out this line.
export RAY_TMPDIR=/mnt/ssd1/ray_tmp # ADD YOUR PATH
export TMPDIR=/mnt/ssd1/tmp # ADD YOUR PATH

# Descobre a raiz do projeto (pasta onde está este script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Garante que o Python encontre os módulos em ./src
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH}"
export PYTHONPATH=$(pwd):$PYTHONPATH
cd MARLlib
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

# Caminho do trainer parametrizado
TRAIN_SCRIPT="${ROOT_DIR}/src/pipelines/train_marllib.py"

# ✅ Lista com TODOS os algoritmos (edite à vontade — pode ter mais
# algoritmos que MAX_PARALLEL sem problema, eles entram na fila)
# algos=("iql" "ipg" "ia2c" "iddpg" "itrpo" "ippo" \
#        "maa2c" "coma" "maddpg" "matrpo" "mappo" \
#        "hatrpo" "happo" "vdn" "qmix" "facmac" \
#        "vda2c" "vdppo")

algos=("coma" "happo" "hatrpo" "ippo" "itrpo" "matrpo" "ia2c" "maa2c" "mappo")
# algos=("itrpo")
# Funcionaram 100%
# IA2C

# Joint q learning does not support individual function
# Iql, qmix, vdn

# Saíram pois precisam de ações continuas
# facmac, iddpg, maddpg

# Não está na lib
# "ipg"

# 🔄 Número de execuções por algoritmo (edite à vontade)
runs_per_algo=10

# 🚦 Quantos algoritmos podem rodar AO MESMO TEMPO (vagas fixas).
# Se algos tiver mais itens que isso, o excedente entra na fila e
# só começa quando uma vaga libera.
MAX_PARALLEL=3

# Pastas de saída ALTERAR CASO NECESSARIO ANTES DE RODAR
LOG_DIR="${ROOT_DIR}/logs"
CC_OUTDIR="${ROOT_DIR}/codecarbon"
METRICS_DIR="${ROOT_DIR}/metrics"

mkdir -p "${LOG_DIR}"
mkdir -p "${CC_OUTDIR}"   # CodeCarbon exige que exista
mkdir -p "${METRICS_DIR}"

# -------------------------------------------------------------
# Fila dinâmica: um contador compartilhado + lock (flock) garante
# que dois workers nunca peguem o mesmo índice de `algos`, mesmo
# lançando tudo em paralelo.
# -------------------------------------------------------------
QUEUE_LOCK="${ROOT_DIR}/.run_all_queue.lock"
QUEUE_IDX_FILE="${ROOT_DIR}/.run_all_queue_idx"
echo 0 > "${QUEUE_IDX_FILE}"

cleanup_queue_files() {
  rm -f "${QUEUE_LOCK}" "${QUEUE_IDX_FILE}"
}
trap cleanup_queue_files EXIT

# Pega o próximo índice disponível de `algos` de forma atômica.
# Retorna um número >= len(algos) quando a fila acabou.
next_algo_idx() {
  local idx
  {
    flock -x 200
    idx=$(cat "${QUEUE_IDX_FILE}")
    echo $((idx + 1)) > "${QUEUE_IDX_FILE}"
  } 200>"${QUEUE_LOCK}"
  echo "${idx}"
}

# -------------------------------------------------------------
# Worker: fica em loop pegando o próximo algoritmo da fila e
# rodando TODAS as suas runs em sequência antes de pedir o próximo.
# -------------------------------------------------------------
worker() {
  local worker_id="$1"

  while true; do
    local idx
    idx="$(next_algo_idx)"

    if (( idx >= ${#algos[@]} )); then
      echo "🧵 [worker ${worker_id}] fila esgotada, encerrando."
      break
    fi

    local algo="${algos[$idx]}"
    echo "🧵 [worker ${worker_id}] pegou algoritmo '${algo}' (posição ${idx}) — ${runs_per_algo} runs em sequência"

    for run in $(seq 1 "${runs_per_algo}"); do
      # Próximo run_id disponível para ESTE algoritmo (não sobrescreve logs antigos)
      local n=1
      while [[ -f "${LOG_DIR}/${algo}_run${n}.log" ]]; do
        n=$((n + 1))
      done
      local run_id="run${n}"
      local log_file="${LOG_DIR}/${algo}_${run_id}.log"
      local metrics_file="${METRICS_DIR}/episode_metrics_${algo}_${run_id}.csv"

      export SUNT_METRICS_FILE="${metrics_file}"

      echo "🚀 [worker ${worker_id}] lançando: ${algo} (${run_id}) -> ${log_file}"

      # Sem '&' no final: roda em primeiro plano DENTRO do worker,
      # ou seja, espera terminar antes de ir pra próxima run/algoritmo.
      # nohup mantém a imunidade a SIGHUP mesmo rodando em foreground.
      nohup python "${TRAIN_SCRIPT}" \
        --algo "${algo}" \
        --cc-run-id "${algo}-${run_id}" \
        --cc-output-dir "${CC_OUTDIR}" \
        > "${log_file}" 2>&1

      echo "✅ [worker ${worker_id}] ${algo} (${run_id}) concluído"
    done

    echo "🏁 [worker ${worker_id}] terminou TODAS as ${runs_per_algo} runs de '${algo}'. Buscando próximo algoritmo..."
  done
}

echo "📋 Fila: ${#algos[@]} algoritmo(s) | ${runs_per_algo} runs cada | ${MAX_PARALLEL} vaga(s) simultânea(s)"

# Lança as MAX_PARALLEL vagas fixas. Cada worker consome a fila
# sozinho até ela acabar.
for w in $(seq 1 "${MAX_PARALLEL}"); do
  worker "${w}" &
  sleep 2
done

# Espera todos os workers terminarem (ou seja, a fila inteira ser consumida)
wait

echo "🏁 Todos os algoritmos (${#algos[@]}) x ${runs_per_algo} runs foram concluídos!"
echo "📄 Logs: ${LOG_DIR}"
echo "🌱 CodeCarbon CSV: ${CC_OUTDIR}/emissions.csv (por padrão)"