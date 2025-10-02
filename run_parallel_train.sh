#!/bin/bash
# ---------------------------------------------------------
# Nohup launcher para MARLlib (CodeCarbon ligado por padrão)
# Estrutura esperada:
#   ./run_all.sh
#   ./src/pipelines/train_marllib.py
#   ./codecarbon  (será criado se não existir)
#   ./logs        (será criado se não existir)
# ---------------------------------------------------------

set -euo pipefail

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

# ✅ Lista com TODOS os algoritmos (edite à vontade)
# algos=("iql" "ipg" "ia2c" "iddpg" "itrpo" "ippo" \
#        "maa2c" "coma" "maddpg" "matrpo" "mappo" \
#        "hatrpo" "happo" "vdn" "qmix" "facmac" \
#        "vda2c" "vdppo")

algos=( "itrpo" "ippo"
       "maa2c" "coma" "matrpo" "mappo"
       "hatrpo" "happo")

# Funcionaram 100%
# IA2C 

# Joint q learning does not support individual function
# Iql, qmix, vdn

# Saíram pois precisam de ações continuas
# facmac, iddpg, maddpg

# Não está na lib
# "ipg"

# 🔄 Número de execuções por algoritmo (edite à vontade)
runs_per_algo=50

# Limite de quantos algoritmos rodam em paralelo por rodada
# (para não estourar memória — ex: 2, 3, etc. 0 = todos juntos)
PARALLEL_LIMIT=0

# Pastas de saída
LOG_DIR="${ROOT_DIR}/logs"
CC_OUTDIR="${ROOT_DIR}/codecarbon"

mkdir -p "${LOG_DIR}"
mkdir -p "${CC_OUTDIR}"   # CodeCarbon exige que exista

# ---------------------------------------------------------
# Função auxiliar para checar memória (opcional)
# ---------------------------------------------------------
get_free_mb() {
  awk '/MemAvailable:/ { printf "%.0f", $2/1024 }' /proc/meminfo
}

wait_for_memory() {
  local min_free_mb="${1:-0}"
  if [[ "${min_free_mb}" -le 0 ]]; then return; fi
  while true; do
    local free_mb
    free_mb=$(get_free_mb)
    if [[ "${free_mb}" -ge "${min_free_mb}" ]]; then break; fi
    echo "[MEM] Aguardando memória livre: ${free_mb}MB (< ${min_free_mb}MB)"
    sleep 5
  done
}

# ---------------------------------------------------------
# Função que lança 1 job e devolve o PID
# ---------------------------------------------------------
launch_job() {
  local algo="$1"
  local run="$2"
  local run_id="run${run}"
  local log_file="${LOG_DIR}/${algo}_${run_id}.log"

  echo "Lançando: ${algo} (Rodada ${run}) -> ${log_file}"

  nohup python "${TRAIN_SCRIPT}" \
    --algo "${algo}" \
    --cc-run-id "${algo}-${run_id}" \
    --cc-output-dir "${CC_OUTDIR}" \
    > "${log_file}" 2>&1 &
  echo $!
}

# ---------------------------------------------------------
# Loop principal (rodada por rodada)
# ---------------------------------------------------------
total_algos=${#algos[@]}
if [[ -z "${PARALLEL_LIMIT}" || "${PARALLEL_LIMIT}" -le 0 || "${PARALLEL_LIMIT}" -gt "${total_algos}" ]]; then
  PARALLEL_LIMIT="${total_algos}"
fi

echo "=== Iniciando ${runs_per_algo} rodadas ==="
echo "Algoritmos: ${algos[*]}"
echo "Paralelismo máximo: ${PARALLEL_LIMIT}"

for run in $(seq 1 "${runs_per_algo}"); do
  echo "---------------------------------------------"
  echo "▶️  Rodada ${run}/${runs_per_algo}"

  for ((i=0; i<total_algos; i+=PARALLEL_LIMIT)); do
    chunk=( "${algos[@]:i:PARALLEL_LIMIT}" )
    echo "  • Sub-lote: ${chunk[*]}"

    # lança o sub-lote
    pids=()
    for algo in "${chunk[@]}"; do
      pid=$(launch_job "${algo}" "${run}")
      pids+=( "$pid" )
      sleep 1
    done

    # espera terminar o sub-lote
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done

  echo "✅ Rodada ${run} concluída."
done

echo "🎉 Todas as ${runs_per_algo} rodadas finalizadas."
echo "📄 Logs: ${LOG_DIR}"
echo "🌱 CodeCarbon CSV: ${CC_OUTDIR}/emissions.csv (por padrão)"
