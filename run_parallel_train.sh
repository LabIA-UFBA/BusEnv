#!/bin/bash
# ---------------------------------------------------------
# Nohup launcher para MARLlib (CodeCarbon ligado por padrão)
# Estrutura esperada:
#   ./run_all.sh
#   ./src/pipelines/train_marllib.py
#   ./codecarbon  (será criado se não existir)
#   ./logs        (será criado se não existir)
# ---------------------------------------------------------

# Modify if necessary before running; if you do not wish to select an output folder, you can comment out this line.
export RAY_TMPDIR=/mnt/ssd1/ray_tmp
export TMPDIR=/mnt/ssd1/tmp

# Descobre a raiz do projeto (pasta onde está este script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Garante que o Python encontre os módulos em ./src
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH}"
export PYTHONPATH=$(pwd):$PYTHONPATH
cd MARLlib
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

#Caminho do trainer parametrizado
TRAIN_SCRIPT="${ROOT_DIR}/src/pipelines/train_marllib.py"

# ✅ Lista com TODOS os algoritmos (edite à vontade)
# algos=("iql" "ipg" "ia2c" "iddpg" "itrpo" "ippo" \
#        "maa2c" "coma" "maddpg" "matrpo" "mappo" \
#        "hatrpo" "happo" "vdn" "qmix" "facmac" \
#        "vda2c" "vdppo")

algos=("mappo" "hatrpo" "itrpo")

# Funcionaram 100%
# IA2C 

# Joint q learning does not support individual function
# Iql, qmix, vdn

# Saíram pois precisam de ações continuas
# facmac, iddpg, maddpg

# Não está na lib
# "ipg"

# 🔄 Número de execuções por algoritmo (edite à vontade) 
runs_per_algo=5

# Pastas de saída ALTERAR CASO NECESSARIO ANTES DE RODAR
LOG_DIR="${ROOT_DIR}/logs"
CC_OUTDIR="${ROOT_DIR}/codecarbon"

mkdir -p "${LOG_DIR}"
mkdir -p "${CC_OUTDIR}"   # CodeCarbon exige que exista

# -------------------------
# Loop principal
# -------------------------
for run in $(seq 1 "${runs_per_algo}"); do
  echo "🚀 Iniciando rodada ${run} de ${runs_per_algo}..."
  
  # Executa todos os algoritmos em paralelo nesta rodada
  for algo in "${algos[@]}"; do

      # Encontrar o próximo número disponível de log para este algoritmo
      next_run=1
      while [[ -f "${LOG_DIR}/${algo}_run${next_run}.log" ]]; do
          next_run=$((next_run + 1))
      done

      run_id="run${next_run}"
      log_file="${LOG_DIR}/${algo}_${run_id}.log"

      METRICS_DIR="${ROOT_DIR}/metrics"
      mkdir -p "${METRICS_DIR}"
      export SUNT_METRICS_FILE="${METRICS_DIR}/episode_metrics_${algo}_${run_id}.csv"

      echo "Lançando: ${algo} -> ${log_file} | metrics -> ${SUNT_METRICS_FILE}"

      nohup python "${TRAIN_SCRIPT}" \
        --algo "${algo}" \
        --cc-run-id "${algo}-${run_id}" \
        --cc-output-dir "${CC_OUTDIR}" \
        > "${log_file}" 2>&1 &

      sleep 2
  done


  # Espera todos os algoritmos desta rodada terminarem antes da próxima
  echo "⏳ Aguardando finalização da rodada ${run}..."
  wait
  echo "✅ Rodada ${run} concluída."
done

echo "🏁 Todas as ${runs_per_algo} rodadas foram executadas com sucesso!"
echo "📄 Logs: ${LOG_DIR}"
echo "🌱 CodeCarbon CSV: ${CC_OUTDIR}/emissions.csv (por padrão)"
