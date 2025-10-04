#!/bin/bash
# ---------------------------------------------------------
# Nohup launcher para MARLlib (CodeCarbon ligado por padrão)
# Estrutura esperada:
#   ./run_all.sh
#   ./src/pipelines/train_marllib.py
#   ./codecarbon  (será criado se não existir)
#   ./logs        (será criado se não existir)
# ---------------------------------------------------------

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

# Pastas de saída
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
    run_id="run${run}"
    log_file="${LOG_DIR}/${algo}_${run_id}.log"

    echo "Lançando: ${algo} (Run ${run}) -> ${log_file}"

    nohup python "${TRAIN_SCRIPT}" \
      --algo "${algo}" \
      --cc-run-id "${algo}-${run_id}" \
      --cc-output-dir "${CC_OUTDIR}" \
      > "${log_file}" 2>&1 &

    # Pequeno atraso para não iniciar tudo no mesmo instante
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
