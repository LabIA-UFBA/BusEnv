import os
import pandas as pd
import pickle
import re
import sys
import pyarrow.parquet as pq
from datetime import timedelta

def process_date_folder(date_str: str, base_data_path: str) -> dict:
    """
    Processa os arquivos Parquet de OD (Origem-Destino) para uma dada data,
    calculando o tempo de viagem entre pares de paradas.
    """
    
    file_path = os.path.join(base_data_path, "OD", f"od-{date_str}.parquet")

    if not os.path.exists(file_path):
        print(f"❌ Arquivo não encontrado para a data: {file_path}")
        return None

    try:
        df_od = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo {file_path}: {e}")
        return None

    # Convertendo colunas de tempo para datetime se ainda não estiverem
    df_od['stop_time'] = pd.to_datetime(df_od['stop_time'])

    time_per_stop_pair = {}

    print(f"📆 Processando {date_str}...")

    # Agrupando por viagem para calcular o tempo entre paradas sequenciais
    # Uma viagem é definida por vehicle, trip_id e direction_id
    grouped_trips = df_od.sort_values(by=['vehicle', 'trip_id', 'direction_id', 'pt_sequence']).groupby(['vehicle', 'trip_id', 'direction_id'])

    progress_counter = 0
    total_rows_to_process = len(df_od)

    for name, group in grouped_trips:
        # Pula se o grupo tiver apenas uma parada, pois não há tempo entre paradas
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            prev_row = group.iloc[i-1]
            current_row = group.iloc[i]

            # Garante que estamos na mesma viagem e direção
            if prev_row['trip_id'] == current_row['trip_id'] and \
               prev_row['direction_id'] == current_row['direction_id']:

                # Calcular a diferença de tempo
                time_diff = (current_row['stop_time'] - prev_row['stop_time']).total_seconds()

                # Apenas considerar tempos positivos (viagens em progresso)
                if time_diff > 0:
                    stop_pair = (prev_row['stop_id'], current_row['stop_id'])
                    if stop_pair not in time_per_stop_pair:
                        time_per_stop_pair[stop_pair] = [0.0, 0] # [total_time, count]

                    time_per_stop_pair[stop_pair][0] += time_diff
                    time_per_stop_pair[stop_pair][1] += 1
            
            progress_counter += 1
            if progress_counter % 10000 == 0: # Atualiza o progresso a cada 10.000 linhas
                print(f"\r⏳ {progress_counter/total_rows_to_process*100:.2f}% das linhas de {date_str}", end='')

    print(f"\r✅ 100.00% das linhas de {date_str} processadas.") # Mensagem final de progresso

    return time_per_stop_pair


### **Fluxo Principal Adaptado**


# ==== FLUXO PRINCIPAL ====
# Ajuste o base_path para onde suas pastas SUNT estão localizadas.
# Exemplo: base_path = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
base_path = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp" 
output_path = "./output" # Onde os arquivos de saída serão salvos

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Número de dias a processar (opcional)
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Erro: argumento inválido. Use um número, ex: python script.py 5")
    sys.exit(1)

# Pega todas as pastas de data dentro da pasta 'OD' para determinar as datas disponíveis
# O padrão de pastas é SUNT/tmp/OD/, SUNT/tmp/AFC/, etc.
# Os arquivos são od-YYYY-MM-DD.parquet, afc-YYYY-MM-DD.parquet
# Precisamos encontrar as datas disponíveis nos arquivos parquet.
# Podemos listar os arquivos dentro de uma das subpastas (ex: OD) e extrair as datas.

available_files = []
od_dir = os.path.join(base_path, "OD")
if os.path.exists(od_dir):
    available_files = [f for f in os.listdir(od_dir) if f.startswith('od-') and f.endswith('.parquet')]
else:
    print(f"❌ Diretório de dados OD não encontrado: {od_dir}")
    sys.exit(1)

date_folders = []
for file_name in available_files:
    match = re.match(r'od-(\d{4}-\d{2}-\d{2})\.parquet', file_name)
    if match:
        date_folders.append(match.group(1))

date_folders.sort() # Garante a ordem cronológica

# Se foi especificado um número de dias, corta a lista
if max_days:
    date_folders = date_folders[:max_days]

combined_sum_amount = {} # Renomeado para refletir o nome do arquivo de saída
combined_averages = {} # Para as médias finais

# Processa cada data
for index, date_folder in enumerate(date_folders):
    print(f"\n--- Processando data {index+1}/{len(date_folders)}: {date_folder} ---")

    # Chama a função de processamento de dados para a data atual
    # Passamos base_path para que a função saiba onde encontrar as subpastas (OD, AFC, etc.)
    result_for_date = process_date_folder(date_folder, base_data_path=base_path)

    if result_for_date is None:
        print(f"🚫 Pulando a data {date_folder} devido a erro ou arquivo não encontrado.")
        continue

    # Salva os resultados intermediários para cada data (se desejado)
    avg_date_path = f"{output_path}/averages_{date_folder}.pkl"
    with open(avg_date_path, "wb") as f:
        pickle.dump(result_for_date, f)
    print(f"\n💾 Resultados intermediários para {date_folder} salvos em {avg_date_path}")

    # Combina os resultados para o total e contagem
    print(f"➕ Combinando dados de {date_folder} com os totais...\n")
    for stop_pair, (total_time, count) in result_for_date.items():
        if stop_pair not in combined_sum_amount:
            combined_sum_amount[stop_pair] = [0.0, 0] # Inicializa com float para total_time
        combined_sum_amount[stop_pair][0] += total_time
        combined_sum_amount[stop_pair][1] += count

# Salva os dados combinados (soma total do tempo e contagem)
print("\n📦 Salvando as somas e contagens combinadas...")
sum_amount_file = f"{output_path}/combined_sum_amount.pkl"
with open(sum_amount_file, "wb") as f:
    pickle.dump(combined_sum_amount, f)
print(f"✅ Somas e contagens combinadas salvas em {sum_amount_file}.")

# Calcula e salva as médias finais
print("\n📊 Calculando e salvando as médias finais...")
final_averages = {
    stop_pair: total_time / count
    for stop_pair, (total_time, count) in combined_sum_amount.items()
    if count > 0 # Evita divisão por zero
}
averages_file = f"{output_path}/combined_averages.pkl"
with open(averages_file, "wb") as f:
    pickle.dump(final_averages, f)
print(f"✅ Médias finais salvas em {averages_file}.")

print("\n🎉 Processamento concluído! Resultados prontos.")