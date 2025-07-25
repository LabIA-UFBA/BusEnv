import os
import re
import sys
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict

# === Caminhos principais ===
BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "./output_obs"
GRAPH_PATH = "./SUNT/data/graph_designer/graph_gtfs_fev_2024.gpickle" # Caminho do grafo GTFS

# Cria diretório de saída se necessário
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Argumento opcional para limitar número de dias ===
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Erro: argumento inválido. Use um número, ex: python script.py 5")
    sys.exit(1)

# === Carrega o grafo urbano ===
print("📦 Carregando grafo GTFS...")
try:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    valid_nodes = set(G.nodes)
    print(f"✅ Grafo carregado com {len(valid_nodes)} nós válidos.")
except Exception as e:
    print(f"❌ Erro ao carregar grafo: {e}")
    sys.exit(1)

# === Função auxiliar para coletar datas ===
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder): # Verifica se o diretório existe
        print(f"❌ Diretório não encontrado: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')] # Filtra arquivos com o prefixo e extensão corretos
    dates = [] # Lista para armazenar as datas extraídas
    for fname in filenames:
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname) # Usa regex para extrair a data do nome do arquivo
        if match:
            dates.append(match.group(1)) # Adiciona a data extraída à lista
    return sorted(dates) # Ordena as datas em ordem cronológica

# === Acumuladores ===
real_routes = dict() # Dicionário para armazenar rotas reais
route_metadata = dict() # Dicionário para armazenar metadados das rotas

dates = get_date_list("OD", "od")
if max_days:
    dates = dates[:max_days] # Limita o número de dias a processar

print(f"\n📅 Processando {len(dates)} dias para extrair rotas reais válidas...")

for i, date_str in enumerate(dates):
    print(f"\n➡️  {i+1}/{len(dates)} - {date_str}") # Indica o progresso do processamento
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet")) # Lê o arquivo OD correspondente à data atual

        required_columns = ["route_short_name", "direction_id", "vehicle", "trip_number", "stop_id", "pt_sequence"] # Colunas necessárias para validação
        if not all(col in df_od.columns for col in required_columns):
            print(f"⚠️ Colunas esperadas ausentes em OD {date_str}")
            continue

        df_od = df_od.dropna(subset=["stop_id", "trip_number", "route_short_name"]) # Remove registros com valores ausentes nas colunas críticas
        df_od["vehicle"] = df_od["vehicle"].astype(str) # Converte o ID do veículo para string
        df_od["stop_id"] = df_od["stop_id"].astype(str)
        df_od["trip_number"] = df_od["trip_number"].astype(int)

        df_od["trip_id"] = df_od.apply(lambda x: f"{x['vehicle']}_{x['route_short_name']}_{x['trip_number']}", axis=1) # Cria um ID de viagem único baseado no veículo, nome da rota e número da viagem

        grouped = df_od.sort_values(by=["vehicle", "trip_id", "direction_id", "pt_sequence"]) \
                       .groupby("trip_id") # Agrupa os dados por ID de viagem

        for trip_id, group in grouped: # Processa cada grupo de dados por viagem
            all_stops = group["stop_id"].tolist()
            filtered_stops = [s for s in all_stops if s in valid_nodes]

            if len(filtered_stops) > 1: # Verifica se há mais de uma parada válida
                if len(filtered_stops) < len(all_stops): # algumas paradas foram removidas por não estarem no grafo
                    missing = set(all_stops) - set(filtered_stops) # Identifica paradas removidas
                    print(f"⚠️ Trip {trip_id}: {len(missing)} paradas removidas por não estarem no grafo.") # Informa sobre paradas removidas

                real_routes[trip_id] = filtered_stops
                route_metadata[trip_id] = {
                    "route_short_name": group["route_short_name"].iloc[0], # Nome da rota
                    "direction_id": group["direction_id"].iloc[0], # ID da direção
                    "vehicle": group["vehicle"].iloc[0], # ID do veículo
                    "trip_number": group["trip_number"].iloc[0] # Número da viagem
                }

    except Exception as e:
        print(f"⚠️ Erro ao processar OD {date_str}: {e}")

# === Salvamento final ===
with open(os.path.join(OUTPUT_PATH, "rotas_reais.pkl"), "wb") as f:
    pickle.dump(real_routes, f)

with open(os.path.join(OUTPUT_PATH, "rotas_metadata.pkl"), "wb") as f:
    pickle.dump(route_metadata, f)

print("\n✅ Arquivos 'rotas_reais.pkl' e 'rotas_metadata.pkl' salvos com sucesso com validação no grafo.")
