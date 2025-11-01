import os
import re
import sys
import pickle
import pandas as pd

# === CONFIGURAÇÕES ===
BASE_PATH = "/media/wesley/Disco_local/tes/BusEnv/SUNT/tmp"
OUTPUT_PATH = "/media/wesley/Disco_local/tes/BusEnv/src/training_observation"
GRAPH_PATH = "/media/wesley/Disco_local/tes/BusEnv/src/viz/graph_gtfs_fev_2024.gpickle"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Leitura opcional do número máximo de dias ===
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # padrão: 1 dia
except ValueError:
    print("⚠️ Argumento inválido. Use um número, ex: python generate_real_routes_dedup.py 1")
    sys.exit(1)

# === Carregar o grafo GTFS ===
print("📦 Carregando grafo GTFS...")
try:
    with open(GRAPH_PATH, "rb") as f:
        import networkx as nx
        G = pickle.load(f)
    valid_nodes = set(G.nodes)
    print(f"✅ Grafo carregado com {len(valid_nodes)} nós válidos.")
except Exception as e:
    print(f"❌ Erro ao carregar grafo: {e}")
    sys.exit(1)


# === Função auxiliar para pegar as datas ===
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder):
        print(f"❌ Diretório não encontrado: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')]
    dates = []
    for fname in filenames:
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname)
        if match:
            dates.append(match.group(1))
    return sorted(dates)


# === Coletar as datas e limitar ===
dates = get_date_list("OD", "od")
if not dates:
    print("❌ Nenhum arquivo OD encontrado!")
    sys.exit(1)

dates = dates[:max_days]
print(f"\n📅 Processando {len(dates)} dia(s): {dates}")

# === Acumuladores ===
real_routes = {}
route_metadata = {}

# === Loop pelos dias ===
for i, date_str in enumerate(dates):
    print(f"\n➡️  {i+1}/{len(dates)} - {date_str}")
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))

        required_cols = ["route_short_name", "direction_id", "vehicle", "trip_number", "stop_id", "pt_sequence"]
        if not all(col in df_od.columns for col in required_cols):
            print(f"⚠️ Colunas esperadas ausentes em {date_str}")
            continue

        df_od = df_od.dropna(subset=["stop_id", "trip_number", "route_short_name"])
        df_od["vehicle"] = df_od["vehicle"].astype(str)
        df_od["stop_id"] = df_od["stop_id"].astype(str)
        df_od["trip_number"] = df_od["trip_number"].astype(int)
        df_od["trip_id"] = df_od.apply(lambda x: f"{x['vehicle']}_{x['route_short_name']}_{x['trip_number']}", axis=1)

        grouped = df_od.sort_values(by=["vehicle", "trip_id", "direction_id", "pt_sequence"]).groupby("trip_id")

        for trip_id, group in grouped:
            all_stops = group["stop_id"].tolist()
            filtered_stops = [s for s in all_stops if s in valid_nodes]
            if len(filtered_stops) > 1:
                real_routes[trip_id] = filtered_stops
                route_metadata[trip_id] = {
                    "route_short_name": group["route_short_name"].iloc[0],
                    "direction_id": group["direction_id"].iloc[0],
                    "vehicle": group["vehicle"].iloc[0],
                    "trip_number": group["trip_number"].iloc[0],
                }

    except Exception as e:
        print(f"⚠️ Erro processando OD {date_str}: {e}")

# === Deduplicação de rotas idênticas (por stops) ===
print("\n🧹 Removendo rotas duplicadas...")
unique_routes = {}
metadata_cleaned = {}
seen = set()

for trip_id, stops in real_routes.items():
    route_key = tuple(stops)
    if route_key not in seen:
        seen.add(route_key)
        unique_routes[trip_id] = stops
        metadata_cleaned[trip_id] = route_metadata[trip_id]
    else:
        pass  # Rota duplicada ignorada

print(f"🧮 Total antes da deduplicação: {len(real_routes)}")
print(f"✅ Total após deduplicação: {len(unique_routes)}")
removed = len(real_routes) - len(unique_routes)
print(f"🚮 Rotas removidas: {removed}")

# === Salvamento final ===
with open(os.path.join(OUTPUT_PATH, "real_routes.pkl"), "wb") as f:
    pickle.dump(unique_routes, f)

with open(os.path.join(OUTPUT_PATH, "route_metadata.pkl"), "wb") as f:
    pickle.dump(metadata_cleaned, f)

print("\n🎉 Arquivos salvos com sucesso em:")
print(f"📁 {os.path.join(OUTPUT_PATH, 'real_routes.pkl')}")
print(f"📁 {os.path.join(OUTPUT_PATH, 'route_metadata.pkl')}")
print("\n✅ Rotas únicas da cidade geradas com sucesso.")
