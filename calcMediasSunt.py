import os
import pandas as pd
import pickle
import re
import glob
from collections import defaultdict

BASE_PATH = "./sunt"
OUTPUT_PATH = "./output"

dias_com_erro = []

def encontrar_arquivos(folder, nome_base):
    arquivos = glob.glob(os.path.join(folder, f"{nome_base}*.parquet"))
    return arquivos[0] if arquivos else None

def process_day_data(date_str):
    folder = os.path.join(BASE_PATH, date_str, "output")

    try:
        boarding_path = encontrar_arquivos(folder, "boarding")
        landing_path = encontrar_arquivos(folder, "landing")
        trips_path = encontrar_arquivos(folder, "trips_time-series")

        if not all([boarding_path, landing_path, trips_path]):
            raise FileNotFoundError("Um ou mais arquivos .parquet estão faltando.")

        boarding_df = pd.read_parquet(boarding_path)
        landing_df = pd.read_parquet(landing_path)
        trips_df = pd.read_parquet(trips_path)

        return boarding_df, landing_df, trips_df

    except Exception as e:
        print(f"⚠️ Arquivo(s) faltando para o dia {date_str}: {e}. Pulando...")
        dias_com_erro.append((date_str, "arquivo_faltando"))
        return None, None, None


def process_boarding(boarding_df, date_str):
    if "stop_id" not in boarding_df.columns or "registers" not in boarding_df.columns:
        print(f"⚠️ Dados de boarding incompletos em {date_str}. Pulando esse dia.")
        dias_com_erro.append((date_str, "boarding_invalido"))
        return {}
    return boarding_df.groupby("stop_id")["registers"].sum().to_dict()


def process_landing(landing_df, date_str):
    if "stop_id_ali" not in landing_df.columns:
        print(f"⚠️ Dados de landing incompletos em {date_str}. Pulando esse dia.")
        dias_com_erro.append((date_str, "landing_invalido"))
        return {}
    return landing_df["stop_id_ali"].value_counts().to_dict()


def process_trips(trips_df, date_str):
    required_cols = {"trip", "hora_ponto", "stop_id", "tempo_total", "lat", "lon"}
    if not required_cols.issubset(set(trips_df.columns)):
        print(f"⚠️ Dados de trips incompletos em {date_str}. Pulando esse dia.")
        dias_com_erro.append((date_str, "trips_invalido"))
        return {}, {}

    edge_times = defaultdict(lambda: [0, 0])  # (soma, contagem)
    coords = {}

    grouped = trips_df.groupby("trip")
    for _, trip_data in grouped:
        trip_data = trip_data.sort_values("hora_ponto")
        stops = trip_data["stop_id"].tolist()
        times = trip_data["tempo_total"].tolist()
        lats = trip_data["lat"].tolist()
        lons = trip_data["lon"].tolist()

        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i+1])
            edge_times[pair][0] += times[i+1]
            edge_times[pair][1] += 1

        for stop, lat, lon in zip(stops, lats, lons):
            coords[stop] = (lat, lon)

    return edge_times, coords

# Inicializa agregadores
combined_boarding = defaultdict(int)
combined_landing = defaultdict(int)
combined_edges = defaultdict(lambda: [0, 0])
combined_coords = {}

# Coleta as pastas de data
date_folders = sorted([
    f for f in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, f)) and re.match(r'\d{4}-\d{2}-\d{2}', f)
])

# Itera por cada dia
for date_str in date_folders:
    print(f"📅 Processando {date_str}")
    boarding_df, landing_df, trips_df = process_day_data(date_str)

    if boarding_df is not None:
        day_boarding = process_boarding(boarding_df, date_str)
        for stop, count in day_boarding.items():
            combined_boarding[stop] += count

    if landing_df is not None:
        day_landing = process_landing(landing_df, date_str)
        for stop, count in day_landing.items():
            combined_landing[stop] += count

    if trips_df is not None:
        day_edges, day_coords = process_trips(trips_df, date_str)
        for pair, (t, c) in day_edges.items():
            combined_edges[pair][0] += t
            combined_edges[pair][1] += c
        for stop, coord in day_coords.items():
            combined_coords[stop] = coord  # sempre substitui (assume fixo)

# Finaliza os tempos médios
final_edge_times = {
    pair: total / count
    for pair, (total, count) in combined_edges.items() if count > 0
}

# Salva tudo
os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(os.path.join(OUTPUT_PATH, "boarding_demand.pkl"), "wb") as f:
    pickle.dump(dict(combined_boarding), f)

with open(os.path.join(OUTPUT_PATH, "landing_demand.pkl"), "wb") as f:
    pickle.dump(dict(combined_landing), f)

with open(os.path.join(OUTPUT_PATH, "edge_times.pkl"), "wb") as f:
    pickle.dump(final_edge_times, f)

with open(os.path.join(OUTPUT_PATH, "stop_coords.pkl"), "wb") as f:
    pickle.dump(combined_coords, f)

with open(os.path.join(OUTPUT_PATH, "dias_com_erro.pkl"), "wb") as f:
    pickle.dump(dias_com_erro, f)

print("✅ Tudo processado e salvo!")
if dias_com_erro:
    print(f"⚠️ {len(dias_com_erro)} dias com problemas foram registrados em dias_com_erro.pkl")
