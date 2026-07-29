#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_daily_data.py
Generates daily .pkl files containing:
1. Average travel time between pairs of stops
2. Demand per stop
3. Vehicle occupancy rate
4. Normalized operating time per vehicle
"""

import os
import re
import sys
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

# ============================================================
# CONFIGURAÇÕES DE CAMINHOS
# ============================================================
BASE_PATH = "/mnt/ssd1/xxxx/xxx/xxxx/tpm"
OUTPUT_PATH = "/mnt/ssd1/xxxx/xxx/xxxx/tpm/daily"
GRAPH_PATH = "/mnt/ssd1/xxxx/xxx/src/viz/graph_gtfs_fev_2024.gpickle"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# PARÂMETROS
# ============================================================
MAX_REASONABLE_DEMAND = 150
MAX_TRAVEL_TIME_SECONDS = 1800
MIN_TRAVEL_TIME_SECONDS = 120
MAX_UPTIME_HOURS = 20
BUS_CAPACITY = 80  # capacidade média por veículo
SERVICE_HOURS = 16

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder):
        print(f"❌ Diretório não encontrado: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".parquet")]
    dates = []
    for fname in filenames:
        match = re.match(fr"{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet", fname)
        if match:
            dates.append(match.group(1))
    return sorted(dates)

def validate_stop(stop_id):
    return stop_id in VALID_NODES

def validate_edge(a, b):
    if not G:
        return True
    a, b = str(a), str(b)
    return G.has_edge(a, b) or (nx.is_directed(G) and G.has_edge(b, a))

# ============================================================
# CARREGAR GRAFO
# ============================================================
print("📦 Carregando grafo GTFS...")
try:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    VALID_NODES = set(map(str, G.nodes))
    print(f"✅ Grafo carregado com {len(VALID_NODES)} nós válidos.")
except Exception as e:
    print(f"❌ Erro ao carregar grafo: {e}")
    G = None
    VALID_NODES = set()

# ============================================================
# LOOP PRINCIPAL
# ============================================================
dates = get_date_list("OD", "od")
print(f"\n📅 Dias detectados: {len(dates)}\n")

for day_idx, date_str in enumerate(dates):
    print(f"\n➡️  {day_idx+1}/{len(dates)} - {date_str}")

    avg_travel_times = defaultdict(list)
    future_demand = defaultdict(int)
    occupancy_rate = defaultdict(list)
    uptime_normalized = defaultdict(float)

    # === OD: tempos de viagem e ocupação ===
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))

        print("\n==================== DEBUG OD ====================")
        print("Arquivo:", date_str)
        print("Total de registros:", len(df_od))
        print("Colunas:", list(df_od.columns))

        df_od["stop_id"] = df_od["stop_id"].astype(str)
        df_od["stop_time"] = pd.to_datetime(df_od["stop_time"])

        print("Nós únicos:", df_od["stop_id"].nunique())
        print("Primeiros stop_ids:")
        print(df_od["stop_id"].head(10).tolist())

        print("Primeiros registros:")
        print(df_od[["stop_id", "trip_id", "vehicle", "pt_sequence"]].head())
        print("=================================================\n")

        valid_nodes = df_od["stop_id"].isin(VALID_NODES).sum()

        print("Nós presentes no grafo:", valid_nodes)
        print("Nós fora do grafo:", len(df_od) - valid_nodes)

        if valid_nodes == 0:
            print("⚠️ Nenhum stop_id do arquivo existe no grafo.")

        total_pairs = 0
        valid_stop_pairs = 0
        valid_edge_pairs = 0
        accepted_pairs = 0

        # Tempo entre paradas
        grouped = df_od.sort_values(by=["vehicle", "trip_id", "direction_id", "pt_sequence"]) \
                       .groupby(["vehicle", "trip_id", "direction_id"])
        for _, group in grouped:
            for i in range(1, len(group)):

                total_pairs += 1

                prev, curr = group.iloc[i - 1], group.iloc[i]

                a, b = prev["stop_id"], curr["stop_id"]

                if not (validate_stop(a) and validate_stop(b)):
                    continue

                valid_stop_pairs += 1

                if not validate_edge(a, b):
                    continue

                valid_edge_pairs += 1

                delta = (curr["stop_time"] - prev["stop_time"]).total_seconds()

                if MIN_TRAVEL_TIME_SECONDS <= delta <= MAX_TRAVEL_TIME_SECONDS:
                    accepted_pairs += 1
                    avg_travel_times[(a, b)].append(delta)

        print("\n========== DEBUG TRAVEL ==========")
        print("Total pares:", total_pairs)
        print("Passaram validate_stop:", valid_stop_pairs)
        print("Passaram validate_edge:", valid_edge_pairs)
        print("Passaram filtro tempo:", accepted_pairs)
        print("==================================")

        # Ocupação média por parada
        if "loading" in df_od.columns:
            df_od["loading"] = pd.to_numeric(df_od["loading"], errors="coerce")
            loading_rows = 0
            loading_valid_nodes = 0
            for stop_id, group in df_od.groupby("stop_id"):
                loading_rows += 1
                if not validate_stop(stop_id):
                    continue
                loading_valid_nodes += 1
                avg_load = group["loading"].mean()
                occupancy = min(avg_load / BUS_CAPACITY, 1.0)
                occupancy_rate[stop_id].append(occupancy)

            print("\n========== DEBUG OCCUPANCY ==========")
            print("Stops com loading:", loading_rows)
            print("Stops válidos:", loading_valid_nodes)
            print("Occupancy geradas:", len(occupancy_rate))
            print("=====================================")
        print(f"   ✓ OD: {len(avg_travel_times)} pares de paradas, {len(occupancy_rate)} ocupações")
    except Exception as e:
        print(f"⚠️ Erro OD {date_str}: {e}")

    # === Boarding: demanda por parada ===
    try:
        df_board = pd.read_parquet(os.path.join(BASE_PATH, "Boarding", f"boarding-{date_str}.parquet"))
        df_board["stop_id"] = df_board["stop_id"].astype(str)
        print("\n========== DEBUG BOARDING ==========")
        print("Registros:", len(df_board))
        print("Nós únicos:", df_board["stop_id"].nunique())

        valid = df_board["stop_id"].isin(VALID_NODES).sum()

        print("Registros em nós válidos:", valid)
        print("====================================")
        df_board = df_board[df_board["stop_id"].apply(validate_stop)]
        stop_counts = df_board.groupby("stop_id").size()
        for stop_id, count in stop_counts.items():
            future_demand[stop_id] = min(count, MAX_REASONABLE_DEMAND)
        print(f"   ✓ Boarding: {len(future_demand)} paradas com demanda")
    except Exception as e:
        print(f"⚠️ Erro Boarding {date_str}: {e}")

    # === LTI: uptime normalizado ===
    try:
        df_lti = pd.read_parquet(os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet"))

        # --- Ajuste para nomes diferentes entre anos ---
        if "start_trip" in df_lti.columns and "end_trip" in df_lti.columns:
            df_lti["start_trip"] = pd.to_datetime(df_lti["start_trip"], errors="coerce", dayfirst=True)
            df_lti["end_trip"] = pd.to_datetime(df_lti["end_trip"], errors="coerce", dayfirst=True)
        elif "inicioProgramado" in df_lti.columns and "fimProgramado" in df_lti.columns:
            df_lti["start_trip"] = pd.to_datetime(df_lti["inicioProgramado"], errors="coerce", dayfirst=True)
            df_lti["end_trip"] = pd.to_datetime(df_lti["fimProgramado"], errors="coerce", dayfirst=True)
        else:
            raise ValueError("⚠️ LTI não possui colunas de tempo reconhecidas.")

        vehicle_col = "vehicle" if "vehicle" in df_lti.columns else "veiculo"
        df_lti = df_lti.dropna(subset=["start_trip", "end_trip", vehicle_col])

        # --- Filtro atualizado de atividades válidas ---
        if "activity" in df_lti.columns:
            df_lti = df_lti[df_lti["activity"].isin(["Viagem Normal", "Viagem Extra"])]
        elif "atividade" in df_lti.columns:
            df_lti = df_lti[df_lti["atividade"].isin(["Viagem Normal", "Viagem Extra"])]

        # --- Cálculo do uptime ---
        for _, row in df_lti.iterrows():
            uptime_hours = (row["end_trip"] - row["start_trip"]).total_seconds() / 3600
            if 0.1 <= uptime_hours <= MAX_UPTIME_HOURS:
                uptime_normalized[row[vehicle_col]] += uptime_hours / SERVICE_HOURS

        print(f"   ✓ LTI: {len(uptime_normalized)} veículos com uptime")

    except Exception as e:
        print(f"⚠️ Erro LTI {date_str}: {e}")


    # === SALVAR RESULTADOS ===
    day_data = {
        "date": date_str,
        "avg_travel_times": {k: np.mean(v) for k, v in avg_travel_times.items()},
        "future_demand": dict(future_demand),
        "occupancy_rate": {k: np.mean(v) for k, v in occupancy_rate.items()},
        "uptime_normalized": dict(uptime_normalized),
    }

    out_file = os.path.join(OUTPUT_PATH, f"daily_data_{date_str}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(day_data, f)
    print(f"💾 Arquivo salvo: {out_file}")

print("\n🏁 Processamento concluído com sucesso.")
