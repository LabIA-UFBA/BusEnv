#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_daily_data.py
Gera arquivos diários .pkl com tempos de viagem, demanda, taxa de ocupação
e uptime bruto por veículo, a partir dos dados SUNT.
"""

import os
import re
import sys
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIGURAÇÕES DE CAMINHOS
# ============================================================
BASE_PATH = "/media/your_user/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "../training_observation"
GRAPH_PATH = "./src/viz/graph_gtfs_fev_2024.gpickle"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# PARÂMETROS GERAIS
# ============================================================
MAX_REASONABLE_DEMAND = 150
MAX_TRAVEL_TIME_SECONDS = 1800
MIN_TRAVEL_TIME_SECONDS = 120
MAX_UPTIME_HOURS = 20
BUS_CAPACITY = 80
SERVICE_HOURS = 16

# ============================================================
# CARREGAR GRAFO
# ============================================================
print("📦 Carregando grafo GTFS...")
try:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    valid_nodes = set(G.nodes)
    print(f"✅ Grafo carregado com {len(valid_nodes)} nós válidos.")
except Exception as e:
    print(f"❌ Erro ao carregar grafo: {e}")
    G = None
    valid_nodes = set()

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
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

def validate_stop_in_graph(stop_id):
    return str(stop_id) in valid_nodes if valid_nodes else True

def validate_edge_in_graph(a, b):
    if not G:
        return True
    a, b = str(a), str(b)
    return G.has_edge(a, b)

def remove_outliers_iqr(data, multiplier=1.5):
    if not data or len(data) < 4:
        return list(data)
    arr = np.array(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return arr[(arr >= lower) & (arr <= upper)].tolist()

# ============================================================
# NOVA FUNÇÃO — PROCESSAMENTO LTI (2024 e 2025)
# ============================================================
def process_lti_file(lti_path):
    """Processa arquivo LTI e retorna uptime bruto por veículo."""
    if not os.path.exists(lti_path):
        print(f"⚠️ Arquivo LTI não encontrado: {lti_path}")
        return {}

    try:
        df = pd.read_parquet(lti_path)
        if df.empty:
            print(f"⚠️ LTI vazio: {lti_path}")
            return {}

        # Detecta formato do arquivo
        if {'vehicle', 'start_trip', 'end_trip'}.issubset(df.columns):
            # Formato 2024
            df['inicio'] = pd.to_datetime(df['start_trip'], errors='coerce', dayfirst=True)
            df['fim'] = pd.to_datetime(df['end_trip'], errors='coerce', dayfirst=True)
            df['veiculo'] = df['vehicle']

        elif {'veiculo', 'inicioRealizado', 'fimRealizado'}.intersection(df.columns):
            # Formato 2025
            df['inicio'] = pd.to_datetime(
                df['inicioRealizado'].fillna(df.get('inicioProgramado')),
                errors='coerce', dayfirst=True
            )
            df['fim'] = pd.to_datetime(
                df['fimRealizado'].fillna(df.get('fimProgramado')),
                errors='coerce', dayfirst=True
            )
            df['veiculo'] = df['veiculo']
        else:
            print(f"⚠️ Estrutura inesperada no arquivo LTI: {lti_path}")
            return {}

        # Calcula uptime
        df['uptime_h'] = (df['fim'] - df['inicio']).dt.total_seconds() / 3600
        df = df[(df['uptime_h'] > 0) & (df['uptime_h'] <= 24)]

        uptime_raw = df.groupby('veiculo')['uptime_h'].apply(list).to_dict()
        avg = df['uptime_h'].mean() if not df.empty else 0
        print(f"   ✓ LTI: {len(uptime_raw)} veículos com uptime (média {avg:.2f}h)")
        return uptime_raw

    except Exception as e:
        print(f"⚠️ Erro ao processar LTI ({lti_path}): {e}")
        return {}

# ============================================================
# LOOP PRINCIPAL
# ============================================================
avg_travel_times = defaultdict(list)
future_demand = defaultdict(list)
occupancy = defaultdict(list)
uptime = defaultdict(list)
stop_activity = defaultdict(lambda: defaultdict(int))
route_frequencies = defaultdict(int)
graph_stats = {'invalid_stops': 0, 'invalid_edges': 0, 'valid_edges': 0}

dates = get_date_list("OD", "od")
print(f"\n📅 Dias detectados: {len(dates)}\n")

for i, date_str in enumerate(dates):
    print(f"\n➡️ {i+1}/{len(dates)} - {date_str}")

    # === OD: tempos de viagem ===
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
        df_od['stop_time'] = pd.to_datetime(df_od['stop_time'])
        grouped = df_od.sort_values(['vehicle', 'trip_id', 'direction_id', 'pt_sequence']).groupby(['vehicle', 'trip_id', 'direction_id'])
        for _, g in grouped:
            for j in range(1, len(g)):
                a, b = g.iloc[j-1]['stop_id'], g.iloc[j]['stop_id']
                if not (validate_stop_in_graph(a) and validate_stop_in_graph(b)):
                    graph_stats['invalid_stops'] += 1
                    continue
                if not validate_edge_in_graph(a, b):
                    graph_stats['invalid_edges'] += 1
                    continue
                delta = (g.iloc[j]['stop_time'] - g.iloc[j-1]['stop_time']).total_seconds()
                if MIN_TRAVEL_TIME_SECONDS <= delta <= MAX_TRAVEL_TIME_SECONDS:
                    avg_travel_times[(str(a), str(b))].append(delta)
                    route_frequencies[g.iloc[j]['route_short_name']] += 1
        print(f"   ✓ OD: {len(avg_travel_times)} pares processados")
    except Exception as e:
        print(f"⚠️ Erro OD {date_str}: {e}")

    # === Boarding: demanda ===
    try:
        df_board = pd.read_parquet(os.path.join(BASE_PATH, "Boarding", f"boarding-{date_str}.parquet"))
        df_board['stop_time'] = pd.to_datetime(df_board['stop_time'], errors='coerce')
        df_board = df_board.dropna(subset=['stop_time', 'stop_id'])
        for _, row in df_board.iterrows():
            stop_activity[row['stop_id']][row['stop_time'].hour] += 1
        stop_counts = df_board.groupby('stop_id').size()
        for sid, count in stop_counts.items():
            future_demand[sid].append(min(count, MAX_REASONABLE_DEMAND))
        print(f"   ✓ Boarding: {len(stop_counts)} paradas")
    except Exception as e:
        print(f"⚠️ Erro Boarding {date_str}: {e}")

    # === OD: ocupação ===
    try:
        if 'loading' in df_od.columns:
            for sid, grp in df_od.groupby('stop_id'):
                if validate_stop_in_graph(sid):
                    rate = min(grp['loading'].mean() / BUS_CAPACITY, 1.0)
                    occupancy[sid].append(rate)
        print(f"   ✓ Ocupação: {len(occupancy)} paradas")
    except Exception as e:
        print(f"⚠️ Erro Ocupação {date_str}: {e}")

    # === LTI: uptime ===
    try:
        lti_path = os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet")
        vehicle_uptime_raw = process_lti_file(lti_path)
        for vid, vals in vehicle_uptime_raw.items():
            uptime[vid].extend(vals)
    except Exception as e:
        print(f"⚠️ Erro LTI {date_str}: {e}")

# ============================================================
# SALVAMENTO DOS RESULTADOS
# ============================================================
def finalize_dict(d):
    out = {}
    for k, v in d.items():
        if v:
            clean = remove_outliers_iqr(v)
            out[k] = np.mean(clean)
    return out

print("\n💾 Salvando arquivos...")

with open(os.path.join(OUTPUT_PATH, "avg_travel_time_AB.pkl"), "wb") as f:
    pickle.dump(finalize_dict(avg_travel_times), f)
with open(os.path.join(OUTPUT_PATH, "future_demand_at_B.pkl"), "wb") as f:
    pickle.dump(finalize_dict(future_demand), f)
with open(os.path.join(OUTPUT_PATH, "occupancy_rate.pkl"), "wb") as f:
    pickle.dump(finalize_dict(occupancy), f)
with open(os.path.join(OUTPUT_PATH, "uptime_normalized.pkl"), "wb") as f:
    pickle.dump(finalize_dict(uptime), f)

with open(os.path.join(OUTPUT_PATH, "route_frequencies.pkl"), "wb") as f:
    pickle.dump(dict(route_frequencies), f)
with open(os.path.join(OUTPUT_PATH, "hourly_stop_activity.pkl"), "wb") as f:
    pickle.dump(dict(stop_activity), f)

print("\n✅ Processamento concluído com sucesso.")
