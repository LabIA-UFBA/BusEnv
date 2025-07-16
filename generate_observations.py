import os
import re
import sys
import pickle
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Diretórios principais
BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "./output_obs"

# Cria diretório de saída se necessário
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Argumento opcional para limitar número de dias
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Erro: argumento inválido. Use um número, ex: python script.py 5")
    sys.exit(1)

# === Funções Auxiliares ===
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

def normalize_id(val, max_val=1_000_000):
    return int(val) / max_val if val and val >= 0 else 0.0

def normalize_time_of_day(dt):
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400

# === Acumuladores ===
avg_travel_times = defaultdict(lambda: [0.0, 0])   # (total_time, count)
future_demand = defaultdict(lambda: [0, 0])       # (total_boardings, count)
occupancy = defaultdict(lambda: [0.0, 0])         # (sum_occupancy, count)
uptime = defaultdict(lambda: [0.0, 0])            # (sum_uptime_seconds, count)

dates = get_date_list("OD", "od")
if max_days:
    dates = dates[:max_days]

print(f"\n📅 Processando {len(dates)} dias...")

for i, date_str in enumerate(dates):
    print(f"\n➡️  {i+1}/{len(dates)} - {date_str}")

    # --- OD: avg_travel_time_AB ---
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
        df_od['stop_time'] = pd.to_datetime(df_od['stop_time'])
        grouped = df_od.sort_values(by=['vehicle', 'trip_id', 'direction_id', 'pt_sequence']) \
                        .groupby(['vehicle', 'trip_id', 'direction_id'])
        for _, group in grouped:
            for i in range(1, len(group)):
                prev = group.iloc[i-1]
                curr = group.iloc[i]
                delta = (curr['stop_time'] - prev['stop_time']).total_seconds()
                if delta > 0:
                    pair = (prev['stop_id'], curr['stop_id'])
                    avg_travel_times[pair][0] += delta
                    avg_travel_times[pair][1] += 1
    except Exception as e:
        print(f"⚠️ Erro OD {date_str}: {e}")

    # --- Boarding: future_demand_at_B ---
    try:
        df_board = pd.read_parquet(os.path.join(BASE_PATH, "Boarding", f"boarding-{date_str}.parquet"))
        if df_board is not None and "stop_id" in df_board.columns:
            df_board.dropna(subset=["stop_id"], inplace=True)

            for stop_id, group in df_board.groupby("stop_id"):
                total = len(group)  # número de embarques estimados no ponto
                future_demand[stop_id][0] += total
                future_demand[stop_id][1] += 1  # conta o número de vezes que vimos esse ponto
        else:
            print(f"⚠️ Coluna 'target_boarding' não encontrada em Boarding {date_str}")
    except Exception as e:
        print(f"⚠️ Erro Boarding {date_str}: {e}")

    # --- AVL: occupancy_rate ---
    try:
        df_avl = pd.read_parquet(os.path.join(BASE_PATH, "AVL", f"avl-lines-{date_str}.parquet"))
        if 'stop_id' in df_avl.columns:
            for stop_id, group in df_avl.groupby('stop_id'):
                count = len(group)
                if count > 0:
                    # Ocupação simulada, pois coluna real está ausente
                    occ_value = 0.5  # valor dummy, pois AVL não possui occupancy_rate real
                    occupancy[stop_id][0] += occ_value * count
                    occupancy[stop_id][1] += count
        else:
            print(f"⚠️ Coluna 'stop_id' ausente em AVL {date_str}")
    except Exception as e:
        print(f"⚠️ Erro AVL {date_str}: {e}")

    # --- LTI: uptime_normalized ---
    try:
        df_lti = pd.read_parquet(os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet"))

        # Verifica quais colunas de tempo estão disponíveis
        if 'start_trip' in df_lti.columns and 'end_trip' in df_lti.columns:
            df_lti['start_trip'] = pd.to_datetime(df_lti['start_trip'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['end_trip'], errors='coerce', dayfirst=True)
        elif 'inicioProgramado' in df_lti.columns and 'fimProgramado' in df_lti.columns:
            df_lti['start_trip'] = pd.to_datetime(df_lti['inicioProgramado'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['fimProgramado'], errors='coerce', dayfirst=True)
        else:
            raise ValueError("⚠️ LTI não possui colunas de tempo reconhecidas.")

        # Garante que a coluna 'vehicle' exista (ou ajusta se tiver nome diferente)
        vehicle_col = 'vehicle' if 'vehicle' in df_lti.columns else 'veiculo'
        df_lti.dropna(subset=['start_trip', 'end_trip', vehicle_col], inplace=True)

        for _, row in df_lti.iterrows():
            uptime_sec = (row['end_trip'] - row['start_trip']).total_seconds()
            if uptime_sec > 0:
                uptime[row[vehicle_col]][0] += uptime_sec
                uptime[row[vehicle_col]][1] += 1
    except Exception as e:
        print(f"⚠️ Erro LTI {date_str}: {e}")


# === Finalização ===
def finalize_avg_dict(d):
    return {k: v[0]/v[1] for k, v in d.items() if v[1] > 0}

with open(os.path.join(OUTPUT_PATH, "avg_travel_time_AB.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(avg_travel_times), f)

with open(os.path.join(OUTPUT_PATH, "future_demand_at_B.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(future_demand), f)

with open(os.path.join(OUTPUT_PATH, "occupancy_rate.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(occupancy), f)

with open(os.path.join(OUTPUT_PATH, "uptime_normalized.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(uptime), f)

print("\n✅ Todos os dados foram processados e salvos.")
