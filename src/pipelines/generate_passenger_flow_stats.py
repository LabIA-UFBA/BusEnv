#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_passenger_flow_stats.py

Gera estatísticas REAIS de embarque/desembarque por parada, a partir dos
dados brutos OD (n-boardings, n-alighting, lag_loading), para substituir a
lógica atual de ocupação por percentual fixo (alpha=0.5 / ALIGHTING_RATE=0.20)
por uma fila de passageiros por parada alimentada por dados reais.

Granularidade (com fallback em cascata, do mais específico ao mais genérico):
  by_route_stop_hour: (trip_id, stop_id, hour) -> {mean_boardings, mean_alight_frac, count}
  by_route_stop:      (trip_id, stop_id)        -> {mean_boardings, mean_alight_frac, mean_intervisit_sec, count}
  by_stop_hour:       (stop_id, hour)            -> {mean_boardings, mean_alight_frac, count}   (pool de todas as rotas)
  by_stop:            stop_id                    -> {mean_boardings, mean_alight_frac, mean_intervisit_sec, count}
  global:             {mean_boardings, mean_alight_frac, mean_intervisit_sec}

Cada média é a média aritmética direta das amostras reais (soma/contagem),
sem nenhuma filtragem estatística de "outliers": n-boardings/n-alighting são
dados de contagem fortemente concentrados em zero (a maioria das visitas não
tem ninguém embarcando/desembarcando), e uma filtragem por IQR sobre esse tipo
de distribuição colapsa (Q1=Q3=0) e descarta quase toda visita com embarque
real como "outlier" — o que sub-estimaria sistematicamente a demanda real.
Preferimos manter o dado bruto fiel, com apenas um filtro de sanidade fixo
(30s-4h) no intervalo entre visitas, para remover falhas óbvias de dado (ex.:
borda do dia), não para "normalizar" a distribuição.

`mean_intervisit_sec` é calculado por (route_short_name, direction_id, stop_id)
a partir do intervalo entre visitas consecutivas de qualquer veículo daquela
rota+direção à parada (aproxima o headway real de linha), e é usado no
simulador para converter "passageiros médios por visita" em uma taxa de
chegada (passageiros/segundo) que alimenta a fila entre visitas de ônibus.

As chaves de rota nos níveis by_route_* usam os mesmos trip_id de
`real_routes.pkl`/`route_metadata.pkl` (ex.: "20001_310_1"), mapeados a partir
de (route_short_name, direction_id) via `route_metadata.pkl`, para que o
ambiente (`sunt_env.py`) possa consultar diretamente por `route_id` sem
tradução adicional.
"""

import argparse
import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================
# CONFIGURAÇÕES DE CAMINHOS
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_PATH = "your base path"
DEFAULT_OUTPUT_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "training_observation"))
GRAPH_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "viz", "graph_gtfs_fev_2024.gpickle"))

# ============================================================
# PARÂMETROS
# ============================================================
MAX_INTERVISIT_SECONDS = 4 * 3600   # ignora gaps > 4h (provável falha de dado / borda do dia)
MIN_INTERVISIT_SECONDS = 30          # ignora gaps quase nulos (provável duplicidade de registro)
MIN_BUCKET_COUNT = 3                 # usado no ambiente para decidir se um bucket é confiável o bastante

_arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_arg_parser.add_argument(
    "--base-path", default=DEFAULT_BASE_PATH,
    help="Directory containing the OD/ (and Boarding/, LTI/) parquet subfolders. "
         "ALL od-YYYY-MM-DD.parquet files found under <base-path>/OD/ are processed "
         "automatically — to update with new/different data, drop the new parquet "
         "files into that folder (or point --base-path at a different folder entirely) "
         "and rerun this script; there is no hardcoded date list or day count.",
)
_arg_parser.add_argument(
    "--output-dir", default=DEFAULT_OUTPUT_PATH,
    help="Where to write stop_passenger_flow.pkl (also where route_metadata.pkl is read from).",
)
_args = _arg_parser.parse_args()

BASE_PATH = _args.base_path
OUTPUT_PATH = os.path.normpath(_args.output_dir)
os.makedirs(OUTPUT_PATH, exist_ok=True)


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


class SumCountAccumulator:
    """Accumulates (sum, count) per key across days, for a plain arithmetic mean."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, agg_df: pd.DataFrame):
        """agg_df: DataFrame indexed by key, with 'sum' and 'count' columns."""
        for key, row in zip(agg_df.index, agg_df.itertuples(index=False)):
            if row.count == 0:
                continue
            self.sums[key] += row.sum
            self.counts[key] += row.count

    def mean(self, key):
        c = self.counts.get(key, 0)
        if c == 0:
            return None
        return self.sums[key] / c

    def count(self, key):
        return self.counts.get(key, 0)

    def keys(self):
        return self.counts.keys()


print("📦 Carregando grafo GTFS...")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
VALID_NODES = set(map(str, G.nodes))
print(f"✅ Grafo carregado com {len(VALID_NODES)} nós válidos.")

print("📦 Carregando route_metadata.pkl...")
with open(os.path.join(OUTPUT_PATH, "route_metadata.pkl"), "rb") as f:
    route_metadata = pickle.load(f)
print(f"✅ {len(route_metadata)} trip_ids conhecidos.")

dates = get_date_list("OD", "od")
print(f"\n📅 Dias detectados: {len(dates)}\n")
if not dates:
    raise SystemExit("Nenhum arquivo OD encontrado em " + BASE_PATH)

# ============================================================
# Acumuladores (sum, count) por nível de granularidade
# ============================================================
board_route_stop_hour = SumCountAccumulator()   # (rsn, did, stop_id, hour)
alight_route_stop_hour = SumCountAccumulator()

board_route_stop = SumCountAccumulator()        # (rsn, did, stop_id) -- todas as horas
alight_route_stop = SumCountAccumulator()

board_stop_hour = SumCountAccumulator()         # (stop_id, hour) -- pool entre rotas
alight_stop_hour = SumCountAccumulator()

board_stop = SumCountAccumulator()              # stop_id -- pool entre rotas e horas
alight_stop = SumCountAccumulator()

intervisit_route_stop = SumCountAccumulator()   # (rsn, did, stop_id)
intervisit_stop = SumCountAccumulator()         # stop_id -- pool entre rotas

board_global_sum, board_global_cnt = 0.0, 0
alight_global_sum, alight_global_cnt = 0.0, 0
intervisit_global_sum, intervisit_global_cnt = 0.0, 0

for day_idx, date_str in enumerate(dates):
    print(f"➡️  {day_idx + 1}/{len(dates)} - {date_str}")
    try:
        df = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
    except Exception as e:
        print(f"⚠️ Erro lendo OD {date_str}: {e}")
        continue

    required = {"route_short_name", "direction_id", "stop_id", "stop_time", "n-boardings", "n-alighting", "lag_loading"}
    if not required.issubset(df.columns):
        print(f"⚠️ Colunas ausentes em {date_str}, pulando.")
        continue

    df["stop_id"] = df["stop_id"].astype(str)
    df = df[df["stop_id"].isin(VALID_NODES)]
    if df.empty:
        continue

    df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")
    df = df.dropna(subset=["stop_time"])
    df["hour"] = df["stop_time"].dt.hour

    df["n-boardings"] = pd.to_numeric(df["n-boardings"], errors="coerce")
    df["n-alighting"] = pd.to_numeric(df["n-alighting"], errors="coerce")
    df["lag_loading"] = pd.to_numeric(df["lag_loading"], errors="coerce")

    df["alight_frac"] = np.where(
        df["lag_loading"] > 0,
        (df["n-alighting"] / df["lag_loading"]).clip(0.0, 1.0),
        np.nan,
    )

    # --- (rota, direção, parada, hora) ---
    g_rsh = df.groupby(["route_short_name", "direction_id", "stop_id", "hour"], observed=True)
    board_route_stop_hour.add(g_rsh["n-boardings"].agg(["sum", "count"]))
    alight_route_stop_hour.add(g_rsh["alight_frac"].agg(["sum", "count"]))  # pandas ignora NaN em sum/count

    # --- (rota, direção, parada), todas as horas ---
    g_rs = df.groupby(["route_short_name", "direction_id", "stop_id"], observed=True)
    board_route_stop.add(g_rs["n-boardings"].agg(["sum", "count"]))
    alight_route_stop.add(g_rs["alight_frac"].agg(["sum", "count"]))

    # --- (parada, hora), pool entre rotas ---
    g_sh = df.groupby(["stop_id", "hour"], observed=True)
    board_stop_hour.add(g_sh["n-boardings"].agg(["sum", "count"]))
    alight_stop_hour.add(g_sh["alight_frac"].agg(["sum", "count"]))

    # --- parada, pool entre rotas e horas ---
    g_s = df.groupby(["stop_id"], observed=True)
    board_stop.add(g_s["n-boardings"].agg(["sum", "count"]))
    alight_stop.add(g_s["alight_frac"].agg(["sum", "count"]))

    # --- globais (dia inteiro, todas as rotas/paradas) ---
    board_global_sum += float(df["n-boardings"].sum())
    board_global_cnt += int(df["n-boardings"].count())
    alight_global_sum += float(df["alight_frac"].sum())
    alight_global_cnt += int(df["alight_frac"].count())

    # --- intervalo entre visitas consecutivas (rota, direção, parada) ---
    g_iv = df.sort_values("stop_time").groupby(["route_short_name", "direction_id", "stop_id"])["stop_time"]
    for key, sub in g_iv:
        if len(sub) < 2:
            continue
        gaps = sub.diff().dt.total_seconds().dropna()
        gaps = gaps[(gaps >= MIN_INTERVISIT_SECONDS) & (gaps <= MAX_INTERVISIT_SECONDS)]
        if gaps.empty:
            continue
        s, c = float(gaps.sum()), int(len(gaps))
        intervisit_route_stop.sums[key] += s
        intervisit_route_stop.counts[key] += c
        intervisit_stop.sums[key[2]] += s
        intervisit_stop.counts[key[2]] += c
        intervisit_global_sum += s
        intervisit_global_cnt += c

print("\n💾 Agregando estatísticas finais...")

by_route_stop_hour = {}
for key in board_route_stop_hour.keys():
    by_route_stop_hour[key] = {
        "mean_boardings": board_route_stop_hour.mean(key),
        "mean_alight_frac": alight_route_stop_hour.mean(key),
        "count": board_route_stop_hour.count(key),
    }

by_route_stop = {}
for key in intervisit_route_stop.keys():
    by_route_stop[key] = {
        "mean_boardings": board_route_stop.mean(key),
        "mean_alight_frac": alight_route_stop.mean(key),
        "mean_intervisit_sec": intervisit_route_stop.mean(key),
        "count": intervisit_route_stop.count(key),
    }

by_stop_hour = {}
for key in board_stop_hour.keys():
    by_stop_hour[key] = {
        "mean_boardings": board_stop_hour.mean(key),
        "mean_alight_frac": alight_stop_hour.mean(key),
        "count": board_stop_hour.count(key),
    }

by_stop = {}
all_stops = set(board_stop.keys()) | set(intervisit_stop.keys())
for stop_id in all_stops:
    by_stop[stop_id] = {
        "mean_boardings": board_stop.mean(stop_id),
        "mean_alight_frac": alight_stop.mean(stop_id),
        "mean_intervisit_sec": intervisit_stop.mean(stop_id),
        "count": board_stop.count(stop_id),
    }

global_stats = {
    "mean_boardings": (board_global_sum / board_global_cnt) if board_global_cnt else 1.0,
    "mean_alight_frac": (alight_global_sum / alight_global_cnt) if alight_global_cnt else 0.2,
    "mean_intervisit_sec": (intervisit_global_sum / intervisit_global_cnt) if intervisit_global_cnt else 600.0,
}

print(f"   by_route_stop_hour: {len(by_route_stop_hour)} buckets")
print(f"   by_route_stop:      {len(by_route_stop)} buckets")
print(f"   by_stop_hour:       {len(by_stop_hour)} buckets")
print(f"   by_stop:            {len(by_stop)} buckets")
print(f"   global:             {global_stats}")

# ============================================================
# Remapear níveis by_route_* de (route_short_name, direction_id) -> trip_id
# ============================================================
print("\n🔁 Remapeando níveis por rota para trip_id (via route_metadata.pkl)...")

# Agrupa por (route_short_name, direction_id) uma única vez, para evitar
# varrer todos os buckets para cada um dos ~668 trip_ids (O(n) em vez de O(n*m)).
grouped_stop_hour = defaultdict(dict)   # (rsn, did) -> {(stop_id, hour): stats}
grouped_stop = defaultdict(dict)        # (rsn, did) -> {stop_id: stats}

for (r, d, stop_id, hour), stats in by_route_stop_hour.items():
    grouped_stop_hour[(r, d)][(stop_id, hour)] = stats

for (r, d, stop_id), stats in by_route_stop.items():
    grouped_stop[(r, d)][stop_id] = stats

final_by_route_stop_hour = {}
final_by_route_stop = {}

for trip_id, meta in route_metadata.items():
    rsn = meta.get("route_short_name")
    did = meta.get("direction_id")
    if rsn is None or did is None:
        continue

    for (stop_id, hour), stats in grouped_stop_hour.get((rsn, did), {}).items():
        final_by_route_stop_hour[(trip_id, stop_id, hour)] = stats

    for stop_id, stats in grouped_stop.get((rsn, did), {}).items():
        final_by_route_stop[(trip_id, stop_id)] = stats

print(f"   final_by_route_stop_hour: {len(final_by_route_stop_hour)} entradas")
print(f"   final_by_route_stop:      {len(final_by_route_stop)} entradas")

output = {
    "by_route_stop_hour": final_by_route_stop_hour,
    "by_route_stop": final_by_route_stop,
    "by_stop_hour": by_stop_hour,
    "by_stop": by_stop,
    "global": global_stats,
    "min_bucket_count": MIN_BUCKET_COUNT,
}

out_file = os.path.join(OUTPUT_PATH, "stop_passenger_flow.pkl")
with open(out_file, "wb") as f:
    pickle.dump(output, f)

print(f"\n✅ Estatísticas de fluxo de passageiros salvas em: {out_file}")
