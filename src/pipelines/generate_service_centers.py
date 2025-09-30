import os
import re
import pickle
import pandas as pd

# Paths
BASE_PATH = "/media/wesley/Disco_local/tes/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "../training_observation"
GRAPH_PATH = "/media/wesley/Disco_local/tes/graph-exploration/src/viz/graph_gtfs_fev_2024.gpickle"
NODE_FEATURES_PATH = os.path.join(BASE_PATH, "graph", "node_features.parquet")  # <-- Ajuste se precisar

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Config ---
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder):
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')]
    dates = []
    for fname in filenames:
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname)
        if match:
            dates.append(match.group(1))
    return sorted(dates)

dates = get_date_list("LTI", "lti")
print(f"📅 Found {len(dates)} days with LTI data")

# --- Collector ---
service_centers = set()
df_metrics_all = []

for day_idx, date_str in enumerate(dates):
    print(f"\n➡️ {day_idx+1}/{len(dates)} - {date_str}")

    try:
        # Load LTI
        df_lti = pd.read_parquet(os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet"))
        print("📋 LTI columns:", df_lti.columns.tolist())

        # Normalizar nome da coluna de atividade
        for col in df_lti.columns:
            if col.lower().startswith("acti") or "ativ" in col.lower():
                df_lti = df_lti.rename(columns={col: "activity"})
                break

        if "activity" not in df_lti.columns:
            print("⚠️ Nenhuma coluna de atividade encontrada.")
            continue

        # Filtrar apenas Saída de Garagem e Recolhe
        df_lti = df_lti[df_lti["activity"].str.lower().isin(
            ["saída de garagem", "recolhe", "leaving the garage", "returning to the garage"]
        )]
        if df_lti.empty:
            print("   ⚠️ No garage activity this day.")
            continue

        # Guardar pontos de início/fim
        day_nodes = set()
        for col in ["codPontoInicio", "codPontoFim"]:
            if col in df_lti.columns:
                for node in df_lti[col].dropna().astype(str):
                    day_nodes.add(node)
                    service_centers.add(node)

        print(f"   ✓ Found {len(df_lti)} garage activities with {len(day_nodes)} unique nodes.")

        # --- Carregar Graph-based Node Features ---
        gbnf_path = os.path.join(BASE_PATH, "Graph-based Node Features", f"graph_node_features-{date_str}.parquet")
        if os.path.exists(gbnf_path):
            df_nodes = pd.read_parquet(gbnf_path)
            df_nodes["node"] = df_nodes["node"].astype(str)

            # Filtrar só os nós garagem
            df_garage_metrics = df_nodes[df_nodes["node"].isin(day_nodes)].copy()
            df_garage_metrics["date"] = date_str

            df_metrics_all.append(df_garage_metrics)
        else:
            print(f"   ⚠️ Node features file not found for {date_str}")

    except Exception as e:
        print(f"⚠️ Error {date_str}: {e}")

# --- Save results ---
final_centers = sorted(service_centers)
output_pkl = os.path.join(OUTPUT_PATH, "service_centers_nodes.pkl")
with open(output_pkl, "wb") as f:
    pickle.dump(final_centers, f)

print("\n✅ Extraction completed!")
print(f"📍 Total unique service centers found: {len(final_centers)}")
print(f"💾 Saved nodes list at: {output_pkl}")

if df_metrics_all:
    df_metrics_all = pd.concat(df_metrics_all, ignore_index=True)
    output_csv = os.path.join(OUTPUT_PATH, "service_centers_metrics.csv")
    df_metrics_all.to_csv(output_csv, index=False)
    print(f"💾 Saved metrics at: {output_csv}")
else:
    print("⚠️ No metrics were saved because no node features were found.")
