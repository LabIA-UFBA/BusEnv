import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "MAPPO"       # IA2C | IPPO | MAPPO e outros

# Cores fixas para cada política
POLICY_COLORS = {
    "Balanced":   "#1f77b4",   # Azul
    "Occupancy":  "#2ca02c",   # Verde
    "Uptime":     "#ff7f0e",   # Laranja
    "Sync":       "#d62728",   # Vermelho
    "Efficiency": "#9467bd"    # Roxo
}

METRICS = [
    "occupancy",
    "uptime",
    "sync",
    "efficiency"
]

# ==========================================================
# Caminhos
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

csv_path = os.path.join(PROJECT_ROOT, "paretoMeans", "policy_vectors.csv")
output_dir = os.path.join(PROJECT_ROOT, "paretoMeans")

# ==========================================================
# Leitura
# ==========================================================

df = pd.read_csv(csv_path)

# Seleciona somente as políticas do algoritmo escolhido
subset = df[df["policy"].str.startswith(ALGORITHM)].copy()

if subset.empty:
    raise ValueError(f"Nenhuma política encontrada para {ALGORITHM}")

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(METRICS))

for _, row in subset.iterrows():

    full_name = row["policy"]

    # Remove "IA2C_" e "_means"
    policy_name = full_name.replace(f"{ALGORITHM}_", "").replace("_means", "")

    plt.plot(
        x,
        row[METRICS].values,
        marker="o",
        linewidth=2.5,
        color=POLICY_COLORS.get(policy_name, "black"),
        label=policy_name
    )

plt.xticks(x, METRICS)

plt.ylim(0, 1)

plt.grid(alpha=0.3)

plt.title(f"{ALGORITHM} - Policy Comparison")

plt.ylabel("Objective Value")

plt.legend(title="Policy")

plt.tight_layout()

output_path = os.path.join(
    output_dir,
    f"{ALGORITHM.lower()}_policy_comparison.png"
)

plt.savefig(output_path, dpi=200)

plt.close()

print(f"Gráfico salvo em:\n{output_path}")