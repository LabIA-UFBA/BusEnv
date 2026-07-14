import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "MAPPO"      # IA2C | IPPO | MAPPO

POLICY_COLORS = {
    "Balanced":   "#1f77b4",   # Azul
    "Occupancy":  "#2ca02c",   # Verde
    "Uptime":     "#ff7f0e",   # Laranja
    "Sync":       "#d62728",   # Vermelho
    "Efficiency": "#9467bd"    # Roxo
}

# ==========================================================
# Caminhos
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

csv_path = os.path.join(
    PROJECT_ROOT,
    "paretoMeans",
    "policy_vectors.csv"
)

output_dir = os.path.join(
    PROJECT_ROOT,
    "paretoMeans"
)

# ==========================================================
# Leitura
# ==========================================================

df = pd.read_csv(csv_path)

subset = df[df["policy"].str.startswith(ALGORITHM)].copy()

if subset.empty:
    raise ValueError(f"Nenhuma política encontrada para {ALGORITHM}")

# ==========================================================
# Plot
# ==========================================================

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

for _, row in subset.iterrows():

    full_name = row["policy"]

    policy_name = (
        full_name
        .replace(f"{ALGORITHM}_", "")
        .replace("_means", "")
    )

    color = POLICY_COLORS.get(policy_name, "black")

    ax.scatter(
        row["occupancy"],
        row["uptime"],
        row["efficiency"],
        s=120,
        color=color
    )

    ax.text(
        row["occupancy"],
        row["uptime"],
        row["efficiency"],
        policy_name,
        fontsize=9
    )

ax.set_xlabel("Occupancy")
ax.set_ylabel("Uptime")
ax.set_zlabel("Efficiency")

ax.set_title(f"{ALGORITHM} - Policy Distribution")

plt.tight_layout()

output_path = os.path.join(
    output_dir,
    f"{ALGORITHM.lower()}_pareto_3d.png"
)

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Gráfico salvo em:\n{output_path}")