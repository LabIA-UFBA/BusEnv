import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "IA2C"      # IA2C | IPPO | MAPPO

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

plt.figure(figsize=(8,6))

for _, row in subset.iterrows():

    full_name = row["policy"]

    policy_name = (
        full_name
        .replace(f"{ALGORITHM}_", "")
        .replace("_means", "")
    )

    color = POLICY_COLORS.get(policy_name, "black")

    plt.scatter(
        row["occupancy"],
        row["efficiency"],
        s=140,
        color=color
    )

    plt.text(
        row["occupancy"] + 0.002,
        row["efficiency"],
        policy_name,
        fontsize=9
    )

plt.xlabel("Occupancy")
plt.ylabel("Efficiency")

plt.title(f"{ALGORITHM} - Pareto Projection")

plt.grid(alpha=0.3)

plt.tight_layout()

output_path = os.path.join(
    output_dir,
    f"{ALGORITHM.lower()}_pareto_projection.png"
)

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Gráfico salvo em:\n{output_path}")