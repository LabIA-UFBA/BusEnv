import os
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

REWARD_COLUMN = "episode_reward_mean"
X_COLUMN = "timesteps_total"

MAX_TIMESTEPS = 200000

# ==========================================================
# CAMINHOS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

RUNS_DIR = os.path.join(
    PROJECT_ROOT,
    "runs_lexicografico"
)

EXPERIMENT = "RUN_OCC_MAX_WEIGHT" # ESCOLHAR AQUI A PASTA QUE VOCE QUER OS GRAFICOS 

EXPERIMENT_DIR = os.path.join(
    RUNS_DIR,
    EXPERIMENT
)

# Onde salvar os gráficos
OUTPUT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "graficos"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# ALGORITMOS
# ==========================================================

ALGORITHMS = [
    "COMA",
    "HAPPO",
    "HATRPO",
    "IA2C",
    "IPPO",
    "ITRPO",
    "MAA2C",
    "MAPPO",
    "MATRPO"
]


# ==========================================================
# Encontrar progress.csv
# ==========================================================

def find_progress_csvs(experiment_dir, algorithm):

    algorithm_dir = os.path.join(
        experiment_dir,
        f"{algorithm}_OCC_MAX_WEIGHT"
    )

    if not os.path.exists(algorithm_dir):
        print(
            f"[AVISO] Pasta não encontrada: "
            f"{algorithm_dir}"
        )
        return []

    pattern = os.path.join(
        algorithm_dir,
        "**",
        "progress.csv"
    )

    csvs = glob.glob(
        pattern,
        recursive=True
    )

    return sorted(csvs)


# ==========================================================
# Carregar UMA curva bruta (ainda sem normalizar)
# ==========================================================

def load_run_curve(csv_path):

    df = pd.read_csv(csv_path)

    for column in (
        REWARD_COLUMN,
        X_COLUMN
    ):

        if column not in df.columns:

            raise ValueError(
                f"\nColuna '{column}' não encontrada em:\n"
                f"{csv_path}\n\n"
                f"Colunas disponíveis:\n"
                f"{list(df.columns)}"
            )

    df = (
        df[
            [
                X_COLUMN,
                REWARD_COLUMN
            ]
        ]
        .dropna()
        .sort_values(X_COLUMN)
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise ValueError(
            f"Nenhum dado válido encontrado em:\n"
            f"{csv_path}"
        )

    return df


# ==========================================================
# Normalização Min-Max
# ==========================================================

def normalize_reward(values, vmin, vmax):

    denominator = vmax - vmin

    if denominator == 0:
        return np.zeros_like(values, dtype=float)

    return np.clip(
        (values - vmin) / denominator,
        0,
        1
    )


# ==========================================================
# 1) CARREGAR TODAS AS CURVAS BRUTAS DE TODOS OS ALGORITMOS
# ==========================================================

raw_curves_by_algorithm = {}

for algorithm in ALGORITHMS:

    csvs = find_progress_csvs(
        EXPERIMENT_DIR,
        algorithm
    )

    if not csvs:
        print(
            f"\n[AVISO] Nenhum progress.csv encontrado "
            f"para {algorithm}"
        )
        continue

    print("\n" + "=" * 80)
    print(algorithm)
    print("=" * 80)
    print(f"Treinos encontrados: {len(csvs)}")

    curves = []

    for csv_path in csvs:
        print(f"  - {csv_path}")
        curves.append(
            load_run_curve(csv_path)
        )

    raw_curves_by_algorithm[algorithm] = curves


# ==========================================================
# 2) MIN E MAX GLOBAIS DA RECOMPENSA
#
#    Calculados sobre TODOS os algoritmos e runs, ANTES de
#    calcular média/AUC/final — assim a tabela e todos os
#    gráficos usam exatamente a mesma escala normalizada.
# ==========================================================

all_rewards = []

for curves in raw_curves_by_algorithm.values():
    for c in curves:
        all_rewards.extend(c[REWARD_COLUMN].values)

if not all_rewards:
    raise ValueError(
        "Nenhum valor de reward encontrado."
    )

reward_min = float(np.min(all_rewards))
reward_max = float(np.max(all_rewards))

print("\n" + "=" * 70)
print("NORMALIZAÇÃO DA REWARD (Min-Max global)")
print("=" * 70)
print(f"Reward mínima global: {reward_min:.4f}")
print(f"Reward máxima global: {reward_max:.4f}")


# ==========================================================
# 3) NORMALIZAR TODAS AS CURVAS
#
#    A partir daqui, REWARD_COLUMN de cada curva já está em
#    escala [0, 1]. Os valores brutos não são mais usados em
#    nenhum lugar do script (tabela, CSV, gráficos de barra
#    e gráfico de curvas usam todos a versão normalizada).
# ==========================================================

normalized_curves_by_algorithm = {}

for algorithm, curves in raw_curves_by_algorithm.items():

    normalized_curves = []

    for c in curves:

        c_norm = c.copy()

        c_norm[REWARD_COLUMN] = normalize_reward(
            c[REWARD_COLUMN].values,
            reward_min,
            reward_max
        )

        normalized_curves.append(c_norm)

    normalized_curves_by_algorithm[algorithm] = normalized_curves


# ==========================================================
# Estatísticas (mean / AUC / final) de UMA curva normalizada
# ==========================================================

def compute_run_reward_stats(df):

    x = df[X_COLUMN].values
    y = df[REWARD_COLUMN].values

    mean_val = float(
        np.mean(y)
    )

    final_val = float(
        y[-1]
    )

    if (
        len(x) >= 2
        and (x.max() - x.min()) > 0
    ):

        if hasattr(np, "trapezoid"):
            auc_raw = float(
                np.trapezoid(y, x=x)
            )
        else:
            auc_raw = float(
                np.trapz(y, x=x)
            )

        auc_norm = (
            auc_raw /
            (x.max() - x.min())
        )

    else:

        auc_norm = mean_val

    return {
        "mean": mean_val,
        "auc": auc_norm,
        "final": final_val,
        "curve": df
    }


# ==========================================================
# Estatísticas de UM algoritmo (entre os diferentes treinos)
# ==========================================================

def calculate_algorithm_stats(curves, algorithm):

    if not curves:
        return None

    run_stats = [
        compute_run_reward_stats(c)
        for c in curves
    ]

    means = [r["mean"] for r in run_stats]
    aucs = [r["auc"] for r in run_stats]
    finals = [r["final"] for r in run_stats]

    return {

        "algorithm": algorithm,
        "num_runs": len(run_stats),

        "mean_of_means": float(np.mean(means)),
        "std_of_means": float(np.std(means)),

        "mean_of_auc": float(np.mean(aucs)),
        "std_of_auc": float(np.std(aucs)),

        "mean_of_final": float(np.mean(finals)),
        "std_of_final": float(np.std(finals)),

        "curves": [r["curve"] for r in run_stats]
    }


# ==========================================================
# CALCULAR TODOS OS ALGORITMOS (já em escala normalizada)
# ==========================================================

results = []

for algorithm in ALGORITHMS:

    curves = normalized_curves_by_algorithm.get(algorithm, [])

    result = calculate_algorithm_stats(
        curves,
        algorithm
    )

    if result is not None:
        results.append(result)


# ==========================================================
# TABELA NO TERMINAL
# ==========================================================

print("\n")
print("=" * 110)
print(
    f"RECOMPENSA NORMALIZADA DE TREINO - {EXPERIMENT}"
)
print("=" * 110)

print(
    f"{'Algoritmo':12s} | "
    f"{'Runs':>5s} | "
    f"{'Média':>14s} | "
    f"{'AUC':>14s} | "
    f"{'Final':>14s}"
)

print("-" * 110)

for r in results:

    print(
        f"{r['algorithm']:12s} | "
        f"{r['num_runs']:5d} | "
        f"{r['mean_of_means']:7.3f} ± {r['std_of_means']:5.3f} | "
        f"{r['mean_of_auc']:7.3f} ± {r['std_of_auc']:5.3f} | "
        f"{r['mean_of_final']:7.3f} ± {r['std_of_final']:5.3f}"
    )


# ==========================================================
# SALVAR RESULTADOS EM CSV
# ==========================================================

results_df = pd.DataFrame([

    {
        "algorithm": r["algorithm"],
        "num_runs": r["num_runs"],

        "mean_reward_norm": r["mean_of_means"],
        "std_reward_norm": r["std_of_means"],

        "mean_auc_norm": r["mean_of_auc"],
        "std_auc_norm": r["std_of_auc"],

        "mean_final_norm": r["mean_of_final"],
        "std_final_norm": r["std_of_final"]
    }

    for r in results
])

results_csv = os.path.join(
    OUTPUT_DIR,
    "reward_auc_comparison_normalized.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

print(
    f"\nTabela salva em:\n"
    f"{results_csv}"
)


# ==========================================================
# GRÁFICOS DE BARRA (média / AUC / final), todos normalizados
# ==========================================================

def bar_chart(names, means, stds, ylabel, title, filename):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        names,
        means,
        yerr=stds,
        capsize=5
    )

    ax.set_xlabel("Algoritmo")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nGráfico salvo em:\n{path}")

    return path


names = [r["algorithm"] for r in results]

bar_chart(
    names,
    [r["mean_of_means"] for r in results],
    [r["std_of_means"] for r in results],
    "Recompensa média normalizada",
    f"{EXPERIMENT} - Recompensa média de treino (normalizada)",
    "reward_mean_comparison_normalized.png"
)

bar_chart(
    names,
    [r["mean_of_auc"] for r in results],
    [r["std_of_auc"] for r in results],
    "AUC normalizada",
    f"{EXPERIMENT} - AUC da curva de treinamento (normalizada)",
    "reward_auc_comparison_normalized.png"
)

bar_chart(
    names,
    [r["mean_of_final"] for r in results],
    [r["std_of_final"] for r in results],
    "Recompensa final normalizada",
    f"{EXPERIMENT} - Recompensa final (normalizada)",
    "reward_final_comparison_normalized.png"
)


# ==========================================================
# FORMATADOR DO EIXO X
# ==========================================================

def format_x_1e5(x, pos):
    if x == 0:
        return "0"
    return f"{int(x / 1e5)}×10⁵"


# ==========================================================
# CURVAS DE TREINAMENTO NORMALIZADAS
#
# X = TIMESTEPS REAIS
# Y = REWARD NORMALIZADA (já normalizada na etapa 3)
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 7))

for r in results:

    curves = r["curves"]

    if not curves:
        continue

    x_min = max(
        c[X_COLUMN].min()
        for c in curves
        if X_COLUMN in c.columns
    )

    x_max = min(
        c[X_COLUMN].max()
        for c in curves
        if X_COLUMN in c.columns
    )

    x_max = min(x_max, MAX_TIMESTEPS)

    if x_max <= x_min:
        print(f"[AVISO] Intervalo inválido para {r['algorithm']}")
        continue

    grid = np.linspace(x_min, x_max, 300)

    interpolated = []

    for c in curves:

        x = pd.to_numeric(c[X_COLUMN], errors="coerce")
        y = pd.to_numeric(c[REWARD_COLUMN], errors="coerce")

        valid = x.notna() & y.notna()

        x = x[valid].values
        y = y[valid].values

        if len(x) < 2:
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        interpolated.append(
            np.interp(grid, x, y)
        )

    if not interpolated:
        print(f"[AVISO] Nenhuma curva válida para {r['algorithm']}")
        continue

    interpolated = np.array(interpolated)

    curve_mean_norm = interpolated.mean(axis=0)
    curve_std_norm = interpolated.std(axis=0)

    lower = np.clip(curve_mean_norm - curve_std_norm, 0, 1)
    upper = np.clip(curve_mean_norm + curve_std_norm, 0, 1)

    ax.plot(
        grid,
        curve_mean_norm,
        label=r["algorithm"],
        linewidth=2
    )

    ax.fill_between(
        grid,
        lower,
        upper,
        alpha=0.15
    )


ax.set_xlim(0, MAX_TIMESTEPS)

ticks = np.arange(0, MAX_TIMESTEPS + 1, 100000)
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(FuncFormatter(format_x_1e5))

ax.set_ylim(0, 1)
ax.set_xlabel("Step")
ax.set_ylabel("Normalized Reward")
ax.set_title(f"{EXPERIMENT} - Curvas de treinamento (normalizadas)")

ax.legend(title="Algorithm")
ax.grid(True, alpha=0.2)

plt.tight_layout()

curve_path = os.path.join(
    OUTPUT_DIR,
    f"{EXPERIMENT.lower()}_reward_curves_normalized.png"
)

plt.savefig(
    curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nGráfico normalizado salvo em:\n{curve_path}")


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)

print(f"\nResultados salvos em:\n{OUTPUT_DIR}")