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

# Algoritmo que você quer analisar através de todos os contextos | ALTERAR AQUI
ALGORITHM = "MATRPO"

# Prefixos de pastas dentro de runs_lexicografico que NÃO são
# contextos de treino (são saídas de gráficos já geradas antes)
IGNORED_PREFIXES = ("graficos",)


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

OUTPUT_DIR = os.path.join(
    RUNS_DIR,
    f"graficos_{ALGORITHM.lower()}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# DESCOBRIR OS CONTEXTOS DE TREINO
#
# Qualquer pasta dentro de runs_lexicografico que não comece
# com um dos prefixos ignorados é tratada como um contexto
# (HIBRID, RUN_OCC_MAX, RUN_OCC_MAX_WEIGHT, WEIGHT, etc.)
# ==========================================================

def find_context_dirs(runs_dir):

    contexts = []

    for entry in sorted(os.listdir(runs_dir)):

        full_path = os.path.join(runs_dir, entry)

        if not os.path.isdir(full_path):
            continue

        if entry.lower().startswith(IGNORED_PREFIXES):
            continue

        contexts.append(entry)

    return contexts


# ==========================================================
# Encontrar progress.csv de UM algoritmo dentro de UM contexto
#
# Usa um curinga (algorithm_*) em vez de montar o sufixo do
# contexto na mão, então funciona independente de a pasta se
# chamar "COMA_OCC_MAX_WEIGHT", "COMA_HIBRID", "COMA_WEIGHT",
# etc.
# ==========================================================

def find_progress_csvs(context_dir, algorithm):

    pattern = os.path.join(
        context_dir,
        f"{algorithm}_*",
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
# 1) CARREGAR AS CURVAS BRUTAS DO ALGORITMO ESCOLHIDO EM
#    TODOS OS CONTEXTOS
# ==========================================================

contexts = find_context_dirs(RUNS_DIR)

print(f"Contextos encontrados: {contexts}")

raw_curves_by_context = {}

for context in contexts:

    context_dir = os.path.join(RUNS_DIR, context)

    csvs = find_progress_csvs(context_dir, ALGORITHM)

    if not csvs:
        print(
            f"\n[AVISO] Nenhum progress.csv de {ALGORITHM} "
            f"encontrado em: {context}"
        )
        continue

    print("\n" + "=" * 80)
    print(f"{ALGORITHM} — {context}")
    print("=" * 80)
    print(f"Treinos encontrados: {len(csvs)}")

    curves = []

    for csv_path in csvs:
        print(f"  - {csv_path}")
        curves.append(
            load_run_curve(csv_path)
        )

    raw_curves_by_context[context] = curves


if not raw_curves_by_context:
    raise ValueError(
        f"Nenhum dado de '{ALGORITHM}' foi encontrado em "
        f"nenhum contexto dentro de:\n{RUNS_DIR}"
    )


# ==========================================================
# 2) MIN E MAX GLOBAIS DA RECOMPENSA
#
#    Calculados sobre TODOS os contextos do algoritmo
#    escolhido, ANTES de calcular média/AUC/final — assim
#    a tabela e todos os gráficos usam a mesma escala e os
#    contextos ficam comparáveis entre si em [0, 1].
# ==========================================================

all_rewards = []

for curves in raw_curves_by_context.values():
    for c in curves:
        all_rewards.extend(c[REWARD_COLUMN].values)

reward_min = float(np.min(all_rewards))
reward_max = float(np.max(all_rewards))

print("\n" + "=" * 70)
print(f"NORMALIZAÇÃO DA REWARD DE {ALGORITHM} (Min-Max global)")
print("=" * 70)
print(f"Reward mínima global: {reward_min:.4f}")
print(f"Reward máxima global: {reward_max:.4f}")


# ==========================================================
# 3) NORMALIZAR TODAS AS CURVAS
# ==========================================================

normalized_curves_by_context = {}

for context, curves in raw_curves_by_context.items():

    normalized_curves = []

    for c in curves:

        c_norm = c.copy()

        c_norm[REWARD_COLUMN] = normalize_reward(
            c[REWARD_COLUMN].values,
            reward_min,
            reward_max
        )

        normalized_curves.append(c_norm)

    normalized_curves_by_context[context] = normalized_curves


# ==========================================================
# Estatísticas (mean / AUC / final) de UMA curva normalizada
# ==========================================================

def compute_run_reward_stats(df):

    x = df[X_COLUMN].values
    y = df[REWARD_COLUMN].values

    mean_val = float(np.mean(y))
    final_val = float(y[-1])

    if (
        len(x) >= 2
        and (x.max() - x.min()) > 0
    ):

        if hasattr(np, "trapezoid"):
            auc_raw = float(np.trapezoid(y, x=x))
        else:
            auc_raw = float(np.trapz(y, x=x))

        auc_norm = auc_raw / (x.max() - x.min())

    else:

        auc_norm = mean_val

    return {
        "mean": mean_val,
        "auc": auc_norm,
        "final": final_val,
        "curve": df
    }


# ==========================================================
# Estatísticas de UM contexto (entre os diferentes treinos)
# ==========================================================

def calculate_context_stats(curves, context):

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

        "context": context,
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
# CALCULAR TODOS OS CONTEXTOS (já em escala normalizada)
# ==========================================================

results = []

for context in contexts:

    curves = normalized_curves_by_context.get(context, [])

    result = calculate_context_stats(curves, context)

    if result is not None:
        results.append(result)


# ==========================================================
# TABELA NO TERMINAL
# ==========================================================

print("\n")
print("=" * 110)
print(f"DESEMPENHO NORMALIZADO DE {ALGORITHM} POR CONTEXTO DE TREINO")
print("=" * 110)

print(
    f"{'Contexto':28s} | "
    f"{'Runs':>5s} | "
    f"{'Média':>14s} | "
    f"{'AUC':>14s} | "
    f"{'Final':>14s}"
)

print("-" * 110)

for r in results:

    print(
        f"{r['context']:28s} | "
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
        "context": r["context"],
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
    f"{ALGORITHM.lower()}_performance_by_context_normalized.csv"
)

results_df.to_csv(results_csv, index=False)

print(f"\nTabela salva em:\n{results_csv}")


# ==========================================================
# GRÁFICOS DE BARRA (média / AUC / final), todos normalizados
# ==========================================================

def bar_chart(names, means, stds, ylabel, title, filename):

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(names, means, yerr=stds, capsize=5)

    ax.set_xlabel("Contexto de treino")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nGráfico salvo em:\n{path}")

    return path


names = [r["context"] for r in results]

bar_chart(
    names,
    [r["mean_of_means"] for r in results],
    [r["std_of_means"] for r in results],
    "Recompensa média normalizada",
    f"{ALGORITHM} - Recompensa média por contexto (normalizada)",
    f"{ALGORITHM.lower()}_reward_mean_by_context_normalized.png"
)

bar_chart(
    names,
    [r["mean_of_auc"] for r in results],
    [r["std_of_auc"] for r in results],
    "AUC normalizada",
    f"{ALGORITHM} - AUC por contexto (normalizada)",
    f"{ALGORITHM.lower()}_auc_by_context_normalized.png"
)

bar_chart(
    names,
    [r["mean_of_final"] for r in results],
    [r["std_of_final"] for r in results],
    "Recompensa final normalizada",
    f"{ALGORITHM} - Recompensa final por contexto (normalizada)",
    f"{ALGORITHM.lower()}_reward_final_by_context_normalized.png"
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
# Uma linha por CONTEXTO (mesmo algoritmo em todos)
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
        print(f"[AVISO] Intervalo inválido para {r['context']}")
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
        print(f"[AVISO] Nenhuma curva válida para {r['context']}")
        continue

    interpolated = np.array(interpolated)

    curve_mean_norm = interpolated.mean(axis=0)
    curve_std_norm = interpolated.std(axis=0)

    lower = np.clip(curve_mean_norm - curve_std_norm, 0, 1)
    upper = np.clip(curve_mean_norm + curve_std_norm, 0, 1)

    ax.plot(
        grid,
        curve_mean_norm,
        label=r["context"],
        linewidth=2
    )

    ax.fill_between(grid, lower, upper, alpha=0.15)


ax.set_xlim(0, MAX_TIMESTEPS)

ticks = np.arange(0, MAX_TIMESTEPS + 1, 100000)
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(FuncFormatter(format_x_1e5))

ax.set_ylim(0, 1)
ax.set_xlabel("Step")
ax.set_ylabel("Normalized Reward")
ax.set_title(f"{ALGORITHM} - Curvas de treinamento por contexto (normalizadas)")

ax.legend(title="Contexto")
ax.grid(True, alpha=0.2)

plt.tight_layout()

curve_path = os.path.join(
    OUTPUT_DIR,
    f"{ALGORITHM.lower()}_reward_curves_by_context_normalized.png"
)

plt.savefig(curve_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGráfico normalizado salvo em:\n{curve_path}")


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)

print(f"\nResultados salvos em:\n{OUTPUT_DIR}")