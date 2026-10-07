import os
import re
import glob
import numpy as np
import pandas as pd


# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "MATRPO"      # IA2C | MAA2C | MAPPO | COMA | etc

# Quantidade máxima de CSVs lidos por experimento:
#   None -> lê todos os que encontrar
#   10   -> lê apenas os 10 primeiros (em ordem alfabética/numérica)
MAX_FILES = 10

# Coluna do eixo X para a AUC. Se não existir no CSV,
# usa-se o índice da linha (0, 1, 2, ...).
EPISODE_COLUMN = "episode"


# ==========================================================
# Caminhos
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RUNS_DIR = os.path.join(
    PROJECT_ROOT,
    "runs_lexicografico"
)


# ==========================================================
# Experimentos
# ==========================================================

EXPERIMENTS = [
    "HIBRID",

    "RUN_ENERGY_MAX",
    "RUN_ENERGY_MAX_WEIGHT",

    "RUN_OCC_MAX",
    "RUN_OCC_MAX_WEIGHT",

    "RUN_SYNC_MAX",
    "RUN_SYNC_MAX_WEIGHT",

    "RUN_UPTIME_MAX",
    "RUN_UPTIME_MAX_WEIGHT",

    "WEIGHT"
]


# ==========================================================
# Configuração dos objetivos
# ==========================================================

OBJECTIVES = [
    "occupancy",
    "uptime",
    "sync",
    "efficiency"
]


# ==========================================================
# AUC
# ==========================================================

# np.trapz foi renomeado para np.trapezoid no NumPy 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def compute_auc(x, y):
    """
    Retorna (auc, auc_normalizada).

    - auc: área sob a curva (regra do trapézio).
    - auc_normalizada: auc / (x_max - x_min). Equivale à média
      da curva ponderada pelo eixo X, e permite comparar runs
      com número de episódios diferente.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove pontos com NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return float("nan"), float("nan")

    # Garante ordem crescente do eixo X
    order = np.argsort(x)
    x, y = x[order], y[order]

    auc = float(_trapezoid(y, x))
    span = x[-1] - x[0]

    auc_norm = auc / span if span > 0 else float("nan")

    return auc, auc_norm


# ==========================================================
# Identificar técnica e contexto
# ==========================================================

def identify_technique_context(experiment):

    if experiment == "HIBRID":
        return "HYBRID", "GLOBAL"

    if experiment == "WEIGHT":
        return "WEIGHT", "GLOBAL"

    contexts = {
        "ENERGY": "RUN_ENERGY_MAX",
        "OCC": "RUN_OCC_MAX",
        "SYNC": "RUN_SYNC_MAX",
        "UPTIME": "RUN_UPTIME_MAX"
    }

    for context, base_name in contexts.items():

        if experiment == base_name:
            return "HYBRID", context

        if experiment == f"{base_name}_WEIGHT":
            return "WEIGHT", context

    return "UNKNOWN", "UNKNOWN"


# ==========================================================
# Ordenação natural (números dentro do caminho)
# ==========================================================

def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path)
    ]


# ==========================================================
# Função para encontrar os CSVs
# ==========================================================

def find_csvs(base_dir, experiment_type, algorithm):

    experiment_dir = os.path.join(
        base_dir,
        experiment_type
    )

    pattern = os.path.join(
        experiment_dir,
        f"{algorithm}_*",
        "**",
        f"episode_metrics_{algorithm.lower()}_*.csv"
    )

    csvs = glob.glob(
        pattern,
        recursive=True
    )

    # Ordenação "natural" (run_2 vem antes de run_10), para que
    # os "primeiros N" sejam sempre os mesmos entre execuções
    csvs.sort(key=natural_key)

    total = len(csvs)

    if MAX_FILES is not None:
        csvs = csvs[:MAX_FILES]

    if csvs:
        print(
            f"[INFO] {experiment_type}: {len(csvs)} de {total} "
            f"CSV(s) selecionado(s)"
        )

    return csvs


# ==========================================================
# Calcular média e AUC
# ==========================================================

def calculate_metrics(csvs, experiment_type):

    if not csvs:

        raise ValueError(
            f"Nenhum CSV encontrado para "
            f"{ALGORITHM}_{experiment_type}"
        )

    technique, context = identify_technique_context(
        experiment_type
    )

    print("\n" + "=" * 80)
    print(f"Experimento : {experiment_type}")
    print(f"Técnica     : {technique}")
    print(f"Contexto    : {context}")
    print(f"CSV(s)      : {len(csvs)}")
    print("=" * 80)

    run_metrics = []

    # ======================================================
    # Métricas de cada run
    # ======================================================

    for csv_path in csvs:

        print(f"  - {csv_path}")

        df = pd.read_csv(csv_path)

        for column in OBJECTIVES:

            if column not in df.columns:

                raise ValueError(
                    f"Coluna '{column}' não encontrada em:\n"
                    f"{csv_path}\n\n"
                    f"Colunas disponíveis:\n"
                    f"{list(df.columns)}"
                )

        # Eixo X da AUC
        if EPISODE_COLUMN in df.columns:
            x = df[EPISODE_COLUMN].values
        else:
            x = np.arange(len(df))

        metrics = {"csv": csv_path}

        for column in OBJECTIVES:

            auc, auc_norm = compute_auc(x, df[column].values)

            metrics[f"{column}_mean"] = df[column].mean()
            metrics[f"{column}_auc"] = auc
            metrics[f"{column}_auc_norm"] = auc_norm

        run_metrics.append(metrics)

    # ======================================================
    # Média entre os runs
    # ======================================================

    result = {
        "algorithm": ALGORITHM,
        "technique": technique,
        "context": context,
        "experiment": experiment_type,
        "num_runs": len(csvs)
    }

    for objective in OBJECTIVES:

        for suffix in ("mean", "auc", "auc_norm"):

            key = f"{objective}_{suffix}"

            values = [run[key] for run in run_metrics]

            result[key] = float(np.nanmean(values))

            # Desvio padrão amostral (ddof=1) entre os runs
            if len(values) > 1:
                result[f"{key}_std"] = float(np.nanstd(values, ddof=1))
            else:
                result[f"{key}_std"] = 0.0

    return result


# ==========================================================
# Encontrar e calcular todos os experimentos
# ==========================================================

results = []

for experiment in EXPERIMENTS:

    csvs = find_csvs(
        RUNS_DIR,
        experiment,
        ALGORITHM
    )

    if not csvs:

        print(
            f"\n[WARNING] Nenhum CSV encontrado para "
            f"{ALGORITHM}_{experiment}"
        )

        continue

    results.append(
        calculate_metrics(csvs, experiment)
    )


# ==========================================================
# DataFrame final
# ==========================================================

results_df = pd.DataFrame(results)


# ==========================================================
# Ordenação
# ==========================================================

context_order = [
    "GLOBAL",
    "ENERGY",
    "OCC",
    "SYNC",
    "UPTIME"
]

technique_order = [
    "HYBRID",
    "WEIGHT"
]

results_df["context"] = pd.Categorical(
    results_df["context"],
    categories=context_order,
    ordered=True
)

results_df["technique"] = pd.Categorical(
    results_df["technique"],
    categories=technique_order,
    ordered=True
)

results_df = results_df.sort_values(
    ["context", "technique"]
).reset_index(drop=True)


# ==========================================================
# Exibir resultados
# ==========================================================

def print_table(title, suffix, fmt):

    width = 12 + 12 + 15 * len(OBJECTIVES) + 8

    print("\n")
    print("=" * width)
    print(f"{title} - {ALGORITHM}")
    print("=" * width)

    header = f"{'Contexto':<12}{'Técnica':<12}"

    for objective in OBJECTIVES:
        header += f"{objective.capitalize():>15}"

    header += f"{'Runs':>8}"

    print(header)
    print("-" * width)

    for _, row in results_df.iterrows():

        line = (
            f"{str(row['context']):<12}"
            f"{str(row['technique']):<12}"
        )

        for objective in OBJECTIVES:
            line += f"{row[f'{objective}_{suffix}']:>15{fmt}}"

        line += f"{row['num_runs']:>8}"

        print(line)

    print("-" * width)


print_table("MÉDIA DOS OBJETIVOS POR CONTEXTO E TÉCNICA", "mean", ".4f")
print_table("AUC DOS OBJETIVOS POR CONTEXTO E TÉCNICA", "auc", ".2f")
print_table("AUC NORMALIZADA POR CONTEXTO E TÉCNICA", "auc_norm", ".4f")


# ==========================================================
# Salvar CSV
# ==========================================================

output_file = os.path.join(
    RUNS_DIR,
    f"{ALGORITHM.lower()}_objective_means_auc_by_context.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nResultado salvo em:")
print(output_file)

print("\n[DONE]")