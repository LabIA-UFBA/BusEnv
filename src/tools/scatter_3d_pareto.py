import os
import glob
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projection="3d")


# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "ITRPO"      # SELECIONAR ALGORITMO

# Colunas de objetivo a serem lidas do episode_metrics_*.csv
# e comparadas entre os contextos. Se o CSV tiver outras
# colunas de objetivo, basta adicionar o nome aqui — o plot
# 3D sempre usa as 3 primeiras da lista.
OBJECTIVE_COLUMNS = [
    "occupancy",
    "sync",
    "efficiency"
]

# Prefixos de pastas dentro de runs_lexicografico que NÃO são
# contextos de treino (são saídas de gráficos já geradas antes)
IGNORED_PREFIXES = ("graficos",)


# ==========================================================
# Caminhos
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

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
# (HIBRID, RUN_OCC_MAX, RUN_OCC_MAX_WEIGHT, WEIGHT, etc.).
#
# Substitui a lista EXPERIMENTS fixa de antes, que só tinha
# 5 pastas — faltavam as variações _WEIGHT e a WEIGHT.
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


EXPERIMENTS = find_context_dirs(RUNS_DIR)

print(f"Contextos encontrados: {EXPERIMENTS}")


# ==========================================================
# Função para encontrar os CSVs
# ==========================================================

def find_csvs(base_dir, experiment_type, algorithm):

    experiment_dir = os.path.join(
        base_dir,
        experiment_type
    )

    # Procura SOMENTE:
    #
    # episode_metrics_ia2c_*.csv
    # episode_metrics_maa2c_*.csv
    # episode_metrics_mappo_*.csv
    #
    # Ignora progress.csv, env_metrics.csv etc.
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

    return sorted(csvs)


# ==========================================================
# Função para calcular a média dos objetivos
# ==========================================================

def calculate_mean(csvs, experiment_type):

    if not csvs:
        # Antes isso derrubava o script inteiro com um
        # ValueError. Agora só avisa e pula o contexto.
        print(
            f"[AVISO] Nenhum CSV encontrado para "
            f"{ALGORITHM}_{experiment_type}"
        )
        return None

    print(f"\n{ALGORITHM}_{experiment_type}")
    print(f"CSV(s) encontrados: {len(csvs)}")

    run_means = []

    for csv_path in csvs:

        print(f"  - {csv_path}")

        df = pd.read_csv(csv_path)

        # Verifica se o CSV possui as colunas necessárias
        for column in OBJECTIVE_COLUMNS:

            if column not in df.columns:

                raise ValueError(
                    f"Coluna '{column}' não encontrada em:\n"
                    f"{csv_path}\n\n"
                    f"Colunas disponíveis:\n"
                    f"{list(df.columns)}"
                )

        # Média desta execução, para cada objetivo
        run_means.append({
            column: df[column].mean()
            for column in OBJECTIVE_COLUMNS
        })

    result = {
        "experiment": experiment_type,
        "num_csvs": len(csvs)
    }

    # Média entre as execuções, para cada objetivo
    for column in OBJECTIVE_COLUMNS:
        result[column] = (
            sum(run[column] for run in run_means)
            / len(run_means)
        )

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

    result = calculate_mean(
        csvs,
        experiment
    )

    if result is not None:
        results.append(result)


if not results:
    raise ValueError(
        f"Nenhum dado de '{ALGORITHM}' foi encontrado em "
        f"nenhum contexto dentro de:\n{RUNS_DIR}"
    )


# ==========================================================
# Mostrar resultados
# ==========================================================

print("\n" + "=" * 75)
print(f"RESULTADOS - {ALGORITHM}")
print("=" * 75)

for result in results:

    objectives_str = " | ".join(
        f"{column.capitalize()}: {result[column]:.4f}"
        for column in OBJECTIVE_COLUMNS
    )

    print(
        f"{result['experiment']:24s} | "
        f"{objectives_str} | "
        f"CSV: {result['num_csvs']}"
    )


# ==========================================================
# Salvar resultados em CSV
# ==========================================================

results_df = pd.DataFrame(results)

results_csv = os.path.join(
    OUTPUT_DIR,
    f"{ALGORITHM.lower()}_objectives_by_context_3d.csv"
)

results_df.to_csv(results_csv, index=False)

print(f"\nTabela salva em:\n{results_csv}")


# ==========================================================
# Plot 3D
#
# Usa as 3 primeiras colunas de OBJECTIVE_COLUMNS. Se você
# adicionar uma 4ª, ela fica de fora do gráfico (mas continua
# na tabela/CSV) — um 4º eixo não dá pra plotar diretamente
# num scatter 3D; nesse caso dá pra usar cor ou tamanho do
# ponto para representá-lo, me avise se quiser isso.
# ==========================================================

if len(OBJECTIVE_COLUMNS) < 3:

    print(
        "\n[AVISO] É necessário pelo menos 3 colunas em "
        "OBJECTIVE_COLUMNS para gerar o scatter 3D. Pulando "
        "o gráfico."
    )

else:

    x_col, y_col, z_col = OBJECTIVE_COLUMNS[:3]

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    for result in results:

        ax.scatter(
            result[x_col],
            result[y_col],
            result[z_col],
            s=140,
            label=result["experiment"]
        )

        ax.text(
            result[x_col],
            result[y_col],
            result[z_col],
            result["experiment"],
            fontsize=9
        )

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.set_zlabel(z_col.capitalize())

    ax.set_title(f"{ALGORITHM} - Comparação dos Contextos")

    ax.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{ALGORITHM.lower()}_experimentos_comparacao_3d.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nGráfico salvo em:\n{output_path}")