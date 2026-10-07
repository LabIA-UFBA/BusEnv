import os
import glob
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# Configuração
# ==========================================================

ALGORITHM = "MATRPO"      # SELECIONAR ALGORITMO

# Colunas de objetivo a serem lidas do episode_metrics_*.csv
# e comparadas entre os contextos. Se o CSV tiver outras
# colunas de objetivo (ex.: "energy", "uptime"), basta
# adicionar o nome aqui.
OBJECTIVE_COLUMNS = [
    "occupancy",
    "sync"
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
# Isso substitui a lista EXPERIMENTS fixa de antes, que só
# tinha 5 pastas — faltavam as variações _WEIGHT e a WEIGHT.
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

    # Procura SOMENTE arquivos:
    # episode_metrics_ia2c_*.csv
    # episode_metrics_maa2c_*.csv
    # episode_metrics_mappo_*.csv
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
        # ValueError. Agora só avisa e pula o contexto —
        # assim um contexto sem dados para esse algoritmo
        # não impede de ver os outros.
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

print("\n" + "=" * 70)
print(f"RESULTADOS - {ALGORITHM}")
print("=" * 70)

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
    f"{ALGORITHM.lower()}_objectives_by_context.csv"
)

results_df.to_csv(results_csv, index=False)

print(f"\nTabela salva em:\n{results_csv}")


# ==========================================================
# Plot
#
# Scatter usando as duas primeiras colunas de
# OBJECTIVE_COLUMNS, um ponto por contexto. Se você adicionar
# um terceiro objetivo à lista, ele fica de fora do gráfico
# (mas continua na tabela/CSV) — me avise se quiser um tipo
# de gráfico diferente para comparar 3+ objetivos ao mesmo
# tempo (ex.: radar chart, ou um scatter por par de eixos).
# ==========================================================

if len(OBJECTIVE_COLUMNS) < 2:
    print(
        "\n[AVISO] É necessário pelo menos 2 colunas em "
        "OBJECTIVE_COLUMNS para gerar o scatter. Pulando o "
        "gráfico."
    )
else:

    x_col, y_col = OBJECTIVE_COLUMNS[0], OBJECTIVE_COLUMNS[1]

    fig, ax = plt.subplots(figsize=(9, 7))

    for result in results:

        ax.scatter(
            result[x_col],
            result[y_col],
            s=140,
            label=result["experiment"]
        )

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.set_title(f"{ALGORITHM} - Comparação dos Contextos")

    ax.legend(title="Contexto")
    ax.grid(alpha=0.3)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{ALGORITHM.lower()}_experimentos_comparacao.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nGráfico salvo em:\n{output_path}")