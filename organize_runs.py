import os
import shutil
import glob


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

# Pasta raiz do projeto BusEnv
BASE_DIR = os.path.abspath(".")

# Pasta onde estão os CSVs
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

# Pasta do experimento
TARGET_ROOT = os.path.join(
    BASE_DIR,
    "runs_lexicografico",
    "RUN_OCC_MAX_WEIGHT"
)

# ==========================================================
# IMPORTANTE
# ==========================================================
# True  -> apenas mostra o que seria movido
# False -> executa os movimentos
DRY_RUN = False


# Algoritmos e prefixos das pastas de treinamento
ALGORITHMS = {
    "COMA": "coma_mlp_sunt_bus",
    "HAPPO": "happo_mlp_sunt_bus",
    "HATRPO": "hatrpo_mlp_sunt_bus",
    "IA2C": "ia2c_mlp_sunt_bus",
    "IPPO": "ippo_mlp_sunt_bus",
    "ITRPO": "itrpo_mlp_sunt_bus",
    "MAA2C": "maa2c_mlp_sunt_bus",
    "MAPPO": "mappo_mlp_sunt_bus",
    "MATRPO": "matrpo_mlp_sunt_bus",
}


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def move_item(src, dst):
    """
    Move um arquivo ou pasta.
    """

    print(f"    {src}")
    print(f" -> {dst}")

    if not DRY_RUN:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def find_algorithm_runs(algorithm, prefix):
    """
    Encontra todas as pastas de treinamento pertencentes
    ao algoritmo.

    Exemplo HAPPO:
        happo_mlp_sunt_bus_..._seed-0
        happo_mlp_sunt_bus_..._seed-1
        happo_mlp_sunt_bus_..._seed-3
        happo_mlp_sunt_bus_..._seed-5
    """

    runs = []

    for name in os.listdir(BASE_DIR):

        path = os.path.join(BASE_DIR, name)

        if not os.path.isdir(path):
            continue

        # Não procurar dentro das pastas de resultados
        if name in [
            "metrics",
            "runs_lexicografico",
            "logs",
            "results",
            "MARLlib"
        ]:
            continue

        if name.lower().startswith(prefix.lower()):
            runs.append(path)

    return sorted(runs)


def find_algorithm_metrics(algorithm):
    """
    Encontra os CSVs do algoritmo.

    Exemplo:
        episode_metrics_coma_run33.csv
        episode_metrics_coma_run34.csv
        episode_metrics_coma_run35.csv
    """

    pattern = os.path.join(
        METRICS_DIR,
        f"episode_metrics_{algorithm.lower()}_run*.csv"
    )

    return sorted(glob.glob(pattern))


# ==========================================================
# ORGANIZAÇÃO
# ==========================================================

def organize_algorithm(algorithm, prefix):

    target_dir = os.path.join(
        TARGET_ROOT,
        f"{algorithm}_OCC_MAX_WEIGHT"
    )

    print("\n" + "=" * 60)
    print(f" {algorithm}")
    print("=" * 60)

    # ------------------------------------------------------
    # Cria pasta do algoritmo
    # ------------------------------------------------------

    print(f"\nPasta destino:")
    print(f"  {target_dir}")

    if not DRY_RUN:
        os.makedirs(target_dir, exist_ok=True)

    # ------------------------------------------------------
    # Encontrar runs
    # ------------------------------------------------------

    runs = find_algorithm_runs(
        algorithm,
        prefix
    )

    print(f"\nRuns encontradas: {len(runs)}")

    for run in runs:

        run_name = os.path.basename(run)

        destination = os.path.join(
            target_dir,
            run_name
        )

        move_item(
            run,
            destination
        )

    # ------------------------------------------------------
    # Encontrar métricas
    # ------------------------------------------------------

    metrics = find_algorithm_metrics(
        algorithm
    )

    print(f"\nCSVs encontrados: {len(metrics)}")

    for metric in metrics:

        metric_name = os.path.basename(metric)

        destination = os.path.join(
            target_dir,
            metric_name
        )

        move_item(
            metric,
            destination
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print(" ORGANIZADOR - RUN_OCC_MAX_WEIGHT")
    print("=" * 60)

    print(f"\nBASE_DIR:")
    print(f"  {BASE_DIR}")

    print(f"\nMETRICS_DIR:")
    print(f"  {METRICS_DIR}")

    print(f"\nTARGET_ROOT:")
    print(f"  {TARGET_ROOT}")

    print(f"\nDRY_RUN: {DRY_RUN}")

    # ------------------------------------------------------
    # Verificações
    # ------------------------------------------------------

    if not os.path.exists(METRICS_DIR):
        print("\nERRO: pasta metrics não encontrada.")
        return

    if not os.path.exists(TARGET_ROOT):
        if DRY_RUN:
            print("\nA pasta destino ainda não existe.")
        else:
            os.makedirs(TARGET_ROOT, exist_ok=True)

    # ------------------------------------------------------
    # Processa todos os algoritmos
    # ------------------------------------------------------

    for algorithm, prefix in ALGORITHMS.items():

        organize_algorithm(
            algorithm,
            prefix
        )

    print("\n" + "=" * 60)
    print(" ORGANIZAÇÃO CONCLUÍDA")
    print("=" * 60)

    if DRY_RUN:
        print("\n⚠️ DRY_RUN está ativado.")
        print("Nada foi movido.")
        print("Altere DRY_RUN = False para executar.")


if __name__ == "__main__":
    main()