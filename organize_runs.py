import os
import shutil

# ===== CONFIGURAÇÕES =====
BASE_DIR = "exp_results/matrpo_mlp_sunt_bus"

FILTER_KEY = "Trainer_sunt_bus_sunt_bus"

RUN_TYPE = "MATRPO-Rain-Bonus"
WEIGHTS = "(0.5,0.7,0.5,0.6)"

DRY_RUN = False
# =========================


def is_valid_run(path, name):
    return os.path.isdir(path) and FILTER_KEY in name


def extract_time_code(name):
    """
    Extrai o código final HH-MM-SS
    Ex: ..._2026-02-18_09-11-44 -> 09-11-44
    """
    try:
        return name.split("_")[-1]
    except Exception:
        return None


def move_state_files(base_dir, original_run_name, new_run_path):
    """
    Move basic-variant-state e experiment_state
    que tenham o mesmo HH-MM-SS da run
    """
    time_code = extract_time_code(original_run_name)
    if not time_code:
        return

    for fname in os.listdir(base_dir):
        if not (
            fname.startswith("basic-variant-state-")
            or fname.startswith("experiment_state-")
        ):
            continue

        if not fname.endswith(time_code):
            continue

        src = os.path.join(base_dir, fname)
        dst = os.path.join(new_run_path, fname)

        print(f"    ↳ movendo state {fname}")

        if not DRY_RUN:
            shutil.move(src, dst)


def main():
    base_dir = os.path.abspath(BASE_DIR)

    target_root = os.path.join(base_dir, f"{RUN_TYPE} {WEIGHTS}")
    os.makedirs(target_root, exist_ok=True)

    runs = [
        name for name in os.listdir(base_dir)
        if is_valid_run(os.path.join(base_dir, name), name)
    ]

    runs.sort()

    print(f"Encontradas {len(runs)} runs válidas.")

    for idx, run_name in enumerate(runs, start=1):
        src = os.path.join(base_dir, run_name)
        dst_name = f"{RUN_TYPE} {WEIGHTS} - {idx}"
        dst = os.path.join(target_root, dst_name)

        print(f"{src}  -->  {dst}")

        if not DRY_RUN:
            shutil.move(src, dst)

        # 👇 NOVA FUNÇÃO AQUI
        move_state_files(base_dir, run_name, dst)

    print("Organização concluída.")


if __name__ == "__main__":
    main()
