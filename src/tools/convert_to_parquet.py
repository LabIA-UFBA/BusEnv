"""Convert progress.csv files from wesley's BusEnv/1runs into parquet, one per seed run.

Only the 5 algorithms and 3 approaches relevant to this analysis are converted.
"""
import glob
import os

import pandas as pd

SRC_ROOT = # Data root with RL execution data
DST_ROOT = # Data root for parquet output

ALGOS = ["ia2c", "maa2c", "itrpo", "matrpo", "hatrpo"]

# folder name -> (algo_dir_suffix, separator-tolerant glob for the algo dir)
APPROACHES = {
    "baseline": "BASELINE",
    "timesfm": "TIMESFM",
    "tfm_Prev3": "TFM-PREV3",
}


def find_algo_dir(approach_folder, algo, suffix):
    pattern = os.path.join(SRC_ROOT, approach_folder, f"{algo}[-_]{suffix}")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return matches[0]


def main():
    converted = 0
    missing = []

    for algo in ALGOS:
        for approach_folder, suffix in APPROACHES.items():
            algo_dir = find_algo_dir(approach_folder, algo, suffix)
            if algo_dir is None:
                missing.append((algo, approach_folder))
                continue

            seed_dirs = sorted(glob.glob(os.path.join(algo_dir, "*-[1-5]")))
            out_dir = os.path.join(DST_ROOT, algo, approach_folder)
            os.makedirs(out_dir, exist_ok=True)

            for seed_dir in seed_dirs:
                csv_path = os.path.join(seed_dir, "progress.csv")
                if not os.path.isfile(csv_path):
                    continue

                seed_num = seed_dir.rsplit("-", 1)[-1]
                df = pd.read_csv(csv_path, low_memory=False)
                out_path = os.path.join(out_dir, f"seed_{seed_num}.parquet")
                df.to_parquet(out_path, index=False)
                converted += 1

    print(f"Converted {converted} progress.csv files to parquet under {DST_ROOT}")
    if missing:
        print("Missing algo/approach combos:")
        for algo, approach in missing:
            print(f"  - {algo} / {approach}")


if __name__ == "__main__":
    main()
