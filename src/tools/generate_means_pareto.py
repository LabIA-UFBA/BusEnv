import os
import glob
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "politicas"
OUTPUT_FOLDER = "paretoMeans"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Look for the CSVs
# ==========================================================

csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

results = []

print(f"Foram encontrados {len(csv_files)} arquivos.\n")

# ==========================================================
# Calculat mean vectors in each politc
# ==========================================================

for file in csv_files:

    df = pd.read_csv(file)

    policy_name = os.path.splitext(os.path.basename(file))[0]

    occ = df["occupancy"].mean()
    uptime = df["uptime"].mean()
    sync = df["sync"].mean()
    eff = df["efficiency"].mean()

    results.append({

        "policy": policy_name,

        "occupancy": occ,
        "uptime": uptime,
        "sync": sync,
        "efficiency": eff

    })

    print(f"{policy_name}")
    print(f" Occupancy : {occ:.4f}")
    print(f" Uptime    : {uptime:.4f}")
    print(f" Sync      : {sync:.4f}")
    print(f" Efficiency: {eff:.4f}")
    print()

# ==========================================================
# Saving final result
# ==========================================================

result_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_FOLDER,
    "policy_vectors.csv"
)

result_df.to_csv(output_file, index=False)

print("===================================")
print("Vetores médios salvos em:")
print(output_file)
print("===================================")