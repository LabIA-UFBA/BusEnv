import os
import pandas as pd
import pickle
import re
import sys
import pyarrow.parquet as pq

def process_date_folder(date_str: str) -> dict:
    date = date_str.split("-")
    file_path = f"./sunt/{date_str}/output/trips_time-series_{date[2]}-{date[1]}-{date[0]}_{date[2]}-{date[1]}-{date[0]}.parquet"

    if not os.path.exists(file_path):
        print(f"❌ File not found for date: {file_path}")
        return None

    df = pd.read_parquet(file_path)
    time_per_stop = {}

    print(f"📆 Processing {date_str}...")
    i = 0
    for index, row in df.iterrows():
        if index == 0:
            continue

        previous_row = df.loc[index - 1]
        if previous_row["linha_atend"] != row["linha_atend"]:
            continue

        stop_pair = (previous_row["stop_id"], row["stop_id"])
        if stop_pair not in time_per_stop:
            time_per_stop[stop_pair] = [0, 0]

        time_per_stop[stop_pair][0] += row["tempo_total"]
        time_per_stop[stop_pair][1] += 1

        if index / len(df.index) * 100 > i:
            i += 1
            print(f"\r⏳ {i}% of {len(df.index)} rows for {date_str}", end='')

    return time_per_stop

# ==== MAIN FLOW ====
base_path = "./sunt"
output_path = "./output"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Número de dias a processar (opcional)
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Erro: argumento inválido. Use um número, ex: python script.py 5")
    sys.exit(1)

# Pega todas as pastas no formato de data
date_folders = [
    f for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f)) and re.match(r'\d{4}-\d{2}-\d{2}', f)
]
date_folders.sort()

# Se foi especificado um número de dias, corta a lista
if max_days:
    date_folders = date_folders[:max_days]

combined_averages = {}

# Processa cada pasta
for index, date_folder in enumerate(date_folders):
    print(f"\n📌 Ta enterando ??? {index+1}/{len(date_folders)}")

    result = process_date_folder(date_folder)

    if result is None:
        continue

    avg_path = f"{output_path}/averages_{date_folder}.pkl"
    with open(avg_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n💾 Saved results for {date_folder}")

    print(f"➕ Combining {date_folder}...\n")

    for stop_pair, (total_time, count) in result.items():
        if stop_pair not in combined_averages:
            combined_averages[stop_pair] = [0, 0]
        combined_averages[stop_pair][0] += total_time
        combined_averages[stop_pair][1] += count

# Salva os dados combinados 
print("\n📦 Saving combined averages...")
with open(f"{output_path}/combined_sum_amount.pkl", "wb") as f:
    pickle.dump(combined_averages, f)

final_averages = {
    stop_pair: total_time / count
    for stop_pair, (total_time, count) in combined_averages.items()
}
with open(f"{output_path}/combined_averages.pkl", "wb") as f:
    pickle.dump(final_averages, f)

print("✅ Tudo pronto! Resultados salvos.")
