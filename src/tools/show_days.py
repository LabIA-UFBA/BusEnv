import os, pickle, pandas as pd

daily_data_path = "/mnt/ssd1/wesley/BusEnv/src/training_observation/daily"
climate_data_path = "/mnt/ssd1/wesley/BusEnv/src/training_observation/climate_data/climate_data_2.pkl"

obras_windows = {
    "batatinha":     ("2024-01-01", "2024-08-27"),
    "duda_mendonca": ("2024-01-01", "2025-03-28"),
}

daily_files = sorted([f for f in os.listdir(daily_data_path) if f.startswith("daily_data_") and f.endswith(".pkl")])
dates = [f.replace("daily_data_", "").replace(".pkl", "") for f in daily_files]

with open(climate_data_path, "rb") as f:
    climate_data = pickle.load(f)

def day_has_rain(date_str):
    date_fmt = date_str.replace("-", "/")
    rows = climate_data[climate_data["date"] == date_fmt]
    return bool((rows["precip"] > 0.0).any()) if not rows.empty else False

def day_in_obra(date_str, obra_name):
    start, end = obras_windows[obra_name]
    return start <= date_str <= end

rows = []
for d in dates:
    rows.append({
        "date": d,
        "rain": day_has_rain(d),
        "batatinha": day_in_obra(d, "batatinha"),
        "duda_mendonca": day_in_obra(d, "duda_mendonca"),
    })

df = pd.DataFrame(rows)
print(f"Total de dias no dataset: {len(df)}")
print(f"Dias com chuva: {df.rain.sum()} ({100*df.rain.mean():.1f}%)")
print(f"Dias na janela Batatinha: {df.batatinha.sum()} ({100*df.batatinha.mean():.1f}%)")
print(f"Dias na janela Duda Mendonça: {df.duda_mendonca.sum()} ({100*df.duda_mendonca.mean():.1f}%)")
print(f"Dias com chuva E alguma obra ativa: {(df.rain & (df.batatinha | df.duda_mendonca)).sum()}")
df.to_csv("context_day_coverage.csv", index=False)