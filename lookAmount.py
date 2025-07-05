import pickle
import pandas as pd

# Caminho do arquivo de saída
output_path = "visao_geral_dataset.txt"

with open(output_path, "w", encoding="utf-8") as out:

    # Pairs de paradas
    with open('./output/combined_sum_amount.pkl', 'rb') as f:
        data = pickle.load(f)

    print("Total de pares:", len(data), file=out)
    print("\nExemplos dos primeiros 5 pares:\n", list(data.items())[:5], file=out)

    # Boarding
    df_boarding = pd.read_parquet('./sunt/2024-03-02/output/03_boarding_02-03-2024_02-03-2024.parquet')
    print("\n--- BOARDING ---", file=out)
    print("Colunas:", df_boarding.columns.tolist(), file=out)
    print("\nPrimeiras 5 linhas:\n", df_boarding.head(), file=out)
    #print("\nTodas as linhas:\n", df_boarding.to_string(index=False), file=out) # Descomente para imprimir todas as linhas

    # Landing
    df_landing = pd.read_parquet('./sunt/2024-03-02/output/04_landing_02-03-2024_02-03-2024.parquet')
    print("\n--- LANDING ---", file=out)
    print("Colunas:", df_landing.columns.tolist(), file=out)
    print("\nPrimeiras 5 linhas:\n", df_landing.head(), file=out)
    #print("\nTodas as linhas:\n", df_landing.to_string(index=False), file=out) # Descomente para imprimir todas as linhas

    # Trips
    df_trips = pd.read_parquet('./sunt/2024-03-02/output/trips_time-series_02-03-2024_02-03-2024.parquet')
    print("\n--- TRIPS ---", file=out)
    print("Colunas:", df_trips.columns.tolist(), file=out)
    print("\nPrimeiras 5 linhas:\n", df_trips.head(), file=out)
    # print("\nTodas as linhas:\n", df_trips.to_string(index=False), file=out) # Descomente para imprimir todas as linhas

    # Top 10 paradas com mais embarques
    boarding_by_stop = df_boarding.groupby('stop_id').size().sort_values(ascending=False)
    print("\n--- TOP 10 PARADAS COM MAIS EMBARQUES ---", file=out)
    print(boarding_by_stop.head(10), file=out)

print(f"\n✅ Exportado para: {output_path}")
