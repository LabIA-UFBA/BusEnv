import pandas as pd
import os

# Path to the main dataset folder
base_path = "SUNT/tmp"

# Path to the output file
output_path = "overview.txt"
output_dir = os.path.dirname(output_path)

# Ensure the output directory exists
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary with the folders and files to be analyzed
datasets = {
    "AFC - Bilhetagem Automática": ("AFC", "afc-2024-03-01.parquet"),
    "AVL - Localização Automática de Veículos": ("AVL", "avl-lines-2024-03-01.parquet"),
    "LTI - Itinerário Linha-Viagem": ("LTI", "lti-2024-07-12.parquet"),
    "Boarding - Embarques Estimados": ("Boarding", "boarding-2024-03-01.parquet"),
    "Alighting - Desembarques Estimados": ("Alighting", "alighting-2024-03-01.parquet"),
    "OD - Origem-Destino Estimada": ("OD", "od-2024-03-01.parquet"),
}

print("Starting to read Parquet files...")

with open(output_path, "w", encoding="utf-8") as out:
    for title, (folder, file) in datasets.items():
        out.write("=" * 50 + "\n")
        out.write(f"{title.center(50)}\n")
        out.write("=" * 50 + "\n")
        file_path = os.path.join(base_path, folder, file)
        try:
            df = pd.read_parquet(file_path)
            out.write(f"File read: {file_path}\n\n")
            out.write("--- GENERAL INFORMATION ---\n")
            out.write(f"Total records: {len(df)}\n")
            out.write(f"Columns: {df.columns.tolist()}\n\n")
            out.write("--- 5 FIRST RECORDS ---\n")
            out.write(df.head().to_string())
            out.write("\n\n")
        except FileNotFoundError:
            out.write(f"ERROR: File not found at '{file_path}'. Check the path.\n\n")
        except Exception as e:
            out.write(f"An error occurred while reading the file {title}: {e}\n\n")

print(f"✅ Analysis successfully exported to: {output_path}")
