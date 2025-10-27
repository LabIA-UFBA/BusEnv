import os
import pandas as pd

# Caminho base dos seus arquivos
BASE_PATH = "/media/wesley/Disco_local/tes/BusEnv/SUNT/tmp"

# Lista de arquivos a inspecionar (adicione outros se quiser)
FILES = [
    os.path.join(BASE_PATH, "OD", "od-2024-03-01.parquet"),
    os.path.join(BASE_PATH, "OD", "od-2025-03-14.parquet"),
    os.path.join(BASE_PATH, "Boarding", "boarding-2024-03-01.parquet"),
    os.path.join(BASE_PATH, "Boarding", "boarding-2025-03-14.parquet"),
    os.path.join(BASE_PATH, "LTI", "lti-2024-03-01.parquet"),
    os.path.join(BASE_PATH, "LTI", "lti-2025-03-18.parquet"),
]

def log_file_info(path):
    print("=" * 80)
    print(f"📄 FILE: {path}")
    if not os.path.exists(path):
        print("❌ Arquivo não encontrado.\n")
        return

    try:
        df = pd.read_parquet(path)
        print(f"✅ Lido com sucesso.")
        print(f"📊 Total de registros: {len(df):,}")
        print(f"📋 Colunas: {list(df.columns)}\n")

        # Mostra 5 primeiras linhas
        print("--- 5 PRIMEIROS REGISTROS ---")
        print(df.head(5))
        print()
    except Exception as e:
        print(f"⚠️ Erro ao ler {path}: {e}\n")

if __name__ == "__main__":
    print("\n🔍 INSPEÇÃO DE ARQUIVOS PARQUET\n")
    for file_path in FILES:
        log_file_info(file_path)
    print("=" * 80)
    print("✅ Inspeção concluída.")
