import pandas as pd
import os

# Caminho da pasta principal do dataset
base_path = "SUNT/tmp"

# Caminho do arquivo de saída
output_path = "visao_geral_dataset.txt"
output_dir = os.path.dirname(output_path)

# Garante que o diretório de saída exista
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dicionário com as pastas e arquivos a serem analisados
datasets = {
    "AFC - Bilhetagem Automática": ("AFC", "afc-2024-03-01.parquet"),
    "AVL - Localização Automática de Veículos": ("AVL", "avl-lines-2024-03-01.parquet"),
    "LTI - Itinerário Linha-Viagem": ("LTI", "lti-2024-07-12.parquet"),
    "Boarding - Embarques Estimados": ("Boarding", "boarding-2024-03-01.parquet"),
    "Alighting - Desembarques Estimados": ("Alighting", "alighting-2024-03-01.parquet"),
    "OD - Origem-Destino Estimada": ("OD", "od-2024-03-01.parquet"),
}

print("Iniciando a leitura dos arquivos Parquet...")

with open(output_path, "w", encoding="utf-8") as out:
    for titulo, (pasta, arquivo) in datasets.items():
        out.write("=" * 50 + "\n")
        out.write(f"{titulo.center(50)}\n")
        out.write("=" * 50 + "\n")
        file_path = os.path.join(base_path, pasta, arquivo)
        try:
            df = pd.read_parquet(file_path)
            out.write(f"Arquivo lido: {file_path}\n\n")
            out.write("--- INFORMAÇÕES GERAIS ---\n")
            out.write(f"Total de registros: {len(df)}\n")
            out.write(f"Colunas: {df.columns.tolist()}\n\n")
            out.write("--- 5 PRIMEIROS REGISTROS ---\n")
            out.write(df.head().to_string())
            out.write("\n\n")
        except FileNotFoundError:
            out.write(f"ERRO: Arquivo não encontrado em '{file_path}'. Verifique o caminho.\n\n")
        except Exception as e:
            out.write(f"Ocorreu um erro ao ler o arquivo {titulo}: {e}\n\n")

print(f"✅ Análise exportada com sucesso para: {output_path}")
