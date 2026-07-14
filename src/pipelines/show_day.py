import pickle
from pprint import pprint

# Caminho do arquivo de um dia
path = "/media/wesley/Disco_local/tes/BusEnv/SUNT/tmp/daily/daily_data_2024-03-01.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print("📅 Dados carregados com sucesso!\n")
print(f"Chaves disponíveis: {list(data.keys())}\n")

# Exibir resumo de cada bloco
for key, val in data.items():
    if isinstance(val, dict):
        print(f"🔹 {key}: {len(val)} entradas")
        # mostra uma amostra
        sample_items = list(val.items())[:3]
        for k, v in sample_items:
            print(f"   {k}: {v}")
        print()
    else:
        print(f"🔸 {key}: tipo {type(val)} -> {val}\n")
