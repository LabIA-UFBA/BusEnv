import pickle
import numpy as np

file_path = 'output_obs/avg_travel_time_AB.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Conteúdo do arquivo '{file_path}':\n")

    if isinstance(data, dict):
        values = list(data.values())
        print(f"Número de pares: {len(values)}")
        print(f"Tempo médio de viagem: {np.mean(values):.2f}")
        print(f"Tempo mínimo: {np.min(values):.2f}")
        print(f"Tempo máximo: {np.max(values):.2f}")
        print(f"Desvio padrão: {np.std(values):.2f}")

        # Se quiser ver os pares com maiores tempos
        print("\nTop 5 maiores tempos:")
        for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  {k}: {v:.2f}")

    else:
        print(f"Tipo de dado não esperado: {type(data)}")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo pickle: {e}")
