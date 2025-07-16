import pickle
import pandas as pd # Importe pandas se você espera que o conteúdo seja um DataFrame

# 'output/combined_sum_amount.pkl' ou 'output/combined_averages.pkl'
file_path = 'output_obs/avg_travel_time_AB.pkl' 

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Conteúdo do arquivo '{file_path}':\n")
    
    # Dependendo do tipo de dado, você pode visualizá-lo de diferentes formas:
    if isinstance(data, dict):
        print("É um dicionário. Primeiros 5 itens:")
        for i, (key, value) in enumerate(data.items()):
            print(f"  {key}: {value}")
            if i >= 4: # Para não imprimir tudo se for muito grande
                break
        if len(data) > 5:
            print(f"  ...e mais {len(data) - 5} itens.")
        
    elif isinstance(data, pd.DataFrame):
        print("É um DataFrame do Pandas. Primeiras 5 linhas:")
        print(data.head())
        
    elif isinstance(data, list):
        print("É uma lista. Primeiros 5 itens:")
        for i, item in enumerate(data):
            print(f"  {item}")
            if i >= 4:
                break
        if len(data) > 5:
            print(f"  ...e mais {len(data) - 5} itens.")
            
    else:
        print(f"Tipo de dado desconhecido: {type(data)}")
        print(data) # Imprime o dado como está

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo pickle: {e}")