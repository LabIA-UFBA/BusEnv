import pickle

# Caminho onde os arquivos foram salvos
pkl_path = "./output_obs"

# Carrega os arquivos
with open(f"{pkl_path}/rotas_reais.pkl", "rb") as f:
    rotas_reais = pickle.load(f)

with open(f"{pkl_path}/rotas_metadata.pkl", "rb") as f:
    rotas_metadata = pickle.load(f)

# Exibe algumas rotas para conferência
print("\n📌 Amostra de rotas reais:")
for i, (trip_id, stops) in enumerate(rotas_reais.items()):
    print(f"\n🚌 Trip ID: {trip_id}")
    print(f"Stops: {stops}")
    
    meta = rotas_metadata.get(trip_id, {})
    print("Metadata:", meta)

    if i >= 4:  # Limita para mostrar só as 5 primeiras
        break
