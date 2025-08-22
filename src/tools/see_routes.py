import pickle

# Path where files were saved
pkl_path = "../training_observation"

# Load the files
with open(f"{pkl_path}/rotas_reais.pkl", "rb") as f:
    real_routes = pickle.load(f)

with open(f"{pkl_path}/rotas_metadata.pkl", "rb") as f:
    route_metadata = pickle.load(f)

# Display some routes for verification
print("\n📌 Sample of real routes:")
for i, (trip_id, stops) in enumerate(real_routes.items()):
    print(f"\n🚌 Trip ID: {trip_id}")
    print(f"Stops: {stops}")

    meta = route_metadata.get(trip_id, {})
    print("Metadata:", meta)

    if i >= 4:  # Limit to show only the first 5
        break
