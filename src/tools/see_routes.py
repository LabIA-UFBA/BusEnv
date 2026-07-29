import pickle

# Path where files were saved
pkl_path = "your base path"

# Load the files
with open(f"{pkl_path}/real_routes.pkl", "rb") as f:
    real_routes = pickle.load(f)

with open(f"{pkl_path}/route_metadata.pkl", "rb") as f:
    route_metadata = pickle.load(f)

# Display some routes for verification
print("\n📌 Sample of real routes:")
for i, (trip_id, stops) in enumerate(real_routes.items()):
    print(f"\n🚌 Trip ID: {trip_id}")
    print(f"Stops: {stops}")

    meta = route_metadata.get(trip_id, {})
    print("Metadata:", meta)

    if i >= 10:  # Limit to show only the first 5
        break
