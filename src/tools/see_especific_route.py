import pickle

# Path where files were saved
pkl_path = "your base path"

# Load the files
with open(f"{pkl_path}/real_routes.pkl", "rb") as f:
    real_routes = pickle.load(f)

with open(f"{pkl_path}/route_metadata.pkl", "rb") as f:
    route_metadata = pickle.load(f)

# =====================================================
# Trip that you want to see the data
# =====================================================
trip_to_show = "20002_1320_1"

if trip_to_show in real_routes:
    print(f"\n🚌 Trip ID: {trip_to_show}")
    print(f"Number of stops: {len(real_routes[trip_to_show])}")
    print("Stops:")

    for i, stop in enumerate(real_routes[trip_to_show], start=1):
        print(f"{i:02d} -> {stop}")

    print("\nMetadata:")
    print(route_metadata.get(trip_to_show, {}))

else:
    print(f"❌ Trip '{trip_to_show}' not found.")