import os
import re
import sys
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict

# === Main Paths ===
BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "./output_obs"
GRAPH_PATH = "./SUNT/data/graph_designer/graph_gtfs_fev_2024.gpickle" # Path to the GTFS graph

# Create output directory if necessary
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Optional argument to limit number of days ===
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Error: Invalid argument. Please use a number, e.g., python script.py 5")
    sys.exit(1)

# === Load the urban graph ===
print("📦 Loading GTFS graph...")
try:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    valid_nodes = set(G.nodes)
    print(f"✅ Graph loaded with {len(valid_nodes)} valid nodes.")
except Exception as e:
    print(f"❌ Error loading graph: {e}")
    sys.exit(1)

# === Auxiliary function to collect dates ===
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder): # Check if the directory exists
        print(f"❌ Directory not found: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')] # Filter files with the correct prefix and extension
    dates = [] # List to store extracted dates
    for fname in filenames:
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname) # Use regex to extract the date from the filename
        if match:
            dates.append(match.group(1)) # Add the extracted date to the list
    return sorted(dates) # Sort the dates in chronological order

# === Accumulators ===
real_routes = dict() # Dictionary to store real routes
route_metadata = dict() # Dictionary to store route metadata

dates = get_date_list("OD", "od")
if max_days:
    dates = dates[:max_days] # Limit the number of days to process

print(f"\n📅 Processing {len(dates)} days to extract valid real routes...")

for i, date_str in enumerate(dates):
    print(f"\n➡️  {i+1}/{len(dates)} - {date_str}") # Indicate the progress of processing
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet")) # Read the OD file for the current date

        required_columns = ["route_short_name", "direction_id", "vehicle", "trip_number", "stop_id", "pt_sequence"] # Required columns for validation
        if not all(col in df_od.columns for col in required_columns):
            print(f"⚠️ Expected columns missing in OD {date_str}")
            continue

        df_od = df_od.dropna(subset=["stop_id", "trip_number", "route_short_name"]) # Remove records with missing values in critical columns
        df_od["vehicle"] = df_od["vehicle"].astype(str) # Convert vehicle ID to string
        df_od["stop_id"] = df_od["stop_id"].astype(str)
        df_od["trip_number"] = df_od["trip_number"].astype(int)

        df_od["trip_id"] = df_od.apply(lambda x: f"{x['vehicle']}_{x['route_short_name']}_{x['trip_number']}", axis=1) # Create a unique trip ID based on vehicle, route name, and trip number

        grouped = df_od.sort_values(by=["vehicle", "trip_id", "direction_id", "pt_sequence"]) \
                       .groupby("trip_id") # Group data by trip ID

        for trip_id, group in grouped: # Process each group of data by trip
            all_stops = group["stop_id"].tolist()
            filtered_stops = [s for s in all_stops if s in valid_nodes]

            if len(filtered_stops) > 1: # Check if there is more than one valid stop
                if len(filtered_stops) < len(all_stops): # Some stops were removed for not being in the graph
                    missing = set(all_stops) - set(filtered_stops) # Identify missing stops
                    print(f"⚠️ Trip {trip_id}: {len(missing)} stops removed for not being in the graph.") # Inform about removed stops

                real_routes[trip_id] = filtered_stops
                route_metadata[trip_id] = {
                    "route_short_name": group["route_short_name"].iloc[0], # Route name
                    "direction_id": group["direction_id"].iloc[0], # Direction ID
                    "vehicle": group["vehicle"].iloc[0], # Vehicle ID
                    "trip_number": group["trip_number"].iloc[0] # Trip number
                }

    except Exception as e:
        print(f"⚠️ Error processing OD {date_str}: {e}")

# === Finalization ===
with open(os.path.join(OUTPUT_PATH, "real_routes.pkl"), "wb") as f:
    pickle.dump(real_routes, f)

with open(os.path.join(OUTPUT_PATH, "route_metadata.pkl"), "wb") as f:
    pickle.dump(route_metadata, f)

print("\n✅ Files 'real_routes.pkl' and 'route_metadata.pkl' saved successfully with graph validation.")
