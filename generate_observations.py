import os
import re
import sys
import pickle
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Path to the base directory and output directory
BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "./output_obs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Optional argument to limit number of days
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Error: Invalid argument. Please use a number, e.g., python script.py 5")
    sys.exit(1)

# === Auxiliary Functions ===
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder) # Path to the subfolder
    if not os.path.exists(folder): # Check if the folder exists
        print(f"❌ Directory not found: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')] # List files that match the prefix and have .parquet extension
    dates = []
    for fname in filenames: # Extract dates from filenames
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname) # Match the date pattern
        if match:
            dates.append(match.group(1))
    return sorted(dates) # Return sorted list of dates

def normalize_id(val, max_val=1_000_000): # Normalize ID values to a range between 0 and 1
    return int(val) / max_val if val and val >= 0 else 0.0 # Normalize to 0.0 if value is None or negative

def normalize_time_of_day(dt):
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400 # Normalize time of day to a value between 0 and 1 (24 hours)

# === Accumulators ===
avg_travel_times = defaultdict(lambda: [0.0, 0])   # (total_time, count)
future_demand = defaultdict(lambda: [0, 0])       # (total_boardings, count)
occupancy = defaultdict(lambda: [0.0, 0])         # (sum_occupancy, count)
uptime = defaultdict(lambda: [0.0, 0])            # (sum_uptime_seconds, count)

dates = get_date_list("OD", "od")
if max_days:
    dates = dates[:max_days]

print(f"\n📅 Processing {len(dates)} days...")

for i, date_str in enumerate(dates): # Loop through each date
    print(f"\n➡️  {i+1}/{len(dates)} - {date_str}") # Display current date being processed

    # --- OD: avg_travel_time_AB ---
    try: # Read the OD data for average travel time
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet")) # Check if the file exists
        df_od['stop_time'] = pd.to_datetime(df_od['stop_time']) # Convert stop_time to datetime
        grouped = df_od.sort_values(by=['vehicle', 'trip_id', 'direction_id', 'pt_sequence']) \
                        .groupby(['vehicle', 'trip_id', 'direction_id']) # Group by vehicle, trip_id, and direction_id
        for _, group in grouped: 
            for i in range(1, len(group)): # Calculate travel time between consecutive stops
                prev = group.iloc[i-1] # Previous stop
                curr = group.iloc[i] # Current stop
                delta = (curr['stop_time'] - prev['stop_time']).total_seconds() # Calculate time difference in seconds
                if delta > 0: # Only consider positive travel times
                    pair = (prev['stop_id'], curr['stop_id']) # Create a pair of stop IDs
                    avg_travel_times[pair][0] += delta # Accumulate total travel time
                    avg_travel_times[pair][1] += 1 # Increment count of travel times for this pair
    except Exception as e:
        print(f"⚠️ Erro OD {date_str}: {e}")

    # --- Boarding: future_demand_at_B ---
    try:
        df_board = pd.read_parquet(os.path.join(BASE_PATH, "Boarding", f"boarding-{date_str}.parquet")) # Read the boarding data
        if df_board is not None and "stop_id" in df_board.columns: # Check if the DataFrame is not empty and contains 'stop_id'
            df_board.dropna(subset=["stop_id"], inplace=True)

            for stop_id, group in df_board.groupby("stop_id"): # Group by stop_id
                total = len(group)  # Count total boardings at this stop
                future_demand[stop_id][0] += total # Accumulate total boardings for this stop
                future_demand[stop_id][1] += 1  # Increment count of occurrences for this stop
        else:
            print(f"⚠️ Column 'target_boarding' not found in Boarding {date_str}")
    except Exception as e:
        print(f"⚠️ Error Boarding {date_str}: {e}")

    # --- AVL: occupancy_rate ---
    try:
        df_avl = pd.read_parquet(os.path.join(BASE_PATH, "AVL", f"avl-lines-{date_str}.parquet")) # Read the AVL data
        if 'stop_id' in df_avl.columns: # Check if 'stop_id' column exists
            for stop_id, group in df_avl.groupby('stop_id'): # Group by stop_id
                count = len(group)
                if count > 0: # Check if there are any records for this stop
                    # Simulated occupancy, as real column is missing
                    occ_value = 0.5  # Dummy value, as AVL does not have real occupancy_rate
                    occupancy[stop_id][0] += occ_value * count # Accumulate total occupancy for this stop
                    occupancy[stop_id][1] += count # Increment count of records for this stop
        else:
            print(f"⚠️ Column 'stop_id' not found in AVL {date_str}")
    except Exception as e:
        print(f"⚠️ Error AVL {date_str}: {e}")

    # --- LTI: uptime_normalized ---
    try:
        df_lti = pd.read_parquet(os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet")) # Read the LTI data

        # Check which time columns are available
        if 'start_trip' in df_lti.columns and 'end_trip' in df_lti.columns: # Prefer 'start_trip' and 'end_trip' if available
            df_lti['start_trip'] = pd.to_datetime(df_lti['start_trip'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['end_trip'], errors='coerce', dayfirst=True)
        elif 'inicioProgramado' in df_lti.columns and 'fimProgramado' in df_lti.columns: # Fallback to 'inicioProgramado' and 'fimProgramado'
            df_lti['start_trip'] = pd.to_datetime(df_lti['inicioProgramado'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['fimProgramado'], errors='coerce', dayfirst=True)
        else:
            raise ValueError("⚠️ LTI does not have recognized time columns.")

        # Ensure the 'vehicle' column exists (or adjust if it has a different name)
        vehicle_col = 'vehicle' if 'vehicle' in df_lti.columns else 'veiculo' # Adjust if necessary
        df_lti.dropna(subset=['start_trip', 'end_trip', vehicle_col], inplace=True) # Drop rows with invalid dates or missing vehicle IDs

        for _, row in df_lti.iterrows(): # Calculate uptime for each vehicle
            uptime_sec = (row['end_trip'] - row['start_trip']).total_seconds() # Calculate uptime in seconds
            if uptime_sec > 0: # Only consider positive uptimes
                uptime[row[vehicle_col]][0] += uptime_sec # Accumulate total uptime seconds for this vehicle
                uptime[row[vehicle_col]][1] += 1 # Increment count of uptime records for this vehicle
    except Exception as e:
        print(f"⚠️ Error LTI {date_str}: {e}")


# === Finalization ===
def finalize_avg_dict(d):
    return {k: v[0]/v[1] for k, v in d.items() if v[1] > 0}

with open(os.path.join(OUTPUT_PATH, "avg_travel_time_AB.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(avg_travel_times), f)

with open(os.path.join(OUTPUT_PATH, "future_demand_at_B.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(future_demand), f)

with open(os.path.join(OUTPUT_PATH, "occupancy_rate.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(occupancy), f)

with open(os.path.join(OUTPUT_PATH, "uptime_normalized.pkl"), "wb") as f:
    pickle.dump(finalize_avg_dict(uptime), f)

print("\n✅ Todos os dados foram processados e salvos.")
