# SUNT dataset needed to run this script
# CHANGE: base_path = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp" to your sunt path

import os
import pandas as pd
import pickle
import re
import sys
import pyarrow.parquet as pq
from datetime import timedelta

def process_date_folder(date_str: str, base_data_path: str) -> dict:
    """
    Process the Parquet files of OD (Origin-Destination) for a given date,
    calculating the travel time between pairs of stops.
    """
    
    file_path = os.path.join(base_data_path, "OD", f"od-{date_str}.parquet")

    if not os.path.exists(file_path):
        print(f"❌ Archive not found for date: {file_path}")
        return None

    try:
        df_od = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None

    # Convert time columns to datetime if they are not already
    df_od['stop_time'] = pd.to_datetime(df_od['stop_time'])

    time_per_stop_pair = {}

    print(f"📆 Processing {date_str}...")

    # Grouping by trip to calculate the time between sequential stops
    # A trip is defined by vehicle, trip_id, and direction_id
    grouped_trips = df_od.sort_values(by=['vehicle', 'trip_id', 'direction_id', 'pt_sequence']).groupby(['vehicle', 'trip_id', 'direction_id'])

    progress_counter = 0
    total_rows_to_process = len(df_od)

    for name, group in grouped_trips:
        # Skip if the group has only one stop, as there is no time between stops
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            prev_row = group.iloc[i-1]
            current_row = group.iloc[i]

            # Ensure we are on the same trip and direction
            if prev_row['trip_id'] == current_row['trip_id'] and \
               prev_row['direction_id'] == current_row['direction_id']:

                # Calculate the time difference
                time_diff = (current_row['stop_time'] - prev_row['stop_time']).total_seconds()

                # Only consider positive times (ongoing trips)
                if time_diff > 0:
                    stop_pair = (prev_row['stop_id'], current_row['stop_id'])
                    if stop_pair not in time_per_stop_pair:
                        time_per_stop_pair[stop_pair] = [0.0, 0] # [total_time, count]

                    time_per_stop_pair[stop_pair][0] += time_diff
                    time_per_stop_pair[stop_pair][1] += 1
            
            progress_counter += 1
            if progress_counter % 10000 == 0: # Update progress every 10,000 rows
                print(f"\r⏳ {progress_counter/total_rows_to_process*100:.2f}% of rows from {date_str}", end='')

    print(f"\r✅ 100.00% of rows from {date_str} processed.") # Final progress message

    return time_per_stop_pair



# ==== Principal ====
base_path = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp" 
output_path = "./output" # Where the output files will be saved

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Number of days to process (optional)
try:
    max_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
except ValueError:
    print("⚠️ Error: Invalid argument. Please use a number, e.g., python script.py 5")
    sys.exit(1)

# Get all date folders within the 'OD' folder to determine available dates
# The folder pattern is SUNT/tmp/OD/, SUNT/tmp/AFC/, etc.
# The files are od-YYYY-MM-DD.parquet, afc-YYYY-MM-DD.parquet
# We need to find the available dates in the parquet files.
# We can list the files inside one of the subfolders (e.g., OD) and extract the dates.

available_files = []
od_dir = os.path.join(base_path, "OD")
if os.path.exists(od_dir):
    available_files = [f for f in os.listdir(od_dir) if f.startswith('od-') and f.endswith('.parquet')]
else:
    print(f"❌ Directory OD not found: {od_dir}")
    sys.exit(1)

date_folders = []
for file_name in available_files:
    match = re.match(r'od-(\d{4}-\d{2}-\d{2})\.parquet', file_name)
    if match:
        date_folders.append(match.group(1))

date_folders.sort() # Guarantee chronological order

# If a number of days was specified, truncate the list
if max_days:
    date_folders = date_folders[:max_days]

combined_sum_amount = {} # Renamed to reflect the output file name
combined_averages = {} # For the final averages

# Process each date
for index, date_folder in enumerate(date_folders):
    print(f"\n--- Processing date {index+1}/{len(date_folders)}: {date_folder} ---")

    # Call the data processing function for the current date
    # Pass base_path so the function knows where to find the subfolders (OD, AFC, etc.)
    result_for_date = process_date_folder(date_folder, base_data_path=base_path)

    if result_for_date is None:
        print(f"🚫 Skipping date {date_folder} due to error or file not found.")
        continue

    # Save intermediate results for each date (if desired)
    avg_date_path = f"{output_path}/averages_{date_folder}.pkl"
    with open(avg_date_path, "wb") as f:
        pickle.dump(result_for_date, f)
    print(f"\n💾 Intermediate results for {date_folder} saved to {avg_date_path}")

    # Combine results for total and count
    print(f"➕ Combining data from {date_folder} with totals...\n")
    for stop_pair, (total_time, count) in result_for_date.items():
        if stop_pair not in combined_sum_amount:
            combined_sum_amount[stop_pair] = [0.0, 0] # Initialize with float for total_time
        combined_sum_amount[stop_pair][0] += total_time
        combined_sum_amount[stop_pair][1] += count

# Save combined data (total time sum and count)
print("\n📦 Saving combined sums and counts...")
sum_amount_file = f"{output_path}/combined_sum_amount.pkl"
with open(sum_amount_file, "wb") as f:
    pickle.dump(combined_sum_amount, f)
print(f"✅ Combined sums and counts saved to {sum_amount_file}.")

# Calculate and save final averages
print("\n📊 Calculating and saving final averages...")
final_averages = {
    stop_pair: total_time / count
    for stop_pair, (total_time, count) in combined_sum_amount.items()
    if count > 0 # Avoid division by zero
}
averages_file = f"{output_path}/combined_averages.pkl"
with open(averages_file, "wb") as f:
    pickle.dump(final_averages, f)
print(f"✅ Final averages saved to {averages_file}.")

print("\n🎉 Processing complete! Results are ready.")