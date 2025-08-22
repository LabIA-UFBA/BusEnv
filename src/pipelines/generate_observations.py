# SUNT dataset needed to run this script
# CHANGE: BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp" to your sunt path


import os
import re
import sys
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from collections import defaultdict

# Path to the base directory and output directory
BASE_PATH = "/media/wesley/Disco_local/graph-exploration/SUNT/tmp"
OUTPUT_PATH = "../training_observation"
GRAPH_PATH = "./src/viz/graph_gtfs_fev_2024.gpickle"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Optional argument to limit number of days
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
    print("⚠️ Continuing without graph validation...")
    G = None
    valid_nodes = set()

# === Configuration ===
# Outlier thresholds
MAX_REASONABLE_DEMAND = 150    # Max boardings per stop per day
MAX_TRAVEL_TIME_SECONDS = 1800 # Max travel time between consecutive stops (30 min for heavy traffic)
MIN_TRAVEL_TIME_SECONDS = 120  # Min travel time between stops (2 minutes)
MAX_UPTIME_HOURS = 20          # Max vehicle uptime per trip

# === Auxiliary Functions ===
def get_date_list(subfolder: str, prefix: str):
    folder = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(folder):
        print(f"❌ Directory not found: {folder}")
        return []
    filenames = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.parquet')]
    dates = []
    for fname in filenames:
        match = re.match(fr'{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet', fname)
        if match:
            dates.append(match.group(1))
    return sorted(dates)

def normalize_id(val, max_val=1_000_000):
    return int(val) / max_val if val and val >= 0 else 0.0

def normalize_time_of_day(dt):
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400

def validate_stop_in_graph(stop_id):
    """Check if stop exists in the graph"""
    if not valid_nodes:
        return True  # No graph validation available
    return str(stop_id) in valid_nodes

def validate_edge_in_graph(stop_a, stop_b):
    """Check if edge exists in the graph"""
    if not G:
        return True  # No graph validation available
    return G.has_edge(str(stop_a), str(stop_b))
    """Remove outliers using IQR method"""
    if len(data) < 4:
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return [x for x in data if lower_bound <= x <= upper_bound]

def remove_outliers_iqr(data, multiplier=1.5):
    """
    Remove outliers usando o método do Intervalo Interquartil (IQR).
    
    Parâmetros:
        data (list ou array-like): Lista de valores numéricos.
        multiplier (float): Fator multiplicador do IQR para definir limites.
    
    Retorna:
        list: Lista com os valores dentro dos limites.
    """
    if not data or len(data) < 4:
        return list(data)  # Não faz nada se houver poucos dados

    data_array = np.array(data)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    return filtered.tolist()


# === Enhanced Accumulators ===
avg_travel_times = defaultdict(list)      # Store all values for better statistics
future_demand = defaultdict(list)         # Store daily counts per stop
occupancy = defaultdict(list)             # Store calculated occupancy values
uptime = defaultdict(list)                # Store uptime values per vehicle
route_frequencies = defaultdict(int)      # Track route frequencies
stop_activity = defaultdict(lambda: defaultdict(int))  # Track hourly activity

# Graph validation counters
graph_validation_stats = {
    'invalid_stops': 0,
    'invalid_edges': 0,
    'valid_edges': 0,
    'total_stops_checked': 0,
    'total_edges_checked': 0
}

dates = get_date_list("OD", "od")
if max_days:
    dates = dates[:max_days]

print(f"\n📅 Processing {len(dates)} days...")

for day_idx, date_str in enumerate(dates):
    print(f"\n➡️  {day_idx+1}/{len(dates)} - {date_str}")
    
    daily_demand_per_stop = defaultdict(int)

    # --- OD: Enhanced avg_travel_time_AB ---
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
        df_od['stop_time'] = pd.to_datetime(df_od['stop_time'])
        
        # Group and sort properly
        grouped = df_od.sort_values(by=['vehicle', 'trip_id', 'direction_id', 'pt_sequence']) \
                        .groupby(['vehicle', 'trip_id', 'direction_id'])
        
        for _, group in grouped:
            for i in range(1, len(group)):
                prev = group.iloc[i-1]
                curr = group.iloc[i]
                
                # Graph validation for stops and edges
                prev_stop = str(prev['stop_id'])
                curr_stop = str(curr['stop_id'])
                
                graph_validation_stats['total_stops_checked'] += 2
                graph_validation_stats['total_edges_checked'] += 1
                
                # Check if both stops exist in graph
                if not (validate_stop_in_graph(prev_stop) and validate_stop_in_graph(curr_stop)):
                    graph_validation_stats['invalid_stops'] += 1
                    continue
                
                # Check if edge exists in graph
                if not validate_edge_in_graph(prev_stop, curr_stop):
                    graph_validation_stats['invalid_edges'] += 1
                    continue
                
                graph_validation_stats['valid_edges'] += 1
                
                delta_seconds = (curr['stop_time'] - prev['stop_time']).total_seconds()
                
                # Enhanced filtering
                if MIN_TRAVEL_TIME_SECONDS <= delta_seconds <= MAX_TRAVEL_TIME_SECONDS:
                    pair = (prev_stop, curr_stop)
                    avg_travel_times[pair].append(delta_seconds)
                    
                    # Track route frequency
                    route_frequencies[curr['route_short_name']] += 1
                    
        print(f"   ✓ OD: {len(avg_travel_times)} stop pairs processed")
    except Exception as e:
        print(f"⚠️ Error OD {date_str}: {e}")

    # --- Boarding: Enhanced future_demand_at_B ---
    try:
        df_board = pd.read_parquet(os.path.join(BASE_PATH, "Boarding", f"boarding-{date_str}.parquet"))
        
        if df_board is not None and "stop_id" in df_board.columns:
            df_board.dropna(subset=["stop_id"], inplace=True)
            
            # Filter only valid stops from graph
            df_board = df_board[df_board['stop_id'].astype(str).apply(validate_stop_in_graph)]
            
            # Convert stop_time to datetime for hourly analysis
            if 'stop_time' in df_board.columns:
                df_board['stop_time'] = pd.to_datetime(df_board['stop_time'], errors='coerce')
                df_board = df_board.dropna(subset=['stop_time'])
                
                # Calculate hourly activity
                for _, row in df_board.iterrows():
                    hour = row['stop_time'].hour
                    stop_activity[row['stop_id']][hour] += 1
            
            # Count boardings per stop for this day
            stop_counts = df_board.groupby("stop_id").size()
            
            for stop_id, count in stop_counts.items():
                # Filter extreme outliers
                if count <= MAX_REASONABLE_DEMAND:
                    daily_demand_per_stop[stop_id] = count
                else:
                    # Use median of that stop if available, else use a reasonable cap
                    if stop_id in future_demand and len(future_demand[stop_id]) > 0:
                        capped_value = min(count, np.median(future_demand[stop_id]) * 2)
                    else:
                        capped_value = MAX_REASONABLE_DEMAND
                    daily_demand_per_stop[stop_id] = capped_value
                    print(f"   📊 Capped demand at stop {stop_id}: {count} → {capped_value}")
        
        # Add daily totals to accumulator
        for stop_id, count in daily_demand_per_stop.items():
            future_demand[stop_id].append(count)
            
        print(f"   ✓ Boarding: {len(daily_demand_per_stop)} stops with demand data")
    except Exception as e:
        print(f"⚠️ Error Boarding {date_str}: {e}")

    # --- OD: Enhanced occupancy_rate calculation ---
    try:
        df_od = pd.read_parquet(os.path.join(BASE_PATH, "OD", f"od-{date_str}.parquet"))
        
        # Use the 'loading' column which represents passenger count
        if 'loading' in df_od.columns and 'stop_id' in df_od.columns:
            df_od['loading'] = pd.to_numeric(df_od['loading'], errors='coerce')
            df_od = df_od.dropna(subset=['loading', 'stop_id'])
            
            # Calculate occupancy rate per stop (assuming bus capacity ~80 passengers)
            bus_capacity = 80
            
            for stop_id, group in df_od.groupby('stop_id'):
                # Validate stop exists in graph
                if not validate_stop_in_graph(str(stop_id)):
                    continue
                    
                # Calculate average occupancy rate for this stop
                avg_loading = group['loading'].mean()
                occupancy_rate = min(avg_loading / bus_capacity, 1.0)  # Cap at 100%
                
                if occupancy_rate > 0:  # Only store meaningful values
                    occupancy[stop_id].append(occupancy_rate)
        
        print(f"   ✓ OD Occupancy: {len(occupancy)} stops processed")
    except Exception as e:
        print(f"⚠️ Error OD Occupancy {date_str}: {e}")

    # --- LTI: Enhanced uptime_normalized ---
    try:
        df_lti = pd.read_parquet(os.path.join(BASE_PATH, "LTI", f"lti-{date_str}.parquet"))

        # Handle different column names
        if 'start_trip' in df_lti.columns and 'end_trip' in df_lti.columns:
            df_lti['start_trip'] = pd.to_datetime(df_lti['start_trip'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['end_trip'], errors='coerce', dayfirst=True)
        elif 'inicioProgramado' in df_lti.columns and 'fimProgramado' in df_lti.columns:
            df_lti['start_trip'] = pd.to_datetime(df_lti['inicioProgramado'], errors='coerce', dayfirst=True)
            df_lti['end_trip'] = pd.to_datetime(df_lti['fimProgramado'], errors='coerce', dayfirst=True)
        else:
            raise ValueError("⚠️ LTI does not have recognized time columns.")

        vehicle_col = 'vehicle' if 'vehicle' in df_lti.columns else 'veiculo'
        df_lti = df_lti.dropna(subset=['start_trip', 'end_trip', vehicle_col])
        
        # Filter only "Normal" activities for more realistic uptime
        if 'activity' in df_lti.columns:
            df_lti = df_lti[df_lti['activity'] == 'Normal']

        vehicle_daily_uptime = defaultdict(float)
        
        for _, row in df_lti.iterrows():
            uptime_sec = (row['end_trip'] - row['start_trip']).total_seconds()
            uptime_hours = uptime_sec / 3600
            
            # Filter reasonable uptime values
            if 0.1 <= uptime_hours <= MAX_UPTIME_HOURS:  # Between 6 minutes and 20 hours
                vehicle_daily_uptime[row[vehicle_col]] += uptime_hours
        
        # Normalize uptime by daily maximum (assuming 16-hour service day)
        max_service_hours = 16
        for vehicle_id, total_hours in vehicle_daily_uptime.items():
            normalized_uptime = min(total_hours / max_service_hours, 1.0)
            uptime[vehicle_id].append(normalized_uptime)
            
        print(f"   ✓ LTI: {len(vehicle_daily_uptime)} vehicles with uptime data")
    except Exception as e:
        print(f"⚠️ Error LTI {date_str}: {e}")

# === Advanced Finalization with Robust Statistics ===
def finalize_enhanced_dict(
    d, 
    remove_outliers=True, 
    use_median=True, 
    apply_log=False  # mantém dados "reais"
):
    """
    Finaliza os dicionários de métricas aplicando remoção de outliers e 
    estatísticas robustas.

    Parâmetros:
        d (dict): chave -> lista de valores diários
        remove_outliers (bool): se aplica IQR para limpeza
        use_median (bool): se True, usa mediana, senão usa média
        apply_log (bool): se True, aplica log1p para reduzir escala

    Retorna:
        result (dict): chave -> valor agregado (já normalizado se configurado)
        stats (dict): estatísticas detalhadas por chave
    """
    result = {}
    stats = {}

    for k, values in d.items():
        if len(values) == 0:
            continue

        clean_values = remove_outliers_iqr(values) if (remove_outliers and len(values) > 3) else values

        # Escolher métrica central
        val = np.median(clean_values) if use_median else np.mean(clean_values)

        # Log opcional
        if apply_log:
            val = np.log1p(val)

        result[k] = val
        stats[k] = {
            'mean': float(np.mean(clean_values)),
            'median': float(np.median(clean_values)),
            'std': float(np.std(clean_values)) if len(clean_values) > 1 else 0,
            'count': len(clean_values),
            'original_count': len(values),
            'outliers_removed': len(values) - len(clean_values)
        }

    return result, stats

# Save processed data with statistics
print("\n📊 Finalizing data with statistical analysis...")

# Travel times
travel_times, travel_stats = finalize_enhanced_dict(avg_travel_times)
with open(os.path.join(OUTPUT_PATH, "avg_travel_time_AB.pkl"), "wb") as f:
    pickle.dump(travel_times, f)
with open(os.path.join(OUTPUT_PATH, "travel_time_stats.pkl"), "wb") as f:
    pickle.dump(travel_stats, f)

# Future demand
demand, demand_stats = finalize_enhanced_dict(future_demand)
with open(os.path.join(OUTPUT_PATH, "future_demand_at_B.pkl"), "wb") as f:
    pickle.dump(demand, f)
with open(os.path.join(OUTPUT_PATH, "demand_stats.pkl"), "wb") as f:
    pickle.dump(demand_stats, f)

# Occupancy
occupancy_final, occupancy_stats = finalize_enhanced_dict(occupancy)
with open(os.path.join(OUTPUT_PATH, "occupancy_rate.pkl"), "wb") as f:
    pickle.dump(occupancy_final, f)
with open(os.path.join(OUTPUT_PATH, "occupancy_stats.pkl"), "wb") as f:
    pickle.dump(occupancy_stats, f)

# Uptime
uptime_final, uptime_stats = finalize_enhanced_dict(uptime)
with open(os.path.join(OUTPUT_PATH, "uptime_normalized.pkl"), "wb") as f:
    pickle.dump(uptime_final, f)
with open(os.path.join(OUTPUT_PATH, "uptime_stats.pkl"), "wb") as f:
    pickle.dump(uptime_stats, f)

# Additional insights
with open(os.path.join(OUTPUT_PATH, "route_frequencies.pkl"), "wb") as f:
    pickle.dump(dict(route_frequencies), f)

with open(os.path.join(OUTPUT_PATH, "hourly_stop_activity.pkl"), "wb") as f:
    pickle.dump(dict(stop_activity), f)

# Print summary statistics
print("\n📈 Summary Statistics:")
print(f"Travel Time Pairs: {len(travel_times)} (avg: {np.mean(list(travel_times.values())):.1f}s)")
print(f"Demand Stops: {len(demand)} (avg: {np.mean(list(demand.values())):.1f} boardings/day)")
print(f"Occupancy Stops: {len(occupancy_final)} (avg: {np.mean(list(occupancy_final.values())):.2f} rate)")
print(f"Vehicle Uptimes: {len(uptime_final)} (avg: {np.mean(list(uptime_final.values())):.2f} normalized)")
print(f"Active Routes: {len(route_frequencies)}")

# Graph validation summary
if valid_nodes:
    print("\n🗺️  Graph Validation Summary:")
    print(f"Valid edges found: {graph_validation_stats['valid_edges']}")
    print(f"Invalid edges filtered: {graph_validation_stats['invalid_edges']}")
    print(f"Invalid stops filtered: {graph_validation_stats['invalid_stops']}")
    edge_validity_rate = (graph_validation_stats['valid_edges'] / 
                         max(graph_validation_stats['total_edges_checked'], 1)) * 100
    print(f"Edge validity rate: {edge_validity_rate:.1f}%")

# Identify potential issues
high_demand_stops = {k: v for k, v in demand.items() if v > 100}
if high_demand_stops:
    print(f"\n⚠️  {len(high_demand_stops)} stops with >100 daily boardings (check if realistic)")

print("\n✅ Enhanced data processing completed with graph validation!")