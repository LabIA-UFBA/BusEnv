import pickle

# Load the pkl file
with open("src/training_observation/occupancy_rate.pkl", "rb") as f:
    occupancy_rate = pickle.load(f)

# Node you want to search
node_id = 44784619

# Search in the dictionary
if node_id in occupancy_rate:
    value = occupancy_rate[node_id]
    print(f"Node {node_id} found: occupancy rate = {value:.4f}")
else:
    print(f"Node {node_id} not found in the file.")
