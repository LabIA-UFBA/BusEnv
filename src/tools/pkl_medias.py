import pickle
import numpy as np

file_path = 'src/training_observation/avg_travel_time_AB.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"File content '{file_path}':\n")

    if isinstance(data, dict):
        values = list(data.values())
        print(f"Number of pairs: {len(values)}")
        print(f"Average travel time: {np.mean(values):.2f}")
        print(f"Minimum time: {np.min(values):.2f}")
        print(f"Maximum time: {np.max(values):.2f}")
        print(f"Standard deviation: {np.std(values):.2f}")

        # If you want to see the pairs with the highest times
        print("\nTop 5 highest times:")
        for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  {k}: {v:.2f}")

    else:
        print(f"Unexpected data type: {type(data)}")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
