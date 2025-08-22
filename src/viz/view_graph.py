import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Uses backend without graphical interface
import matplotlib.pyplot as plt
import pickle

with open('./graph_gtfs_fev_2024.gpickle', 'rb') as f:
    G = pickle.load(f)

print("Nodes:", G.nodes())
print("Edges:", G.edges())

nx.draw(G, with_labels=True)
plt.savefig("graph_output.png")  # Saves the graph as an image
print("Graph saved as 'graph_output.png'")
