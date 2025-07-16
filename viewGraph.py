import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Usa backend sem interface gráfica
import matplotlib.pyplot as plt
import pickle

with open('./SUNT/data/graph_designer/graph_gtfs_fev_2024.gpickle', 'rb') as f:
    G = pickle.load(f)

print("Nós:", G.nodes())
print("Arestas:", G.edges())

nx.draw(G, with_labels=True)
plt.savefig("graph_output.png")  # Salva o gráfico como imagem
print("Gráfico salvo como 'graph_output.png'")
