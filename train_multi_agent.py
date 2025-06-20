import os
import pickle
import networkx as nx
from multi_agent_sunt_env import parallel_env  # seu novo env multiagente

if __name__ == "__main__":
    # Carrega o grafo urbano do SUNT
    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
        G = pickle.load(f)

    # Cria o ambiente com 2 agentes
    env = parallel_env(network=G, actions_amount=9, max_steps=100, num_agents=2)

    # Inicializa o ambiente
    observations, infos = env.reset()
    print("Observações iniciais:")
    print(observations)

    done = {agent: False for agent in env.agents}
    trunc = {agent: False for agent in env.agents}
    step_count = 1

    while any(not (done[agent] or trunc[agent]) for agent in env.agents):
        actions = {}
        for agent in env.agents:
            if not (done[agent] or trunc[agent]):
                # Ação aleatória como placeholder
                actions[agent] = env.action_space(agent).sample()

        # Avança o ambiente
        observations, rewards, done, trunc, infos = env.step(actions)

        print(f"\n[Step {step_count}]")
        print("Ações:", actions)
        print("Recompensas:", rewards)
        print("Finais:", done)
        print("Truncados:", trunc)

        step_count += 1
