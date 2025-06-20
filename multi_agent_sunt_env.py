import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gymnasium import spaces
import numpy as np
import random
import pickle
import gymnasium.utils.seeding  # Importar seeding

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "graph_exploration_v0"}

    # network: Grafo do tipo networkx.Graph
    # actions_amount: Número de ações possíveis (em geral, o máximo de vizinhos que um nó pode ter)
    # stopClass: Classe de parada personalizada (opcional)
    # rewardClass: Classe de recompensa personalizada (opcional)
    # initial_nodes, target_nodes: nós de início e destino (se não passados, são escolhidos aleatoriamente)
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2, stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None, render_mode=None):
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._num_agents = num_agents

        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(self.network.nodes()))}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # Cria os agentes
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(num_agents))))
        
        # Estados individuais
        self.states = {}   # Guarda o nó atual de cada agente
        self.targets = {}  # Guarda o nó alvo de cada agente
        self.steps = {}    # Contador de passos por agente
        self.delays = {}   # Dicionário de delays por agente
        self.estimated_times = {}  # Tempo estimado percorrido por agente
        self.expected_times = {}   # Tempo ótimo (ideal) por agente

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes
    
    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]

        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}

        observations = {}
        infos = {}

        nodes = list(self.network.nodes)

        for agent in self.agents:
            initial = random.choice(nodes)
            target = random.choice(nodes)
            
            while target == initial and len(nodes) > 1:
                target = random.choice(nodes)

            # Grava os estados
            self.states[agent] = initial
            self.targets[agent] = target
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}  # se quiser simular delays individuais

            # Chama sua função de delay
            self.generate_random_delay(initial, target)

            # Define peso com delays dinâmicos (padrão 1)
            def edge_weight(u, v, d):
                key = (min(str(u), str(v)), max(str(u), str(v)))
                return self.reward.waitTimeDict.get(key, (1, 1))[0]

            # Calcula o caminho ótimo com Dijkstra
            try:
                path = nx.shortest_path(self.network, initial, target, weight=edge_weight)
                self.expected_times[agent] = sum(
                    self.reward.waitTimeDict.get(
                        (min(str(path[i]), str(path[i+1])), max(str(path[i]), str(path[i+1]))),
                        (1, 1)
                    )[0]
                    for i in range(len(path) - 1)
                )
            except nx.NetworkXNoPath:
                self.expected_times[agent] = float("inf")

            # Observação inicial: (estado atual, destino)
            observations[agent] = np.array([
                self.node_to_idx[initial],
                self.node_to_idx[target]
            ], dtype=np.int64)

            infos[agent] = {}

        return observations, infos

    
    def step(self, actions):
        """
        Parâmetro:
            actions: dict do tipo {agent_0: action_0, agent_1: action_1, ...}

        Retorna:
            observations, rewards, terminations, truncations, infos
            (todos no formato dict por agente)
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            self.steps[agent] += 1
            previous_state = self.states[agent]
            neighbors = list(self.network.neighbors(previous_state))

            # Ação inválida: penaliza e trunca
            if actions[agent] >= len(neighbors):
                reward = -150
                terminated = False
                truncated = True
                obs = np.array([
                    self.node_to_idx[self.states[agent]],
                    self.node_to_idx[self.targets[agent]]
                ], dtype=np.int64)

                observations[agent] = obs
                rewards[agent] = reward
                terminations[agent] = terminated
                truncations[agent] = truncated
                infos[agent] = {"invalid_action": True}
                continue

            # Move para o novo estado
            new_state = neighbors[actions[agent]]
            self.states[agent] = new_state

            # Cálculo de tempo com delays
            edge = (min(str(previous_state), str(new_state)), max(str(previous_state), str(new_state)))
            wait_time, _ = self.reward.waitTimeDict.get(edge, (1, 1))
            delay = self.dynamicDelays.get(edge, 0)
            total_time = wait_time + delay
            self.estimated_times[agent] += total_time

            # Calcula recompensa e término do episódio para o agente
            reward = self.reward.getReward(
                new_state, previous_state, actions[agent], self.targets[agent],
                self.network, self.estimated_times[agent], self.expected_times[agent], delay
            )

            terminated = self.stop.isTerminated(
                new_state, previous_state, actions[agent], self.targets[agent], self.network
            )

            truncated = self.steps[agent] >= self.max_steps

            obs = np.array([
                self.node_to_idx[new_state],
                self.node_to_idx[self.targets[agent]]
            ], dtype=np.int64)

            observations[agent] = obs
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "count": self.steps[agent],
                "delay": delay,
                "time_spent": self.estimated_times[agent]
            }

        # Remove agentes que terminaram ou truncaram
        self.agents = [
            agent for agent in self.agents if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos


    @functools.lru_cache(maxsize=None) # Decorador para cache de resultados
    def observation_space(self, agent): # Define o espaço de observação para cada agente
        return spaces.Box(
            low=0,
            high=len(self.network.nodes()) - 1,
            shape=(2,),
            dtype=np.int64
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): # Define o espaço de ação para cada agente
        return spaces.Discrete(self.actions_amount)
    
    def generate_random_delay(self, start, target):
        try:
            # Encontra o caminho mais curto entre o start e o target
            shortest_path = nx.shortest_path(self.network, source=start, target=target, weight='weight')
            path_edges = list(zip(shortest_path, shortest_path[1:])) # Cria uma lista de arestas do caminho mais curto
            total_time = 0

            if not path_edges:
                return  # Sem arestas para atrasar

            # Calcula tempo total das arestas do caminho
            for a, b in path_edges:
                a, b = str(a), str(b)
                edge_key = (min(a, b), max(a, b)) # Ordena os nós da aresta para evitar duplicação
                if edge_key in self.reward.waitTimeDict:
                    edge_time = self.reward.waitTimeDict[edge_key][0]
                    total_time += edge_time # Soma o tempo de espera da aresta
                else:
                    x = 0 
                    #print(f"[AVISO] Aresta {edge_key} não está no waitTimeDict!")

            average_time = total_time / len(path_edges)
            #print(f"Média de tempo das arestas do caminho ótimo: {average_time}")

            # Escolhe uma das arestas do caminho para atrasar
            delay_u, delay_v = random.choice(path_edges)
            delay_u, delay_v = str(delay_u), str(delay_v) # Ordena os nós da aresta para evitar duplicação
            delay_edge_key = (min(delay_u, delay_v), max(delay_u, delay_v)) # Aresta escolhida para atraso

            if delay_edge_key in self.reward.waitTimeDict: # Verifica se a aresta escolhida está no waitTimeDict
                delay = average_time * 5  # Simula congestionamento pesado
                self.dynamicDelays = {
                    delay_edge_key: delay # Aresta escolhida para atraso com o tempo de atraso aplicado
                }
                #print(f"Aresta atrasada: {delay_edge_key}, atraso aplicado: {delay}")
            else:
                #print(f"[ERRO] Aresta escolhida para atraso {delay_edge_key} não está no waitTimeDict.")
                self.dynamicDelays = {}

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.dynamicDelays = {}

# Essa é a classe base para as classes de recompensa e parada
class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError

# Essa é a classe base para as classes de parada
class StopConditionBaseClass():
    def isTerminated(self, state, previousState, action, target, graph):
        raise NotImplementedError

# Essa é a classe padrão de recompensa, que calcula a recompensa com base no tempo total e na quantidade de viagens 
class DefaultReward(RewardBaseClass):
    def __init__(self) -> None:
        super().__init__()
        with open('./output/combined_sum_amount.pkl', 'rb') as f:
            self.waitTimeDict = pickle.load(f)

    def getReward(self, state, previousState, action, target, graph, estimated_time_so_far, max_expected_time, delay):
        # Usa aresta ordenada como chave
        edge = (min(str(previousState), str(state)), max(str(previousState), str(state)))
        wait_time, amount = self.waitTimeDict.get(edge, (1, 1))  # padrão seguro

        total_time = (wait_time / 3600) + delay
        reward = -total_time  # Penaliza tempo gasto

        # Bônus se chegar no destino
        if state == target:
            delay_ratio = estimated_time_so_far / max_expected_time if max_expected_time > 0 else 1
            if delay_ratio <= 1.2:
                reward += 500_000
            elif delay_ratio <= 1.5:
                reward += 250_000
            else:
                reward -= 500_000
            print(f"Reward: {reward}, Estimated time so far: {estimated_time_so_far}, Max expected time: {max_expected_time}")
            print("delay_ratio: ", delay_ratio)

        return reward


# Essa é a classe padrão de parada, que termina o episódio quando o agente chega ao nó alvo   
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target
