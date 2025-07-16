import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gymnasium import spaces
import numpy as np
import random
import pickle
import gymnasium.utils.seeding  # Importar seeding
from gymnasium.spaces import Discrete

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "graph_exploration_v0"}

    # network: Grafo do tipo networkx.Graph
    # actions_amount: Número de ações possíveis (em geral, o máximo de vizinhos que um nó pode ter)
    # stopClass: Classe de parada personalizada (opcional)
    # rewardClass: Classe de recompensa personalizada (opcional)
    # initial_nodes, target_nodes: nós de início e destino (se não passados, são escolhidos aleatoriamente)
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2, stopClass=None, rewardClass=None,
                initial_nodes=None, target_nodes=None, render_mode=None,
                avg_travel_time_AB=None, future_demand_at_B=None,
                occupancy_rate=None, uptime_normalized=None):

        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._num_agents = num_agents

        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(self.network.nodes()))}  # Mapeia nós para índices
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}  # Mapeia índices para nós

        # Cria os agentes
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(num_agents))))  # Mapeia nomes de agentes para índices

        # Estados individuais
        self.states = {}           # Guarda o nó atual de cada agente
        self.targets = {}          # Guarda o nó alvo de cada agente
        self.steps = {}            # Contador de passos por agente
        self.delays = {}           # Dicionário de delays por agente
        self.estimated_times = {}  # Tempo estimado percorrido por agente
        self.expected_times = {}   # Tempo ótimo (ideal) por agente

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes

        # === Substitui os dados derivados de DataFrames por observações prontas ===

        self.avg_travel_time_AB = avg_travel_time_AB           # (nodeA, nodeB) -> tempo médio
        self.future_demand_at_B = future_demand_at_B           # node_id -> demanda estimada
        self.occupancy_rate = occupancy_rate                   # agent_id -> taxa de ocupação (0.0 a 1.0)
        self.uptime_normalized = uptime_normalized             # agent_id -> tempo de operação (0.0 a 1.0)

        # Relógio global da simulação (começa às 6h00)
        self.current_time = 6 * 60 * 60  # começa às 6h00, em segundos
        self.headways = {}         # Histórico de chegada por ponto
        self.sync_stats = {}       # Regularidade dos veículos nas paradas

        self.service_center_node = random.choice(list(self.network.nodes))

        # Estado interno dos agentes
        self.agents = {}
        for agent_id in self.possible_agents:
            self.agents[agent_id] = {
                "location": None,
                "occupancy": self.occupancy_rate.get(agent_id, 0.0),  # Inicializa com valor real se houver
                "uptime": self.uptime_normalized.get(agent_id, 1.0),  # Idem
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": None,                  # Lista de nós
                "route_idx": 0,                  # Começa no início da rota
                "needs_service": False          # Flag para manutenção
            }

            # self.action_space = Discrete(3)  # Espaço de ação discreto (3 ações possíveis: 0, 1, 2)

        # Parâmetros de recompensa
        self.reward_weights = {
            "occ_penalty": 1.0,
            "uptime_bonus": 1.0,
            "sync_score": 1.0,
            "energy_efficiency": 1.0
        }
        self.occupancy_range = (0.6, 0.9)  # Faixa ideal de ocupação

        # Apenas para compatibilidade futura com wrappers que esperam .observation_spaces e .action_spaces
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

        self.default_travel_time = np.mean(list(self.avg_travel_time_AB.values())) # Tempo médio de viagem padrão
        self.max_travel_time = 3250.0 # Tempo maximo encontrado no dataset (5 minutos e 25 segundos)
 



    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]  # Reinicia lista de agentes ativos
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}
        self.current_time = 6 * 60 * 60  # Reinicia o relógio global para 6h00 (em segundos)
        self.headways = {}  # trabalhando direto com dados agregados
        self.sync_stats = {}  # Regularidade dos veículos nas paradas
        self.agents_state = {}
        observations = {}
        infos = {}

        available_nodes = list(self.network.nodes)

        for agent in self.agents:
            initial = random.choice(available_nodes)
            target = random.choice(available_nodes)
            
            while target == initial and len(available_nodes) > 1:
                target = random.choice(available_nodes)

            # Estado interno do agente
            self.agents_state[agent] = {
                "location": initial,
                "occupancy": float(self.occupancy_rate.get((initial, target), 0.0)),  # taxa de ocupação
                "uptime": float(self.uptime_normalized.get(agent, 1.0)),  # tempo de operação
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
            }

            self.states[agent] = initial
            self.targets[agent] = target
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}

            # calcula o caminho mais curto entre o nó inicial e o alvo
            try:
                path = nx.shortest_path(self.network, initial, target, weight="travel_time")
            except nx.NetworkXNoPath:
                path = []
            
            self.agents_state[agent]["route"] = path
            self.agents_state[agent]["route_idx"] = 0
            self.targets[agent] = path[-1] if path else target  # garante que o alvo seja o último nó do caminho

            # Calcula o tempo esperado com base em dados reais
            self.expected_times[agent] = sum(
                self.avg_travel_time_AB.get((path[i], path[i + 1]), self.default_travel_time) # Usa o tempo médio de viagem padrão se não houver dados
                for i in range(len(path) - 1)
            ) if path else float("inf")

            travel_time = self.avg_travel_time_AB.get((initial, target), self.default_travel_time)
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0) # Normaliza o tempo de viagem para 0.0 a 1.0

            # Cria observação baseada na estrutura definida
            observations[agent] = np.array([
                self.current_time / (24 * 60 * 60),  # current_time_normalized (0.0 a 1.0)
                self.occupancy_rate.get((initial, target), 0.0),  # occupancy_rate
                normalized_travel_time,  # avg_travel_time_AB
                self.future_demand_at_B.get(target, 0.0),  # future_demand_at_B
                self.uptime_normalized.get(agent, 1.0),  # uptime_normalized
                1.0 if self.agents_state[agent]["maintenance_status"] == "ok" else 0.0,  # maintenance_status
                self.node_to_idx[initial],  # curr_node_id_norm
                self.node_to_idx[target],  # next_node_id_norm
            ], dtype=np.float32)

            infos[agent] = {
                "path": path,
                "expected_time": self.expected_times[agent],
            }

        return observations, infos


    
    def step(self, actions):
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
            state = self.agents_state[agent]
            route = state["route"]
            idx = state["route_idx"]
            
            print(f"AGENTE: {agent}, IDX: {idx}, ROTA: {route}")
            if idx >= len(route):
                print(f"[ERRO] Agente {agent} excedeu a rota. IDX={idx}, TAM={len(route)}")
                continue  # ou return terminateds, truncateds, rewards, obs, infos

            curr_node = route[idx]

            action = actions[agent]

            if action == 0: # WAIT
                reward = -0.1  # penalidade leve
                self.current_time += 1
                terminated = False
                truncated = False

            elif action == 1: # MOVE_TO next stop
                if idx + 1 < len(route):
                    next_node = route[idx + 1]
                    travel_time = self.avg_travel_time_AB.get((curr_node, next_node))
                    if travel_time is None:
                        # print(f"[WARN] Tempo de viagem ausente para ({curr_node}, {next_node}). Usando fallback.")
                        travel_time = self.default_travel_time # Usa o tempo médio de viagem padrão se não houver dados

                    occupancy = self.occupancy_rate.get((curr_node, next_node), 0.0)
                    self.current_time += travel_time
                    self.estimated_times[agent] += travel_time

                    self.agents_state[agent]["route_idx"] += 1
                    self.agents_state[agent]["occupancy"] = occupancy
                    self.states[agent] = next_node

                    if next_node not in self.headways:
                        self.headways[next_node] = []
                    self.headways[next_node].append(self.current_time)

                    reward = self.reward.getReward(
                        new_state=next_node,
                        previous_state=curr_node,
                        action=action,
                        target=route[-1],
                        network=self.network,
                        estimated_time=self.estimated_times[agent],
                        expected_time=self.expected_times[agent],
                        delay=0,
                        agent_state=state,
                        headways=self.headways[next_node]
                    )

                    terminated = next_node == route[-1]  # fim da rota?
                    truncated = self.steps[agent] >= self.max_steps

                else:
                    # Já está no último ponto da rota: penaliza se tentar mover de novo
                    reward = -1.0
                    terminated = False
                    truncated = False

            elif action == 2:
                # SERVICE_CENTER
                sc_node = self.get_nearest_service_center(curr_node)
                travel_time = self.avg_travel_time_AB.get((curr_node, sc_node), self.default_travel_time) # Usa o tempo médio de viagem padrão se não houver dados
                self.current_time += travel_time
                self.estimated_times[agent] += travel_time

                # Reset de manutenção, combustível, uptime
                self.agents_state[agent]["fuel"] = 100.0
                self.agents_state[agent]["uptime"] = 1.0
                self.agents_state[agent]["maintenance_status"] = "ok"
                self.states[agent] = sc_node

                reward = -0.5  # pequeno custo por manutenção
                terminated = False
                truncated = self.steps[agent] >= self.max_steps

            else:
                # Ação inválida
                reward = -10.0
                terminated = False
                truncated = True

            # Atualiza observação
            curr_node = self.states[agent]
            route_idx = self.agents_state[agent]["route_idx"]
            next_node = route[route_idx + 1] if route_idx + 1 < len(route) else curr_node


            travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0)

            # print("print(self.current_time) ", self.current_time)

            observations[agent] = np.array([
                self.current_time / (24 * 60 * 60),  # current_time_normalized (0.0 a 1.0)
                self.occupancy_rate.get((curr_node, next_node), 0.0),
                normalized_travel_time, # Usa o tempo médio de viagem padrão se não houver dados
                self.future_demand_at_B.get(next_node, 0.0),
                self.uptime_normalized.get(agent, 1.0),
                1.0 if self.agents_state[agent]["maintenance_status"] == "ok" else 0.0,
                self.node_to_idx[curr_node],
                self.node_to_idx[next_node],
            ], dtype=np.float32)

            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "count": self.steps[agent],
                "occupancy": self.agents_state[agent]["occupancy"],
                "location": curr_node,
                "next_stop": next_node,
                "headways": self.headways.get(curr_node, []),
            }

        self.agents = [
            agent for agent in self.agents if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=np.array([
                0.0,    # time_of_day_norm
                0.0,    # occupancy_rate
                0.0,    # avg_travel_time_AB (normalizado)
                0.0,    # future_demand_at_B
                0.0,    # uptime
                0.0,    # maintenance_status
                0.0,    # curr_node_id
                0.0     # next_node_id
            ], dtype=np.float32),
            high=np.array([
                1.0,    # time_of_day_norm
                1.0,    # occupancy_rate
                1.0,    # avg_travel_time_AB (normalizado!)
                1e6,    # future_demand_at_B (mantém valor realista)
                1.0,    # uptime
                1.0,    # manutenção ok
                2e9,    # curr_node_id
                2e9     # next_node_id
            ], dtype=np.float32)
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

    def get_nearest_service_center(self, current_node):
        # Encontra o nó mais próximo do centro de serviço, ainda não está dinamico
        return self.service_center_node

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
    def __init__(self, waitTimeDict=None, reward_weights=None, occupancy_range=(0.6, 0.9)):
        super().__init__()
        with open('./output/combined_sum_amount.pkl', 'rb') as f:
            self.waitTimeDict = pickle.load(f)
        
        self.reward_weights = reward_weights or {
            "occ_penalty": 1.0,         # W1
            "uptime_bonus": 1.0,        # W2
            "sync_score": 1.0,          # W3
            "energy_efficiency": 1.0    # W4
        }

        self.occupancy_range = occupancy_range  # Faixa ideal (ex: 60% a 90%)

    def getReward(self, new_state, previous_state, action, target, network, estimated_time, expected_time, delay, agent_state=None, headways=None):
        reward = 0.0

        # 1. Penalidade por ocupação fora da faixa ideal
        if agent_state is not None:
            occupancy = agent_state.get("occupancy", 0.0)
            min_occ, max_occ = self.occupancy_range
            if occupancy < min_occ:
                occ_penalty = (min_occ - occupancy) ** 2 # Penaliza se a ocupação estiver abaixo do mínimo
            elif occupancy > max_occ:
                occ_penalty = (occupancy - max_occ) ** 2 # Penaliza se a ocupação estiver acima do máximo
            else:
                occ_penalty = 0.0
            reward -= self.reward_weights["occ_penalty"] * occ_penalty

        # 2. Bônus por uptime
        if agent_state is not None:
            uptime = agent_state.get("uptime", 1.0) # tempo de atividade normalizado (0.0 a 1.0)
            reward += self.reward_weights["uptime_bonus"] * uptime # Bônus proporcional ao uptime   

        # 3. Regularidade (sincronização/headways)
        sync_score = 0.0
        if headways and len(headways) > 1:
            intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)] # Calcula os intervalos entre chegadas no ponto de onibus
            if intervals:
                avg_interval = sum(intervals) / len(intervals) # Intervalo médio
                std = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5 # Desvio padrão dos intervalos
                sync_score = -std  # Penaliza irregularidade
                reward += self.reward_weights["sync_score"] * sync_score # Bônus proporcional à regularidade

        # 4. Eficiência energética (estimado vs esperado)
        if expected_time > 0:
            travel_efficiency = max(0.0, 1 - (estimated_time / expected_time))
            reward += self.reward_weights["energy_efficiency"] * travel_efficiency
        else:
            reward += 0.0  # Nenhum bônus se não houver tempo esperado

        return reward


# Essa é a classe padrão de parada, que termina o episódio quando o agente chega ao nó alvo   
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2
