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
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2, stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None, render_mode=None, boarding_df=None, landing_df=None, trips_df=None):
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._num_agents = num_agents

        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(self.network.nodes()))} # Mapeia nós para índices
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()} # Mapeia índices para nós

        # Cria os agentes
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(num_agents)))) # Mapeia nomes de agentes para índices
        
        # Estados individuais
        self.states = {}   # Guarda o nó atual de cada agente
        self.targets = {}  # Guarda o nó alvo de cada agente
        self.steps = {}    # Contador de passos por agente
        self.delays = {}   # Dicionário de delays por agente
        self.estimated_times = {}  # Tempo estimado percorrido por agente
        self.expected_times = {}   # Tempo ótimo (ideal) por agente

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes

        # Novos elementos para observações e recompensas (baseado na estrutura da imagem)

        ## 1. Dados de demanda
        self.boarding_demand = self._extrair_demanda_de_embarque(boarding_df)
        self.alighting_demand = self._extrair_demanda_de_desembarque(landing_df)

        ## 2. Grafo com tempos reais e posições
        self.edge_times = self._extrair_tempos_de_arestas(trips_df)         # (a, b) -> tempo médio
        self.stop_coords = self._extrair_coordenadas_de_paradas(trips_df)       # stop_id -> (lat, lon)

        ## 3. Tempo atual e controle de sincronização
        self.current_time = 6 * 60  # Começa às 6h00 (em minutos) "relógio global" da simulação. Começa às 6h00 e avança a cada passo
        self.headways = {}         # stop_id -> armazena o histórico de tempos de chegada dos ônibus em cada parada
        self.sync_stats = {}       # controle de regularidade por ponto (se os ônibus estão espaçados igualmente, evitando comboios) (Ver Como tratar isso no treinamento)

        ## 4. Estado dos agentes
        self.agents = {}
        for agent_id in self.possible_agents: # Inicializa um dicionário com o estado individual de cada ônibus para a observação e recompensa
            self.agents[agent_id] = {
                "location": None, # posição atual do agente
                "occupancy": 0.0, # taxa de ocupação (0.0 a 1.0)
                "uptime": 1.0, # tempo de atividade (0.0 a 1.0)
                "fuel": 100.0, # 100% de combustível
                "maintenance_status": "ok", # status de manutenção
                "schedule": [],     # se você for simular horários
            }

        ## 5. Parâmetros de recompensa
        self.reward_weights = { # Controla como calcular a recompensa a ideia é permitir que você module o peso de cada termo
            "occ_penalty": 1.0, # Penalidade por ocupação
            "uptime_bonus": 1.0, # Bônus por tempo de atividade
            "sync_score": 1.0, # Pontuação de sincronização
            "energy_efficiency": 1.0 # Eficiência energética
        }
        self.occupancy_range = (0.6, 0.9) # Faixa de ocupação (60% a 90%), define o intervalo ideal para ocupação (penaliza se o ônibus estiver muito cheio ou muito vazio). PROVISORIO

    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]

        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}

        self.current_time = 6 * 60  # Reinicia o relógio global para 6h00

        self.headways = {stop: [] for stop in self.stop_coords}  # Reinicia o histórico de tempos de chegada dos ônibus em cada parada
        self.sync_stats = {stop: {"expected":0, "actual": []} for stop in self.stop_coords}  # Reinicia o controle de regularidade por ponto

        self.agents_state = {}  # Reinicia o estado dos agentes
        observations = {}
        infos = {}

        available_nodes = list(self.network.nodes)

       # print(self.boarding_demand if self.boarding_demand else "Nenhuma demanda de embarque disponível.")
       # print(self.alighting_demand if self.alighting_demand else "Nenhuma demanda de desembarque disponível.")
       # print(self.edge_times if self.edge_times else "Nenhum tempo de aresta disponível.")
       # print(self.stop_coords if self.stop_coords else "Nenhuma coordenada de parada disponível.")

        for agent in self.agents: # Inicializa cada agente
            initial = random.choice(available_nodes)
            target = random.choice(available_nodes)
            
            while target == initial and len(available_nodes) > 1:
                target = random.choice(available_nodes)

            # Estado interno do agente 
            self.agents_state[agent] = {
                "location": initial, # Posição inicial do agente
                "occupancy":0.0, # Taxa de ocupação inicial (0.0 a 1.0)
                "uptime": 1.0, # Tempo de atividade inicial
                "fuel": 100.0, # Combustível inicial (100%)
                "maintenance_status": "ok", # Status de manutenção inicial
                "schedule": [], # Horário inicial (se for simular horários)
            }
            
            # Grava os estados
            self.states[agent] = initial
            self.targets[agent] = target
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}  # Se quiser simular delays individuais

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

            infos[agent] = { # Informações adicionais para cada agente
                "path": path if 'path' in locals() else [], # Caminho ótimo calculado
                "expected_time": self.expected_times[agent] # Tempo esperado para o caminho
            }

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

            # --- 1. Tempo de deslocamento (sem delays reais por enquanto) ---
            edge = (min(str(previous_state), str(new_state)), max(str(previous_state), str(new_state)))
            wait_time, _ = self.reward.waitTimeDict.get(edge, (1, 1))
            # delay = self.dynamicDelays.get(edge, 0)
            total_time = wait_time # + delay 
            self.estimated_times[agent] += total_time
            self.current_time += total_time # Avança o relógio global

            # --- 2. Ocupação do agente (simples por enquanto) ---
            boarding = self.boarding_demand.get(new_state, 0) # Demanda de embarque na nova parada
            alighting = self.alighting_demand.get(new_state, 0) # Demanda de desembarque na nova parada
            old_occupancy = self.agents_state[agent]["occupancy"] # Ocupação anterior do agente
            new_occupancy = max(0.0, min(1.0, old_occupancy + (boarding - alighting) / 100.0)) # Atualiza a ocupação do agente (normaliza entre 0.0 e 1.0)

            self.agents_state[agent]["occupancy"] = new_occupancy # Atualiza o estado do agente

            # --- 3. Headways (controle de sincronização) ---
            if new_state in self.headways:
                self.headways[new_state] = []
            
            self.headways[new_state].append(self.current_time) # Adiciona o tempo atual ao histórico de headways

            # --- 4. Recompensa ---
            # Calcula recompensa e término do episódio para o agente
            reward = self.reward.getReward(
                new_state, previous_state, actions[agent], self.targets[agent],
                self.network, self.estimated_times[agent], self.expected_times[agent], 0,  # delay = 0
                agent_state=self.agents_state[agent],
                headways=self.headways[new_state]
            )

            terminated = self.stop.isTerminated(
                new_state, previous_state, actions[agent], self.targets[agent], self.network
            )

            truncated = self.steps[agent] >= self.max_steps

            obs = np.array([
                self.node_to_idx[new_state], # Estado atual do agente
                self.node_to_idx[self.targets[agent]] # Destino do agente
            ], dtype=np.int64)

            observations[agent] = obs
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = { # Informações adicionais para cada agente
                "count": self.steps[agent],
                "occupancy": new_occupancy, # Taxa de ocupação atual do agente
                "time_spent": self.estimated_times[agent], # Tempo total estimado percorrido pelo agente
                "headways": self.headways[new_state], # Histórico de tempos de chegada dos ônibus na nova parada
            }

        # Remove agentes que terminaram ou truncaram
        self.agents = [
            agent for agent in self.agents if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human": # Renderiza o ambiente se o modo de renderização for "human"
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
    
    def _extrair_demanda_de_embarque(self, boarding_df):
        # Agrupa embarques por stop_id
        if boarding_df is None:
            return {}
        return dict(boarding_df.groupby("stop_id")["registers"].sum())

    def _extrair_demanda_de_desembarque(self, landing_df):
        # Agrupa desembarques por stop_id_ali
        if landing_df is None:
            return {}
        return dict(landing_df.groupby("stop_id_ali").size())

    def _extrair_tempos_de_arestas(self, trips_df):
        if trips_df is None:
            return {}
        edge_times = {}
        grouped = trips_df.groupby("trip")
        for _, trip_data in grouped:
            trip_data = trip_data.sort_values("hora_ponto")
            stops = trip_data["stop_id"].tolist()
            times = trip_data["tempo_total"].tolist()
            for i in range(len(stops)-1):
                pair = (stops[i], stops[i+1])
                edge_times[pair] = times[i+1]  # tempo entre paradas
        return edge_times

    def _extrair_coordenadas_de_paradas(self, trips_df):
        if trips_df is None:
            return {}
        return dict(zip(trips_df["stop_id"], zip(trips_df["lat"], trips_df["lon"])))

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
            "occ_penalty": 1.0, # W1
            "uptime_bonus": 1.0, # W2
            "sync_score": 1.0, # W3
            "energy_efficiency": 1.0 # W4
        }
        self.occupancy_range = occupancy_range  # Faixa de ocupação ideal (60% a 90%)

    def getReward(self, new_state, previous_state, action, target, graph, est_time, expected_time, delay, agent_state=None, headways=None):
        
        reward = 0.0 

        # 1. Penalidade por ocupação fora da faixa ideal
        if agent_state is not None:
            occupancy = agent_state["occupancy"]
            min_occ, max_occ = self.occupancy_range
            if occupancy < min_occ:
                occ_penalty = (min_occ - occupancy) ** 2 # Penaliza se a ocupação estiver abaixo do mínimo
            elif occupancy > max_occ:
                occ_penalty = (occupancy - max_occ) ** 2 # Penaliza se a ocupação estiver acima do máximo
            else:
                occ_penalty = 0.0 # Ocupação ideal, sem penalidade
            reward -= self.reward_weights["occ_penalty"] * occ_penalty
        
        # 2. Bônus por uptime — quanto mais ativo, melhor
        if agent_state is not None:
            uptime = agent_state("uptime", 1,0) # Tempo de atividade do agente (0.0 a 1.0)
            reward += self.reward_weights["uptime_bonus"] * uptime # Bônus proporcional ao uptime
        
        # 3.Regularidade (headways) - Pontuação de sincronização — quanto mais espaçados, melhor
        sync_score = 0.0
        if headways and len(headways) > 1:
            intervals = [headways[i+1] - headways[i] for i in range(len(headways) - 1)]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                std = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5 # Desvio padrão dos intervalos
                sync_score = -std # quanto mais regular (menor desvio padrão), melhor
                reward += self.reward_weights["sync_score"] * sync_score
        
        # 4. Eficiência energética (quanto menor o tempo/delay, melhor)
        travel_efficiency = max(0.0, 1 - (est_time / (expected_time + 1e-5))) # Evita divisão por zero
        reward += self.reward_weights["energy_efficiency"] * travel_efficiency # Bônus proporcional


        return reward # Essa é a recompensa final calculada com base nos critérios definidos


# Essa é a classe padrão de parada, que termina o episódio quando o agente chega ao nó alvo   
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target
