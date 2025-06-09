import gymnasium as gym
from gymnasium import spaces
import pickle
import networkx as nx
import numpy as np
import random

# Essa é a classe do ambiente personalizado, onde o agente pode se mover em um grafo
class GraphExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    
    # network: Grafo do tipo networkx.Graph
    # actions_amount: Número de ações possíveis (em geral, o máximo de vizinhos que um nó pode ter)
    # stopClass: Classe de parada personalizada (opcional)
    # rewardClass: Classe de recompensa personalizada (opcional)
    # initial, target: nós de início e destino (se não passados, são escolhidos aleatoriamente)
    def __init__(self, network: nx.Graph, actions_amout: int, max_steps: int, stopClass = None, rewardClass = None, initial = None, target = None):
        super(GraphExplorationEnv, self).__init__()
        self.network = network

        self.initial = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if initial is None else initial
        self.state = self.initial
        self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if target is None else target

        # Garante que o nó inicial e o nó alvo sejam diferentes
        # Se o nó inicial e o nó alvo forem iguais, escolhe um novo nó alvo aleatório
        while(self.state == self.target and self.network.number_of_nodes != 1):
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]


        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        # Define o espaço de ação e o espaço de observação
        # O espaço de ação é discreto, com o número de ações igual ao número de vizinhos do nó atual
        # O espaço de observação é um vetor de dois elementos: o nó atual e o nó alvo
        self.action_space = gym.spaces.Discrete(actions_amout)
        # self.observation_space = gym.spaces.Box(low=0, high=np.array([self.network.number_of_nodes() - 1 , self.network.number_of_nodes() - 1]), shape=(2,), dtype=np.int64)
        
        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(self.network.nodes()))} 
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        self.observation_space = spaces.Box(
            low=0, 
            high=len(self.network.nodes()) - 1, 
            shape=(2,), 
            dtype=np.int64)

        
        # Define o número máximo de passos por episódio
        self.max_steps = max_steps
     
        self.count = 0 # Contador de Passos
        
        # Um delay dinamico em alguns nós, eventos que aumentam o tempo de viagem em certas arestas ao longo do tempo
        self.dynamicDelays = {} # dicionário que armazena o tempo de espera dinâmico em cada aresta (u, v) do grafo

        # Acumulador de tempo que o agente gastou
        self.estimated_time_so_far = 0

        # Tempo estimado do trajeto ótimo (útil para comparar e penalizar atrasos)
        self.max_expected_time = 0
    
    # Reseta o ambiente, escolhendo um nó inicial e um nó alvo aleatórios
    # Sorteia novo estado inicial e destino 
    def reset(self,seed=None):
        super().reset(seed=seed)
        self.initial = random.choice(list(self.network.nodes))
        self.target = random.choice(list(self.network.nodes))
        
        while self.target == self.initial and self.network.number_of_nodes() > 1: # Garante que o nó inicial e o nó alvo sejam diferentes
            self.target = random.choice(list(self.network.nodes))

        self.state = self.initial # Define o estado inicial como o nó inicial
        self.estimated_time_so_far = 0 # Reinicia o tempo estimado
        self.count = 0 # Reinicia o contador de passos

        self.generate_random_delay(self.initial,self.target) # Geração de atraso simulado em uma aresta aleatória

        # Definição de peso da aresta
        def edge_weight(u, v, d): 
            key = (min(str(u), str(v)), max(str(u), str(v))) # Ordena os nós da aresta para evitar duplicação
            return self.reward.waitTimeDict.get(key, (1, 1))[0] # Pega o tempo de espera da aresta, se não existir, usa (1, 1) como padrão
        
        # Calcula o caminho mais curto entre o nó inicial e o nó alvo, considerando o tempo de espera
        # Dijkstra é usado para encontrar o caminho mais curto entre dois nós em um grafo ponderado
        try:
            path = nx.shortest_path(self.network, self.initial, self.target, weight=edge_weight) # Calcula o caminho mais curto entre o nó inicial e o nó alvo
            self.max_expected_time = sum( # Soma o tempo de espera de cada aresta no caminho
                self.reward.waitTimeDict.get((min(str(path[i]), str(path[i+1])), max(str(path[i]), str(path[i+1]))), (1, 1))[0] # Tempo de espera da aresta
                for i in range(len(path)-1)
            )
            #print("Caminho ótimo entre o nó inicial e o alvo: ", path)
            #print("Tempo estimado do trajeto ótimo: ", self.max_expected_time)
        
        except nx.NetworkXNoPath: # Se não houver caminho entre o nó inicial e o alvo, trata a exceção
            #print("Nenhum caminho encontrado entre o node inicial e o alvo")
            self.max_expected_time = float('inf') # Define o tempo estimado como infinito, pois não há caminho

        # obs = (self.state, self.target) # Retorna a observação inicial: uma tupla (estado atual, destino) 
        obs = np.array([
            self.node_to_idx[self.initial],
            self.node_to_idx[self.target]
        ], dtype=np.int64)

        return obs, {}
    
    
    # Realiza um passo no ambiente, movendo-se para um nó vizinho
    def step(self,action):
        self.count += 1

        # possibleNextStates: são os vizinhos do estado atual (os possíveis próximos estados)
        possibleNextStates = list(self.network.neighbors(self.state))
        print(possibleNextStates, "Possible next states") 
        print("action:", action)
        previousState = self.state

        # Se o nó atual não tem vizinhos, termina o episódio imediatamente | VER SE É INTERESSANTE MESMO FAZER ISSO
        if len(possibleNextStates) == 0:
            return self._make_step_return(previousState, reward=0, terminated=True)

       # if action >= len(possibleNextStates): # Se a ação escolhida for maior que o número de vizinhos, lança um erro
       #     raise ValueError(f"Ação {action} inválida. Apenas {len(possibleNextStates)} vizinhos disponíveis.")
        
        if action >= len(possibleNextStates):
            reward = -150  # Penalidade por ação inválida
            terminated = False
            obs = np.array([
                self.node_to_idx[self.state],
                self.node_to_idx[self.target]
            ], dtype=np.int64)
            print(reward)
            # trunca o episódio, mas não termina
            return obs, reward, terminated, True, {}


        # Atualiza o estado com base na ação
        self.state = possibleNextStates[action]

        # Antes de mudar de estado, calula-se o tempo real da aresta 
        # Aresta é uma tupla (u, v) onde u é o nó anterior e v é o nó atual
        # Aresta é ordenada para evitar duplicação, (u, v) e (v, u) serem considerados diferentes
        edge = (min(str(previousState), str(self.state)), max(str(previousState), str(self.state))) # Ordena os nós da aresta para evitar duplicação
        wait_time, _ = self.reward.waitTimeDict.get(edge, (1, 1)) # Pega o tempo de espera da aresta, se não existir, usa (1, 1) como padrão
        delay = self.dynamicDelays.get(edge, 0) # Tempo de espera dinâmico (se houver)
        total_time = wait_time + delay # Tempo total da aresta

        self.estimated_time_so_far += total_time # Acumula o tempo estimado

        # print(f"Current state: {self.state}, Target: {self.target}, Estimated time so far: {self.estimated_time_so_far}, Delay: {delay}, Total time: {total_time}")

        # Usa a rewardClass e stopClass para calcular recompensa e término do episódio
        reward = self.reward.getReward(
            self.state, previousState, action, self.target, self.network, 
            self.estimated_time_so_far, self.max_expected_time, delay
        )

        terminated = self.stop.isTerminated(self.state, previousState, action, self.target, self.network)

        # print("Reward on Step:",reward, "Terminated:",terminated)

        # obs = (self.state, self.target)
        obs = np.array([
            self.node_to_idx[self.state],
            self.node_to_idx[self.target]
        ], dtype=np.int64) 

        # Episódio truncado por limite de passos
        if self.count >= self.max_steps:
            print(f"[AVISO] Episódio truncado após {self.count} passos.")
            return obs, reward, False, True, {"count": self.count}

        print("step executado com sucesso, recompensa:", reward, "terminado:", terminated)
        # Retorna: observação, recompensa, se o episódio terminou, se o episódio foi truncado (False), e um dicionário com metadados (count de passos)
        return obs, reward, terminated, False, {"count" : self.count}

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

    # Método auxiliar para criar o retorno do passo
    def _make_step_return(self, state, reward, terminated):
        obs = np.array([
            self.node_to_idx[state], # Estado atual
            self.node_to_idx[self.target] # Nó alvo
        ], dtype=np.int64)
        return obs, reward, terminated, False, {"count": self.count} 

    def render(self):
        pass  

    def close(self):
        pass

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
        with open('./output/combined_sum_amount.pkl', 'rb') as f: # Carrega o dicionário de tempos de espera e quantidades de viagens do DataSet
            self.waitTimeDict = pickle.load(f) # waitTimeDict[(u, v)] = (tempo de espera, quantidade)

    # A recompensa é calculada com base no tempo total e na quantidade de viagens
    def getReward(self, state, previousState, action, target, graph, estimated_time_so_far, max_expected_time, delay):
        
        totalTime = self.waitTimeDict[(previousState, state)][0]
        amount = self.waitTimeDict[(previousState, state)][1]

        # reward = 0 # A recompensa padrão é negativa: -totalTime / amount, incentivando caminhos com menor tempo médio

        # Recompensa negativa pelo tempo gasto
        # A recompensa é negativa, pois o agente deve minimizar o tempo gasto
        # Isso casa perfeitamente com algoritmos como Q-learning ou DQN, que maximizam retorno acumulado
        print(f"Total time: {totalTime}, Delay: {delay}, Estimated time so far: {estimated_time_so_far}, Max expected time: {max_expected_time}")
        reward = - ((totalTime / 3600) + delay) # Convertendo para horas

        # Se o agente chega no destino, dá um bônus proporcional ao tempo estimado e ao tempo máximo esperado
        if state == target:
            delay_ratio = estimated_time_so_far / max_expected_time if max_expected_time > 0 else 1
            if delay_ratio <= 1.2:
                reward += 500000 # Grandesa escolhida para ter um impacto significativo na recompensa total
            elif delay_ratio <= 1.5:
                reward += 250000
            else:
                reward -= 500000 # Penaliza se o tempo estimado for muito maior que o esperado
            print(f"Reward: {reward}, Estimated time so far: {estimated_time_so_far}, Max expected time: {max_expected_time}")
            print("delay_ratio: ", delay_ratio)
        
        return reward

# Essa é a classe padrão de parada, que termina o episódio quando o agente chega ao nó alvo   
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target
    