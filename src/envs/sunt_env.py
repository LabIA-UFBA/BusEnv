import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gym import spaces
import numpy as np
import random
import pickle
import gym.utils.seeding  # import seeding
from gym.spaces import Discrete
import csv
import os

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "graph_exploration_v0"}

    # network: A networkx.Graph type object representing the graph
    # actions_amount: The number of possible actions (generally, the maximum number of neighbors a node can have)
    # stopClass: Custom stop class (optional)
    # rewardClass: Custom reward class (optional)
    # initial_nodes, target_nodes: start and target nodes (if not passed, they are chosen randomly)
    # render_mode: "human" or "rgb_array" (optional)
    # avg_travel_time_AB, future_demand_at_B, occupancy_rate, uptime_normalized: dicts with precomputed data
    # real_routes: dict mapping agent_id to a fixed route (list of nodes)
    # route_metadata: dict with metadata for each route (optional)
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2,
                 stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None,
                 render_mode=None, avg_travel_time_AB=None, future_demand_at_B=None,
                 occupancy_rate=None, uptime_normalized=None,
                 real_routes=None, route_metadata=None):  

        # --- Basic configuration ---
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps 
        self.render_mode = render_mode

        # --- Agentes ---
        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()  # <- MARLlib precisa disso
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # --- Stop/Reward classes ---
        self.stop = DefaultStopClass() if stopClass is None else stopClass
        self.reward = DefaultReward() if rewardClass is None else rewardClass

        # --- Map nodes to indexes ---
        self.node_to_idx = {str(n): i for i, n in enumerate(self.network.nodes)}  
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # --- Internal states per agent ---
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes

        # --- External data / features ---
        self.avg_travel_time_AB = avg_travel_time_AB or {}
        self.future_demand_at_B = future_demand_at_B or {}
        self.occupancy_rate = occupancy_rate or {}
        self.uptime_normalized = uptime_normalized or {}

        # --- Global clock and statistics ---
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents} # Every agent starts at 6:00 AM 
        self.headways = {}
        self.sync_stats = {}

        self.service_center_node = random.choice(list(self.network.nodes))

        # --- Internal structure of agents ---
        self.agent_states = {}
        for agent_id in self.possible_agents:
            self.agent_states[agent_id] = {
                "location": None,
                "occupancy": 0.0,
                "uptime": 1.0,
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": None,
                "route_idx": 0,
                "needs_service": False,
                "going_forward": True,
            }

        # --- Reward parameters ---
        self.reward_weights = {
            "occ_penalty": 1.0,
            "uptime_bonus": 1.0,
            "sync_score": 1.0,
            "energy_efficiency": 1.0
        }
        self.occupancy_range = (0.6, 0.9)

        # --- Observation and action spaces ---
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        # --- Defaults / limits ---
        if self.avg_travel_time_AB:
            self.default_travel_time = np.mean(list(self.avg_travel_time_AB.values()))
        else:
            self.default_travel_time = 1.0

        self.max_travel_time = 3250.0
        self.max_capacity = 80

        self.real_routes = real_routes or {}
        self.route_metadata = route_metadata or {}
        self.agent_routes = {}

        self.metrics_file = "env_metrics.csv" # To log metrics for analysis

        if not os.path.exists(self.metrics_file): # Create the metrics file if it doesn't exist
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "env_steps", "mean_reward", "total_reward", "fairness"])


    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]  
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents}  # 6:00 AM
        self.headways = {}
        self.sync_stats = {}
        self.agent_states = {}

        observations = {}
        self.infos = {}

        for agent in self.agents:
            if agent not in self.agent_routes:  
                trip_id, path = random.choice(list(self.real_routes.items()))
                print(f"[DEBUG] Route chosen for {agent} (Trip ID: {trip_id}): {path}")
                self.agent_routes[agent] = path

            path = self.agent_routes[agent]
            if len(path) < 2:
                raise ValueError(f"Invalid route for {agent}: {path}")

            initial = path[0]
            target = path[-1]

            self.agent_states[agent] = {
                "location": initial,
                "occupancy": int(self.occupancy_rate.get(int(initial), 0.0)),
                "uptime": float(self.uptime_normalized.get(initial, 1.0)),
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": path,
                "route_idx": 0,
            }

            self.states[agent] = initial
            self.targets[agent] = target
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}

            self.expected_times[agent] = sum(
                self.avg_travel_time_AB.get((path[i], path[i + 1]), self.default_travel_time)
                for i in range(len(path) - 1)
            )

            next_node = path[1]
            travel_time = self.avg_travel_time_AB.get((initial, next_node), self.default_travel_time)
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0)

            obs_array = np.array([
                self.agent_times[agent] / (24 * 60 * 60),
                self.agent_states[agent]["occupancy"],
                normalized_travel_time,
                self.future_demand_at_B.get(next_node, 0.0),
                self.uptime_normalized.get(agent, 1.0),
                1.0 if self.agent_states[agent]["maintenance_status"] == "ok" else 0.0,
                self.node_to_idx[str(initial)],
                self.node_to_idx[str(next_node)],
            ], dtype=np.float32)

            # APPLY CLIPPING HERE!
            #    Use os limites (low/high) que você definiu no seu observation_space
            clipped_obs = np.clip(
                obs_array,
                self.observation_space(agent).low,  # Accessing the limits of the Box space
                self.observation_space(agent).high, # Accessing the limits of the Box space
            )

            observations[agent] = clipped_obs


            self.infos[agent] = {
                "chosen_route": path,
                "expected_time": self.expected_times[agent],
            }

        #print(f"[RESET] Environment reset. Agents: {self.agents}")

        self.current_episode_metrics = {  # Metrics for the current episode
            "rewards": {agent: 0.0 for agent in self.agents},
            "steps": {agent: 0 for agent in self.agents},
            "done": False
        }

        # return only observations
        return observations

    
    def step(self, actions):
        if not actions:  # If there are no actions, return empty observations
            self.agents = []
            return {}, {}, {}, {}, {}  # Observations, rewards, terminations, truncations, infos

        observations = {}  # Observations for each agent
        rewards = {}       # Rewards for each agent
        terminations = {}  # Terminations for each agent
        truncations = {}   # Truncations for each agent
        infos = {}

        for agent in self.agents:  # Checks if the agent is active
            self.steps[agent] += 1  # Increments the agent's step counter
            state = self.agent_states[agent]  # Internal state of the agent
            route = state["route"]  # Route of the agent
            idx = state["route_idx"]  # Current index in the route

            if idx >= len(route):  # Verifies if the agent has exceeded the route
                #print(f"[ERROR] Agent {agent} exceeded the route. IDX={idx}, LEN={len(route)}")
                terminations[agent] = True
                truncations[agent] = False
                rewards[agent] = -1.0
                continue

            curr_node = route[idx]  # Current node of the agent
            self.states[agent] = curr_node  # Ensures synchronization

            action = actions[agent]  # Action chosen by the agent
            #print(f"[DEBUG] action: {action} for agent: {agent}")

            # ================= WAIT =================
            if action == 0:  
                reward = -0.1  # Penalty for waiting
                elapsed = 60.0  # Assume 1 minute of waiting
                self.agent_times[agent] += elapsed # Update agent's internal clock
                state["uptime"] = max(state["uptime"] - elapsed / (12 * 3600), 0.0)
                state["fuel"] = max(state["fuel"] - elapsed / 300.0, 0.0)
                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = False

            # ================= MOVE =================
            elif action == 1:  
                going_forward = state.get("going_forward", True)
                route_length = len(route)

                if going_forward:
                    if idx + 1 < route_length:
                        next_node = route[idx + 1]
                        self.agent_states[agent]["route_idx"] += 1
                    else:
                        state["going_forward"] = False
                        self.agent_states[agent]["route_idx"] -= 1
                        next_node = route[self.agent_states[agent]["route_idx"]]
                else:  
                    if idx > 0:
                        self.agent_states[agent]["route_idx"] -= 1
                        next_node = route[self.agent_states[agent]["route_idx"]]
                    else:
                        state["going_forward"] = True
                        self.agent_states[agent]["route_idx"] += 1
                        next_node = route[self.agent_states[agent]["route_idx"]]

                direction = "➡️ going forward" if state.get("going_forward", True) else "⬅️ going backward"
                #print(f"[MOVE] Agent {agent} | {direction} | {curr_node} -> {next_node} "
                #      f"(t={self.current_time/3600:.2f}h, occ={state.get('occupancy',0):.1f}, "
                #      f"fuel={state.get('fuel',0):.1f}, uptime={state.get('uptime',0):.2f})")

                travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)

                prev_occ = state.get("occupancy", 0.0)
                if int(curr_node) in self.occupancy_rate:
                    expected_occ = self.occupancy_rate[int(curr_node)]
                    alpha = 0.5
                    new_occ = (1 - alpha) * prev_occ + alpha * expected_occ
                    occupancy = max(0.0, min(new_occ, 1.0))
                    #print(f"[DEBUG] Expected Occupancy: {expected_occ:.2f}, Previous Occupancy: {prev_occ:.2f}, New Occupancy: {occupancy:.2f}")
                else:
                    #print(f"[DEBUG] No Expected Occupancy for Node {curr_node}. Using Previous Occupancy: {prev_occ:.2f}")
                    occupancy = prev_occ

                state["occupancy"] = occupancy
                self.agent_times[agent] += travel_time
                self.estimated_times[agent] += travel_time
                state["uptime"] = max(state["uptime"] - travel_time / (12 * 3600), 0.0)
                state["fuel"] = max(state["fuel"] - travel_time / 300.0, 0.0)
                self.states[agent] = next_node

                if next_node not in self.headways:
                    self.headways[next_node] = []
                self.headways[next_node].append(self.agent_times[agent])

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

                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = self.steps[agent] >= self.max_steps

            # ================= SERVICE CENTER =================
            elif action == 2:  
                sc_node = self.get_nearest_service_center(curr_node)

                try:
                    path = nx.shortest_path(
                        self.network, source=curr_node, target=sc_node,
                        weight=lambda u, v, d: self.avg_travel_time_AB.get((u, v), self.default_travel_time)
                    )

                    total_travel_time = 0.0
                    total_fuel_cost = 0.0

                    for u, v in zip(path[:-1], path[1:]):
                        edge_time = self.avg_travel_time_AB.get((u, v), self.default_travel_time)
                        edge_time *= 0.3
                        total_travel_time += edge_time
                        total_fuel_cost += edge_time / 300.0

                    #print(f"[SERVICE_CENTER] Agent {agent} traveling path {path} "
                    #      f"with total travel time={total_travel_time:.2f}, fuel cost={total_fuel_cost:.2f}")
                except nx.NetworkXNoPath:
                    #print(f"[SERVICE_CENTER][ERROR] No path from {curr_node} to {sc_node}")
                    reward = -10.0
                    terminated = False
                    truncated = False
                else:
                    reward = 0.0
                    if state["fuel"] > 0.8 and state["uptime"] > 0.8:
                        reward -= 0.5 * total_travel_time  

                    if state["fuel"] < total_fuel_cost:
                        #print(f"[SERVICE_CENTER][FAIL] Agent {agent} insufficient fuel "
                        #      f"({state['fuel']:.2f}) needs {total_fuel_cost:.2f}")
                        reward = -20.0
                    else:
                        self.agent_times[agent] += total_travel_time
                        self.estimated_times[agent] += total_travel_time
                        state["fuel"] = max(state["fuel"] - total_fuel_cost, 0.0)
                        state["uptime"] = max(state["uptime"] - total_travel_time / (12 * 3600), 0.0)

                        state["fuel"] = 100.0
                        state["uptime"] = 1.0
                        state["maintenance_status"] = "ok"
                        self.states[agent] = sc_node
                        reward = -1.0 * (1 + total_travel_time / 600.0)

                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = self.steps[agent] >= self.max_steps

            else:
                reward = -10.0
                terminated = False
                truncated = True

            # ================= OBSERVATION UPDATE =================
            route_idx = self.agent_states[agent]["route_idx"]
            curr_node = self.states[agent]
            next_node = route[route_idx + 1] if route_idx + 1 < len(route) else curr_node

            travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0)
            
            #print(f"[STEP] agent: {agent}")
            #print(f"[STEP] self.future_demand_at_B.get(next_node, 0.0): {self.future_demand_at_B.get(next_node, 0.0)}")
            #print(f"[STEP]  self.agent_times[agent] : {self.agent_times[agent]}") 
            #print(f"[STEP]  self.agent_times[agent] / (24 * 60 * 60): {self.agent_times[agent] / (24 * 60 * 60)}")
            #print(f"[STEP]  normalized_travel_time: {normalized_travel_time}")
            #print(f"[STEP]  self.occupancy_rate.get(curr_node, 0.0): {self.occupancy_rate.get(int(curr_node), 0.0)}")
            #print(f"[STEP]  state['occupancy']: {state['occupancy']}")
            #print(f"[STEP]  state['uptime']: {state['uptime']}")
            #print(f"[STEP]  state['fuel']: {state['fuel']}")
            #print(f"[STEP]  curr_node: {curr_node}")
            #print(f"[STEP]  next_node: {next_node}")
            #print(f"[STEP]  travel_time: {travel_time}")
            #print(f"[STEP]  self.node_to_idx[str(curr_node)]: {self.node_to_idx[str(curr_node)]}")
            #print(f"[STEP]  self.node_to_idx[str(next_node)]: {self.node_to_idx[str(next_node)]}")

            # 1. Crie o array de observação como antes
            obs_array = np.array([
                self.agent_times[agent] / (24 * 60 * 60),
                state["occupancy"],
                normalized_travel_time,
                self.future_demand_at_B.get(next_node, 0.0),
                state["uptime"],
                1.0 if state["maintenance_status"] == "ok" else 0.0,
                self.node_to_idx[str(curr_node)],
                self.node_to_idx[str(next_node)],
            ], dtype=np.float32)

            # APPLY CLIPPING HERE!
            clipped_obs = np.clip(
                obs_array,
                self.observation_space(agent).low,
                self.observation_space(agent).high
            )

            observations[agent] = clipped_obs

            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "count": self.steps[agent],
                "occupancy": state["occupancy"],
                "location": curr_node,
                "next_stop": next_node,
                "headways": self.headways.get(curr_node, []),
            }
            
            if self.agent_times[agent] >= 24 * 3600:
                print(f"[END OF DAY] Simulation ended at {self.agent_times[agent]/3600:.2f}h (>= 24h).")

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]

        # Update current episode metrics
        total_reward = sum(rewards.values())
        mean_reward = np.mean(list(rewards.values()))

        # Fairness (Gini coefficient sobre recompensas)
        def gini(x):
            if np.amin(x) < 0:
                x = np.array(x) - np.amin(x)  # shift values to be non-negative
            x = np.sort(np.array(x))
            n = len(x)
            if n == 0:
                return 0.0
            index = np.arange(1, n + 1)
            return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-8)

        fairness = 1 - gini(list(rewards.values())) if rewards else 0.0

        if not hasattr(self, "metrics_history"):  # Initialize metrics history if not present
            self.metrics_history = []
        
        self.metrics_history.append({
            "step": sum(self.steps.values()),  # Total steps taken by all agents
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "fairness": fairness
        })

        # === save metrics on CSV ===
        if not hasattr(self, "episode_counter"):
            self.episode_counter = 0
        self.episode_counter += 1

        env_steps = sum(self.steps.values())
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_counter,
                env_steps,
                mean_reward,
                total_reward,
                fairness
            ])


        # fusing terminations + truncations → dones
        dones = {a: (terminations[a] or truncations[a]) for a in rewards}
        dones["__all__"] = all(dones.values())

        for agent in self.agents: # Add episode reward and fairness to infos
            infos[agent] = {
                **infos.get(agent, {}),
                "episode_reward": rewards[agent],
                "mean_reward_episode": mean_reward,
                "fairness": fairness
            }

        return observations, rewards, dones, infos



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
    def action_space(self, agent): # Define the action space for each agent
        return spaces.Discrete(self.actions_amount)
    
    def generate_random_delay(self, start, target):
        try:
            # Finds the shortest path between the start and target
            shortest_path = nx.shortest_path(self.network, source=start, target=target, weight='weight')
            path_edges = list(zip(shortest_path, shortest_path[1:])) # Creates a list of edges from the shortest path
            total_time = 0

            if not path_edges:
                return  # No edges to delay

            # Calculates total time of the edges in the path
            for a, b in path_edges:
                a, b = str(a), str(b)
                edge_key = (min(a, b), max(a, b)) # Sorts the nodes of the edge to avoid duplication
                if edge_key in self.reward.waitTimeDict:
                    edge_time = self.reward.waitTimeDict[edge_key][0]
                    total_time += edge_time # Sums the waiting time of the edge
                else:
                    x = 0 
                    #print(f"[AVISO] Aresta {edge_key} não está no waitTimeDict!")

            average_time = total_time / len(path_edges)
            #print(f"Média de tempo das arestas do caminho ótimo: {average_time}")

            # Chooses a random edge in the path to apply the delay
            delay_u, delay_v = random.choice(path_edges)
            delay_u, delay_v = str(delay_u), str(delay_v) # Sorts the nodes of the edge to avoid duplication
            delay_edge_key = (min(delay_u, delay_v), max(delay_u, delay_v)) # Chosen edge for delay

            if delay_edge_key in self.reward.waitTimeDict: # Checks if the chosen edge is in waitTimeDict
                delay = average_time * 5  # Simulates heavy congestion
                self.dynamicDelays = {
                    delay_edge_key: delay # Chosen edge for delay with applied delay time
                }
                #print(f"Aresta atrasada: {delay_edge_key}, atraso aplicado: {delay}")
            else:
                #print(f"[ERRO] Aresta escolhida para atraso {delay_edge_key} não está no waitTimeDict.")
                self.dynamicDelays = {}

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.dynamicDelays = {}

    def get_nearest_service_center(self, current_node):
        # Finds the nearest service center node, not dynamic yet
        return self.service_center_node

# This is the base class for reward classes
class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError

# This is the base class for stop classes
class StopConditionBaseClass():
    def isTerminated(self, state, previousState, action, target, graph):
        raise NotImplementedError
        
class DefaultReward(RewardBaseClass):
    """
    Recompensa composta e NORMALIZADA para [-1, +1] por passo.
    Componentes:
      - occ: penaliza fora do intervalo ideal (quadrática, 0..1, sinal -)
      - uptime: bônus direto (0..1, sinal +)
      - sync: mede regularidade dos headways vs alvo (0..1, sinal +)
      - efficiency: 1 - (estimated/expected) truncado (0..1, sinal +)
    """
    def __init__(self, waitTimeDict=None, reward_weights=None, occupancy_range=(0.6, 0.9),
                 target_headway_seconds: float = 600.0,  # 10 minutos
                 max_sync_rel_std: float = 1.0          # >1 é truncado
                 ):
        super().__init__()
        # self.waitTimeDict pode ser usado se necessário para outras métricas
        self.waitTimeDict = waitTimeDict or {}

        # Pesos ajustáveis (soma não precisa dar 1; fazemos média ponderada)
        self.reward_weights = reward_weights or {
            "occ_penalty": 0.6,        # menor que 1 para não dominar
            "uptime_bonus": 0.6,
            "sync_score": 0.3,         # começe baixo; aumente depois
            "energy_efficiency": 0.5
        }

        self.occupancy_range = occupancy_range
        self.target_headway = float(target_headway_seconds)
        self.max_sync_rel_std = float(max_sync_rel_std)

    def _occ_component(self, occupancy: float) -> float:
        """
        Retorna um valor em [0, 1], onde 0 = perfeito no intervalo ideal; 1 = muito fora.
        Depois aplicamos sinal negativo ao compor o reward.
        """
        min_occ, max_occ = self.occupancy_range
        if occupancy < min_occ:
            return min(1.0, (min_occ - occupancy) ** 2 / (min_occ ** 2 + 1e-8))
        if occupancy > max_occ:
            return min(1.0, (occupancy - max_occ) ** 2 / ((1.0 - max_occ) ** 2 + 1e-8))
        return 0.0

    def _sync_component(self, headways: list) -> float:
        """
        Mede regularidade em [0, 1]: 1 = perfeito (intervalos bem próximos ao alvo),
        0 = muito irregular (desvio relativo >= max_sync_rel_std).
        """
        if not headways or len(headways) < 3:
            return 0.0  # Sem informação suficiente para avaliar regularidade

        # intervalos em segundos
        intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)]
        # remove ruídos/intervalos inválidos
        intervals = [x for x in intervals if x > 0]
        if len(intervals) < 2:
            return 0.0

        # Desvio RMS versus alvo
        diffs = [(x - self.target_headway) for x in intervals]
        mean_sq = sum(d * d for d in diffs) / len(diffs)
        rms = mean_sq ** 0.5  # em segundos

        # Desvio relativo normalizado (0=perfeito, 1=limite ruim)
        rel = min(1.0, rms / (self.max_sync_rel_std * self.target_headway + 1e-8))

        # Converter para "score" em [0,1], onde 1 é bom
        return 1.0 - rel

    def _efficiency_component(self, estimated_time: float, expected_time: float) -> float:
        """
        Eficiência de viagem em [0,1]. 1 = igual/menor que o esperado; 0 = pior que o esperado.
        """
        if expected_time <= 0:
            return 0.0
        ratio = estimated_time / (expected_time + 1e-8)
        return float(np.clip(1.0 - ratio, 0.0, 1.0))

    def getReward(
        self,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None
    ):
        # Componentes normalizadas
        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))

        # Combinação ponderada (mantendo cada termo em [-1, +1])
        # occ_pen entra com sinal NEGATIVO
        w = self.reward_weights
        reward = (
            -w["occ_penalty"] * occ_pen +
             w["uptime_bonus"] * uptime +
             w["sync_score"] * sync +
             w["energy_efficiency"] * eff
        )

        # Normaliza pela soma dos pesos para manter magnitude em ~[-1, +1]
        weight_sum = (abs(w["occ_penalty"]) + w["uptime_bonus"] + w["sync_score"] + w["energy_efficiency"])
        if weight_sum > 0:
            reward = reward / weight_sum

        # Clip final para segurança numérica
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward


# This is the default stop class, which terminates the episode when the agent reaches the target node or takes the SERVICE_CENTER action
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2
