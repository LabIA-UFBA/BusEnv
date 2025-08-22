import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gymnasium import spaces
import numpy as np
import random
import pickle
import gymnasium.utils.seeding  # import seeding
from gymnasium.spaces import Discrete

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


        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._num_agents = num_agents

        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        self.node_to_idx = {str(n): i for i, n in enumerate(self.network.nodes)} # Map nodes to indices

        # self.node_to_idx = {node: idx for idx, node in enumerate(sorted(self.network.nodes()))}  # Map nodes to indices
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}  # Map indices to nodes

        # Create agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(num_agents))))  # Map agent names to indices

        # Individual states
        self.states = {}           # Stores the current node of each agent
        self.targets = {}          # Stores the target node of each agent
        self.steps = {}            # Stores the step count for each agent
        self.delays = {}           # Stores the delays for each agent
        self.estimated_times = {}  # Stores the estimated travel time for each agent
        self.expected_times = {}   # Stores the optimal (ideal) travel time for each agent

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes

        # === replace DataFrame-derived data with ready observations ===

        self.avg_travel_time_AB = avg_travel_time_AB           # (nodeA, nodeB) -> average time
        self.future_demand_at_B = future_demand_at_B           # node_id -> estimated demand
        self.occupancy_rate = occupancy_rate                   # agent_id -> occupancy rate (0.0 to 1.0)
        self.uptime_normalized = uptime_normalized             # agent_id -> uptime (0.0 to 1.0)

        # Global simulation clock (starts at 6:00 AM)
        self.current_time = 6 * 60 * 60  # starts at 6:00 AM, in seconds
        self.headways = {}         # Arrival history by point
        self.sync_stats = {}       # Vehicle regularity at stops

        self.service_center_node = random.choice(list(self.network.nodes))

        # Internal state of agents
        self.agents = {}
        for agent_id in self.possible_agents:
            self.agents[agent_id] = {
                "location": None,
                "occupancy": self.occupancy_rate.get(agent_id, 0.0),  # Initializes with real value if available
                "uptime": self.uptime_normalized.get(agent_id, 1.0),  # Same
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": None,                  # List of nodes
                "route_idx": 0,                  # Starts at the beginning of the route
                "needs_service": False          # Flag for maintenance
            }

        # Reward parameters
        self.reward_weights = {
            "occ_penalty": 1.0,
            "uptime_bonus": 1.0,
            "sync_score": 1.0,
            "energy_efficiency": 1.0
        }
        self.occupancy_range = (0.6, 0.9)  # Ideal occupancy range

        # Just for future compatibility with wrappers that expect .observation_spaces and .action_spaces
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

        self.default_travel_time = np.mean(list(self.avg_travel_time_AB.values())) # Default average travel time
        self.max_travel_time = 3250.0 # Maximum time found in the dataset (5 minutes and 25 seconds)

        self.real_routes = real_routes or {}
        self.route_metadata = route_metadata or {}
        self.agent_routes = {}  # Maps agent IDs to their fixed routes

 



    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]  # Resets the agents to the initial list
        self.states = {} # Stores the states of the agents
        self.targets = {} # Stores the targets of the agents
        self.steps = {} # Stores the step count for each agent
        self.delays = {} # Stores the delays for each agent
        self.estimated_times = {} # Stores the estimated times for each agent
        self.expected_times = {} # Stores the expected times for each agent
        self.current_time = 6 * 60 * 60  # Resets the global clock to 6:00 AM (in seconds)
        self.headways = {}  # Working directly with aggregated data
        self.sync_stats = {}  # Vehicle regularity at stops
        self.agents_state = {} # Internal state of agents
        observations = {}
        infos = {}

        for agent in self.agents:
            # === Fixed route for each agent ===
            if agent not in self.agent_routes: # If the agent doesn't have a fixed route, choose randomly
                trip_id, path = random.choice(list(self.real_routes.items())) # Choose a random route
                print(f"[DEBUG] Route chosen for {agent} (Trip ID: {trip_id}): {path}")
                self.agent_routes[agent] = path

            path = self.agent_routes[agent] # Gets the fixed route for the agent
            if len(path) < 2: # Checks if the route is valid
                raise ValueError(f"Invalid route for {agent}: {path}")

            initial = path[0] # Starting point of the agent
            target = path[-1] # Destination point of the agent

            self.agents_state[agent] = { # Internal state of the agent
                "location": initial,
                "occupancy": float(self.occupancy_rate.get(agent, 0.0)), # Agent occupancy rate
                "uptime": float(self.uptime_normalized.get(agent, 1.0)), # Agent uptime
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": path,
                "route_idx": 0
            }

            self.states[agent] = initial # Sets the initial state of the agent to the starting point
            self.targets[agent] = target # Sets the target of the agent to the destination point
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}

            # Expected time based on the route
            self.expected_times[agent] = sum( # Calculates the expected time for the route
                self.avg_travel_time_AB.get((path[i], path[i + 1]), self.default_travel_time) # Average travel time
                for i in range(len(path) - 1) # Gets the average time between each pair of nodes in the route
            )

            next_node = path[1] # Next node in the route
            travel_time = self.avg_travel_time_AB.get((initial, next_node), self.default_travel_time) # Average travel time between the initial node and the next node
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0) # Normalizes the travel time

            observations[agent] = np.array([
                self.current_time / (24 * 60 * 60), # Time of day normalized (0.0 to 1.0)
                self.occupancy_rate.get(agent, 0.0), # Agent occupancy rate
                normalized_travel_time, # Normalized average travel time
                self.future_demand_at_B.get(next_node, 0.0), # Future demand at the next node
                self.uptime_normalized.get(agent, 1.0), # Normalized uptime (0.0 to 1.0)
                1.0 if self.agents_state[agent]["maintenance_status"] == "ok" else 0.0, # Maintenance status (0.0 or 1.0)
                self.node_to_idx[str(initial)], # Maps the initial node to the index
                self.node_to_idx[str(next_node)], # Maps the next node to the index
            ], dtype=np.float32)

            infos[agent] = {
                "chosen_route": path, # Chosen route for the agent
                "expected_time": self.expected_times[agent]
            }

        return observations, infos


    
    def step(self, actions):
        if not actions: # If there are no actions, return empty observations
            self.agents = []
            return {}, {}, {}, {}, {} # Observations, rewards, terminations, truncations, infos

        observations = {} # Observations for each agent
        rewards = {} # Rewards for each agent
        terminations = {} # Terminations for each agent
        truncations = {} # Truncations for each agent
        infos = {}

        for agent in self.agents: # Checks if the agent is active
            self.steps[agent] += 1 # Increments the agent's step counter
            state = self.agents_state[agent] # Internal state of the agent
            route = state["route"] # Route of the agent
            idx = state["route_idx"] # Current index in the route

            if idx >= len(route): # Verifies if the agent has exceeded the route
                print(f"[ERRO] Agente {agent} excedeu a rota. IDX={idx}, TAM={len(route)}")
                terminations[agent] = True
                truncations[agent] = False
                rewards[agent] = -1.0
                continue

            curr_node = route[idx] # Current node of the agent
            self.states[agent] = curr_node  # Ensures synchronization

            action = actions[agent] # Action chosen by the agent
            print(f"[DEBUG] action: {action}")

            if action == 0:  # WAIT
                reward = -0.1  # Penalty for waiting
                elapsed = 60.0  # assume 1 minute of waiting
                self.current_time += elapsed # Updates the global time
                state["uptime"] = max(state["uptime"] - elapsed / (12 * 3600), 0.0) # update the uptime
                state["fuel"] = max(state["fuel"] - elapsed / 300.0, 0.0) # Updates fuel
                terminated = False
                truncated = False

            elif action == 1:  # MOVE
                if idx + 1 < len(route): # verifies if there is a next node in the route
                    next_node = route[idx + 1] # Move to next node in the route
                    travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time) # Average travel time between the current node and the next node
                    prev_occ = state.get("occupancy", 0.0) # Previous occupancy rate

                    if(curr_node, next_node) in self.occupancy_rate:
                        expected_occ = self.occupancy_rate[(curr_node,next_node)] # Expected occupancy rate between the current node and the next node
                        
                        delta_occ = expected_occ - prev_occ # Difference between expected and previous occupancy
                        new_occ = prev_occ + delta_occ # New occupancy is previous occupancy plus difference
                        # new_occ = max(min(new_occ, 1.0), 0.0) # Ensures the new occupancy is between 0.0 and 1.0
                        occupancy = new_occ # Updates the agent's occupancy rate
                    else:
                        occupancy = prev_occ # If no occupancy rate is defined, keeps the previous occupancy

                    # occupancy = self.occupancy_rate.get((curr_node, next_node), 0.0) # Occupancy rate between the current node and the next node

                    self.current_time += travel_time # Updates the global time
                    self.estimated_times[agent] += travel_time # Updates the agent's estimated time
                    self.agents_state[agent]["route_idx"] += 1 # Advances the agent's route index
                    self.agents_state[agent]["occupancy"] = occupancy # Updates the agent's occupancy rate

                    # Updates uptime and fuel
                    state["uptime"] = max(state["uptime"] - travel_time / (12 * 3600), 0.0) # Updates uptime
                    state["fuel"] = max(state["fuel"] - travel_time / 300.0, 0.0) # Updates fuel

                    self.states[agent] = next_node # Updates the agent's state to the next node

                    if next_node not in self.headways: # If the next node does not have a arrival history
                        self.headways[next_node] = []
                    self.headways[next_node].append(self.current_time) # Adds the current time to the arrival history of the next node

                    print(f"[STEP Action MOVE] self.estimated_times[agent]: {self.estimated_times[agent]}")
                    print(f"[STEP Action MOVE] self.expected_times[agent]: {self.expected_times[agent]}")
                    print(f"[STEP Action MOVE] self.headways[next_node]: {self.headways[next_node]}")
                    print(f"[STEP Action MOVE] travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time): {travel_time}")
                    print(f"[STEP]  self.current_time: {self.current_time}")
                    print(f"[STEP]  self.occupancy_rate.get((curr_node, next_node), 0.0): {self.occupancy_rate.get((curr_node, next_node), 0.0)}")
                    print(f"[STEP]  state['uptime']: {state['uptime']}")

                    reward = self.reward.getReward( # Calculates the reward based on the current state
                        new_state=next_node, # New state of the agent
                        previous_state=curr_node, # Previous state of the agent
                        action=action, # Action taken by the agent
                        target=route[-1], # Target node of the route
                        network=self.network,
                        estimated_time=self.estimated_times[agent], # Estimated time of the agent
                        expected_time=self.expected_times[agent], # Expected time of the route
                        delay=0,
                        agent_state=state,
                        headways=self.headways[next_node]
                    )

                    terminated = next_node == route[-1] # finishes if the agent reached the end of the route
                    truncated = self.steps[agent] >= self.max_steps # truncates if the agent exceeded the maximum number of steps

                else:
                    reward = -1.0  # already at the end of the route, cannot move
                    terminated = False
                    truncated = False

            elif action == 2:  # SERVICE_CENTER
                sc_node = self.get_nearest_service_center(curr_node) # Gets the nearest service center node
                travel_time = self.avg_travel_time_AB.get((curr_node, sc_node), self.default_travel_time) # Travel time to the service center
                self.current_time += travel_time # Updates the global time
                self.estimated_times[agent] += travel_time # Updates the agent's estimated time

                state["fuel"] = 100.0 # Refuels the agent
                state["uptime"] = 1.0 # Resets the uptime
                state["maintenance_status"] = "ok" # Resets the maintenance status
                self.states[agent] = sc_node # Updates the agent's state to the service center

                reward = -0.5 # Penalizes the action of going to the service center
                terminated = False
                truncated = self.steps[agent] >= self.max_steps # truncates if the agent exceeded the maximum number of steps

            else:
                reward = -10.0
                terminated = False
                truncated = True

            # Updated observation
            route_idx = self.agents_state[agent]["route_idx"]
            curr_node = self.states[agent]
            next_node = (
                route[route_idx + 1] if route_idx + 1 < len(route) else curr_node
            )

            travel_time = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)
            normalized_travel_time = min(travel_time / self.max_travel_time, 1.0)

            print(f"[STEP] self.future_demand_at_B.get(next_node, 0.0): {self.future_demand_at_B.get(next_node, 0.0)}")
            print(f"[STEP]  self.current_time / (24 * 60 * 60): {self.current_time / (24 * 60 * 60)}")
            print(f"[STEP]  normalized_travel_time: {normalized_travel_time}")
            print(f"[STEP]  self.occupancy_rate.get((curr_node, next_node), 0.0): {self.occupancy_rate.get((curr_node, next_node), 0.0)}")
            print(f"[STEP]  state['uptime']: {state['uptime']}")
            print(f"[STEP]  curr_node: {curr_node}")
            print(f"[STEP]  next_node: {next_node}")
            print(f"[STEP]  travel_time: {travel_time}")
            print(f"[STEP]  self.node_to_idx[str(curr_node)]: {self.node_to_idx[str(curr_node)]}")
            print(f"[STEP]  self.node_to_idx[str(next_node)]: {self.node_to_idx[str(next_node)]}")


            observations[agent] = np.array([
                self.current_time / (24 * 60 * 60), # Normalized time of day (0.0 to 1.0)
                self.occupancy_rate.get((curr_node, next_node), 0.0), # Occupancy rate between current node and next node
                normalized_travel_time, # Normalized average travel time
                self.future_demand_at_B.get(next_node, 0.0), # Future demand at next node
                state["uptime"], # Normalized uptime (0.0 to 1.0)
                1.0 if state["maintenance_status"] == "ok" else 0.0, # Maintenance status (0.0 or 1.0)
                self.node_to_idx[str(curr_node)], # Maps the current node to the index
                self.node_to_idx[str(next_node)] # Maps the next node to the index
            ], dtype=np.float32)

            rewards[agent] = reward # Calculates the reward for the agent
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "count": self.steps[agent], # Agent step counter
                "occupancy": state["occupancy"], # Agent occupancy rate
                "location": curr_node, # Agent's current location
                "next_stop": next_node, # Next node in the agent's route
                "headways": self.headways.get(curr_node, []), # Arrival history at the current node
            }

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])] # Remove terminated or truncated agents

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

# This is the default reward class, which calculates the reward based on total time and number of trips
class DefaultReward(RewardBaseClass):
    def __init__(self, waitTimeDict=None, reward_weights=None, occupancy_range=(0.6, 0.9)):
        super().__init__()
        #with open('./output/combined_sum_amount.pkl', 'rb') as f:
        #    self.waitTimeDict = pickle.load(f)

        self.reward_weights = reward_weights or {
            "occ_penalty": 1.0,         # W1
            "uptime_bonus": 1.0,        # W2
            "sync_score": 1.0,          # W3
            "energy_efficiency": 1.0    # W4
        }

        self.occupancy_range = occupancy_range  # ideal range (ex: 60% a 90%)

    def getReward(self, new_state, previous_state, action, target, network, estimated_time, expected_time, delay, agent_state=None, headways=None):
        reward = 0.0

        # 1. Penalty for occupancy outside ideal range
        if agent_state is not None:
            occupancy = agent_state.get("occupancy", 0.0)
            min_occ, max_occ = self.occupancy_range
            if occupancy < min_occ:
                occ_penalty = (min_occ - occupancy) ** 2 # Penalizes if occupancy is below minimum
            elif occupancy > max_occ:
                occ_penalty = (occupancy - max_occ) ** 2 # Penalizes if occupancy is above maximum
            else:
                occ_penalty = 0.0
            reward -= self.reward_weights["occ_penalty"] * occ_penalty

        # 2. Bonus for uptime
        if agent_state is not None:
            uptime = agent_state.get("uptime", 1.0) # normalized uptime (0.0 to 1.0)
            reward += self.reward_weights["uptime_bonus"] * uptime # Bonus proportional to uptime

        # 3. Regularity (synchronization/headways)
        sync_score = 0.0
        if headways and len(headways) > 1:
            intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)] # Calculates intervals between arrivals
            if intervals: # Check if intervals list is not empty
                avg_interval = sum(intervals) / len(intervals) # Average interval
                std = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5 # Standard deviation of intervals
                sync_score = -std  # Penalizes irregularity
                reward += self.reward_weights["sync_score"] * sync_score # Bonus proportional to regularity

        # 4. Energy efficiency (estimated vs expected)
        if expected_time > 0:
            travel_efficiency = max(0.0, 1 - (estimated_time / expected_time))
            reward += self.reward_weights["energy_efficiency"] * travel_efficiency
        else:
            reward += 0.0  # No bonus if there is no expected time

        return reward


# This is the default stop class, which terminates the episode when the agent reaches the target node or takes the SERVICE_CENTER action
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2
