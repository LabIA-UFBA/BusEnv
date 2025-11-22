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
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2, agents_per_route=None,
                use_only_mean_data=None, stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None,
                render_mode=None, avg_travel_time_AB=None, future_demand_at_B=None,
                occupancy_rate=None, uptime_normalized=None,
                real_routes=None, route_metadata=None):  

        # --- Basic configuration ---
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps 
        self.render_mode = render_mode

        # --- Agents ---
        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()  # <- MARLlib needs this
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # --- Stop/Reward classes ---
        self.stop = DefaultStopClass() if stopClass is None else stopClass
        self.reward = DefaultReward() if rewardClass is None else rewardClass

        # --- Node index mapping ---
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
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents}  # Every agent starts at 6:00 AM 
        self.headways = {}
        self.sync_stats = {}

        # --- Agent presence tracking ---
        # Maps each node (stop) to the list of agents currently there
        self.node_occupancy = {}  

        # Stores last known positions of all agents (to detect proximity changes)
        self.agent_positions = {agent: None for agent in self.possible_agents}

        # Threshold (in seconds) for spacing between agents in the same route
        self.min_headway_time = 300.0  # 5 minutes minimum gap between agents on same route


        self.service_center_node = random.choice(list(self.network.nodes))

        # --- Internal structure of agents ---
        self.agent_states = {}
        for agent_id in self.possible_agents:
            self.agent_states[agent_id] = {
                "location": None,  # current location of the agent
                "occupancy": 0.0,
                "uptime": 1.0,
                "fuel": 100.0,
                "maintenance_status": "ok", # can be "ok", "needs_service", etc.
                "schedule": [],
                "route": None,
                "route_idx": 0,
                "needs_service": False, # indicates if the agent needs maintenance
                "going_forward": True, # indicates direction along the route
                "status": "active",  # state is always active
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
        # Ensure the action space supports what we need
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        # --- Defaults / limits ---
        if self.avg_travel_time_AB:
            self.default_travel_time = np.mean(list(self.avg_travel_time_AB.values()))
        else:
            self.default_travel_time = 1.0

        # --- Daily Data Control ---
        self.daily_data_path = "/mnt/ssd1/wesley/BusEnv/src/training_observation/daily"  # Path to daily data files
        self.daily_files = sorted([
            f for f in os.listdir(self.daily_data_path)
            if f.startswith("daily_data_") and f.endswith(".pkl")
        ])
        self.current_day_index = 0
        self.total_days = len(self.daily_files)
        print(f"[DAILY DATA] {self.total_days} days detected for training.")

        # --- Simulation control flags ---
        self._advance_day = False
        self.day_done = False          # Ensures consistent state for daily progression
        self.sim_done = False          # Used for final termination when all days end
        self.simulated_days = 0        # Total simulated days counter

        # --- Limits and constants ---
        self.max_travel_time = 3250.0
        self.max_capacity = 80

        # --- Route handling ---
        self.real_routes = real_routes or {}
        self.route_metadata = route_metadata or {}
        self.agent_routes = {}
        self.agents_per_route = agents_per_route  # Number of agents sharing the same route
        self.fixed_agent_routes = None

        # --- Coordination control for agents on same route ---
        # Stores the last time an agent from each route advanced (for coordination)
        self.route_last_move_time = {}

        # Controls which agent is the "leader" on each route (the first to move)
        self.route_leader = {}


        # --- Logging and metrics ---
        self.metrics_file = "env_metrics.csv"  # To log metrics for analysis
        self._printed_day_end = set()
        self.last_logged_day = -1
        self.episode_step_counter = 0  # Counts total environment steps per episode
        
        
        self.use_only_mean_data = use_only_mean_data   # 0 = use only mean data, 1 = use daily data

        # Create metrics file if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "env_steps", "mean_reward", "total_reward", "fairness"])
        
        # --- Fallback averages for missing daily data ---
        self.avg_travel_time_AB_mean = avg_travel_time_AB or {} 
        self.future_demand_at_B_mean = future_demand_at_B or {} 
        self.occupancy_rate_mean = occupancy_rate or {}
        self.uptime_normalized_mean = uptime_normalized or {}


    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        print("================ RESETTING ENVIRONMENT ================")
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # --- Inicialização persistente ---
        if not hasattr(self, "current_day_index"):
            self.current_day_index = 0
        if not hasattr(self, "simulated_days"):
            self.simulated_days = 0
        if not hasattr(self, "day_done"):
            self.day_done = False
        if not hasattr(self, "sim_done"):
            self.sim_done = False

        # --- Estruturas básicas ---
        self.agents = self.possible_agents[:]
        self.agent_done = {a: False for a in self.possible_agents}
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents}  # start at 6 AM every new day
        self.headways = {}
        self.sync_stats = {}
        self.agent_states = {}
        self.infos = {}
        observations = {}

        # --- Reset agent coordination and presence tracking ---
        self.node_occupancy = {node: [] for node in self.network.nodes}  # which agents are currently at each node
        self.agent_positions = {agent: None for agent in self.possible_agents}
        self.route_last_move_time = {route_id: 0.0 for route_id in self.real_routes.keys()}
        self.route_leader = {}  # will be updated dynamically in step()


        # --- Move on to the next day if the previous one has ended ---
        print(f"📅 [ENV] Current day: {self.current_day_index + 1}/{self.total_days}")
        print(f"📆 [ENV] Total days in simulation: {self.total_days}")
        print(f"🔄 [ENV] Day done flag: {self.day_done}")
        print(f"🗓️ [ENV] Total simulated days so far: {self.simulated_days}")

        if (self.day_done or getattr(self, "agents_finished_previous_day", False)) and self.total_days > 0:
            self.current_day_index = (self.current_day_index + 1) % self.total_days
            self.simulated_days += 1
            next_file = self.daily_files[self.current_day_index]
            next_date = next_file.replace("daily_data_", "").replace(".pkl", "")
            print(f"\n🔁 [ENV] Advancing to next day: {next_date} ({self.current_day_index + 1}/{self.total_days})")
            print(f"📆 [ENV] Total simulated days so far: {self.simulated_days}")
            self.day_done = False  # reset flag
            self.agents_finished_previous_day = False # reset flag

        # --- Loads daily data ---
        try:
            self.load_current_day_data()
        except Exception as e:
            print(f"⚠️ Error loading daily data: {e}. Using fallback averages.")
            self._use_fallbacks()

        # --- Initialization of fixed routes (if they don't already exist) ---
        if not hasattr(self, "fixed_agent_routes") or self.fixed_agent_routes is None:
            self.fixed_agent_routes = {}
            num_agents = len(self.agents)
            routes = list(self.real_routes.items())

            # CHANGE: function to decide number of agents per route (based on length)
            def _agents_for_path(path_len: int, default_agents: int) -> int:
                if path_len < 15:
                    return 1
                if path_len <= 30:
                    return 1 # Antes era 2 
                # >30 uses the user's default (ensures >=1)
                return max(1, int(default_agents))

            default_agents_per_route = getattr(self, "agents_per_route", 1)

            agent_routes_assignment = []
            agent_idx = 0

            # CHANGE: dynamic allocation based on route length
            for trip_id, path in routes:
                if agent_idx >= num_agents:
                    break

                k = _agents_for_path(len(path), default_agents_per_route)
                # does not exceed the total remaining agents
                k = min(k, num_agents - agent_idx)
                if k <= 0:
                    break

                assigned_agents = self.agents[agent_idx:agent_idx + k]
                agent_routes_assignment.append((trip_id, path, assigned_agents))

                for agent in assigned_agents:
                    self.fixed_agent_routes[agent] = path

                # DEBUG: show decision per route
                route_preview = " → ".join(str(n) for n in path[:5]) + (" → ..." if len(path) > 5 else "")
                print(f"[ROUTE SPLIT] trip={trip_id} | len={len(path)} | assigned={k} | agents={', '.join(assigned_agents)} | path={route_preview}")

                agent_idx += k

            print("\n=== [ROUTE ASSIGNMENT DEBUG - INITIALIZED ONCE] ===")
            for trip_id, path, assigned_agents in agent_routes_assignment:
                route_preview = " → ".join(str(n) for n in path[:5])
                if len(path) > 5:
                    route_preview += " → ..."
                print(f"Trip ID: {trip_id:<10} | Agents: {', '.join(assigned_agents)} | "
                      f"Route length: {len(path):<3} | Path: {route_preview}")
            print("====================================================\n")

        # --- Usa sempre as rotas fixas ---
        self.agent_routes = self.fixed_agent_routes

        # --- Inicializa estados de cada agente ---
        for agent in self.agents:
            if agent not in self.agent_routes:
                raise ValueError(f"[RESET ERROR] Agent {agent} did not receive a route!")

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
                "status": "active",
                "going_forward": True,
            }

            # --- Register initial position for coordination ---
            self.agent_positions[agent] = initial
            if initial not in self.node_occupancy:
                self.node_occupancy[initial] = []
            self.node_occupancy[initial].append(agent)

            # If this is the first agent on this route, mark as leader
            route_id = [k for k, v in self.real_routes.items() if v == path]
            if route_id:
                rid = route_id[0]
                if rid not in self.route_leader:
                    self.route_leader[rid] = agent


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

            clipped_obs = np.clip(
                obs_array,
                self.observation_space(agent).low,
                self.observation_space(agent).high,
            )

            observations[agent] = clipped_obs
            self.infos[agent] = {
                "chosen_route": path,
                "expected_time": self.expected_times[agent],
                "status": "active",
            }

        # --- Métricas de episódio ---
        self.current_episode_metrics = {
            "rewards": {agent: 0.0 for agent in self.agents},
            "steps": {agent: 0 for agent in self.agents},
            "done": False
        }

        # --- Debug: presence summary ---
        active_points = {node: agents for node, agents in self.node_occupancy.items() if agents}
        print(f"📍 [RESET DEBUG] Agent initial positions per node: {active_points}")

        print("✅ [RESET COMPLETE] All agents initialized with status='active'.")
        return observations
    
    def step(self, actions):
        """
        Executes a multi-agent simulation step.
        Includes active PARK-blocking logic so agents can only park after 24h,
        travel time clamping, and robust day-end handling.
        """

        # --- Output containers ---
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Safety: ensure agent_states exists
        if not hasattr(self, "agent_states"):
            raise RuntimeError("agent_states not initialized. Call reset() before step().")

        # Global limits
        TRAVEL_TIME_CAP = 1800.0   # 30 minutes max per edge
        END_OF_DAY = 24 * 3600.0      # End of the day in seconds

        # --- Reset dynamic presence tracking for this step ---
        for node in self.node_occupancy:
            self.node_occupancy[node] = []  # clear per-step occupancy

        #if self.episode_step_counter % 100 == 0:
            #print(f"[DEBUG] Step {self.episode_step_counter}: Cleared node_occupancy map.")

        # Record latest positions as we go
        step_node_positions = {}


        # Loop through all possible agents (fixed set)
        for agent in self.possible_agents:
            state = self.agent_states[agent]

            # Skip finished agents
            if state.get("status") == "finished":
                observations[agent] = np.zeros_like(self.observation_space(agent).low, dtype=np.float32)
                rewards[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {"status": "finished"}
                continue

            # Get action (default = WAIT)
            action = actions.get(agent, 0)

            # --- Active agent ---
            self.steps[agent] += 1
            route = state["route"]
            idx = state["route_idx"]
            curr_node = route[idx]

            # === ACTION 0: WAIT ===
            if action == 0:
                reward = -0.1 # Antes era  -0.1
                elapsed = 60.0  # 1 minute wait
                self.agent_times[agent] += elapsed
                self.estimated_times[agent] += elapsed
                state["uptime"] = max(state["uptime"] - elapsed / (12 * 3600), 0.0)
                state["fuel"] = max(state["fuel"] - elapsed / 300.0, 0.0)

            # === ACTION 1: MOVE ===
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

                # Clamp travel time to avoid large jumps
                travel_time_raw = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)
                travel_time = min(travel_time_raw, TRAVEL_TIME_CAP)

                self.agent_times[agent] += travel_time
                self.estimated_times[agent] += travel_time

                # Update occupancy
                prev_occ = state.get("occupancy", 0.0)
                if int(curr_node) in self.occupancy_rate:
                    expected_occ = self.occupancy_rate[int(curr_node)]
                    alpha = 0.5
                    occupancy = (1 - alpha) * prev_occ + alpha * expected_occ
                    occupancy = max(0.0, min(occupancy, 1.0))
                else:
                    occupancy = prev_occ

                state["occupancy"] = occupancy
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

            # === ACTION 2: SERVICE CENTER ===
            elif action == 2:
                sc_node = self.get_nearest_service_center(curr_node)
                try:
                    path = nx.shortest_path(
                        self.network,
                        source=curr_node,
                        target=sc_node,
                        weight=lambda u, v, d: self.avg_travel_time_AB.get((u, v), self.default_travel_time)
                    )

                    total_travel_time = 0.0
                    total_fuel_cost = 0.0
                    for u, v in zip(path[:-1], path[1:]):
                        edge_time_raw = self.avg_travel_time_AB.get((u, v), self.default_travel_time)
                        edge_time = min(edge_time_raw, TRAVEL_TIME_CAP) * 0.3
                        total_travel_time += edge_time
                        total_fuel_cost += edge_time / 300.0

                except nx.NetworkXNoPath:
                    reward = -10# antes era -10.0
                else:
                    if state["fuel"] < total_fuel_cost:
                        reward = -20 # em vez de -20
                        print("total_fuel_cost: ", total_fuel_cost)
                        print("state[fuel]: ", state["fuel"])
                        print("VEIO PRO SERVICE CENTER SEM CONDIÇÃO DE CHEGAR")
                    else:
                        self.agent_times[agent] += total_travel_time
                        self.estimated_times[agent] += total_travel_time
                        state["fuel"] = max(state["fuel"] - total_fuel_cost, 0.0)
                        state["uptime"] = max(state["uptime"] - total_travel_time / (12 * 3600), 0.0)
                        state["fuel"] = 100.0
                        state["uptime"] = 1.0
                        state["maintenance_status"] = "ok"
                        self.states[agent] = sc_node
                        reward = -1.0 * (1 + total_travel_time / 600.0) # antes era -1.0 * 
                        #reward = 0

            # === INVALID ACTION ===
            else:
                reward = -10.0

            # Presence and coordination tracking
            current_pos = self.states[agent]
            self.agent_positions[agent] = current_pos

            if current_pos not in self.node_occupancy:
                self.node_occupancy[current_pos] = []
            self.node_occupancy[current_pos].append(agent)

            #if len(self.node_occupancy[current_pos]) > 1:  # detect overlap
               # print(f"[DEBUG] Overlap at node {current_pos}: {self.node_occupancy[current_pos]}")

            route_id = [k for k, v in self.real_routes.items() if v == state["route"]]
            if route_id:
                rid = route_id[0]
                self.route_last_move_time[rid] = self.agent_times[agent]

            # === End-of-day to agents, putting the finished status ===
            if self.agent_times[agent] >= END_OF_DAY:
                state["status"] = "finished"
                observations[agent] = np.zeros_like(self.observation_space(agent).low, dtype=np.float32)
                rewards[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {"status": "finished", "reason": "24h_limit"}
                continue

            # === Observation update ===
            route_idx = self.agent_states[agent]["route_idx"]
            curr_node = self.states[agent]
            next_node = route[route_idx + 1] if route_idx + 1 < len(route) else curr_node

            tt_next_raw = self.avg_travel_time_AB.get((curr_node, next_node), self.default_travel_time)
            tt_next = min(tt_next_raw, TRAVEL_TIME_CAP)
            normalized_travel_time = min(tt_next / self.max_travel_time, 1.0)

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

            observations[agent] = np.clip(
                obs_array,
                self.observation_space(agent).low,
                self.observation_space(agent).high
            )

            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {"status": "active"}

            # Update route leader dynamically
            if route_id:
                rid = route_id[0]
                leader = self.route_leader.get(rid)
                if leader is None or self.agent_times[agent] > self.agent_times.get(leader, 0.0):
                    self.route_leader[rid] = agent
                    #print(f"[DEBUG] {agent} is now leader of route {rid}")

        # === Global post-processing ===
        all_finished = all(self.agent_times[a] >= END_OF_DAY for a in self.possible_agents)

        dones = {a: all_finished for a in self.possible_agents}
        dones["__all__"] = all_finished

        if all_finished:
            print("🌙 [ENV] All agents finished 24h — day finished. Awaiting reset() to advance to next day.")
            self.day_done = True
            self.agents_finished_previous_day = True
        else:
            self.day_done = False

        if self.episode_step_counter % 50 == 0:
            active_nodes = {node: ags for node, ags in self.node_occupancy.items() if ags}
            #print(f"[DEBUG] Node occupancy snapshot: {active_nodes}")

        
        #print(f"[STEP SUMMARY] Active: {sum(1 for a in self.possible_agents if self.agent_states[a]['status'] == 'active')} "
        #    f"| Parked: {sum(1 for a in self.possible_agents if self.agent_states[a]['status'] == 'parked')} "
        #    f"| Done flag: {all_parked}")

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
    
    def load_current_day_data(self):
        """Loads the data for the current day and falls back to averages if necessary."""

        if getattr(self, "use_only_mean_data", 0) == 1:
            print("📘 [ENV] Using ONLY MEAN DATA (daily files ignored).")
            self._load_mean_values()
            return

        if self.total_days == 0:
            print("⚠️ No daily data files found. Using default averages.")
            self._use_fallbacks()
            return

        file_path = os.path.join(self.daily_data_path, self.daily_files[self.current_day_index])
        with open(file_path, "rb") as f:
            day_data = pickle.load(f)

        print(f"\n📅 [ENV] Loading daily data for {day_data.get('date', 'unknown')} "
            f"({self.current_day_index + 1}/{self.total_days})")

        # Load with fallback preservation
        self.avg_travel_time_AB = day_data.get("avg_travel_times", self.avg_travel_time_AB)
        self.future_demand_at_B = day_data.get("future_demand", self.future_demand_at_B)
        self.occupancy_rate = day_data.get("occupancy_rate", self.occupancy_rate)
        self.uptime_normalized = day_data.get("uptime_normalized", self.uptime_normalized)

        # Apply explicit fallbacks when missing
        self._use_fallbacks()

    def _load_mean_values(self):
        """ Always forces the environment to use only mean values """
        self.avg_travel_time_AB = self.avg_travel_time_AB_mean
        self.future_demand_at_B = self.future_demand_at_B_mean
        self.occupancy_rate = self.occupancy_rate_mean
        self.uptime_normalized = self.uptime_normalized_mean

        print("⚠️ [ENV] Mean travel times loaded.")
        print("⚠️ [ENV] Mean future demand loaded.")
        print("⚠️ [ENV] Mean occupancy rate loaded.")
        print("⚠️ [ENV] Mean uptime normalized loaded.")

    def _use_fallbacks(self):
        """Applies average fallback values where data is missing."""
        if not getattr(self, "avg_travel_time_AB", None):
            self.avg_travel_time_AB = self.avg_travel_time_AB_mean
            print("⚠️ Using fallback avg_travel_time_AB_mean")
        if not getattr(self, "future_demand_at_B", None):
            self.future_demand_at_B = self.future_demand_at_B_mean
            print("⚠️ Using fallback future_demand_at_B_mean")
        if not getattr(self, "occupancy_rate", None):
            self.occupancy_rate = self.occupancy_rate_mean
            print("⚠️ Using fallback occupancy_rate_mean")
        if not getattr(self, "uptime_normalized", None):
            self.uptime_normalized = self.uptime_normalized_mean
            print("⚠️ Using fallback uptime_normalized_mean")


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
    Compound reward and NORMALIZED to [-1, +1] per step.
    Components:
      - occ: penalizes out of ideal range (quadratic, 0..1, sign -)
      - uptime: direct bonus (0..1, sign +)
      - sync: measures regularity of headways vs target (0..1, sign +)
      - efficiency: 1 - (estimated/expected) truncated (0..1, sign +)
    """
    def __init__(self, waitTimeDict=None, reward_weights=None, occupancy_range=(0.6, 0.9),
                 target_headway_seconds: float = 600.0,  # 10 minutos
                 max_sync_rel_std: float = 1.0          # >1 é truncado
                 ):
        super().__init__()
        # self.waitTimeDict can be used if needed for other metrics
        self.waitTimeDict = waitTimeDict or {}

        # Adjustable weights (sum doesn't need to be 1; we do weighted average)
        self.reward_weights = reward_weights or {
            "occ_penalty": 0.5,        # less than 1 to not dominate
            "uptime_bonus": 0.7,
            "sync_score": 0.5,         
            "energy_efficiency": 0.6
        }

        self.occupancy_range = occupancy_range
        self.target_headway = float(target_headway_seconds)
        self.max_sync_rel_std = float(max_sync_rel_std)

    def _occ_component(self, occupancy: float) -> float:
        """
        Returns a value in [0, 1], where 0 = perfect in ideal range; 1 = far off.
        Then we apply negative sign when composing the reward
        """
        min_occ, max_occ = self.occupancy_range
        if occupancy < min_occ:
            return min(1.0, (min_occ - occupancy) ** 2 / (min_occ ** 2 + 1e-8))
        if occupancy > max_occ:
            return min(1.0, (occupancy - max_occ) ** 2 / ((1.0 - max_occ) ** 2 + 1e-8))
        return 0.0

    def _sync_component(self, headways: list) -> float:
        """
        Measures regularity in [0, 1]: 1 = perfect (intervals very close to target),
        0 = very irregular (relative deviation >= max_sync_rel_std)
        """
        if not headways or len(headways) < 3:
            return 0.0  # Not enough information to assess regularity

        # intervals in seconds
        intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)]
        # remove noise/invalid intervals
        intervals = [x for x in intervals if x > 0]
        if len(intervals) < 2:
            return 0.0

        # RMS deviation versus target
        diffs = [(x - self.target_headway) for x in intervals]
        mean_sq = sum(d * d for d in diffs) / len(diffs)
        rms = mean_sq ** 0.5  # in seconds

        # Normalized relative deviation (0=perfect, 1=bad limit)
        rel = min(1.0, rms / (self.max_sync_rel_std * self.target_headway + 1e-8))

        # Convert to "score" in [0,1], where 1 is good
        return 1.0 - rel

    def _efficiency_component(self, estimated_time: float, expected_time: float) -> float:
        """
        Travel efficiency in [0,1]. 1 = equal/to less than expected; 0 = worse than expected.
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
        # Normalized components
        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))

        # Weighted combination (keeping each term in [-1, +1])
        # occ_pen enters with NEGATIVE sign
        w = self.reward_weights
        reward = (
            -w["occ_penalty"] * occ_pen +
             w["uptime_bonus"] * uptime +
             w["sync_score"] * sync +
             w["energy_efficiency"] * eff
        )

        # Normalize by the sum of weights to keep magnitude around ~[-1, +1]
        weight_sum = (abs(w["occ_penalty"]) + w["uptime_bonus"] + w["sync_score"] + w["energy_efficiency"])
        if weight_sum > 0:
            reward = reward / weight_sum

        # Clip final for numerical stability
        # reward += 0.2
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward


# This is the default stop class, which terminates the episode when the agent reaches the target node or takes the SERVICE_CENTER action
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2
