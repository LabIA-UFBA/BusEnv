import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gym import spaces
import numpy as np
import sys
import random
import pickle
import gym.utils.seeding  # import seeding
from gym.spaces import Discrete
import csv
import os
import pandas as pd
from collections import defaultdict
import fcntl

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
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2, use_rain: bool = False, agents_per_route=None,
                use_only_mean_data=None, stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None,
                render_mode=None, avg_travel_time_AB=None, future_demand_at_B=None,
                occupancy_rate=None, uptime_normalized=None,
                real_routes=None, route_metadata=None,risk_horizon_steps=5, enable_risk_feature=True, occupancy_source="real", reward_raining_type ="normal",
                metrics_file_objectives=None, passenger_flow_stats=None,
                record_replay=False, replay_output_dir=None):

        # --- Basic configuration ---
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps 
        self.render_mode = render_mode

        # np.set_printoptions(threshold=sys.maxsize)

        # --- Agents ---
        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()  # <- MARLlib needs this
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # --- Stop/Reward classes ---
        self.stop = DefaultStopClass() if stopClass is None else stopClass
        self.reward = DefaultReward() if rewardClass is None else rewardClass
        self.reward.env = self

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

        # --- Real passenger flow stats (generate_passenger_flow_stats.py) ---
        # Cascading fallback levels: by_route_stop_hour -> by_route_stop -> by_stop_hour -> by_stop -> global
        self.passenger_flow_stats = passenger_flow_stats or {}

        # --- Per-(route, stop) passenger queue state (fixes static-occupancy bug) ---
        # Keyed by (route_id, node_id) -> passengers currently waiting at that stop for that route.
        self.stop_waiting_passengers = {}
        self.last_stop_update_time = {}
        # agent -> route_id (trip_id key into self.real_routes), resolved once per agent in reset()
        # instead of re-scanning self.real_routes.items() on every step().
        self.agent_route_id = {}

        # --- Replay logging (for the game-like map visualization) ---
        self.record_replay = bool(record_replay)
        self.replay_output_dir = replay_output_dir or "replays"
        self.replay_log = []

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
        # (the live weight dict is DefaultReward.reward_weights, set on self.reward
        # above; occupancy_range is duplicated here because _compute_occ_risk_for_agent,
        # an env method, also needs it for the forecast-based risk feature.)
        self.occupancy_range = (0.6, 0.9)

        # --- Risk feature (forecast-based) ---
        # Computes the probability of leaving the ideal occupancy range in the next N route-steps
        # (no time-of-day computation; purely step-ahead along the route)
        self.enable_risk_feature = bool(enable_risk_feature)
        self.risk_horizon_steps = int(risk_horizon_steps)
        # route_id -> {seq_idx(0-based along route): predicted occupancy_norm in [0,1]}
        self.occ_pred_by_route_seq = defaultdict(dict)

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
        self.daily_data_path = "/mnt/ssd1/wesley/BusEnv/src/training_observation/daily_may"  # Path to daily data ONLY USING MAY
        self.daily_files = sorted([
            f for f in os.listdir(self.daily_data_path)
            if f.startswith("daily_data_") and f.endswith(".pkl")
        ])
        self.current_day_index = 0
        self.current_service_day = 0
        self.total_days = len(self.daily_files)
        print(f"[DAILY DATA] {self.total_days} days detected for training.")

        self.occupancy_source = occupancy_source # "real" | "quantum_qru" | "quantum_lstm" | "timesfm_ft" | "timesfm" | "naive"

        self.quantum_data_path = ("/mnt/ssd1/wesley/BusEnv/src/training_observation/quantum_data") # Quamtum data path loading

        self.prediction_data_path = ("/mnt/ssd1/wesley/BusEnv/src/training_observation/prediction") # Prediction data path loading

        self.rates_data_path = ("/mnt/ssd1/wesley/BusEnv/src/training_observation/exports_rates")

        self.quantum_routes = [
            "20001_310_1",
            "20001_310_2",
            "20002_1320_1",
            "20002_1320_10",
            "20002_1367_5",
        ]

        self.prediction_routes = [
            "20001_310_1",
            "20001_310_2",
            "20002_1320_1",
            "20002_1320_10",
            "20002_1367_5",
        ]

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
        self.metrics_file_objectives = metrics_file_objectives or "/mnt/ssd1/wesley/BusEnv/metrics/episode_metrics.csv"
        self._printed_day_end = set()
        self.last_logged_day = -1
        self.episode_step_counter = 0  # Counts total environment steps per episode
        
        
        self.use_only_mean_data = use_only_mean_data   # 1 = use only mean data, 0 = use daily data

        # ==== Rain variables ====

        # --- Climate / Rain control ---
        self.use_rain = use_rain   

        # --- Climate data (Rain) ---
        if self.use_rain:
            # If the component is up, then we use it 
            with open(
                '/mnt/ssd1/wesley/BusEnv/src/training_observation/climate_data/climate_data_2.pkl',
                'rb'
            ) as f:
                self.climate_data = pickle.load(f)
            print("RAIN TRUE, DADOS CARREGADOS EM", self.climate_data)
        else:
            # When theres is no use of the rain
            self.climate_data = None

        self.last_rain_eff = {agent: 0.0 for agent in self.agents}

        self.reward_type = reward_raining_type # normal | penalization | bonus

        self.date = None


        # Create metrics file if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "env_steps", "mean_reward", "total_reward", "fairness"])
        
        os.makedirs(os.path.dirname(self.metrics_file_objectives), exist_ok=True)
        if not os.path.exists(self.metrics_file_objectives):
            with open(self.metrics_file_objectives, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "episode",
                    "occupancy",
                    "uptime",
                    "sync",
                    "efficiency",
                    "Occupancy Percentage"
                ])
        
        self.episode_counter = 0 # Episode Count

        # --- Fallback averages for missing daily data ---
        self.avg_travel_time_AB_mean = avg_travel_time_AB or {} 
        self.future_demand_at_B_mean = future_demand_at_B or {} 
        self.occupancy_rate_mean = occupancy_rate or {}
        self.uptime_normalized_mean = uptime_normalized or {}

        # --- Manual selection of routes you can change for the ones that you want to use--- 
        self.manual_route_groups = {
            "20001_310_1":  ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
            "20001_310_2":  ["agent_5", "agent_6", "agent_7", "agent_8", "agent_9"],
            "20002_1320_1":  ["agent_10", "agent_11", "agent_12", "agent_13", "agent_14"],
            "20002_1320_10":  ["agent_15", "agent_16", "agent_17", "agent_18", "agent_19"],
            "20002_1367_5":  ["agent_20", "agent_21", "agent_22", "agent_23", "agent_24"],
        }

        # Element of the new reward with the occupancy
        self.farf_prediction_steps = 3      # n - How many steps i am going to predict

        self.debug_quantum = False   # DEBUG MODE ON
        self.debug = False   



    @property
    def num_agents(self):
        return self._num_agents
    
    def reset(self, seed=None, options=None):
        print("================ RESETTING ENVIRONMENT ================")
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # --- Defensive date initialization ---
        if not hasattr(self, "date") or self.date is None:
            if hasattr(self, "daily_files") and len(self.daily_files) > 0:
                first_file = self.daily_files[self.current_day_index]
                self.date = (
                    first_file
                    .replace("daily_data_", "")
                    .replace(".pkl", "")
                )
            else:
                self.date = "1970-01-01"  # absolut fallback 

        # --- Inicialização persistente ---
        if not hasattr(self, "current_day_index"):
            self.current_day_index = 0
        if not hasattr(self, "simulated_days"):
            self.simulated_days = 0
        if not hasattr(self, "day_done"):
            self.day_done = False
        if not hasattr(self, "sim_done"):
            self.sim_done = False

        # --- basic structure ---
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

        # --- Reset per-day passenger queues and replay log ---
        self.stop_waiting_passengers = {}
        self.last_stop_update_time = {}
        self.replay_log = []

        # --- Reset agent coordination and presence tracking ---
        self.node_occupancy = {node: [] for node in self.network.nodes}  # which agents are currently at each node
        self.agent_positions = {agent: None for agent in self.possible_agents}
        self.route_last_move_time = {route_id: 0.0 for route_id in self.real_routes.keys()}
        self.route_leader = {}  # will be updated dynamically in step()
        self.episode_step_counter = 0

        self.episode_objectives = {
            agent: []
            for agent in self.agents
        }

        self.episode_occupancy_values = {a: [] for a in self.possible_agents}
        self.episode_occupancy_ideal_flags = {a: [] for a in self.possible_agents}

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
            self.date = next_date
            print(f"\n🔁 [ENV] Advancing to next day: {next_date} ({self.current_day_index + 1}/{self.total_days})")
            print(f"📆 [ENV] Total simulated days so far: {self.simulated_days}")
            print("self.date used only when the raining is activate", self.date)
            self.day_done = False  # reset flag
            self.agents_finished_previous_day = False # reset flag

        # --- Loads daily data ---
        try:
            self.load_current_day_data()
        except Exception as e:
            print(f"⚠️ Error loading daily data: {e}. Using fallback averages.")
            self._use_fallbacks()
        
        # --- Initialization of fixed routes (if they dont already exist) ---
        if not hasattr(self, "fixed_agent_routes") or self.fixed_agent_routes is None:
            self.fixed_agent_routes = {}
            num_agents = len(self.agents)
            routes = list(self.real_routes.items())

            manual_route_groups = getattr(self, "manual_route_groups", {}) or {}

            manually_assigned_agents = set()
            used_trip_ids = set()

            for trip_id, agent_list in manual_route_groups.items():
                if trip_id not in self.real_routes:
                    print(f"[ROUTE MANUAL] AVISO: trip_id '{trip_id}' não existe em real routes, ignorando.")
                    continue
                
                path = self.real_routes[trip_id]

                for agent in agent_list:
                    if agent not in self.agents:
                        print(f"[ROUTE MANUAL] AVISO: agente '{agent}' não existe em self.agents, ignorando.")
                        continue
                    self.fixed_agent_routes[agent] = path
                    manually_assigned_agents.add(agent)
                
                used_trip_ids.add(trip_id)
                route_preview = " -> ".join(str(n) for n in path)
                print(
                    f"[ROUTE MANUAL] trip={trip_id} | len={len(path)} | "
                    f"agents={', '.join(agent_list)} | path={route_preview}"
                )
            
            remaining_agents = [a for a in self.agents if a not in manually_assigned_agents]

            remaining_routes = [(tid, p) for tid, p in routes if tid not in used_trip_ids]

            # CHANGE: function to decide number of agents per route (based on length)
            def _agents_for_path(path_len: int, default_agents: int) -> int:
                if path_len < 15:
                    return int(default_agents) # Antes era 1 
                if path_len <= 30:
                    return int(default_agents) # Antes era 2 
                # >30 uses the user's default (ensures >=1)
                return max(1, int(default_agents)) # DO JEITO QUE TA AQUI EU TO SEMPRE RETORNANDO POR ROTA O VALOR QUE EU SETO POR ENQUANTO

            default_agents_per_route = getattr(self, "agents_per_route", 1)

            agent_routes_assignment = []
            agent_idx = 0
            num_remaining_agents = len(remaining_agents)

            # dynamic allocation based on route length
            for trip_id, path in remaining_routes:
                if agent_idx >= num_remaining_agents:
                    break

                k = _agents_for_path(len(path), default_agents_per_route)
                # does not exceed the total remaining agents
                k = min(k, num_remaining_agents - agent_idx)
                if k <= 0:
                    break

                assigned_agents = remaining_agents[agent_idx:agent_idx + k]
                agent_routes_assignment.append((trip_id, path, assigned_agents))

                for agent in assigned_agents:
                    self.fixed_agent_routes[agent] = path

                # DEBUG: show decision per route
                route_preview = " → ".join(str(n) for n in path)
                print(
                    f"[ROUTE SPLIT] trip={trip_id} | len={len(path)} | assigned={k} | "
                    f"agents={', '.join(assigned_agents)} | path={route_preview}"
                )

                agent_idx += k

            print("\n=== [ROUTE ASSIGNMENT DEBUG - INITIALIZED ONCE] ===")
            for trip_id, path, assigned_agents in agent_routes_assignment:
                route_preview = " → ".join(str(n) for n in path[:5])
                if len(path) > 5:
                    route_preview += " → ..."
                print(f"Trip ID: {trip_id:<10} | Agents: {', '.join(assigned_agents)} | "
                      f"Route length: {len(path):<3} | Path: {route_preview}")
            print("=====================================================\n")

        # --- Usa sempre as rotas fixas ---
        self.agent_routes = self.fixed_agent_routes

        # --- Resolve route_id (trip_id) for every agent up front, in one pass, so that
        # pairwise-headway lookups below never depend on loop ordering between agents
        # that share the same route. ---
        path_to_route_id = {tuple(v): k for k, v in self.real_routes.items()}
        self.agent_route_id = {
            agent: path_to_route_id.get(tuple(path), "unknown")
            for agent, path in self.agent_routes.items()
        }

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
                "occupancy": float(self.occupancy_rate.get(initial, 0.0)),
                "uptime": float(self.uptime_normalized.get(initial, 1.0)),
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": path,
                "route_idx": 0,
                "status": "active",
                "going_forward": True,
            }
            
            print("self.agent_states[agent]: ", self.agent_states[agent])

            # --- Register initial position for coordination ---
            self.agent_positions[agent] = initial
            if initial not in self.node_occupancy:
                self.node_occupancy[initial] = []
            self.node_occupancy[initial].append(agent)

            # If this is the first agent on this route, mark as leader
            rid = self.agent_route_id.get(agent, "unknown")
            if rid != "unknown" and rid not in self.route_leader:
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

            occ_risk = self._compute_occ_risk_for_agent(agent, horizon=self.risk_horizon_steps)

            # --- Climate / Rain: initial observation ---
            if self.use_rain and self.climate_data is not None:
                time_sec = self.agent_times[agent]
                hour_of_day = int(time_sec // 3600)
                hour_str = f"{hour_of_day:02d}00 UTC"

                date_str = str(self.date).replace('-', '/')
                row = self.climate_data[
                    (self.climate_data['date'] == date_str) &
                    (self.climate_data['hour'] == hour_str)
                ]

                is_raining = False
                if not row.empty:
                    is_raining = row['precip'].iloc[0] > 0.0
            else:
                # Rain off
                is_raining = False

            leader_gap, follower_gap = self._compute_pairwise_headway_gaps(agent)

            obs_array = np.array([
                self.agent_times[agent] / (24 * 60 * 60),
                self.agent_states[agent]["occupancy"],
                normalized_travel_time,
                self.future_demand_at_B.get(str(next_node), 0.0),
                self.uptime_normalized.get(agent, 1.0),
                1.0 if self.agent_states[agent]["maintenance_status"] == "ok" else 0.0,
                self.node_to_idx[str(initial)],
                self.node_to_idx[str(next_node)],
                1.0 if is_raining else 0.0,
                occ_risk,
                self._normalize_headway_gap(leader_gap),
                self._normalize_headway_gap(follower_gap),
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
        
        # Validação de observação de 1 agente especifico, primeiro da fila 
        sample_agent = self.agents[0] 
        print("\n[OBS DEBUG - RESET]")
        print(f"Agent: {sample_agent}")
        print(f"Raw obs: {obs_array}")
        print(f"Clipped obs: {clipped_obs}")
        print(f"Obs space low: {self.observation_space(sample_agent).low}")
        print(f"Obs space high: {self.observation_space(sample_agent).high}")

        print("\n[TIME DEBUG]")
        print(f"Simulated days: {self.simulated_days}")
        print(f"Current day index: {self.current_day_index}")
        print(f"Agent start time (seconds): {list(self.agent_times.items())[:3]}")



        # --- Debug: presence summary ---
        active_points = {node: agents for node, agents in self.node_occupancy.items() if agents}
        print(f"📍 [RESET DEBUG] Agent initial positions per node: {active_points}")

        print("✅ [RESET COMPLETE] All agents initialized with status='active'.")
        print("\n=== [RESET SUMMARY] ===")
        print(f"Agents: {len(self.agents)}")
        print(f"Routes: {len(set(tuple(r) for r in self.agent_routes.values()))}")
        print(f"Leaders per route: {self.route_leader}")
        print("Agent → Route length:")
        for agent, route in self.agent_routes.items():
            print(f"  {agent}: len={len(route)} start={route[0]} end={route[-1]}")
        print("=======================\n")
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
        self.episode_step_counter += 1


        # Safety: ensure agent_states exists
        if not hasattr(self, "agent_states"):
            raise RuntimeError("agent_states not initialized. Call reset() before step().")

        # Global limits
        TRAVEL_TIME_CAP = 1800.0   # 30 minutes max per edge
        END_OF_DAY = 24 * 3600.0      # End of the day in seconds

        # --- Reset dynamic presence tracking for this step ---
        for node in self.node_occupancy:
            self.node_occupancy[node] = []  # clear per-step occupancy

        if self.episode_step_counter % 50 == 0:
            print(f"\n--- [STEP {self.episode_step_counter}] ---")


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

            # --- Climate / Rain: check current weather ---
            if self.use_rain and self.climate_data is not None:
                time_sec = self.agent_times[agent]
                hour_of_day = int(time_sec // 3600)
                hour_str = f"{hour_of_day:02d}00 UTC"

                date_str = self.date.replace('-', '/')
                row = self.climate_data[
                    (self.climate_data['date'] == date_str) &
                    (self.climate_data['hour'] == hour_str)
                ]

                is_raining = False
                if not row.empty:
                    is_raining = row['precip'].iloc[0] > 0.0
            else:
                # Rain off
                is_raining = False


            # === ACTION 0: WAIT ===
            if action == 0:
                if action == 0 and self.steps[agent] % 20 == 0:
                    print(f"[WAIT][{agent}] time={self.agent_times[agent]:.0f}s")
                reward = 0 # Antes era  -0.1
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

                self.agent_times[agent] += travel_time # Avançando o relogio do agente baseado no tempo da viagem
                self.estimated_times[agent] += travel_time

                # --- Resolve this agent's route id once (reused for the queue key below and
                # for the headway key further down, instead of re-scanning self.real_routes). ---
                route_id = self.agent_route_id.get(agent, "unknown")

                # Update occupancy
                prev_occ = state.get("occupancy", 0.0)

                if self.occupancy_source == "real":
                    # =====================================================
                    # Real occupancy based on SUNT Historic: per-(route, stop) passenger QUEUE.
                    #
                    # Boarding/alighting happens when the bus ARRIVES at next_node (not
                    # when it departs curr_node, which the previous implementation used).
                    # The queue accumulates passengers between visits at a real arrival
                    # rate (pax/sec, derived from historical boardings-per-visit divided
                    # by the historical inter-visit time) and DEPLETES when a bus boards —
                    # this is what makes a second bus passing 1 minute after the first see
                    # a near-empty queue instead of the same static value.
                    # =====================================================
                    now_time = self.agent_times[agent]
                    arrival_rate = self._get_arrival_rate(route_id, next_node, now_time, going_forward)
                    alight_frac = self._get_alight_frac(route_id, next_node, now_time, going_forward)
                elif self.occupancy_source in ("timesfm", "naive"):
                    # =====================================================
                    # Prediction-based occupancy (Quantum / TimesFM / etc.)
                    # =====================================================
                    now_time = self.agent_times[agent]
                    arrival_rate = self._get_predicted_arrival_rate(route_id, next_node, now_time, going_forward)
                    alight_frac = self._get_predicted_alight_frac(route_id, next_node, now_time, going_forward)
                else:
                    arrival_rate = alight_frac = None  # quantum: ramo antigo abaixo

                if np.random.rand() < 0.02 and self.debug:
                    print(
                        f"[RATE SOURCE] source={self.occupancy_source} | "
                        f"route={route_id} | node={next_node} | "
                        f"arrival_rate={arrival_rate:.5f} | alight_frac={alight_frac:.3f}"
                    )
                
                if arrival_rate is not None:
                    qkey = (route_id, next_node)

                    if qkey not in self.stop_waiting_passengers:
                        queue = self._initial_queue_estimate(route_id, next_node, now_time, going_forward)
                    else:
                        last_t = self.last_stop_update_time.get(qkey, now_time)
                        elapsed = max(0.0, now_time - last_t)
                        queue = self.stop_waiting_passengers[qkey] + arrival_rate * elapsed

                    prev_onboard = prev_occ * self.max_capacity
                    alighting = prev_onboard * alight_frac
                    onboard_after_alighting = max(0.0, prev_onboard - alighting)
                    remaining_capacity = max(0.0, self.max_capacity - onboard_after_alighting)
                    boarding = min(queue, remaining_capacity)

                    self.stop_waiting_passengers[qkey] = max(0.0, queue - boarding)
                    self.last_stop_update_time[qkey] = now_time

                    occupancy = max(0.0, min((onboard_after_alighting + boarding) / self.max_capacity, 1.0))

                    if np.random.rand() < 0.05 and self.debug:
                        print(
                            f"[QUEUE] route={route_id} node={next_node} "
                            f"prev_occ={prev_occ:.3f} queue_before_board={queue:.2f} "
                            f"boarded={boarding:.2f} alighted={alighting:.2f} "
                            f"queue_after={self.stop_waiting_passengers[qkey]:.2f} "
                            f"new_occ={occupancy:.3f}"
                        )
                else:
                    if curr_node in self.occupancy_rate:
                        occupancy = max(0.0, min(self.occupancy_rate[curr_node], 1.0))
                        if self.debug_quantum and self.episode_step_counter % 50 == 0:
                            print(
                                f"🧪 [Quantum-OCC USED] "
                                f"day={self.current_day_index} | "
                                f"step={self.episode_step_counter} | "
                                f"agent={agent} | "
                                f"node={curr_node} | "
                                f"Quantum_occ={occupancy:.4f} | "
                                f"source={self.occupancy_source}"
                            )
                    else:
                        occupancy = prev_occ
                  
                """
                CODIGO ANTIGO ONDE SO OS DADOS DO SUNT TEM A LOGICA DE DESOCUPAÇÃO

                if self.occupancy_source != "real":
                    # =====================================================
                    # Prediction-based occupancy (Quantum / TimesFM / etc.)
                    # Unchanged: still driven by the static occupancy_rate override
                    # built by _override_occupancy_with_predictions_node_level()
                    # =====================================================
                    if curr_node in self.occupancy_rate:
                        occupancy = self.occupancy_rate[curr_node]
                        occupancy = max(0.0, min(occupancy, 1.0))

                        if self.debug_quantum and self.episode_step_counter % 50 == 0:
                            print(
                                f"🧪 [PRED-OCC USED] "
                                f"day={self.current_day_index} | "
                                f"step={self.episode_step_counter} | "
                                f"agent={agent} | "
                                f"node={curr_node} | "
                                f"pred_occ={occupancy:.4f} | "
                                f"source={self.occupancy_source}"
                            )
                    else:
                        occupancy = prev_occ

                else:
                    # =====================================================
                    # Real occupancy: per-(route, stop) passenger QUEUE.
                    #
                    # Boarding/alighting happens when the bus ARRIVES at next_node (not
                    # when it departs curr_node, which the previous implementation used).
                    # The queue accumulates passengers between visits at a real arrival
                    # rate (pax/sec, derived from historical boardings-per-visit divided
                    # by the historical inter-visit time) and DEPLETES when a bus boards —
                    # this is what makes a second bus passing 1 minute after the first see
                    # a near-empty queue instead of the same static value.
                    # =====================================================
                    qkey = (route_id, next_node)
                    now_time = self.agent_times[agent]

                    arrival_rate = self._get_arrival_rate(route_id, next_node, now_time, going_forward)
                    alight_frac = self._get_alight_frac(route_id, next_node, now_time, going_forward)

                    if qkey not in self.stop_waiting_passengers:
                        # First visit of the day to this (route, stop): seed with the
                        # expected per-visit boarding count rather than 0 — the historical
                        # average is itself already conditioned on periodic servicing.
                        queue = self._initial_queue_estimate(route_id, next_node, now_time, going_forward)
                    else:
                        last_t = self.last_stop_update_time.get(qkey, now_time)
                        elapsed = max(0.0, now_time - last_t)
                        queue = self.stop_waiting_passengers[qkey] + arrival_rate * elapsed

                    prev_onboard = prev_occ * self.max_capacity
                    alighting = prev_onboard * alight_frac
                    onboard_after_alighting = max(0.0, prev_onboard - alighting)

                    remaining_capacity = max(0.0, self.max_capacity - onboard_after_alighting)
                    boarding = min(queue, remaining_capacity)

                    self.stop_waiting_passengers[qkey] = max(0.0, queue - boarding)
                    self.last_stop_update_time[qkey] = now_time

                    occupancy = max(0.0, min((onboard_after_alighting + boarding) / self.max_capacity, 1.0))

                    if np.random.rand() < 0.05 and self.debug:
                        print(
                            f"[QUEUE] route={route_id} node={next_node} "
                            f"prev_occ={prev_occ:.3f} queue_before_board={queue:.2f} "
                            f"boarded={boarding:.2f} alighted={alighting:.2f} "
                            f"queue_after={self.stop_waiting_passengers[qkey]:.2f} "
                            f"new_occ={occupancy:.3f}"
                        )
                """

                state["occupancy"] = occupancy

                # Future occupancies always come from the configured occupancy source.
                # Depending on occupancy_source this dictionary may contain:
                #   - real historical occupancy (SUNT)
                #   - Quantum predictions
                #   - LSTM predictions
                #   - TimesFM predictions
                state["predicted_occupancies"] = self._get_future_occupancies(
                    agent=agent,
                    route=route,
                    current_occupancy=occupancy
                )
                # state["uptime"] = min(1.0, state["uptime"] / (travel_time_hors + 1e-8))
                travel_time_hors = travel_time / (12 * 3600) 
                old = state["uptime"] 
                state["uptime"] = max(state["uptime"] - travel_time / (12 * 3600), 0.0) # O 12 aqui estou assumindo que 12 horas sem service center é o maximo que o agente consegue
                state["fuel"] = max(state["fuel"] - travel_time / 300.0, 0.0)

                if np.random.rand() < 0.01 and self.debug:
                    print(
                        f"[UPTIME MOVE] "
                        f"old={old:.4f} "
                        f"travel_h={travel_time_hors:.4f} "
                        f"new={state['uptime']:.4f}"
                    )

                self.states[agent] = next_node

                """
                print(
                    f"[MOVE][{agent}] "
                    f"{curr_node} → {next_node} | "
                    f"route_idx={self.agent_states[agent]['route_idx']} | "
                    f"time+={travel_time:.1f}s | "
                    f"fuel={state['fuel']:.1f} | "
                    f"uptime={state['uptime']:.3f}"
                )

              
                # --- DEBUG AQUI ---
                total_sec = int(self.agent_times[agent])
                hours = total_sec // 3600
                minutes = (total_sec % 3600) // 60

                print(
                    f"[DEBUG][{agent}] time={hours:02d}:{minutes:02d} | "
                    f"estimated_time={self.estimated_times[agent]:.1f} | "
                    f"expected_time={self.expected_times[agent]:.1f} | "
                    f"node={next_node}"
                )
                """

                #if next_node not in self.headways:
                #    self.headways[next_node] = []
                #self.headways[next_node].append(self.agent_times[agent])

                # --- direção ---
                direction = state.get("going_forward", True)  

                # --- Montando a chave correta ---
                key = (route_id, next_node, direction) 

                # --- armazenamento ---
                if key not in self.headways:
                    self.headways[key] = []

                self.headways[key].append(self.agent_times[agent])

                # --- janela temporal ---
                MAX_HEADWAY_TIME = 1800 # Limpamos a lista após 30 minutos (1800)
                self.headways[key] = [
                    t for t in self.headways[key]
                    if self.agent_times[agent] - t <= MAX_HEADWAY_TIME # O que entra na lista de headways é o horário absoluto de chegada naquele ponto pro agente 
                ]

                if self.episode_step_counter % 50 == 0 and self.debug:  # DEBUG HADWAY
                    print(
                        f"[HEADWAY DEBUG] key={key} | "
                        f"agent={agent} | "
                        f"time={self.agent_times[agent]:.1f} | "
                        f"n={len(self.headways[key])} | "
                        f"times={sorted(self.headways[key])}"
                    )
                """
                reward = self.reward.getRewardHard( 
                    new_state=next_node,
                    previous_state=curr_node,
                    action=action,
                    target=route[-1],
                    network=self.network,
                    estimated_time=self.estimated_times[agent],
                    expected_time=self.expected_times[agent],
                    delay=0,
                    agent_state=state,
                    headways=self.headways[key], # headways=self.headways[next_node]
                )

                """
                # CHAMADA PADRÃO COM O RAIN (CHUVA)
                reward = self.reward.getReward( 
                    agent=agent,
                    new_state=next_node,
                    previous_state=curr_node,
                    action=action,
                    target=route[-1],
                    network=self.network,
                    estimated_time=self.estimated_times[agent],
                    expected_time=self.expected_times[agent],
                    delay=0,
                    agent_state=state,
                    headways=self.headways[key], # headways=self.headways[next_node]
                    is_raining=is_raining,
                    reward_type=self.reward_type,
                    last_rain_eff=self.last_rain_eff.get(agent, 0.0)
                )

                vector = self.reward.getVectorReward(
                    agent=agent,
                    new_state=next_node,
                    previous_state=curr_node,
                    action=action,
                    target=route[-1],
                    network=self.network,
                    estimated_time=self.estimated_times[agent],
                    expected_time=self.expected_times[agent],
                    delay=0,
                    agent_state=state,
                    headways=self.headways[key],
                    is_raining=is_raining,
                    reward_type=self.reward_type,
                    last_rain_eff=self.last_rain_eff.get(agent, 0.0)
                )

                print(
                    f"[OBJECTIVE APPEND] "
                    f"{agent} "
                    f"time={self.agent_times[agent]:.0f} "
                    f"step={self.steps[agent]} "
                    f"len_before={len(self.episode_objectives[agent])}"
                )

                self.episode_objectives[agent].append(vector)
                # print("self.episode_objectives[agent]: ", self.episode_objectives[agent]) print(np.round(vector, 3))
                
                
                self.last_rain_eff[agent] = self.reward._efficiency_component(
                    float(self.estimated_times[agent]),
                    float(self.expected_times[agent])
                )

                if self.episode_step_counter % 100 == 0:
                    print(
                        f"[STEP→REWARD] "
                        f"agent={agent} | "
                        f"action={action} | "
                        f"node={curr_node}->{next_node} | "
                        f"time={self.agent_times[agent]:.0f}s | "
                        f"reward={reward:.3f}"
                    )



                """
                print(
                    f"[REWARD][{agent}] "
                    f"node={next_node} | "
                    f"est={self.estimated_times[agent]:.1f} | "
                    f"exp={self.expected_times[agent]:.1f} | "
                    f"headway_n={len(self.headways[next_node])} | "
                    f"reward={reward:.3f}"
                )
                """

                self.estimated_times[agent] = 0.0

            # === ACTION 2: SERVICE CENTER ===
            elif action == 2:
                sc_node = self.get_nearest_service_center(curr_node)
                if action == 2 and self.steps[agent] % 20 == 0:
                    print(f"[SERVICE CENTER][{agent}] time={self.agent_times[agent]:.0f}s")
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
                    reward = 0 # antes era -10.0
                else:
                    if state["fuel"] < total_fuel_cost:
                        reward = 0 # em vez de -20
                        print("total_fuel_cost: ", total_fuel_cost)
                        print("state[fuel]: ", state["fuel"])
                        print("VEIO PRO SERVICE CENTER SEM CONDIÇÃO DE CHEGAR")
                    else:
                        # self.agent_times[agent] += total_travel_time # Modelo anterior
                        # self.estimated_times[agent] += total_travel_time
                        penalized_time = total_travel_time * 0.4  # Redução de tempo para um penalização mais leve no tempo
                        self.agent_times[agent] += penalized_time  # Redução de tempo para um penalização mais leve no tempo
                        self.estimated_times[agent] += penalized_time 
                        state["fuel"] = max(state["fuel"] - total_fuel_cost, 0.0)
                        # total_travel_time_hors = total_travel_time / (12 * 3600)
                        state["uptime"] = max(state["uptime"] - total_travel_time / (12 * 3600), 0.0)
                        state["fuel"] = 100.0
                        state["uptime"] = min(state["uptime"] + 0.3, 1.0) # state["uptime"] = 1.0
                        state["maintenance_status"] = "ok"
                        self.states[agent] = sc_node
                        reward = 0 # PADRÃO
                        # reward = -0.2 * (total_travel_time / 600.0) # IMPACTO NA RECOMPENSA 

            # === INVALID ACTION ===
            else:
                reward = 0


            assert 0.0 <= state["fuel"] <= 100.0, f"Fuel inválido: {state['fuel']}"
            assert 0.0 <= state["uptime"] <= 1.0, f"Uptime inválido: {state['uptime']}"
            assert self.agent_times[agent] >= 0.0

            """
            # padroniza a chave
            node = self.states[agent]

            # garante que exista lista
            if node not in self.headways:
                self.headways[node] = []

            headways_list = self.headways[node]
            
            if should_calc_reward:
                reward = self.reward.getReward(
                    new_state=self.states[agent],
                    previous_state=curr_node,
                    action=action,
                    target=route[-1],
                    network=self.network,
                    estimated_time=self.estimated_times[agent],
                    expected_time=self.expected_times[agent],
                    delay=0,
                    agent_state=state,
                    headways=headways_list
                )
            """

            # Presence and coordination tracking
            current_pos = self.states[agent]
            self.agent_positions[agent] = current_pos

            if current_pos not in self.node_occupancy:
                self.node_occupancy[current_pos] = []
            self.node_occupancy[current_pos].append(agent)

            """
            if len(self.node_occupancy[current_pos]) > 1:
                print(
                    f"[OCCUPANCY] node={current_pos} "
                    f"agents={self.node_occupancy[current_pos]}"
                )
            """

            #if len(self.node_occupancy[current_pos]) > 1:  # detect overlap
               # print(f"[DEBUG] Overlap at node {current_pos}: {self.node_occupancy[current_pos]}")

            rid = self.agent_route_id.get(agent, "unknown")
            if rid != "unknown":
                self.route_last_move_time[rid] = self.agent_times[agent]

            # === End-of-day to agents, putting the finished status ===
            if self.agent_times[agent] >= END_OF_DAY:
                print(
                    f"[OBS ZEROED][END_OF_DAY] agent={agent} "
                    f"time={self.agent_times[agent]:.0f}s "
                    f"step={self.episode_step_counter}"
                )
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

            occ_risk = self._compute_occ_risk_for_agent(agent, horizon=self.risk_horizon_steps)
            leader_gap, follower_gap = self._compute_pairwise_headway_gaps(agent)

            # === Observation update ===
            if state.get("status") != "active":
                print(
                    f"[WARN] OBS about to be computed for non-active agent "
                    f"{agent} status={state.get('status')}"
                )

            obs_array = np.array([
                self.agent_times[agent] / (24 * 60 * 60),
                state["occupancy"],
                normalized_travel_time,
                self.future_demand_at_B.get(str(next_node), 0.0),
                state["uptime"],
                1.0 if state["maintenance_status"] == "ok" else 0.0,
                self.node_to_idx[str(curr_node)],
                self.node_to_idx[str(next_node)],
                1.0 if is_raining else 0.0,
                occ_risk,
                self._normalize_headway_gap(leader_gap),
                self._normalize_headway_gap(follower_gap),
            ], dtype=np.float32)

            observations[agent] = np.clip(
                obs_array,
                self.observation_space(agent).low,
                self.observation_space(agent).high
            )

            if not observations[agent].any():
                print(
                    f"[WARN][ZERO OBS AFTER CLIP] agent={agent} "
                    f"time={self.agent_times[agent]:.0f}s "
                    f"status={state.get('status')}"
                )

            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {"status": "active"}

            if self.record_replay:
                self.replay_log.append({
                    "sim_time_sec": self.agent_times[agent],
                    "agent_id": agent,
                    "route_id": rid,
                    "curr_node": curr_node,
                    "next_node": next_node,
                    "action": int(action),
                    "occupancy": float(state["occupancy"]),
                    "waiting": float(self.stop_waiting_passengers.get((rid, curr_node), 0.0)),
                })

            # Update route leader dynamically
            if rid != "unknown":
                leader = self.route_leader.get(rid)
                if leader is None or self.agent_times[agent] > self.agent_times.get(leader, 0.0):
                    self.route_leader[rid] = agent
                   # print(
                   #     f"[LEADER CHANGE] route={rid} "
                   #     f"new={agent} "
                   #     f"time={self.agent_times[agent]:.0f}"
                   # )

        # === Global post-processing ===
        all_finished = all(self.agent_times[a] >= END_OF_DAY for a in self.possible_agents)

        dones = {a: all_finished for a in self.possible_agents}
        dones["__all__"] = all_finished

        if all_finished:
            print("🌙 [ENV] All agents finished 24h — day finished. Awaiting reset() to advance to next day.")
            episode_vectors = []
            pct_ideal_occupancy_per_agent = []
            for a in self.possible_agents:
                print(f"{a}: time={self.agent_times[a]:.0f}s steps={self.steps[a]}")

                print("=" * 60)
                print(a)
                print("episode_objectives len =", len(self.episode_objectives[a]))

                for i, obj in enumerate(self.episode_objectives[a][:5]):
                    print(i, obj, type(obj))
                
                vectors = np.array(self.episode_objectives[a]) # I take the episode values ​​for the metrics

                if len(vectors) == 0:
                    total_objectives = np.zeros(4, dtype=np.float32)
                    mean_objectives = np.zeros(4, dtype=np.float32)
                else:
                    total_objectives = vectors.sum(axis=0) # Sums all vectors collected in the episode for this agent 
                    mean_objectives = vectors.mean(axis=0) # Mean for all vectors collected in the episode for this agent

                print("vectors.shape =", vectors.shape)

                #total_objectives = vectors.sum(axis=0) # Sums all vectors collected in the episode for this agent 
                #mean_objectives = vectors.mean(axis=0) # Mean for all vectors collected in the episode for this agent

                # NOVO: % of the steps that this agent spent within the ideal occupancy
                occ_flags = np.array(self.episode_occupancy_ideal_flags[a])
                pct_ideal_occupancy = float(occ_flags.mean() * 100) if len(occ_flags) > 0 else 0.0
                pct_ideal_occupancy_per_agent.append(pct_ideal_occupancy) 

                infos[a]["episodic_return_vector"] = total_objectives
                infos[a]["episodic_mean_objectives"] = mean_objectives
                infos[a]["pct_ideal_occupancy"] = pct_ideal_occupancy 
                infos[a]["status"] = "finished"
                infos[a]["reason"] = "24h_limit"

                print("mean_objectives =", mean_objectives)
                print(type(mean_objectives))
                print(np.shape(mean_objectives))
                
                episode_vectors.append(mean_objectives)

                if self.debug:
           
                    print(f"\n{a}")

                    print(f"Mean Objectives: {mean_objectives.round(3)}")

                    print(f"Total Sun Objectives: {total_objectives.round(3)}")

                    print(f"% Ocupação Ideal: {pct_ideal_occupancy:.1f}%")

                    print(f"\n Valores brutos do Agente, vector: {vectors.round(3)}")
            
            episode_vectors = np.array(episode_vectors)

            episode_mean = episode_vectors.mean(axis=0)

            # Overall ideal daily occupancy percentage, averaged across all agents

            pct_ideal_occupancy_day = (
                float(np.mean(pct_ideal_occupancy_per_agent))
                if pct_ideal_occupancy_per_agent else 0.0
            )

            self.episode_counter += 1 # Counting the episodes of the training

            with open(self.metrics_file_objectives, "a", newline="") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    print(
                        "[CSV]",
                        "PID =", os.getpid(),
                        "PATH =", os.path.abspath(self.metrics_file_objectives),
                        "EP =", self.episode_counter,
                    )
                    
                    
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_counter,
                        float(episode_mean[0]),
                        float(episode_mean[1]),
                        float(episode_mean[2]),
                        float(episode_mean[3]),
                        pct_ideal_occupancy_day,
                    ])
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            if self.debug:
                print("\n==============================")
                print(f"Episode Mean Objectives: {episode_mean.round(3)}") # occupancy_score, uptime_score, sync_score and efficiency_score
                print(f"Episode % Ideal Occupancy: {pct_ideal_occupancy_day:.1f}%")
                print("==============================")

            if self.record_replay:
                self._flush_replay_log()

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

        if self.episode_step_counter % 100 == 0 and self.debug:
            for a, obs in observations.items():
                print(f"[OBS SNAPSHOT][{a}] {obs}")

        if self.episode_step_counter % 50 == 0 and self.debug:
            print("\n[STEP SUMMARY]")
            for a in self.possible_agents:
                print(
                    f"  {a}: "
                    f"status={self.agent_states[a].get('status')} | "
                    f"time={self.agent_times[a]:.0f}s | "
                    f"route_idx={self.agent_states[a]['route_idx']}"
                )

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
                0.0,    # next_node_id
                0.0,    # is_raining (0 ou 1)
                0.0,    # occ_risk_out
                -1.0,   # headway_leader_norm (signed: <0 bunched up, >0 gap too large)
                -1.0,   # headway_follower_norm
            ], dtype=np.float32),
            high=np.array([
                1.0,    # time_of_day_norm
                1.0,    # occupancy_rate
                1.0,    # avg_travel_time_AB (normalizado!)
                1e6,    # future_demand_at_B (mantém valor realista)
                1.0,    # uptime
                1.0,    # manutenção ok
                2e9,    # curr_node_id
                2e9,    # next_node_id
                1.0,    # is_raining (0 ou 1)
                1.0,    # occ_risk_out
                1.0,    # headway_leader_norm
                1.0,    # headway_follower_norm
            ], dtype=np.float32)
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): # Define the action space for each agent
        return spaces.Discrete(self.actions_amount)
    
    # ------------------------------------------------------------------
    # Risk feature helpers (route-step-ahead, no time computations)
    # ------------------------------------------------------------------
    def _advance_route_idx_bounce(self, idx: int, going_forward: bool, route_length: int):
        """Advance one position with the same 'bounce at ends' rule used in MOVE."""
        if route_length <= 1:
            return idx, going_forward

        if going_forward:
            if idx + 1 < route_length:
                return idx + 1, True
            # bounce back
            return max(0, idx - 1), False
        else:
            if idx - 1 >= 0:
                return idx - 1, False
            # bounce forward
            return min(route_length - 1, idx + 1), True

    def _future_route_indices(self, idx: int, going_forward: bool, route_length: int, horizon: int):
        """List of future indices (excluding current) for the next 'horizon' route-steps."""
        out = []
        i = int(idx)
        d = bool(going_forward)
        for _ in range(max(0, int(horizon))):
            i, d = self._advance_route_idx_bounce(i, d, route_length)
            out.append(i)
        return out

    def _get_predicted_occupancy_for_route_seq(self, route_id: str, seq_idx: int):
        """Fetch occupancy prediction aligned to a route's pt_sequence (0-based)."""
        if not route_id:
            return None
        mp = getattr(self, "occ_pred_by_route_seq", None)
        if not mp:
            return None
        v = mp.get(route_id, {}).get(int(seq_idx))
        return None if v is None else float(v)

    def _compute_occ_risk_for_agent(self, agent: str, horizon: int = None) -> float:
        """Probability of leaving the ideal occupancy range in the next N route-steps."""
        if not getattr(self, "enable_risk_feature", False):
            return 0.0

        H = int(self.risk_horizon_steps if horizon is None else horizon)
        if H <= 0:
            return 0.0

        state = self.agent_states.get(agent)
        if not state:
            return 0.0

        route = state.get("route") or []
        route_length = len(route)
        if route_length < 2:
            return 0.0

        idx = int(state.get("route_idx", 0))
        going_forward = bool(state.get("going_forward", True))
        route_id = state.get("route_id")

        min_occ, max_occ = self.occupancy_range

        future_idxs = self._future_route_indices(idx, going_forward, route_length, H)
        out_count = 0

        for j in future_idxs:
            # seq_idx is aligned with route order (same convention as: seq = pt_sequence - 1)
            occ_pred = self._get_predicted_occupancy_for_route_seq(route_id, j)

            if occ_pred is None:
                # fallback: use node-level occupancy estimate (possibly overridden by predictions)
                try:
                    node_int = int(route[j])
                except Exception:
                    node_int = None
                if node_int is not None:
                    occ_pred = float(self.occupancy_rate.get(node_int, 0.0))
                else:
                    occ_pred = 0.0

            if occ_pred < min_occ or occ_pred > max_occ:
                out_count += 1

        return float(out_count) / float(H)

    # ------------------------------------------------------------------
    # Real passenger flow lookup (generate_passenger_flow_stats.py) +
    # per-(route, stop) queue helpers
    # ------------------------------------------------------------------
    def _get_flow_stats(self, route_id: str, node, now_time: float, going_forward: bool = True) -> dict:
        """
        Cascading fallback into self.passenger_flow_stats, from most to least specific:
          1) by_route_stop_hour  (route+stop+hour of day)
          2) by_route_stop       (route+stop, all hours pooled)
          3) by_stop_hour        (stop+hour, pooled across every route serving it)
          4) by_stop              (stop, all routes/hours pooled)
          5) global               (last-resort scalar)

        Levels 1-2 (route-specific) are only used when going_forward=True: the
        historical OD data only ever recorded the direction a trip_id actually
        drove. The simulated "bounce back" leg (going_forward=False) retraces
        that same node list in reverse, which has no real-world counterpart —
        using the forward route's stats for it would misrepresent a path no
        real bus ever drove, so the backward leg falls back straight to the
        stop-level pools instead.
        """
        stats = getattr(self, "passenger_flow_stats", None)
        node = str(node)
        hour = int((now_time // 3600) % 24)
        min_count = int(stats.get("min_bucket_count", 1)) if stats else 1

        result = {"mean_boardings": None, "mean_alight_frac": None, "mean_intervisit_sec": None}

        def _fill(entry):
            # Skip buckets with too few real samples to trust as a mean estimate —
            # they fall through to the next, broader level instead.
            if not entry or entry.get("count", 0) < min_count:
                return
            for k in result:
                if result[k] is None and entry.get(k) is not None:
                    result[k] = entry[k]

        if stats:
            if going_forward:
                _fill(stats.get("by_route_stop_hour", {}).get((route_id, node, hour)))
                _fill(stats.get("by_route_stop", {}).get((route_id, node)))

            _fill(stats.get("by_stop_hour", {}).get((node, hour)))
            _fill(stats.get("by_stop", {}).get(node))

            g = stats.get("global", {})
        else:
            g = {}

        if result["mean_boardings"] is None:
            result["mean_boardings"] = g.get("mean_boardings", 1.0)
        if result["mean_alight_frac"] is None:
            result["mean_alight_frac"] = g.get("mean_alight_frac", 0.2)
        if result["mean_intervisit_sec"] is None:
            result["mean_intervisit_sec"] = g.get("mean_intervisit_sec", 600.0)

        return result

    def _get_arrival_rate(self, route_id: str, node, now_time: float, going_forward: bool = True) -> float:
        """Passengers/second accumulating at this (route, stop) between bus visits."""
        flow = self._get_flow_stats(route_id, node, now_time, going_forward)
        intervisit = max(float(flow["mean_intervisit_sec"]), 1.0)
        return max(0.0, float(flow["mean_boardings"])) / intervisit

    def _get_alight_frac(self, route_id: str, node, now_time: float, going_forward: bool = True) -> float:
        """Fraction of onboard passengers that alight when a bus reaches this stop."""
        flow = self._get_flow_stats(route_id, node, now_time, going_forward)
        return float(np.clip(flow["mean_alight_frac"], 0.0, 1.0))

    def _get_predicted_arrival_rate(self, route_id, node, now_time, going_forward=True):
        hour = int((now_time // 3600) % 24)
        entry = getattr(self, "predicted_flow_stats", {}).get((route_id, str(node), hour))
        if entry is None or entry.get("boardings_per_min") is None:
            return 0.0
        return max(0.0, float(entry["boardings_per_min"])) / 60.0  # pax/min -> pax/seg


    def _get_predicted_alight_frac(self, route_id, node, now_time, going_forward=True):
        hour = int((now_time // 3600) % 24)
        entry = getattr(self, "predicted_flow_stats", {}).get((route_id, str(node), hour))
        if entry is None or entry.get("alighting_rate") is None:
            return 0.2  # mesmo default do ramo real
        return float(np.clip(entry["alighting_rate"], 0.0, 1.0))
    
    
    def _initial_queue_estimate(self, route_id: str, node, now_time: float, going_forward: bool = True) -> float:
        """
        Seed value for a (route, stop) queue the first time it's touched in a
        simulated day. Using the historical mean-boardings-per-visit (rather than
        0) avoids an artificially empty queue for the very first bus of the day,
        since that historical average is itself already conditioned on a bus
        periodically servicing the stop.
        """
        flow = self._get_flow_stats(route_id, node, now_time, going_forward)
        return max(0.0, float(flow["mean_boardings"]))

    # ------------------------------------------------------------------
    # Pairwise headway (agent coordination / synchronization)
    # ------------------------------------------------------------------
    def _route_loop_position(self, agent: str):
        """
        Linearizes a bouncing route into a single monotonic "conveyor belt"
        coordinate over one round trip (0..loop_len-1), so that two agents on
        the same route can be compared by position regardless of direction.
        """
        state = self.agent_states.get(agent)
        if not state:
            return None, 0
        route_length = len(state["route"])
        if route_length <= 1:
            return 0, 1
        idx = state["route_idx"]
        going_forward = state.get("going_forward", True)
        loop_len = 2 * (route_length - 1)
        pos = idx if going_forward else (loop_len - idx)
        return pos, loop_len

    def _compute_pairwise_headway_gaps(self, agent: str):
        """
        Returns (leader_gap, follower_gap): the simulated-time gap (seconds) to
        the immediate preceding bus (leader, ahead on the route loop) and the
        immediate following bus (follower, behind), among other ACTIVE agents
        sharing this agent's route. Returns None for a side with no active peer.

        This replaces the population-wide RMS-of-all-recent-arrivals headway
        metric (self.headways / _sync_component) with a signal that is specific
        to THIS agent's own gap to its neighbors, which is what an agent can
        actually act on to avoid bunching or closing an excessive gap.
        """
        route_id = self.agent_route_id.get(agent)
        if not route_id or route_id == "unknown":
            return None, None

        my_pos, loop_len = self._route_loop_position(agent)
        if my_pos is None or loop_len <= 0:
            return None, None

        peers = [
            a for a in self.possible_agents
            if a != agent
            and self.agent_route_id.get(a) == route_id
            and self.agent_states.get(a, {}).get("status") == "active"
        ]
        if not peers:
            return None, None

        best_ahead = None   # smallest positive (peer_pos - my_pos) mod loop_len -> leader
        best_behind = None  # smallest positive (my_pos - peer_pos) mod loop_len -> follower

        for peer in peers:
            peer_pos, peer_loop_len = self._route_loop_position(peer)
            if peer_pos is None or peer_loop_len != loop_len:
                continue

            ahead_delta = (peer_pos - my_pos) % loop_len
            behind_delta = (my_pos - peer_pos) % loop_len

            # Note: >= 0, not > 0 — a peer at the EXACT same position (delta == 0) is the
            # maximum-bunching case (two buses literally at the same stop) and must count
            # as both "ahead" and "behind" with a zero gap, not be treated as "no peer".
            if ahead_delta >= 0 and (best_ahead is None or ahead_delta < best_ahead[0]):
                best_ahead = (ahead_delta, peer)
            if behind_delta >= 0 and (best_behind is None or behind_delta < best_behind[0]):
                best_behind = (behind_delta, peer)

        my_time = self.agent_times.get(agent, 0.0)
        leader_gap = None
        follower_gap = None
        if best_ahead is not None:
            leader_gap = abs(my_time - self.agent_times.get(best_ahead[1], my_time))
        if best_behind is not None:
            follower_gap = abs(my_time - self.agent_times.get(best_behind[1], my_time))

        return leader_gap, follower_gap

    def _normalize_headway_gap(self, gap) -> float:
        """
        Signed, normalized headway-gap feature for the observation space:
        0.0 = on the target headway (or no known peer -> neutral/"assume on
        schedule"), negative = bunched up (gap smaller than target), positive
        = larger gap than target. Clipped to [-1, 1].
        """
        if gap is None:
            return 0.0
        target = float(getattr(self.reward, "target_headway", 600.0))
        max_headway = 1800.0
        return float(np.clip((gap - target) / max_headway, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Replay logging (game-like map visualization)
    # ------------------------------------------------------------------
    def _flush_replay_log(self):
        """Writes the accumulated per-step replay log for the finished day to disk."""
        if not self.replay_log:
            return
        os.makedirs(self.replay_output_dir, exist_ok=True)
        fname = f"replay_ep{self.episode_counter}_{self.date}.json"
        out_path = os.path.join(self.replay_output_dir, fname)
        try:
            import json
            with open(out_path, "w") as f:
                json.dump(self.replay_log, f)
            print(f"[REPLAY] Saved {len(self.replay_log)} events to {out_path}")
        except Exception as e:
            print(f"[REPLAY] Failed to save replay log: {e}")
        self.replay_log = []


    def get_nearest_service_center(self, current_node):
        # Finds the nearest service center node, not dynamic yet
        return self.service_center_node
    
    def load_current_day_data(self):
        """Loads base data (daily or mean), always extracts service_day from daily,
        and applies quantum override if enabled.
        """

        service_day = None

        # =====================================================
        # 1️⃣ TRY TO LOAD DAILY (FOR SERVICE DAY)
        # =====================================================
        if self.total_days > 0:
            file_path = os.path.join(
                self.daily_data_path,
                self.daily_files[self.current_day_index]
            )

            with open(file_path, "rb") as f:
                day_data = pickle.load(f)

            service_day = day_data.get("date")

            print(
                f"\n📅 [ENV] Loading daily metadata for {service_day} "
                f"({self.current_day_index + 1}/{self.total_days})"
            )
        else:
            day_data = {}
            print("⚠️ No daily files found. service_day unavailable.")

        # =====================================================
        # 2️⃣ BASE DATA SELECTION
        # =====================================================
        if getattr(self, "use_only_mean_data", 0) == 1:
            print("📘 [ENV] Using MEAN values as base.")
            self._load_mean_values()
        else:
            print("📘 [ENV] Using DAILY values as base (with mean fallback).")

            self.avg_travel_time_AB = day_data.get(
                "avg_travel_times", self.avg_travel_time_AB
            )
            self.future_demand_at_B = day_data.get(
                "future_demand", self.future_demand_at_B
            )
            self.occupancy_rate = day_data.get(
                "occupancy_rate", self.occupancy_rate
            )
            self.uptime_normalized = day_data.get(
                "uptime_normalized", self.uptime_normalized
            )

            self._use_fallbacks()

        # =====================================================
        # 3️⃣ OCCUPANCY PREDICTION OVERRIDE
        # =====================================================
        if self.occupancy_source != "real":

            print(
                f"📈 [ENV] Trying occupancy override | "
                f"source={self.occupancy_source} | "
                f"service_day={service_day}"
            )

            self.occ_pred_by_route_seq = defaultdict(dict)

            self._override_occupancy_with_predictions_node_level(
                service_day=service_day
            )
            if self.occupancy_source in ("timesfm", "naive"): # Pra ser usado só para essas novas fontes de dados
                self._load_predicted_rates(service_day=service_day)

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

    def _override_occupancy_with_predictions_node_level(self, service_day):
        """
        Overrides occupancy_rate at NODE LEVEL using quantum or LSTM predictions.

        - Expects quantum predictions in ABSOLUTE passengers
        - Converts to normalized occupancy ∈ [0,1]
        - Follows the SAME logic used in generate_daily_data.py
        - Falls back gracefully when data is missing
        """

        BUS_CAPACITY = 80  # MUST match generate_daily_data.py

        print(
            f"⚛️ [ENV] Occupancy override attempt START | "
            f"source={self.occupancy_source} | "
            f"service_day={service_day}"
        )

        if service_day is None:
            print("⚠️ [ENV] No service_day found. Skipping occupancy.")
            return

        column_map = {

            # Quantum
            "quantum_qru": "y_pred_QRU",
            "quantum_lstm": "y_pred_LSTM",

            # TimesFM
            "timesfm": "y_pred_timesfm",
            "timesfm_ft": "y_pred_timesfm_ft",
            "naive": "y_pred_naive",
        }

        pred_column = column_map.get(self.occupancy_source)
        if pred_column is None:
            print(
                f"⚠️ [ENV] Unknown occupancy_source={self.occupancy_source}. "
                f"Skipping override."
            )
            return

        node_predictions = defaultdict(list)

        # ===============================
        # Load quantum predictions
        # ===============================
        if self.occupancy_source.startswith("quantum"):
            routes = self.quantum_routes
            base_path = self.quantum_data_path
        else:
            routes = self.prediction_routes
            base_path = self.prediction_data_path

        for route_id in routes:
            route_dir = os.path.join(base_path, route_id)

            if not os.path.isdir(route_dir):
                print(f"⚠️ [ENV] route dir not found | route={route_id}")
                continue

            csv_files = [f for f in os.listdir(route_dir) if f.endswith(".csv")]
            if not csv_files:
                print(f"⚠️ [ENV] No CSV found | route={route_id}")
                continue

            csv_path = os.path.join(route_dir, csv_files[0])

            print(f"⚛️ [ENV] Loading CSV | route={route_id} | file={csv_files[0]}")

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"⚠️ [ENV] Failed to read {csv_path}: {e}")
                continue

            day_df = df[df["service_day"] == service_day]
            if day_df.empty:
                print(f"⚠️ [ENV] No rows for route={route_id} | day={service_day}")
                continue

            if route_id not in self.real_routes:
                print(f"⚠️ [ENV] route_id={route_id} not found in real_routes")
                continue

            path = self.real_routes[route_id]

            for _, row in day_df.iterrows():
                seq = int(row["pt_sequence"]) - 1
                if seq < 0 or seq >= len(path):
                    continue

                value = row[pred_column]
                if pd.isna(value):
                    continue

                node_id = int(path[seq])
                raw_passengers = float(value)
                occupancy_norm = min(raw_passengers / BUS_CAPACITY, 1.0)

                node_predictions[node_id].append(occupancy_norm)

                if len(node_predictions[node_id]) == 1:
                    print(
                        f"🧪 [PRED-OCC RAW→NORM] "
                        f"source={self.occupancy_source} | "
                        f"day={service_day} | "
                        f"route={route_id} | "
                        f"node={node_id} | "
                        f"raw={raw_passengers:.2f} | "
                        f"norm={occupancy_norm:.4f}"
                    )

        # ===============================
        # Apply overrides
        # ===============================
        if not node_predictions:
            print(
                f"⚠️ [ENV] No quantum node-level data for {service_day}. "
                f"Using classical occupancy."
            )
            return

        overridden_nodes = 0
        for node, values in node_predictions.items():
            self.occupancy_rate[node] = sum(values) / len(values)
            overridden_nodes += 1

        # Build per-route, per-sequence prediction map (used by risk feature)
        # We store 0-based seq_idx (seq = pt_sequence - 1) to match route_idx convention.
        try:
            self.occ_pred_by_route_seq = {
                rid: {s: (sum(vals) / len(vals)) for s, vals in seq_map.items() if vals}
                for rid, seq_map in route_seq_predictions.items()
            }
        except Exception as e:
            print(f"⚠️ [ENV] Failed building occ_pred_by_route_seq: {e}")
            self.occ_pred_by_route_seq = defaultdict(dict)

        if self.debug_quantum and self.occ_pred_by_route_seq:
            # show a small sample for sanity
            sample_rid = next(iter(self.occ_pred_by_route_seq.keys()))
            sample_items = sorted(self.occ_pred_by_route_seq[sample_rid].items())[:5]
            print(f"🧪 [Q-PRED MAP] route={sample_rid} sample(seq->occ)={sample_items}")
        print(
            f"✅ [ENV] Occupancy predictions applied | "
            f"source={self.occupancy_source} | "
            f"nodes_updated={overridden_nodes}"
        )
    
    def _get_future_occupancies(self, agent, route, current_occupancy):
        """
        Returns the predicted occupancy for the next N stops in the
        direction the bus is currently traveling.

        Parameters
        ----------
        agent : str
            Agent id.

        route : list
            Complete route of the agent.

        current_occupancy : float
            Used as fallback if a future node has no prediction.

        Returns
        -------
        list[float]
            Occupancies for the next N stops.
        """

        future_occupancies = []

        lookahead = self.farf_prediction_steps

        state = self.agent_states[agent]
        current_idx = state["route_idx"]
        going_forward = state.get("going_forward", True)

        for step in range(1, lookahead + 1):

            if going_forward:
                future_idx = current_idx + step
            else:
                future_idx = current_idx - step

            # chegou ao fim da rota
            if future_idx < 0 or future_idx >= len(route):
                break

            future_node = route[future_idx]

            future_occ = self.occupancy_rate.get(
                future_node,
                current_occupancy
            )

            future_occupancies.append(float(future_occ))

        while len(future_occupancies) < lookahead:
            future_occupancies.append(current_occupancy)
        
        return future_occupancies

    def _load_predicted_rates(self, service_day):
        """
        Loads per-(route, stop, hour) boarding-rate and alighting-rate predictions,
        for occupancy_source in {"timesfm", "naive"} — the two sources that ship
        exports_rates files. Quantum sources (quantum_qru/quantum_lstm) don't have
        these files and keep using the older direct occupancy_rate override.

        alighting uses the *ratio* column (alighting_per_veh / lag_loading_per_veh,
        precomputed upstream) rather than the direct alighting-rate prediction —
        per the data provider, predicting alighting and loading separately and then
        dividing outperformed predicting the rate directly.

        Populates self.predicted_flow_stats: dict[(route_id, stop_id, hour)] -> {
            "boardings_per_min": float,
            "alighting_rate": float,
        }
        """
        print("🔥 LOADED PATCHED VERSION 🔥 ")
        column_map = {
            "timesfm": {"boardings": "y_pred_timesfm_boardings_per_min",
                        "alighting": "y_pred_timesfm_ratio_alighting_rate"},
            "naive":   {"boardings": "y_pred_naive_boardings_per_min",
                        "alighting": "y_pred_naive_alighting_rate"},
        }

        cols = column_map.get(self.occupancy_source)
        self.predicted_flow_stats = {}

        if cols is None:
            print(f"⚠️ [ENV] source={self.occupancy_source} has no rate files. Skipping rates load.")
            return
        if service_day is None:
            print("⚠️ [ENV] No service_day found. Skipping predicted rates load.")
            return

        for route_id in self.prediction_routes:
            route_dir = os.path.join(self.rates_data_path, route_id)
            if not os.path.isdir(route_dir):
                print(f"⚠️ [ENV] rates dir not found | route={route_id}")
                continue

            boardings_df = self._read_first_csv_matching(route_dir, "rates_boardings_per_min")
            alighting_df = self._read_first_csv_matching(route_dir, "rates_alighting_rate")

            if boardings_df is None or alighting_df is None:
                print(f"⚠️ [ENV] Missing rate CSVs | route={route_id}")
                continue

            for df, col_key, dict_key in (
                (boardings_df, cols["boardings"], "boardings_per_min"),
                (alighting_df, cols["alighting"], "alighting_rate"),
            ):
                if col_key not in df.columns:
                    print(f"⚠️ [ENV] Column {col_key} missing in file for route={route_id}")
                    continue

                day_df = df[df["service_day"] == service_day]
                for _, row in day_df.iterrows():
                    val = row.get(col_key)
                    if pd.isna(val):
                        continue
                    stop_id = str(int(row["stop_id"]))
                    hour = int(row["hour"])
                    key = (route_id, stop_id, hour)
                    self.predicted_flow_stats.setdefault(key, {})[dict_key] = float(val)

        if np.random.rand() < 0.02 and self.debug:
            for h in [2, 7, 12, 18, 22]:
                real_vals = []
                for route_id in self.prediction_routes:
                    # pega um stop qualquer da rota pra comparar
                    stop = self.real_routes[route_id][0]
                    real_vals.append(self._get_flow_stats(route_id, stop, h*3600, True)["mean_alight_frac"])
                pred_vals = [v["alighting_rate"] for (r, s, hh), v in self.predicted_flow_stats.items() if hh == h and "alighting_rate" in v]
                print(f"hour={h:02d} | real_mean={np.mean(real_vals):.3f} | pred_mean={np.mean(pred_vals):.3f}")
            
            vals = [v["alighting_rate"] for v in self.predicted_flow_stats.values() if "alighting_rate" in v]
            print(f"alight_frac: min={min(vals):.3f} max={max(vals):.3f} mean={np.mean(vals):.3f}")

        print(
            f"✅ [ENV] Predicted rates loaded | "
            f"source={self.occupancy_source} | keys={len(self.predicted_flow_stats)}"
        )


    def _read_first_csv_matching(self, directory, prefix):
        matches = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".csv")]
        if not matches:
            return None
        try:
            return pd.read_csv(os.path.join(directory, matches[0]))
        except Exception as e:
            print(f"⚠️ [ENV] Failed to read {matches[0]}: {e}")
            return None

# This is the base class for reward classes
class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError
    def getRewardSoftMin(self, state, previousState, action, target, graph):
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
                 max_sync_rel_std: float = 1.0,          # >1 é truncado
                 softmin_temperature=0.2):
        super().__init__()
        # self.waitTimeDict can be used if needed for other metrics
        self.waitTimeDict = waitTimeDict or {}

        # Adjustable weights (sum doesn't need to be 1; we do weighted average).
        # Previously uptime_bonus/sync_score/energy_efficiency defaulted to 0.0 and no
        # training script overrode them, so the "compound" reward this class documents
        # was in practice 100% occupancy_score. Defaulting all four components to equal
        # weight restores the intended multi-objective signal (including making the
        # headway-synchronization term below actually reach the policy gradient).
        self.reward_weights = reward_weights or {
            "occ_penalty": 1.0,
            "uptime_bonus": 0.0,
            "sync_score": 0.0,
            "energy_efficiency": 0.0
        }

        self.occupancy_range = occupancy_range
        self.target_headway = float(target_headway_seconds)
        self.max_sync_rel_std = float(max_sync_rel_std)
        self.softmin_temperature = float(softmin_temperature)

        # =====================================================
        # FARF occupancy reward parameters
        # =====================================================

        self.farf_prediction_steps = 3      # Number of future stops considered
        self.farf_gamma = 0.8               # Discount factor
        
        # Occupation factor parameters
        self.farf_desired_occ = 0.75
        self.farf_delta = 0.15
        self.farf_p = 0.10
        self.farf_overcrowding_scale = 2.0

        weights = [
            self.farf_gamma ** i
            for i in range(self.farf_prediction_steps + 1)
        ]

        weight_sum = sum(weights)

        self.farf_alphas = [
            w / weight_sum
            for w in weights
        ]

    def _occ_component(self, occupancy: float) -> float:
        """
        Returns a value in [0, 1], where 0 = perfect in ideal range; 1 = far off
        Then we apply negative sign when composing the reward
        """
        min_occ, max_occ = self.occupancy_range
        if np.random.rand() < 000.1: # DEBUG
            print(f"occupancy = {occupancy:.3f}")
        
        if occupancy < min_occ:
            return min(1.0, (min_occ - occupancy) ** 2 / (min_occ ** 2 + 1e-8))
        if occupancy > max_occ:
            return min(1.0, (occupancy - max_occ) ** 2 / ((1.0 - max_occ) ** 2 + 1e-8))
        return 0.0


    def _occupation_factor(self, occupancy: float):
        """
        FARF occupation factor Θ(x).

        Returns:
            [-s*k , 1]
        """

        g = np.exp(
            -((occupancy - self.farf_desired_occ) ** 2) /
            (2 * self.farf_p ** 2)
        )

        k = np.exp(
            -(self.farf_delta ** 2) /
            (2 * self.farf_p ** 2)
        )

        if occupancy < self.farf_desired_occ - self.farf_delta:
            return g - k

        elif occupancy > self.farf_desired_occ + self.farf_delta:
            return self.farf_overcrowding_scale * (g - k)

        else:
            return (g - k) / (1.0 - k)

    def _farf_occupancy_reward(
        self,
        current_occ,
        predicted_occupancies
    ):
        """
        Computes the FARF occupancy reward.

        R = Σ a_i · Θ_i

        where:
            a_i : normalized temporal weights
            Θ_i : occupation factor for the current/future stop
        """

        values = [current_occ] + predicted_occupancies

        reward = 0.0

        for alpha, occ in zip(self.farf_alphas, values):

            reward += alpha * self._occupation_factor(occ)

        return reward

    def _sync_component(self, headways: list) -> float:
        """
        Measures regularity in [0, 1]: 1 = perfect (intervals very close to target),
        0 = very irregular (relative deviation >= max_sync_rel_std)
        """
        
        if np.random.rand() < 000.1:
            print(f"headways = {headways}")

        if not headways or len(headways) < 2: # Estou passando todos os agentes e não os agentes na rota especifica
            # print("SYNC RETURN 0 -> poucos ônibus:", headways)
            return 0.0  # Not enough information to assess regularity

        headways = sorted(headways)  # ORDENAR
        
        # intervals in seconds
        intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)]
        # <<< DEBUG AQUI >>>
        #if len(intervals) > 0 and random.random() < 0.05:  # amostragem
        #    print(
        #        f"[SYNC DEBUG] intervals={intervals} | "
        #        f"target={self.target_headway}"
        #    )
        
        # remove noise/invalid intervals não existe “intervalo negativo” entre dois veículos então é removido se tiver
        intervals = [x for x in intervals if x > 0]
        if len(intervals) < 2:
            return 0.0

        # Desvio RMS (Root Mean Square) dos intervalos em relação ao valor ideal (600 (10 minutos))
        # Para cada intervalo real, eu to medindo o quanto ele errou em relação ao desejado 10 minutos 
        diffs = [(x - self.target_headway) for x in intervals]
        mean_sq = sum(d * d for d in diffs) / len(diffs)
        rms = mean_sq ** 0.5  # in seconds

        # Normalized relative deviation (0=perfect, 1=bad limit)
        rel = min(1.0, rms / (self.max_sync_rel_std * self.target_headway + 1e-8)) # rms erro absoluto (em segundos) é normalizado

        # Convert to "score" in [0,1], where 1 is good
        return 1.0 - rel

    def _pairwise_sync_component(self, leader_gap, follower_gap) -> float:
        """
        Headway-regularity score in [0, 1] based on THIS agent's own gap to its
        immediate leader/follower on the route (see
        parallel_env._compute_pairwise_headway_gaps), rather than the population-wide
        RMS of every recent arrival used by _sync_component. This is what actually
        drives the coordination signal into scalarize()/getObjectives() below; a
        neutral 0.5 is returned when no active peer is known on either side (e.g. a
        route currently has only one active bus), since we have no basis to judge
        bunching/gaps in that case.
        """
        gaps = [g for g in (leader_gap, follower_gap) if g is not None]
        if not gaps:
            return 0.5

        scores = []
        for gap in gaps:
            rel = min(1.0, abs(gap - self.target_headway) / (self.max_sync_rel_std * self.target_headway + 1e-8))
            scores.append(1.0 - rel)

        return float(np.mean(scores))

    def _efficiency_component(self, estimated_time: float, expected_time: float) -> float:
        """
        Travel efficiency in [0,1]. 1 = equal/to less than expected; 0 = worse than expected.
        """
        if expected_time <= 0:
            return 0.0
        ratio = estimated_time / (expected_time + 1e-8)
        return float(np.clip(1.0 - ratio, 0.0, 1.0))
    
    def getReward_OLD(
        self,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None, is_raining=False, 
        reward_type="normal", last_rain_eff=0.0 
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

        # --- Base efficiency ---
        eff = self._efficiency_component(float(estimated_time), float(expected_time))
        
        # --- Rain efficiency used only when rains ---
        modified_eff = eff
        rain_bonus_reward = 0.0

        # ======================================================
        # 🌧️ RAIN LOGIC (can be turned OFF with reward_type="normal")
        # ======================================================
        if is_raining and reward_type != "normal":

            # -------- Penalization --------
            if reward_type == "penalization":
                rain_strength = 0.2 # Penalization intensity Lamda on article 
                # modified_eff = eff * rain_strength * (1.0 - eff) # If it's raining, efficiency is penalized more
                modified_eff = eff * (1.0 - rain_strength * (1.0 - eff))

            # -------- Bonus on efficiency --------
            elif reward_type == "bonus":
                #rain_bonus_eff = 0.2 # Bonus intensity
                #modified_eff = np.clip(eff + rain_bonus_eff * eff, 0.0, 1.0) # If it's raining, efficiency is rewarded more
                modified_eff = eff

           # -------- Relative improvement bonus (reward-level) --------
            # rain_bonus_reward = 0.2 if eff > last_rain_eff else 0.0 INICIALMENTE VAMOS USAR SO O BONUS OU PENALIZAÇÃO

        # Safety clip
        modified_eff = float(np.clip(modified_eff, 0.0, 1.0))

        
        if reward_type != "normal" and np.random.rand() < 0.005:
            print(
                f"[RAIN DBG] raining={is_raining} "
                f"eff_base={eff:.3f} "
                f"eff_modified={modified_eff:.3f} "
                f"last_rain_eff={last_rain_eff:.3f}"
            )

        assert 0.0 <= occ_pen <= 1.0, f"occ_pen fora do range: {occ_pen}"
        assert 0.0 <= uptime <= 1.0, f"uptime fora do range: {uptime}"
        assert 0.0 <= sync <= 1.0, f"sync fora do range: {sync}"
        assert 0.0 <= eff <= 1.0, f"eff fora do range: {eff}"


        # Weighted combination (keeping each term in [-1, +1])
        # occ_pen enters with NEGATIVE sign
        w = self.reward_weights

        # ------------------------------------------------------
        # Efficiency weight (ONLY bonus doubles it)
        # ------------------------------------------------------
        eff_weight = w["energy_efficiency"]
        if is_raining and reward_type == "bonus":
            eff_weight = 2.0 * w["energy_efficiency"]
        
        
        reward = (
            -w["occ_penalty"] * occ_pen +
             w["uptime_bonus"] * uptime + # uptime esta sendo um positivo que esta tendendo a zero, enquanto ele vai
             w["sync_score"] * sync +
             eff_weight * modified_eff + rain_bonus_reward # w["energy_efficiency"] * eff
        )

        # Normalize by the sum of weights to keep magnitude around ~[-1, +1]
        weight_sum = (abs(w["occ_penalty"]) + w["uptime_bonus"] + w["sync_score"] + w["energy_efficiency"])
        if weight_sum > 0:
            reward = reward / weight_sum

        assert weight_sum >= 0.0, f"weight_sum negativo: {weight_sum}"
        
        # Clip final for numerical stability
        # reward += 0.2
        # print(f"\n--- Recompensa Final Normalizada: {final_reward:.4f} ---")
        reward = float(np.clip(reward, -1.0, 1.0))

        if np.random.rand() < 0.002:
            print(
                f"[EFF CHECK] "
                f"rain={is_raining} | "
                f"eff_base={eff:.3f} | "
                f"modified_eff={modified_eff:.3f} | "
                f"est={estimated_time:.1f} | "
                f"exp={expected_time:.1f}"
            )

        if np.random.rand() < 0.002:
            print(
                f"[REWARD BREAKDOWN] "
                f"occ={occ_pen:.3f} | "
                f"uptime={uptime:.3f} | "
                f"sync={sync:.3f} | "
                f"eff={eff:.3f} | "
                f"rain={is_raining} | "
                f"bonus={rain_bonus_reward:.3f} | "
                f"final={reward:.3f}"
            )


        if np.random.rand() < 0.005:  # ~0.5% dos steps
            print(
                f"[REWARD DBG] "
                f"occ_pen={occ_pen:.3f} "
                f"uptime={uptime:.3f} "
                f"sync={sync:.3f} "
                f"eff={eff:.3f} | "
                f"weights={w} | "
                f"final={reward:.3f}"
            )

        return reward

    def getReward(
        self,
        agent,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None, is_raining=False, 
        reward_type="normal", last_rain_eff=0.0 
    ):

        objectives = self.getObjectives(
            agent,
            new_state,
            previous_state,
            action,
            target,
            network,
            estimated_time,
            expected_time,
            delay,
            agent_state,
            headways,
            is_raining,
            reward_type,
            last_rain_eff

        )

        rain_bonus_reward = 0.0

        eff_weight = self.reward_weights["energy_efficiency"]

        if is_raining and reward_type == "bonus":

            eff_weight *= 2.0

        reward = self.scalarize(

            objectives,

            eff_weight,

            rain_bonus_reward
        )

        if np.random.rand() < 0.002:
            print(
                f"[EFF CHECK] "
                f"rain={is_raining} | "
                f"eff={objectives['efficiency_score']:.3f} | "
                f"est={estimated_time:.1f} | "
                f"exp={expected_time:.1f}"
            )

        if np.random.rand() < 0.002:
            print(
                f"[REWARD BREAKDOWN] "
                f"occ={objectives['occupancy_score']:.3f} | "
                f"uptime={objectives['uptime_score']:.3f} | "
                f"sync={objectives['sync_score']:.3f} | "
                f"eff={objectives['efficiency_score']:.3f} | "
                f"rain={is_raining} | "
                f"bonus={rain_bonus_reward:.3f} | "
                f"final={reward:.3f}"
            )


        if np.random.rand() < 0.005:  # ~0.5% dos steps
            print(
                f"[REWARD DBG] "
                f"occ={objectives['occupancy_score']:.3f} "
                f"uptime={objectives['uptime_score']:.3f} "
                f"sync={objectives['sync_score']:.3f} "
                f"eff={objectives['efficiency_score']:.3f} "
                f"weights={self.reward_weights} "
                f"final={reward:.3f}"
            )

        return reward

    # ===========================
    # RECOVERING TRAINING OBJECTIVES 
    # ===========================
    def getObjectives(
        self,
        agent,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None,
        is_raining=False,
        reward_type="normal",
        last_rain_eff=0.0
    ):
        """
        Calculates all objectives independently.

        Nothing is scalarized here.
        Nothing uses weights.

        This method becomes the "heart" of the reward morl system
        """

        # Normalized components
        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0
        occupancy = 0.0 

        if agent_state is not None:

            # =====================================================
            # OLD OCCUPANCY REWARD (Baseline)
            # =====================================================
            #
            # occupancy = float(agent_state.get("occupancy", 0.0))
            # occ_pen = self._occ_component(occupancy)
            # occ_score = 1.0 - occ_pen # ESTAMOS INVERTENDO A PENALIDADE PARA QUE TUDO SEJA O MESMO SENTIDO, QUANTO MENOR PIOR 
            # =====================================================
            # NEW FARF OCCUPANCY REWARD
            # =====================================================
            occupancy = float(agent_state.get("occupancy", 0.0))

            predicted = agent_state.get("predicted_occupancies", [])

            occ_score = self._farf_occupancy_reward(
                occupancy,
                predicted
            )
            # =====================================================
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

    
                
        
        
        # Population-level RMS metric, kept only as a secondary debug diagnostic —
        # NOT what feeds scalarize() below (see _pairwise_sync_component).
        _sync_population_debug = self._sync_component(headways or [])

        leader_gap, follower_gap = (None, None)
        if hasattr(self, "env") and self.env is not None:
            leader_gap, follower_gap = self.env._compute_pairwise_headway_gaps(agent)
        sync = self._pairwise_sync_component(leader_gap, follower_gap)

        # registra pra métrica de fim de episódio
        if hasattr(self, "env") and self.env is not None:

            # Valor bruto de ocupação
            self.env.episode_occupancy_values[agent].append(occupancy)

            # =====================================================
            # BASELINE
            # =====================================================
            # self.env.episode_occupancy_ideal_flags[agent].append(
            #     1.0 if occ_pen == 0.0 else 0.0
            # )

            # =====================================================
            # FARF
            # =====================================================
            inside_interval = (
                abs(occupancy - self.farf_desired_occ) <= self.farf_delta
            )

            self.env.episode_occupancy_ideal_flags[agent].append(
                1.0 if inside_interval else 0.0
            )

        # --- Base efficiency ---
        eff = self._efficiency_component(float(estimated_time), float(expected_time))
        
        # --- Rain efficiency used only when rains ---
        modified_eff = eff
        rain_bonus_reward = 0.0

        # ======================================================
        # 🌧️ RAIN LOGIC (can be turned OFF with reward_type="normal")
        # ======================================================
        if is_raining and reward_type != "normal":

            # -------- Penalization --------
            if reward_type == "penalization":
                rain_strength = 0.2 # Penalization intensity Lamda on article 
                # modified_eff = eff * rain_strength * (1.0 - eff) # If it's raining, efficiency is penalized more
                modified_eff = eff * (1.0 - rain_strength * (1.0 - eff))

            # -------- Bonus on efficiency --------
            elif reward_type == "bonus":
                #rain_bonus_eff = 0.2 # Bonus intensity
                #modified_eff = np.clip(eff + rain_bonus_eff * eff, 0.0, 1.0) # If it's raining, efficiency is rewarded more
                modified_eff = eff

           # -------- Relative improvement bonus (reward-level) --------
            # rain_bonus_reward = 0.2 if eff > last_rain_eff else 0.0 INICIALMENTE VAMOS USAR SO O BONUS OU PENALIZAÇÃO

        modified_eff = float(np.clip(modified_eff, 0.0, 1.0))

        # transforma penalidade em score

        # assert 0 <= occ_score <= 1 # OLD OCCUPANCY REWARD (Baseline)
        assert -1.0 <= occ_score <= 1.0 # FARF occupancy reward may become negative.
        assert 0 <= uptime <= 1
        assert 0 <= sync <= 1
        assert 0 <= modified_eff <= 1

        return {

                "occupancy_score": occ_score,

                "uptime_score": uptime,

                "sync_score": sync,

                "efficiency_score": modified_eff

        }
    

    # ===========================
    # NOVO MÉTODO
    # ===========================
    def getVectorReward(
        self,
        agent, 
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None,
        headways=None,
        is_raining=False,
        reward_type="normal",
        last_rain_eff=0.0
    ):

        obj = self.getObjectives(
            agent, 
            new_state,
            previous_state,
            action,
            target,
            network,
            estimated_time,
            expected_time,
            delay,
            agent_state,
            headways,
            is_raining,
            reward_type,
            last_rain_eff

        )

        return np.array([

            obj["occupancy_score"],

            obj["uptime_score"],

            obj["sync_score"],

            obj["efficiency_score"]

        ])

    def scalarize(
        self,
        objectives,
        eff_weight,
        rain_bonus_reward=0.0
    ):

        w = self.reward_weights

        reward = (

            w["occ_penalty"] *
            objectives["occupancy_score"]

            +

            w["uptime_bonus"] *
            objectives["uptime_score"]

            +

            w["sync_score"] *
            objectives["sync_score"]

            +

            eff_weight *
            objectives["efficiency_score"]

            +

            rain_bonus_reward

        )

        weight_sum = (

            w["occ_penalty"]

            +

            w["uptime_bonus"]

            +

            w["sync_score"]

            +

            eff_weight

        )

        if weight_sum > 0:

            reward /= weight_sum

        reward = float(np.clip(reward, -1.0, 1.0))

        return reward
        

    # METODOS MORL DE RECOMPENSA LINEAR 
    def getRewardHard(
        self,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None
    ):
 
       # Hard-min (minimax) scalarization - Worst-Case Optimization (Minimax / Max-Min)
       # - converte componentes para objetivos "maior é melhor"
       # - calcula reward = min(obj_i) (considerando pesos)
       # - normaliza/clipa para manter contrato [-1, 1]


        # --- 1) extrai componentes (mesma lógica que já tinha) ---
        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))

        # --- 2) prepara objetivos "maior é melhor" ---
        # occ_pen é uma penalidade (0 = ótimo, 1 = muito ruim), então invertemos:
        obj_occ = -occ_pen        # agora: maior é melhor (menos penalidade => maior)
        obj_uptime = uptime       # já maior é melhor
        obj_sync = sync           # já maior é melhor
        obj_eff = eff             # já maior é melhor

        # --- 3) aplica pesos por objetivo (se peso == 0 => ignorar) ---
        w = self.reward_weights  # dicionário esperado: occ_penalty, uptime_bonus, sync_score, energy_efficiency
        w_occ = abs(w.get("occ_penalty", 0.0))
        w_up  = float(w.get("uptime_bonus", 0.0))
        w_sync = float(w.get("sync_score", 0.0))
        w_eff = float(w.get("energy_efficiency", 0.0))

        # Se um peso for zero, colocamos um sentinel alto (1.0) para que ele não seja o min.
        # Alternativa: simplesmente não incluir o objetivo na lista; optamos por incluir sentinel
        # para preservar a consistência de normalização.
        objs = []
        if w_occ != 0.0:
            objs.append(w_occ * obj_occ)
        else:
            objs.append(1.0)

        if w_up != 0.0:
            objs.append(w_up * obj_uptime)
        else:
            objs.append(1.0)

        if w_sync != 0.0:
            objs.append(w_sync * obj_sync)
        else:
            objs.append(1.0)

        if w_eff != 0.0:
            objs.append(w_eff * obj_eff)
        else:
            objs.append(1.0)

        # --- 4) normalização simples por maior peso ativo (evita escala estranha) ---
        max_w = max(w_occ, w_up, w_sync, w_eff, 1.0)

        # ### ALTERAÇÃO MINIMAX: HARD-MIN ###
        reward = min(objs) / max_w
        # ### FIM DA ALTERAÇÃO MINIMAX ###

        # --- 5) garantia de contrato e estabilidade ---
        reward = float(np.clip(reward, -1.0, 1.0))
        
        if np.random.rand() < 0.005:  # ~0.5% dos steps
            print(
                f"[REWARD DBG] "
                f"occ_pen={occ_pen:.3f} "
                f"uptime={uptime:.3f} "
                f"sync={sync:.3f} "
                f"eff={eff:.3f} | "
                f"weights={w} | "
                f"final={reward:.3f}"
            )
        
        return reward

    def getRewardSoftMin(
            self,
            new_state, previous_state, action, target, network,
            estimated_time, expected_time, delay,
            agent_state=None, headways=None
        ):
            # ============================================================
            # 1 - EXTRAÇÃO DOS COMPONENTES (OBJETIVOS SEPARADOS)
            # ============================================================
            # Aqui nós explicitamente mantemos múltiplos objetivos R_i,
            # o que caracteriza o problema como Multi-Objective RL (MORL).
            #
            # Cada componente é normalizado para [0, 1] antes da agregação.

            occ_pen = 0.0   # penalidade de ocupação (quanto mais longe do ideal, pior)
            uptime = 0.0    # tempo de atividade do veículo
            sync = 0.0      # regularidade de headways
            eff = 0.0       # eficiência temporal

            if agent_state is not None:
                occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
                uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

            sync = self._sync_component(headways or [])
            eff = self._efficiency_component(float(estimated_time), float(expected_time))

            # ============================================================
            # 2 - DEFINIÇÃO DOS OBJETIVOS (MAIOR = MELHOR)
            # ============================================================
            # Para aplicar Soft Max-Min, todos os objetivos
            # precisam estar no mesmo sentido semântico:
            # - valores maiores significam comportamento melhor
            #
            # Por isso, penalidades são invertidas.

            obj_occ = -occ_pen     # quanto menor a penalidade, maior o objetivo
            obj_uptime = uptime
            obj_sync = sync
            obj_eff = eff

            # ============================================================
            # 3) SELEÇÃO DOS OBJETIVOS ATIVOS E APLICAÇÃO DE PESOS
            # ============================================================
            # Diferente de uma soma ponderada clássica, aqui os pesos
            # NÃO definem diretamente a importância final,
            # mas apenas a escala relativa de cada objetivo.
            #
            # Objetivos com peso zero são removidos da escalarização pra não dar BO 

            w = self.reward_weights
            objs = []
            labels = []

            if w["occ_penalty"] > 0:
                objs.append(abs(w["occ_penalty"]) * obj_occ)
                labels.append("occ")

            if w["uptime_bonus"] > 0:
                objs.append(w["uptime_bonus"] * obj_uptime)
                labels.append("uptime")

            if w["sync_score"] > 0:
                objs.append(w["sync_score"] * obj_sync)
                labels.append("sync")

            if w["energy_efficiency"] > 0:
                objs.append(w["energy_efficiency"] * obj_eff)
                labels.append("eff")

            objs = np.array(objs, dtype=np.float32)

            # ============================================================
            # 4) SOFT MAX-MIN (SOFTMIN SCALARIZATION)
            # ============================================================
            # Este é o núcleo MORL do método.
            #
            # A ideia do Soft Max-Min é:
            #   - dar mais peso aos objetivos com PIOR desempenho
            #   - sem ignorar completamente os outros objetivos
            #
            # Isso é feito aplicando um softmin sobre os objetivos.

            T = self.softmin_temperature  # temperatura controla quão "duro" é o min no exemplo base ta pra 0.2
            # T → 0  => aproxima minimax (hard min)
            # T alto => aproxima média ponderada

            # Softmin é implementado como softmax sobre o negativo
            logits = -objs / (T + 1e-8)

            # Estabilidade numérica (remove o maior logit)
            weights = np.exp(logits - np.max(logits))
            weights = weights / (np.sum(weights) + 1e-8)

            # A reward final é uma combinação ponderada,
            # onde objetivos piores recebem mais peso automaticamente.
            reward = float(np.sum(weights * objs))

            # ============================================================
            # 5) NORMALIZAÇÃO FINAL
            # ============================================================
            # Garante contrato esperado pelo algoritmo de RL
            # e evita instabilidade numérica.
            reward = float(np.clip(reward, -1.0, 1.0))

            # ============================================================
            # 6) DEBUG (INTERPRETABILIDADE)
            # ============================================================
            # Esse log é extremamente útil para validar MORL:
            # pra conseguir ver explicitamente
            #   - quais objetivos estão piores
            #   - como os pesos se redistribuem dinamicamente
            if np.random.rand() < 0.005:
                dbg = ", ".join(f"{l}={o:.3f}" for l, o in zip(labels, objs))
                print(
                    f"[SOFTMIN DBG] T={T:.2f} | "
                    f"objs=[{dbg}] | "
                    f"weights={weights.round(3)} | "
                    f"reward={reward:.3f}"
                )

            return reward

    def getRewardMaxMedian(
            self,
            new_state, previous_state, action, target, network,
            estimated_time, expected_time, delay,
            agent_state=None, headways=None
        ):
        # ============================================================
        # 1) EXTRAÇÃO DOS COMPONENTES (OBJETIVOS MORL)
        # ============================================================
        # Cada componente representa um objetivo distinto R_i.
        # Todos são normalizados previamente para [0, 1]

        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))

        # ============================================================
        # 2) CONVERSÃO PARA OBJETIVOS "MAIOR = MELHOR"
        # ============================================================
        # Para aplicar qualquer escalarização MORL,
        # todos os objetivos precisam ter a mesma semântica.
        #
        # Penalidades são invertidas.

        obj_occ = -occ_pen
        obj_uptime = uptime
        obj_sync = sync
        obj_eff = eff

        # ============================================================
        # 3) SELEÇÃO DOS OBJETIVOS ATIVOS
        # ============================================================
        # Diferente da soma ponderada:
        # - pesos NÃO entram como multiplicadores
        # - eles funcionam apenas como liga/desliga de objetivos
        #
        # Isso preserva a propriedade do Max-Median que é o que estamos aplicando aqui no caso

        w = self.reward_weights
        objs = []
        labels = []

        if w["occ_penalty"] > 0:
            objs.append(obj_occ)
            labels.append("occ")

        if w["uptime_bonus"] > 0:
            objs.append(obj_uptime)
            labels.append("uptime")

        if w["sync_score"] > 0:
            objs.append(obj_sync)
            labels.append("sync")

        if w["energy_efficiency"] > 0:
            objs.append(obj_eff)
            labels.append("eff")

        objs = np.array(objs, dtype=np.float32)

        # ============================================================
        # 4) MAX-MEDIAN (ESCALARIZAÇÃO POR MEDIANA)
        # ============================================================
        # Aqui está o núcleo da técnica:
        #
        # - Ordenamos implicitamente os objetivos
        # - Selecionamos o valor mediano
        #
        # Esse valor representa o "desempenho típico"
        # da política naquele step.

        reward = float(np.median(objs))

        # ============================================================
        # 5) NORMALIZAÇÃO FINAL
        # ============================================================
        # Garante compatibilidade com o algoritmo de RL
        # e estabilidade numérica.

        reward = float(np.clip(reward, -1.0, 1.0))

        # ============================================================
        # 6) DEBUG (INTERPRETABILIDADE MORL)
        # ============================================================
        # Útil para verificar:
        # - quais objetivos estão extremos tanto pra cima quanto pra baixo
        # - qual deles está definindo a mediana, tambvém pra ver se tem algum bug rolando

        if np.random.rand() < 0.005:
            dbg = ", ".join(f"{l}={o:.3f}" for l, o in zip(labels, objs))
            print(
                f"[MAX-MEDIAN DBG] "
                f"objs=[{dbg}] | "
                f"median={reward:.3f}"
            )

        if np.random.rand() < 0.005:  # ~0.5% dos steps
            print(
                f"[REWARD DBG] "
                f"occ_pen={occ_pen:.3f} "
                f"uptime={uptime:.3f} "
                f"sync={sync:.3f} "
                f"eff={eff:.3f} | "
                f"weights={w} | "
                f"final={reward:.3f}"
            )

        return reward

    def getRewardLowerQuantile(
        self,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None,
        alpha=1/3
    ):
        # ============================================================
        # CONTEXTO GERAL ESSE AQUI ME CONFUNDIU UM POUCO
        # ============================================================
        # Este método implementa a escalarização Multi Objetivo conhecida
        # como Lower Quantile Optimization.
        #
        # A ideia central NÃO é:
        #   - otimizar a média dos objetivos
        #   - nem otimizar apenas o pior caso
        #
        # Mas sim:
        #   - otimizar a "faixa inferior" dos objetivos
        #
        # Em outras palavras:
        #   "garanta que os objetivos ruins não estejam ruins demais",
        # sem se tornar excessivamente conservador.
        #
        # Matematicamente igual no artigo:
        #   f(R1,...,Rn) = Q_α({Ri})
        #
        # onde α define qual fração inferior dos objetivos importa.
        # Neste experimento, α = 1/3.
        # ============================================================


        # ============================================================
        # 1) EXTRAÇÃO DOS COMPONENTES DE RECOMPENSA
        # ============================================================
        # Aqui extraímos os mesmos componentes já usados nas outras
        # funções de reward.
        #
        # IMPORTANTE:
        #   Cada componente é normalizado para [0, 1].
        #   Neste ponto ainda NÃO fazemos nenhuma agregação.

        occ_pen = 0.0   # Penalidade de ocupação (0 = ideal, 1 = muito ruim)
        uptime = 0.0    # Fração de tempo ativo (0 = ruim, 1 = perfeito)
        sync = 0.0      # Regularidade de headways (0 = ruim, 1 = perfeito)
        eff = 0.0       # Eficiência temporal (0 = ruim, 1 = perfeito)

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))


        # ============================================================
        # 2) CONVERSÃO PARA OBJETIVOS "MAIOR = MELHOR"
        # ============================================================
        # Para MORL, todos os objetivos precisam ter o mesmo sentido
        # semântico: valores maiores indicam comportamento melhor.
        #
        # - uptime, sync e eff já seguem essa lógica
        # - occ_pen é uma penalidade, então invertimos o sinal

        obj_occ = -occ_pen      # menor penalidade → valor maior
        obj_uptime = uptime
        obj_sync = sync
        obj_eff = eff


        # ============================================================
        # 3) APLICAÇÃO DE PESOS (ESCALA RELATIVA)
        # ============================================================
        # Diferente de uma soma ponderada tradicional:
        #   → aqui os pesos NÃO definem contribuição final direta
        #   → eles apenas ajustam a escala relativa entre objetivos
        #
        # Objetivos com peso zero são ignorados completamente,
        # evitando que entrem na ordenação e afetem o quantil, aquele α la

        w = self.reward_weights
        objs = []
        labels = []

        if w["occ_penalty"] > 0:
            objs.append(abs(w["occ_penalty"]) * obj_occ)
            labels.append("occ")

        if w["uptime_bonus"] > 0:
            objs.append(w["uptime_bonus"] * obj_uptime)
            labels.append("uptime")

        if w["sync_score"] > 0:
            objs.append(w["sync_score"] * obj_sync)
            labels.append("sync")

        if w["energy_efficiency"] > 0:
            objs.append(w["energy_efficiency"] * obj_eff)
            labels.append("eff")

        # Converte para numpy para facilitar ordenação
        objs = np.array(objs, dtype=np.float32)


        # ============================================================
        # 4) LOWER QUANTILE OPTIMIZATION (NÚCLEO DO MÉTODO)
        # ============================================================
        # Passo-chave da técnica:
        #
        # 1) Ordenamos os objetivos do pior para o melhor
        # 2) Selecionamos o quantil inferior Q_α
        #
        # Exemplo com 4 objetivos:
        #   sorted = [-1.0, 0.0, 0.9, 1.0]
        #
        # α = 1/3 → índice ≈ 1
        # reward = 0.0
        #
        # Ou seja:
        #   - ignoramos o pior extremo
        #   - focamos na fronteira inferior aceitável

        sorted_objs = np.sort(objs)  # crescente: pior → melhor

        n = len(sorted_objs)
        assert n > 0, "Nenhum objetivo ativo para escalarização"

        # Índice do quantil inferior
        # (n - 1) garante que o índice fique no range válido
        q_idx = int(np.floor(alpha * (n - 1)))

        reward = float(sorted_objs[q_idx])


        # ============================================================
        # 5) NORMALIZAÇÃO FINAL
        # ============================================================
        # Garante que o reward respeite o contrato esperado
        # pelo algoritmo de RL (ex: PPO, DQN, A2C, etc.)
        #
        # Também evita explosões numéricas.

        reward = float(np.clip(reward, -1.0, 1.0))


        # ============================================================
        # 6) DEBUG E INTERPRETABILIDADE
        # ============================================================
        # Este log é pra ajudar no debug é útil para:
        #   - validar o comportamento da escalarização
        #   - entender quais objetivos estão segurando a política
        #
        # Ele mostra:
        #   - valores individuais
        #   - ordenação
        #   - valor do quantil escolhido

        if np.random.rand() < 0.005:
            dbg = ", ".join(f"{l}={o:.3f}" for l, o in zip(labels, objs))
            print(
                f"[LOWER-Q DBG] α={alpha:.2f} | "
                f"objs=[{dbg}] | "
                f"sorted={sorted_objs.round(3)} | "
                f"Qα={reward:.3f}"
            )

        return reward




# This is the default stop class, which terminates the episode when the agent reaches the target node or takes the SERVICE_CENTER action
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2