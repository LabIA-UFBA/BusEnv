import os
import pickle
import time
import numpy as np
from gym.spaces import Dict as GymDict

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

from ray.rllib.models import ModelCatalog
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Import custom BaseMLP
from models.base_mlp import BaseMLPCustom

# ------------------------------
# Custom Environment Wrapper
# ------------------------------
class RLlibSuntBus(MultiAgentEnv):
    def __init__(self, env_config):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Load graph
        graph_path = os.path.join(BASE_DIR, "viz/graph_gtfs_fev_2024.gpickle")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        # Load observations
        obs_dir = os.path.join(BASE_DIR, "training_observation")

        def load_pickle(filename):
            path = os.path.join(obs_dir, filename)
            with open(path, "rb") as f:
                return pickle.load(f)

        avg_travel_time_AB = load_pickle("avg_travel_time_AB.pkl")
        future_demand_at_B = load_pickle("future_demand_at_B.pkl")
        occupancy_rate = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

        # Parallel environment
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=1000000,
            num_agents=5,
            avg_travel_time_AB=avg_travel_time_AB,
            future_demand_at_B=future_demand_at_B,
            occupancy_rate=occupancy_rate,
            uptime_normalized=uptime_normalized,
            real_routes=real_routes,
            route_metadata=route_metadata,
        )

        # Supersuit wrappers
        self.env = pad_observations_v0(self.env)
        self.env = pad_action_space_v0(self.env)

        # Agents
        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        # Spaces
        self.observation_space = GymDict({"obs": self.env.observation_space(self.agents[0])})
        self.action_space = self.env.action_space(self.agents[0])
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

    def reset(self):
        obs = self.env.reset()
        self.agents = list(obs.keys())
        return {a: {"obs": np.array(o)} for a, o in obs.items()}

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        obs = {a: {"obs": np.array(o[a])} for a in o.keys()}
        rewards = {a: r.get(a, 0.0) for a in r.keys()}
        dones = {"__all__": all(d.values())}
        infos = {a: info.get(a, {}) for a in info.keys()}
        self.agents = [a for a in self.agents if not d.get(a, False)]
        return obs, rewards, dones, infos

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def close(self):
        self.env.close()

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000000,
            "agent_id": self.agents,
            "share_observation_space": self.observation_space,
            "policy_mapping_info": {
                "sunt_bus": {
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                    "policy_map": {
                        agent_id: f"policy_{i}" for i, agent_id in enumerate(self.agents)
                    }
                }
            }
        }

# ------------------------------
# Register environment
# ------------------------------
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus
env_tuple = marl.make_env(environment_name="sunt_bus", map_name="sunt_bus", force_coop=False)

# ------------------------------
# Register custom model
# ------------------------------
ModelCatalog.register_custom_model("custom_ac", BaseMLPCustom)

# ------------------------------
# Select algorithm and configure model
# ------------------------------
algo = marl.algos.ia2c(hyperparam_source="common")

model_config = {
    "custom_model": "custom_ac",
    "model_arch_args": {
        "core_arch": "mlp",
        "fc_layer": 2,
        "out_dim_fc_0": 128,
        "out_dim_fc_1": 128
    },
    "custom_model_config": {
        "num_agents": env_tuple[0].get_env_info()["num_agents"],
        "mask_flag": False,
        "global_state_flag": False,
    },
    "fcnet_activation": "relu",
}


# Build the model tuple para MARLlib
model = (BaseMLPCustom, model_config)

# ------------------------------
# Execution configurations
# ------------------------------
run_config = {
    "local_mode": False,
    "stop": {"timesteps_total": 1000000},
    "checkpoint_freq": 200,
    "num_gpus": 0,
    "num_workers": 2,
    "share_policy": "individual",
}

custom_config = {
    "lr": 0.0003,          # lr_1
    "batch_episode": 20,    # batch_size
    "gamma": 0.99,          # discount
    "vf_loss_coeff": 1.0,   # weight do critic
    "entropy_coeff": 0.01,  # exploração
    "use_gae": True,
    "lambda": 1.0
}

final_config = run_config.copy()
final_config.update(custom_config)
stop_conditions = final_config.pop("stop")

# ------------------------------
# Train
# ------------------------------
algo.fit(
    env=env_tuple,
    model=model,
    stop=stop_conditions,
    **final_config
)
