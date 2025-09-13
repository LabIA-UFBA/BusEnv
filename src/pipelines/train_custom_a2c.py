# train_custom_a2c.py
import os
import pickle
import time
import numpy as np
from gym.spaces import Dict as GymDict

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from models.custom_a2c import CustomA2CTrainer
from models.base_mlp import BaseMLP
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.multi_agent_env import MultiAgentEnv


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
        def load_pickle(fname):
            with open(os.path.join(obs_dir, fname), "rb") as f:
                return pickle.load(f)

        avg_travel_time_AB = load_pickle("avg_travel_time_AB.pkl")
        future_demand_at_B = load_pickle("future_demand_at_B.pkl")
        occupancy_rate = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

        # Parallel env
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=10000,
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

        # Agentes
        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        # Spaces
        self.observation_space = GymDict({
            "obs": self.env.observation_space(self.agents[0])
        })
        self.action_space = self.env.action_space(self.agents[0])

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


# ------------------------------
# Register env + model
# ------------------------------
register_env("sunt_bus", lambda cfg: RLlibSuntBus(cfg))
ModelCatalog.register_custom_model("BaseMLP", BaseMLP)


# ------------------------------
# RLlib Training Config
# ------------------------------
trainer_config = {
    "env": "sunt_bus",
    "framework": "torch",
    "num_workers": 0,
    "num_gpus": 0,
    "lr": 3e-4,
    "train_batch_size": 20 * 1000,
    "rollout_fragment_length": 1000,
    "use_critic": True,
    "use_gae": True,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "model": {
        "custom_model": "BaseMLP",
        "custom_model_config": {
            "num_agents": 5,
            "global_state_flag": False,
            "mask_flag": False,
            "model_arch_args": {
                "fc_layer": 2,
                "out_dim_fc_0": 64,
                "out_dim_fc_1": 64
            }
        },
    },
    "logger_config": {
        "type": "ray.tune.logger.TBXLogger",  # TensorBoard logger
        "logdir": "./exp_results",            # onde salvar os arquivos
    },
    "log_level": "INFO"
}


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    trainer = CustomA2CTrainer(config=trainer_config)
    for i in range(10):  # 10 iterations só para testar
        results = trainer.train()
        print(f"Iter {i}: reward={results['episode_reward_mean']}")
        if i % 5 == 0:
            chkpt = trainer.save()
            print(f"Checkpoint saved at {chkpt}")
