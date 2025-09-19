# train_custom_a2c.py
import os
import pickle
import time
from datetime import datetime
import numpy as np
from gym.spaces import Dict as GymDict

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

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
            max_steps=1000,
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


# ------------------------------
# Register environment
# ------------------------------
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus
env_tuple = marl.make_env(environment_name="sunt_bus", map_name="sunt_bus", force_coop=False)
env_instance = env_tuple[0]
register_env("sunt_bus", lambda cfg: env_instance)
print("✅ Environment successfully registered.")


# ------------------------------
# Register custom model
# ------------------------------
ModelCatalog.register_custom_model("BaseMLP", BaseMLP)


# ------------------------------
# Trainer config
# ------------------------------
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("exp_results", f"run_{run_id}")

trainer_config = {
    "env": "sunt_bus",
    "framework": "torch",
    "num_workers": 2,
    "num_gpus": 0,
    "lr": 3e-4,
    "batch_mode": "complete_episodes",
    "rollout_fragment_length": -1,
    "train_batch_size": 20,
    "use_critic": True,
    "use_gae": True,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "model": {
        "custom_model": "BaseMLP",
        "custom_model_config": {
            "num_agents": env_instance.num_agents,
            "global_state_flag": False,
            "mask_flag": False,
            "model_arch_args": {
                "fc_layer": 2,
                "out_dim_fc_0": 128,
                "out_dim_fc_1": 128
            }
        },
    },
    "logger_config": {
        "type": "ray.tune.logger.TBXLogger",
        "logdir": log_dir,
    },
    "log_level": "INFO",
}


# ------------------------------
# Training loop
# ------------------------------
if __name__ == "__main__":
    trainer = CustomA2CTrainer(config=trainer_config)
    checkpoint_interval = 50
    stop_episodes = 1000
    iteration = 0

    while True:
        results = trainer.train()
        iteration += 1

        print(f"\n=== Iteration {iteration} ===")
        print(f"Episodes total: {results.get('episodes_total', 'NA')}")
        print(f"Timesteps total: {results.get('timesteps_total', 'NA')}")
        print(f"Mean reward: {results.get('episode_reward_mean', 'NA')}")
        print(f"Min/Max reward: {results.get('episode_reward_min', 'NA')} / {results.get('episode_reward_max', 'NA')}")

        learner_info = results.get("info", {}).get("learner", {}).get("default_policy", {})
        learner_stats = learner_info.get("learner_stats", {})
        if learner_info:
            print(f"Policy loss: {learner_stats.get('policy_loss', 'NA')}")
            print(f"Value loss: {learner_stats.get('vf_loss', 'NA')}")
            print(f"Entropy: {learner_stats.get('policy_entropy', 'NA')}")

        # Save checkpoint every N iterations
        if iteration % checkpoint_interval == 0:
            ckpt_path = trainer.save(log_dir)
            print(f">>> Checkpoint saved at: {ckpt_path}")

        # Stop condition
        if results.get("episodes_total", 0) >= stop_episodes:
            print(">>> Stop condition reached!")
            break

    trainer.stop()
    print(">>> Training finished successfully!")
