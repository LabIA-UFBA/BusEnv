import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import TBXLoggerCallback
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.tuner import Tuner # Ray Tune API
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import pickle


# --- Environment creator ---
def env_creator(config):
    from src.envs.sunt_env import parallel_env

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # === Load the graph ===
    with open(os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle"), "rb") as f:
        G = pickle.load(f)

    # === Load the observation files ===
    obs_dir = os.path.join(BASE_DIR, "training_observation")

    with open(os.path.join(obs_dir, "avg_travel_time_AB.pkl"), "rb") as f:
        avg_travel_time_AB = pickle.load(f)

    with open(os.path.join(obs_dir, "future_demand_at_B.pkl"), "rb") as f:
        future_demand_at_B = pickle.load(f)

    with open(os.path.join(obs_dir, "occupancy_rate.pkl"), "rb") as f:
        occupancy_rate = pickle.load(f)

    with open(os.path.join(obs_dir, "uptime_normalized.pkl"), "rb") as f:
        uptime_normalized = pickle.load(f)

    with open(os.path.join(obs_dir, "real_routes.pkl"), "rb") as f:
        real_routes = pickle.load(f)

    with open(os.path.join(obs_dir, "route_metadata.pkl"), "rb") as f:
        route_metadata = pickle.load(f)

    # === Create the environment ===
    env = parallel_env(
        network=G,
        actions_amount=3,
        max_steps=100,
        num_agents=3,
        avg_travel_time_AB=avg_travel_time_AB,
        future_demand_at_B=future_demand_at_B,
        occupancy_rate=occupancy_rate,
        uptime_normalized=uptime_normalized,
        real_routes=real_routes,
        route_metadata=route_metadata
    )

    # === Wrappers ===
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)

    return env


# Register the custom environment
register_env("sunt_env", lambda config: env_creator(config))

# --- Ray init ---
try:
    ray.init(ignore_reinit_error=True)
    print("✅ Ray initialized (GPU if available).")
except Exception as e:
    print(f"⚠️ Ray init failed ({e}). Falling back to CPU.")
    ray.init(ignore_reinit_error=True, num_gpus=0)
    print("✅ Ray initialized in CPU-only mode.")

# --- Env spaces ---
env = env_creator({})
agents = env.par_env.possible_agents
obs_space = env.observation_space[agents[0]]
act_space = env.action_space[agents[0]]

# Shared policy
policies = {
    "shared_policy": (None, obs_space, act_space, {})
}
policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

# --- PPO Config ---
config = (
    PPOConfig()
    .environment(env="sunt_env")
    .framework("torch")
    .rollouts(num_rollout_workers=1)      # ✅ FIX: use rollouts instead of env_runners
    .training(train_batch_size=4000, gamma=0.99)
    .resources(num_gpus=0)
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
)

# Training with Tuner, Hyperparameter Tuning
tuner = Tuner(  # The Tuner is the heart of experimentation with ray.tune
    "PPO", # PPO algorithm from Ray RLlib
    run_config=RunConfig( # Pass a RunConfig (from ray.train) to control execution, checkpoints, logs, etc
        stop={"training_iteration": 5},
        storage_path=os.path.abspath("./results"), # local_dir ="./results",
        name="ppo_sunt_experiment",
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=2
        ),
        verbose=1,
        callbacks=[TBXLoggerCallback()] # Using TensorBoard for logging
    ),
    tune_config=TuneConfig(), # Passing an empty TuneConfig (but can define hyperparameter search)
    param_space=config.to_dict() # Passing the param_space with the PPO parameters via .to_dict()
)

# tuner.fit()
results = tuner.fit() # Running the training process
# print("Best result:", results.get_best_result(metric="episode_reward_mean", mode="max"))