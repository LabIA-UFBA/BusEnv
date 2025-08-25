import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import RunConfig  
from ray.tune import TuneConfig
from ray.tune.tuner import Tuner # Ray Tune API
from pettingzoo.utils import parallel_to_aec # Importing PettingZoo for Ray RLlib compatibility
from supersuit import pad_observations_v0, pad_action_space_v0 # Importing Supersuit for Ray RLlib compatibility
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv 
from ray.tune.logger import TBXLoggerCallback




# Creating and registering a multi-agent environment compatible with RLlib
def env_creator(config):
    import pickle
    from src.envs.sunt_env import parallel_env # Import the custom environment

    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # Get the base directory

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

    # === Load the real routes and their metadata ===
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
        route_metadata=route_metadata  # <-- Pass the real routes and metadata
    )

    # === Wrappers ===
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)

    return env


register_env("sunt_env", lambda config: env_creator(config))  # Register the custom environment in Ray RLlib

# --- Try GPU first, fall back to CPU if it fails ---
try:
    ray.init(ignore_reinit_error=True)  # Let Ray auto-detect GPUs
    print("✅ Ray initialized with GPU (if available).")
except Exception as e:
    print(f"⚠️ Ray GPU init failed ({e}). Falling back to CPU.")
    ray.init(ignore_reinit_error=True, num_gpus=0)
    print("✅ Ray initialized in CPU-only mode.")

ray.init(ignore_reinit_error=True) # Initializing Ray

env = env_creator({})
agents = env.par_env.possible_agents
obs_space = env.observation_space[agents[0]]
act_space = env.action_space[agents[0]]


# Shared policy
policies = {
    "shared_policy": (None, obs_space, act_space, {})
}
policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

config = ( # Ray RLlib configuration
    PPOConfig() # Setting up the PPO algorithm configuration
    .environment(env="sunt_env") # Defining the environment
    .framework("torch") # Using PyTorch as the framework
    .env_runners(num_env_runners=1) #.rollouts(num_rollout_workers=1)
    .training(train_batch_size=4000, gamma=0.99) # Training batch size
    .resources(num_gpus=0)
    .multi_agent( # Configuring multi-agents with a shared policy
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