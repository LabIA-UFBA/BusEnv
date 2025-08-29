import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import IMPALAConfig  # Import IMPALA algorithm configuration
from ray.train import RunConfig  
from ray.tune import TuneConfig
from ray.tune.tuner import Tuner
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv 
from ray.tune.logger import TBXLoggerCallback

# Suppress Ray v2 migration warnings
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

# --- Environment creation and registration ---
def env_creator(config):
    import pickle
    from src.envs.sunt_env import parallel_env  # Import the custom parallel environment

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Base directory of the project

    # === Load the network graph ===
    with open(os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle"), "rb") as f:
        G = pickle.load(f)

    # === Load precomputed observation data ===
    obs_dir = os.path.join(BASE_DIR, "training_observation")

    with open(os.path.join(obs_dir, "avg_travel_time_AB.pkl"), "rb") as f:
        avg_travel_time_AB = pickle.load(f)

    with open(os.path.join(obs_dir, "future_demand_at_B.pkl"), "rb") as f:
        future_demand_at_B = pickle.load(f)

    with open(os.path.join(obs_dir, "occupancy_rate.pkl"), "rb") as f:
        occupancy_rate = pickle.load(f)

    with open(os.path.join(obs_dir, "uptime_normalized.pkl"), "rb") as f:
        uptime_normalized = pickle.load(f)

    # === Load real routes and metadata ===
    with open(os.path.join(obs_dir, "real_routes.pkl"), "rb") as f:
        real_routes = pickle.load(f)

    with open(os.path.join(obs_dir, "route_metadata.pkl"), "rb") as f:
        route_metadata = pickle.load(f)

    # === Create the parallel PettingZoo environment ===
    env = parallel_env(
        network=G,
        actions_amount=3,
        max_steps=1000000,
        num_agents=3,
        avg_travel_time_AB=avg_travel_time_AB,
        future_demand_at_B=future_demand_at_B,
        occupancy_rate=occupancy_rate,
        uptime_normalized=uptime_normalized,
        real_routes=real_routes,
        route_metadata=route_metadata
    )

    # === Apply observation and action space padding wrappers ===
    env = pad_observations_v0(env)  # Pad observation spaces to have uniform shape
    env = pad_action_space_v0(env)  # Pad action spaces to have uniform size
    env = ParallelPettingZooEnv(env)  # Wrap environment for RLlib multi-agent support

    return env

# Register the custom environment with Ray RLlib
register_env("sunt_env", lambda config: env_creator(config))

# --- Initialize Ray ---
try:
    ray.init(ignore_reinit_error=True)  # Try GPU first, if available
    print("✅ Ray initialized with GPU (if available).")
except Exception as e:
    print(f"⚠️ Ray GPU init failed ({e}). Falling back to CPU.")
    ray.init(ignore_reinit_error=True, num_gpus=0)
    print("✅ Ray initialized in CPU-only mode.")

# --- Environment setup for multi-agent configuration ---
env = env_creator({})
agents = env.par_env.possible_agents  # Get the list of agent IDs
obs_space = env.observation_space[agents[0]]  # Observation space of a single agent
act_space = env.action_space[agents[0]]       # Action space of a single agent

# --- Select algorithm ---
ALGO = "PPO"  # Options: "PPO", "IMPALA"

# --- Define shared policy for all agents ---
policies = {
    "shared_policy": (None, obs_space, act_space, {})  # None -> default policy class
}
policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"  # Map all agents to shared policy

# --- Algorithm-specific configuration ---
if ALGO == "PPO":
    config = (
        PPOConfig()
        .environment(env="sunt_env")  # Set environment
        .framework("torch")          # Use PyTorch backend
        .env_runners(num_env_runners=1)  # Number of environment runners
        .training(train_batch_size=4000, gamma=0.99)  # Training batch size & discount factor
        .resources(num_gpus=0)       # GPU resources
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
    )
    algo_name = "PPO"
    exp_name = "ppo_sunt_experiment"

elif ALGO == "IMPALA":
    config = (
        IMPALAConfig()
        .environment(env="sunt_env")
        .framework("torch")
        .env_runners(num_env_runners=2)  # IMPALA benefits from more workers
        .resources(num_gpus=0)
        .training(
            gamma=0.99,
            lr=0.0005,           # Learning rate
            train_batch_size=512, # IMPALA uses smaller batches
            num_sgd_iter=1,       # Number of SGD iterations per batch
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
    )
    algo_name = "IMPALA"
    exp_name = "impala_sunt_experiment"

else:
    raise ValueError(f"Unsupported algorithm: {ALGO}")

# --- Setup Ray Tune Tuner ---
tuner = Tuner(
    algo_name,
    run_config=RunConfig(
        stop={"training_iteration": 10},  # Stop after 10 training iterations
        storage_path=os.path.abspath("./results"),  # Directory to store results
        name=exp_name,  # Experiment name
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_at_end=True,          # Save checkpoint at the end
            checkpoint_frequency=2           # Save every 2 iterations
        ),
        verbose=1,                           # Logging verbosity
        callbacks=[TBXLoggerCallback()]      # TensorBoard logging callback
    ),
    tune_config=TuneConfig(),
    param_space=config.to_dict()  # Convert RLlib config to param_space for Tuner
)

# --- Start training ---
results = tuner.fit()
print(f"Training completed with {ALGO}!")
