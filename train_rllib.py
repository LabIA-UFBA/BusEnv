import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import RunConfig  
from ray.tune import TuneConfig
from ray.tune.tuner import Tuner
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from multi_agent_sunt_env import parallel_env
from ray.rllib.utils.pre_checks.env  import check_env




def env_creator(config):
    import pickle
    from multi_agent_sunt_env import parallel_env

    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
        G = pickle.load(f)

    env = parallel_env(
        network=G,
        actions_amount=9,
        max_steps=100,
        num_agents=2
    )
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)
    # env = parallel_to_aec(env)

    check_env(env)

    return env



register_env("sunt_env", lambda config: env_creator(config))

ray.init(ignore_reinit_error=True)

env = env_creator({})
agents = env.par_env.possible_agents
obs_space = env.observation_space[agents[0]]
act_space = env.action_space[agents[0]]


# Política compartilhada
policies = {
    "shared_policy": (None, obs_space, act_space, {})
}
policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

config = (
    PPOConfig()
    .environment(env="sunt_env")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(train_batch_size=4000, gamma=0.99)
    .resources(num_gpus=0)
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
)

# Treinamento com Tuner (novo Tune API)
tuner = Tuner(
    "PPO",
    run_config=RunConfig(
        stop={"training_iteration": 5},
        local_dir="./results",
        name="ppo_sunt_experiment",
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=1
        ),
        verbose=1
    ),
    tune_config=TuneConfig(),
    param_space=config.to_dict()
)

tuner.fit()
