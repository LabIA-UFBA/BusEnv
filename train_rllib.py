import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import RunConfig  
from ray.tune import TuneConfig
from ray.tune.tuner import Tuner # Ray Tune API
from pettingzoo.utils import parallel_to_aec # Importando o PettingZoo para compatibilidade com Ray RLlib
from supersuit import pad_observations_v0, pad_action_space_v0 # Importando o Supersuit para compatibilidade com Ray RLlib
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv 
from ray.rllib.utils.pre_checks.env  import check_env




def env_creator(config):
    import pickle
    from multi_agent_sunt_env import parallel_env # Importando o ambiente personalizado

    with open('./SUNT/data/graph_designer/graph_gtfs_fev_2024.gpickle', 'rb') as f:
        G = pickle.load(f)

    # Carrega os arquivos de observação 
    with open("output_obs/avg_travel_time_AB.pkl", "rb") as f:
        avg_travel_time_AB = pickle.load(f)

    with open("output_obs/future_demand_at_B.pkl", "rb") as f:
        future_demand_at_B = pickle.load(f)

    with open("output_obs/occupancy_rate.pkl", "rb") as f:
        occupancy_rate = pickle.load(f)

    with open("output_obs/uptime_normalized.pkl", "rb") as f:
        uptime_normalized = pickle.load(f)

    env = parallel_env(
        network=G,
        actions_amount=3,
        max_steps=100,
        num_agents=2,
        avg_travel_time_AB=avg_travel_time_AB,
        future_demand_at_B=future_demand_at_B,
        occupancy_rate=occupancy_rate,
        uptime_normalized=uptime_normalized
    )

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)

    check_env(env) # Verificando se o ambiente está correto para uso com Ray RLlib

    return env


register_env("sunt_env", lambda config: env_creator(config)) # Registrando o ambiente personalizado no Ray RLlib

ray.init(ignore_reinit_error=True) # Inicializando o Ray

env = env_creator({})
agents = env.par_env.possible_agents
obs_space = env.observation_space[agents[0]]
act_space = env.action_space[agents[0]]


# Política compartilhada
policies = {
    "shared_policy": (None, obs_space, act_space, {})
}
policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

config = ( # Configuração do Ray RLlib
    PPOConfig()
    .environment(env="sunt_env")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(train_batch_size=4000, gamma=0.99) # Tamanho do lote de treinamento
    .resources(num_gpus=0)
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
)

# Treinamento com Tuner (novo Tune API)
tuner = Tuner( 
    "PPO", # Algoritmo PPO do Ray RLlib
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
