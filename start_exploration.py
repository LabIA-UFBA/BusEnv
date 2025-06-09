import os
import networkx as nx
import pickle
from GraphExplorationEnv import GraphExplorationEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.env_util import make_vec_env
import copy


if __name__ == "__main__":
    # Carrega o grafo
    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
        G = pickle.load(f)

    # Define o número máximo de ações com base no nó mais conectado
    ACTIONS = max([len(list(G.neighbors(n))) for n in G.nodes])

    # Cria o ambiente
    raw_env = GraphExplorationEnv(network=G, actions_amout=ACTIONS)

    # Verifica se o ambiente está de acordo com o padrão Gym
    check_env(raw_env, warn=True)

    # Diretórios de log
    log_dir = "./monitor_logs/"
    tensorboard_log_dir = "./tensorboard_graph_exploration/"
    eval_log_dir = "./eval_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # Envelopa o ambiente com Monitor para logging de recompensas
    monitored_env = Monitor(raw_env, log_dir)

    env = TimeLimit(monitored_env, max_episode_steps=200)

    # Wrap com DummyVecEnv
    env = make_vec_env(lambda: monitored_env, n_envs=1) # Use n_envs=1 para simplificar o exemplo


    # Inicializa o modelo DQN
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.9,
        tensorboard_log=tensorboard_log_dir
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=eval_log_dir) 

    # Treinamento com log para TensorBoard
    model.learn(
        total_timesteps=5000,
        tb_log_name="DQN_GraphRun",
        progress_bar=True,
        callback=callback
    )

    # Salva o modelo treinado
    model.save("dqn_graph_exploration")

    # Salva o ambiente (opcional)
    with open("env.pkl", "wb") as f:
        pickle.dump(raw_env, f)
