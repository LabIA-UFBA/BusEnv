import os
import pickle
import time
import pprint
import numpy as np
from gym.spaces import Dict as GymDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0


# ------------------------------
# Classe RLlibSuntBus
# ------------------------------
class RLlibSuntBus(MultiAgentEnv):
    def __init__(self, env_config):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Carregar grafo
        graph_path = os.path.join(BASE_DIR, "viz/graph_gtfs_fev_2024.gpickle")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {graph_path}")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        # Carregar observações
        obs_dir = os.path.join(BASE_DIR, "training_observation")

        def load_pickle(filename):
            path = os.path.join(obs_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        avg_travel_time_AB = load_pickle("avg_travel_time_AB.pkl")
        future_demand_at_B = load_pickle("future_demand_at_B.pkl")
        occupancy_rate = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

        # Criar env paralelizado
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=1000000,
            num_agents=1,
            avg_travel_time_AB=avg_travel_time_AB,
            future_demand_at_B=future_demand_at_B,
            occupancy_rate=occupancy_rate,
            uptime_normalized=uptime_normalized,
            real_routes=real_routes,
            route_metadata=route_metadata,
        )

        # Wrappers do supersuit
        self.env = pad_observations_v0(self.env)
        self.env = pad_action_space_v0(self.env)

        # Agentes
        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        # Espaços
        self.observation_space = GymDict({
            "obs": self.env.observation_space(self.agents[0])
        })
        self.action_space = self.env.action_space(self.agents[0])
        self.action_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }

    def reset(self):
        """Reseta o env paralelo e retorna obs no formato RLlib."""
        original_obs = self.env.reset()  # dict: {agent_id: obs}
        self.agents = list(original_obs.keys())
        obs = {agent: {"obs": np.array(o)} for agent, o in original_obs.items()}
        return obs

    def step(self, action_dict):
        """Executa uma etapa no env paralelo com action_dict do RLlib."""
        o, r, d, info = self.env.step(action_dict)

        obs = {agent: {"obs": np.array(o[agent])} for agent in o.keys()}
        rewards = {agent: r.get(agent, 0.0) for agent in r.keys()}
        dones = {"__all__": all(d.values())}
        infos = {agent: info.get(agent, {}) for agent in info.keys()}

        self.agents = [a for a in self.agents if not d.get(a, False)]
        return obs, rewards, dones, infos

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
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
        return env_info
    


# ------------------------------
# Registrar ambiente no MARLlib
# ------------------------------
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus

# ------------------------------
# Dicionário de Configuração do Ambiente
# ------------------------------
env_config = {
    "map_name": "sunt_bus",
}

# 1. CRIAR A TUPLA COM A INSTÂNCIA DO AMBIENTE
env_tuple = marl.make_env(environment_name="sunt_bus", map_name="sunt_bus", force_coop=False)

# ------------------------------
# Selecionar algoritmo e configurar modelo
# ------------------------------
algo = marl.algos.ia2c(hyperparam_source="common")

model_config = {
    "core_arch": "mlp",
    "encode_layer": "128-128",
}

# 2. CONSTRUIR O MODELO (já estava correto, usando env_tuple)
model = marl.build_model(env_tuple, algo, model_config)

# ------------------------------
# Configurações de execução
# ------------------------------
run_config = {
    "local_mode": False,
    "stop": {"episodes_total": 1000},
    "checkpoint_freq": 50,
    "num_gpus": 0,
    "num_workers": 2,
    "share_policy": "individual",
}

custom_config = {
    "lr": 0.0003,
    "batch_episode": 20,
}

final_config = run_config.copy()
final_config.update(custom_config)
stop_conditions = final_config.pop("stop")

# ------------------------------
# Treinar (Com a chamada final e correta)
# ------------------------------
# 3. PASSE A MESMA `env_tuple` PARA A FUNÇÃO FIT
algo.fit(
    env=env_tuple, # <-- MUDANÇA PRINCIPAL AQUI
    model=model,
    stop=stop_conditions,
    **final_config
)