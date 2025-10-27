
import os
import pickle
import time
import argparse
import warnings
import numpy as np
from gym.spaces import Dict as GymDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from contextlib import nullcontext

# Quiet PettingZoo deprecation chatter
warnings.filterwarnings("ignore", message="The observation_spaces dictionary is deprecated")
warnings.filterwarnings("ignore", message="The action_spaces dictionary is deprecated")

# ---------- Optional CodeCarbon wrapper ----------
def _make_tracker(enabled: bool, *, project_name: str, output_dir: str = None):
    """
    Create a CodeCarbon tracker if enabled; otherwise return a no-op ctx.
    Ensures output_dir exists when provided.
    """
    if not enabled:
        return nullcontext(), None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        from codecarbon import EmissionsTracker
    except Exception:
        print("[codecarbon] CodeCarbon is not installed. pip install codecarbon", flush=True)
        return nullcontext(), None

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        save_to_file=True,
        save_to_api=False,
    )

    class _TrackerCtx:
        def __enter__(self):
            tracker.start()
            return tracker
        def __exit__(self, exc_type, exc, tb):
            try:
                emissions = tracker.stop()
                if emissions is not None:
                    print(f"[codecarbon] {project_name} emissions: {emissions:.6f} kg CO₂eq")
            except Exception as e:
                print(f"[codecarbon] Failed to stop tracker: {e}")

    return _TrackerCtx(), tracker

# ------------------------------
# RLlibSuntBus Environment
# ------------------------------
class RLlibSuntBus(MultiAgentEnv):
    def __init__(self, env_config):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Load graph
        graph_path = os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"File not found: {graph_path}")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        # Load observations
        obs_dir = os.path.join(BASE_DIR, "training_observation")

        def load_pickle(filename):
            path = os.path.join(obs_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        avg_travel_time_AB = load_pickle("avg_travel_time_AB.pkl")
        future_demand_at_B = load_pickle("future_demand_at_B.pkl")
        occupancy_rate = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

        # Create parallel env
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=1000,
            num_agents=2,
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

        # Agents and spaces
        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        # Base space após wrappers (assume Box compatível entre agentes)
        self._base_obs_space = self.env.observation_space(self.agents[0])
        # Shape/dtype "alvo" para todas as observações
        self._obs_shape = tuple(getattr(self._base_obs_space, "shape", ()))
        if len(self._obs_shape) == 0:
            # fallback: trata como escalar
            self._obs_shape = (1,)
        self._obs_size = int(np.prod(self._obs_shape))
        self._obs_dtype = np.float32

        # Exponha um Dict({"obs": space}) para compatibilidade com o que você já tinha
        self.observation_space = GymDict({"obs": self._base_obs_space})
        self.action_space = self.env.action_space(self.agents[0])
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

        # Termina todos juntos quando qualquer um terminar (lockstep)
        self._team_done = True

    # ---------- helpers ----------
    def _fix_obs(self, o):
        """Força dtype=float32 e shape fixo. Trunca ou 'padd' com zeros se necessário."""
        x = np.asarray(o, dtype=self._obs_dtype)
        if x.shape != self._obs_shape:
            flat = x.ravel()
            if flat.size < self._obs_size:
                pad = np.zeros(self._obs_size - flat.size, dtype=self._obs_dtype)
                flat = np.concatenate([flat, pad], axis=0)
            elif flat.size > self._obs_size:
                flat = flat[:self._obs_size]
            x = flat.reshape(self._obs_shape)
        return x

    def _wrap_obs_dict(self, obs_dict):
        """Converte dict simples em {agent: {'obs': array(...)}} com shape/dtype fixos."""
        wrapped = {}
        for agent in self.agents:
            raw = obs_dict.get(agent, np.zeros(self._obs_shape, dtype=self._obs_dtype))
            wrapped[agent] = {"obs": self._fix_obs(raw)}
        return wrapped

    def _default_action(self, agent):
        """Ação 'no-op' robusta para Discrete/MultiDiscrete."""
        sp = self.action_spaces[agent]
        # MultiDiscrete tem atributo nvec
        if hasattr(sp, "nvec"):
            return np.zeros_like(sp.nvec, dtype=np.int64)
        # Discrete → 0
        if hasattr(sp, "n"):
            return 0
        # Fallback: tenta 0
        return 0

    # ---------- MultiAgentEnv API ----------
    def reset(self):
        original_obs = self.env.reset()
        # Garante conjunto completo e ordenado de agentes durante o episódio
        self.agents = self.env.possible_agents.copy()

        # Alguns envs não retornam todos os agentes no reset → preencha
        for a in self.agents:
            original_obs.setdefault(a, np.zeros(self._obs_shape, dtype=self._obs_dtype))

        return self._wrap_obs_dict(original_obs)

    def step(self, action_dict):
        # Preenche ações ausentes para manter lockstep
        for a in self.agents:
            if a not in action_dict:
                action_dict[a] = self._default_action(a)

        o, r, d, info = self.env.step(action_dict)

        # Se qualquer agente terminou e lockstep ativo → todos terminam
        if self._team_done and any(d.get(a, False) for a in self.agents):
            for a in self.agents:
                d[a] = True

        # Garanta chaves para todos os agentes
        for a in self.agents:
            if a not in o:
                o[a] = np.zeros(self._obs_shape, dtype=self._obs_dtype)
            if a not in r:
                r[a] = 0.0
            if a not in d:
                d[a] = False
            if a not in info:
                info[a] = {}

        obs = self._wrap_obs_dict(o)
        rewards = {a: float(r[a]) for a in self.agents}
        dones = {"__all__": all(d.get(a, False) for a in self.agents)}
        infos = {a: info[a] for a in self.agents}

        # Mantém a lista de agentes estável até o próximo reset (não remova no meio do episódio)
        # if dones["__all__"]:  # nada a fazer aqui; reset tratará na próxima chamada

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
            "episode_limit": 1_000_000,
            "agent_id": self.agents,
            "share_observation_space": self.observation_space,
            "policy_mapping_info": {
                "sunt_bus": {
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                    "policy_map": {agent_id: f"policy_{i}" for i, agent_id in enumerate(self.agents)},
                }
            },
        }


# Register environment
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus

# ------------------------------
# Algorithm Configs (commented placeholders)
# ------------------------------
ALGO_CONFIGS = {
    # "iql": {"lr": 0.0003, "batch_episode": 20},
    # "ipg": {"lr": 0.0003, "batch_episode": 20},
    # "ia2c": {"lr": 0.0003, "batch_episode": 20},
    # "iddpg": {"lr": 0.0010, "batch_episode": 25},
    # "itrpo": {"lr": 0.0003, "batch_episode": 20},
    # "ippo": {"lr": 0.0005, "batch_episode": 30},
    # "maa2c": {"lr": 0.0003, "batch_episode": 20},
    # "coma": {"lr": 0.0001, "batch_episode": 10},
    # "maddpg": {"lr": 0.0010, "batch_episode": 25},
    # "matrpo": {"lr": 0.0003, "batch_episode": 20},
    # "mappo": {"lr": 0.0003, "batch_episode": 50},
    # "hatrpo": {"lr": 0.0003, "batch_episode": 20},
    # "happo": {"lr": 0.0003, "batch_episode": 20},
    # "vdn": {"lr": 0.0003, "batch_episode": 20},
    # "qmix": {"lr": 0.0003, "batch_episode": 20},
    # "facmac": {"lr": 0.0010, "batch_episode": 25},
    # "vda2c": {"lr": 0.0003, "batch_episode": 20},
    # "vdppo": {"lr": 0.0003, "batch_episode": 20},
    
}

DEFAULT_CONFIG = {"lr": 0.0003, "batch_episode": 20, "sgd_minibatch_size": 128}

# ------------------------------
# Main entrypoint
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MARLlib algorithms on SUNT Bus env")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=[
            "iql","ipg","ia2c","iddpg","itrpo","ippo",
            "maa2c","coma","maddpg","matrpo","mappo",
            "hatrpo","happo","vdn","qmix","facmac",
            "vda2c","vdppo"
        ],
        help="Which MARLlib algorithm to train.",
    )
    # CodeCarbon args
    parser.add_argument("--cc-run-id", default=None, help="Unique run identifier for CodeCarbon logging.")
    parser.add_argument("--cc-output-dir", default="./codecarbon", help="Directory to store CodeCarbon CSV/JSON.")
    parser.add_argument("--no-cc", action="store_true", help="Disable CodeCarbon tracking.")

    args = parser.parse_args()

    # Ensure CC dir exists (prevents OSError)
    outdir = args.cc_output_dir if not args.no_cc else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Environment
    env_tuple = marl.make_env(environment_name="sunt_bus", map_name="sunt_bus", force_coop=False)

    # Algorithm
    algo_ctor = getattr(marl.algos, args.algo)
    algo = algo_ctor(hyperparam_source="common")

    model_config = {"core_arch": "mlp", "encode_layer": "128-128"}
    model = marl.build_model(env_tuple, algo, model_config)

    # Base run config
    run_config = {
        "local_mode": False,
        "stop": {"timesteps_total": 10000},  # adjust as needed
        "checkpoint_freq": 200,
        "num_gpus": 0,         # adjust as needed
        "num_workers": 0,     # adjust as needed
        "share_policy": "individual",
    }




    custom_config = ALGO_CONFIGS.get(args.algo, DEFAULT_CONFIG)
    final_config = {**run_config, **custom_config}
    stop_conditions = final_config.pop("stop")

    # CodeCarbon tracker
    tracker_ctx, _ = _make_tracker(
        enabled=not args.no_cc,
        project_name=f"marllib:{args.algo}:{args.cc_run_id or 'default'}",
        output_dir=outdir,
    )

    # Train
    with tracker_ctx:
        algo.fit(env=env_tuple, model=model, stop=stop_conditions, **final_config)

if __name__ == "__main__":
    main()
