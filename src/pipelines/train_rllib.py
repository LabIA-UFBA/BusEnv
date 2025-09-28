# src/pipelines/train_rllib.py

import os
import argparse
import pickle

# (opcional) silenciar alguns avisos de migração
os.environ.setdefault("RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS", "0")

import ray
from ray import tune
from ray.tune.registry import register_env

# === RLlib (API antiga do Ray 1.x) ===
from ray.rllib.agents.ppo import PPOTrainer as PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer as ImpalaTrainer

# === PettingZoo / SuperSuit (sem parallel_to_aec) ===
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


# --------------------------------------------------------------------------------------
# Ambiente
# --------------------------------------------------------------------------------------
def env_creator(config):
    """Cria o ambiente paralelo do PettingZoo e aplica wrappers + wrapper do RLlib."""
    from src.envs.sunt_env import parallel_env  # import local para evitar custo no --help

    # Base do projeto (um nível acima deste arquivo)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # === Carrega o grafo ===
    with open(os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle"), "rb") as f:
        G = pickle.load(f)

    # === Observações pré-computadas ===
    obs_dir = os.path.join(BASE_DIR, "training_observation")

    with open(os.path.join(obs_dir, "avg_travel_time_AB.pkl"), "rb") as f:
        avg_travel_time_AB = pickle.load(f)
    with open(os.path.join(obs_dir, "future_demand_at_B.pkl"), "rb") as f:
        future_demand_at_B = pickle.load(f)
    with open(os.path.join(obs_dir, "occupancy_rate.pkl"), "rb") as f:
        occupancy_rate = pickle.load(f)
    with open(os.path.join(obs_dir, "uptime_normalized.pkl"), "rb") as f:
        uptime_normalized = pickle.load(f)

    # === Rotas reais e metadados ===
    with open(os.path.join(obs_dir, "real_routes.pkl"), "rb") as f:
        real_routes = pickle.load(f)
    with open(os.path.join(obs_dir, "route_metadata.pkl"), "rb") as f:
        route_metadata = pickle.load(f)

    # === Cria env paralelo PettingZoo ===
    env = parallel_env(
        network=G,
        actions_amount=config.get("actions_amount", 3),
        max_steps=config.get("max_steps", 1_000_000),
        num_agents=config.get("num_agents", 5),
        avg_travel_time_AB=avg_travel_time_AB,
        future_demand_at_B=future_demand_at_B,
        occupancy_rate=occupancy_rate,
        uptime_normalized=uptime_normalized,
        real_routes=real_routes,
        route_metadata=route_metadata,
    )

    # === Wrappers ===
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)  # wrapper do RLlib p/ multiagente

    return env


# registra o ambiente no RLlib
register_env("sunt_env", lambda cfg: env_creator(cfg or {}))


# --------------------------------------------------------------------------------------
# Args (para o "--help" funcionar no seu CLI)
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Treino RLlib (Ray 1.x) com PettingZoo.")
    p.add_argument("--algo", choices=["PPO", "IMPALA"], default="PPO", help="Algoritmo a usar")
    p.add_argument("--stop-iters", type=int, default=5, help="Parar após N iterações de treino")
    p.add_argument("--num-workers", type=int, default=1, help="Workers de rollout (num_workers)")
    p.add_argument("--num-gpus", type=int, default=0, help="GPUs por trial (num_gpus)")
    p.add_argument("--local-dir", default="./results", help="Diretório de resultados do Tune")
    p.add_argument("--exp-name", default=None, help="Nome do experimento (opcional)")
    p.add_argument("--train-batch-size", type=int, default=4000, help="train_batch_size (PPO)")
    p.add_argument("--gamma", type=float, default=0.99, help="fator de desconto")
    # parâmetros do ambiente
    p.add_argument("--actions-amount", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--num-agents", type=int, default=5)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()

    # Inicializa Ray
    try:
        ray.init(ignore_reinit_error=True)
        print("✅ Ray inicializado.")
    except Exception as e:
        print(f"⚠️ Ray init falhou ({e}). Tentando CPU-only.")
        ray.init(ignore_reinit_error=True, num_gpus=0)

    # Instancia um env para descobrir spaces (ParallelPettingZooEnv fornece spaces comuns)
    tmp_env = env_creator({
        "actions_amount": args.actions_amount,
        "max_steps": args.max_steps,
        "num_agents": args.num_agents,
    })
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    # (opcional) fechar o env temporário se houver método close
    if hasattr(tmp_env, "close"):
        try:
            tmp_env.close()
        except Exception:
            pass

    # Política compartilhada
    policies = {
        "shared_policy": (None, obs_space, act_space, {})
    }
    policy_mapping_fn = lambda agent_id, *_, **__: "shared_policy"

    # Config comum
    base_config = {
        "env": "sunt_env",
        "framework": "torch",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "gamma": args.gamma,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        # repassar configurações do env
        "env_config": {
            "actions_amount": args.actions_amount,
            "max_steps": args.max_steps,
            "num_agents": args.num_agents,
        },
    }

    # Algoritmo específico
    if args.algo == "PPO":
        trainer_cls = PPOTrainer
        algo_name = "PPO"
        exp_name = args.exp_name or "ppo_sunt_experiment"
        algo_config = {
            "train_batch_size": args.train_batch_size,
        }
    else:
        trainer_cls = ImpalaTrainer
        algo_name = "IMPALA"
        exp_name = args.exp_name or "impala_sunt_experiment"
        algo_config = {
            "lr": 5e-4,
            "train_batch_size": 512,
        }

    config = {**base_config, **algo_config}

    # Logger compatível (callbacks novos vs. loggers antigos)
    logger_kwargs = {}
    try:
        # Ray com Callback
        from ray.tune.logger import TBXLoggerCallback
        logger_kwargs["callbacks"] = [TBXLoggerCallback()]
    except Exception:
        # Ray 1.x clássico
        try:
            from ray.tune.logger import TBXLogger
            logger_kwargs["loggers"] = [TBXLogger]
        except Exception:
            # sem TensorBoardX disponível – segue sem logger extra
            pass

    # Executa treino
    results = tune.run(
        run_or_experiment=trainer_cls,
        name=exp_name,
        stop={"training_iteration": args.stop_iters},
        config=config,
        local_dir=os.path.abspath(args.local_dir),
        checkpoint_at_end=True,
        checkpoint_freq=2,
        verbose=3,
        **logger_kwargs,
    )

    print(f"✅ Training completed with {algo_name}!")
    return results


if __name__ == "__main__":
    main()
