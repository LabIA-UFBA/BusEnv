import os
import pickle
import time
import argparse
import numpy as np
from gym.spaces import Dict as GymDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from contextlib import nullcontext
from codecarbon import EmissionsTracker  # needs pip install codecarbon

# ------------------------------
# RLlibSuntBus Environment
# ------------------------------
class RLlibSuntBus(MultiAgentEnv):
    # ... (same as before, no changes)
    pass

# Register environment
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus

# ------------------------------
# Algorithm Configs (commented placeholders)
# ------------------------------

ALGO_CONFIGS = {
    # "iql": {"lr": 0.0003, "batch_episode": 20},
    # "ipg": {"lr": 0.0003, "batch_episode": 20},
    # "ia2c": {"lr": 0.0003, "batch_episode": 20},
    # "iddpg": {"lr": 0.001, "batch_episode": 25},
    # "itrpo": {"lr": 0.0003, "batch_episode": 20},
    # "ippo": {"lr": 0.0005, "batch_episode": 30},
    # "maa2c": {"lr": 0.0003, "batch_episode": 20},
    # "coma": {"lr": 0.0001, "batch_episode": 10},
    # "maddpg": {"lr": 0.001, "batch_episode": 25},
    # "matrpo": {"lr": 0.0003, "batch_episode": 20},
    # "mappo": {"lr": 0.0003, "batch_episode": 50},
    # "hatrpo": {"lr": 0.0003, "batch_episode": 20},
    # "happo": {"lr": 0.0003, "batch_episode": 20},
    # "vdn": {"lr": 0.0003, "batch_episode": 20},
    # "qmix": {"lr": 0.0003, "batch_episode": 20},
    # "facmac": {"lr": 0.001, "batch_episode": 25},
    # "vda2c": {"lr": 0.0003, "batch_episode": 20},
    # "vdppo": {"lr": 0.0003, "batch_episode": 20},
}
DEFAULT_CONFIG = {"lr": 0.0003, "batch_episode": 20}

# ------------------------------
# Main entrypoint
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MARLlib algorithms on SUNT Bus env")
    parser.add_argument("--algo", type=str, required=True,
                        choices=["iql","ipg","ia2c","iddpg","itrpo","ippo",
                                 "maa2c","coma","maddpg","matrpo","mappo",
                                 "hatrpo","happo","vdn","qmix","facmac",
                                 "vda2c","vdppo"],
                        help="Which MARLlib algorithm to train.")
    parser.add_argument("--cc-run-id", default=None,
                        help="Unique run identifier for CodeCarbon logging.")
    parser.add_argument("--cc-output-dir", default="./codecarbon",
                        help="Directory to store CodeCarbon CSV/JSON.")
    parser.add_argument("--no-cc", action="store_true",
                        help="Disable CodeCarbon tracking.")
    args = parser.parse_args()

    # ---------------- Env ----------------
    env_tuple = marl.make_env(environment_name="sunt_bus", map_name="sunt_bus", force_coop=False)

    # ---------------- Algo ----------------
    algo_ctor = getattr(marl.algos, args.algo)
    algo = algo_ctor(hyperparam_source="common")

    model_config = {"core_arch": "mlp", "encode_layer": "128-128"}
    model = marl.build_model(env_tuple, algo, model_config)

    run_config = {
        "local_mode": False,
        "stop": {"timesteps_total": 400000},
        "checkpoint_freq": 200,
        "num_gpus": 1,
        "num_workers": 2,
        "share_policy": "individual",
    }

    custom_config = ALGO_CONFIGS.get(args.algo, DEFAULT_CONFIG)
    final_config = {**run_config, **custom_config}
    stop_conditions = final_config.pop("stop")

    # ---------------- CodeCarbon ----------------
    tracker_ctx = nullcontext()
    tracker = None
    if not args.no_cc:
        tracker = EmissionsTracker(
            project_name=f"marllib:{args.algo}:{args.cc_run_id or 'default'}",
            output_dir=args.cc_output_dir,
            save_to_file=True,
            save_to_api=False,
        )
        tracker.start()
        tracker_ctx = tracker

    # ---------------- Train ----------------
    try:
        algo.fit(env=env_tuple, model=model, stop=stop_conditions, **final_config)
    finally:
        if tracker:
            emissions = tracker.stop()
            print(f"[codecarbon] {args.algo} {args.cc_run_id} emissions: {emissions:.6f} kg CO₂eq")


if __name__ == "__main__":
    main()

