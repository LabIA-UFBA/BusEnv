#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_eval_replay.py

Runs one simulated day of the SUNT bus env with record_replay=True and saves
the resulting per-step replay log, for the game-like map visualization
(src/viz/replay_export.py + src/viz/render_replay.py).

Two modes:
  --mode random      Uses random actions (no checkpoint needed) — gives an
                      immediate "before training" / untrained-baseline replay.
  --mode checkpoint  Restores a trained policy from a Ray RLlib checkpoint
                      directory (as produced by src/pipelines/train_rllib.py,
                      which uses a single shared_policy for every agent) and
                      uses it to pick actions — gives an "after training" replay.

For checkpoints produced by the MARLlib scripts (train_marllib.py /
train_marllib_a2c.py / train_custom_a2c.py) instead of train_rllib.py: MARLlib
has ~18 different algorithm classes (ippo, mappo, coma, qmix, ...), each with
its own Trainer subclass and restore quirks, so there isn't one single
"--mode checkpoint" call that fits all of them. Swap the `_load_rllib_policy`
call below for your algorithm's `marl.algos.<algo>(...)` + the
`{"model_path": ..., "params_path": ...}` restore dict — see
evaluate_marllib.py's ALGO_CONFIGS / RLlibSuntBus for the exact env wiring to
reuse; the replay-recording env setup and step loop below stay the same.

Uso:
    python src/pipelines/generate_eval_replay.py --mode random --out replays/before.json
    python src/pipelines/generate_eval_replay.py --mode checkpoint \
        --checkpoint /mnt/ssd1/ray_results/.../checkpoint_000100/checkpoint-100 \
        --out replays/after.json
"""

import argparse
import os
import pickle
import shutil
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))

from envs.sunt_env import parallel_env  # noqa: E402


def build_env(replay_output_dir, num_agents=25, agents_per_route=5, use_rain=False, metrics_file_objectives=None):
    obs_dir = os.path.join(BASE_DIR, "training_observation")

    def load_pickle(name):
        with open(os.path.join(obs_dir, name), "rb") as f:
            return pickle.load(f)

    with open(os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle"), "rb") as f:
        G = pickle.load(f)

    # Default metrics path is a per-user hardcoded path in sunt_env.py's own
    # constructor default; override it with something under our own output dir
    # so this script doesn't depend on another user's directory being writable.
    metrics_file_objectives = metrics_file_objectives or os.path.join(replay_output_dir, "episode_metrics.csv")

    env = parallel_env(
        network=G,
        actions_amount=3,
        agents_per_route=agents_per_route,
        use_only_mean_data=0,
        max_steps=10000,
        num_agents=num_agents,
        use_rain=use_rain,
        metrics_file_objectives=metrics_file_objectives,
        avg_travel_time_AB=load_pickle("avg_travel_time_AB.pkl"),
        future_demand_at_B=load_pickle("future_demand_at_B.pkl"),
        occupancy_rate=load_pickle("occupancy_rate.pkl"),
        uptime_normalized=load_pickle("uptime_normalized.pkl"),
        real_routes=load_pickle("real_routes.pkl"),
        route_metadata=load_pickle("route_metadata.pkl"),
        passenger_flow_stats=load_pickle("stop_passenger_flow.pkl"),
        occupancy_source="real",
        reward_raining_type="normal",
        record_replay=True,
        replay_output_dir=replay_output_dir,
    )
    return env


def _load_rllib_policy(checkpoint_path):
    """
    Restores a raw-RLlib PPOTrainer checkpoint (train_rllib.py's format: one
    literal shared_policy for every agent) and returns a callable
    obs -> action using its old-RLlib-1.x compute_action API.
    """
    from ray.rllib.agents.ppo import PPOTrainer

    trainer = PPOTrainer(config={"num_workers": 0, "framework": "torch"})
    trainer.restore(checkpoint_path)

    def act(obs):
        return trainer.compute_action(obs, policy_id="shared_policy")

    return act


def run_episode(env, act_fn=None, max_steps=20000):
    obs = env.reset()
    steps = 0
    while steps < max_steps:
        actions = {}
        for agent, o in obs.items():
            if act_fn is None:
                actions[agent] = env.action_space(agent).sample()
            else:
                actions[agent] = act_fn(o)
        obs, rewards, dones, infos = env.step(actions)
        steps += 1
        if dones.get("__all__"):
            break
    return steps


def main():
    parser = argparse.ArgumentParser(description="Generate a replay log for the game-like map viewer")
    parser.add_argument("--mode", choices=["random", "checkpoint"], default="random")
    parser.add_argument("--checkpoint", default=None, help="Path to a Ray RLlib checkpoint file (--mode checkpoint)")
    parser.add_argument("--out", required=True, help="Output replay JSON path")
    parser.add_argument("--num-agents", type=int, default=25)
    args = parser.parse_args()

    if args.mode == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode checkpoint")

    replay_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "_replay_tmp")
    os.makedirs(replay_tmp_dir, exist_ok=True)

    env = build_env(replay_output_dir=replay_tmp_dir, num_agents=args.num_agents)

    act_fn = None
    if args.mode == "checkpoint":
        act_fn = _load_rllib_policy(args.checkpoint)

    before_files = set(os.listdir(replay_tmp_dir))
    steps = run_episode(env, act_fn=act_fn)
    after_files = set(os.listdir(replay_tmp_dir))

    new_files = sorted(after_files - before_files)
    if not new_files:
        raise SystemExit(
            f"Nenhum replay foi salvo em {replay_tmp_dir} após {steps} steps — "
            "o dia simulado talvez não tenha terminado (24h) dentro do limite de steps."
        )

    produced = os.path.join(replay_tmp_dir, new_files[-1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    shutil.move(produced, args.out)
    shutil.rmtree(replay_tmp_dir, ignore_errors=True)

    print(f"✅ Replay ({args.mode}, {steps} steps) salvo em {args.out}")


if __name__ == "__main__":
    main()
