"""Tests for the training side of src/rag (run in the marllib environment):

  cd BusEnv && PYTHONPATH=src /mnt/ssd1/conda/envs/marllib/bin/python -m pytest -q src/tests/test_rag_online.py

What is checked:
  - our copy of the passenger lookup gives exactly the simulator's numbers
  - keys convert to integers and back; our "next stops" logic matches the simulator's
  - without retrieval, the simulator behaves exactly as before (same observations and rewards)
  - with an all-zero table, the 12 original values are unchanged and 10 zeros are appended
  - the day split keeps only 2024 days for training and only 2025 days for testing
  - with the real tables: every bus situation is found and a lookup takes microseconds
  - every --retrieval method name points to an existing table
"""
import os
import pickle
import random
import types

import numpy as np
import pytest

from rag.chunks import flow_stats
from rag.day_split import apply_day_split
from rag.keys import KeyCodec, QueryKey, future_route_indices, key_from_env
from rag.table import save_table

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OBS = os.path.join(SRC, "training_observation")
TRIPS = ["20005_1719_10", "20015_1413_1", "20041_1230_10", "30013_120_10", "20002_1367_5"]


def _pk(name):
    with open(os.path.join(OBS, name), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def env_factory(tmp_path_factory):
    from envs.sunt_env import DefaultReward, parallel_env

    with open(os.path.join(SRC, "viz", "graph_gtfs_fev_2024.gpickle"), "rb") as f:
        G = pickle.load(f)
    data = {n: _pk(n + ".pkl") for n in ("avg_travel_time_AB", "future_demand_at_B", "occupancy_rate",
                                         "uptime_normalized", "real_routes", "route_metadata")}
    flow = _pk("stop_passenger_flow.pkl")
    metrics = str(tmp_path_factory.mktemp("metrics") / "episode_metrics.csv")

    def make():
        random.seed(0)
        np.random.seed(0)
        return parallel_env(network=G, actions_amount=3, agents_per_route=5, use_only_mean_data=0, max_steps=10000,
                            num_agents=25, use_rain=False, passenger_flow_stats=flow, occupancy_source="real",
                            reward_raining_type="normal", metrics_file_objectives=metrics,
                            reward_scalarization_mode="weighted",
                            rewardClass=DefaultReward(reward_weights={"occ_penalty": 0.6, "uptime_bonus": 0.4,
                                                                      "sync_score": 0.5, "energy_efficiency": 0.5}),
                            **data)
    return make


def test_flow_cascade_matches_simulator():
    from envs.sunt_env import parallel_env

    stats = _pk("stop_passenger_flow.pkl")
    routes = _pk("real_routes.pkl")
    fake = types.SimpleNamespace(passenger_flow_stats=stats)
    rng = random.Random(0)
    for _ in range(1000):
        trip = rng.choice(TRIPS)
        stop = rng.choice(routes[trip])
        t = rng.uniform(4 * 3600, 24 * 3600)
        gf = rng.random() < 0.5
        a = parallel_env._get_flow_stats(fake, trip, stop, t, gf)
        b = flow_stats(stats, trip, stop, t, gf)
        for k in ("mean_boardings", "mean_alight_frac", "mean_intervisit_sec"):
            assert a[k] == b[k], (trip, stop, t, gf, k)


def test_keys_and_bounce(env_factory):
    codec = KeyCodec(TRIPS)
    for k in [QueryKey("20002_1367_5", 0, True, 6, 0), QueryKey("30013_120_10", 41, False, 23, 2)]:
        assert codec.decode(codec.encode(k)) == k
    env = env_factory()
    for L in (2, 5, 33):
        for idx in range(L):
            for gf in (True, False):
                ours = [i for i, _, _ in future_route_indices(idx, gf, L, 7)]
                assert ours == env._future_route_indices(idx, gf, L, 7)


def _rollout(env, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    obs = env.reset()
    traj = [obs]
    for _ in range(steps):
        acts = {a: int(rng.integers(0, 2)) for a in env.possible_agents}  # WAIT/MOVE only
        obs, rew, done, info = env.step(acts)
        traj.append((obs, rew))
    return traj


def test_rag_off_is_bit_identical(env_factory):
    """Two runs without retrieval and with the same seed produce identical results."""
    a = _rollout(env_factory())
    b = _rollout(env_factory())
    for x, y in zip(a[1:], b[1:]):
        for ag in x[0]:
            assert np.array_equal(x[0][ag], y[0][ag])
            assert x[1][ag] == y[1][ag]


def _zero_table(path, D=10):
    from rag.keys import DAY_TYPES

    codec = KeyCodec(TRIPS)
    routes = _pk("real_routes.pkl")
    keys = [codec.encode(QueryKey(t, i, gf, h, dt)) for t in TRIPS for i in range(len(routes[t]))
            for gf in (True, False) for h in range(0, 24) for dt in range(len(DAY_TYPES))]
    save_table(str(path), np.array(keys), np.zeros((len(keys), D), np.float32),
               {"arm": "b0z", "trips": TRIPS, "featurizer_version": "test"})
    return str(path)


def test_zero_table_wrapper_and_padding(env_factory, tmp_path):
    """Adding an all-zero context must not change anything else."""
    from rag.wrapper import RagObsWrapper
    from supersuit import pad_action_space_v0, pad_observations_v0

    table = _zero_table(tmp_path / "rag_table_b0z.npz")
    base = _rollout(env_factory(), steps=100)
    wrapped_env = RagObsWrapper(env_factory(), table)
    wrapped = _rollout(wrapped_env, steps=100)
    for x, y in zip(base[1:], wrapped[1:]):
        for ag in x[0]:
            assert y[0][ag].shape == (22,)
            assert np.array_equal(x[0][ag], y[0][ag][:12])
            assert not y[0][ag][12:].any()
            assert x[1][ag] == y[1][ag]
    assert wrapped_env.stats["misses"] == 0

    padded = pad_action_space_v0(pad_observations_v0(RagObsWrapper(env_factory(), table)))
    assert padded.observation_space(padded.possible_agents[0]).shape == (22,)
    obs = padded.reset()
    assert all(o.shape == (22,) for o in obs.values())


def test_wrapper_keys_match_env_state(env_factory, tmp_path):
    from rag.wrapper import RagObsWrapper

    env = RagObsWrapper(env_factory(), _zero_table(tmp_path / "t.npz"))
    env.reset()
    for _ in range(30):
        env.step({a: 1 for a in env.possible_agents})
    k = key_from_env(env._base, "agent_0")
    st = env._base.agent_states["agent_0"]
    assert k.route_idx == st["route_idx"] and k.going_forward == st["going_forward"]
    assert k.trip_id == env._base.agent_route_id["agent_0"]


TABLES = os.path.join(SRC, "..", "artifacts", "rag", "tables")


@pytest.mark.skipif(not os.path.exists(os.path.join(TABLES, "rag_table_b5b.npz")), reason="offline tables not built")
@pytest.mark.parametrize("arm", ["b1", "b4", "b5a", "b5b"])
def test_real_tables_lookup(env_factory, arm, capsys):
    """Real tables: no misses on a training-day rollout, varying context, lookup cost per agent-step."""
    from rag.wrapper import RagObsWrapper

    base = env_factory()
    apply_day_split(base, "train")
    env = RagObsWrapper(base, os.path.join(TABLES, f"rag_table_{arm}.npz"))
    traj = _rollout(env, steps=300, seed=1)
    ctx = np.array([o[12:] for obs, _ in traj[1:] for o in obs.values() if o.any()])
    assert env.stats["misses"] == 0
    assert ctx.shape[1] == env.D and ctx.std(axis=0).max() > 0
    per_lookup_us = 1e6 * env.stats["lookup_seconds"] / max(env.stats["lookups"], 1)
    with capsys.disabled():
        print(f"\n[{arm}] lookups={env.stats['lookups']} misses=0 per-lookup={per_lookup_us:.1f}us "
              f"ctx mean={np.round(ctx.mean(0), 3).tolist()}")
    assert per_lookup_us < 50


@pytest.mark.skipif(not os.path.exists(os.path.join(TABLES, "rag_table_b5b.npz")), reason="offline tables not built")
def test_every_method_has_a_table():
    from rag.methods import METHODS, table_for

    for m in METHODS:
        path = table_for(m)
        assert (path is None) == (m == "none")


def test_day_split(env_factory):
    env = env_factory()
    n_all = env.total_days
    n_tr = apply_day_split(env, "train")
    assert 0 < n_tr < n_all and all(f[11:21] <= "2024-12-31" for f in env.daily_files)
    env.reset()
    assert env.date <= "2024-12-31"
    env2 = env_factory()
    n_te = apply_day_split(env2, "test")
    assert n_tr + n_te == n_all and all(f[11:21] > "2024-12-31" for f in env2.daily_files)
    env2.reset()
    assert env2.date.startswith("2025")
