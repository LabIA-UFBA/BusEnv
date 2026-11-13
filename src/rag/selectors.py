"""Ways of choosing which candidate chunks to keep for each bus situation (offline only).

Simple rules:
  select_all      keep every candidate ("lookahead")
  select_topk     keep the K best by one score (BM25, dense, hybrid)
  select_random   keep K random candidates
  select_oracle   keep the chunks labelled useful (best possible, for comparison)

Learned (they decide how many and which chunks to keep):
  RL-Chunk  one agent walks through the candidates from the nearest stop to the farthest
            and decides keep / skip for each one.
  MA-Chunk  several agents, one per score (BM25, dense, proximity, ...), take turns
            proposing chunks; they share what has already been chosen and can stop early.

RL-Chunk and MA-Chunk are NOT copied into BusEnv. They are imported from their own git
repositories, wherever they are cloned (paths below, or the env vars RL_CHUNK_PIPELINE and
MA_CHUNK_REPO). Their code is used unchanged.

Every selector returns the chosen rows of the candidate table.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

RL_CHUNK_PIPELINE = os.environ.get("RL_CHUNK_PIPELINE", "/mnt/ssd1/rafael/git_rl_chunk/RL-Chunk/src/rl_pipeline")
MA_CHUNK_REPO = os.environ.get("MA_CHUNK_REPO", "/mnt/ssd1/rafael/git_rl_chunk/MA-Chunk")


# ----------------------------------------------------------------------------------------
# Simple rules
# ----------------------------------------------------------------------------------------
def select_all(pools: pd.DataFrame) -> pd.DataFrame:
    return pools


def select_topk(pools: pd.DataFrame, scorer: str, k: int) -> pd.DataFrame:
    return pools[pools[f"rank_{scorer}"] < k]


def select_random(pools: pd.DataFrame, k: int, seed: int = 20261113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.random(len(pools)), index=pools.index)
    rank = r.groupby(pools["qkey"]).rank(method="first") - 1
    return pools[rank < k]


def select_oracle(pools: pd.DataFrame, k_max: int) -> pd.DataFrame:
    """The chunks labelled useful (at most k_max per key). Best possible selection, for comparison."""
    u = pools[pools["useful"] == 1].copy()
    u["_w"] = np.abs(u["z"]) / np.sqrt(u["hop"])
    u["_r"] = u.groupby("qkey")["_w"].rank(ascending=False, method="first")
    return u[u["_r"] <= k_max].drop(columns=["_w", "_r"])


def select_by_ids(pools: pd.DataFrame, selections: Dict[int, Sequence[int]]) -> pd.DataFrame:
    """Turns {key: [candidate positions]} (what RL-Chunk / MA-Chunk return) into the chosen rows."""
    keyset = {(int(q), int(c)) for q, cs in selections.items() for c in cs}
    mask = [(int(q), int(c)) in keyset for q, c in zip(pools["qkey"], pools["cand_pos"])]
    return pools[mask]


# ----------------------------------------------------------------------------------------
# RL-Chunk: its own environment, with only the data source replaced
# ----------------------------------------------------------------------------------------
def _import_rlchunk():
    # RL-Chunk's config refuses to load without this variable; no OpenAI call is ever made here
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    if RL_CHUNK_PIPELINE not in sys.path:
        sys.path.insert(0, RL_CHUNK_PIPELINE)
    import env as rlc_env  # noqa: E402  (RL-Chunk/src/rl_pipeline/env.py)
    import config as rlc_config  # noqa: E402
    return rlc_env, rlc_config


def rlchunk_lookup(pools: pd.DataFrame, sim_col: str = "s_dense", bm25_col: str = "s_bm25") -> Dict[int, tuple]:
    """What RL-Chunk sees for each key: the candidates from nearest to farthest stop, plus two
    scores per candidate, both rescaled to 0..1 within the key: similarity (dense) and BM25.
    -> {key: (candidate positions, similarity, bm25)}"""
    out = {}
    for qkey, g in pools.sort_values(["qkey", "cand_pos"]).groupby("qkey", sort=False):
        s = g[sim_col].to_numpy(dtype=np.float32)
        lo, hi = float(s.min()), float(s.max())
        sim = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        b = np.clip(g[bm25_col].to_numpy(dtype=np.float32), 0, 1)
        out[int(qkey)] = (g["cand_pos"].astype(int).tolist(), sim.astype(np.float32), b)
    return out


def make_bus_chunk_env_class():
    rlc_env, _ = _import_rlchunk()

    class BusChunkSelectionEnv(rlc_env.ChunkSelectionEnv):
        """RL-Chunk's ChunkSelectionEnv, unchanged except that chunks and scores come from our table
        instead of being computed from a web page."""
        LOOKUP: Dict[int, tuple] = {}

        def _chunks_and_scores(self, row_idx, row):
            return self.LOOKUP[int(row["qkey"])]

    return BusChunkSelectionEnv, rlc_env


def train_rlchunk(train_keys: Sequence[int], lookup: Dict[int, tuple], timesteps: int, seed: int, k_max: int,
                  algo: str = "ppo"):
    """Trains RL-Chunk with the same settings as RL-Chunk's own train.py
    (PPO, n_steps=128, ent_coef=0.01). k_max = most chunks it may keep per key."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO

    Env, rlc_env = make_bus_chunk_env_class()
    Env.LOOKUP = lookup
    rlc_env.MAX_SELECTED_CHUNKS = k_max  # RL-Chunk reads its chunk budget from this module constant
    # RL-Chunk's reset() expects a "query" column; its text is not used because scores are precomputed
    rows = pd.DataFrame({"qkey": list(train_keys), "query": [str(k) for k in train_keys]})

    def _init():
        return Monitor(Env(rows, continuous_action=False, row_order="random", seed=seed), info_keywords=("n_selected",))

    venv = DummyVecEnv([_init])
    if algo == "ppo":
        model = PPO("MultiInputPolicy", venv, n_steps=128, ent_coef=0.01, seed=seed, verbose=0, device="cpu")
    elif algo == "recurrent_ppo":
        model = RecurrentPPO("MultiInputLstmPolicy", venv, n_steps=128, ent_coef=0.01, seed=seed, verbose=0, device="cpu")
    else:
        raise ValueError(algo)
    model.learn(total_timesteps=timesteps)
    return model


def rollout_rlchunk(model, keys: Sequence[int], lookup: Dict[int, tuple], k_max: int, algo: str = "ppo") -> Dict[int, List[int]]:
    """Runs the trained RL-Chunk once per key (no randomness) -> {key: chosen candidate positions}."""
    Env, rlc_env = make_bus_chunk_env_class()
    Env.LOOKUP = lookup
    rlc_env.MAX_SELECTED_CHUNKS = k_max
    out = {}
    for q in keys:
        row = pd.Series({"qkey": int(q), "query": str(q)})
        env = Env(row.to_frame().T, continuous_action=False, row_order="sequential")
        obs, _ = env.reset(row=row)
        done, lstm, start = False, None, np.ones((1,), dtype=bool)
        while not done:
            if algo == "recurrent_ppo":
                action, lstm = model.predict(obs, state=lstm, episode_start=start, deterministic=True)
                start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        out[int(q)] = [int(c) for c in env.selected_chunks]
    return out


# ----------------------------------------------------------------------------------------
# MA-Chunk
# ----------------------------------------------------------------------------------------
def _import_machunk():
    if MA_CHUNK_REPO not in sys.path:
        sys.path.insert(0, MA_CHUNK_REPO)
    from ma_chunk import env_ma, train_ma  # noqa: E402
    return env_ma, train_ma


def machunk_queries(pools: pd.DataFrame, keys: Sequence[int], agents: Sequence[str], top_m: int,
                    tokens_per_chunk: int, score_suffix: str = ""):
    """Packs each key's candidates in MA-Chunk's input format (one QueryData per key).

    tokens_per_chunk is the "price" unit of a chunk: MA-Chunk pays lambda * tokens / 1000 for
    each chunk it keeps, so with 100 and lambda = 1 each chunk costs 0.1.
    The candidate position is used as chunk id, so the answer maps straight back to our table.
    """
    env_ma, _ = _import_machunk()
    sub = pools[pools["qkey"].isin(set(int(k) for k in keys))]
    qs = []
    for qkey, g in sub.groupby("qkey", sort=False):
        df = pd.DataFrame({
            "interaction_id": int(qkey), "chunk_id": g["cand_pos"].to_numpy(), "text": "",
            "tokens": tokens_per_chunk, "emb": list(g["emb"].to_numpy()),
            "useful": g["useful"].to_numpy(), "n_support": g["n_support"].to_numpy(),
        })
        for a in agents:
            df[f"s_{a}"] = g[f"s_{a}{score_suffix}" if f"s_{a}{score_suffix}" in g else f"s_{a}"].to_numpy()
            df[f"rank_{a}"] = g[f"rank_{a}{score_suffix}" if f"rank_{a}{score_suffix}" in g else f"rank_{a}"].to_numpy()
        qs.append(env_ma.QueryData(df, top_m, list(agents), list(agents)))
    return qs


def train_machunk(queries, agents, lam: float, timesteps: int, seed: int, k_max: int, tokens_per_chunk: int,
                  share_scores: bool = True, algo: str = "ppo"):
    """Trains MA-Chunk with its own model settings (train_ma.make_model).
    share_scores=True lets each agent also see the other agents' scores ("messages")."""
    env_ma, train_ma = _import_machunk()
    env_kwargs = dict(lam=lam, token_budget=k_max * tokens_per_chunk, max_chunks=k_max, share_scores=share_scores,
                      universe=list(agents))
    env = env_ma.MAChunkEnv(queries, seed=seed, **env_kwargs)
    model = train_ma.make_model(algo, env, seed)
    model.learn(total_timesteps=timesteps)
    return model, env_kwargs


def rollout_machunk(model, queries, env_kwargs) -> Dict[int, List[int]]:
    """Runs the trained MA-Chunk once per key (no randomness) -> {key: chosen candidate positions}."""
    _, train_ma = _import_machunk()
    rows = train_ma.rollout(model, queries, env_kwargs)
    return {int(r["interaction_id"]): [int(c) for c in r["chunk_ids"]] for r in rows}


# ----------------------------------------------------------------------------------------
# How well a selection matches the "useful" labels
# ----------------------------------------------------------------------------------------
def retrieval_metrics(pools: pd.DataFrame, selected: pd.DataFrame) -> Dict[str, float]:
    """precision = share of chosen chunks that are useful;
    recall_utility = share of the useful chunks that were chosen (capped at 1);
    n_sel = chunks chosen per key; empty_rate = share of keys where nothing was chosen."""
    keys = pools["qkey"].unique()
    n_useful_sel = selected.groupby("qkey")["useful"].sum().reindex(keys, fill_value=0)
    n_sel = selected.groupby("qkey").size().reindex(keys, fill_value=0)
    n_support = pools.groupby("qkey")["n_support"].first().reindex(keys)
    has_useful = pools.groupby("qkey")["useful"].sum().reindex(keys) > 0
    precision = np.where(n_sel > 0, n_useful_sel / n_sel.clip(lower=1), np.nan)
    recall = np.minimum(1.0, n_useful_sel / n_support)
    p = np.nanmean(precision) if np.isfinite(precision).any() else 0.0
    r = float(recall[has_useful].mean()) if has_useful.any() else 0.0
    return {"n_sel_mean": float(n_sel.mean()), "n_sel_std": float(n_sel.std()), "n_sel_median": float(n_sel.median()),
            "precision": float(p), "recall_utility": r, "f1": float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
            "empty_rate": float((n_sel == 0).mean())}
