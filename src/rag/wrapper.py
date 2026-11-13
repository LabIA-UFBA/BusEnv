"""Adds the context numbers to each bus's observation during training.

Wraps the BusEnv simulator: after every reset() and step(), it looks up each bus's
situation in the context table and appends the numbers to the bus's 12 original
observation values (12 -> 12 + 10). Nothing else changes: same rewards, same actions.

Order of wrappers:  parallel_env -> RagObsWrapper -> SuperSuit padding -> RLlibSuntBus
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from gym import spaces
from pettingzoo.utils.wrappers import BaseParallelWraper

from rag.keys import key_from_env
from rag.table import ContextTable


class RagObsWrapper(BaseParallelWraper):
    def __init__(self, env, table_path: str, verify_hash: bool = True):
        super().__init__(env)
        self.table = ContextTable(table_path, verify_hash=verify_hash)
        self.D = self.table.D
        self._base = env  # the simulator itself; bus positions and clocks are read from it
        # new observation space = original 12 values + D context values in [-1, 1]
        base_space = env.observation_space(env.possible_agents[0])
        low = np.concatenate([base_space.low, -np.ones(self.D, dtype=np.float32)]).astype(np.float32)
        high = np.concatenate([base_space.high, np.ones(self.D, dtype=np.float32)]).astype(np.float32)
        self._space = spaces.Box(low=low, high=high, dtype=np.float32)
        # PettingZoo also keeps an old-style dict of spaces; keep it consistent
        self._observation_spaces = {a: self._space for a in env.possible_agents}
        self._action_spaces = {a: env.action_space(a) for a in env.possible_agents}
        self.stats = {"lookups": 0, "misses": 0, "lookup_seconds": 0.0}

    # SuperSuit and MARLlib ask for the spaces through these two methods
    def observation_space(self, agent):
        return self._space

    def action_space(self, agent):
        return self._base.action_space(agent)

    def __getattr__(self, name):
        """Anything not defined here (e.g. num_agents) is read from the simulator."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _augment(self, obs: Dict[str, np.ndarray], infos: Optional[dict] = None) -> Dict[str, np.ndarray]:
        """Appends each bus's context numbers to its observation (zeros if none are found)."""
        out = {}
        t0 = time.perf_counter()
        for agent, o in obs.items():
            key = key_from_env(self._base, agent)
            ctx = self.table.get(key)
            self.stats["lookups"] += 1
            if ctx is None:
                ctx = self.table.zeros()
                if key is not None:  # an active bus whose situation is missing from the table
                    self.stats["misses"] += 1
                    if infos is not None and isinstance(infos.get(agent), dict):
                        infos[agent]["rag_miss"] = 1
            out[agent] = np.concatenate([np.asarray(o, dtype=np.float32), ctx]).astype(np.float32)
        self.stats["lookup_seconds"] += time.perf_counter() - t0
        return out

    def reset(self):
        obs = self.env.reset()
        self.agents = self.env.agents
        return self._augment(obs)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        self.agents = self.env.agents
        return self._augment(obs, infos), rewards, dones, infos
