"""Tests for the offline side of src/rag (run in the offline environment, Python 3.9):

  cd BusEnv && PYTHONPATH=src /mnt/ssd1/rafael/venvs/busenv-rag/bin/python -m pytest -q src/tests/test_rag_offline.py

Builds a small example (two routes, two hours, random numbers instead of text scores) and checks:
  - candidates go from the nearest stop to the farthest (about 4 per stop); every chunk says
    which file its numbers came from
  - labels: only clearly unusual chunks are marked useful
  - the 10 context numbers are finite, within range, and zero when nothing was chosen
  - top-K / random / oracle respect their limits; a table can be saved and read back
  - RL-Chunk and MA-Chunk train for a few steps and return valid choices
"""
import os

import numpy as np
import pandas as pd
import pytest

from rag.chunks import SuntData, add_ranks, add_structural_scores, build_pools, enumerate_keys, numeric_emb
from rag.featurize import D, compute_norms, featurize
from rag.keys import KeyCodec
from rag.labels import l1_labels
from rag import selectors as S
from rag.table import ContextTable, save_table

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GTFS = "/mnt/ssd1/wesley/BusEnv/SUNT/tmp/GTFS/BUS"
TRIPS = ["20002_1367_5", "30013_120_10"]
H = 10


@pytest.fixture(scope="module")
def built():
    from rag.places import load_gtfs_places, load_gtfs_route_names

    places = load_gtfs_places(os.path.join(GTFS, "stops.txt")) if os.path.exists(GTFS) else {}
    names = load_gtfs_route_names(os.path.join(GTFS, "routes.txt")) if os.path.exists(GTFS) else {}
    data = SuntData(os.path.join(SRC, "training_observation"), os.path.join(SRC, "training_observation", "daily"),
                    "2024-03-01", "2024-12-31", places, names)
    codec = KeyCodec(TRIPS)
    keys = enumerate_keys(data, TRIPS, [7, 17])
    pools, reg = build_pools(data, keys, codec, H)
    chunks = reg.frame()
    pools = add_structural_scores(pools, chunks, data, codec, H)
    rng = np.random.default_rng(0)
    for a in ("bm25", "dense"):  # stand-ins for the text scorers
        pools[f"s_{a}"] = rng.random(len(pools)).astype(np.float32)
    sc = ["bm25", "dense", "place", "proximity", "eta", "salience", "support"]
    pools = add_ranks(pools, sc)
    pools["s_rrf"] = sum(1.0 / (60 + pools[f"rank_{a}"]) for a in sc) / (len(sc) / 60)
    pools = add_ranks(pools, ["rrf"])
    pools = l1_labels(pools)
    pools["emb"] = list(numeric_emb(pools, chunks, H))
    return data, codec, keys, pools, chunks


def test_pools_route_order_and_provenance(built):
    data, codec, keys, pools, chunks = built
    assert len(pools["qkey"].unique()) == len(keys)
    per_key = pools.groupby("qkey").size()
    assert 3 * H <= per_key.median() <= 4 * H
    hops_sorted = pools.groupby("qkey")["hop"].apply(lambda h: (np.diff(h.to_numpy()) >= 0).all())
    assert hops_sorted.all()
    assert chunks["provenance"].notna().all()
    assert set(chunks["ctype"]) == {"stop_hour", "segment", "stop_activity"}


def test_labels(built):
    *_, pools, _ = built
    assert (pools["n_support"] >= 1).all()
    assert (np.abs(pools.loc[pools["useful"] == 1, "z"]) >= 0.5).all()
    assert 0 < pools["useful"].mean() < 0.6


def test_featurizer_and_tables(built, tmp_path):
    data, codec, keys, pools, chunks = built
    norms = compute_norms(chunks)
    all_keys = pools["qkey"].unique()
    k_all, f_all = featurize(pools, chunks, all_keys, norms, 10, H)
    k_none, f_none = featurize(pools.iloc[0:0], chunks, all_keys, norms, 10, H)
    k_top, f_top = featurize(S.select_topk(pools, "rrf", 4), chunks, all_keys, norms, 10, H)
    assert f_all.shape == (len(all_keys), D) and np.isfinite(f_all).all()
    assert (f_all >= -1).all() and (f_all <= 1).all()
    assert not f_none.any()
    assert np.array_equal(k_all, k_none) and np.array_equal(k_all, k_top)
    assert (f_top[:, 9] == 1).all()
    p = tmp_path / "rag_table_test.npz"
    save_table(str(p), k_top, f_top, {"arm": "test", "trips": TRIPS})
    t = ContextTable(str(p))
    k0 = codec.decode(int(k_top[0]))
    assert np.allclose(t.get(k0), f_top[0])


def test_fixed_selectors_budget(built):
    *_, pools, _ = built
    for sel in (S.select_topk(pools, "bm25", 5), S.select_random(pools, 5), S.select_oracle(pools, 10)):
        assert sel.groupby("qkey").size().max() <= 10
    assert S.select_topk(pools, "dense", 5).groupby("qkey").size().eq(5).all()
    m = S.retrieval_metrics(pools, S.select_oracle(pools, 10))
    assert m["precision"] == pytest.approx(1.0)


def test_rlchunk_smoke(built):
    *_, pools, _ = built
    lookup = S.rlchunk_lookup(pools)
    keys = list(lookup)
    model = S.train_rlchunk(keys, lookup, timesteps=2048, seed=0, k_max=10)
    sel = S.rollout_rlchunk(model, keys[:50], lookup, k_max=10)
    assert len(sel) == 50 and all(len(v) <= 10 for v in sel.values())


def test_machunk_smoke(built):
    *_, pools, _ = built
    agents = ["bm25", "dense", "place", "proximity", "eta", "salience", "support"]
    keys = list(pools["qkey"].unique())
    qs = S.machunk_queries(pools, keys, agents, top_m=8, tokens_per_chunk=100)
    model, kw = S.train_machunk(qs, agents, lam=1.0, timesteps=4096, seed=0, k_max=10, tokens_per_chunk=100)
    sel = S.rollout_machunk(model, qs[:50], kw)
    assert len(sel) == 50 and all(len(v) <= 10 for v in sel.values())
