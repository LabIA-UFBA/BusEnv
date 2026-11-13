#!/usr/bin/env python3
"""Offline step 2 of 3: train RL-Chunk or MA-Chunk to choose chunks, and save their choices.

  --method rlchunk   RL-Chunk: one agent, keep/skip each candidate from nearest to farthest stop
  --method machunk   MA-Chunk: one agent per score, taking turns; can stop early

Fair testing: there are 5 routes, so each method is trained 5 times, each time leaving one
route out, and that model makes the choices for the route it never saw.

MA-Chunk pays a price (--lams, "lambda") for every chunk it keeps: higher price = fewer
chunks. Give several prices; export_rag_tables.py keeps the one that best matches the labels.

  PYTHONPATH=src /mnt/ssd1/rafael/venvs/busenv-rag/bin/python src/pipelines/train_rag_selector.py \
      --method machunk --lams 0.5 1 2 --out artifacts/rag/selectors

Trained models and choices are saved under artifacts/ (not committed to git).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag import selectors as S  # noqa: E402
from rag.keys import KeyCodec  # noqa: E402

MA_AGENTS = ["bm25", "dense", "place", "proximity", "eta", "salience", "support"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["rlchunk", "machunk"], required=True)
    ap.add_argument("--pools", default="artifacts/rag/pools.parquet")
    ap.add_argument("--build-manifest", default=None)
    ap.add_argument("--out", default="artifacts/rag/selectors")
    ap.add_argument("--k-max", type=int, default=10, help="most chunks a method may keep per situation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None, help="training steps per route left out (default: rlchunk 40k, machunk 200k)")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--variant", default="", help="'_noplace' = use text scores computed without stop names")
    # MA-Chunk
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0])
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--tokens-per-chunk", type=int, default=100, help="price unit: each kept chunk costs lambda*tokens/1000")
    ap.add_argument("--no-share-scores", action="store_true", help="MA-Chunk agents do not see each other's scores")
    ap.add_argument("--max-keys-per-fold", type=int, default=None, help="for quick tests: train on fewer situations")
    ap.add_argument("--drop-agents", nargs="*", default=[], help="remove MA-Chunk agents, e.g. --drop-agents salience")
    ap.add_argument("--tag-suffix", default="", help="appended to output names, e.g. _nosal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    pools = pd.read_parquet(args.pools)
    bm = json.load(open(args.build_manifest or os.path.join(os.path.dirname(args.pools), "build_manifest.json")))
    codec = KeyCodec(bm["trips"])
    trip_of = {int(k): codec.decode(int(k)).trip_id for k in pools["qkey"].unique()}
    pools["trip_id"] = pools["qkey"].map(trip_of)
    trips = bm["trips"]
    rng = np.random.default_rng(args.seed)
    v = args.variant

    if args.method == "rlchunk":
        lookup = S.rlchunk_lookup(pools, sim_col="s_dense" + v, bm25_col="s_bm25" + v)
        runs = [("rlchunk" + v, None)]
    else:
        agents = [a for a in MA_AGENTS if not (v == "_noplace" and a == "place") and a not in args.drop_agents]
        runs = [(f"machunk{v}{args.tag_suffix}_lam{lam:g}", lam) for lam in args.lams]

    for tag, lam in runs:
        selections, fold_info = {}, []
        for fold, test_trip in enumerate(trips):  # leave this route out of training
            train_keys = [k for k, t in trip_of.items() if t != test_trip]
            test_keys = [k for k, t in trip_of.items() if t == test_trip]
            if args.max_keys_per_fold and len(train_keys) > args.max_keys_per_fold:
                train_keys = list(rng.choice(train_keys, args.max_keys_per_fold, replace=False))
            tf = time.time()
            if args.method == "rlchunk":
                steps = args.timesteps or 40_000
                model = S.train_rlchunk(train_keys, lookup, steps, args.seed + fold, args.k_max, args.algo)
                sel = S.rollout_rlchunk(model, test_keys, lookup, args.k_max, args.algo)
            else:
                steps = args.timesteps or 200_000
                q_train = S.machunk_queries(pools, train_keys, agents, args.top_m, args.tokens_per_chunk, v)
                q_test = S.machunk_queries(pools, test_keys, agents, args.top_m, args.tokens_per_chunk, v)
                model, env_kwargs = S.train_machunk(q_train, agents, lam, steps, args.seed + fold, args.k_max,
                                                    args.tokens_per_chunk, share_scores=not args.no_share_scores,
                                                    algo=args.algo)
                sel = S.rollout_machunk(model, q_test, env_kwargs)
            model.save(os.path.join(args.out, f"{tag}_fold{fold}_{test_trip}.zip"))
            selections.update(sel)
            sub = pools[pools["trip_id"] == test_trip]
            m = S.retrieval_metrics(sub, S.select_by_ids(sub, sel))
            fold_info.append({"fold": fold, "test_trip": test_trip, "train_keys": len(train_keys),
                              "test_keys": len(test_keys), "timesteps": steps, "seconds": round(time.time() - tf, 1), **m})
            print(f"[{tag}] fold {fold} ({test_trip}) {fold_info[-1]}", flush=True)

        sel_df = pd.DataFrame({"qkey": list(selections.keys()), "cand_pos": list(selections.values())})
        sel_df.to_parquet(os.path.join(args.out, f"{tag}_selections.parquet"))
        overall = S.retrieval_metrics(pools, S.select_by_ids(pools, selections))
        info = {"tag": tag, "method": args.method, "lambda": lam, "variant": v, "k_max": args.k_max,
                "seed": args.seed, "algo": args.algo, "agents": None if args.method == "rlchunk" else agents,
                "top_m": args.top_m, "tokens_per_chunk": args.tokens_per_chunk,
                "share_scores": not args.no_share_scores, "folds": fold_info, "overall": overall,
                "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(args.out, f"{tag}_info.json"), "w") as f:
            json.dump(info, f, indent=2, default=float)
        print(f"[{tag}] overall {overall}", flush=True)


if __name__ == "__main__":
    main()
