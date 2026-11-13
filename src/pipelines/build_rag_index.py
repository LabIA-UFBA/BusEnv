#!/usr/bin/env python3
"""Offline step 1 of 3: build the chunks, the candidates for every bus situation, their
scores and the "useful" labels.

Run in the offline environment (Python 3.9), from the BusEnv folder:

  PYTHONPATH=src /mnt/ssd1/rafael/venvs/busenv-rag/bin/python src/pipelines/build_rag_index.py \
      --gtfs-dir /mnt/ssd1/wesley/BusEnv/SUNT/tmp/GTFS/BUS --out artifacts/rag

Writes to --out:
  chunks.parquet        every chunk, with its numbers, where they came from, and its text
  queries.parquet       every bus situation (key) and its search text
  pools.parquet         one row per (situation, candidate chunk): all scores, ranks, labels
  build_manifest.json   settings used, train/test days, fingerprints of the input files
Takes about 5 minutes for the 5 study routes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.chunks import (SuntData, add_ranks, add_structural_scores, build_pools, enumerate_keys,  # noqa: E402
                        numeric_emb, render_chunk, render_query)
from rag.keys import DAY_TYPES, KeyCodec  # noqa: E402
from rag.labels import l1_labels  # noqa: E402
from rag.places import load_gtfs_places, load_gtfs_route_names  # noqa: E402
from rag.table import sha256_file  # noqa: E402

# the 5 routes BusEnv's training script assigns buses to (sunt_env.py manual_route_groups)
STUDY_TRIPS = ["20005_1719_10", "20015_1413_1", "20041_1230_10", "30013_120_10", "20002_1367_5"]
SCORERS = ["bm25", "dense", "place", "proximity", "eta", "salience", "support"]
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-dir", default=os.path.join(SRC, "training_observation"))
    ap.add_argument("--daily-dir", default=os.path.join(SRC, "training_observation", "daily"))
    ap.add_argument("--gtfs-dir", required=True, help="folder with GTFS stops.txt and routes.txt")
    ap.add_argument("--out", default="artifacts/rag")
    ap.add_argument("--trips", nargs="+", default=STUDY_TRIPS)
    ap.add_argument("--hours", nargs=2, type=int, default=[6, 23], help="first and last hour of the day to cover")
    ap.add_argument("--horizon", type=int, default=10, help="how many stops ahead to consider")
    ap.add_argument("--train-start", default="2024-03-01")
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--label-coverage", type=float, default=0.6)
    ap.add_argument("--label-zmin", type=float, default=0.5)
    ap.add_argument("--dense-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-text-scorers", action="store_true", help="for quick tests: set BM25/dense scores to 0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    stops_txt, routes_txt = os.path.join(args.gtfs_dir, "stops.txt"), os.path.join(args.gtfs_dir, "routes.txt")
    places, route_names = load_gtfs_places(stops_txt), load_gtfs_route_names(routes_txt)
    data = SuntData(args.obs_dir, args.daily_dir, args.train_start, args.train_end, places, route_names)
    print(f"[data] {len(data.train_days)} training days; {len(places)} GTFS stops", flush=True)

    codec = KeyCodec(args.trips)
    keys = enumerate_keys(data, args.trips, range(args.hours[0], args.hours[1] + 1))
    pools, reg = build_pools(data, keys, codec, args.horizon)
    chunks = reg.frame()
    print(f"[pools] {len(keys)} keys, {len(chunks)} chunks, {len(pools)} candidate rows "
          f"(mean {len(pools) / len(keys):.1f}/key)  {time.time() - t0:.0f}s", flush=True)

    chunks["text"] = [render_chunk(r, data, True) for r in chunks.to_dict("records")]
    chunks["text_noplace"] = [render_chunk(r, data, False) for r in chunks.to_dict("records")]
    for col in ("street", "neighborhood", "landmark"):
        chunks[col] = [data.place(s).get(col) for s in chunks["stop_id"]]

    qcodes = [codec.encode(k) for k in keys]
    queries = pd.DataFrame({
        "qkey": qcodes, "trip_id": [k.trip_id for k in keys], "route_idx": [k.route_idx for k in keys],
        "going_forward": [k.going_forward for k in keys], "hour": [k.hour for k in keys],
        "day_type": [k.day_type for k in keys],
        "text": [render_query(k, data, True) for k in keys],
        "text_noplace": [render_query(k, data, False) for k in keys],
    })

    pools = add_structural_scores(pools, chunks, data, codec, args.horizon)
    print(f"[structural] done  {time.time() - t0:.0f}s", flush=True)

    # text scores twice: with stop names, and without them ("_noplace", to measure their effect)
    for variant, ctext, qtext in (("", "text", "text"), ("_noplace", "text_noplace", "text_noplace")):
        qd = dict(zip(queries["qkey"].astype(int), queries[qtext]))
        if args.skip_text_scorers:
            pools["s_bm25" + variant] = 0.0
            pools["s_dense" + variant] = 0.0
            continue
        from rag.scorers import DENSE_MODEL, bm25_scores, dense_scores
        pools["s_bm25" + variant] = bm25_scores(pools, chunks[ctext], qd)
        print(f"[bm25{variant}] done  {time.time() - t0:.0f}s", flush=True)
        pools["s_dense" + variant] = dense_scores(pools, chunks[ctext], qd, args.dense_model or DENSE_MODEL, args.device)
        print(f"[dense{variant}] done  {time.time() - t0:.0f}s", flush=True)
    # without stop names the "place" score cannot be computed either
    pools["s_place_noplace"] = 0.0

    pools = add_ranks(pools, SCORERS + ["bm25_noplace", "dense_noplace"])
    # hybrid score: reciprocal rank fusion, i.e. a candidate ranked high by many scorers wins
    rrf_k = 60
    pools["s_rrf"] = sum(1.0 / (rrf_k + pools[f"rank_{a}"]) for a in SCORERS) / (len(SCORERS) / rrf_k)
    pools = add_ranks(pools, ["rrf"])
    pools = l1_labels(pools, args.label_coverage, args.label_zmin)
    pools["emb"] = list(numeric_emb(pools, chunks, args.horizon))

    chunks.to_parquet(os.path.join(args.out, "chunks.parquet"))
    queries.to_parquet(os.path.join(args.out, "queries.parquet"))
    pools.to_parquet(os.path.join(args.out, "pools.parquet"))

    per_key = pools.groupby("qkey")
    manifest = {
        "trips": args.trips, "key_fields": ["trip_id", "route_idx", "going_forward", "hour", "day_type"],
        "day_types": list(DAY_TYPES), "hours": args.hours, "horizon": args.horizon,
        "train_start": args.train_start, "train_end": args.train_end,
        "train_days": [os.path.basename(f)[11:21] for f in data.train_days],
        "scorers": SCORERS, "rrf_k": rrf_k,
        "label": {"type": "L1-deviation-coverage", "coverage": args.label_coverage, "z_min": args.label_zmin},
        "dense_model": None if args.skip_text_scorers else (args.dense_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "n_keys": len(keys), "n_chunks": int(len(chunks)), "n_pool_rows": int(len(pools)),
        "useful_rate": float(pools["useful"].mean()),
        "keys_with_no_useful": float((per_key["useful"].sum() == 0).mean()),
        "mean_n_support": float(per_key["n_support"].first().mean()),
        "chunk_levels": chunks.groupby(["ctype", "level"]).size().rename("n").reset_index().to_dict("records"),
        "inputs_sha256": {name: sha256_file(os.path.join(args.obs_dir, name)) for name in (
            "stop_passenger_flow.pkl", "travel_time_stats.pkl", "avg_travel_time_AB.pkl",
            "hourly_stop_activity.pkl", "real_routes.pkl", "route_metadata.pkl")},
        "gtfs_sha256": {"stops.txt": sha256_file(stops_txt), "routes.txt": sha256_file(routes_txt)},
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps({k: manifest[k] for k in ("n_keys", "n_chunks", "n_pool_rows", "useful_rate",
                                               "keys_with_no_useful", "mean_n_support", "build_seconds")}, indent=2))


if __name__ == "__main__":
    main()
