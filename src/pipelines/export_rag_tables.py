#!/usr/bin/env python3
"""Offline step 3 of 3: build one lookup table per retrieval method, plus a quality report.

One table per method (names as in --retrieval of train_marllib.py, see rag/methods.py):
  zeros, lookahead, bm25, dense, hybrid, rl-chunk, ma-chunk, random,
  and checks: shuffled, oracle, ma-chunk-no-salience, *-no-names.

Fair comparison: the top-K methods keep K chunks, where K is the typical number kept by the
better learned method (RL-Chunk or MA-Chunk), so all methods use a similar amount of context.
For MA-Chunk, the price (lambda) whose choices best match the "useful" labels is used.

  PYTHONPATH=src /mnt/ssd1/rafael/venvs/busenv-rag/bin/python src/pipelines/export_rag_tables.py \
      --rag-dir artifacts/rag

Writes artifacts/rag/tables/rag_table_<arm>.npz/.json and offline_retrieval_metrics.csv.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag import selectors as S  # noqa: E402
from rag.featurize import FEATURE_NAMES, FEATURIZER_VERSION, compute_norms, featurize  # noqa: E402
from rag.table import save_table, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-dir", default="artifacts/rag")
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--k", type=int, default=None, help="force the number of chunks kept by the top-K methods")
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    d = args.rag_dir
    sel_dir, out_dir = os.path.join(d, "selectors"), os.path.join(d, "tables")
    os.makedirs(out_dir, exist_ok=True)

    bm_path = os.path.join(d, "build_manifest.json")
    bm = json.load(open(bm_path))
    chunks = pd.read_parquet(os.path.join(d, "chunks.parquet")).set_index("chunk_id", drop=False)
    pools = pd.read_parquet(os.path.join(d, "pools.parquet"))
    all_keys = pools["qkey"].unique()
    norms = compute_norms(chunks.loc[pools["chunk_id"].unique()])
    H = int(bm["horizon"])

    # --- choices made by RL-Chunk and MA-Chunk (from train_rag_selector.py) ---------------
    def load_sel(tag):
        p = os.path.join(sel_dir, f"{tag}_selections.parquet")
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)
        return S.select_by_ids(pools, dict(zip(df["qkey"], df["cand_pos"])))

    metrics, chosen = {}, {}
    rl = load_sel("rlchunk")
    if rl is not None:
        chosen["b5a"] = ("rlchunk", rl)
    ma_tags = sorted(os.path.basename(p)[:-len("_selections.parquet")]
                     for p in glob.glob(os.path.join(sel_dir, "machunk_lam*_selections.parquet")))
    best_ma = None
    for tag in ma_tags:
        sel = load_sel(tag)
        m = S.retrieval_metrics(pools, sel)
        metrics[f"machunk_grid/{tag}"] = m
        if best_ma is None or m["f1"] > best_ma[2]["f1"]:
            best_ma = (tag, sel, m)
    if best_ma:
        chosen["b5b"] = (best_ma[0], best_ma[1])

    adaptive = {arm: S.retrieval_metrics(pools, sel) for arm, (_, sel) in chosen.items()}
    better = max(adaptive, key=lambda a: adaptive[a]["f1"]) if adaptive else None
    K = args.k or (max(1, int(round(adaptive[better]["n_sel_median"]))) if better else 5)
    print(f"[K] equal-budget K={K} (from {better or 'default'}); adaptive={adaptive}")

    # --- one selection per method ------------------------------------------------------
    arms = {
        "b0z": pools.iloc[0:0],
        "b1": S.select_all(pools),
        "b2": S.select_topk(pools, "bm25", K),
        "b3": S.select_topk(pools, "dense", K),
        "b4": S.select_topk(pools, "rrf", K),
        "b6": S.select_random(pools, K, args.seed),
        "a8": S.select_oracle(pools, args.k_max),
        "a11_b2": S.select_topk(pools, "bm25_noplace", K),
        "a11_b3": S.select_topk(pools, "dense_noplace", K),
    }
    for arm, (_, sel) in chosen.items():
        arms[arm] = sel
    for tag, arm in (("rlchunk_noplace", "a11_b5a"),):
        sel = load_sel(tag)
        if sel is not None:
            arms[arm] = sel
    nosal = sorted(glob.glob(os.path.join(sel_dir, "machunk_nosal_lam*_selections.parquet")))
    if nosal:  # A1: MA-Chunk without the salience scorer (label-circularity check)
        arms["a1"] = load_sel(os.path.basename(nosal[0])[:-len("_selections.parquet")])
    ma_np = sorted(glob.glob(os.path.join(sel_dir, "machunk_noplace_lam*_selections.parquet")))
    if ma_np:
        best = max((os.path.basename(p)[:-len("_selections.parquet")] for p in ma_np),
                   key=lambda t: S.retrieval_metrics(pools, load_sel(t))["f1"])
        arms["a11_b5b"] = load_sel(best)

    common = {
        "trips": bm["trips"], "key_fields": bm["key_fields"], "horizon": H, "k_max": args.k_max, "K": K,
        "featurizer_version": FEATURIZER_VERSION, "feature_names": list(FEATURE_NAMES), "norms": norms,
        "train_start": bm["train_start"], "train_end": bm["train_end"],
        "build_manifest_sha256": sha256_file(bm_path), "label": bm["label"], "scorers": bm["scorers"],
        "dense_model": bm.get("dense_model"),
    }
    rows = []
    for arm, sel in arms.items():
        keys, feats = featurize(sel, chunks, all_keys, norms, args.k_max, H)
        src = chosen.get(arm, (None,))[0]
        info = None
        if src:
            ip = os.path.join(sel_dir, f"{src}_info.json")
            info = json.load(open(ip)) if os.path.exists(ip) else None
        m = S.retrieval_metrics(pools, sel) if arm != "b0z" else {}
        save_table(os.path.join(out_dir, f"rag_table_{arm}.npz"), keys, feats,
                   {**common, "arm": arm, "selector_source": src, "selector_info": info, "offline_metrics": m})
        if arm == better or (better is None and arm == "b4"):
            best_keys, best_feats = keys, feats
        rows.append({"arm": arm, **m})
        print(f"[table] {arm}: {m}", flush=True)

    # "shuffled" check: the better learned method's numbers, but given to the wrong situations.
    # If training with this works as well as the real table, the content does not matter.
    perm = np.random.default_rng(args.seed).permutation(len(best_keys))
    save_table(os.path.join(out_dir, "rag_table_a7.npz"), best_keys, best_feats[perm],
               {**common, "arm": "a7", "selector_source": f"shuffled({better or 'b4'})", "offline_metrics": {}})
    rows.append({"arm": "a7"})

    rep = pd.DataFrame(rows)
    for k, m in metrics.items():
        rep = pd.concat([rep, pd.DataFrame([{"arm": k, **m}])], ignore_index=True)
    rep.to_csv(os.path.join(out_dir, "offline_retrieval_metrics.csv"), index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
