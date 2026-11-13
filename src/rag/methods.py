"""The retrieval methods a user can choose, and which context table each one uses.

Train with:   python src/pipelines/train_marllib.py --algo mappo --retrieval ma-chunk
(or set RAG_RETRIEVAL=ma-chunk). The default "none" trains the original BusEnv.

Each method name maps to a table built offline by export_rag_tables.py. The short code
in brackets is the arm name used in the experiment documents.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts", "rag", "tables")

# method name -> table file suffix
METHODS = {
    "none": None,          # [B0]  original BusEnv, no extra context
    "zeros": "b0z",        # [B0z] extra inputs present but always 0 (control for input size)
    "lookahead": "b1",     # [B1]  everything about the next stops, no selection
    "bm25": "b2",          # [B2]  text search (BM25), top K chunks
    "dense": "b3",         # [B3]  embedding search (multilingual model + FAISS), top K chunks
    "hybrid": "b4",        # [B4]  BM25 + embeddings + structural scores combined, top K chunks
    "rl-chunk": "b5a",     # [B5a] RL-Chunk decides which chunks to keep (single agent)
    "ma-chunk": "b5b",     # [B5b] MA-Chunk decides which chunks to keep (several scorer agents)
    "random": "b6",        # [B6]  K random chunks (control)
    # ablations / controls
    "ma-chunk-no-salience": "a1",  # [A1] MA-Chunk without the salience scorer
    "shuffled": "a7",      # [A7]  real numbers, but attached to the wrong situations (control)
    "oracle": "a8",        # [A8]  the labelled "useful" chunks (upper bound)
    "bm25-no-names": "a11_b2",       # [A11] same as bm25, but stop names removed from the text
    "dense-no-names": "a11_b3",
    "rl-chunk-no-names": "a11_b5a",
    "ma-chunk-no-names": "a11_b5b",
}


def table_for(method: str, tables_dir: Optional[str] = None) -> Optional[str]:
    """Returns the table path for a method name, or None for "none"."""
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method!r}; choose one of: {', '.join(METHODS)}")
    suffix = METHODS[method]
    if suffix is None:
        return None
    path = os.path.join(tables_dir or os.environ.get("RAG_TABLES_DIR", DEFAULT_TABLES_DIR), f"rag_table_{suffix}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"table for {method!r} not found at {path}; run the offline build (RUNBOOK §2)")
    return os.path.abspath(path)
