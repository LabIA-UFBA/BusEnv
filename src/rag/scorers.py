"""Text-based scores for the candidate chunks (offline only).

bm25   classic keyword search: how many words of the bus's query appear in the chunk's
       text (rare words count more). Divided by the best score of that key, so 0..1.
dense  meaning-based search: the query and each chunk are turned into vectors by a
       multilingual sentence model (the text is Portuguese), and the score is the cosine
       similarity. FAISS computes it, restricted to that key's candidates.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from rag.places import fold

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    """Lowercase, accent-free words, e.g. "Avenida Luiz Viana" -> ["avenida", "luiz", "viana"]."""
    return _TOKEN_RE.findall(fold(text))


def bm25_scores(pools: pd.DataFrame, chunk_texts: pd.Series, query_texts: dict) -> np.ndarray:
    """BM25 score of every (key, candidate) row. Word rarity is measured over all chunks."""
    from rank_bm25 import BM25Okapi

    ids = chunk_texts.index.to_numpy()
    pos = {int(c): i for i, c in enumerate(ids)}
    bm25 = BM25Okapi([tokenize(t) for t in chunk_texts.tolist()])
    out = np.zeros(len(pools), dtype=np.float32)
    for qkey, g in pools.groupby("qkey", sort=False):
        doc_idx = [pos[int(c)] for c in g["chunk_id"].to_numpy()]
        s = np.asarray(bm25.get_batch_scores(tokenize(query_texts[int(qkey)]), doc_idx), dtype=np.float32)
        m = float(s.max()) if len(s) else 0.0
        out[g.index.to_numpy()] = s / m if m > 0 else 0.0
    return out


def encode(texts, model_name: str = DENSE_MODEL, device: str = None, batch_size: int = 256) -> np.ndarray:
    """Sentence vectors (length 1) for a list of texts."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def dense_scores(pools: pd.DataFrame, chunk_texts: pd.Series, query_texts: dict,
                 model_name: str = DENSE_MODEL, device: str = None) -> np.ndarray:
    """Embedding similarity of every (key, candidate) row, computed with a FAISS index."""
    import faiss

    ids = chunk_texts.index.to_numpy().astype(np.int64)
    c_emb = encode(chunk_texts.tolist(), model_name, device)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(c_emb.shape[1]))  # exact inner-product search
    index.add_with_ids(c_emb, ids)
    qkeys = list(query_texts.keys())
    q_emb = encode([query_texts[k] for k in qkeys], model_name, device)
    q_pos = {int(k): i for i, k in enumerate(qkeys)}
    out = np.zeros(len(pools), dtype=np.float32)
    for qkey, g in pools.groupby("qkey", sort=False):
        cand = g["chunk_id"].to_numpy().astype(np.int64)
        # only search among this key's candidates
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(cand))
        D_, I_ = index.search(q_emb[q_pos[int(qkey)]][None, :], len(cand), params=params)
        score = dict(zip(I_[0].tolist(), D_[0].tolist()))
        out[g.index.to_numpy()] = [score.get(int(c), 0.0) for c in cand]
    return out
