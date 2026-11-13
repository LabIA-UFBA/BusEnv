"""The lookup table: the only file the training side reads.

Each experiment arm has two files:
  rag_table_<arm>.npz   the keys (integers) and, for each key, its 10 context numbers
  rag_table_<arm>.json  a description of how the table was made (settings, file hashes)

The .json also stores a hash of the .npz, so a modified or mismatched table is detected
when it is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

import numpy as np

from rag.keys import KeyCodec


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Fingerprint of a file, used to check that inputs and tables did not change."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_table(path_npz: str, keys: np.ndarray, feats: np.ndarray, manifest: Dict) -> str:
    """Saves the table and its description. Refuses duplicate keys and NaN/inf values."""
    keys = np.asarray(keys, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.float32)
    order = np.argsort(keys)
    keys, feats = keys[order], feats[order]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate keys in context table")
    if not np.all(np.isfinite(feats)):
        raise ValueError("non-finite values in context table")
    os.makedirs(os.path.dirname(os.path.abspath(path_npz)), exist_ok=True)
    np.savez_compressed(path_npz, keys=keys, feats=feats)
    manifest = dict(manifest)
    manifest["n_keys"] = int(len(keys))
    manifest["D"] = int(feats.shape[1])
    manifest["table_sha256"] = sha256_file(path_npz)
    with open(_manifest_path(path_npz), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest["table_sha256"]


def _manifest_path(path_npz: str) -> str:
    return os.path.splitext(path_npz)[0] + ".json"


class ContextTable:
    """Loads a table into memory and answers "what are the context numbers for this key?".

    get() returns None for an unknown key; the caller then uses zeros().
    """

    def __init__(self, path_npz: str, verify_hash: bool = True):
        with open(_manifest_path(path_npz)) as f:
            self.manifest = json.load(f)
        if verify_hash and sha256_file(path_npz) != self.manifest["table_sha256"]:
            raise ValueError(f"hash mismatch for {path_npz}")
        data = np.load(path_npz)
        keys, feats = data["keys"], data["feats"].astype(np.float32)
        self.D = int(feats.shape[1])               # number of context numbers per key
        self.codec = KeyCodec(self.manifest["trips"])
        self._index = {int(k): i for i, k in enumerate(keys)}
        self._feats = feats
        self._zeros = np.zeros(self.D, dtype=np.float32)
        self.arm = self.manifest.get("arm")

    def get(self, key) -> Optional[np.ndarray]:
        if key is None or not self.codec.has_trip(key.trip_id):
            return None
        try:
            i = self._index.get(self.codec.encode(key))
        except ValueError:  # key outside the table's range (e.g. an hour never built)
            return None
        return None if i is None else self._feats[i]

    def zeros(self) -> np.ndarray:
        return self._zeros
