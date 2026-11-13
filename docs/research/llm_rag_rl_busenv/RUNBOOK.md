# RUNBOOK — Phase 1 (retrieval-augmented context for BusEnv)

Exact commands for the Phase 1 implementation in `src/rag/` (docs 04, 06 and 07). Run them from the `BusEnv/` folder.

## 0. Environments

| Side | Interpreter | Notes |
|---|---|---|
| Offline (chunks, scorers, RL-Chunk, MA-Chunk, tables) | `/mnt/ssd1/rafael/venvs/busenv-rag/bin/python` (Python 3.9) | Clean venv built from MA-Chunk's pinned versions plus `faiss-cpu`, with CPU torch: numpy 1.26.4, pandas 2.3.3, gymnasium 1.1.1, SB3 / sb3-contrib 2.6.0, scikit-learn 1.5.1, rank-bm25 0.2.2, sentence-transformers 4.1.0, transformers 4.57.6, faiss-cpu 1.13.0, spaCy 3.8.5 (thinc 8.3.4). The existing Python 3.9 conda envs share an incomplete `~/.local` stack, so they were not used. |
| Online (training, evaluation) | `/mnt/ssd1/conda/envs/marllib/bin/python` (Python 3.8, ray 1.8, MARLlib 1.0.3) | No new dependencies. **Requires the RLlib patch links to be valid** (see §5). |

**RL-Chunk and MA-Chunk are separate git repositories. They are not part of BusEnv and nothing from them is committed here.** BusEnv imports them from wherever they are cloned, and uses their code unchanged:

| Repository | Default location | Override with |
|---|---|---|
| RL-Chunk (`RafaelCostaF/RL-Chunk`) | `/mnt/ssd1/rafael/git_rl_chunk/RL-Chunk` | `RL_CHUNK_PIPELINE=<clone>/src/rl_pipeline` |
| MA-Chunk (`RafaelCostaF/MA-Chunk`) | `/mnt/ssd1/rafael/git_rl_chunk/MA-Chunk` | `MA_CHUNK_REPO=<clone>` |

The trained RL-Chunk and MA-Chunk models go to `artifacts/rag/selectors/`, which git ignores. Only the offline steps need these repositories; training BusEnv only reads the finished tables.

The GTFS feed (for stop names) is read from `/mnt/ssd1/wesley/BusEnv/SUNT/tmp/GTFS/BUS`.

```bash
export PYTHONPATH=src
OFF=/mnt/ssd1/rafael/venvs/busenv-rag/bin/python
ON=/mnt/ssd1/conda/envs/marllib/bin/python
```

## 1. Tests

```bash
$ON  -m pytest -q src/tests/test_rag_online.py    # flow cascade == simulator, RAG-off bit-identical, wrapper, day split
$OFF -m pytest -q src/tests/test_rag_offline.py   # pools, labels, featurizer, selectors, RL-Chunk / MA-Chunk smoke
```

## 2. Offline build

```bash
# 2.1 chunks, keys, candidate pools, scores (BM25, multilingual dense + FAISS, structural), L1 labels
$OFF src/pipelines/build_rag_index.py --gtfs-dir /mnt/ssd1/wesley/BusEnv/SUNT/tmp/GTFS/BUS --out artifacts/rag

# 2.2 adaptive selectors, leave-one-route-out CV (test-fold selections)
$OFF src/pipelines/train_rag_selector.py --method rlchunk --out artifacts/rag/selectors          # B5a
$OFF src/pipelines/train_rag_selector.py --method machunk --lams 0.5 1 2 --out artifacts/rag/selectors   # B5b
# ablation A11 (stop names off), optional:
$OFF src/pipelines/train_rag_selector.py --method rlchunk --variant _noplace --out artifacts/rag/selectors
$OFF src/pipelines/train_rag_selector.py --method machunk --variant _noplace --lams 1 --out artifacts/rag/selectors

# 2.3 one context table per arm + offline retrieval metrics
$OFF src/pipelines/export_rag_tables.py --rag-dir artifacts/rag
#   -> artifacts/rag/tables/rag_table_<arm>.npz/.json, offline_retrieval_metrics.csv
```

## 3. Training: choose the retrieval method

Pick the method with `--retrieval`:

```bash
$ON src/pipelines/train_marllib.py --algo mappo --retrieval ma-chunk --day-split train --seed 1
```

| `--retrieval` | What each bus additionally sees | Doc code |
|---|---|---|
| `none` (default) | nothing: the original BusEnv | B0 |
| `zeros` | 10 extra inputs that are always 0 (control for the bigger input) | B0z |
| `lookahead` | a summary of everything about the next 10 stops, no selection | B1 |
| `bm25` | the 6 chunks that best match by keyword search | B2 |
| `dense` | the 6 chunks that best match by meaning (multilingual embeddings + FAISS) | B3 |
| `hybrid` | the 6 best by all scores combined | B4 |
| `rl-chunk` | the chunks RL-Chunk decides to keep | B5a |
| `ma-chunk` | the chunks MA-Chunk decides to keep | B5b |
| `random` | 6 random chunks (control) | B6 |
| `shuffled`, `oracle`, `ma-chunk-no-salience`, `*-no-names` | checks and ablations | A7, A8, A1, A11 |

The list lives in `src/rag/methods.py`. `RAG_RETRIEVAL=<method>` works too, instead of the flag.

Other options:
- `--day-split train|test|none`: simulate 2024 days, 2025 days, or all days (default `none`, the old behaviour).
- `RAG_REWARD_WEIGHTS='{"occ_penalty":0.6,"uptime_bonus":0.4,"sync_score":0.5,"energy_efficiency":0.5}'`: the same reward for every compared run. Without it, the script's original weights are used.
- `SUNT_METRICS_DIR`, `RAY_RESULTS_DIR`, `RAY_TMPDIR`, `TMPDIR`: output folders. The defaults point to `/mnt/ssd1/...` folders this account cannot write to.

With no new options, `train_marllib.py` behaves exactly as before.

To compare several methods at once, `run_rag_grid.sh` trains every method × algorithm × seed, e.g.:
```bash
METHODS="none hybrid rl-chunk ma-chunk random" ALGOS="mappo" SEEDS="1 2 3 4 5" ./run_rag_grid.sh
```

## 4. Deviations from the plan (and why)

1. **Rain and road works are not query-key fields** (doc 04 §4 listed them). No Phase 1 chunk type is conditioned on rain or road works, so including them would only duplicate identical selections. Day type *is* a key field, because segment chunks are aggregated per day type from **training days only**.
2. **Segment chunks use training-day, per-day-type aggregates** of the daily edge travel times, falling back to `travel_time_stats.pkl` and then to the simulator's default. Test days therefore never leak into the context.
3. **RL-Chunk is trained with the same leave-one-route-out protocol as MA-Chunk**, not one model per route trained and tested on the same keys. RL-Chunk's own notes flag train = test as a weakness (`aamas/BARREIRAS.md`). Hyperparameters are unchanged (PPO `MultiInputPolicy`, `n_steps=128`, `ent_coef=0.01`, 40k steps per fold).
4. **MA-Chunk cost unit:** `tokens = 100` per chunk and `token_budget = 100 · K_max`, so λ = 1 costs 0.1 per chunk. With `tokens = 1` the price would be negligible.
5. **sunt_bus.yaml** is vendored at `src/configs/marllib/sunt_bus.yaml` (copied from `/mnt/ssd1/wesley/BusEnv/MARLlib`). `rag/marllib_compat.make_sunt_env` uses it only when MARLlib's own config folder has none.
6. **Corridor chunks** (doc 04 §3.2, optional) are not implemented yet. Stop names are used through the text renderings and the `place` scorer.

## 5. Known blocker: broken RLlib patch links in the `marllib` conda env

MARLlib patches Ray RLlib by **symlinking** six files into `site-packages/ray/rllib/`. In `/mnt/ssd1/conda/envs/marllib` they point to `/mnt/ssd1/rafael/graph-exploration/MARLlib/marllib/patch/...`, which no longer exists (the links date from 2025-10-02). Every RLlib import that touches them fails, e.g. `ModuleNotFoundError: No module named 'ray.rllib.models.preprocessors'`.

The broken links are:
```
ray/rllib/models/preprocessors.py
ray/rllib/policy/torch_policy.py
ray/rllib/policy/rnn_sequencing.py
ray/rllib/execution/train_ops.py
ray/rllib/execution/replay_buffer.py
ray/rllib/utils/exploration/ornstein_uhlenbeck_noise.py
```

Fix: clone upstream MARLlib (commit `80e9973`, the one the old copy used) to a stable place and repoint the six links.
```bash
git clone https://github.com/Replicable-MARL/MARLlib.git /mnt/ssd1/rafael/venvs/MARLlib
git -C /mnt/ssd1/rafael/venvs/MARLlib checkout 80e9973
E=/mnt/ssd1/conda/envs/marllib
for l in $(find $E -xtype l); do
  old=$(readlink "$l"); new=${old/\/mnt\/ssd1\/rafael\/graph-exploration\/MARLlib/\/mnt\/ssd1\/rafael\/venvs\/MARLlib}
  echo "$l | $old -> $new" >> /mnt/ssd1/rafael/venvs/marllib_symlink_repair.log
  ln -sfn "$new" "$l"
done
find $E -xtype l | wc -l     # expect 0
```
