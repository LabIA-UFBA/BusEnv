# Phase 1 — Implementation Status and Offline Results

> Status on 2026-11-13. Commands: `RUNBOOK.md`. Design: docs 04, 06, 07. Artifacts (git-ignored): `artifacts/rag/`.

## 1. What is implemented and verified

| Step (doc 07) | File(s) | Status |
|---|---|---|
| 1 Keys | `src/rag/keys.py` | ✅ round-trip and bounce logic tested against `parallel_env._future_route_indices` |
| 2 Chunks + places | `src/rag/chunks.py`, `src/rag/places.py` | ✅ flow cascade equal to `parallel_env._get_flow_stats` on 1,000 random cases; every chunk has provenance |
| 3 Queries / pools | `src/pipelines/build_rag_index.py` | ✅ 26,352 keys, 14,410 chunks, 1,022,220 candidate rows (38.8 per key), built in 5 min |
| 4 Scorers | `src/rag/scorers.py`, `chunks.add_structural_scores` | ✅ BM25 (global IDF), multilingual dense (`paraphrase-multilingual-MiniLM-L12-v2`) with FAISS `IndexFlatIP` + `IDSelectorBatch`, place, proximity, ETA, salience, support, RRF |
| 5 L1 labels | `src/rag/labels.py` | ✅ 17.2% of candidates useful; mean n_support 6.7; 3 keys with no useful chunk |
| 6a RL-Chunk | `src/rag/selectors.py`, `train_rag_selector.py --method rlchunk` | ✅ `ChunkSelectionEnv` subclass (only `_chunks_and_scores` overridden), leave-one-route-out, 40k PPO steps per fold |
| 6b MA-Chunk | same, `--method machunk` | ✅ `MAChunkEnv` + `make_model` + `rollout` imported from MA-Chunk unchanged; 7 scorer agents, `share_scores`; λ ∈ {0.5, 1, 2}; 200k PPO steps per fold |
| 7 Featurizer + tables | `src/rag/featurize.py`, `src/rag/table.py`, `export_rag_tables.py` | ✅ D = 10, normalization shared by all arms, sha256 per table |
| 8 Wrapper | `src/rag/wrapper.py` | ✅ RAG off is bit-identical; zero table keeps the first 12 dims; SuperSuit pads to 22; real tables: 0 misses, about 6.5 µs per lookup |
| 9 Day split | `src/rag/day_split.py` | ✅ train = 144 days (2024), test = 39 days (2025) |
| 10 `train_marllib.py` switch | `--retrieval <method>` (list in `src/rag/methods.py`), `--day-split`, `--seed`, `--timesteps`, `--num-workers`, `--local-dir`; env vars `RAG_REWARD_WEIGHTS`, `SUNT_METRICS_DIR`, `RAY_RESULTS_DIR` | ✅ written; defaults reproduce the old behaviour. ⛔ **training not yet run** (§3) |
| 10b MARLlib config fallback | `src/rag/marllib_compat.py`, `src/configs/marllib/sunt_bus.yaml` | ✅ written; ⛔ not exercised (§3) |
| 11 Held-out evaluator | `src/pipelines/evaluate_rag.py` | ⏳ **not written yet.** MARLlib's own restore path forces a render-mode evaluation that BusEnv does not support, so a manual restore and rollout is needed, and that can only be tested once RLlib imports work (§3) |
| 12 Analysis | `src/tools/rag_analysis.py` | ⏳ pending (needs training runs) |
| Grid launcher | `run_rag_grid.sh` | ✅ written (syntax-checked); not run |
| Tests | `src/tests/test_rag_online.py` (10), `src/tests/test_rag_offline.py` (6) | ✅ all passing |

## 2. Offline results (held-out routes, against L1 labels)

Selections come from leave-one-route-out cross-validation: each key was selected by a model that never saw its route. Equal-budget K = 6, the median of the better adaptive arm (MA-Chunk).

| Arm | Selector | Chunks/key (mean ± sd) | Precision | Recall (utility) | F1 |
|---|---|---|---|---|---|
| B1 | all candidates (look-ahead) | 38.8 ± 2.7 | 0.172 | 1.000 | 0.293 |
| B2 | BM25 top-6 | 6 | 0.254 | 0.228 | 0.241 |
| B3 | dense (FAISS) top-6 | 6 | 0.249 | 0.214 | 0.230 |
| B4 | hybrid RRF top-6 | 6 | 0.471 | 0.440 | 0.455 |
| **B5a** | **RL-Chunk** | 8.9 ± 1.9 (median 10) | 0.260 | 0.331 | 0.291 |
| **B5b** | **MA-Chunk** (λ = 0.5) | 6.5 ± 1.6 | **0.820** | **0.834** | **0.827** |
| B6 | random 6 | 6 | 0.172 | 0.155 | 0.163 |
| A8 | oracle (labels) | 6.6 ± 2.2 | 1.000 | 0.991 | 0.996 |

MA-Chunk λ grid: λ = 0.5 → 6.5 chunks, F1 0.827 (chosen); λ = 1 → 4.3 chunks, P 0.927, F1 0.774; λ = 2 → stops at once (0 chunks).

**Per-scorer AUC for `useful`** (as in MA-Chunk's `BARREIRAS.md` B9):

| salience | RRF | proximity | ETA | BM25 | support | place | dense |
|---|---|---|---|---|---|---|---|
| 0.936 | 0.780 | 0.702 | 0.610 | 0.609 | 0.580 | 0.556 | 0.524 |

**Stop names (ablation A11, offline):**

| | with names | names off |
|---|---|---|
| dense AUC | 0.524 | 0.403 |
| BM25 AUC | 0.609 | 0.648 |
| dense top-6 F1 | 0.230 | 0.085 |
| BM25 top-6 F1 | 0.241 | 0.284 |
| RL-Chunk F1 | 0.291 | 0.141 |
| MA-Chunk F1 (λ = 1) | 0.774 | 0.777 |

**Reading [I]:**
- **MA-Chunk beats every other selector by a wide margin offline, with a variable budget.** Most of its advantage comes from the `salience` scorer, which is close to the L1 label definition (AUC 0.936). That is exactly the circularity flagged in doc 02 §4.3, and ablation A1 measures it (§2.1). The offline numbers measure agreement with a proxy label. **Only the downstream MARL experiment shows whether any of this helps bus control.**
- **RL-Chunk reproduces its QA behaviour**: it mostly fills the budget (median 10). Its only signals are the text similarities, which are weak here.
- **Stop names matter for the text-based selectors.** They roughly double RL-Chunk's and dense retrieval's agreement with the labels, slightly hurt BM25, and make no difference to MA-Chunk, which leans on the structural scorers.

### 2.1 A1 (MA-Chunk without salience)

| Arm | Chunks/key | Precision | Recall | F1 |
|---|---|---|---|---|
| B5b MA-Chunk (all 7 scorers, λ = 0.5) | 6.5 ± 1.6 | 0.820 | 0.834 | 0.827 |
| **A1 MA-Chunk without salience** (λ = 0.5) | 5.2 ± 0.5 | 0.435 | 0.352 | 0.389 |
| B4 hybrid RRF top-6 (includes salience) | 6 | 0.471 | 0.440 | 0.455 |
| B5a RL-Chunk | 8.9 | 0.260 | 0.331 | 0.291 |

**Reading [I]:**
- Most of MA-Chunk's offline margin comes from the salience scorer, i.e. from the label definition, as anticipated.
- Without it, MA-Chunk still beats RL-Chunk, BM25, dense and random, with fewer chunks. It does not beat a fixed RRF that includes salience.
- This makes the downstream MARL comparison essential. The `a1` table is exported, so A1 can also be run downstream.

## 3. Blocker: the `marllib` conda env cannot import RLlib

`/mnt/ssd1/conda/envs/marllib` has six RLlib files that MARLlib installs as **symlinks** into `site-packages/ray/rllib/`. They point to `/mnt/ssd1/rafael/graph-exploration/MARLlib/...`, which no longer exists; the links date from 2025-10-02. Any MARLlib training fails with `ModuleNotFoundError: No module named 'ray.rllib.models.preprocessors'`.

The fix is in `RUNBOOK.md` §5: clone upstream MARLlib at `80e9973` and repoint the six links. It changes files inside the conda environment, so it was **not** applied automatically and needs the user's approval. Until then the following are blocked:
- the training smoke test;
- `evaluate_rag.py` (manual checkpoint restore and test-day rollout);
- the B0 vs arms grid.

## 4. Next steps

1. Fix the RLlib links (user action or approval).
2. Smoke-train MAPPO for 3k steps on b0z, then on b5b, to check checkpoints and metrics.
3. Write and test `evaluate_rag.py` on the smoke checkpoint.
4. Run the minimum experiment (doc 06 / proposal §8 Q7): B0, B1, B4, B5a, B5b, B6, A7 × MAPPO × 10 seeds, with `run_rag_grid.sh`.
5. `rag_analysis.py`: AUC, test-day metrics, Wilcoxon + Holm, Friedman + Nemenyi.
