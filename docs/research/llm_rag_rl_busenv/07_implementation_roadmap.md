# 07 — Implementation Roadmap

> Written for the coding agent that will implement Phase 1. Every change lists: **why**, **what file**, **whether it is a separate module**, **complexity** (S ≤ ½ day, M ≤ 2 days, L > 2 days) and **scientific value**. `sunt_env.py`, `DefaultReward`, the action space and the MARLlib configs are **not modified** in Phase 1.

## 0. Two environments, one contract

| Side | Python env | Responsibilities | New dependencies |
|---|---|---|---|
| **Offline** | the MA-Chunk env (Python 3.9, `MA-Chunk/requirements.txt`) | chunks, queries, scoring (BM25, FAISS, numeric), L1 labels, MA-Chunk training and selection, featurization, table export | `faiss-cpu` (only addition; MA-Chunk already has `rank-bm25` and `sentence-transformers`) |
| **Online** | `marllib` conda env (Python 3.8) | `RagObsWrapper` lookup, day split, evaluation | **none** (numpy, pandas only) |

**The contract between the two:** `rag_table_<arm>.npz` (keys `int64[N]` encoded from `QueryKey`, values `float32[N, D]`) plus `manifest.json`. The manifest holds:
- `arm`, `D`, `H`, `K`, `lambda`, `scorers`;
- `featurizer_version`, `key_encoding`;
- `train_days`, `test_days`;
- the sha256 of every input artifact;
- the MA-Chunk commit `3948cc7`;
- the BusEnv commit.

## 1. Phase 1 steps (in order)

### Step 1 — `src/rag/keys.py` (shared, pure Python/numpy) · **S**
- **Why:** both sides must encode `QueryKey = (trip_id, route_idx, going_forward, hour, day_type, rain, obras)` the same way.
- **What:**
  - `encode(key) -> int64` and `decode`;
  - `key_from_env(env, agent)`, which reads `agent_route_id`, `agent_states`, `agent_times`, `date`, `climate_data` and `obras_*`, mirroring `sunt_env.py:710–745`;
  - `day_type(date)`.
- **Separate module:** yes.
- **Value:** a single source of truth; prevents train/serve skew.
- **Test:** round-trip; `key_from_env` on a fresh `reset()` matches a hand-computed key.

### Step 2 — `src/rag/chunks.py` + `src/pipelines/build_rag_chunks.py` · **M**
- **Why:** build the three chunk types of doc 04 §3, with provenance.
- **What:**
  - `stop_hour` chunks must call **the same lookup cascade** as `parallel_env._get_flow_stats`, either by importing it or by copying it with a unit test that compares the two on 1,000 random keys. That includes the forward-only rule for route levels.
  - `segment` chunks from `travel_time_stats.pkl`; `stop_activity` from `hourly_stop_activity.pkl`.
  - **Place fields** from GTFS `stops.txt` / `routes.txt` (doc 04 §3.2): `street`, `neighborhood`, `cep`, `landmark` (regex parse, keeping the unparsed `stop_name`), plus `lat`/`lon` from the graph. Optional `corridor` chunks aggregated by `street`.
  - `z_route_hour`, Portuguese `text` rendering (with and without place fields, for ablation A11), numeric `emb`.
  - Output `artifacts/rag/chunks.parquet`.
- **Separate module:** yes.
- **Value:** reproducible, traceable knowledge base.
- **Test:**
  - every chunk has non-null `provenance`;
  - the stop_hour values equal `env._get_arrival_rate` / `_get_alight_frac` for the same key;
  - the place-field parse rates are ≥ the figures in doc 04 §3.2 (219 / 153 / 148 of 225).

### Step 3 — `src/pipelines/build_rag_queries.py` · **S**
- **Why:** enumerate every reachable `QueryKey` for the configured routes and hours (doc 04 §4, about 105k), each with its candidate set C(q) (horizon H, ETA hours).
- **What:** output `artifacts/rag/queries.parquet` (key, ETA per hop) and `pools/<H>.parquet` (one row per (key, candidate)).
- **Separate module:** yes.
- **Test:** |C(q)| ≈ 4H on average; candidates follow `_future_route_indices` order.

### Step 4 — `src/rag/scorers.py` · **M**
- **Why:** these are the MA-Chunk "agents" and the baselines B2–B4.
- **What:** write `s_<scorer>` and `rank_<scorer>` columns, plus `rrf`, for:
  - `bm25` (`rank_bm25`, accent-folded);
  - `dense` (multilingual sentence-transformers + FAISS `IndexFlatIP`);
  - `place` (street/neighbourhood continuity);
  - `proximity`, `eta`, `salience`, `support`.
- **Separate module:** yes (offline env only).
- **Value:** heterogeneous signals; per-scorer AUC is itself a reportable result.
- **Test:** ranks are a permutation within each key; the dense scorer is deterministic (fixed model revision).

### Step 5 — `src/rag/labels.py` (L1) · **S**
- **Why:** an LLM-free `useful` / `n_support` label for MA-Chunk training and retrieval metrics (doc 02 §4.3).
- **Separate module:** yes.
- **Test:** `n_support ≥ 1` for every key; the label is independent of every scorer except salience (check the correlations).

### Step 6a — RL-Chunk training and selection: `src/pipelines/train_rag_selector.py --method rlchunk` · **S–M**
- **Why:** single-agent adaptive selector, arm B5a (doc 02 §2.1).
- **What:**
  - Install RL-Chunk from `git_rl_chunk/RL-Chunk` (pin the commit; record its hash in the manifest).
  - Subclass `rl_pipeline.env.ChunkSelectionEnv` as `BusChunkSelectionEnv`, overriding **only** `_chunks_and_scores`. It returns the candidate chunk ids in route order, the dense similarity scaled to [0, 1] per key, and the BM25 score normalized by the per-key maximum.
  - Reuse `train.py`'s SB3 construction: PPO main, R-PPO optional; 40k steps; one model per route (`trip_id`).
  - Run `env.run_episode` for deterministic selection on all keys.
  - Set `OPENAI_API_KEY=unused`, because `config.py:30–34` requires it at import; no OpenAI call happens.
- **Separate module:** yes (offline).
- **Test:**
  - the chunk count is not constant across keys (std > 0); this guards against the "always 10" collapse recorded in `BARREIRAS.md` B1;
  - its first-hop picks are not 100% of episodes.

### Step 6b — MA-Chunk training and selection: `src/pipelines/train_rag_selector.py --method machunk` · **M**
- **Why:** multi-agent adaptive selector, arm B5b.
- **What:**
  - Import `ma_chunk.env_ma.QueryData` / `MAChunkEnv` and `ma_chunk.train_ma.make_model` / `rollout` from the MA-Chunk repository, pinned at `3948cc7` and installed with `pip install -e`. **Do not copy or fork MA-Chunk.**
  - `agents = [bm25, dense, place, proximity, eta, salience, support]`, `tokens = 1` per chunk, `token_budget = max_chunks = K_max`.
  - 5-fold CV grouped by `trip_id`; λ grid {0.1, 0.3, 1.0}; save the fold models.
  - Run deterministic selection on all keys.
- **Separate module:** yes (offline).
- **Value:** the core of the method.
- **Test:** reproduce MA-Chunk's own `tests/` on the new pool schema; the mean number of chunks is in [1, K_max] and not constant (guards against the RL-Chunk degeneration recorded in `BARREIRAS.md` B1).

### Step 7 — `src/rag/featurize.py` + `src/pipelines/export_rag_tables.py` · **S**
- **Why:** the fixed D = 10 pooled featurizer (doc 04 §7), shared by all arms B1–B6. Normalization constants are taken from **training keys** only.
- **What:** write `rag_table_<arm>.npz` and `manifest.json` for B0z (zeros), B1, B2, B3, B4, B5a, B5b and B6.
- **Separate module:** yes; pure numpy, importable from both environments.
- **Test:** all values are finite and within [−1, 1] or [0, 1]; B0z is all zeros; the tables have the same key set.

### Step 8 — `src/rag/wrapper.py` (`RagObsWrapper`) · **S**
- **Why:** the only online integration point (doc 01 §9, option A).
- **What:**
  - Subclass `pettingzoo.utils.wrappers.BaseParallelWraper`.
  - Override `observation_space(agent)` → `Box(12+D)` and keep `_observation_spaces` consistent (doc 04 §8).
  - In `reset` / `step`, concatenate `table[key_from_env(env, a)]`; a missing key gives zeros and increments `infos[a]["rag_miss"]`.
  - Record the lookup latency.
- **Separate module:** yes.
- **Test:**
  - With `mode="off"`, the env is unwrapped and the observations are bit-identical to the original over a 500-step rollout with a fixed seed and actions.
  - With `mode="b0z"`, the first 12 dims are equal to the original and the rest are 0.
  - The `pad_observations_v0` output shape is (12+D,).

### Step 9 — `src/rag/day_split.py` · **S**
- **Why:** train/test day split (doc 06 §3) without editing `sunt_env.py`.
- **What:** `apply_split(env, split="train"|"test", manifest)` filters `env.daily_files`, resets `current_day_index = 0` and sets `total_days`.
- **Separate module:** yes.
- **Test:** after `apply_split("test")`, the first `reset()` loads 2025-01-01 and no 2024 file is ever loaded.

### Step 10 — Edits to `src/pipelines/train_marllib.py` · **S** (about 15 lines)
- **Why:** turn RAG on or off per run.
- **What:**
  - Read `RAG_MODE` and `RAG_TABLE` (env vars, or `env_config`).
  - In `RLlibSuntBus.__init__`, wrap `parallel_env` **before** `pad_observations_v0` (`train_marllib.py:126`) and call `apply_split(..., "train")`.
  - Make the reward weight vector and the tag consistent (`:97–98`) and log both.
  - Include `RAG_MODE`, the table hash and the seed in `run_tag` and the CodeCarbon project name.
- **Separate module:** no; a minimal edit to an existing file.
- **Value:** experiment switch; `RAG_MODE=off` reproduces today's behaviour.

### Step 11 — New `src/pipelines/evaluate_rag.py` · **M**
- **Why:** `evaluate_marllib.py` is not a real evaluator. It is a copy of the training script with `num_agents=20` (doc 06 §8). A held-out evaluation is needed.
- **What:**
  - Restore a MARLlib checkpoint (see `generate_eval_replay.py:87–118` for the RLlib restore pattern; adapt it to MARLlib's policy naming).
  - Build the same wrapper stack with `apply_split("test")` and `record_replay=True`.
  - Run deterministically over all 39 test days.
  - Write the per-day and per-context metrics of doc 06 §4.1.
- **Separate module:** yes.

### Step 12 — `src/tools/rag_analysis.py` · **S**
- **Why:** waiting time, headway coefficient of variation and fleet utilization from replays; AUC (reuse `calc_media_auc_porRun.py`); Wilcoxon + Holm, Friedman + Nemenyi and CD diagrams.
- **Separate module:** yes.

### Step 13 — Configuration and reproducibility · **S**
- Document (or vendor) `sunt_bus.yaml` (doc 01, warning box in §1).
- Add `docs/research/llm_rag_rl_busenv/RUNBOOK.md` with the exact commands for both environments.
- Do not hard-code new absolute paths: every new module takes paths as arguments.

## 2. Effort summary

| Steps | Complexity | Notes |
|---|---|---|
| 1, 3, 5, 7, 8, 9, 10, 12, 13 | S each | |
| 2, 4, 6a, 6b, 11 | M each (6a S–M) | 6a and 6b are mostly wiring existing RL-Chunk and MA-Chunk code |
| **Phase 1 total** | about **2–3 weeks** of focused work, plus about 2–3 days of compute (doc 06 §7) | |

Optional step: fix the `occ_risk` key bug (doc 01 §4, `sunt_env.py:1599`). It is **S**, but it changes the baseline, so apply it either to all arms or to none (decision **[Q1]**).

## 3. Phase 2 (only if Phase 1 supports H1 and H2/H3)

| Step | What | Complexity |
|---|---|---|
| P2.1 | `src/rag/llm_context.py`: fixed prompt over the selected chunks (+ GTFS names) → JSON schema (doc 05 §3); cache keyed by the hash of the selected-chunk set; temperature 0; store prompt, output and model id | M |
| P2.2 | Validation: invalid-JSON rate, agreement with numeric features, a 50-key manual audit for hallucinations | S |
| P2.3 | Export `rag_table_b7.npz` (D + D′); run arm B7 and ablation A10 | S |

## 4. Phase 3 (only with evidence)

| Extension | Change needed | Why it is deferred |
|---|---|---|
| Dynamic query (occupancy and headway bins in the key) | key encoding, table size × bins | cannot be fully precomputed if continuous |
| L2 labels (downstream-outcome attribution) | rollout logging of B0 | couples labels to a specific policy |
| Shared or fleet-level retrieval (one channel per route; buses as MA-Chunk agents) | new environment design | changes the MA-Chunk formulation |
| Joint training (selector reward = MARL return) | two-level loop across incompatible stacks | non-stationarity, engineering cost |
| Day-specific demand in the simulator (per-day flow stats) | `sunt_env.py` data loading | makes day-analog retrieval meaningful (doc 01 §10) |
| Graph-based retrieval (shared stops across routes, GTFS graph neighbourhood) | candidate generator | needs evidence that cross-route context matters |
| MA-Chunk gate as "use context or not" bandit | small extra model | only useful if context sometimes hurts |
| New actions (stop skipping, dispatch) | action space + transition | changes the research objective |

## 5. Acceptance checklist for Phase 1

- [ ] `RAG_MODE=off` gives bit-identical observations and rewards to the unmodified BusEnv (Step 8 test).
- [ ] All arms B0z–B6 share key sets, D and featurizer version (manifest check).
- [ ] No 2025 data enters chunk statistics used for training-time normalization or the selector (manifest `train_days`).
- [ ] RL-Chunk and MA-Chunk each select a variable number of chunks (std > 0) and achieve above-random P@K on test keys.
- [ ] Online lookup latency is < 50 µs/agent-step, and RSS growth per worker is < 50 MB.
- [ ] 10 seeds × 2 algorithms × 9 arms trained and evaluated on 39 test days; statistics produced per doc 06 §6.
- [ ] Every commit message references the project card (§6).

## 6. Version control and project tracking

- **Branch:** all research and Phase 1 work goes on `Research-RAG-BusEnv` in `LabIA-UFBA/BusEnv`.
- **Project card:** every push must be linked to the GitHub Projects card, issue **`kunumi-labia-ufba/opt_repo#68`**: <https://github.com/orgs/kunumi-labia-ufba/projects/4/views/1?pane=issue&itemId=253911103&issue=kunumi-labia-ufba%7Copt_repo%7C68>.
  - Because the issue lives in a **different repository** (`opt_repo`) from BusEnv, every commit message must end with the **full cross-repository reference**: `Refs kunumi-labia-ufba/opt_repo#68`. A bare `#68` would point to BusEnv's own issue 68.
  - Use `Refs`, not `Closes` or `Fixes`, so the card is not closed automatically on merge.
  - Optionally, link the branch or its pull request in the issue's **Development** sidebar.
- **Who commits:** coding agents prepare changes and propose the commit message. The **user** commits and pushes (workspace rule in `Graph_sunt_project/CLAUDE.md`).
- **What not to commit:** generated artifacts (`artifacts/rag/`, tables, Ray results, replays). Commit the code that generates them and the `manifest.json` schema.
