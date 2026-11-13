# 06 — Experimental Protocol

> Workstream F. **[V]** = verified, **[I]** = design, **[Q]** = open question.

## 1. Hypotheses

- **H1 (context helps).** Adding retrieved downstream context to the observation improves MARL performance over the original BusEnv (B0).
- **H2 (selection helps).** At equal average feature budget, **adaptive selection (RL-Chunk B5a and/or MA-Chunk B5b)** beats fixed top-K retrieval (B2–B4) and random selection (B6).
- **H3 (retrieval vs look-ahead).** The adaptive arms match or beat the **direct look-ahead** control (B1), which uses all H downstream chunks with no selection, while using fewer chunks.
- **H4 (Phase 2 only).** Offline LLM features add value beyond the numeric features they were derived from.
- **H5 (multi-agent vs single-agent selection).** MA-Chunk (B5b) beats RL-Chunk (B5a) in downstream MARL performance and/or uses fewer chunks for the same performance. This tests whether MA-Chunk's documented QA advantage (doc 02 §3.2) carries over to a control setting.

H2 and H3 are what make the work a contribution about *retrieval and selection* rather than "more features". **A result where B1 ≈ B5a ≈ B5b > B0 is informative and must be reported as such**: context helps, but selection does not.

## 2. Arms

Every arm except B0 uses the same wrapper, the same featurizer and the same D = 10 (doc 04 §7). Only the selection differs.

| Arm | Observation | Selection |
|---|---|---|
| **B0** | original `Box(12)` | — (no wrapper; bit-identical to current BusEnv) |
| **B0z** | `Box(12+D)`, context = zeros | — (controls for input width and initialization) |
| **B1** | `Box(12+D)` | **all** chunks in C(q) (direct look-ahead, no retrieval) |
| **B2** | `Box(12+D)` | BM25 top-K |
| **B3** | `Box(12+D)` | dense (MiniLM + FAISS `IndexFlatIP`) top-K |
| **B4** | `Box(12+D)` | hybrid RRF top-K |
| **B5a** | `Box(12+D)` | **RL-Chunk** (SB3 PPO, single agent, route-order walk, label-free graduated reward, one model per route; doc 02 §2.1) |
| **B5b** | `Box(12+D)` | **MA-Chunk** (PPO, scorers as agents, `share_scores=True`, λ chosen on validation keys, variable ≤ K_max) |
| **B6** | `Box(12+D)` | random K from C(q) (fixed seed per key) |
| B7 (Phase 2) | `Box(12+D+D′)` | best of B5a/B5b + cached LLM features |

K = the median number of chunks selected on training keys by the better adaptive arm on validation keys (equal-budget rule, doc 04 §6).

## 3. Fixed experimental conditions

These conditions are pinned explicitly because the current scripts are ambiguous:

| Setting | Value | Reason |
|---|---|---|
| Routes and agents | the 5 `manual_route_groups` trip_ids, 25 agents (5 per route) | as in `train_marllib.py:102–125` |
| Reward | `DefaultReward` with an **explicitly recorded** weight vector **[Q3 in doc 01]**; recommended: the class defaults `{0.6, 0.4, 0.5, 0.5}` (`sunt_env.py:2273–2278`) so that all four objectives are active | the script's weights and tag currently disagree (`train_marllib.py:97–98`) |
| `reward_scalarization_mode` | `"weighted"` (Phase 1); `"hybrid_lexico"` as a robustness check | |
| `use_rain` | **[Q]** False (current default) or True. Must be the same for all arms; if False, the query's rain field is constant | |
| `occupancy_source` | `"real"` | demand model of doc 01 §6 |
| Algorithms | **MAPPO** and **IPPO** (MARLlib, `hyperparam_source="common"`, `core_arch="mlp"`, `encode_layer="128-128"`) | PPO family, the team's most used algorithms; IPPO has no centralized critic, which tests whether the context matters per agent |
| Budget | `timesteps_total = 200000` (current), **[Q]** extend to 500k if learning curves have not plateaued in B0 | |
| Seeds | **10** per (arm, algorithm), matching the team's run1–run10 convention (`codecarbon/emissions.csv`); minimum 5 | |
| Day split | **train**: 2024-03-01 → 2024-12-31 (144 days); **test**: 2025-01-01 → 2025-03-14 (39 days) [V counts] | chronological; no test day is seen in training |
| Day restriction mechanism | the wrapper or `RLlibSuntBus` sets `env.daily_files` / `env.total_days` to the split after construction (no change to `sunt_env.py`) | `sunt_env.py:161–167` reads the folder at init |
| MA-Chunk selector | trained on **training-period keys** only, 5-fold CV by `trip_id`, λ chosen on held-out folds | no leakage of the selector into test |

**Context shift to report [V].** The Batatinha road-works window ends 2024-08-30 and Duda Mendonça ends 2025-03-28 (`sunt_env.py:304–308`). In the test period, therefore, only routes `20041_1230_10` and `30013_120_10` are in the "obras" context. Results must be reported per context.

**Caveat about what the "history" is [V/I].** Demand statistics (`stop_passenger_flow.pkl`) are aggregated over the whole period, and the simulator itself uses them (doc 01 §6). Inside the simulator, retrieved demand chunks are therefore *the simulator's own expected values*: the experiment tests whether agents benefit from access to the right part of the model's knowledge. A **realism check** (optional, Phase 1.5) breaks this identity:
- rebuild chunks from training-period OD only;
- run the test-period simulator with flow statistics rebuilt from test-period OD;
- `generate_passenger_flow_stats.py --base-path` points at a folder with only those days' parquet (symlinks).

## 4. Metrics

### 4.1 Transportation (evaluation on the 39 test days, deterministic policy)

| Metric | How it is measured | Already available? |
|---|---|---|
| Occupancy score, uptime score, sync score, efficiency score | per context, from `episode_metrics_<run>.csv` (`sunt_env.py:1432–1458`) | **yes** [V] |
| % steps in the ideal occupancy band | same CSV (`Occupancy Percentage`) | **yes** [V] |
| **Passenger waiting (queue) at visited stops** | mean of `waiting` in replay logs (`record_replay=True`, `sunt_env.py:1309–1319`); waiting time ≈ queue / arrival rate (Little's law) | logged, needs an analysis script |
| **Headway regularity** | coefficient of variation of consecutive same-route arrival gaps per stop, from replay timestamps | logged, needs an analysis script |
| Leader/follower headway deviation | mean \|obs features 10–11\| over steps | computable in the wrapper |
| Fleet utilization | share of WAIT / MOVE / SERVICE_CENTER actions, from replay `action` | logged |
| Average travel time | mean MOVE edge time against the expected time (efficiency component) | yes (efficiency) |
| Demand satisfaction | boarded / (boarded + left waiting at end of day). **Needs** a small eval-only logger of `boarding` (`sunt_env.py:844`); the wrapper cannot separate boarding from alighting | **no**: optional, eval-only |

### 4.2 Reinforcement learning

| Metric | Source |
|---|---|
| Episode reward (mean, final) | RLlib `progress.csv` (`episode_reward_mean`) |
| **AUC of the learning curve** (normalized) | existing tools `src/tools/calc_media_auc_porRun.py` and `calc_media_auc_porContext.py` [V exist] |
| Convergence speed | timesteps to reach 90% of B0's final reward |
| Stability | std across seeds; std of the last 20% of the curve |
| Sample efficiency | AUC at 50k / 100k / 200k timesteps |
| Robustness | metrics split by context (normal/obras/rain), hour band (peak 06–09 and 16–19 vs off-peak), day type |

### 4.3 Retrieval and compute

| Metric | Where |
|---|---|
| P@K, R@K, nDCG@K, utility U, against L1 labels on **test keys** | offline (`baselines.py`-style evaluation) |
| Per-scorer AUC for `useful` (as `BARREIRAS.md` B9) | offline |
| Mean / std number of selected chunks; STOP rate | offline |
| Offline build time (scoring, MA-Chunk training, selection) | offline |
| Table size; online lookup latency (µs/step, timed in the wrapper) | online |
| Peak RSS per Ray worker; CPU time; energy | CodeCarbon, already integrated in `train_marllib.py:22–58` [V] |
| Tokens, cost, invalid-JSON rate (Phase 2) | offline LLM cache |

## 5. Ablations

| ID | Ablation | Question |
|---|---|---|
| A1 | B5b without the `salience` scorer | Is MA-Chunk just re-deriving the L1 label? |
| A2 | B5b with only structural scorers (proximity, ETA, salience, support, place) | Do lexical and dense scorers add anything? (MA-Chunk's "misleading agent" analysis) |
| A3 | H ∈ {5, 10, 20} | Does selection matter more as the candidate pool grows? |
| A4 | λ ∈ {0.1, 0.3, 1.0} | budget vs. performance trade-off |
| A5 | slot featurizer (K_max × fields) instead of pooled | Does the representation limit the gain? |
| A6 | supervised selector (`sup_select.py` logreg + greedy) instead of PPO | Is RL needed for selection? |
| A7 | **context from a shuffled key** (permutation control) | Does the *information* matter, or only the extra input? |
| A8 | oracle selection from L1 labels | upper bound of selection |
| A9 | MA-Chunk `single` (RRF, one agent) vs `ma` vs `ma_msg` | Does the multi-agent selection structure help here? |
| A10 | Phase 2: LLM-only vs numeric + LLM vs numeric-only | H4 |
| A11 | **Stop names off**: chunk and query text rendered without place fields (doc 04 §3.2, §5.1); `place` scorer removed | Do stop and place names improve BM25, dense, RL-Chunk and MA-Chunk retrieval quality and downstream MARL? |
| A12 | RL-Chunk with the L1-label reward (`useful − cost`) instead of its similarity reward | Is a B5a vs B5b gap due to the architecture or to the reward? |
| A13 | RL-Chunk walk order: route order vs hybrid-rank order | Does RL-Chunk's sequential walk benefit from BusEnv's natural ordering? |

**Priority if compute is limited:** A7 and A8 first (validity), then A6, A1 and A11, then the rest. A11–A13 can first be assessed **offline** (retrieval metrics against L1 labels); run the full MARL comparison only if the offline difference is meaningful.

## 6. Statistical analysis

- **Unit:** one (arm, algorithm, seed) training run, evaluated on the 39 test days. Use per-seed means over days for the tests; keep per-day values for the plots.
- **Primary comparisons:** each adaptive arm (B5a, B5b) against each of {B0, B1, B4, B6}, plus **B5b vs B5a** (H5). Use the **Wilcoxon signed-rank test paired by seed** on the AUC and on the test-period scalarized reward, with **Holm** correction. Report Hodges–Lehmann shift, Cliff's δ and bootstrap 95% CIs.
- **All arms:** Friedman test with Nemenyi post-hoc and a critical-difference diagram (blocks = algorithm × seed). This matches the statistics the team already uses (`friedman_nemenyi_from_tfevents.py` and the CD-diagram tooling in the former graph-exploration repository; also the MA-Chunk skill's Friedman/Nemenyi/Bonferroni–Dunn plots).
- **Multiple metrics:** declare **one primary endpoint** in advance: AUC of the scalarized reward. Treat the transportation metrics as secondary, with Holm correction within each family.
- **Power note [I]:** with 10 seeds, a paired Wilcoxon test detects only large effects (the minimum two-sided p is about 0.002). If effects are small, increase the seeds before adding arms.

## 7. Compute budget [I]

- Core grid: 9 arms (B0, B0z, B1, B2, B3, B4, B5a, B5b, B6) × 2 algorithms × 10 seeds = **180 runs**.
- At about 0.25–0.4 h per run with 10 workers (`codecarbon/emissions.csv` median 0.18 h for 200k steps), that is about **45–72 h** sequentially on the 64-core machine, or about 15–25 h running 3 at a time.
- Offline retrieval and selection add under 2 h in total.
- Ablations A1–A9 at 5 seeds × MAPPO only add about 45 runs.

## 8. Procedure (step by step)

1. Freeze the BusEnv commit and record the MARLlib config hash (`sunt_bus.yaml`) and the reward weights.
2. Build `chunks.parquet` and `queries.parquet` from **training-period** artifacts. For the simulator-identical setting, these are the existing pickles.
3. Score candidates, build L1 labels, train RL-Chunk (one PPO model per route) and MA-Chunk (5-fold by trip_id, choose λ), and produce `rag_table_<arm>.npz` and `manifest.json` (hashes, H, K, λ, scorer list) for each arm.
4. Report the offline retrieval metrics (§4.3) on test keys.
5. Train every (arm, algorithm, seed) on training days with CodeCarbon on.
6. Evaluate every checkpoint on test days with `record_replay=True` using a **new evaluation script**. The current `evaluate_marllib.py` is a training-script copy: its parser says "Train MARLlib algorithms" and it uses `num_agents=20` against 25 in training (`evaluate_marllib.py:100`, `:289`) [V].
7. Compute the metrics and statistics; produce CD diagrams, learning curves and per-context tables.
8. Run Phase 2 only if H1 and (H2 or H3) hold.
