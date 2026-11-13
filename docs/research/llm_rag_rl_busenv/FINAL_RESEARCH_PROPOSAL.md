# Final Research Proposal — Adaptive Retrieval (RL-Chunk and MA-Chunk) as Decision Context for Multi-Agent Bus Control in BusEnv

> Consolidates docs 01–07 in this folder. **[V]** = verified in code, data or primary sources; **[I]** = interpretation or recommendation; **[Q]** = open question. Date: 2026-11-13. BusEnv branch `retrieval-augmented-rl` @ `d2b6a8f`; MA-Chunk @ `3948cc7`.

## 1. Motivation

BusEnv agents decide **when to hold** (WAIT vs MOVE) and **when to go to maintenance**. They make these decisions from a 12-feature observation that covers only the **current and next stop** [V] (doc 01 §4).

The simulator's reward, however, depends on what happens **downstream**: queues build up at the next stops at arrival rates that depend on route, stop and hour, and the next edges have their own travel times [V] (doc 01 §6). SUNT history already contains this information, aggregated in the pickles BusEnv ships. The policy never sees it.

At the same time, the user's two adaptive chunk selectors learn *how much* and *which* context to pass downstream:
- **RL-Chunk** (single agent): a sequential SELECT/SKIP walk with a label-free similarity reward [V] (doc 02 §2).
- **MA-Chunk** (AAMAS 2027 code release): retrievers as agents, a STOP action, budget pricing, redundancy awareness, and training from per-chunk labels without an LLM [V] (doc 02 §3).

Both can be reused with small adapters (doc 02 §2.1, §4). Comparing them in a control setting is part of the study.

The literature review found **no work in which a learned adaptive retriever feeds a control policy rather than an LLM reader**, and none that uses retrieval to build observations for multi-agent bus control (doc 03 §10, gaps 1–2). The closest work, Dong & Gayah (arXiv:2608.10207), adds **static** offline LLM stop embeddings to a single DQN holding controller.

## 2. Research question and hypotheses

> Does query-specific, adaptive selection of SUNT-derived context, appended to each bus agent's observation, improve multi-agent bus control in BusEnv? Is the gain due to *selection*, not just to *more information*?

- **H1:** context improves over the original environment (B5a/B5b > B0).
- **H2:** adaptive selection beats fixed top-K and random selection at equal average budget (B5a/B5b > B2, B3, B4, B6).
- **H3:** adaptive selection matches or beats direct look-ahead of all downstream chunks while using fewer of them (B5a/B5b ≥ B1).
- **H4 (Phase 2):** offline LLM features add value beyond the numbers they are derived from.
- **H5:** multi-agent selection (MA-Chunk, B5b) beats single-agent selection (RL-Chunk, B5a) in downstream control performance, or matches it with fewer chunks.

## 3. Proposed architecture (Phase 1)

```mermaid
flowchart TB
  subgraph OFF["Offline · MA-Chunk env (Py 3.9)"]
    D[Existing SUNT-derived pickles\nstop_passenger_flow · travel_time_stats · hourly_stop_activity · real_routes] --> CH[Structured chunks\nstop_hour · segment · stop_activity\n+ provenance + text rendering]
    N[GTFS stop/route names\nstreet · neighborhood · landmark] --> CH
    QK[All query keys\n(trip, stop idx, direction, hour, day type, rain, road works)] --> CAND[Candidates: next H stops/edges at ETA hour]
    CH --> CAND --> SC[Scorers\nBM25 · FAISS multilingual dense · place · proximity · ETA · salience · support]
    SC --> LBL[L1 proxy labels\n(no LLM)]
    SC --> RLC[RL-Chunk PPO, B5a\nroute-order walk, SELECT / SKIP\nlabel-free reward]
    SC & LBL --> MA[MA-Chunk PPO, B5b\nscorers as agents\nSKIP / SEND / STOP, price λ]
    RLC & MA --> FZ[Pooled featurizer → 10 numbers]
    FZ --> TB[(rag_table_arm.npz + manifest)]
  end
  subgraph ON["Online · marllib env (Py 3.8) — BusEnv unchanged"]
    ENV[parallel_env\nsunt_env.py] --> W[RagObsWrapper\nkey from env state → lookup → concat]
    TB --> W
    W --> PAD[SuperSuit pad] --> RL[RLlibSuntBus → MARLlib MAPPO/IPPO]
  end
```

Six design decisions, each with its reason:
1. **Structured chunks from existing artifacts only.** Every value traces to raw OD/Boarding columns through existing pipelines (doc 04 §3). Nothing is invented.
2. **Stop and place names (GTFS) are used for matching, never as values.** Street, neighbourhood and landmark parsed from `stop_name` and `stop_desc` (225 of 226 study stops matched) make lexical and dense retrieval match *places* and corridors. They also enable a `place` scorer and optional cross-route `corridor` chunks (doc 04 §3.2). Ablation A11 measures their effect.
3. **Query = discrete decision context, so everything is precomputed** (about 105k keys). RL-Chunk, MA-Chunk, BM25, FAISS and the dense encoder run **offline** in their own Python environment. BusEnv only performs a dictionary lookup. This avoids the Python, numpy and torch version conflict [V] (doc 02 §3.3), adds no latency in the training loop, and makes every arm a hash-identified file.
4. **RL-Chunk and MA-Chunk are both reused, with minimal adapters.**
   - RL-Chunk: a subclass overrides only `_chunks_and_scores`. It walks candidates in route order and keeps its label-free reward (doc 02 §2.1).
   - MA-Chunk: unchanged. Its "agents" are SUNT scorers, trained from LLM-free L1 labels (doc 02 §4.3).
   - Both are trained separately and then frozen.
5. **Retrieved context reaches the policy only as 10 numeric features.** The reward, transition and action space are unchanged, so any gain comes from better-informed decisions (doc 05 §5).
6. **Integration is a PettingZoo wrapper placed before SuperSuit.** `RLlibSuntBus` takes the wider observation space automatically [V] (`train_marllib.py:139–144`). With `RAG_MODE=off`, the behaviour is the current BusEnv, bit for bit.

## 4. Implementation strategy

- **Phase 1:** steps 1–13 of doc 07. New modules go in `src/rag/` and `src/pipelines/build_*`, `train_rag_selector.py`, `evaluate_rag.py`, plus about 15 lines in `train_marllib.py`. Estimated at 2–3 weeks of work plus 2–3 days of compute.
- **Phase 2:** only if H1 and (H2 or H3) hold. An offline, cached LLM turns each unique selected-chunk set into a fixed JSON schema of features (doc 05 §3).
- **Phase 3:** only with evidence. Dynamic queries, labels attributed from downstream outcomes, fleet-level shared retrieval, joint training, per-day demand in the simulator, graph retrieval (doc 07 §4).

## 5. Experiments (summary of doc 06)

- **Arms:**
  - B0 (original)
  - B0z (zero context)
  - B1 (direct look-ahead)
  - B2 BM25, B3 FAISS dense, B4 hybrid RRF
  - **B5a RL-Chunk**
  - **B5b MA-Chunk**
  - B6 random
  - B7 (Phase 2: the better of B5a/B5b plus LLM features)

  All arms except B0 share the featurizer and the observation size.
- **Setup:**
  - MAPPO and IPPO, 10 seeds each;
  - 25 agents on 5 routes;
  - chronological split: 144 training days (2024) and 39 test days (2025);
  - reward weights pinned and logged.
- **Metrics:**
  - four reward components and the ideal-occupancy percentage (already logged);
  - waiting, headway coefficient of variation and fleet utilization from replay logs;
  - learning-curve AUC (existing tools);
  - retrieval P@K, nDCG and number of chunks against L1 labels;
  - latency, memory and energy (CodeCarbon is already integrated).
- **Validity controls and ablations:**
  - a shuffled-key permutation control (A7);
  - an oracle selection upper bound (A8);
  - a supervised selector instead of RL (A6);
  - a run without the salience scorer (A1);
  - **stop names off** (A11);
  - RL-Chunk with the label reward (A12) and with a different walk order (A13).
- **Statistics:** Wilcoxon signed-rank paired by seed with Holm correction for B5a and B5b against B0, B1, B4 and B6, and for B5b against B5a; Friedman and Nemenyi tests with a critical-difference diagram over all arms. The single pre-registered primary endpoint is the AUC of the scalarized reward.

## 6. Expected contributions

1. **Method:** the first use, to our knowledge, of *learned adaptive retrievers* (RL-Chunk and MA-Chunk) as the context provider for a **control** policy rather than an LLM reader, with LLMs kept offline (doc 03 gaps 1, 2 and 4). The study includes a head-to-head test of single-agent against multi-agent selection in that setting (H5).
2. **Domain:** retrieval-augmented observations for **multi-agent bus control on real SUNT data** from Salvador; no Latin American study of this kind was found (doc 03 gap 6).
3. **Evaluation:** ablations that separate *information content* from *selection* (B0z, B1, A7, A8), which is missing from bus-control papers (doc 03 gap 7).
4. **Engineering:** a reproducible design with no impact on BusEnv's code or behaviour when RAG is off. Precomputed context tables let any MARLlib algorithm use it.

A negative result for H2 or H3 is also a contribution: it shows when adaptive retrieval is *not* worth it for control.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Simple look-ahead (B1) already captures the gain | Report it honestly; ablation A3 (larger horizon H) tests whether selection matters as the candidate pool grows |
| L1 label partly overlaps with the salience scorer | Ablation A1, per-scorer AUC, MARL outcome as primary endpoint |
| Demand chunks equal the simulator's own expectations (static flow statistics) | State this explicitly; optional realism check with chunks from training-period data and a simulator built from test-period data (doc 06 §3) |
| Bugs or ambiguity in existing code that affect baselines (`occ_risk` key, weight vector vs tag, eval script with 20 agents) | Doc 01 §4 and §7, doc 06 §8: decide once and apply the same to all arms |
| Too few seeds for small effects | 10 seeds; add seeds before adding arms |

## 8. Answers to the final questions

1. **What is the simplest useful way to integrate RLChunk into BusEnv?**
   Run **both RL-Chunk and MA-Chunk offline** over the same structured SUNT candidate pools.
   - RL-Chunk uses a one-method subclass, a route-order walk and its own label-free reward.
   - MA-Chunk uses SUNT scorers as agents and L1 labels.

   Precompute a 10-feature context vector per decision key for each selector. Append it to each bus's observation through a PettingZoo wrapper. No LLM, no reward change, no action change.
   - MA-Chunk is expected to be stronger: the original RL-Chunk degenerates on QA (`BARREIRAS.md` B1).
   - RL-Chunk's sequential walk fits BusEnv's natural route order, so the comparison is open and is tested as H5.

2. **Which existing BusEnv components need to change?**
   Only `src/pipelines/train_marllib.py` (about 15 lines): wrap the env when `RAG_MODE != off`, apply the day split, and make the reward weights and tag consistent. Evaluation needs a **new** `evaluate_rag.py`, because `evaluate_marllib.py` is a training-script copy with 20 agents. Everything else is new code in `src/rag/` and new pipeline scripts.

3. **Which parts can remain completely untouched?**
   - `src/envs/sunt_env.py` (environment, transition, demand model, `DefaultReward`, observation and action spaces);
   - the MARLlib algorithms and configs;
   - the data pipelines and pickles;
   - the forecasting module;
   - the visualization;
   - the existing analysis tools, which are reused.

4. **What data can actually be used to construct chunks?**
   - `stop_passenger_flow.pkl`: mean boardings, alighting fraction and inter-visit time per route, stop and hour, from OD `n-boardings`, `n-alighting`, `lag_loading`, `stop_time`;
   - `travel_time_stats.pkl`: edge travel-time mean, median, standard deviation and count, from OD `stop_time` differences;
   - `hourly_stop_activity.pkl`: Boarding record counts per stop and hour;
   - the route structure, from `real_routes.pkl` and `route_metadata.pkl`;
   - query context from INMET rain and the road-works windows;
   - **GTFS stop and route names** (`stop_name` → street / neighbourhood / postcode, `stop_desc` landmark, `route_long_name`), used for text rendering, the `place` scorer and optional corridor grouping. They are never used as measurements. Note that road-works names do not appear in stop names, so road-works proximity must stay geometric.

   Not in Phase 1: daily per-edge travel times (leakage, size), the quantum and prediction folders (only 7 trips), LTI uptime (keyed by vehicle) (doc 04 §3).

5. **How should retrieval results influence the RL agent?**
   Only through the observation: a fixed-length, order-independent numeric vector (demand ahead, peak position, alighting, travel time and its variability ahead, stop activity, selection count), concatenated with the 12 original features. The policy, reward and actions are unchanged, so effects can be attributed to information.

6. **Is an LLM necessary for the first experiment?**
   **No.** The selector trains on LLM-free labels and the policy consumes numbers. An LLM appears only in Phase 2: offline, cached, producing a fixed JSON schema, and evaluated against the numeric features it was derived from.

7. **What is the minimum viable experiment that could demonstrate a scientific contribution?**
   - Arms: B0, B1, B4 (hybrid top-K), B5a (RL-Chunk), B5b (MA-Chunk), B6 (random), plus the A7 shuffled-key control.
   - MAPPO only, 5 routes and 25 agents, train on 2024 days and test on 39 days of 2025, 10 seeds.
   - Primary endpoint: AUC of the scalarized reward.
   - Secondary endpoints: test-day occupancy-band percentage, sync and efficiency scores, and headway coefficient of variation.
   - That is about 70 runs (about 18–28 h of compute) plus under 2 h offline.

   The claim that *adaptive selection of historical context improves multi-agent control* would be supported by:
   - an adaptive arm > B0 (H1);
   - that arm > B4 and B6 at equal budget, and ≥ B1 with fewer chunks (H2/H3);
   - A7 falling back to B0z.

   B5b against B5a answers H5.

8. **What should be implemented first, second, and third?**
   1. **First:** the online side and the baseline integrity checks.
      - `keys.py`, `wrapper.py`, `day_split.py` and the `train_marllib.py` switch;
      - a B0z table (zeros) and a B1 look-ahead table;
      - the bit-identical test with `RAG_MODE=off`;
      - `evaluate_rag.py`.

      This also delivers the first scientific check, B1 against B0.
   2. **Second:** the offline knowledge base and conventional retrieval.
      - chunks with place fields from stop names, queries, scorers (BM25, multilingual FAISS, place, structural), L1 labels;
      - tables for B2, B3, B4 and B6;
      - offline retrieval metrics, including A11 (names on/off).
   3. **Third:** the two adaptive selectors.
      - RL-Chunk: subclass, one PPO model per route, B5a table;
      - MA-Chunk: 5-fold by trip_id, choose λ, B5b table;
      - run the full grid, the ablations A7, A8, A6, A1, A11 and A12, and the statistics.
      - Phase 2 (LLM) only after this, and only if H1 and (H2 or H3) hold.

## 9. Open questions for the team

- **[Q1]** Fix the `occ_risk` key bug (`sunt_env.py:1599`) before the study, for all arms? (doc 01 §4)
- **[Q2]** Which reward weight vector is official? (`train_marllib.py:97–98` weights ≠ tag)
- **[Q3]** `use_rain` on or off for the study? (It changes the context dimension of the query.)
- **[Q4]** Commit or document `sunt_bus.yaml`; the local `MARLlib/` folder is empty.
- **[Q5]** A holiday calendar for `day_type`?
- **[Q6]** Training budget: keep 200k timesteps or extend after checking B0 plateaus?
- **[Q7]** Phase 2 LLM choice (local open-weights for reproducibility vs API) and whether GTFS stop names may be included in prompts.
- **[Q8]** Road-work coordinates: are they available outside the notebook that produced `route_obras_proximity.csv`? They would allow stop-level (geometric) road-works context.

## 9b. Version control and project tracking

- **Branch:** `Research-RAG-BusEnv` (`LabIA-UFBA/BusEnv`).
- **Card:** every push must be linked to the GitHub Projects card, issue **`kunumi-labia-ufba/opt_repo#68`** (<https://github.com/orgs/kunumi-labia-ufba/projects/4/views/1?pane=issue&itemId=253911103&issue=kunumi-labia-ufba%7Copt_repo%7C68>).
- **How:** every commit message ends with `Refs kunumi-labia-ufba/opt_repo#68`. Use the full cross-repository form, because the issue is in `opt_repo`, not in BusEnv.
- **Details:** doc 07 §6.

## 10. Document map

| Doc | Content |
|---|---|
| `01_busenv_architecture_analysis.md` | lifecycle, observation, action, reward, demand model, data provenance, integration points, issues found |
| `02_rlchunk_adaptation.md` | RL-Chunk and MA-Chunk analysis, both adapted (B5a/B5b), reuse/adapt/drop, adapter interfaces, label design |
| `03_literature_review.md` | 34 papers reviewed in detail plus 13 briefly noted, comparison table, gaps, implementation insights, 47 references (key ones independently re-verified) |
| `04_rag_architecture.md` | chunk schema, stop names as place context, query, candidates, scorers, selection arms, featurizer, wrapper |
| `05_llm_rl_integration.md` | four integration strategies compared; why Phase 1 has no LLM |
| `06_experimental_protocol.md` | hypotheses, arms, fixed conditions, metrics, ablations, statistics, budget |
| `07_implementation_roadmap.md` | step-by-step Phase 1 plan with files, tests, complexity; Phases 2–3 |
