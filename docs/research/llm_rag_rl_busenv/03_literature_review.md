# Workstream C: Literature Review: Retrieval, RL and LLMs for Multi-Agent Public Transit Control

**Scope.** This review covers prior work for a project that uses RAG plus **RLChunk** (RL-based, query-specific adaptive chunk selection), and optionally LLMs, to retrieve relevant information from SUNT-derived data (Salvador, Bahia, Brazil) and feed it as context to the MARL bus agents of **BusEnv**. The design keeps environment changes small and keeps the LLM out of the environment hot loop.

**Date of search:** 2026-11-13.

**How papers were verified.**
- Bibliographic metadata (title, authors, year, arXiv ID, DOI, journal-ref/comment) was checked against the **arXiv API**.
- Venues were cross-checked with the **Semantic Scholar Graph API**, **Crossref** and **OpenAlex**.
- Abstracts were read from arXiv/OpenAlex/Semantic Scholar.
- Exploratory discovery and gap checks used web search.

**Legend.**
- **[V]**: verified from metadata or the abstract.
- **[I]**: our interpretation or recommendation.
- **"not verified"**: we could not confirm the detail from a primary source.
- Reported results are copied only as stated in each paper's abstract. We did not read every full text, so architecture and experimental details beyond the abstract are kept to a minimum and marked when uncertain.

**Size.** 34 papers are reviewed in detail (Sections 1–7). 13 more are noted briefly as context (Section 8). Reference list: 47 entries.

**Independent spot-check (2026-11-13).** The seven most decision-relevant references were re-checked against the primary registries, separately from the review itself. Title, authors and date or venue matched in every case:
- arXiv API: 2608.10207 (Dong & Gayah), 2505.07233 (DynamicRAG), 2512.14465 (Context-Picker), 2510.14337 (Stop-RAG), 2510.26242 (REG-TSC).
- Crossref: 10.1016/j.tre.2025.104142 (Yu, Wang & Ma, *Transportation Research Part E* 200) and 10.1609/aaai.v40i27.39394 (MAICC, AAAI 2026, pp. 22363–22371).

---

## 0. Project context (for relevance judgements)

- **BusEnv** (PettingZoo `ParallelEnv`, trained with RLlib/MARLlib):
  - Each bus is an agent on a real Salvador GTFS route.
  - `Box(12,)` observation: time of day, occupancy, average travel time to the next stop, historical future demand at the next stop, uptime, maintenance status, current/next node ids, rain flag, occupancy-risk forecast, and signed headway to the leader/follower.
  - `Discrete(3)` actions: WAIT / MOVE / SERVICE_CENTER.
  - Multi-objective reward: occupancy, uptime, headway synchronization, efficiency. A lexicographic/MORL branch also exists.
  - Several observation features (travel time, demand) already come from **lookups into pickled historical SUNT statistics**. They are a primitive, non-adaptive form of retrieval.
- **RLChunk / MA-Chunk.** A PPO policy (stable-baselines3) chooses which candidate chunks (from BM25/FAISS) to keep and when to stop, instead of a fixed top-k. MA-Chunk is a multi-agent variant. The current reward is based on downstream LLM answer quality (ROUGE, BERTScore, RAGAS-style metrics).

---

## 1. Retrieval-augmented RL and episodic memory

### 1.1 Neural Episodic Control (NEC)
- **Authors:** Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adrià Puigdomènech, Oriol Vinyals, Demis Hassabis, Daan Wierstra, Charles Blundell.
- **Year / venue:** 2017, ICML 2017 [V, Semantic Scholar]. **ID:** arXiv:1703.01988. **Status:** peer-reviewed.
- **Objective:** make deep RL data-efficient by assimilating new experiences quickly [V].
- **Architecture / mechanisms:** a semi-tabular value function. A buffer of past experience holds slowly changing state embeddings and rapidly updated value estimates, and the value of a state is read by looking up similar stored embeddings [V abstract]. The lookup is k-nearest-neighbour over a "differentiable neural dictionary" (from the paper body as we recall it; not re-verified here).
- **Setup / results:** "a wide range of environments" (Atari in the paper). The agent learns significantly faster than other general-purpose deep RL agents [V abstract].
- **Limitations:** single agent. The memory grows with experience. Keys are learned, so memory contents drift as the encoder trains.
- **Relevance [I]:** this is the canonical "retrieve similar past states → value/feature" pattern. For BusEnv, an episodic store keyed by (stop, time-bin, headway, occupancy) could provide nearest-neighbour value or feature estimates. It is a non-LLM baseline that RLChunk-based retrieval should beat.
- *Related:* Blundell et al., "Model-Free Episodic Control", arXiv:1606.04460, 2016 (preprint) [V]. It is the tabular precursor and reports faster learning than deep RL on several domains.

### 1.2 Retrieval-Augmented Reinforcement Learning (R2A)
- **Authors:** Anirudh Goyal, Abram L. Friesen, Andrea Banino, Théophane Weber, Nan Rosemary Ke, Adrià Puigdomènech Badia, Arthur Guez, Mehdi Mirza, et al.
- **Year / venue:** 2022, ICML 2022 [V, Semantic Scholar]. **ID:** arXiv:2202.08417. **Status:** peer-reviewed.
- **Objective:** let the agent condition on a whole dataset of experiences at decision time, instead of distilling everything into weights [V].
- **Architecture / mechanisms:** the RL agent is augmented with a **retrieval process parameterised as a neural network** that has direct access to a dataset of experiences. The dataset can be the agent's own experience, expert demonstrations or other sources. The retrieval is trained to fetch information useful in the current context [V].
- **Setup / results:** integrated into an offline DQN agent and an online R2D2 agent [V].
  - Offline multi-task problems: retrieval-augmented DQN avoids task interference and learns faster.
  - Atari: retrieval-augmented R2D2 learns significantly faster and scores higher than the baseline [V].
- **Limitations [I]:** retrieval is learned end-to-end with the agent, which is costly. Evaluated only in single-agent game domains.
- **Relevance [I]:** this is the closest conceptual template for "retrieval output becomes an extra input to the policy". RLChunk plays the role of R2A's retrieval process. The difference is that RLChunk is a *discrete select/stop* policy over BM25/FAISS candidates, and it can be trained separately from the MARL policy (see Section 10).

### 1.3 Large-Scale Retrieval for Reinforcement Learning
- **Authors:** Peter C. Humphreys, Arthur Guez, Olivier Tieleman, Laurent Sifre, Théophane Weber, Timothy Lillicrap.
- **Year / venue:** 2022, NeurIPS 2022 [V, arXiv comment]. **ID:** arXiv:2206.05314. **Status:** peer-reviewed.
- **Objective:** support the agent's parametric computation with large-scale, context-sensitive database lookups. New information can then be used without retraining, just by augmenting the dataset [V].
- **Mechanisms:** fast approximate nearest-neighbour retrieval over **tens of millions** of expert demonstration states. The agent attends to the retrieved data end-to-end. Setting: offline RL in 9x9 Go [V].
- **Results:** attending to retrieved data significantly boosts prediction accuracy and game-play performance compared with using the same demonstrations only as training trajectories [V].
- **Limitations:** a single board game with a perfectly observed state. The retrieval corpus consists of expert states.
- **Relevance [I]:** shows that retrieval over historical data can add value beyond using the same data for training. This is the main scientific argument for retrieving from SUNT history at decision time instead of only training on it. The point that the corpus can be updated without retraining fits a transit operator well (new GTFS feed, new month of AVL data).

### 1.4 Retrieval-Augmented Decision Transformer (RA-DT)
- **Authors:** Thomas Schmied, Fabian Paischer, Vihang Patil, Markus Hofmarcher, Razvan Pascanu, Sepp Hochreiter.
- **Year / venue:** 2024. arXiv preprint; peer-reviewed venue not verified. **ID:** arXiv:2410.07071.
- **Objective:** in-context RL without putting entire episodes in context [V].
- **Mechanisms:** an external memory of past experiences. Only sub-trajectories relevant to the current situation are retrieved. The retrieval component **requires no training and can be domain-agnostic** [V].
- **Setup / results:** grid-worlds, robotics simulations and procedurally generated games. On grid-worlds it outperforms baselines while using a fraction of their context length. The authors also point out limits of in-context RL on complex environments [V].
- **Relevance [I]:** supports using a **frozen, training-free retriever** (for example a frozen sentence encoder plus FAISS) as a first step in BusEnv. Retrieving *sub-trajectories* matches the idea of retrieving "what happened on this route segment at this hour on similar days".

### 1.5 Multi-agent In-context Coordination via Decentralized Memory Retrieval (MAICC)
- **Authors:** Tao Jiang, Zichuan Lin, Lihe Li, Yi-Chen Li, Cong Guan, Lei Yuan, Zongzhang Zhang, Yang Yu, et al.
- **Year / venue:** 2026, AAAI 2026, Vol. 40 No. 27, pp. 22363–22371 [V, arXiv journal-ref]. **DOI:** 10.1609/aaai.v40i27.39394. **ID:** arXiv:2511.10030. **Status:** peer-reviewed.
- **Objective:** fast adaptation and coordination in cooperative MARL with decentralized execution [V].
- **Mechanisms [V]:**
  - A centralized embedding model learns trajectory representations.
  - Decentralized models approximate it to obtain team-level task information.
  - Relevant trajectories are retrieved as context and combined with each agent's current sub-trajectory.
  - A memory mechanism balances test-time online data against offline memory.
  - A hybrid utility score mixes individual and team returns for credit assignment.
- **Setup / results:** Level-Based Foraging and SMAC v1/v2. Faster adaptation to unseen tasks than existing methods [V].
- **Limitations [I]:** generic game benchmarks, transformer-based in-context RL rather than MAPPO-style training, and no transit or real data.
- **Relevance [I]:** **the closest work to "retrieval in MARL"** that we found. It shows that decentralized retrieval with a centrally trained embedding works in CTDE. That maps naturally onto MAPPO/HAPPO in MARLlib, and onto MA-Chunk (one retriever per agent or per route, trained with a team-level signal).

---

## 2. MARL with shared memory and communication

### 2.1 Improving Coordination in Small-Scale Multi-Agent Deep RL through Memory-driven Communication (MD-MADDPG)
- **Authors:** Emanuele Pesce, Giovanni Montana.
- **Year / venue:** 2020, *Machine Learning* (Springer) [V]. **DOI:** 10.1007/s10994-019-05864-5. **ID:** arXiv:1901.03887. **Status:** peer-reviewed.
- **Objective / mechanism:** agents learn an explicit communication protocol through a **shared memory device**. They learn read and write operations end-to-end with DDPG-based CTDE, and infer a shared representation of the world [V].
- **Results:** better coordination in scenarios with up to six agents. The paper also includes channel-corruption tests and memory visualisation [V].
- **Limitations:** small scale (≤6 agents). The memory is learned, not grounded in external data.
- **Relevance [I]:** a learned shared memory is the "internal" counterpart of an external retrieval store. In BusEnv, buses on the same route could share a route-level retrieval context (the same retrieved chunks), which acts as implicit communication without learning a protocol. That is easier to train with many buses.
- *Related (brief, Section 8):* DIAL/RIAL (Foerster et al., 2016) and CommNet (Sukhbaatar et al., 2016) for learned communication. MAPPO (Yu et al., 2022) is the main algorithm BusEnv uses.

---

## 3. LLMs and RL

### 3.1 Survey on LLM-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods
- **Authors:** Yuji Cao, Huan Zhao, Yuheng Cheng, Ting Shu, Yue Chen, Guolong Liu, Gaoqi Liang, Junhua Zhao, et al.
- **Year / venue:** IEEE Transactions on Neural Networks and Learning Systems [V]. **DOI:** 10.1109/TNNLS.2024.3497992. **ID:** arXiv:2404.00282. **Status:** peer-reviewed survey.
- **Content:** a taxonomy of LLM roles in RL: **information processor, reward designer, decision-maker, generator**. For each role it reviews methods and open challenges [V].
- **Relevance [I]:** gives vocabulary for positioning the project. Our intended LLM roles are (a) *information processor* (offline summarisation/embedding of SUNT records and stop context into features) and possibly (b) *reward designer* (offline), never *decision-maker* in the hot loop. RLChunk adds something this taxonomy does not name explicitly: a **learned retrieval selector** between the data store and the RL agent.

### 3.2 Large Language Models Are Semi-Parametric Reinforcement Learning Agents (REMEMBERER)
- **Authors:** Danyang Zhang, Lu Chen, Situo Zhang, Hongshen Xu, Zihan Zhao, Kai Yu.
- **Year / venue:** 2023, NeurIPS 2023 [V, Semantic Scholar]. **ID:** arXiv:2306.07929. **Status:** peer-reviewed.
- **Mechanisms:** an LLM agent with a long-term **experience memory**. The memory is updated by "RL with Experience Memory" (RLEM), so the system learns from successes and failures without fine-tuning the LLM [V].
- **Results:** exceeds prior SOTA success rate by 4% and 2% on two task sets [V].
- **Limitations [I]:** the LLM is the policy (in the loop). Evaluated on text/web task sets.
- **Relevance [I]:** shows that an experience memory with RL-updated values can stand in for parametric learning. In our design the policy is the MARL network, not an LLM. The memory-update idea could still inform how RLChunk's corpus is curated (keep the chunks that helped).

---

## 4. RAG, adaptive retrieval and RL-based retrieval

### 4.1 Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.
- **Year / venue:** arXiv 2023; ICLR (2024 conference) [V Semantic Scholar venue = ICLR]. **ID:** arXiv:2310.11511. **Status:** peer-reviewed.
- **Mechanisms:** a single LM retrieves **on demand** and critiques the retrieved passages and its own output using special "reflection tokens". This makes retrieval adaptive rather than a fixed number of passages [V]. It is trained with supervised targets, not online RL (based on our reading of the paper; treat as not verified here).
- **Results:** Self-RAG (7B/13B) outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning and fact verification, with better factuality and citation accuracy [V].
- **Relevance [I]:** motivates "retrieve only when needed". For BusEnv this means an agent at a quiet stop at 03:00 may need no retrieved context, while an agent at a congested hub at peak hour needs more. RLChunk's STOP action can learn this.

### 4.2 Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity
- **Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park.
- **Year / venue:** 2024, NAACL 2024 [V]. **ID:** arXiv:2403.14403. **Status:** peer-reviewed.
- **Mechanisms:** a small LM classifier predicts query complexity and routes each query to no retrieval, single-step retrieval or iterative retrieval [V].
- **Results:** better overall efficiency and accuracy on open-domain QA than adaptive-retrieval baselines [V].
- **Relevance [I]:** a cheap, *non-RL* routing baseline. In BusEnv, a hand-crafted or learned "context complexity" signal (peak vs. off-peak, rain, headway deviation) could gate retrieval. It is a useful ablation against RLChunk.

### 4.3 Active Retrieval Augmented Generation (FLARE)
- **Authors:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, et al.
- **Year / venue:** 2023, EMNLP 2023 [V]. **ID:** arXiv:2305.06983. **Status:** peer-reviewed.
- **Mechanisms:** decides *when and what* to retrieve during generation. It uses a prediction of the upcoming sentence as a forward-looking query, and re-retrieves when that prediction contains low-confidence tokens [V].
- **Results:** superior or competitive performance on four long-form knowledge-intensive tasks [V].
- **Relevance [I]:** **forward-looking retrieval** suits transit well: query with the *next* stop and the *expected arrival time*, not the current state. BusEnv already has `next_node_id` and a travel-time estimate, so the query key is "(next stop, ETA time-bin)".

### 4.4 Query Rewriting for Retrieval-Augmented LLMs (Rewrite-Retrieve-Read)
- **Authors:** Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan.
- **Year / venue:** 2023, EMNLP 2023 [V]. **ID:** arXiv:2305.14283. **Status:** peer-reviewed.
- **Mechanisms:** a small trainable rewriter LM is trained **with RL from the frozen LLM reader's feedback** to reformulate queries before web search [V].
- **Results:** consistent improvements on open-domain and multiple-choice QA [V].
- **Relevance [I]:** an early example of "RL trains a small retrieval-side module against a frozen downstream consumer". This is the pattern we propose: RLChunk trained against a frozen or fixed MARL policy, or against a proxy.

### 4.5 Search-R1: Training LLMs to Reason and Leverage Search Engines with RL
- **Authors:** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han.
- **Year / venue:** 2025. arXiv preprint; peer-reviewed venue **not verified**. **ID:** arXiv:2503.09516.
- **Mechanisms:** RL (outcome-based reward) trains an LLM to issue multiple search queries during step-by-step reasoning. **Retrieved-token masking** keeps RL training stable [V].
- **Results:** +41% (Qwen2.5-7B) and +20% (Qwen2.5-3B) over RAG baselines on seven QA datasets [V].
- **Limitations [I]:** expensive LLM-scale RL. The LLM itself is the policy.
- **Relevance [I]:** important as context for why *not* to put the LLM in the loop. The idea of masking retrieved content from the gradient is worth keeping: retrieved features should be treated as inputs, not as something the MARL policy can change.

### 4.6 DynamicRAG: LLM Outputs as Feedback for Dynamic Reranking in RAG
- **Authors:** Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han.
- **Year / venue:** 2025. arXiv preprint; venue not verified. **ID:** arXiv:2505.07233.
- **Mechanisms:** the reranker is an **RL agent that sets both the order and the number (k) of retrieved documents** per query. Its reward comes from LLM output quality [V].
- **Results:** state of the art among models of the same size on seven knowledge-intensive datasets [V].
- **Relevance [I]:** **the closest published work to RLChunk's core idea** (query-specific variable k learned by RL). It has the same weakness for our purpose: the reward is LLM answer quality. In BusEnv the consumer is a control policy, and the reward must come from control outcomes or a proxy.

### 4.7 Stop-RAG: Value-Based Retrieval Control for Iterative RAG
- **Authors:** Jaewan Park, Solbee Cho, Jay-Yoon Lee.
- **Year / venue:** 2025, NeurIPS 2025 MTI-LLM Workshop [V, arXiv comment]. **ID:** arXiv:2510.14337. **Status:** workshop paper.
- **Mechanisms:** iterative RAG is cast as a finite-horizon MDP. A value-based controller trained with forward-view Q(λ) targets decides when to stop retrieving. It works with black-box APIs [V].
- **Results:** beats fixed-iteration and prompt-based stopping on multi-hop QA [V].
- **Relevance [I]:** directly supports RLChunk's **STOP action** formulation. A value-based alternative to PPO (also compatible with RLChunk's DQN variants) is a natural ablation.

### 4.8 Context-Picker: Dynamic Context Selection Using Multi-Stage RL
- **Authors:** Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu.
- **Year / venue:** 2025. arXiv preprint. **ID:** arXiv:2512.14465.
- **Mechanisms:** context selection is reframed as finding a **minimal sufficient evidence subset**. Two-stage RL: stage 1 maximises recall, stage 2 prunes redundancy. Dense rewards come from offline leave-one-out "minimal sufficient set" mining [V].
- **Results:** beats strong RAG baselines on five long-context and multi-hop QA datasets [V].
- **Relevance [I]:** the **offline leave-one-out reward mining** is directly transferable. For each BusEnv state in a logged rollout, we can compute which retrieved SUNT chunks actually change the critic's value estimate or a demand forecast, and use that as a dense, cheap reward for RLChunk without running the environment.

### 4.9 M-RAG: RAG with Multiple Partitions (MARL)
- **Authors:** Zheng Wang, Shu Xian Teo, Jieer Ouyang, Yongjun Xu, Wei Shi.
- **Year / venue:** 2024, ACL 2024 [V, arXiv comment]. **ID:** arXiv:2405.16420. **Status:** peer-reviewed.
- **Mechanisms:** the memory database is split into partitions. **Multi-agent RL** agents select a partition and generate text, and are optimised jointly [V].
- **Results:** +11% (summarisation), +8% (machine translation), +12% (dialogue) over baselines across seven datasets [V].
- **Relevance [I]:** partitioned retrieval maps naturally onto transit, with partitions by route, by time-of-day band or by stop cluster. It also gives MA-Chunk an analogue: one retrieval agent per partition.

### 4.10 Improving RAG through Multi-Agent RL (MMOA-RAG)
- **Authors:** Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, et al.
- **Year / venue:** 2025, NeurIPS 2025 [V, arXiv comment]. **ID:** arXiv:2501.15228. **Status:** peer-reviewed.
- **Mechanisms:** each RAG pipeline module (query rewriting, retrieval, filtering, generation) is a cooperative RL agent. All share a unified reward (answer F1) [V].
- **Results:** improves overall pipeline performance over baselines on several QA benchmarks [V].
- **Relevance [I]:** provides the template for an *eventual* joint formulation in which RLChunk agents and bus agents cooperate under the shared BusEnv reward. Our recommendation is to start decoupled (Section 10).

---

## 5. Structured, graph and temporal retrieval

### 5.1 From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)
- **Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, et al.
- **Year / venue:** 2024. arXiv preprint (Microsoft). **ID:** arXiv:2404.16130.
- **Mechanisms:** an LLM builds an entity knowledge graph from documents, then pre-generates community summaries. Answers are assembled map-reduce style from the community summaries [V].
- **Results:** substantial improvements in answer comprehensiveness and diversity over conventional RAG on global sensemaking questions over corpora of about 1M tokens [V].
- **Limitations [I]:** LLM-heavy indexing. Designed for text sensemaking, not numeric control.
- **Relevance [I]:** the SUNT network (2,871 stops, 4,526 edges) is already a graph. "Community summaries" have a cheap analogue: **precomputed stop-cluster or corridor profiles** (demand by hour, travel-time variability, bunching frequency), built offline and retrieved by node id. An LLM is optional and would only be used to write human-readable profiles.

### 5.2 TableRAG: Million-Token Table Understanding with Language Models
- **Authors:** Si-An Chen, Lesly Miculicich, Julian Martin Eisenschlos, Zifeng Wang, Zilong Wang, Yanfei Chen, Yasuhisa Fujii, Hsuan-Tien Lin, et al.
- **Year / venue:** 2024, NeurIPS 2024 [V]. **ID:** arXiv:2410.04739. **Status:** peer-reviewed.
- **Mechanisms:** query expansion plus **schema retrieval and cell retrieval** pick out the relevant parts of large tables before they reach the LM [V].
- **Results:** highest retrieval quality and new SOTA on large-scale table understanding (new million-token benchmarks built from Arcade and BIRD-SQL) [V].
- **Relevance [I]:** SUNT-derived data is **tabular** (AVL/APC records, GTFS tables, per-stop statistics). TableRAG argues for retrieving at the row, column or cell level rather than naive text chunks. For RLChunk, a "chunk" could be a table slice, such as (route, stop, hour, day-type) aggregate rows.

### 5.3 It's About Time: Incorporating Temporality in Retrieval Augmented Language Models (TempRALM)
- **Authors:** Anoushka Gade, Jorjeta Jetcheva.
- **Year / venue:** 2024. arXiv preprint; venue not verified. **ID:** arXiv:2401.13222.
- **Mechanisms:** the retriever scores documents by **semantic and temporal relevance** to the query. It needs no pretraining and no re-indexing [V].
- **Results:** up to 74% improvement over a baseline RALM [V].
- **Relevance [I]:** transit data is strongly time-dependent (hour, weekday, season, rain). A temporal-relevance term (same hour bin, same day type, recent months weighted higher) should be part of the BM25/FAISS score, or a hard filter, before RLChunk selects.

### 5.4 Retrieval Augmented Time Series Forecasting (RAF)
- **Authors:** Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz, Samet Oymak.
- **Year / venue:** 2024. arXiv preprint; venue not verified. **ID:** arXiv:2411.08249.
- **Mechanisms:** retrieves related time-series segments and adds them to the input of time-series foundation models (e.g., Chronos) [V].
- **Results:** better forecasting accuracy across domains. Gains grow with model size [V].
- **Relevance [I]:** BusEnv already uses forecasts (`future_demand_at_B`, `occ_risk_out`). RAF suggests a concrete retrieval target: **past demand or travel-time windows similar to the current one** (same stop, similar recent trajectory), summarised as a few numbers in the observation.

---

## 6. Public transport control with RL (bus holding / bunching)

### 6.1 Dynamic Holding Control to Avoid Bus Bunching: A Multi-Agent Deep RL Framework
- **Authors:** Jiawei Wang, Lijun Sun.
- **Year / venue:** 2020, *Transportation Research Part C* [V Crossref]. **DOI:** 10.1016/j.trc.2020.102661. **Status:** peer-reviewed.
- **Mechanisms:**
  - Each bus is an agent that considers all other buses, not only its immediate leader or follower.
  - Headway-based reward.
  - Actor-critic with a **joint action tracker**, trained with PPO [V, abstract summary via Semantic Scholar].
- **Results:** outperforms baselines that use only local information (numerical experiments) [V].
- **Relevance [I]:** the reference MARL formulation for bus bunching. BusEnv's headway-aware reward and leader/follower observations follow the same line of work. Wang & Sun rely on *instantaneous* state. Retrieval of historical context is what our project would add.

### 6.2 Reducing Bus Bunching with Asynchronous Multi-Agent RL
- **Authors:** Jiawei Wang, Lijun Sun.
- **Year / venue:** 2021, IJCAI 2021 [V]. **DOI:** 10.24963/ijcai.2021/60. **ID:** arXiv:2105.00376. **Status:** peer-reviewed.
- **Mechanisms:** route-level fleet control as asynchronous MARL (agents act only on stop arrival). A critic uses a **graph attention network** to approximate other agents' marginal contributions [V].
- **Setup / results:** real-world bus services with passenger demand derived from **smart-card data**. Outperforms headway-based control and existing MARL methods [V].
- **Relevance [I]:** shows the value of real demand data and graph structure, both available in SUNT. It also points to asynchrony as a possible confound: BusEnv steps all agents synchronously, so retrieval latency per step is uniform.

### 6.3 Robust Dynamic Bus Control: A Distributional MARL Approach (IQNC-M)
- **Authors:** Jiawei Wang, Lijun Sun.
- **Year / venue:** IEEE Transactions on Intelligent Transportation Systems (2022 per arXiv journal-ref) [V]. **DOI:** 10.1109/TITS.2022.3229527. **ID:** arXiv:2111.01946. **Status:** peer-reviewed.
- **Mechanisms:** implicit quantile networks plus an **interpretable meta-learning module that injects global information** into distributional MARL [V].
- **Results:** handles extreme events (traffic perturbations, service interruptions, demand surges) better than traditional holding and SOTA MARL baselines in simulators built from real services and demand [V].
- **Relevance [I]:** the "global information module" is functionally close to what retrieved context would provide. It is learned rather than retrieved. Retrieval of *similar past disruptions* from SUNT history is a natural alternative to evaluate.

### 6.4 From Observation to Intervention: Bus Holding Control from Historical Uncontrolled Trajectories with Offline RL (Offline-GMAC)
- **Authors:** Minyu Shen, Xiaowen Bi, Weihua Gu, Zhandong Xu, Yuezhong Wu, Feng Xiao, Lu Li.
- **Year / venue:** 2026. arXiv preprint. **ID:** arXiv:2610.14632.
- **Mechanisms:** an offline graph-based multi-agent actor-critic trained only on **logged GPS data from normal uncontrolled operations**. GNN credit assignment. A data augmentation scheme synthesises unperformed holding actions by exploiting the equivalence between link travel-time delays and holding. Comes with a Wasserstein bound on the synthesised transitions [V].
- **Results:** comparable to, and sometimes better than, the online counterpart on real bus data [V].
- **Relevance [I]:** strong evidence that historical AVL logs (like SUNT) are usable for transit MARL. Here the logs serve as *training data*. Using them as a *retrieval corpus at decision time* remains open (Section 9).

### 6.5 Real-time Bus Holding Control on a Transit Corridor Based on MARL (MARL-H)
- **Authors:** Weiya Chen, Kunlin Zhou, Chunxiao Chen.
- **Year / venue:** 2016, IEEE ITSC 2016 [V]. **DOI:** 10.1109/ITSC.2016.7795538. **Status:** peer-reviewed.
- **Mechanisms:** coordination graphs and sparse cooperative Q-learning for coordinated holding [V].
- **Results:** simulations show advantages for real-time coordinated holding [V]. Numeric results not verified.
- **Relevance [I]:** early tabular MARL baseline. Useful for the related-work narrative that bus MARL is well established while retrieval augmentation is not.

---

## 7. LLMs in intelligent transportation and public transit

### 7.1 Large Language Model-Enhanced RL for Generic Bus Holding Control Strategies
- **Authors:** Jiajie Yu, Yuhong Wang, Wei Ma.
- **Year / venue:** 2025, *Transportation Research Part E*, vol. 200 [V OpenAlex/IDEAS]. **DOI:** 10.1016/j.tre.2025.104142. **ID:** arXiv:2410.10212. **Status:** peer-reviewed.
- **Mechanisms:** LLM modules (reward initializer, reward modifier, performance analyzer, reward refiner) **generate and iteratively refine the RL reward function** from training and test feedback. Ineffective rewards are filtered out [V].
- **Setup / results:** scenarios varying the number of lines, stops and demand. Per a secondary summary these include a synthetic single line and a real dual-line system in Beijing (the Beijing detail is not verified from the full text). Better than vanilla RL, an LLM-only controller, physics-based feedback and optimisation-based controllers [V abstract].
- **Limitations [I]:** the LLM shapes the *reward*, not the *observation*. Repeated full RL training loops are costly.
- **Relevance [I]:** **the most directly comparable LLM+RL bus paper.** It keeps the LLM outside the control loop (offline reward design), which matches our constraint. Our project differs by enriching the **observation** through learned retrieval. The two approaches are complementary: BusEnv's multi-objective weights could also be LLM-tuned offline.

### 7.2 Mitigating Bus Bunching with RL Enhanced by Semantic Stop Embedding
- **Authors:** Xin Dong, Vikash V. Gayah.
- **Year / venue:** 2026. arXiv preprint. **ID:** arXiv:2608.10207.
- **Mechanisms:** an LLM is used **offline** to turn heterogeneous stop information (physical attributes, surrounding activity, historical operational characteristics) into **fixed semantic embeddings**. These feed a deep Q-learning event-driven holding controller **without real-time LLM inference** [V].
- **Setup / results:** stochastic simulations calibrated with observed data from two bus routes [V].
  - Compared with the best calibrated Daganzo baseline: −32.0% headway variability, −69.2% bunching events, −24.0% passenger waiting time.
  - Route-specific stop identifiers do not help the spacing-only controller; semantic stop information does.
  - Cross-route: zero-shot transfer is limited, warm-start fine-tuning helps, cold-start gives the best final performance [V].
- **Limitations [I]:** embeddings are static per stop, with no query- or time-specific retrieval. Two routes. A single controller (the abstract does not say whether it is multi-agent).
- **Relevance [I]:** **the strongest precedent for our design.** It shows that offline LLM-derived features appended to an RL state help bus control, and that raw stop ids do not. BusEnv currently feeds raw `curr_node_id`/`next_node_id` as observations, so this paper directly motivates replacing or augmenting them with embeddings. Our contribution would add (i) **dynamic, query-specific retrieval** (time, weather, headway state), (ii) **RLChunk's adaptive selection**, and (iii) **multi-agent** use with real Salvador data.

### 7.3 TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models
- **Authors:** Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai. (The arXiv v1 lists five authors without W. Liang; the journal version lists six [V Crossref].)
- **Year / venue:** 2024, *Transport Policy* [V]. **DOI:** 10.1016/j.tranpol.2024.03.006. **ID:** arXiv:2309.06719. **Status:** peer-reviewed.
- **Mechanisms:** ChatGPT orchestrates **traffic foundation models as tools**. It breaks tasks down, processes traffic data and supports decisions through dialogue [V].
- **Results:** demonstrated qualitatively and as a system. No quantitative control benchmark is mentioned in the abstract [V].
- **Relevance [I]:** an example of LLM-as-orchestrator for *analysis*, not control. A useful contrast: we use LLMs (if at all) offline to build the retrieval corpus.

### 7.4 LLMLight: Large Language Models as Traffic Signal Control Agents
- **Authors:** Siqi Lai, Zhao Xu, Weijia Zhang, Hao Liu, Hui Xiong.
- **Year / venue:** KDD 2025 (Proc. 31st ACM SIGKDD) [V]. **DOI:** 10.1145/3690624.3709379. **ID:** arXiv:2312.16044. **Status:** peer-reviewed.
- **Mechanisms:** the LLM is the decision agent, prompted with real-time traffic conditions. **LightGPT** is a backbone specialised for traffic signal control [V].
- **Results:** outperforms nine baselines and ten LLMs on ten real and synthetic datasets, with evaluation by 15 human experts [V].
- **Limitations [I]:** an LLM in the decision loop at every signal phase, which is costly for the large multi-agent fleets BusEnv targets.
- **Relevance [I]:** a counterpoint showing that LLM-as-policy is feasible for TSC. It also justifies our choice to keep a small MARL policy and move the LLM offline.

### 7.5 iLLM-TSC: Integrating RL and LLM for Traffic Signal Control Policy Improvement
- **Authors:** Aoyu Pang, Maonan Wang, Man-On Pun, Chung Shue Chen, Xi Xiong.
- **Year / venue:** 2024 arXiv. Semantic Scholar lists *IEEE Transactions on Vehicular Technology* with DOI 10.1109/TVT.2026.3674284; not independently confirmed. **ID:** arXiv:2407.06025.
- **Mechanisms:** RL decides first, then an LLM checks whether the decision is reasonable and corrects it. This targets imperfect observations and rare events outside the reward. It integrates without changing the RL system [V].
- **Results:** −17.5% average waiting time under degraded communication versus traditional RL [V].
- **Relevance [I]:** the "LLM as post-hoc verifier" pattern. For us it is an evaluation and analysis tool (explain or flag odd bus actions offline), not a hot-loop component.

### 7.6 RAG-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles (REG-TSC)
- **Authors:** Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang.
- **Year / venue:** 2025. arXiv preprint. **ID:** arXiv:2510.26242.
- **Mechanisms:** distributed LLM agents, with a **Reviewer-based Emergency RAG** that retrieves knowledge and guidance from **historical cases**. "Reward-guided Reinforced Refinement" fine-tunes the LLM agents with a reward-weighted likelihood loss and priority-sampled experience across heterogeneous intersections [V].
- **Results:** on three real networks (17–177 intersections): −42.00% travel time, −62.31% queue length, −83.16% emergency-vehicle waiting time [V].
- **Limitations [I]:** LLM agents in the loop. Signal control, not transit. Retrieval is not learned adaptively.
- **Relevance [I]:** **the closest work combining RAG over historical cases, RL-style training and multi-agent traffic control** that we found. It confirms that retrieving historical cases helps multi-agent traffic decisions. It also leaves open (a) a non-LLM policy, (b) learned adaptive retrieval and (c) public transit.

### 7.7 TransitGPT: A Generative AI-based Framework for Interacting with GTFS Data Using LLMs
- **Authors:** Saipraneeth Devunuri, Lewis Lehe.
- **Year / venue:** arXiv 2024; *Public Transport* (Springer) per Semantic Scholar, DOI 10.1007/s12469-025-00395-w (not independently confirmed). **ID:** arXiv:2412.06831.
- **Mechanisms:** the LLM writes Python code that is executed on a server holding the GTFS feed. Prompt-only, no fine-tuning, and the LLM has no direct access to the feed [V].
- **Results:** evaluated with GPT-4o and Claude-3.5-Sonnet on a 100-task benchmark. The authors report that it improves the accessibility and usability of transit data [V].
- **Relevance [I]:** GTFS is relational and multi-file, and the authors argue that naive text-chunk retrieval struggles with it (per web-search summary of the paper; not verified in full text). For SUNT, an **offline code or SQL step that materialises per-(route, stop, hour) aggregates as retrievable rows** is more reliable than chunking raw GTFS text. The LLM could help write that step once, offline.

---

## 8. Briefly noted (context; verified metadata, abstract-level only)

| Paper | ID / venue [V] | Why noted |
|---|---|---|
| Blundell et al., *Model-Free Episodic Control* (2016) | arXiv:1606.04460, preprint | Tabular episodic memory; precursor to NEC |
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | arXiv:2005.11401, NeurIPS 2020 | Original RAG (dense retriever + seq2seq) |
| Foerster et al., *Learning to Communicate with Deep MARL* (2016) | arXiv:1605.06676, NeurIPS 2016 | RIAL/DIAL learned communication in CTDE |
| Sukhbaatar et al., *Learning Multiagent Communication with Backpropagation* (CommNet, 2016) | arXiv:1605.07736, NIPS 2016 | Continuous learned communication |
| Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO) | arXiv:2103.01955, NeurIPS 2022 D&B | Main algorithm family used in BusEnv |
| Hu et al., *MARLlib* (2022) | arXiv:2210.13708; venue not verified | BusEnv's training library; needs a fixed observation `Box` |
| Sun, Huang, Pompili, *LLM-based Multi-Agent RL: Current and Future Directions* (2024) | arXiv:2405.11106, preprint (submitted to IEEE RA-L) | LLM-MARL survey, communication focus |
| Alesiani & Gkiotsalitis, *RL-Based Bus Holding for High-Frequency Services* (2018) | DOI 10.1109/ITSC.2018.8569473, ITSC 2018 | Single-agent RL holding; Singapore circular line, 400 training / 200 evaluation day-episodes |
| Menda et al., *Deep RL for Event-Driven Multi-Agent Decision Processes* | DOI 10.1109/TITS.2018.2848264, IEEE T-ITS (2019 issue) | Asynchronous multi-agent policy optimisation; bus holding case study |
| Da et al., *Open-TI: Open Traffic Intelligence with Augmented Language Model* | arXiv:2401.00211; Int. J. Machine Learning and Cybernetics, DOI 10.1007/s13042-024-02190-8 | LLM orchestrating traffic tools and TSC training |
| Wang et al., *LLM-Assisted Light* (2024) | arXiv:2403.08337; venue not verified | LLM-centred TSC with tools; −20.4% waiting time under sensor outage vs. RL |
| Devunuri, Qiam, Lehe, *ChatGPT for GTFS: Benchmarking LLMs on GTFS Understanding and Retrieval* | arXiv:2308.02618; *Public Transport* per arXiv comment (DOI not verified) | Program synthesis beats zero-shot (up to 93% simple / 61% complex queries, GPT-4) |
| Nie, Sun, Ma, *Exploring the Roles of LLMs in Reshaping Transportation Systems: A Survey, Framework, and Roadmap* | DOI 10.1016/j.ait.2025.100003, *AI for Transportation* 2025 | LLM4TR taxonomy: information processors, knowledge encoders, component generators, decision facilitators |

Also seen during search but not reviewed: DeepRetrieval (arXiv:2503.00223, RL-trained query generation), Nie et al. 2023 robust multi-strategy bus control (arXiv:2308.08179), and GARLIC (GPT-augmented MARL for ride-hailing dispatch; listed as AAAI 2025 in mlanthology; not verified further).

---

## 9. Comparison table

Legend for "LLM in loop?": **No** = no LLM; **Offline** = LLM used only before or after training; **Train** = LLM inside the training loop but not needed at deployment; **Yes** = LLM queried at decision time.

| Paper | Retrieval mechanism | RL role | LLM in loop? | Domain |
|---|---|---|---|---|
| NEC (Pritzel 2017) | kNN over episodic buffer of embeddings | Value estimation from memory | No | Atari / games |
| R2A (Goyal 2022) | Learned neural retrieval over experience dataset | Policy conditioned on retrieved data | No | Atari, offline multi-task |
| Large-scale retrieval (Humphreys 2022) | Approx. NN over tens of millions of expert states | Offline RL policy/value attends to retrieval | No | 9x9 Go |
| RA-DT (Schmied 2024) | Training-free, domain-agnostic sub-trajectory retrieval | In-context RL (Decision Transformer) | No | Grid-worlds, robotics, games |
| MAICC (Jiang 2026) | Decentralized trajectory retrieval with centrally trained embeddings | Cooperative MARL, in-context | No | LBF, SMAC |
| MD-MADDPG (Pesce 2020) | Learned shared memory (read/write) | MARL communication | No | Particle-type tasks (≤6 agents) |
| REMEMBERER (Zhang 2023) | Experience memory lookup | RL updates memory values; LLM is policy | Yes | Text/web task sets |
| Self-RAG (Asai 2023/24) | On-demand retrieval + reflection tokens | None (supervised) | Yes (LLM is the system) | Open QA, fact verification |
| Adaptive-RAG (Jeong 2024) | Classifier routes none/single/iterative retrieval | None | Yes | Open QA |
| FLARE (Jiang 2023) | Forward-looking, confidence-triggered retrieval | None | Yes | Long-form generation |
| Rewrite-Retrieve-Read (Ma 2023) | Web search with RL-trained query rewriter | RL trains rewriter vs. frozen reader | Yes | QA |
| Search-R1 (Jin 2025) | Multi-turn search during reasoning | RL trains LLM to search | Yes | QA |
| DynamicRAG (Sun 2025) | RL reranker picks order and k | RL agent = reranker (variable k) | Yes (reward from LLM) | Knowledge-intensive QA |
| Stop-RAG (Park 2025) | Iterative retrieval with learned stopping | Value-based stop controller | Yes | Multi-hop QA |
| Context-Picker (Zhu 2025) | Minimal sufficient subset selection | Two-stage RL selector, LOO-mined rewards | Yes | Long-context QA |
| M-RAG (Wang 2024) | Partitioned memory, agent picks partition | MARL over partitions + generation | Yes | Summarisation, MT, dialogue |
| MMOA-RAG (Chen 2025) | Full RAG pipeline modules as agents | Cooperative MARL, shared F1 reward | Yes | QA |
| GraphRAG (Edge 2024) | Entity graph + community summaries | None | Yes (index + answer) | Text sensemaking |
| TableRAG (Chen 2024) | Schema + cell retrieval with query expansion | None | Yes | Large tables |
| TempRALM (Gade 2024) | Semantic + temporal relevance scoring | None | Yes | Temporal QA |
| RAF (Tire 2024) | Retrieval of similar time-series segments | None | No (TS foundation model) | Time-series forecasting |
| Wang & Sun 2020 (TR-C) | None (instantaneous state) | MARL holding (PPO, joint action tracker) | No | Bus bunching |
| Wang & Sun 2021 (IJCAI) | None (GAT critic) | Asynchronous MARL | No | Bus bunching, smart-card demand |
| IQNC-M (Wang & Sun 2022) | None (learned global-info module) | Distributional MARL + meta-learning | No | Bus control under disruptions |
| Offline-GMAC (Shen 2026) | None (historical logs as training data) | Offline multi-agent actor-critic | No | Bus holding, real GPS logs |
| MARL-H (Chen 2016) | None | Sparse cooperative Q-learning | No | Bus holding |
| LLM-enhanced RL bus holding (Yu 2025) | None | Standard RL; LLM designs reward | Offline/Train (reward design) | Bus holding (multi-line) |
| Semantic stop embedding (Dong & Gayah 2026) | Static per-stop LLM embeddings (lookup) | DQN holding controller | Offline only | Bus bunching (2 routes) |
| TrafficGPT (Zhang 2024) | Tool calls to traffic foundation models | None | Yes | Traffic management |
| LLMLight (Lai 2025) | None (prompted state) | LLM is policy (LightGPT tuned) | Yes | Traffic signal control |
| iLLM-TSC (Pang 2024) | None | RL policy + LLM verifier | Yes | Traffic signal control |
| REG-TSC (Li 2025) | RAG over historical emergency cases | Reward-weighted fine-tuning of LLM agents | Yes | Multi-intersection TSC |
| TransitGPT (Devunuri 2024) | LLM-generated code over GTFS | None | Yes | Transit data QA |
| **Proposed: RL-Chunk / MA-Chunk in BusEnv [I]** | BM25/FAISS + structural scorers over SUNT-derived chunks, **RL-selected, variable k**, time-aware | RL-Chunk (PPO, sequential walk) and MA-Chunk (PPO, scorers as agents) select context offline, compared head-to-head; MARL (MAPPO/IPPO) controls buses | **Offline only** (optional embedding/summarisation) | Bus fleet control, Salvador |

---

## 10. Research gaps

These statements reflect our search (arXiv/Semantic Scholar/Crossref plus targeted web searches on "retrieval-augmented multi-agent RL", "LLM bus holding", "RAG public transit", "retrieval-augmented RL traffic signal control", "retrieval augmented observation MARL bus") up to 2026-11-13. They are phrased as absence of evidence, not proof of absence.

1. **We found no work that uses retrieval (RAG-style, over historical or contextual data) to build observation features for multi-agent RL bus control.**
   - Bus MARL papers (Wang & Sun 2020, 2021, 2022; Chen 2016; Offline-GMAC 2026) use instantaneous state, learned global modules, or historical data *for training only*.
   - The nearest work, Dong & Gayah 2026, injects **static** offline LLM stop embeddings into a **single DQN controller** (per the abstract). There is no query-time retrieval and no adaptivity.

2. **We found no work in which a learned adaptive retriever (variable-k selection or stopping) feeds a downstream *control policy* rather than an LLM reader.**
   - DynamicRAG, Stop-RAG, Context-Picker, Rewrite-Retrieve-Read, Search-R1, M-RAG and MMOA-RAG all reward *text answer quality*.
   - Retrieval-augmented RL (R2A, Humphreys, RA-DT, MAICC) uses either a fixed kNN/top-k or end-to-end learned attention. None uses an explicit RL select/stop policy over candidates.
   - This is the methodological gap RLChunk/MA-Chunk fills [I]. (Doc 02 adapts both: RL-Chunk, single-agent, and MA-Chunk, the newer multi-agent formulation, compared head-to-head as arms B5a/B5b.)

3. **Retrieval in MARL is very recent and so far only on generic benchmarks.** MAICC (AAAI 2026) is the only cooperative-MARL retrieval method we found, on LBF and SMAC. We found no application to transport or to real operational data.

4. **Where retrieval appears in traffic control, it sits inside LLM agents.** REG-TSC retrieves historical cases, but its LLM agents are in the decision loop, and it targets signals, not buses. We found no RAG-enhanced traffic or transit work in which the deployed policy is a lightweight MARL network and the LLM is offline only.

5. **Temporal and tabular retrieval has not been connected to sequential control.**
   - TempRALM (temporal RAG) and RAF (time-series RAG) address QA and forecasting.
   - TableRAG and TransitGPT address tabular and GTFS QA.
   - We found no work that uses time-aware retrieval over tabular AVL/APC or GTFS records as RL state augmentation.

6. **Geographic and data gap.** We found no LLM/RAG + RL transit-control work using **Brazilian or Latin American** (including SUNT/Salvador) data. The works we found use Beijing (Yu et al., per secondary summary), Singapore (Alesiani & Gkiotsalitis), North American routes (Dong & Gayah, location not verified) or synthetic corridors. This should be stated cautiously, since regional venues were not searched exhaustively.

7. **Evaluation gap.** None of the bus-control papers reviewed report ablations that isolate the *information content* of the observation (for example: no context vs. fixed top-k vs. adaptive retrieval vs. oracle features). Dong & Gayah's stop-id vs. semantic-embedding comparison is the only partial example.

**Positioning statement [I]:** *"RL-Chunk / MA-Chunk in BusEnv" would be, to our knowledge, the first study of learned, query-specific adaptive retrieval over historical transit data as decision-time context for multi-agent bus control, with LLMs confined to offline preprocessing.*

---

## 11. Practical implementation insights (what to borrow)

All items are **[I]** recommendations derived from the papers above.

1. **Retrieval output as extra fixed-size observation features (minimal env change).**
   - MARLlib/RLlib need a fixed `Box` observation, so extend BusEnv's `Box(12,)` to `Box(12 + d,)`.
   - RLChunk selects a *variable* number of chunks. Pool them into a fixed vector, e.g.:
     - mean/max-pooled chunk embeddings projected to d = 8–32;
     - plus `n_selected / k_max`;
     - plus a "retrieved-nothing" flag.
   - This follows R2A/Humphreys (retrieval feeds the policy input) while leaving the MARL algorithm untouched.

2. **Keep the LLM offline; cache everything (Dong & Gayah; Yu et al.).**
   - Build per-stop or per-corridor text profiles and embeddings once, with an LLM if desired.
   - Store them in a lookup table keyed by `(node_id, hour_bin, day_type, is_raining)`.
   - During `step()`, retrieval is a dictionary or FAISS lookup on precomputed vectors: no network calls, deterministic, and fast enough for millions of MARL steps.

3. **Start with frozen, training-free retrieval (RA-DT), then add RLChunk.**
   - Stage 0: hand-picked features (current BusEnv).
   - Stage 1: fixed top-k BM25/FAISS with a frozen encoder.
   - Stage 2: RLChunk adaptive selection.
   - Stage 3 (optional): joint or cooperative training (MMOA-RAG/M-RAG style).
   - Each stage is an ablation row.

4. **Train RLChunk decoupled first, with a proxy reward (Rewrite-Retrieve-Read, Context-Picker).**
   - Joint training with the MARL return is high-variance and expensive.
   - Practical proxies:
     - (a) accuracy of predicting realised next-segment travel time, boarding demand or headway from the selected chunks, using held-out SUNT days;
     - (b) leave-one-out influence of each chunk on a frozen MAPPO critic's value estimate, which gives dense rewards in the style of Context-Picker's minimal sufficient sets;
     - (c) a cost penalty per selected chunk (RLChunk's existing length or efficiency term).
   - Then fine-tune RLChunk against the frozen MARL policy's return.

5. **Make retrieval time-aware and forward-looking (TempRALM, FLARE, RAF).**
   - Build the query from the *next* stop and the *expected arrival time*, not only the current state.
   - Add a temporal relevance term (same hour bin and weekday type, recency decay), or hard-filter candidates before RLChunk.
   - Retrieve *similar recent windows* of demand or travel time (RAF), not only static profiles.

6. **Use structured, table-shaped chunks (TableRAG, TransitGPT).**
   - Define a "chunk" as a materialised aggregate row, e.g. `(route, stop, hour, day_type) → mean/var travel time, boarding rate, alighting fraction, bunching frequency, rain effect`, with an optional one-line text rendering for BM25.
   - Generate these with offline pandas or SQL (possibly LLM-assisted, as in TransitGPT). Avoid chunking raw GTFS text.

7. **Exploit the graph (GraphRAG, Wang & Sun 2021).**
   - Precompute corridor or stop-cluster profiles (community-summary analogue) over the 2,871-stop / 4,526-edge network.
   - Allow retrieval of the *k next stops* on the route, not just the next one.
   - Replace or augment raw `curr_node_id`/`next_node_id` with embeddings, since Dong & Gayah found that raw stop ids did not help while semantic stop information did.

8. **Multi-agent retrieval design (MAICC, M-RAG, MD-MADDPG).**
   - Options:
     - (i) one shared RLChunk policy with an agent or route id in its query (parameter sharing, like MAPPO);
     - (ii) partitioned corpora per route or time band (M-RAG);
     - (iii) route-level shared context that all buses on a route see, acting as implicit communication.
   - Use team-level signals for credit assignment (MAICC's hybrid utility).

9. **Prevent temporal leakage.** Split SUNT data by *day* (train, validation, test). Index only training days for retrieval, and never retrieve records from the simulated day itself or later. Offline RL work on logs (Offline-GMAC) faces the same distribution and leakage issues.

10. **Use LLMs where they help without slowing the loop.**
    - Offline reward-weight design or tuning for the multi-objective reward (Yu et al. 2025).
    - Offline semantic stop embeddings (Dong & Gayah 2026).
    - Post-hoc explanation or sanity-checking of logged bus actions (iLLM-TSC-style verifier, used only for analysis).
    - Avoid LLM-as-policy (LLMLight, REMEMBERER, REG-TSC) in BusEnv, given fleet size and step counts.

11. **Evaluation protocol.**
    - Baselines: no retrieval; current hand-crafted features; fixed top-k; random-k; Adaptive-RAG-style rule gating; NEC-style kNN episodic features; static semantic embeddings (Dong & Gayah); RLChunk; oracle (true future demand or travel time).
    - Metrics: BusEnv's four objectives (occupancy band, uptime, headway synchronisation, efficiency), plus headway coefficient of variation, bunching count, waiting time, chunks selected per step, and wall-clock overhead per step.
    - Test robustness under held-out days, rain days and demand surges, following the disruption tests in IQNC-M.

---

## References

All entries were verified against arXiv, Semantic Scholar, Crossref or OpenAlex unless marked.

1. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. arXiv:2310.11511.
2. Blundell, C., Uria, B., Pritzel, A., Li, Y., Ruderman, A., Leibo, J. Z., Rae, J., Wierstra, D., et al. *Model-Free Episodic Control.* arXiv:1606.04460, 2016 (preprint).
3. Cao, Y., Zhao, H., Cheng, Y., Shu, T., Chen, Y., Liu, G., Liang, G., Zhao, J., et al. *Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods.* IEEE TNNLS. DOI:10.1109/TNNLS.2024.3497992. arXiv:2404.00282.
4. Chen, S.-A., Miculicich, L., Eisenschlos, J. M., Wang, Z., Wang, Z., Chen, Y., Fujii, Y., Lin, H.-T., et al. *TableRAG: Million-Token Table Understanding with Language Models.* NeurIPS 2024. arXiv:2410.04739.
5. Chen, W., Zhou, K., Chen, C. *Real-time bus holding control on a transit corridor based on multi-agent reinforcement learning.* IEEE ITSC 2016. DOI:10.1109/ITSC.2016.7795538.
6. Chen, Y., Yan, L., Sun, W., Ma, X., Zhang, Y., Wang, S., Yin, D., Yang, Y., et al. *Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning* (MMOA-RAG). NeurIPS 2025 (per arXiv comment). arXiv:2501.15228.
7. Da, L., Liou, K., Chen, T., Zhou, X., Luo, X., Yang, Y., Wei, H. *Open-TI: Open Traffic Intelligence with Augmented Language Model.* Int. J. Machine Learning and Cybernetics. DOI:10.1007/s13042-024-02190-8. arXiv:2401.00211.
8. Devunuri, S., Lehe, L. *TransitGPT: A Generative AI-based Framework for Interacting with GTFS Data Using Large Language Models.* arXiv:2412.06831. *Public Transport*, DOI:10.1007/s12469-025-00395-w (per Semantic Scholar; not independently confirmed).
9. Devunuri, S., Qiam, S., Lehe, L. *ChatGPT for GTFS: Benchmarking LLMs on GTFS Understanding and Retrieval.* arXiv:2308.02618 (journal: *Public Transport* per arXiv comment; DOI not verified).
10. Dong, X., Gayah, V. V. *Mitigating Bus Bunching with Reinforcement Learning Enhanced by Semantic Stop Embedding.* arXiv:2608.10207, 2026 (preprint).
11. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130, 2024 (preprint).
12. Foerster, J. N., Assael, Y. M., de Freitas, N., Whiteson, S. *Learning to Communicate with Deep Multi-Agent Reinforcement Learning.* NeurIPS 2016. arXiv:1605.06676.
13. Gade, A., Jetcheva, J. *It's About Time: Incorporating Temporality in Retrieval Augmented Language Models.* arXiv:2401.13222, 2024 (preprint).
14. Goyal, A., Friesen, A. L., Banino, A., Weber, T., Ke, N. R., Badia, A. P., Guez, A., Mirza, M., et al. *Retrieval-Augmented Reinforcement Learning.* ICML 2022. arXiv:2202.08417.
15. Hu, S., Zhong, Y., Gao, M., Wang, W., Dong, H., Liang, X., Li, Z., Chang, X., et al. *MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library.* arXiv:2210.13708 (venue not verified).
16. Humphreys, P. C., Guez, A., Tieleman, O., Sifre, L., Weber, T., Lillicrap, T. *Large-Scale Retrieval for Reinforcement Learning.* NeurIPS 2022. arXiv:2206.05314.
17. Jeong, S., Baek, J., Cho, S., Hwang, S. J., Park, J. C. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL 2024. arXiv:2403.14403.
18. Jiang, T., Lin, Z., Li, L., Li, Y.-C., Guan, C., Yuan, L., Zhang, Z., Yu, Y., et al. *Multi-agent In-context Coordination via Decentralized Memory Retrieval* (MAICC). AAAI 2026, 40(27):22363–22371. DOI:10.1609/aaai.v40i27.39394. arXiv:2511.10030.
19. Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., et al. *Active Retrieval Augmented Generation* (FLARE). EMNLP 2023. arXiv:2305.06983.
20. Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., Han, J. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.* arXiv:2503.09516, 2025 (preprint; venue not verified).
21. Lai, S., Xu, Z., Zhang, W., Liu, H., Xiong, H. *LLMLight: Large Language Models as Traffic Signal Control Agents.* KDD 2025. DOI:10.1145/3690624.3709379. arXiv:2312.16044.
22. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401.
23. Li, X., Guo, Q., Chen, J., Guo, Z., Xu, S., Li, L., Zhang, L. *Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles* (REG-TSC). arXiv:2510.26242, 2025 (preprint).
24. Ma, X., Gong, Y., He, P., Zhao, H., Duan, N. *Query Rewriting for Retrieval-Augmented Large Language Models.* EMNLP 2023. arXiv:2305.14283.
25. Menda, K., Chen, Y.-C., Grana, J., Bono, J. W., Tracey, B. D., Kochenderfer, M. J. *Deep Reinforcement Learning for Event-Driven Multi-Agent Decision Processes.* IEEE T-ITS, 2019. DOI:10.1109/TITS.2018.2848264.
26. Alesiani, F., Gkiotsalitis, K. *Reinforcement Learning-Based Bus Holding for High-Frequency Services.* IEEE ITSC 2018. DOI:10.1109/ITSC.2018.8569473.
27. Nie, T., Sun, J., Ma, W. *Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap.* Artificial Intelligence for Transportation, 2025. DOI:10.1016/j.ait.2025.100003.
28. Pang, A., Wang, M., Pun, M.-O., Chen, C. S., Xiong, X. *iLLM-TSC: Integration Reinforcement Learning and Large Language Model for Traffic Signal Control Policy Improvement.* arXiv:2407.06025 (IEEE TVT per Semantic Scholar, DOI 10.1109/TVT.2026.3674284; not independently confirmed).
29. Park, J., Cho, S., Lee, J.-Y. *Stop-RAG: Value-Based Retrieval Control for Iterative RAG.* NeurIPS 2025 MTI-LLM Workshop. arXiv:2510.14337.
30. Pesce, E., Montana, G. *Improving Coordination in Small-Scale Multi-Agent Deep Reinforcement Learning through Memory-driven Communication.* Machine Learning, 2020. DOI:10.1007/s10994-019-05864-5. arXiv:1901.03887.
31. Pritzel, A., Uria, B., Srinivasan, S., Puigdomènech, A., Vinyals, O., Hassabis, D., Wierstra, D., Blundell, C. *Neural Episodic Control.* ICML 2017. arXiv:1703.01988.
32. Schmied, T., Paischer, F., Patil, V., Hofmarcher, M., Pascanu, R., Hochreiter, S. *Retrieval-Augmented Decision Transformer: External Memory for In-context RL.* arXiv:2410.07071, 2024 (venue not verified).
33. Shen, M., Bi, X., Gu, W., Xu, Z., Wu, Y., Xiao, F., Li, L. *From Observation to Intervention: Mastering Bus Holding Control from Historical Uncontrolled Trajectories with Offline Reinforcement Learning.* arXiv:2610.14632, 2026 (preprint).
34. Sukhbaatar, S., Szlam, A., Fergus, R. *Learning Multiagent Communication with Backpropagation.* NIPS 2016. arXiv:1605.07736.
35. Sun, C., Huang, S., Pompili, D. *LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions.* arXiv:2405.11106, 2024 (preprint).
36. Sun, J., Zhong, X., Zhou, S., Han, J. *DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation.* arXiv:2505.07233, 2025 (venue not verified).
37. Tire, K., Taga, E. O., Ildiz, M. E., Oymak, S. *Retrieval Augmented Time Series Forecasting.* arXiv:2411.08249, 2024 (preprint).
38. Wang, J., Sun, L. *Dynamic holding control to avoid bus bunching: A multi-agent deep reinforcement learning framework.* Transportation Research Part C, 2020. DOI:10.1016/j.trc.2020.102661.
39. Wang, J., Sun, L. *Reducing Bus Bunching with Asynchronous Multi-Agent Reinforcement Learning.* IJCAI 2021. DOI:10.24963/ijcai.2021/60. arXiv:2105.00376.
40. Wang, J., Sun, L. *Robust Dynamic Bus Control: A Distributional Multi-agent Reinforcement Learning Approach.* IEEE T-ITS. DOI:10.1109/TITS.2022.3229527. arXiv:2111.01946.
41. Wang, M., Pang, A., Kan, Y., Pun, M.-O., Chen, C. S., Huang, B. *LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments.* arXiv:2403.08337, 2024 (venue not verified).
42. Wang, Z., Teo, S. X., Ouyang, J., Xu, Y., Shi, W. *M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions.* ACL 2024. arXiv:2405.16420.
43. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., Wu, Y. *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games.* NeurIPS 2022 Datasets & Benchmarks. arXiv:2103.01955.
44. Yu, J., Wang, Y., Ma, W. *Large language model-enhanced reinforcement learning for generic bus holding control strategies.* Transportation Research Part E, 200, 2025. DOI:10.1016/j.tre.2025.104142. arXiv:2410.10212.
45. Zhang, D., Chen, L., Zhang, S., Xu, H., Zhao, Z., Yu, K. *Large Language Models Are Semi-Parametric Reinforcement Learning Agents* (REMEMBERER). NeurIPS 2023. arXiv:2306.07929.
46. Zhang, S., Fu, D., Liang, W., Zhang, Z., Yu, B., Cai, P. *TrafficGPT: Viewing, processing and interacting with traffic foundation models.* Transport Policy, 2024. DOI:10.1016/j.tranpol.2024.03.006. arXiv:2309.06719.
47. Zhu, S., Xu, C., Ke, K., Yu, C. *Context-Picker: Dynamic Context Selection Using Multi-Stage Reinforcement Learning.* arXiv:2512.14465, 2025 (preprint).
