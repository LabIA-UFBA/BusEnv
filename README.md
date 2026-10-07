# CHiP-MARL — Contextual Hybrid Prioritization for Multi-Objective MARL

CHiP-MARL is a **contextual hybrid reward aggregation mechanism** for multi-objective multi-agent reinforcement learning (MARL). It combines conventional weighted scalarization with a **tolerance-based lexicographic component**, so explicit objective priorities can be enforced while compensatory trade-offs among objectives are preserved.

CHiP-MARL is not a new MARL algorithm. It is a reward layer between the environment and an existing MARL method: it receives a normalized multi-objective vector and returns a scalar reward.

- **Weighted component**: continuous, compensatory trade-offs among objectives.
- **Tolerance-based lexicographic component**: an explicit priority ordering, with admissible deviations before a priority violation is penalized.
- **Context-dependent priorities**: the priority ordering changes with the operating conditions.
- **Objective-space evaluation**: results can be compared both in scalar reward and in the underlying objectives.

---

## Reward Formulation

### Objectives

At each step, the environment produces a normalized objective vector $\mathbf{v}_t = (v_{\mathrm{occ}}, v_{\mathrm{uptime}}, v_{\mathrm{sync}}, v_{\mathrm{eff}})$ in $[0,1]^4$, where larger values are better. The objectives are computed independently of the aggregation mechanism, so the same vector can be aggregated by weighted scalarization or by CHiP-MARL.

| Objective | Description |
|---|---|
| **Occupancy** | Vehicle occupancy relative to the desired operating range |
| **Uptime** | Operational availability of the vehicle |
| **Synchronization** | Service regularity and headway synchronization |
| **Efficiency** | Operational efficiency, including travel-time performance |

### Weighted component

$$
R_w(\mathbf{v}_t) = \frac{\sum_{j=1}^{m} w_j\, v_{t,j}}{\sum_{j=1}^{m} w_j}, \qquad w_j \ge 0,\ \sum_j w_j > 0,
$$

where $m$ is the number of objectives. An improvement in one objective can compensate for a reduction in another, according to the weights.

### Tolerance-based lexicographic component

The deviation of each objective from its ideal value is $\Delta_j = 1 - v_j$. For context $c$, objectives are inspected in the priority order $\Pi_c = (o_1, \ldots, o_m)$, where $o_1$ has the highest priority. Each objective has a tolerance $\epsilon_j \in [0,1)$.

The first priority violation is

$$
k = \min \{\, i : \Delta_{o_i} > \epsilon_{o_i} \,\}.
$$

If no objective exceeds its tolerance, $R_{\mathrm{prio}} = 1$. Otherwise, the normalized severity of the violation is

$$
\delta_k = \min\!\left( \frac{\Delta_{o_k} - \epsilon_{o_k}}{1 - \epsilon_{o_k}},\ 1 \right),
\qquad
R_{\mathrm{prio}} = \frac{k - \delta_k}{m}.
$$

The tolerance is **not** a weight and does not enter the weighted average. It only defines how much deviation from the ideal value is admissible before an objective counts as violated, which gives a *soft* lexicographic preference: small variations in a high-priority objective do not dominate the reward.

### Hybrid reward

$$
R_{\mathrm{CHiP}} = \lambda\, R_{\mathrm{prio}} + (1 - \lambda)\, R_w .
$$

| $\lambda$ | Interpretation |
|---|---|
| $0$ | Pure weighted scalarization |
| $0 < \lambda < 1$ | Hybrid aggregation |
| $1$ | Pure tolerance-based lexicographic aggregation |

The current configuration uses $\lambda = 0.5$, which gives equal influence to both components.

---

## Operating Contexts

CHiP-MARL supports four operating contexts: normal operation, rain, road works, and rain with road works,

$$
\mathcal{C} = \{ c_{\mathrm{normal}},\ c_{\mathrm{rain}},\ c_{\mathrm{works}},\ c_{\mathrm{works+rain}} \}.
$$

Each context $c$ has a preference specification $\Theta_c = (\Pi_c, \boldsymbol{\epsilon}, \lambda)$. Only the priority ordering $\Pi_c$ depends on the context; the tolerances $\boldsymbol{\epsilon}$ and the coefficient $\lambda$ are shared by all contexts.

| Context | Priority ordering $\Pi_c$ |
|---|---|
| **Normal** | Occupancy $\succ$ Synchronization $\succ$ Efficiency $\succ$ Uptime |
| **Rain** | Efficiency $\succ$ Occupancy $\succ$ Synchronization $\succ$ Uptime |
| **Road works** | Synchronization $\succ$ Efficiency $\succ$ Occupancy $\succ$ Uptime |
| **Rain + road works** | Efficiency $\succ$ Synchronization $\succ$ Occupancy $\succ$ Uptime |

Tolerances (same $[0,1]$ scale as the objective scores):

$$
\epsilon_{\mathrm{occ}} = 0.50, \quad \epsilon_{\mathrm{uptime}} = 0.17, \quad \epsilon_{\mathrm{sync}} = 0.78, \quad \epsilon_{\mathrm{eff}} = 0.65 .
$$

At each simulation step, the active context is determined externally from the prevailing operating conditions and given to the reward mechanism, which applies the corresponding ordering $\Pi_c$. Context therefore changes how the same objective vector is prioritized, not how the objectives are defined.

---

## Integration with MARL

At each step, the learner receives a scalar reward and keeps optimizing its usual discounted return:

$$
\mathbf{v}_t \;\rightarrow\; R_w, \qquad
(\mathbf{v}_t, \Pi_c, \boldsymbol{\epsilon}) \;\rightarrow\; R_{\mathrm{prio}}, \qquad
(R_w, R_{\mathrm{prio}}, \lambda) \;\rightarrow\; R_{\mathrm{CHiP}} .
$$

Because the mechanism is independent of the policy optimization procedure, it can be combined with different MARL algorithms.

---

## Project Structure

```text
src/
├─ envs/          # Multi-agent environment
├─ pipelines/     # Data processing and experiment pipelines
├─ tools/         # Data utilities and analysis
├─ models/        # MARL models and policy components
├─ training/      # Training entrypoints and configurations
├─ tests/         # Automated tests
└─ viz/           # Visualization and replay utilities

replays/          # Generated replay files and viewers
logs/             # Experimental outputs
```

The exact structure may vary between experiments and branches.

---

## Installation

### 1. Create the environment

```bash
conda create -n chip-marl python=3.8 -y
conda activate chip-marl

python -m pip install --upgrade \
    "pip==21.0" "setuptools==65.5.0" "wheel==0.38.0"
```

### 2. Install MARLlib

Run this **outside** the CHiP-MARL directory.

```bash
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib

python -m pip install -r requirements.txt

cd marllib/patch
python add_patch.py -y
cd ../..

python -m pip install marllib
```

If your environment needs the legacy dependency stack:

```bash
python -m pip install "gym==0.20.0"
python -m pip install "protobuf>=3.19.0,<3.21.0"
python -m pip install "pydantic==1.10.13"
```

### 3. Install CHiP-MARL

Go back to the root of this repository, then:

```bash
python -m pip install -e ".[rllib,data,viz,test]"
```

### 4. Verify

```bash
python --version
pytest -q
```

---

## Training

```bash
# Available MARLlib training options
marllib train-marllib-a2c -- --help

# Custom training configuration
marllib train-custom-a2c -- --help

# Several algorithms / configurations
bash run_parallel_train.sh
```

The same CHiP-MARL reward can be given to different MARL algorithms, so the effect of reward aggregation can be studied independently of the policy optimization method.

---

## Evaluation

CHiP-MARL should be evaluated in the **original objective space**, in addition to scalar return, because two aggregation mechanisms can give different scalar rewards while producing similar objective vectors. Recommended reporting:

- mean of each normalized objective and its standard deviation across independent runs;
- scalar return and its normalized area under the learning curve (AUC);
- comparison between CHiP-MARL and weighted scalarization, per objective.
