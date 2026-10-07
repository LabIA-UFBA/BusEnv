# CHiP-MARL — Contextual Hybrid Prioritization for Multi-Objective MARL

CHiP-MARL is a **contextual hybrid reward aggregation mechanism** for multi-objective multi-agent reinforcement learning (MARL). It combines conventional weighted scalarization with a **tolerance-based lexicographic component**, allowing explicit objective priorities to be incorporated while preserving compensatory trade-offs among objectives.

The mechanism operates on a normalized multi-objective vector and produces a scalar learning signal that can be consumed by existing MARL algorithms.

---

## Overview

Multi-objective reinforcement learning commonly requires several performance dimensions to be represented by a single scalar reward. A weighted sum provides a simple and continuous trade-off, but it may allow an important objective to be sacrificed when improvements in other objectives compensate for its degradation.

CHiP-MARL addresses this limitation by combining two complementary components:

- **Weighted component** — captures continuous trade-offs among objectives.
- **Tolerance-based lexicographic component** — preserves an explicit priority ordering while allowing admissible deviations before a priority violation is penalized.

The resulting reward is:

$$
R_{\mathrm{CHiP}}
=
\lambda_c R_{\mathrm{prio}}
+
(1-\lambda_c)R_w
$$

where:

- $R_w$ is the normalized weighted scalarization;
- $R_{\mathrm{prio}}$ is the tolerance-based lexicographic score;
- $\lambda_c \in [0,1]$ controls the relative contribution of both components;
- $c$ denotes the current operating context.

---

## Key Features

- **Hybrid reward aggregation** combining weighted and lexicographic preferences.
- **Tolerance-based prioritization**, preventing small objective deviations from being treated as immediate violations.
- **Context-dependent priorities**, allowing the relative importance of objectives to change with operating conditions.
- **Normalized objective space**, with all objectives represented on the $[0,1]$ scale.
- **Compatibility with existing MARL algorithms**, since CHiP-MARL provides a scalar learning signal rather than a new policy optimization algorithm.
- **Multiple operating contexts**, each associated with its own objective priority ordering.
- **Objective-space evaluation**, enabling comparison of scalar reward and the underlying multi-objective behavior.

---

## Reward Formulation

### Objective Vector

At each simulation step, the system produces a normalized objective vector

$$
\mathbf{v}_t =
\left(
 v_{\mathrm{occ}},
 v_{\mathrm{uptime}},
 v_{\mathrm{sync}},
 v_{\mathrm{eff}}
\right),
$$

where larger values indicate better performance.

The four objectives represent:

| Objective | Description |
|---|---|
| **Occupancy** | Vehicle occupancy relative to the desired operating range |
| **Uptime** | Operational availability of the vehicle |
| **Synchronization** | Service regularity and headway synchronization |
| **Efficiency** | Operational efficiency, including travel-time performance |

The objective values are computed independently of the aggregation mechanism. Therefore, the same objective vector can be evaluated using conventional weighted scalarization or CHiP-MARL.

---

## Weighted Component

The first component uses conventional normalized weighted scalarization:

$$
R_w(\mathbf{v}_t)
=
\frac{
\sum_{j=1}^{m} w_j v_{t,j}
}{
\sum_{j=1}^{m} w_j
},
$$

where $w_j \geq 0$ and $\sum_j w_j > 0$.

This component provides a smooth and compensatory preference model: an improvement in one objective can compensate for a reduction in another according to the corresponding weights.

---

## Tolerance-Based Lexicographic Component

CHiP-MARL introduces an explicit priority structure through a tolerance-based lexicographic mechanism.

For each objective, the deviation from its ideal normalized value is computed as:

$$
\Delta_i = 1-v_i.
$$

For a given context $c$, objectives are inspected according to the priority ordering

$$
\Pi_c=(o_1,o_2,\ldots,o_m),
$$

where $o_1$ is the highest-priority objective.

The mechanism identifies the first objective whose deviation exceeds its contextual tolerance:

$$
k =
\min
\left\{
 i :
 \Delta_{o_i} > \epsilon_{c,o_i}
\right\}.
$$

If a violation occurs, its normalized severity is:

$$
\delta_k =
\min
\left(
\frac{
\Delta_{o_k}-\epsilon_{c,o_k}
}{
1-\epsilon_{c,o_k}
},
1
\right).
$$

The lexicographic component is then:

$$
R_{\mathrm{prio}}
=
\frac{k-\delta_k}{m}.
$$

When no objective exceeds its corresponding tolerance:

$$
R_{\mathrm{prio}}=1.
$$

### Interpretation

The tolerance does **not** act as another weight in the weighted average. Instead, it defines an admissible deviation from the ideal value.

Consequently, a higher-priority objective does not immediately dominate the reward because of small variations. Only when its deviation exceeds the specified tolerance does the lexicographic mechanism register a priority violation.

This results in a **soft lexicographic preference model**, combining hierarchical prioritization with operational flexibility.

---

## Hybrid Reward

The final CHiP-MARL reward is:

$$
R_{\mathrm{CHiP}}
=
\lambda_c R_{\mathrm{prio}}
+
(1-\lambda_c)R_w.
$$

The coefficient $\lambda_c$ controls the relative influence of the two aggregation mechanisms:

| $\lambda_c$ | Interpretation |
|---|---|
| $0$ | Pure weighted scalarization |
| $0<\lambda_c<1$ | Hybrid aggregation |
| $1$ | Pure tolerance-based lexicographic aggregation |

In the current configuration, $\lambda_c=0.5$, giving equal influence to the weighted and lexicographic components.

---

## Context-Dependent Preferences

CHiP-MARL supports a finite set of operating contexts:

$$
\mathcal{C} =
\{
 c_{\mathrm{normal}},
 c_{\mathrm{rain}},
 c_{\mathrm{works}},
 c_{\mathrm{works+rain}}
\}.
$$

These contexts represent normal operation, rainfall, road works, and the simultaneous occurrence of rainfall and road works.

Each context defines a preference specification:

$$
\Theta_c =
\left(
\Pi_c,
\boldsymbol{\epsilon}_c,
\lambda_c
\right).
$$

The supported priority specifications are:

| Context | Priority ordering |
|---|---|
| **Normal** | Occupancy $\succ$ Synchronization $\succ$ Efficiency $\succ$ Uptime |
| **Rain** | Efficiency $\succ$ Occupancy $\succ$ Synchronization $\succ$ Uptime |
| **Road works** | Synchronization $\succ$ Efficiency $\succ$ Occupancy $\succ$ Uptime |
| **Rain + road works** | Efficiency $\succ$ Synchronization $\succ$ Occupancy $\succ$ Uptime |

The tolerance values are defined on the same normalized $[0,1]$ scale as the objective scores:

$$
\epsilon_{\mathrm{occ}}=0.50,
\qquad
\epsilon_{\mathrm{uptime}}=0.17,
\qquad
\epsilon_{\mathrm{sync}}=0.78,
\qquad
\epsilon_{\mathrm{eff}}=0.65.
$$

These tolerance values are shared across the operating contexts, and $\lambda_c=0.5$ is used for every context.

At each simulation step, the current operating context is externally determined from the prevailing operating conditions and provided to the reward mechanism. The corresponding priority ordering is then used to evaluate the objective deviations.

---

## Preference Specification

The complete contextual preference configuration is:

$$
\Theta_c =
\left(
\Pi_c,
\boldsymbol{\epsilon}_c,
\lambda_c
\right).
$$

This separates three concepts:

1. **Priority** — which objective is considered first.
2. **Tolerance** — how much deviation from the ideal value is admissible.
3. **Mixing coefficient** — how strongly the lexicographic component influences the final scalar reward.

Context therefore changes the prioritization applied to the same objective vector rather than redefining the objectives themselves.

---

## Integration with MARL

CHiP-MARL does not introduce a new MARL optimization algorithm. Instead, it acts as a **reward aggregation layer** between the environment and an existing MARL method.

At each step:

$$
\mathbf{v}_t
\xrightarrow{\mathbf{w}}
R_w,
$$

$$
(\mathbf{v}_t,\Pi_c,\boldsymbol{\epsilon}_c)
\xrightarrow{}
R_{\mathrm{prio}},
$$

$$
(R_w,R_{\mathrm{prio}},\lambda_c)
\xrightarrow{}
R_{\mathrm{CHiP}}.
$$

The resulting scalar reward is supplied to the learning algorithm, which can continue to optimize its conventional discounted return.

This design makes CHiP-MARL independent of the underlying policy optimization procedure and allows it to be combined with different MARL algorithms.


---

## Project Structure

A typical project structure is:

```text
src/
├─ envs/                       # Multi-agent environment
├─ pipelines/                 # Data processing and experiment pipelines
├─ tools/                     # Data utilities and analysis
├─ models/                    # MARL models and policy components
├─ training/                  # Training entrypoints and configurations
├─ tests/                     # Automated tests
└─ viz/                       # Visualization and replay utilities

replays/                      # Generated replay files and viewers
logs/                         # Experimental outputs
```

The exact directory structure may vary depending on the experiment or implementation branch.

---

## Installation

The project can be used with a Conda-based Python environment.

### 1. Create the environment

```bash
conda create -n chip-marl python=3.8 -y
conda activate chip-marl
```

### 2. Upgrade the required packaging tools

```bash
python -m pip install --upgrade \
    "pip==21.0" \
    "setuptools==65.5.0" \
    "wheel==0.38.0"
```

### 3. Install the required MARL framework

```bash
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib

python -m pip install -r requirements.txt

cd marllib/patch
python add_patch.py -y
cd ../..

python -m pip install marllib
```

### 4. Install the project

```bash
python -m pip install -e ".[rllib,data,viz,test]"
```

For environments requiring the legacy dependency stack:

```bash
python -m pip install "gym==0.20.0"
python -m pip install "protobuf>=3.19.0,<3.21.0"
python -m pip install "pydantic==1.10.13"
```

### 5. Verify the installation

```bash
python --version
python -m pip --version
pytest -q
```

---

## Training

Training commands depend on the selected MARL algorithm and experiment configuration.

Examples:

```bash
# Show available MARLlib training options
marllib train-marllib-a2c -- --help

# Run a custom training configuration
marllib train-custom-a2c -- --help

# Run the standard training entrypoint
marllib train
```

For multiple algorithms or configurations:

```bash
bash run_parallel_train.sh
```

The same CHiP-MARL reward mechanism can be supplied to different MARL algorithms, allowing the effect of reward aggregation to be studied independently of the policy optimization method.

---

## Analysis and Evaluation

CHiP-MARL experiments should be evaluated in the **original objective space** in addition to scalar return.

Recommended metrics include:

- normalized objective means;
- standard deviation across independent runs;
- area under the learning curve (AUC);
- scalar reward and AUC;
- comparisons between CHiP-MARL and weighted scalarization;
- objective-wise trade-offs.

Reporting objective-level results is important because two aggregation mechanisms can produce different scalar rewards while yielding similar objective vectors.

---

## Research Use

CHiP-MARL is intended for research involving:

- multi-objective MARL;
- preference-aware reward aggregation;
- contextual objective prioritization;
- tolerance-based lexicographic methods;
- comparisons between weighted and hierarchical preference models.

The method is particularly suited to scenarios in which the relative importance of objectives may change according to operating conditions.

---


