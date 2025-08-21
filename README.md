# 🚍 Graph Exploration – Multi-Agent Urban Bus Simulation (Modular)

This project provides a **multi-agent reinforcement learning environment** for urban bus operations, based on real-world data from the **Salvador Urban Network Transportation (SUNT)** system.  
It has been refactored into a **modular package** with a clean structure, clear separation of concerns, and a unified CLI.

---

## 🚌 Overview

The **Multi-Agent Urban Bus Simulation Environment** is built on top of real public transportation data from Salvador (Brazil).  
It simulates the operation of multiple buses as independent agents navigating a real city transit network, enabling the development and testing of intelligent control strategies for public transport.

Key aspects:
- Realistic, **data-driven** training scenarios.
- Focus on **optimizing service efficiency** and **passenger experience**.
- Uses **boarding, alighting, and travel time data** from actual operations.

---

## 🎯 Objectives

Agents (buses) are trained to:
- Reduce passenger waiting time at stops.
- Maintain regular headways (time between buses).
- Balance occupancy (avoid overcrowding or emptiness).
- Operate efficiently regarding energy and maintenance.

The system applies **Multi-Agent Reinforcement Learning (MARL)**, where each bus acts autonomously but cooperates implicitly through a **shared reward function**.

---

## 📊 Observation Outputs

During training, the environment generates key metrics such as:

- **avg_travel_time_AB** → Average travel time between reference stops.  
- **future_demand_at_B** → Predicted passenger demand at stop B.  
- **occupancy_rate** → Proportion of current bus capacity in use.  
- **uptime_normalized** → Normalized availability of a bus in operation.  

These signals provide feedback to agents and can be used for both monitoring and reward shaping.

---

## 🎮 Actions

Each bus (agent) can choose among three actions:

- **WAIT** → Delay before continuing, to avoid clustering and improve headway.  
- **MOVE** → Proceed to the next stop.  
- **SERVICE_CENTER** → Divert to maintenance when required (low fuel or maintenance issues).  

---

## 🎯 Reward Function

The reward combines:
- Passenger service quality (shorter waits, demand satisfaction).  
- Operational efficiency (balanced occupancy, timely trips).  
- Maintenance/fuel management (penalties for ignoring issues).  
- Traffic flow & coordination (avoid idling or bus bunching).  

This ensures agents balance **service quality, fleet efficiency, and sustainability**.

---

## 🛠 Training Setup

The environment integrates:

- **Ray RLlib** → Distributed reinforcement learning.  
- **PettingZoo** → Multi-agent environment API.  
- **SuperSuit** → Wrappers for preprocessing.  
- **Gymnasium** → Standard action/observation API.  

**Configuration:**
- Each bus is an agent.
- Scenario based on real SUNT data (routes, stops, demand).  
- Each episode ≈ one simulated operational day.  
- PPO (Proximal Policy Optimization) with shared policy.  

**Scaling:**  
Supports **hundreds of agents in parallel**, leveraging Ray’s distributed training.

---

## 📂 Project Structure

```
src/sunt_training/
├─ __init__.py
├─ __main__.py
├─ cli.py                  # Unified CLI
├─ envs/
│  └─ sunt_env.py          # PettingZoo environment
├─ pipelines/
│  ├─ observations.py      # Observation generation
│  ├─ real_routes.py       # Real route generation
│  ├─ stats.py             # Metrics/statistics
│  └─ train_rllib.py       # RLlib training
├─ tools/
│  ├─ look_amount.py
│  ├─ pkl_medias.py
│  ├─ see_routes.py
│  └─ view_pkl.py
└─ viz/
   └─ view_graph.py
```

- **`legacy/`** → original scripts kept unchanged, for reference.  
- **CLI** exposes subcommands mapping to these modules.  
- Hardcoded paths from original scripts remain; consider replacing with `.env` or config files.  

---

## ⚡ Installation & Usage

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. Install dependencies in editable mode
pip install -e ".[rllib,data,viz]"

# 3. Run tests
pytest -q
```

### CLI usage
All commands are unified under `graphx`:

```bash
graphx train -- --help
graphx obs -- --help
graphx routes -- --help
graphx env-sunt --
```

Arguments after `--` are passed directly to the original scripts.

---

## ✅ Next Steps

- Replace **hardcoded paths** with configuration files or environment variables.  
- Extend **tests** to cover pipelines, tools, and environment logic.  
- Add support for **experiment tracking** (e.g., MLflow, Weights & Biases).  
- Modularize reward and observation design for more flexible experimentation.  
