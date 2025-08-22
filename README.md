# 🚍 Graph Exploration – Multi-Agent Urban Bus Simulation (Modular)

This project provides a **multi-agent reinforcement learning environment** for urban bus operations, based on real-world data from the **Salvador Urban Network Transportation (SUNT)** system.  
It has been refactored into a **modular package** with a clean structure, clear separation of concerns, and a unified CLI.

---

## 🚌 Overview

The **Multi-Agent Urban Bus Simulation Environment** is built on top of real public transportation data from Salvador (Brazil).  
It simulates the operation of multiple buses as independent agents navigating a real transit network, enabling the development and testing of intelligent control strategies for public transport.

Key aspects:
- Realistic, **data-driven** training scenarios.  
- Focus on **optimizing service efficiency** and **passenger experience**.  
- Uses **boarding, alighting, and travel time data** from actual operations.  

---

## 🎯 Objectives

Agents (buses) are trained to:
- Reduce passenger waiting time at stops.  
- Maintain regular headways (time between buses).  
- Balance occupancy (avoid overcrowding or running empty).  
- Operate efficiently regarding energy and maintenance.  

The system applies **Multi-Agent Reinforcement Learning (MARL)**, where each bus acts autonomously but cooperates implicitly through a **shared reward function**.

---

## 📊 Observations

During training, the environment generates key metrics such as:

- **avg_travel_time_AB** → Average travel time between reference stops.  
- **future_demand_at_B** → Predicted passenger demand at stop B.  
- **occupancy_rate** → Proportion of bus capacity in use.  
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
- **Gymnasium** → Standard API.  

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
src/
├─ envs/                        # PettingZoo environments
├─ pipelines/                   # observations, routes, stats, RLlib training
├─ tools/                       # data utilities and analysis
├─ viz/                         # graph visualization
├─ tests/                       # automated tests
├─ training_observation/        # training observations (unzip real_routes.zip here)
├─ output_observation_travel_time_sum_amout/  # experimental outputs
└─ __pycache__/                 # python cache
```

- **CLI** exposes subcommands mapping to these modules.  
- Some scripts still use **hardcoded paths** → recommended to migrate to configs or `.env`.  

---

## ⚡ Installation & Usage

```bash
# 1. Create a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2. Upgrade basic tools
pip install --upgrade pip setuptools

# 3. Install dependencies in editable mode
pip install -e ".[rllib,data,viz,test]"

# 4. (Linux / macOS) Export PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

#    (Windows PowerShell)
$env:PYTHONPATH = (Get-Location).Path + ";" + $env:PYTHONPATH

# 5. Unzip the real route data (required for training)
unzip training_observation/real_routes.zip -d training_observation/

# 6. Run tests
pytest -q
```

### CLI (`graphx`)

```bash
# RLlib training (reinforcement learning experiments)
graphx train -- --help

# Dataset statistics (mean, std, etc.)
graphx stats -- --help

# Dataset size and item counts
graphx look-amount -- --help

# Compute averages across PKL files
graphx pkl-medias -- --help

# Explore and analyze route files
graphx see-routes -- --help

# View the content of PKL files interactively
graphx view-pkl -- --help

# Visualize graphs
graphx view-graph -- --help

# Run the SUNT environment entrypoint
graphx env-sunt --
```

---

