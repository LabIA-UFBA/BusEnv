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

- **MARLlib** → Framework for MARL built on Ray RLlib.
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

Before proceeding, make sure you have **Conda** installed.  
👉 Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/download).  
On WSL/Linux, you can install Miniconda with:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
~/miniconda3/bin/conda init
exec "$SHELL"
```


```bash

## 1. create conda enviroment
conda create -n marllib python=3.8 -y
conda activate marllib

## 2. Adjust tools for MARLlib
pip install "pip==21" "setuptools==65.5.0" "wheel==0.38.0"
pip install "gym==0.20.0"

## 3. Clone MARLlib
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib

## 4. Install MARLlib dependencies
pip install -r requirements.txt

## 5. Apply patches
cd marllib/patch
python add_patch.py -y
cd ../..

## 6. install MARLlib
pip install marllib
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

## 7. Install your project in editable mode with extras
pip install -e ".[rllib,data,viz,test]"

## 8. Fix protobuf version for Ray/RLlib
pip install "protobuf>=3.19.0,<3.21.0"

## 9. adjust PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

## 10. Unpack route data
unzip src/training_observation/real_routes.zip -d src/training_observation/

## 11. Run tests
pytest -q

## 12. Place the configuration folder 
Place the sunt_bus.yaml file that is inside the src folder inside the config folder in the path "/MARLlib/marllib/envs/base_env/config"

## 13. [Extra] Run Custom 
If you need to run the custom environment, go to the a2c.py file in the path "/MARLlib/marllib/marl/algos/core/IL" and make the following changes within this file:
1 - Add the import "from models.custom_a3c_torch_policy import CustomA3CTorchPolicy"
2 - Where it says "IA2CTorchPolicy = A3CTorchPolicy.with_updates" replace it with "IA2CTorchPolicy = CustomA3CTorchPolicy.with_updates"

```

### CLI (`graphx`)

```bash
# Train with MARLlib A2C (default)
marllib train-marllib-a2c -- --help

# Train with MARLlib custom A2C
marllib train-custom-a2c -- --help

# Train with RLlib
marllib train -- --help

# Dataset statistics (mean, std, etc.)
marllib stats -- --help

# Dataset size and item counts
marllib look-amount -- --help

# Compute averages across PKL files
marllib pkl-medias -- --help

# Explore and analyze route files
marllib see-routes -- --help

# View the content of PKL files interactively
marllib view-pkl -- --help

# Visualize graphs
marllib view-graph -- --help

# Visualize a specific node information
marllib view-especific-node -- --help

# Visualize training metrics
marllib view-metrics -- --help

# Run the SUNT environment entrypoint
marllib env-sunt --


---

