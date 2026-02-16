# 🚍 Bus Env – Multi-Agent Urban Bus Simulation (Modular)

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
## 1) Create and activate the environment
conda create -n marllib python=3.8 -y
conda activate marllib

## 2) Confirm that we are using the environment's Python/Pip
which python
python --version
python -m pip --version

## 3) Adjust the tools in the marllib env (uses the env's own pip)
python -m pip install --upgrade "pip==21.0" "setuptools==65.5.0" "wheel==0.38.0"

## 4) Gym compatible (old API)
python -m pip install "gym==0.20.0"

## 5. Clone MARLlib
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib

## 6. Install MARLlib dependencies
python -m pip install -r requirements.txt

## 7. Apply patches
cd marllib/patch
python add_patch.py -y
cd ../..

## 8. install MARLlib
python -m pip install marllib
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

## 9. Install your project in editable mode with extras
python -m pip install -e ".[rllib,data,viz,test]"

## 10. Fix protobuf version for Ray/RLlib
python -m pip install "protobuf>=3.19.0,<3.21.0"
pip install "pydantic==1.10.12"

## 11. adjust PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

## 12. Unpack route data
unzip src/training_observation/real_routes.zip -d src/training_observation/

## 13. Run tests
pytest -q

## 14. Place the configuration folder
mv src/sunt_bus.yaml MARLlib/marllib/envs/base_env/config/


## 15. [Extra] Run Custom 
If you need to run the custom model, go to the a2c.py file in the path "/MARLlib/marllib/marl/algos/core/IL" and make the following changes within this file:
1 - Add the import "from models.custom_a3c_torch_policy import CustomA3CTorchPolicy"
2 - Where it says "IA2CTorchPolicy = A3CTorchPolicy.with_updates" replace it with "IA2CTorchPolicy = CustomA3CTorchPolicy.with_updates"


## 16. Code Carbon
python -m pip install codecarbon
# 1) Stop Ray and any running nohup training processes
ray stop

# 2) Remove existing Pydantic (2.x) and its core module
python -m pip uninstall -y pydantic pydantic-core

# 3) Install Pydantic 1.10.x (compatible with Ray and MARLlib)
python -m pip install "pydantic==1.10.13"

# 4) (Optional) Install an older version of typing-extensions for compatibility
python -m pip install "typing-extensions<4.6" -q

# 5) (Optional) Reinstall CodeCarbon without dependencies to avoid upgrading Pydantic again
#    Using --no-deps ensures that no package updates Pydantic automatically.
python -m pip install --upgrade --no-deps codecarbon

# 6) Verify the installation
python - << 'PY'
import pydantic, ray
print("pydantic:", pydantic.__version__)
print("ray:", ray.__version__)
PY


```

### CLI (`graphx`)

```bash
# Train with MARLlib A2C (default)
marllib train-marllib-a2c -- --help

# Train with MARLlib custom A2C
marllib train-custom-a2c -- --help

# Train with RLlib
marllib train

# Train with a series of algs
bash run_parallel_train.sh

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

<img width="1261" height="619" alt="Real-world Transportation Data (3)" src="https://github.com/user-attachments/assets/12aab8c5-c712-4a18-95c5-1ad9872d2900" />
