# 🚍 BusEnv — Multi-Agent Reinforcement Learning for Urban Bus Fleet Control

BusEnv is a **multi-agent reinforcement learning (MARL) environment** for urban bus fleet control, built on real operational data from the **Salvador Urban Network Transportation (SUNT)** system (Salvador, Brazil). Buses are modeled as independent agents navigating a real GTFS transit network, with passenger demand, boarding/alighting, and travel times derived from real Automatic Vehicle Location (AVL) / Automatic Passenger Counting (APC)-style records rather than synthetic distributions.

The environment follows the [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` API and is trained via [Ray RLlib](https://docs.ray.io/en/releases-1.8.0/rllib.html) or [MARLlib](https://github.com/Replicable-MARL/MARLlib).

---

## Highlights

- **Data-driven passenger demand.** Boarding/alighting are not a fixed percentage of bus capacity: each (route, stop) maintains its own passenger queue, replenished at a real arrival rate derived from historical per-visit boarding counts and inter-visit intervals, and depleted when a bus actually boards passengers — two buses serving the same stop minutes apart are no longer credited with the same ridership.
- **Multi-agent coordination via headway awareness.** Each agent observes a signed, normalized gap to its immediate leading and following bus on the same route (not just a population-wide regularity statistic), and a matching reward term is weighted into the scalarized reward — enabling agents to react to bunching or excessive gaps.
- **Multi-objective reward.** Four weighted components — occupancy, uptime, headway synchronization, and travel-time efficiency — are computed every step (`getObjectives`) and can be scalarized (`scalarize`) or consumed directly as a reward vector for MORL algorithms.
- **Real network and routes.** 2,871 stops and 4,526 edges from Salvador's Feb-2024 GTFS feed, with 668 real historical trip patterns as candidate routes.
- **Interactive replay visualization.** A self-contained, browser-based viewer (real OpenStreetMap basemap, draggable timeline, editable route/stop labels, PNG export) for inspecting and presenting simulated episodes — see [`src/viz/README.md`](src/viz/README.md).

---

## Environment formulation

**Agents.** Each bus is an independent agent (`agent_0`, `agent_1`, …), assigned a fixed real route (an ordered sequence of GTFS stops) that it repeatedly traverses back and forth for the length of a simulated operational day (24h).

**Observation space** — `Box(12,)` per agent:

| # | Feature | Description |
|---|---|---|
| 0 | `time_of_day_norm` | Simulated time of day, normalized to [0, 1] |
| 1 | `occupancy_rate` | Current bus occupancy, fraction of capacity |
| 2 | `avg_travel_time_AB` | Normalized travel time to the next stop |
| 3 | `future_demand_at_B` | Historical boarding demand at the next stop |
| 4 | `uptime` | Normalized time since last maintenance |
| 5 | `maintenance_status` | 1.0 if operational, 0.0 otherwise |
| 6 | `curr_node_id` | Current stop (graph node index) |
| 7 | `next_node_id` | Next stop (graph node index) |
| 8 | `is_raining` | Weather flag (optional, data-driven) |
| 9 | `occ_risk_out` | Forecast risk of leaving the ideal occupancy band |
| 10 | `headway_leader_norm` | Signed, normalized time gap to the leading bus on the route |
| 11 | `headway_follower_norm` | Signed, normalized time gap to the following bus on the route |

**Action space** — `Discrete(3)`: `WAIT`, `MOVE` (advance to the next stop), `SERVICE_CENTER` (divert to a maintenance node).

**Reward** — a weighted combination of four objectives, each normalized to `[0, 1]`:

| Objective | What it measures |
|---|---|
| Occupancy | Deviation from an ideal occupancy band (default 0.6–0.9 of capacity) |
| Uptime | Time available before the next required maintenance |
| Synchronization | Pairwise headway regularity vs. a target interval (default 10 min) |
| Efficiency | Actual vs. expected travel time |

**Passenger demand model.** Each `(route, stop)` pair has its own passenger queue. Between visits, it accumulates at an arrival rate (passengers/second) derived from real historical data; when a bus arrives, boarding is `min(queue, remaining capacity)` and the queue depletes accordingly, while alighting reduces onboard occupancy by a historical per-stop fraction. See [Data & demand pipeline](#data--demand-pipeline) below.

---

## Data & demand pipeline

| Source | Content |
|---|---|
| `src/viz/graph_gtfs_fev_2024.gpickle` | Salvador GTFS network graph (Feb 2024): 2,871 stops with lat/lon, 4,526 edges with distance |
| `src/training_observation/real_routes.zip` | 668 real historical trip patterns (ordered stop sequences) |
| `src/training_observation/daily_may/` | Per-day travel-time / occupancy / uptime reference data |
| Raw AVL/APC-style parquet (`OD`, `Boarding`, `LTI`) | Per-visit boarding/alighting counts and timestamps, May 2024 (30 days) |

**`src/pipelines/generate_passenger_flow_stats.py`** aggregates the raw per-visit records into `stop_passenger_flow.pkl`: mean boardings, mean alighting fraction, and mean inter-visit interval, at a cascading granularity — `(route, stop, hour of day)` → `(route, stop)` → `(stop, hour of day)` → `(stop)` → a global fallback — so sparse combinations fall back to a broader, still-real estimate rather than an arbitrary default. Aggregation is a direct sum/count mean; no distribution-shape-sensitive outlier filtering is applied, since per-visit boarding/alighting counts are heavily zero-inflated and such filtering was found to systematically suppress real demand.

To refresh these statistics with new or additional data, re-run the same script — it auto-discovers every `od-YYYY-MM-DD.parquet` file under the configured data folder (no hardcoded date range):

```bash
python3 src/pipelines/generate_passenger_flow_stats.py --base-path /path/to/SUNT/tpm
```

---

## Visualization

`src/viz/replay_export.py` + `src/viz/render_replay.py` turn a recorded episode (`record_replay=True`) into a self-contained HTML viewer: real OpenStreetMap basemap, a draggable timeline with step-by-step event navigation, live per-stop queue and per-bus occupancy indicators, editable route/stop labels, and PNG export for figures. `src/pipelines/generate_eval_replay.py` generates the replay logs, either from a random (untrained) policy or a trained checkpoint, enabling side-by-side before/after-training comparisons on a shared map scale.

See **[`src/viz/README.md`](src/viz/README.md)** for the exact commands.

---

## 🎯 Objectives

Agents (buses) are trained to:
- Reduce passenger waiting time at stops.
- Maintain regular headways (time between buses).
- Balance occupancy (avoid overcrowding or running empty).
- Operate efficiently regarding energy and maintenance.

The system applies **Multi-Agent Reinforcement Learning (MARL)**, where each bus acts autonomously and can coordinate with nearby buses on its route through the headway-aware observation and reward described above.

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
- PPO (Proximal Policy Optimization), independent or shared policies depending on the training script.

**Scaling:**
Supports **hundreds of agents in parallel**, leveraging Ray's distributed training.

---

## 📂 Project Structure

```
src/
├─ envs/                        # PettingZoo environment (sunt_env.py)
├─ pipelines/                   # data generation, training entrypoints, replay generation
├─ tools/                       # data utilities and analysis
├─ viz/                         # graph visualization + interactive replay viewer
├─ models/                      # custom RLlib/MARLlib model + policy classes
├─ tests/                       # automated tests
├─ training_observation/        # precomputed reference data (unzip real_routes.zip here)
└─ output_observation_travel_time_sum_amout/  # experimental outputs
replays/                        # generated replay logs / scenes / HTML viewers (not tracked in git)
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

# Regenerate real-data passenger flow statistics (boarding/alighting/demand)
python3 src/pipelines/generate_passenger_flow_stats.py

# Generate a replay log + interactive HTML viewer (see src/viz/README.md)
python3 src/pipelines/generate_eval_replay.py --mode random --out replays/before.json
python3 src/viz/replay_export.py --replay replays/before.json --out replays/scene_before.json
python3 src/viz/render_replay.py --scene replays/scene_before.json -o replays/viewer.html
```

---

## Citation

If you use BusEnv in academic work, please cite this repository:

```bibtex
@misc{busenv2026,
  title        = {BusEnv: A Multi-Agent Reinforcement Learning Environment for Urban Bus Fleet Control},
  author       = {LabIA-UFBA},
  year         = {2026},
  howpublished = {\url{https://github.com/LabIA-UFBA/BusEnv}}
}
```

*(Replace with your paper's official reference once published.)*

## License

MIT — see `pyproject.toml`.
