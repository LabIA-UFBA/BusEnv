# Replay Viewer — Quick Start

Generates a self-contained HTML map (real OpenStreetMap tiles, draggable
timeline) that replays a simulated day of the bus fleet. This is the fastest
path to *see the simulation run*.

## 1. View the untrained agent (works right now, no model needed)

```bash
cd /mnt/ssd1/rafael/Graph_sunt_project/BusEnv

# 1a. Simulate one day with random actions, recording every step
/mnt/ssd1/conda/envs/marllib/bin/python3 src/pipelines/generate_eval_replay.py \
  --mode random --out replays/before.json --num-agents 25

# 1b. Convert the replay log into a map "scene"
python3 src/viz/replay_export.py --replay replays/before.json \
  --out replays/scene_before.json --label "Untrained baseline"

# 1c. Render the HTML viewer
python3 src/viz/render_replay.py --scene replays/scene_before.json \
  -o replays/viewer.html
```

Open `replays/viewer.html` in any browser (double-click it, or paste
`file:///mnt/ssd1/rafael/Graph_sunt_project/BusEnv/replays/viewer.html` into
the address bar). **Requires internet access** — it loads the map tiles and
two small JS libraries (Leaflet, html2canvas) from public CDNs.

Step 1a must run under the `marllib` conda env (it needs `pettingzoo`/`ray`/
`torch`); steps 1b/1c are plain Python and run with any `python3`.

## 2. View a trained agent ("after training")

**The checkpoints already on this machine
(`/mnt/ssd1/ray_results/{ippo,mappo,maa2c,itrpo,hatrpo,ia2c}_mlp_sunt_bus/...`)
were all trained on 2026-07-21, before this session's environment fixes, and
use a 10-dimensional observation space. The current environment now uses 12
dimensions (2 new headway-sync features) and different occupancy dynamics —
loading one of those old checkpoints against the current env will fail with
a shape mismatch. They cannot be used as-is.**

To get a real "after training" replay, train a fresh checkpoint against the
current env, then point the replay generator at it:

```bash
# Train (produces checkpoints under /mnt/ssd1/ray_results/ by default)
/mnt/ssd1/conda/envs/marllib/bin/python3 src/pipelines/train_rllib.py

# Once you have a checkpoint file, e.g.:
#   /mnt/ssd1/ray_results/<experiment>/checkpoint_000100/checkpoint-100
/mnt/ssd1/conda/envs/marllib/bin/python3 src/pipelines/generate_eval_replay.py \
  --mode checkpoint --checkpoint /path/to/checkpoint-100 \
  --out replays/after.json --num-agents 25

python3 src/viz/replay_export.py --replay replays/after.json \
  --out replays/scene_after.json --label "Trained policy"

# Side-by-side comparison, synced timeline + synced pan/zoom:
python3 src/viz/render_replay.py --scene replays/scene_before.json \
  --scene2 replays/scene_after.json -o replays/comparison.html
```

`--mode checkpoint` in `generate_eval_replay.py` currently supports checkpoints
from `train_rllib.py` (one shared policy for every agent). Checkpoints trained
via the MARLlib scripts (`train_marllib.py`, `train_marllib_a2c.py`,
`train_custom_a2c.py` — one policy per agent, e.g. `ippo`/`mappo`/`maa2c`) need
a small adaptation: swap `_load_rllib_policy()` in `generate_eval_replay.py`
for the algorithm-specific Trainer class (e.g. `from
marllib.marl.algos.core.IL.ppo import IPPOTrainer` for ippo), register
`BaseMLPCustom` (`src/models/base_mlp.py`) as `"Base_Model"` via
`ModelCatalog.register_custom_model`, and restore with `policy_id=f"policy_{i}"`
per agent — follow `train_marllib_a2c.py` for the exact config shape.

## 3. Using the viewer

| Control | Effect |
|---|---|
| Drag the timeline slider | Jump to any moment of the simulated day (06:00–24:00+) |
| ⏮ Prev / Next ⏭ | Jump to the previous/next actual bus event (precise step-by-step) |
| ▶ Play + speed dropdown | Animate; 900× is a good default (full day in ~96s) |
| Scroll / drag on the map | Zoom / pan (native Leaflet controls) |
| "Show map" checkbox | Toggle the OpenStreetMap basemap off for a clean, tile-free figure |
| Click a route name in the legend | Rename it (e.g. for a paper figure) |
| Click a stop marker | Rename that stop via the popup |
| "Export PNG" | Downloads the current view (map + legend, with your renamed labels) as a PNG |

Circle size on a stop = passengers currently waiting. Circle color on a bus =
occupancy (light blue → dark blue, empty → full). Line color + dash pattern
identify each route (see the legend).
