#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_export.py

Converts a raw replay log (JSON produced by parallel_env._flush_replay_log,
when record_replay=True) into a "scene" JSON for the Leaflet-based HTML
viewer (render_replay.py): keeps real lat/lon for every stop (Leaflet does its
own map projection, so no custom projection is needed here), assigns each
route a validated categorical color + a distinct line-dash pattern (so routes
stay distinguishable even without color), and groups events by agent/stop for
animation in the browser.

Usage:
    python src/viz/replay_export.py --replay replays/replay_random.json \
        --out replays/scene_before.json --label "Untrained baseline"
"""

import argparse
import json
import os
import pickle
import zipfile

# Categorical palette (validated: docs/dataviz skill reference palette.md),
# fixed order, never cycled/reordered. Distinct dash patterns are assigned
# alongside color so routes are never distinguishable by hue alone.
ROUTE_STYLES = [
    {"color": "#2a78d6", "dash": None},       # slot 1 blue   - solid
    {"color": "#eb6834", "dash": "8 4"},      # slot 2 orange - dashed
    {"color": "#1baf7a", "dash": "2 4"},      # slot 3 aqua   - dotted
    {"color": "#eda100", "dash": "10 4 2 4"}, # slot 4 yellow - dash-dot
    {"color": "#e87ba4", "dash": "1 5"},      # slot 5 magenta- fine dotted
]
FALLBACK_STYLE = {"color": "#898781", "dash": "4 4"}  # "Other" — muted ink, never a new hue


def load_graph_coords(graph_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    coords = {}
    for node, data in G.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            coords[str(node)] = (float(lat), float(lon))
    return coords


def load_pickle_or_zip(training_observation_dir, name):
    pkl_path = os.path.join(training_observation_dir, f"{name}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    zip_path = os.path.join(training_observation_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            with z.open(f"{name}.pkl") as f:
                return pickle.load(f)
    return {}


def default_route_name(route_id, route_metadata):
    meta = route_metadata.get(route_id)
    if not meta:
        return f"Route {route_id}"
    short_name = meta.get("route_short_name", route_id)
    direction = meta.get("direction_id")
    direction_label = {"I": "outbound", "V": "inbound"}.get(direction, direction)
    if direction_label:
        return f"Route {short_name} ({direction_label})"
    return f"Route {short_name}"


def build_scene(replay_events, coords, real_routes, route_metadata, label):
    routes_used = sorted({e["route_id"] for e in replay_events if e.get("route_id") and e["route_id"] != "unknown"})

    stops_used = set()
    for e in replay_events:
        stops_used.add(str(e["curr_node"]))
        stops_used.add(str(e["next_node"]))

    routes = {}
    for i, rid in enumerate(routes_used):
        path = real_routes.get(rid)
        if not path:
            continue
        style = ROUTE_STYLES[i] if i < len(ROUTE_STYLES) else FALLBACK_STYLE
        routes[rid] = {
            "name": default_route_name(rid, route_metadata),
            "color": style["color"],
            "dash": style["dash"],
            "stops": [str(n) for n in path],
        }
        stops_used.update(routes[rid]["stops"])

    stops = {n: {"lat": coords[n][0], "lon": coords[n][1]} for n in stops_used if n in coords}

    per_agent = {}
    for e in replay_events:
        per_agent.setdefault(e["agent_id"], []).append(e)
    for evs in per_agent.values():
        evs.sort(key=lambda r: r["sim_time_sec"])

    # Per-stop waiting-queue timeline. "curr_node" in each logged event is the stop the
    # bus just arrived at THIS step (sunt_env.py's step() already advances
    # self.states[agent] to the arrival node before the event is recorded), which is
    # exactly the key self.stop_waiting_passengers is read under.
    stop_waiting = {}
    for e in replay_events:
        node = str(e["curr_node"])
        stop_waiting.setdefault(node, []).append([e["sim_time_sec"], e.get("waiting", 0.0)])
    for samples in stop_waiting.values():
        samples.sort(key=lambda s: s[0])

    return {
        "label": label,
        "stops": stops,
        "routes": routes,
        "agents": {
            agent: [
                {
                    "t": e["sim_time_sec"],
                    "curr": str(e["curr_node"]),
                    "next": str(e["next_node"]),
                    "action": e.get("action", 0),
                    "occ": e.get("occupancy", 0.0),
                    "route": e.get("route_id"),
                }
                for e in evs
            ]
            for agent, evs in per_agent.items()
        },
        "stop_waiting": stop_waiting,
    }


def main():
    parser = argparse.ArgumentParser(description="Export a replay log into a scene JSON for render_replay.py")
    parser.add_argument("--replay", required=True, help="Path to a replay_*.json log produced by the env")
    parser.add_argument("--out", required=True, help="Output scene JSON path")
    parser.add_argument("--label", default="Replay", help="Scene label shown in the viewer (e.g. 'Untrained baseline')")
    parser.add_argument(
        "--graph",
        default=os.path.join(os.path.dirname(__file__), "graph_gtfs_fev_2024.gpickle"),
        help="Path to the GTFS graph pickle (for stop lat/lon)",
    )
    parser.add_argument(
        "--training-observation-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "training_observation"),
        help="Directory containing real_routes.pkl/.zip and route_metadata.pkl",
    )
    args = parser.parse_args()

    with open(args.replay, "r") as f:
        replay_events = json.load(f)

    coords = load_graph_coords(args.graph)
    obs_dir = os.path.normpath(args.training_observation_dir)
    real_routes = load_pickle_or_zip(obs_dir, "real_routes")
    route_metadata = load_pickle_or_zip(obs_dir, "route_metadata")

    scene = build_scene(replay_events, coords, real_routes, route_metadata, args.label)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scene, f)

    print(f"Scene saved to {args.out} ({len(scene['agents'])} agents, {len(scene['stops'])} stops, {len(scene['routes'])} routes)")


if __name__ == "__main__":
    main()
