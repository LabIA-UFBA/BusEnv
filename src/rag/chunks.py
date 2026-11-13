"""Builds the "chunks" (small pieces of information) and, for every bus situation, the list
of candidate chunks about the road ahead.

There are three kinds of chunk, all computed from data files BusEnv already has:
  stop_hour      passenger demand at one stop of one route, at one hour
                 (average boardings per visit, share of people getting off, time between buses)
  segment        travel time between two consecutive stops, for one type of day
  stop_activity  how busy a stop is at one hour (count of boarding records)

Every chunk records which file its numbers came from ("provenance").

For each bus situation (key), the candidates are the chunks about the next H stops on its
route, at the hour the bus is expected to get there. That list is called the key's "pool".

Uses only pandas/numpy so it also runs in the training environment, where a test checks
that our passenger lookup gives exactly the simulator's numbers.
"""
from __future__ import annotations

import glob
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rag.keys import DAY_TYPES, KeyCodec, QueryKey, day_type_of, future_route_indices
from rag.places import fold

TRAVEL_TIME_CAP = 1800.0  # the simulator caps each segment at 30 min (sunt_env.py:672)
DAY_TYPE_PT = ("dia útil", "sábado", "domingo")
CTYPES = ("stop_hour", "segment", "stop_activity")


# ----------------------------------------------------------------------------------------
# Passenger lookup, copied from the simulator (parallel_env._get_flow_stats, sunt_env.py:1630)
# ----------------------------------------------------------------------------------------
def flow_stats(stats: dict, route_id: str, node, now_time: float, going_forward: bool = True) -> dict:
    """Passenger numbers for one stop at one time, looked up exactly as the simulator does.

    It tries the most specific statistic first and falls back to broader ones when there
    are too few samples: route+stop+hour -> route+stop -> stop+hour -> stop -> city-wide.
    Route-specific statistics are only used on the outbound leg (real data only exists in
    that direction). Also returns which level answered ("_levels"), stored as provenance.
    """
    node = str(node)
    hour = int((now_time // 3600) % 24)
    min_count = int(stats.get("min_bucket_count", 1)) if stats else 1
    result = {"mean_boardings": None, "mean_alight_frac": None, "mean_intervisit_sec": None}
    levels = {}

    def _fill(entry, level):
        if not entry or entry.get("count", 0) < min_count:
            return
        for k in result:
            if result[k] is None and entry.get(k) is not None:
                result[k] = entry[k]
                levels[k] = (level, int(entry.get("count", 0)))

    if stats:
        if going_forward:
            _fill(stats.get("by_route_stop_hour", {}).get((route_id, node, hour)), "route_stop_hour")
            _fill(stats.get("by_route_stop", {}).get((route_id, node)), "route_stop")
        _fill(stats.get("by_stop_hour", {}).get((node, hour)), "stop_hour")
        _fill(stats.get("by_stop", {}).get(node), "stop")
        g = stats.get("global", {})
    else:
        g = {}
    defaults = {"mean_boardings": 1.0, "mean_alight_frac": 0.2, "mean_intervisit_sec": 600.0}
    for k, dv in defaults.items():
        if result[k] is None:
            result[k] = g.get(k, dv)
            levels[k] = ("global", 0)
    result["_levels"] = levels
    return result


def arrival_rate(flow: dict) -> float:
    """Passengers arriving per second = boardings per visit / seconds between visits (as the simulator)."""
    return max(0.0, float(flow["mean_boardings"])) / max(float(flow["mean_intervisit_sec"]), 1.0)


# ----------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------
class SuntData:
    """Loads the BusEnv data files the chunks are built from, plus GTFS stop/route names."""

    def __init__(self, obs_dir: str, daily_dir: str, train_start: str, train_end: str,
                 places: Optional[dict] = None, route_names: Optional[dict] = None):
        def _pk(name):
            with open(os.path.join(obs_dir, name), "rb") as f:
                return pickle.load(f)

        self.obs_dir, self.daily_dir = obs_dir, daily_dir
        self.flow = _pk("stop_passenger_flow.pkl")
        self.tt_stats = _pk("travel_time_stats.pkl")
        self.avg_tt = _pk("avg_travel_time_AB.pkl")
        self.activity = _pk("hourly_stop_activity.pkl")
        self.routes = _pk("real_routes.pkl")
        self.route_meta = _pk("route_metadata.pkl")
        # travel time the simulator uses when a segment has no data (sunt_env.py:153-156)
        self.default_tt = float(np.mean(list(self.avg_tt.values())))
        self.places = places or {}
        self.route_names = route_names or {}
        self.train_start, self.train_end = train_start, train_end
        self.train_days = self._train_day_files()
        self.seg_daytype = self._segment_daytype_stats()

    def _train_day_files(self) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.daily_dir, "daily_data_*.pkl")))
        out = []
        for f in files:
            d = os.path.basename(f)[len("daily_data_"):-len(".pkl")]
            if self.train_start <= d <= self.train_end:
                out.append(f)
        return out

    def _segment_daytype_stats(self) -> Dict[Tuple[str, str, int], Tuple[float, float, int]]:
        """Average travel time of each segment per type of day, using TRAINING days only
        (so the test days never influence the context). -> {(a, b, day_type): (mean, std, n_days)}"""
        acc = defaultdict(list)
        for f in self.train_days:
            with open(f, "rb") as fh:
                day = pickle.load(fh)
            dt = day_type_of(day.get("date") or os.path.basename(f)[11:21])
            for (a, b), v in day.get("avg_travel_times", {}).items():
                acc[(str(a), str(b), dt)].append(float(v))
        return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}

    def segment_value(self, a: str, b: str, dt: int) -> Tuple[float, float, int, str]:
        """Travel time of segment a->b on day type dt: training-day average if available,
        else the whole-period average, else the simulator's default. Returns (mean, std, count, source)."""
        v = self.seg_daytype.get((str(a), str(b), dt))
        if v is not None:
            return v[0], v[1], v[2], "train_daytype"
        s = self.tt_stats.get((str(a), str(b)))
        if s is not None:
            return float(s["mean"]), float(s["std"]), int(s["count"]), "period"
        return self.default_tt, 0.0, 0, "default"

    def place(self, stop: str) -> dict:
        return self.places.get(str(stop), {})

    def route_label(self, trip: str) -> Tuple[str, str, str]:
        m = self.route_meta.get(trip, {})
        short = str(m.get("route_short_name", trip))
        return short, self.route_names.get(short, ""), str(m.get("direction_id", ""))


# ----------------------------------------------------------------------------------------
# Chunk registry
# ----------------------------------------------------------------------------------------
class ChunkRegistry:
    """Creates each chunk only once (many keys share the same chunk) and gives it an integer id."""

    def __init__(self, data: SuntData):
        self.d = data
        self.rows: List[dict] = []
        self._ids: Dict[tuple, int] = {}

    def _add(self, key: tuple, row: dict) -> int:
        cid = self._ids.get(key)
        if cid is None:
            cid = len(self.rows)
            row["chunk_id"] = cid
            self.rows.append(row)
            self._ids[key] = cid
        return cid

    def stop_hour(self, trip: str, stop: str, hour: int, dir_lookup: bool) -> int:
        key = ("stop_hour", trip, str(stop), int(hour), bool(dir_lookup))
        if key in self._ids:
            return self._ids[key]
        fl = flow_stats(self.d.flow, trip, stop, hour * 3600 + 1800, dir_lookup)
        lvl, cnt = fl["_levels"]["mean_boardings"]
        return self._add(key, {
            "ctype": "stop_hour", "trip_id": trip, "stop_id": str(stop), "stop_b": None, "hour": int(hour),
            "dir_fwd": bool(dir_lookup), "day_type": -1,
            "value_main": arrival_rate(fl), "value_aux": float(fl["mean_alight_frac"]),
            "value_std": float("nan"), "boardings": float(fl["mean_boardings"]),
            "intervisit": float(fl["mean_intervisit_sec"]), "count": cnt, "level": lvl,
            "provenance": f"stop_passenger_flow.pkl:by_{lvl}",
        })

    def segment(self, a: str, b: str, dt: int) -> int:
        key = ("segment", str(a), str(b), int(dt))
        if key in self._ids:
            return self._ids[key]
        mean, std, cnt, lvl = self.d.segment_value(a, b, dt)
        prov = {"train_daytype": "daily/daily_data_*.pkl:avg_travel_times (training days, by day type)",
                "period": "travel_time_stats.pkl", "default": "mean(avg_travel_time_AB.pkl) (simulator fallback)"}[lvl]
        return self._add(key, {
            "ctype": "segment", "trip_id": None, "stop_id": str(a), "stop_b": str(b), "hour": -1,
            "dir_fwd": None, "day_type": int(dt), "value_main": min(mean, TRAVEL_TIME_CAP), "value_aux": float("nan"),
            "value_std": std, "boardings": float("nan"), "intervisit": float("nan"), "count": cnt, "level": lvl,
            "provenance": prov,
        })

    def stop_activity(self, stop: str, hour: int) -> int:
        key = ("stop_activity", str(stop), int(hour))
        if key in self._ids:
            return self._ids[key]
        hist = self.d.activity.get(str(stop)) or {}
        tot = float(sum(hist.values())) if hist else 0.0
        cnt = int(hist.get(int(hour), 0)) if hist else 0
        return self._add(key, {
            "ctype": "stop_activity", "trip_id": None, "stop_id": str(stop), "stop_b": None, "hour": int(hour),
            "dir_fwd": None, "day_type": -1, "value_main": (cnt / tot) if tot > 0 else 0.0, "value_aux": float("nan"),
            "value_std": float("nan"), "boardings": float("nan"), "intervisit": float("nan"), "count": cnt,
            "level": "stop_hour" if hist else "missing", "provenance": "hourly_stop_activity.pkl",
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("chunk_id", drop=False).sort_index()


# ----------------------------------------------------------------------------------------
# Query enumeration and candidate pools
# ----------------------------------------------------------------------------------------
def enumerate_keys(data: SuntData, trips: Sequence[str], hours: Sequence[int]) -> List[QueryKey]:
    """Every possible bus situation: each stop of each route, both directions, each hour, each day type."""
    keys = []
    for trip in trips:
        L = len(data.routes[trip])
        for idx in range(L):
            for gf in (True, False):
                for h in hours:
                    for dt in range(len(DAY_TYPES)):
                        keys.append(QueryKey(trip, idx, gf, int(h), dt))
    return keys


def build_pools(data: SuntData, keys: Sequence[QueryKey], codec: KeyCodec, horizon: int
                ) -> Tuple[pd.DataFrame, ChunkRegistry]:
    """For every key, lists the candidate chunks about the next `horizon` stops.

    For each upcoming stop we add: its demand at the expected arrival hour and the hour
    after, the travel time of the segment leading to it, and its activity at that hour.
    The expected arrival time is accumulated from the segment travel times.
    Returns one row per (key, candidate), ordered from the nearest stop to the farthest.
    "hop" = how many stops ahead (1 = next stop).
    """
    reg = ChunkRegistry(data)
    cols = defaultdict(list)
    for key in keys:
        route = data.routes[key.trip_id]
        t0 = key.hour * 3600 + 1800.0
        cum = 0.0
        prev = route[key.route_idx]
        seen = set()
        pos = 0
        for hop, (j, _d_after, d_lookup) in enumerate(
                future_route_indices(key.route_idx, key.going_forward, len(route), horizon), start=1):
            stop = route[j]
            seg_id = reg.segment(prev, stop, key.day_type)
            cum += min(reg.rows[seg_id]["value_main"], TRAVEL_TIME_CAP)
            eta = t0 + cum
            eta_h = int(eta // 3600) % 24
            cands = [reg.stop_hour(key.trip_id, stop, eta_h, d_lookup)]  # demand at arrival hour
            if eta_h + 1 <= 23:
                cands.append(reg.stop_hour(key.trip_id, stop, eta_h + 1, d_lookup))
            cands += [seg_id, reg.stop_activity(stop, eta_h)]
            for cid in cands:
                if cid in seen:  # the same chunk can appear twice after the bus turns around
                    continue
                seen.add(cid)
                cols["qkey"].append(codec.encode(key))
                cols["cand_pos"].append(pos)
                cols["chunk_id"].append(cid)
                cols["hop"].append(hop)
                cols["eta_h"].append(eta / 3600.0)
                pos += 1
            prev = stop
    pools = pd.DataFrame(cols)
    return pools, reg


# ----------------------------------------------------------------------------------------
# Structural scorers and salience (numpy only)
# ----------------------------------------------------------------------------------------
def add_structural_scores(pools: pd.DataFrame, chunks: pd.DataFrame, data: SuntData, codec: KeyCodec,
                          horizon: int) -> pd.DataFrame:
    """Adds the scores that need no text model. Every score is between 0 and 1, higher = more relevant:

      proximity  1 / stops ahead (the next stop matters most)
      eta        how well the chunk's hour matches when the bus will arrive there
      salience   how unusual the chunk's value is for this route ("z" = standard deviations
                 away from the route's normal value at that hour / day type)
      support    how many data samples back the chunk (more samples = more reliable)
      place      is the chunk's stop on the same street (1.0) or neighbourhood (0.6) as the
                 bus's current stop, discounted with distance (uses the GTFS stop names)
    """
    c = chunks.loc[pools["chunk_id"].to_numpy()]
    ctype = c["ctype"].to_numpy()
    p = pools.copy()
    p["ctype"] = ctype
    keys = [codec.decode(k) for k in p["qkey"].to_numpy()]
    p["trip_id"] = [k.trip_id for k in keys]
    p["q_hour"] = [k.hour for k in keys]

    # z = how far the chunk's value is from what is normal on this route, compared with:
    #   demand chunks   -> the route's stops at the same hour
    #   segment chunks  -> the route's segments on the same type of day
    #   activity chunks -> the route's stops at the same hour
    z = np.zeros(len(p))
    val = c["value_main"].to_numpy(dtype=float)
    ref_mean, ref_std = _reference_stats(chunks, data)
    for i, (t, ct, ch, cdt, v) in enumerate(zip(p["trip_id"], ctype, c["hour"], c["day_type"], val)):
        g = (ct, t, int(ch) if ct != "segment" else int(cdt))
        mu, sd = ref_mean.get(g, v), ref_std.get(g, 0.0)
        z[i] = (v - mu) / sd if sd > 1e-9 else 0.0
    p["z"] = np.clip(z, -5, 5)

    hop = p["hop"].to_numpy(dtype=float)
    p["s_proximity"] = 1.0 / hop
    chunk_hour = c["hour"].to_numpy(dtype=float)
    eta = p["eta_h"].to_numpy(dtype=float)
    s_eta = np.where(chunk_hour >= 0, np.exp(-np.abs(chunk_hour + 0.5 - eta)), 0.5)
    p["s_eta"] = s_eta
    p["s_salience"] = np.minimum(np.abs(p["z"].to_numpy()) / 3.0, 1.0)
    cnt = c["count"].to_numpy(dtype=float)
    maxc = chunks.groupby("ctype")["count"].max().to_dict()
    p["s_support"] = np.log1p(cnt) / np.log1p(np.array([max(maxc.get(t, 1), 1) for t in ctype], dtype=float))

    # place score: is the chunk's stop on the bus's current street / neighbourhood?
    s_place = np.zeros(len(p))
    stop_ref = np.where(ctype == "segment", c["stop_b"].to_numpy(), c["stop_id"].to_numpy())
    for i, (k, s) in enumerate(zip(keys, stop_ref)):
        cur = data.place(data.routes[k.trip_id][k.route_idx])
        pl = data.place(s)
        m = 0.0
        if cur.get("street") and pl.get("street") and fold(cur["street"]) == fold(pl["street"]):
            m = 1.0
        elif cur.get("neighborhood") and pl.get("neighborhood") and fold(cur["neighborhood"]) == fold(pl["neighborhood"]):
            m = 0.6
        s_place[i] = m / math.sqrt(hop[i])
    p["s_place"] = s_place
    return p.drop(columns=["trip_id", "q_hour"])


def _reference_stats(chunks: pd.DataFrame, data: SuntData):
    """The "normal" value (mean and spread) of each chunk type on each route, used to compute z."""
    means, stds = {}, {}
    sh = chunks[chunks["ctype"] == "stop_hour"]
    for (t, h), g in sh.groupby(["trip_id", "hour"]):
        means[("stop_hour", t, int(h))] = g["value_main"].mean()
        stds[("stop_hour", t, int(h))] = g["value_main"].std(ddof=0)
    # segment and activity chunks are shared between routes, so group them by each route's own stops
    seg = chunks[chunks["ctype"] == "segment"].set_index(["stop_id", "stop_b", "day_type"])["value_main"]
    act = chunks[chunks["ctype"] == "stop_activity"].set_index(["stop_id", "hour"])["value_main"]
    trips = sh["trip_id"].dropna().unique()
    for t in trips:
        route = [str(s) for s in data.routes[t]]
        stops = set(route)
        edges = set(zip(route[:-1], route[1:])) | set(zip(route[1:], route[:-1]))
        sv = seg[[(a, b) in edges for a, b, _ in seg.index]]
        for dt, g in sv.groupby(level=2):
            means[("segment", t, int(dt))] = g.mean()
            stds[("segment", t, int(dt))] = g.std(ddof=0)
        av = act[[s in stops for s, _ in act.index]]
        for h, g in av.groupby(level=1):
            means[("stop_activity", t, int(h))] = g.mean()
            stds[("stop_activity", t, int(h))] = g.std(ddof=0)
    return means, stds


def add_ranks(p: pd.DataFrame, scorers: Sequence[str]) -> pd.DataFrame:
    """For each score s_<a>, adds rank_<a>: 0 = best candidate of that key (ties: nearer stop first)."""
    p = p.sort_values(["qkey", "cand_pos"]).reset_index(drop=True)
    for a in scorers:
        p[f"rank_{a}"] = (p.groupby("qkey")[f"s_{a}"].rank(ascending=False, method="first").astype(int) - 1)
    return p


# ----------------------------------------------------------------------------------------
# Text versions of chunks and keys (Portuguese), used by the BM25 and embedding search.
# The "without place" variant drops street / neighbourhood names (to measure their effect).
# ----------------------------------------------------------------------------------------
def _place_str(pl: dict, with_place: bool) -> str:
    if not with_place or not pl:
        return ""
    parts = [pl.get("street"), ("bairro " + pl["neighborhood"]) if pl.get("neighborhood") else None, pl.get("landmark")]
    return " | ".join(x for x in parts if x)


def render_chunk(row: dict, data: SuntData, with_place: bool = True) -> str:
    """One chunk as a sentence, e.g. "[demanda] linha 1719 ... parada X | Via Regional | hora 07 | ..."."""
    ct = row["ctype"]
    if ct == "stop_hour":
        short, long_, dirid = data.route_label(row["trip_id"])
        sentido = dirid if row["dir_fwd"] else "retorno"
        pl = _place_str(data.place(row["stop_id"]), with_place)
        return (f"[demanda] linha {short} {('(' + long_ + ')') if (with_place and long_) else ''} sentido {sentido} "
                f"parada {row['stop_id']} {('| ' + pl) if pl else ''} | hora {int(row['hour']):02d} | "
                f"embarques médios {row['boardings']:.2f} por visita; fração de desembarque {row['value_aux']:.2f}; "
                f"intervalo entre visitas {row['intervisit']:.0f} s; n={int(row['count'])}; nível {row['level']}")
    if ct == "segment":
        pa = _place_str(data.place(row["stop_id"]), with_place)
        pb = _place_str(data.place(row["stop_b"]), with_place)
        std = row["value_std"]
        return (f"[trecho] de parada {row['stop_id']} {('(' + pa + ')') if pa else ''} para parada {row['stop_b']} "
                f"{('(' + pb + ')') if pb else ''} | {DAY_TYPE_PT[int(row['day_type'])]} | tempo médio "
                f"{row['value_main']:.0f} s; desvio {0.0 if std != std else std:.0f} s; n={int(row['count'])}; nível {row['level']}")
    pl = _place_str(data.place(row["stop_id"]), with_place)
    return (f"[atividade] parada {row['stop_id']} {('| ' + pl) if pl else ''} | hora {int(row['hour']):02d} | "
            f"{int(row['count'])} registros de embarque; {100 * row['value_main']:.1f}% do dia")


def render_query(key: QueryKey, data: SuntData, with_place: bool = True, n_next: int = 3) -> str:
    """The search query for a bus situation: route, direction, the next 3 stops, hour and day type."""
    route = data.routes[key.trip_id]
    short, long_, dirid = data.route_label(key.trip_id)
    nxt = [route[j] for j, _, _ in future_route_indices(key.route_idx, key.going_forward, len(route), n_next)]
    if with_place:
        desc = []
        for s in nxt:
            pl = data.place(s)
            desc.append(f"{pl.get('street') or s}{(' (' + pl['neighborhood'] + ')') if pl.get('neighborhood') else ''}")
        stops = ", ".join(desc)
        head = f"linha {short} ({long_})" if long_ else f"linha {short}"
    else:
        stops = ", ".join(f"parada {s}" for s in nxt)
        head = f"linha {short}"
    sentido = dirid if key.going_forward else "retorno"
    return f"{head} sentido {sentido} próximas paradas: {stops} | hora {key.hour:02d} | {DAY_TYPE_PT[key.day_type]}"


# ----------------------------------------------------------------------------------------
# Small numeric vector per candidate. MA-Chunk compares these vectors to avoid picking
# two chunks that say almost the same thing.
# ----------------------------------------------------------------------------------------
def numeric_emb(p: pd.DataFrame, chunks: pd.DataFrame, horizon: int) -> np.ndarray:
    c = chunks.loc[p["chunk_id"].to_numpy()]
    onehot = np.stack([(c["ctype"].to_numpy() == t).astype(float) for t in CTYPES], axis=1)
    hop = p["hop"].to_numpy(dtype=float)[:, None] / horizon
    eta = (p["eta_h"].to_numpy(dtype=float)[:, None] % 24) / 24.0
    zc = np.clip(p["z"].to_numpy(dtype=float), -3, 3)[:, None] / 3.0
    e = np.concatenate([onehot, hop, eta, zc], axis=1)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    return e.astype(np.float32)
