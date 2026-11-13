"""The "key" that identifies a bus's situation, used to look up its context in the table.

A key is: (route, position on the route, direction, hour of the day, type of day).
Two buses in the same situation get the same key, and therefore the same context.

Rain and road works are not part of the key: no piece of context depends on them yet,
so adding them would only create duplicate rows with identical values.
"""
from __future__ import annotations

import datetime as _dt
from typing import NamedTuple, Optional, Sequence

DAY_TYPES = ("weekday", "saturday", "sunday")
N_HOURS = 24
MAX_ROUTE_IDX = 256  # upper limit for stops per route (the study routes have at most 60)


class QueryKey(NamedTuple):
    trip_id: str         # route id, e.g. "20002_1367_5"
    route_idx: int       # index of the current stop in the route's stop list
    going_forward: bool  # True on the outbound leg, False on the way back
    hour: int            # hour of the day, 0-23
    day_type: int        # 0 weekday, 1 Saturday, 2 Sunday (index into DAY_TYPES)


def day_type_of(date: str) -> int:
    """'YYYY-MM-DD' -> 0 weekday, 1 Saturday, 2 Sunday. Holidays are treated as normal days."""
    wd = _dt.date.fromisoformat(str(date)[:10]).weekday()
    return 0 if wd < 5 else (1 if wd == 5 else 2)


class KeyCodec:
    """Converts a QueryKey to a single integer and back (the table stores integers).

    The integer depends on the order of the route list, so that list is saved in the
    table's manifest and must be the same when reading.
    """

    def __init__(self, trips: Sequence[str]):
        self.trips = list(trips)
        self._trip_idx = {t: i for i, t in enumerate(self.trips)}

    def encode(self, key: QueryKey) -> int:
        t = self._trip_idx[key.trip_id]
        if not (0 <= key.route_idx < MAX_ROUTE_IDX and 0 <= key.hour < N_HOURS and 0 <= key.day_type < 3):
            raise ValueError(f"key out of range: {key}")
        code = t
        code = code * MAX_ROUTE_IDX + int(key.route_idx)
        code = code * 2 + int(bool(key.going_forward))
        code = code * N_HOURS + int(key.hour)
        code = code * 3 + int(key.day_type)
        return int(code)

    def decode(self, code: int) -> QueryKey:
        code, day_type = divmod(int(code), 3)
        code, hour = divmod(code, N_HOURS)
        code, gf = divmod(code, 2)
        t, idx = divmod(code, MAX_ROUTE_IDX)
        return QueryKey(self.trips[t], idx, bool(gf), hour, day_type)

    def has_trip(self, trip_id: str) -> bool:
        return trip_id in self._trip_idx


def key_from_env(env, agent: str) -> Optional[QueryKey]:
    """Reads the current situation of one bus straight from the simulator's state.

    Uses the same values the simulator uses to build the bus's own observation, so the
    context always matches what the bus sees. Returns None when the bus has finished its
    day or its route is unknown (it then gets an all-zero context).
    """
    st = env.agent_states.get(agent)
    if not st or st.get("status") == "finished":
        return None
    trip = env.agent_route_id.get(agent, "unknown")
    if trip == "unknown":
        return None
    hour = int(env.agent_times[agent] // 3600) % N_HOURS
    date = getattr(env, "date", None)
    if not date:
        return None
    return QueryKey(trip, int(st["route_idx"]), bool(st.get("going_forward", True)), hour, day_type_of(date))


def future_route_indices(idx: int, going_forward: bool, route_length: int, horizon: int):
    """The next `horizon` stops a bus will visit, as (stop index, direction after the move,
    direction used for the passenger lookup).

    Buses go to the end of the route and come back, so the path "bounces" at both ends,
    exactly like the simulator's MOVE action.

    The third value exists because of a simulator detail: when it looks up passenger
    numbers for a move, it uses the direction the bus had *before* turning around.
    We copy that so our numbers match the simulator's.
    """
    out = []
    i, d = int(idx), bool(going_forward)
    if route_length <= 1:
        return out
    for _ in range(max(0, int(horizon))):
        d_lookup = d
        if d:
            if i + 1 < route_length:
                i += 1
            else:          # reached the last stop: turn around
                d = False
                i -= 1
        else:
            if i > 0:
                i -= 1
            else:          # back at the first stop: turn around
                d = True
                i += 1
        out.append((i, d, d_lookup))
    return out
