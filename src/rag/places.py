"""Street, neighbourhood and postcode for each bus stop, taken from the GTFS stop names.

In the SUNT GTFS feed, a stop's "name" is really its street address, for example:
  "Avenida Ulysses Guimarães 4355-4499 - Novo Horizonte Salvador - BA 41218-700 Brasil"
We split it into street ("Avenida Ulysses Guimarães"), neighbourhood ("Novo Horizonte")
and postcode ("41218-700"). These are used only to match and describe stops in text;
they are never used as transport data.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from typing import Dict, Optional

# street, optional house number, optional " - neighbourhood", then the city name
_STREET_RE = re.compile(
    r"^(?P<street>.+?)(?:\s+[\dS/N][\w\-/]*)?\s+(?:-\s+(?P<bairro>.+?)\s+)?Salvador"
)
_CEP_RE = re.compile(r"\b(\d{5}(?:-\d{3})?)\b")  # Brazilian postcode


def fold(text: str) -> str:
    """Lowercase and remove accents, so "São Marcos" and "sao marcos" compare equal."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def parse_stop_name(name: Optional[str]) -> Dict[str, Optional[str]]:
    """Splits one stop name into street, neighbourhood and postcode (None when not found)."""
    out = {"street": None, "neighborhood": None, "cep": None}
    if not name:
        return out
    m = _STREET_RE.match(name)
    if m:
        out["street"] = m.group("street").strip() or None
        b = m.group("bairro")
        out["neighborhood"] = b.strip() if b else None
    c = _CEP_RE.search(name)
    if c:
        out["cep"] = c.group(1)
    return out


def load_gtfs_places(stops_txt: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Reads GTFS stops.txt into {stop_id: {stop_name, street, neighborhood, cep, landmark}}.

    "landmark" comes from the optional stop_desc column (e.g. "Passeio Publico").
    """
    places = {}
    with open(stops_txt, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row["stop_id"]).strip()
            if sid.endswith(".0"):  # some ids are stored as floats, e.g. "12345.0"
                sid = sid[:-2]
            p = parse_stop_name(row.get("stop_name"))
            p["stop_name"] = row.get("stop_name") or None
            desc = (row.get("stop_desc") or "").strip()
            p["landmark"] = desc or None
            places.setdefault(sid, p)
    return places


def load_gtfs_route_names(routes_txt: str) -> Dict[str, str]:
    """Reads GTFS routes.txt into {line number: long name}, e.g. "1413" -> "Boca da Mata x Lapa"."""
    names = {}
    with open(routes_txt, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.setdefault(str(row["route_short_name"]).strip(), (row.get("route_long_name") or "").strip())
    return names
