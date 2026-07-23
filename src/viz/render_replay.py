#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_replay.py

Builds a single self-contained HTML file that replays one or two scene JSONs
(produced by replay_export.py) as a game-like map view of Salvador's bus
routes on a real OpenStreetMap basemap (via Leaflet — native mouse-wheel zoom
and drag-to-pan), with a draggable timeline, per-route legend with editable
names, editable stop labels, and a PNG export button for paper figures.
Passing two scenes renders them side by side (e.g. "untrained" vs "trained")
with synced pan/zoom and a single shared timeline.

Requires internet access when the HTML is opened (loads Leaflet + OpenStreetMap
tiles + html2canvas from public CDNs) — there is no offline/tile-free mode by
design, since the whole point of this view is to show real street context.
Use the "Show map" checkbox to hide the tiles for a cleaner, tile-free figure
if a paper figure should not include OpenStreetMap imagery/attribution.

Usage:
    python src/viz/render_replay.py --scene replays/scene_before.json -o before.html
    python src/viz/render_replay.py --scene replays/scene_before.json \
        --scene2 replays/scene_after.json -o comparison.html
"""

import argparse
import json
import os


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>BusEnv Replay Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --hairline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    --accent:         #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --hairline:       #2c2c2a;
      --border:         rgba(255,255,255,0.10);
      --accent:         #3987e5;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --hairline:       #2c2c2a;
    --border:         rgba(255,255,255,0.10);
    --accent:         #3987e5;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body.viz-root {
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header { padding: 14px 18px 6px; }
  header h1 { font-size: 18px; margin: 0 0 2px; }
  header p { margin: 0; color: var(--text-secondary); font-size: 13px; max-width: 900px; }
  .controls {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 10px 14px; background: var(--surface-1); margin: 10px 18px; border-radius: 10px;
    border: 1px solid var(--border);
  }
  .controls button, .controls label {
    background: var(--accent); color: white; border: none; border-radius: 6px;
    padding: 7px 12px; font-size: 13px; cursor: pointer; white-space: nowrap;
  }
  .controls label.toggle, .controls label.checkbox {
    background: transparent; color: var(--text-primary); border: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px;
  }
  .controls button:hover { opacity: 0.9; }
  .controls input[type=range] { flex: 1; min-width: 180px; }
  .controls select { padding: 5px; border-radius: 6px; }
  .time-label { font-variant-numeric: tabular-nums; min-width: 78px; text-align: center; font-size: 14px; }
  .panels { display: flex; gap: 14px; padding: 0 18px 18px; flex-wrap: wrap; align-items: flex-start; }
  .panel {
    background: var(--surface-1); border-radius: 10px; padding: 10px; flex: 1 1 480px;
    min-width: 340px; border: 1px solid var(--border);
  }
  .panel h2 { font-size: 14px; margin: 2px 6px 8px; font-weight: 600; }
  .panel h2 input {
    font: inherit; font-weight: 600; border: none; background: transparent; color: inherit;
    border-bottom: 1px dashed var(--border); width: 100%;
  }
  .map-container { width: 100%; height: 520px; border-radius: 8px; overflow: hidden; position: relative; }
  /* Legend floats in a map corner (Leaflet control) so it's captured together
     with the map on export, instead of a separate block below it. */
  .legend-control {
    background: var(--surface-1); color: var(--text-secondary);
    padding: 8px 10px; font-size: 11px; border-radius: 8px; max-width: 210px;
    border: 1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .legend-control .hint { color: var(--text-muted); margin-bottom: 6px; }
  .legend-control .route-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .legend-control .swatch { width: 20px; height: 0; border-top-width: 3px; border-top-style: solid; flex: none; }
  .legend-control input.route-name {
    flex: 1; font: inherit; font-size: 11px; color: var(--text-primary); background: transparent;
    border: none; border-bottom: 1px solid transparent; padding: 1px 0; min-width: 0;
  }
  .legend-control input.route-name:hover, .legend-control input.route-name:focus {
    border-bottom: 1px solid var(--border); outline: none;
  }
  .occ-scale { display: flex; align-items: center; gap: 5px; margin-top: 6px; font-size: 10px; }
  .occ-scale .ramp {
    width: 80px; height: 8px; border-radius: 4px; flex: none;
    background: linear-gradient(to right, #cde2fb, #256abf, #0d366b);
  }
  .leaflet-popup-content input { font: inherit; width: 140px; }
</style>
</head>
<body class="viz-root">
<header>
  <h1>BusEnv — Route Replay Viewer</h1>
  <p>Drag the timeline to any moment of the simulated day to inspect bus positions and stop queues.
     Scroll to zoom, drag to pan. Route and stop names are editable — click a name to rename it before exporting a figure.</p>
</header>

<div class="controls">
  <button id="stepBackBtn" title="Jump to previous event">⏮ Prev</button>
  <button id="playBtn">▶ Play</button>
  <button id="stepFwdBtn" title="Jump to next event">Next ⏭</button>
  <select id="speedSel">
    <option value="60">60×</option>
    <option value="300">300×</option>
    <option value="900" selected>900×</option>
    <option value="1800">1800×</option>
    <option value="3600">3600×</option>
    <option value="7200">7200×</option>
  </select>
  <input type="range" id="timeSlider" min="0" max="86400" step="1" value="21600"/>
  <div class="time-label" id="timeLabel">06:00:00</div>
  <label class="checkbox"><input type="checkbox" id="basemapToggle" checked/> Show map</label>
  <button id="exportBtn">Export PNG</button>
</div>

<div class="panels" id="panels"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
const SCENES = __SCENES_JSON__;

// Sequential blue ramp (magnitude encoding for bus occupancy), light -> dark.
const OCC_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"];
function occColor(occ) {
  occ = Math.max(0, Math.min(1, occ || 0));
  const idx = occ * (OCC_RAMP.length - 1);
  const lo = Math.floor(idx), hi = Math.min(OCC_RAMP.length - 1, lo + 1);
  const frac = idx - lo;
  const c1 = hexToRgb(OCC_RAMP[lo]), c2 = hexToRgb(OCC_RAMP[hi]);
  const r = Math.round(c1[0] + (c2[0] - c1[0]) * frac);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * frac);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * frac);
  return `rgb(${r},${g},${b})`;
}
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ---------- Per-scene editable name state ----------
const routeNames = SCENES.map(scene => {
  const names = {};
  for (const rid in scene.routes) names[rid] = scene.routes[rid].name;
  return names;
});
const stopNames = SCENES.map(() => ({}));
function stopLabel(sceneIdx, stopId) {
  return stopNames[sceneIdx][stopId] || `Stop ${stopId}`;
}

// ---------- Build panels ----------
const panelsDiv = document.getElementById("panels");
const maps = [];
const tileLayers = [];
const busMarkers = SCENES.map(() => ({}));
const stopMarkers = SCENES.map(() => ({}));
const legendRows = SCENES.map(() => ({})); // [sceneIdx][routeId] -> {swatch, input}

// Dash pattern presets — identity is never color-alone (each route also gets a
// distinct line pattern), and both are editable from the map or the legend.
const DASH_PRESETS = [
  { name: "Solid", value: "" },
  { name: "Dashed", value: "8 4" },
  { name: "Dotted", value: "2 4" },
  { name: "Dash-dot", value: "10 4 2 4" },
  { name: "Fine dotted", value: "1 5" },
];

// ---------- Shared floating route-style editor (name + color + dash pattern) ----------
// Triggered from either the legend swatch or clicking the route line on the map itself.
const editorEl = document.createElement("div");
editorEl.id = "routeEditor";
editorEl.style.cssText = "position:fixed; z-index:10000; display:none; background:var(--surface-1); " +
  "color:var(--text-primary); border:1px solid var(--border); border-radius:8px; padding:10px; " +
  "box-shadow:0 4px 16px rgba(0,0,0,0.25); font-size:12px; min-width:200px;";
editorEl.innerHTML = `
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <strong>Route style</strong>
    <button id="routeEditorClose" style="background:none;border:none;color:inherit;cursor:pointer;font-size:14px;">×</button>
  </div>
  <label style="display:block;margin-bottom:6px;">Name<br/>
    <input id="routeEditorName" style="width:100%;font:inherit;"/>
  </label>
  <label style="display:block;margin-bottom:6px;">Color
    <input id="routeEditorColor" type="color" style="vertical-align:middle;margin-left:6px;"/>
  </label>
  <label style="display:block;">Dash pattern<br/>
    <select id="routeEditorDash" style="width:100%;font:inherit;">
      ${DASH_PRESETS.map(d => `<option value="${d.value}">${d.name}</option>`).join("")}
    </select>
  </label>
`;
document.body.appendChild(editorEl);

let editingRoute = null; // {sceneIdx, routeId}

function applyRouteStyle(i, rid) {
  const route = SCENES[i].routes[rid];
  if (route._layer) {
    route._layer.setStyle({ color: route.color, dashArray: route.dash || null });
    route._layer.setTooltipContent(routeNames[i][rid]);
  }
  const row = legendRows[i][rid];
  if (row) {
    row.swatch.style.borderTopColor = route.color;
    row.swatch.style.borderTopStyle = route.dash ? "dashed" : "solid";
    row.input.value = routeNames[i][rid];
  }
}

function openRouteEditor(i, rid, clientX, clientY) {
  editingRoute = { i, rid };
  const route = SCENES[i].routes[rid];
  document.getElementById("routeEditorName").value = routeNames[i][rid];
  document.getElementById("routeEditorColor").value = route.color;
  document.getElementById("routeEditorDash").value = route.dash || "";
  editorEl.style.display = "block";
  const maxLeft = window.innerWidth - 220, maxTop = window.innerHeight - 180;
  editorEl.style.left = Math.min(clientX, maxLeft) + "px";
  editorEl.style.top = Math.min(clientY, maxTop) + "px";
}

document.getElementById("routeEditorClose").addEventListener("click", () => { editorEl.style.display = "none"; });
document.getElementById("routeEditorName").addEventListener("input", (ev) => {
  if (!editingRoute) return;
  const { i, rid } = editingRoute;
  routeNames[i][rid] = ev.target.value;
  applyRouteStyle(i, rid);
});
document.getElementById("routeEditorColor").addEventListener("input", (ev) => {
  if (!editingRoute) return;
  const { i, rid } = editingRoute;
  SCENES[i].routes[rid].color = ev.target.value;
  applyRouteStyle(i, rid);
});
document.getElementById("routeEditorDash").addEventListener("change", (ev) => {
  if (!editingRoute) return;
  const { i, rid } = editingRoute;
  SCENES[i].routes[rid].dash = ev.target.value || null;
  applyRouteStyle(i, rid);
});

SCENES.forEach((scene, i) => {
  const panel = document.createElement("div");
  panel.className = "panel";

  const h2 = document.createElement("h2");
  const titleInput = document.createElement("input");
  titleInput.value = scene.label || `Scenario ${i + 1}`;
  h2.appendChild(titleInput);
  panel.appendChild(h2);

  const mapDiv = document.createElement("div");
  mapDiv.className = "map-container";
  mapDiv.id = `map-${i}`;
  panel.appendChild(mapDiv);
  panelsDiv.appendChild(panel);

  // preferCanvas: true — renders routes/stops/buses on a single <canvas> overlay
  // instead of SVG, which html2canvas captures reliably (SVG panes with Leaflet's
  // CSS transforms are a common cause of markers going missing from the export).
  const map = L.map(mapDiv.id, { zoomControl: true, attributionControl: true, preferCanvas: true });
  maps.push(map);

  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    crossOrigin: true,
    attribution: "&copy; OpenStreetMap contributors",
  });
  tiles.addTo(map);
  tileLayers.push(tiles);

  // ---------- Legend: a floating control anchored to a map corner (captured
  // together with the map on export, instead of a separate block below it). ----------
  const legendControl = L.control({ position: "topright" });
  legendControl.onAdd = function () {
    const div = L.DomUtil.create("div", "legend-control");
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);

    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = "Click a line (or its swatch) to edit color/pattern/name. Circle size = passengers waiting.";
    div.appendChild(hint);

    for (const rid in scene.routes) {
      const route = scene.routes[rid];
      const row = document.createElement("div");
      row.className = "route-row";
      const swatch = document.createElement("div");
      swatch.className = "swatch";
      swatch.style.borderTopColor = route.color;
      swatch.style.borderTopStyle = route.dash ? "dashed" : "solid";
      swatch.style.cursor = "pointer";
      swatch.title = "Click to edit color/pattern";
      swatch.addEventListener("click", (ev) => openRouteEditor(i, rid, ev.clientX, ev.clientY));
      const input = document.createElement("input");
      input.className = "route-name";
      input.value = routeNames[i][rid];
      input.addEventListener("input", () => {
        routeNames[i][rid] = input.value;
        if (route._layer) route._layer.setTooltipContent(input.value);
      });
      row.appendChild(swatch);
      row.appendChild(input);
      div.appendChild(row);
      legendRows[i][rid] = { swatch, input };
    }

    const occRow = document.createElement("div");
    occRow.className = "occ-scale";
    occRow.innerHTML = '<span>Empty</span><div class="ramp"></div><span>Full</span>';
    div.appendChild(occRow);

    return div;
  };
  legendControl.addTo(map);

  for (const rid in scene.routes) {
    const route = scene.routes[rid];
    const latlngs = route.stops
      .map(sid => scene.stops[sid])
      .filter(Boolean)
      .map(s => [s.lat, s.lon]);
    if (latlngs.length < 2) continue;
    const layer = L.polyline(latlngs, {
      color: route.color, weight: 3, opacity: 0.85,
      dashArray: route.dash || null,
    }).addTo(map);
    layer.bindTooltip(routeNames[i][rid]);
    layer.on("click", (ev) => openRouteEditor(i, rid, ev.originalEvent.clientX, ev.originalEvent.clientY));
    route._layer = layer;
  }

  for (const stopId in scene.stops) {
    const s = scene.stops[stopId];
    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 3, weight: 1, color: "#333333",
      fillColor: "#898781", fillOpacity: 0.55,
    }).addTo(map);
    marker.bindPopup(() => {
      const div = document.createElement("div");
      const label = document.createElement("div");
      label.textContent = stopLabel(i, stopId) + ` (id ${stopId})`;
      label.style.marginBottom = "6px";
      const input = document.createElement("input");
      input.placeholder = "Rename this stop…";
      input.value = stopNames[i][stopId] || "";
      input.addEventListener("change", () => {
        stopNames[i][stopId] = input.value || null;
        marker.setTooltipContent(stopLabel(i, stopId));
      });
      div.appendChild(label);
      div.appendChild(input);
      return div;
    });
    marker.bindTooltip(stopLabel(i, stopId));
    stopMarkers[i][stopId] = marker;
  }

  for (const agentId in scene.agents) {
    const marker = L.circleMarker([0, 0], {
      radius: 6, weight: 1.5, color: "#000000", fillOpacity: 0.95,
    });
    marker.bindTooltip(agentId, { permanent: false });
    busMarkers[i][agentId] = marker;
  }
});

// Shared initial bounds across all panels, so side-by-side scenes are comparable at the same scale.
function computeBounds() {
  const pts = [];
  for (const scene of SCENES) {
    for (const id in scene.stops) pts.push([scene.stops[id].lat, scene.stops[id].lon]);
  }
  return pts.length ? L.latLngBounds(pts) : L.latLngBounds([[-13, -39], [-12.7, -38.3]]);
}
const BOUNDS = computeBounds();
maps.forEach(map => map.fitBounds(BOUNDS, { padding: [20, 20] }));

// Sync pan/zoom across panels when comparing two scenarios.
if (maps.length === 2) {
  let syncing = false;
  maps.forEach((map, i) => {
    const other = maps[1 - i];
    map.on("move zoom", () => {
      if (syncing) return;
      syncing = true;
      other.setView(map.getCenter(), map.getZoom(), { animate: false });
      syncing = false;
    });
  });
}

document.getElementById("basemapToggle").addEventListener("change", (ev) => {
  tileLayers.forEach(tiles => {
    if (ev.target.checked) tiles.addTo(maps[tileLayers.indexOf(tiles)]);
    else tiles.remove();
  });
});

// ---------- Lookup helpers ----------
function findBracket(events, t) {
  if (events.length === 0) return null;
  if (t <= events[0].t) return [events[0], events[0], 0];
  if (t >= events[events.length - 1].t) {
    const last = events[events.length - 1];
    return [last, last, 0];
  }
  let lo = 0, hi = events.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (events[mid].t <= t) lo = mid; else hi = mid;
  }
  const prev = events[lo], next = events[hi];
  const span = next.t - prev.t;
  return [prev, next, span > 0 ? (t - prev.t) / span : 0];
}

function waitingAt(samples, t) {
  if (!samples || samples.length === 0) return 0;
  if (t <= samples[0][0]) return 0;
  let lo = 0, hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid][0] <= t) lo = mid; else hi = mid;
  }
  return samples[lo][1];
}

// ---------- Global list of event times, for the step buttons ----------
const GLOBAL_TIMES = Array.from(new Set(
  SCENES.flatMap(scene => Object.values(scene.agents).flatMap(evs => evs.map(e => e.t)))
)).sort((a, b) => a - b);

function nextEventTime(t) {
  for (const et of GLOBAL_TIMES) if (et > t + 1e-6) return et;
  return t;
}
function prevEventTime(t) {
  for (let i = GLOBAL_TIMES.length - 1; i >= 0; i--) if (GLOBAL_TIMES[i] < t - 1e-6) return GLOBAL_TIMES[i];
  return t;
}

// ---------- Rendering ----------
function renderAt(t) {
  SCENES.forEach((scene, i) => {
    for (const stopId in scene.stops) {
      const waiting = waitingAt(scene.stop_waiting[stopId], t);
      const r = 2 + Math.min(16, Math.sqrt(Math.max(0, waiting)) * 2.4);
      const marker = stopMarkers[i][stopId];
      marker.setRadius(r);
      marker.setStyle({ fillOpacity: waiting > 0.05 ? 0.75 : 0.35 });
      marker.setTooltipContent(`${stopLabel(i, stopId)} — waiting: ${waiting.toFixed(1)}`);
    }

    for (const agentId in scene.agents) {
      const events = scene.agents[agentId];
      const bracket = findBracket(events, t);
      const marker = busMarkers[i][agentId];
      if (!bracket) { if (maps[i].hasLayer(marker)) marker.remove(); continue; }
      const [prev, next, frac] = bracket;
      const prevXY = scene.stops[prev.curr];
      const nextXY = scene.stops[next.curr];
      if (!prevXY || !nextXY) { if (maps[i].hasLayer(marker)) marker.remove(); continue; }
      const lat = prevXY.lat + (nextXY.lat - prevXY.lat) * frac;
      const lon = prevXY.lon + (nextXY.lon - prevXY.lon) * frac;
      marker.setLatLng([lat, lon]);
      const occ = prev.occ !== undefined ? prev.occ : 0;
      marker.setStyle({ fillColor: occColor(occ) });
      marker.setTooltipContent(`${agentId} — ${routeNames[i][prev.route] || prev.route} — occupancy: ${(occ * 100).toFixed(0)}%`);
      if (!maps[i].hasLayer(marker)) marker.addTo(maps[i]);
    }
  });
}

// ---------- Timeline controls ----------
const slider = document.getElementById("timeSlider");
const timeLabel = document.getElementById("timeLabel");
const playBtn = document.getElementById("playBtn");
const speedSel = document.getElementById("speedSel");

function fmtTime(t) {
  t = Math.max(0, Math.round(t));
  const hh = String(Math.floor(t / 3600)).padStart(2, "0");
  const mm = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
  const ss = String(Math.floor(t % 60)).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

let playing = false;
let lastFrameTs = null;

function setTime(t) {
  t = Math.max(0, Math.min(86400, t));
  slider.value = t;
  timeLabel.textContent = fmtTime(t);
  renderAt(t);
}

slider.addEventListener("input", () => setTime(parseFloat(slider.value)));

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
  lastFrameTs = null;
  if (playing) requestAnimationFrame(tick);
});

document.getElementById("stepFwdBtn").addEventListener("click", () => {
  playing = false; playBtn.textContent = "▶ Play";
  setTime(nextEventTime(parseFloat(slider.value)));
});
document.getElementById("stepBackBtn").addEventListener("click", () => {
  playing = false; playBtn.textContent = "▶ Play";
  setTime(prevEventTime(parseFloat(slider.value)));
});

function tick(ts) {
  if (!playing) return;
  if (lastFrameTs === null) lastFrameTs = ts;
  const dtReal = (ts - lastFrameTs) / 1000;
  lastFrameTs = ts;
  const speed = parseFloat(speedSel.value);
  let t = parseFloat(slider.value) + dtReal * speed;
  if (t >= 86400) { t = 86400; playing = false; playBtn.textContent = "▶ Play"; }
  setTime(t);
  if (playing) requestAnimationFrame(tick);
}

document.getElementById("exportBtn").addEventListener("click", () => {
  const panels = document.getElementById("panels");
  html2canvas(panels, { useCORS: true, scale: 2, backgroundColor: null }).then(canvas => {
    const link = document.createElement("a");
    link.download = `busenv_replay_${fmtTime(parseFloat(slider.value)).replace(/:/g, "-")}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }).catch(err => {
    alert("Export failed (this can happen if the map tiles haven't fully loaded yet, or the browser blocked a cross-origin canvas read). Try again after the map finishes loading, or hide the basemap and export a tile-free figure.");
    console.error(err);
  });
});

setTime(6 * 3600); // start at 6 AM, matching the simulated day's start
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render one or two replay scenes into a self-contained HTML viewer")
    parser.add_argument("--scene", required=True, help="Path to a scene JSON (from replay_export.py)")
    parser.add_argument("--scene2", default=None, help="Optional second scene JSON, rendered side by side")
    parser.add_argument("-o", "--out", default="replay_viewer.html", help="Output HTML path")
    args = parser.parse_args()

    scenes = []
    with open(args.scene, "r") as f:
        scenes.append(json.load(f))
    if args.scene2:
        with open(args.scene2, "r") as f:
            scenes.append(json.load(f))

    html = HTML_TEMPLATE.replace("__SCENES_JSON__", json.dumps(scenes))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)

    print(f"Viewer saved to {args.out} — open it in a browser (internet access required for the map tiles).")


if __name__ == "__main__":
    main()
