"""Self-contained interactive graph viewer (one HTML file, no network access needed).

Two views: services (dependency graph) and functions (call graph). Pan, zoom, drag, search,
filter by service / edge kind / confidence, click a node for its callers, callees and
blast radius, and a list of god nodes.
"""
from __future__ import annotations

import html
import json

from ..models import Dependency, Service
from .callgraph import CallGraph


def graph_data(project: str, services: list[Service], deps: list[Dependency], cg: CallGraph,
               stats: dict | None = None) -> dict:
    svc_rows = [{"id": s.name, "kind": s.kind, "language": s.language, "endpoints": len(s.endpoints),
                 "summary": (s.summary or "")[:400]} for s in services if s.kind != "shared"]
    svc_names = {s["id"] for s in svc_rows}
    svc_edges = [{"s": d.source, "t": d.target, "k": d.kind, "ev": (d.evidence or [""])[0][:160]}
                 for d in deps if d.source in svc_names and d.target in svc_names]
    nodes = [{"id": uid, "n": n["name"], "k": n["kind"], "s": n["service"], "f": n["file"], "l": n["start_line"]}
             for uid, n in cg.nodes.items()]
    edges = [{"s": e["source"], "t": e["target"], "k": e["kind"], "c": e["confidence"], "r": e["resolution"],
              "l": e["line"], **({"d": e["detail"]} if e.get("detail") else {})} for e in cg.edges]
    entry = [{"h": e["handler"], "m": e["method"], "p": e["path"]} for e in cg.entry_points if e.get("handler")]
    gods = [{"id": g["id"], "n": g["name"], "s": g["service"], "in": g["fan_in"], "out": g["fan_out"],
             "dep": g["transitive_dependents"]} for g in cg.god_nodes(12)] if cg else []
    return {"project": project, "stats": stats or {}, "services": svc_rows, "serviceEdges": svc_edges,
            "nodes": nodes, "edges": edges, "entry": entry, "gods": gods}


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", html.escape(f"{data['project']} - code-navigator graph"))
            .replace("__DATA__", payload))


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#ffffff;--panel:#f6f7f9;--text:#1f2328;--muted:#656d76;--border:#d8dee4;--accent:#2563eb;--hi:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--panel:#161b22;--text:#e6edf3;--muted:#8d96a0;--border:#30363d;--accent:#58a6ff;--hi:#f87171}}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
#app{display:grid;grid-template-columns:260px 1fr 320px;grid-template-rows:44px 1fr;height:100%}
header{grid-column:1/4;display:flex;align-items:center;gap:16px;padding:0 14px;border-bottom:1px solid var(--border);background:var(--panel)}
header b{font-size:14px}header span{color:var(--muted)}
aside{overflow:auto;padding:12px;border-right:1px solid var(--border);background:var(--panel)}
#detail{border-right:0;border-left:1px solid var(--border)}
h3{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
label{display:flex;align-items:center;gap:6px;cursor:pointer;padding:1px 0}
input[type=search]{width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text)}
.seg{display:flex;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.seg button{flex:1;padding:6px;border:0;background:var(--bg);color:var(--text);cursor:pointer}
.seg button.on{background:var(--accent);color:#fff}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
.ln{width:18px;height:0;border-top:2px solid;display:inline-block;flex:none}
#canvas{width:100%;height:100%;display:block;cursor:grab}
main{position:relative;overflow:hidden}
#hint{position:absolute;left:10px;bottom:8px;color:var(--muted);font-size:12px;pointer-events:none}
.item{padding:3px 6px;border-radius:4px;cursor:pointer;display:flex;justify-content:space-between;gap:8px}
.item:hover{background:var(--border)}
.item small{color:var(--muted);white-space:nowrap}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}
.tag{display:inline-block;padding:0 6px;border-radius:10px;border:1px solid var(--border);color:var(--muted);font-size:11px;margin-right:4px}
button.act{margin-top:8px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);cursor:pointer}
@media (max-width:900px){html,body{height:auto}#app{height:auto;grid-template-columns:1fr;grid-template-rows:44px auto 70vh auto}header{grid-column:1}aside{border:0;overflow:visible}main{border-top:1px solid var(--border);border-bottom:1px solid var(--border)}}
</style>
</head>
<body>
<div id="app">
<header><b id="title"></b><span id="stats"></span></header>
<aside>
  <div class="seg"><button id="vFn" class="on">Functions</button><button id="vSvc">Services</button></div>
  <h3>Search</h3><input id="q" type="search" placeholder="function, class or service, then Enter">
  <h3>Services</h3><div id="svcs"></div>
  <h3>Edges</h3><div id="kinds"></div>
  <h3>Min confidence <span id="cv"></span></h3><input id="conf" type="range" min="0" max="1" step="0.05" value="0.5" style="width:100%">
  <h3>God nodes</h3><div id="gods"></div>
</aside>
<main><canvas id="canvas"></canvas><div id="hint">drag to pan, scroll to zoom, click a node</div></main>
<aside id="detail"><h3>Details</h3><div id="info" style="color:var(--muted)">Click a node to see what it calls, what calls it and what a change to it would affect.</div></aside>
</div>
<script>
const D = __DATA__;
const PALETTE = ["#2563eb","#16a34a","#d97706","#9333ea","#dc2626","#0891b2","#db2777","#65a30d","#7c3aed","#ea580c","#0d9488","#4f46e5"];
const KIND = {calls:["#8b949e",[]],instantiates:["#8b949e",[2,3]],http:["#2563eb",[6,4]],async:["#d97706",[6,4]],
  grpc:["#7c3aed",[6,4]],database:["#059669",[2,3]],inherits:["#9333ea",[]],implements:["#9333ea",[4,3]],
  imports:["#c9d1d9",[1,3]],depends_on:["#8b949e",[2,3]],config:["#8b949e",[2,3]]};
const svcColor = {}; D.services.forEach((s, i) => svcColor[s.id] = s.kind === "infra" ? "#6b7280" : PALETTE[i % PALETTE.length]);
const color = s => svcColor[s] || "#9ca3af";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("title").textContent = D.project;
const st = D.stats || {};
$("stats").textContent = `${D.services.filter(s=>s.kind!=="infra").length} services · ${D.nodes.filter(n=>n.k!=="file").length} functions/classes · ${D.edges.length} edges`;

let view = "fn", minConf = 0.5, selected = null, hover = null, impact = null, query = "";
const hiddenSvc = new Set(), kindsOn = new Set(["calls","instantiates","http","async","inherits","implements","grpc","database","depends_on"]);

/* ---------- graph model per view ---------- */
function build() {
  let nodes, edges;
  if (view === "svc") {
    nodes = D.services.map(s => ({id: s.id, n: s.id, k: s.kind === "infra" ? "infra" : "service", s: s.id, f: s.language, l: null, sum: s.summary}));
    edges = D.serviceEdges.map(e => ({s: e.s, t: e.t, k: e.k, c: 1, r: e.ev}));
  } else {
    nodes = D.nodes.map(n => ({...n})); edges = D.edges.map(e => ({...e}));
  }
  const byId = new Map(nodes.map(n => [n.id, n]));
  edges = edges.filter(e => byId.has(e.s) && byId.has(e.t));
  nodes.forEach(n => { n.deg = 0; n.out = []; n.inc = []; });
  edges.forEach(e => { const a = byId.get(e.s), b = byId.get(e.t); e.a = a; e.b = b; a.out.push(e); b.inc.push(e); if (e.k !== "imports") { a.deg++; b.deg++; } });
  layoutInit(nodes);
  return {nodes, edges, byId};
}
let G;

/* ---------- layout ---------- */
function layoutInit(nodes) {
  const svcs = [...new Set(nodes.map(n => n.s))]; const R = 60 + 40 * Math.sqrt(nodes.length);
  const center = {}; svcs.forEach((s, i) => { const a = 2 * Math.PI * i / Math.max(1, svcs.length); center[s] = [R * Math.cos(a), R * Math.sin(a)]; });
  nodes.forEach(n => { const c = center[n.s] || [0, 0]; n.cx = c[0]; n.cy = c[1];
    n.x = c[0] + (Math.random() - .5) * 80; n.y = c[1] + (Math.random() - .5) * 80; n.vx = 0; n.vy = 0; });
}
let alpha = 1;
function visible(n) { return !hiddenSvc.has(n.s) && (view === "svc" || n.k !== "file" || kindsOn.has("imports")); }
function edgeOn(e) { return kindsOn.has(e.k) && (e.c ?? 1) >= minConf && visible(e.a) && visible(e.b); }
function tick() {
  const ns = G.nodes.filter(visible), N = ns.length; if (!N) return;
  const rep = view === "svc" ? 9000 : 1400, cell = 120, grid = new Map();
  for (const n of ns) { const k = Math.floor(n.x / cell) + "," + Math.floor(n.y / cell); (grid.get(k) || grid.set(k, []).get(k)).push(n); }
  for (const n of ns) {
    const gx = Math.floor(n.x / cell), gy = Math.floor(n.y / cell);
    for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
      const b = grid.get((gx + dx) + "," + (gy + dy)); if (!b) continue;
      for (const m of b) { if (m === n) continue; let x = n.x - m.x, y = n.y - m.y, d2 = x * x + y * y + .01;
        if (d2 > 90000) continue; const f = rep / d2 * alpha; n.vx += x * f / Math.sqrt(d2); n.vy += y * f / Math.sqrt(d2); }
    }
  }
  const L = view === "svc" ? 160 : 70;
  for (const e of G.edges) { if (!edgeOn(e) || e.a === e.b) continue; const a = e.a, b = e.b;
    let x = b.x - a.x, y = b.y - a.y, d = Math.sqrt(x * x + y * y) || 1; const f = (d - L) * .02 * alpha * (e.k === "imports" ? .2 : 1);
    x /= d; y /= d; a.vx += x * f; a.vy += y * f; b.vx -= x * f; b.vy -= y * f; }
  for (const n of ns) { if (n === drag) continue; n.vx += (n.cx - n.x) * .004 * alpha; n.vy += (n.cy - n.y) * .004 * alpha;
    n.vx *= .82; n.vy *= .82; n.x += n.vx; n.y += n.vy; }
  alpha = Math.max(.02, alpha * .985);
}

/* ---------- canvas ---------- */
const cv = $("canvas"), ctx = cv.getContext("2d"); let W = 0, H = 0, scale = 1, tx = 0, ty = 0, dpr = 1;
function resize() { const r = cv.getBoundingClientRect(); dpr = window.devicePixelRatio || 1; W = r.width; H = r.height; cv.width = W * dpr; cv.height = H * dpr; }
window.addEventListener("resize", resize);
const radius = n => view === "svc" ? 10 + Math.sqrt(n.deg) * 3 : (n.k === "file" ? 3 : 3.5 + Math.sqrt(n.deg) * 1.6);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function focusSet() {
  if (impact) return impact;
  const c = selected || hover; if (!c) return null;
  const s = new Set([c.id]); c.out.concat(c.inc).forEach(e => { if (edgeOn(e)) { s.add(e.a.id); s.add(e.b.id); } }); return s;
}
function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * (W / 2 + tx), dpr * (H / 2 + ty));
  const fs = focusSet(), text = css("--text"), muted = css("--muted"), hi = css("--hi");
  for (const e of G.edges) { if (!edgeOn(e)) continue; const on = !fs || (fs.has(e.a.id) && fs.has(e.b.id));
    const [col, dash] = KIND[e.k] || KIND.calls; ctx.globalAlpha = on ? (fs ? .95 : .55) : .06; ctx.strokeStyle = col;
    ctx.lineWidth = (on && fs ? 1.6 : 1) / scale; ctx.setLineDash(dash.map(v => v / scale)); ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.lineTo(e.b.x, e.b.y); ctx.stroke();
    if (e.a !== e.b && (on || !fs)) { const ang = Math.atan2(e.b.y - e.a.y, e.b.x - e.a.x), r = radius(e.b) + 2 / scale, hx = e.b.x - Math.cos(ang) * r, hy = e.b.y - Math.sin(ang) * r, s = 5 / scale;
      ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx - s * Math.cos(ang - .45), hy - s * Math.sin(ang - .45)); ctx.lineTo(hx - s * Math.cos(ang + .45), hy - s * Math.sin(ang + .45)); ctx.closePath(); ctx.fillStyle = col; ctx.fill(); } }
  ctx.setLineDash([]);
  const q = query.toLowerCase();
  for (const n of G.nodes) { if (!visible(n)) continue; const on = !fs || fs.has(n.id), match = q && n.n.toLowerCase().includes(q);
    ctx.globalAlpha = on ? 1 : .12; ctx.beginPath(); const r = radius(n);
    if (n.k === "class" || n.k === "service") ctx.rect(n.x - r, n.y - r, 2 * r, 2 * r); else ctx.arc(n.x, n.y, r, 0, 7);
    ctx.fillStyle = n.k === "file" ? muted : color(n.s); ctx.fill();
    if (n === selected || match || (impact && impact.has(n.id) && n !== selected)) { ctx.lineWidth = 2.5 / scale; ctx.strokeStyle = n === selected ? text : (match ? css("--accent") : hi); ctx.stroke(); }
    const big = view === "svc" || n.deg >= 6, show = on && (fs ? fs.has(n.id) : (scale > 1.3 || big)) && n.k !== "file";
    if (show || n === hover || match) { ctx.globalAlpha = on ? 1 : .4; ctx.fillStyle = text; ctx.font = `${12 / scale}px -apple-system,Segoe UI,sans-serif`;
      ctx.fillText(view === "svc" ? n.n : n.n.split(".").slice(-2).join("."), n.x + r + 3 / scale, n.y + 4 / scale); } }
  ctx.globalAlpha = 1;
}
function loop() { if (alpha > .021 || drag) tick(); draw(); requestAnimationFrame(loop); }

/* ---------- interaction ---------- */
const world = (cx, cy) => { const r = cv.getBoundingClientRect(); return [(cx - r.left - W / 2 - tx) / scale, (cy - r.top - H / 2 - ty) / scale]; };
function pick(cx, cy) { const [x, y] = world(cx, cy); let best = null, bd = 1e9;
  for (const n of G.nodes) { if (!visible(n)) continue; const d = Math.hypot(n.x - x, n.y - y); if (d < radius(n) + 4 / scale && d < bd) { bd = d; best = n; } } return best; }
let drag = null, pan = null, moved = false;
cv.addEventListener("mousedown", ev => { moved = false; const n = pick(ev.clientX, ev.clientY); if (n) { drag = n; alpha = Math.max(alpha, .3); } else pan = [ev.clientX - tx, ev.clientY - ty]; cv.style.cursor = "grabbing"; });
window.addEventListener("mousemove", ev => { if (drag) { const [x, y] = world(ev.clientX, ev.clientY); drag.x = x; drag.y = y; drag.vx = drag.vy = 0; moved = true; }
  else if (pan) { tx = ev.clientX - pan[0]; ty = ev.clientY - pan[1]; moved = true; }
  else if (ev.target === cv) { hover = pick(ev.clientX, ev.clientY); cv.style.cursor = hover ? "pointer" : "grab"; } });
window.addEventListener("mouseup", ev => { if (!moved && ev.target === cv) select(pick(ev.clientX, ev.clientY)); drag = null; pan = null; cv.style.cursor = "grab"; });
cv.addEventListener("wheel", ev => { ev.preventDefault(); const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left - W / 2, my = ev.clientY - r.top - H / 2;
  const k = Math.exp(-ev.deltaY * .0015), ns = Math.min(8, Math.max(.08, scale * k)); tx = mx - (mx - tx) * ns / scale; ty = my - (my - ty) * ns / scale; scale = ns; }, {passive: false});
function center(n) { tx = -n.x * scale; ty = -n.y * scale; }

/* ---------- details panel ---------- */
function blast(n) { // what may break if n changes: callers (transitively), subclasses, consumers of what it publishes
  const seen = new Set([n.id]), q = [n];
  while (q.length) { const c = q.shift();
    for (const e of c.inc) if (e.k !== "imports" && e.k !== "async" && (e.c ?? 1) >= minConf && !seen.has(e.a.id)) { seen.add(e.a.id); q.push(e.a); }
    for (const e of c.out) if (e.k === "async" && (e.c ?? 1) >= minConf && !seen.has(e.b.id)) { seen.add(e.b.id); q.push(e.b); } }
  return seen;
}
function list(title, rows) { if (!rows.length) return ""; return `<h3>${title} (${rows.length})</h3>` + rows.map(([node, label]) =>
  `<div class="item" data-id="${esc(node.id)}"><span><span class="sw" style="background:${color(node.s)}"></span> ${esc(node.n)}</span><small>${esc(label)}</small></div>`).join(""); }
function select(n) {
  selected = n; impact = null;
  if (!n) { $("info").innerHTML = '<span style="color:var(--muted)">Click a node to see what it calls, what calls it and what a change to it would affect.</span>'; return; }
  const lab = e => e.k + (e.d ? " · " + e.d : "") + (e.c != null && e.c < 1 ? " · " + e.c.toFixed(2) : "");
  const callees = n.out.filter(e => !["inherits","implements","imports"].includes(e.k)).map(e => [e.b, lab(e)]);
  const callers = n.inc.filter(e => !["inherits","implements","imports"].includes(e.k)).map(e => [e.a, lab(e)]);
  const bases = n.out.filter(e => e.k === "inherits" || e.k === "implements").map(e => [e.b, e.k]);
  const subs = n.inc.filter(e => e.k === "inherits" || e.k === "implements").map(e => [e.a, e.k]);
  const eps = view === "fn" ? D.entry.filter(e => e.h === n.id) : [];
  const svc = view === "svc" ? D.services.find(s => s.id === n.id) : null;
  $("info").innerHTML = `<div style="font-size:15px;font-weight:600;word-break:break-all">${esc(n.n)}</div>
    <div style="margin:4px 0"><span class="tag">${esc(n.k)}</span><span class="tag" style="color:${color(n.s)}">${esc(n.s)}</span></div>
    ${n.l ? `<div class="mono">${esc(n.f)}:${n.l}</div>` : `<div class="mono">${esc(n.f || "")}</div>`}
    ${svc && svc.summary ? `<p>${esc(svc.summary)}</p>` : ""}
    ${eps.map(e => `<div><span class="tag">${esc(e.m)}</span><span class="mono">${esc(e.p)}</span></div>`).join("")}
    <button class="act" id="imp">Show what a change affects</button>
    ${list(view === "svc" ? "Depends on" : "Calls", callees)}${list(view === "svc" ? "Used by" : "Called by", callers)}
    ${list("Extends / implements", bases)}${list("Subclasses / implementations", subs)}`;
  $("imp").onclick = () => { impact = blast(n); const k = impact.size - 1;
    $("imp").textContent = k ? `${k} node${k > 1 ? "s" : ""} may be affected (highlighted)` : "Nothing depends on this"; };
  document.querySelectorAll("#info .item").forEach(el => el.onclick = () => { const m = G.byId.get(el.dataset.id); if (m) { select(m); center(m); } });
}

/* ---------- controls ---------- */
function renderFilters() {
  $("svcs").innerHTML = D.services.map(s => `<label><input type="checkbox" data-s="${esc(s.id)}" ${hiddenSvc.has(s.id) ? "" : "checked"}><span class="sw" style="background:${color(s.id)}"></span>${esc(s.id)}</label>`).join("");
  document.querySelectorAll("#svcs input").forEach(i => i.onchange = () => { i.checked ? hiddenSvc.delete(i.dataset.s) : hiddenSvc.add(i.dataset.s); alpha = .5; });
  const present = [...new Set(G.edges.map(e => e.k))].sort();
  $("kinds").innerHTML = present.map(k => `<label><input type="checkbox" data-k="${k}" ${kindsOn.has(k) ? "checked" : ""}><span class="ln" style="border-color:${(KIND[k] || KIND.calls)[0]};${(KIND[k] || KIND.calls)[1].length ? "border-top-style:dashed" : ""}"></span>${k} <small style="color:var(--muted)">${G.edges.filter(e => e.k === k).length}</small></label>`).join("");
  document.querySelectorAll("#kinds input").forEach(i => i.onchange = () => { i.checked ? kindsOn.add(i.dataset.k) : kindsOn.delete(i.dataset.k); alpha = .5; });
  $("gods").innerHTML = view === "svc" ? '<span style="color:var(--muted)">Switch to Functions</span>' : D.gods.map(g =>
    `<div class="item" style="display:block" data-id="${esc(g.id)}" title="${esc(g.n)}"><div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"><span class="sw" style="background:${color(g.s)}"></span> ${esc(g.n.split(".").slice(-2).join("."))}</div><small>${g.in} callers · ${g.out} callees · ${g.dep} affected by a change</small></div>`).join("") || '<span style="color:var(--muted)">none</span>';
  document.querySelectorAll("#gods .item").forEach(el => el.onclick = () => { const m = G.byId.get(el.dataset.id); if (m) { select(m); center(m); } });
}
function setView(v) { view = v; $("vFn").classList.toggle("on", v === "fn"); $("vSvc").classList.toggle("on", v === "svc");
  G = build(); selected = hover = impact = null; select(null); alpha = 1; tx = ty = 0; scale = v === "svc" ? 1 : .8; renderFilters(); }
$("vFn").onclick = () => setView("fn"); $("vSvc").onclick = () => setView("svc");
$("conf").oninput = e => { minConf = +e.target.value; $("cv").textContent = minConf.toFixed(2); };
$("q").oninput = e => { query = e.target.value.trim(); };
$("q").onkeydown = e => { if (e.key === "Enter" && query) { const q = query.toLowerCase();
  const m = G.nodes.filter(n => visible(n) && n.n.toLowerCase().includes(q)).sort((a, b) => b.deg - a.deg)[0]; if (m) { select(m); center(m); } } };
window.addEventListener("keydown", e => { if (e.key === "Escape") { select(null); } });
$("cv").textContent = minConf.toFixed(2);
resize(); setView(D.nodes.length ? "fn" : "svc"); requestAnimationFrame(loop);
</script>
</body>
</html>
"""
