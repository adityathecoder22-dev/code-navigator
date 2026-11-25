"""Service dependency graph: export, impact analysis and simple architecture checks."""
from __future__ import annotations

from collections import defaultdict, deque

from ..models import Dependency, Service

EDGE_STYLE = {
    "http": ('#2563eb', 'solid'),
    "grpc": ('#7c3aed', 'solid'),
    "async": ('#d97706', 'dashed'),
    "database": ('#059669', 'dotted'),
    "depends_on": ('#6b7280', 'dotted'),
    "config": ('#6b7280', 'dashed'),
}


def graph_json(services: list[Service], edges: list[Dependency]) -> dict:
    return {
        "nodes": [{"id": s.name, "kind": s.kind, "language": s.language,
                   "endpoints": len(s.endpoints), "summary": s.summary} for s in services if s.kind != "shared"],
        "edges": [e.to_dict() for e in edges],
    }


def to_dot(services: list[Service], edges: list[Dependency], highlight: set[str] | None = None) -> str:
    highlight = highlight or set()
    lines = ["digraph services {", '  rankdir=LR;', '  graph [fontname="Helvetica", bgcolor="transparent"];',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             '  edge [fontname="Helvetica", fontsize=9];']
    for s in services:
        if s.kind == "shared":
            continue
        if s.kind == "infra":
            attrs = 'shape=cylinder, fillcolor="#ecfdf5", color="#059669"'
        else:
            attrs = 'fillcolor="#eff6ff", color="#2563eb"'
        if s.name in highlight:
            attrs += ', penwidth=2.5, fillcolor="#fef3c7", color="#d97706"'
        label = f"{s.name}\\n({s.language})"
        lines.append(f'  "{s.name}" [label="{label}", {attrs}];')
    for e in edges:
        color, style = EDGE_STYLE.get(e.kind, ("#6b7280", "solid"))
        lines.append(f'  "{e.source}" -> "{e.target}" [label="{e.kind}", color="{color}", fontcolor="{color}", style={style}];')
    lines.append("}")
    return "\n".join(lines)


def impact_analysis(service: str, services: list[Service], edges: list[Dependency]) -> dict:
    """Who is affected if `service` changes or fails? (reverse reachability)"""
    incoming: dict[str, list[Dependency]] = defaultdict(list)
    outgoing: dict[str, list[Dependency]] = defaultdict(list)
    for e in edges:
        if e.kind == "depends_on":
            continue  # startup ordering, not a runtime call
        incoming[e.target].append(e)
        outgoing[e.source].append(e)

    affected: dict[str, int] = {}
    paths: dict[str, list[str]] = {service: [service]}
    queue = deque([service])
    while queue:
        cur = queue.popleft()
        for e in incoming.get(cur, []):
            if e.source not in affected and e.source != service:
                affected[e.source] = len(paths[cur])
                paths[e.source] = [e.source] + paths[cur]
                queue.append(e.source)

    svc = next((s for s in services if s.name == service), None)
    return {
        "service": service,
        "exists": svc is not None,
        "direct_dependents": sorted({e.source for e in incoming.get(service, [])}),
        "transitive_dependents": [{"service": s, "distance": d, "path": " -> ".join(paths[s])}
                                  for s, d in sorted(affected.items(), key=lambda x: (x[1], x[0]))],
        "depends_on": [{"service": e.target, "kind": e.kind, "evidence": e.evidence[:3]} for e in outgoing.get(service, [])],
        "called_by": [{"service": e.source, "kind": e.kind, "evidence": e.evidence[:3]} for e in incoming.get(service, [])],
        "endpoints": svc.endpoints if svc else [],
        "topics_published": svc.topics_published if svc else [],
        "topics_consumed": svc.topics_consumed if svc else [],
    }


def architecture_checks(services: list[Service], edges: list[Dependency]) -> list[dict]:
    """Detect common microservice smells from the dependency graph."""
    findings: list[dict] = []
    app = {s.name for s in services if s.kind == "app"}
    runtime = [e for e in edges if e.kind in ("http", "grpc", "async") and e.source in app and e.target in app]
    graph: dict[str, set[str]] = defaultdict(set)
    for e in runtime:
        graph[e.source].add(e.target)

    # cycles
    seen_cycles = set()
    for start in app:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in graph.get(node, ()):
                if nxt == start and len(path) > 1:
                    key = frozenset(path)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        findings.append({"type": "cyclic_dependency", "severity": "high",
                                         "detail": " -> ".join(path + [start])})
                elif nxt not in path and len(path) < 6:
                    stack.append((nxt, path + [nxt]))

    # shared databases
    db_users: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.kind == "database" and e.source in app:
            db_users[e.target].add(e.source)
    for db, users in db_users.items():
        if len(users) > 1:
            findings.append({"type": "shared_database", "severity": "medium",
                             "detail": f"{db} is used by {', '.join(sorted(users))}"})

    # hubs
    fan_in: dict[str, int] = defaultdict(int)
    for e in runtime:
        fan_in[e.target] += 1
    for s, n in graph.items():
        if len(n) >= 5:
            findings.append({"type": "high_fan_out", "severity": "low",
                             "detail": f"{s} calls {len(n)} services: {', '.join(sorted(n))}"})
    for s, n in fan_in.items():
        if n >= 5:
            findings.append({"type": "high_fan_in", "severity": "low",
                             "detail": f"{s} is called by {n} services (critical dependency)"})
    return findings
