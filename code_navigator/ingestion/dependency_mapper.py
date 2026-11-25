"""Build the service dependency graph from configuration and code."""
from __future__ import annotations

import posixpath
from collections import defaultdict

from ..models import CodeUnit, Dependency, Service, SourceFile
from ..utils import log, names_match
from . import code_analysis as ca
from .config_parser import ConfigInfo


class _EdgeSet:
    def __init__(self):
        self.edges: dict[tuple[str, str, str], Dependency] = {}

    def add(self, source: str, target: str, kind: str, evidence: str) -> None:
        if not source or not target or source == target or "_shared" in (source, target):
            return
        key = (source, target, kind)
        dep = self.edges.get(key)
        if dep is None:
            dep = self.edges[key] = Dependency(source, target, kind, [])
        if evidence not in dep.evidence and len(dep.evidence) < 6:
            dep.evidence.append(evidence)

    def all(self) -> list[Dependency]:
        return sorted(self.edges.values(), key=lambda d: (d.source, d.target, d.kind))


def _kind_for(target: Service, app_kind: str) -> str:
    if target.kind == "app":
        return app_kind
    return "async" if target.language == "broker" else "database"


def _resolve(name: str, services: list[Service]) -> str | None:
    """Map a hostname / identifier to a known service name."""
    name = name.lower()
    if name in ("localhost", "127.0.0.1", "0.0.0.0", "example.com"):
        return None
    host = name.split(".")[0]  # order-service.default.svc.cluster.local -> order-service
    for svc in services:
        if names_match(svc.name, host):
            return svc.name
    return None


def build_dependencies(files: list[SourceFile], units: list[CodeUnit], services: list[Service],
                       info: ConfigInfo, compose_map: dict[str, str], k8s_map: dict[str, str]) -> list[Dependency]:
    edges = _EdgeSet()
    svc_by_name = {s.name: s for s in services}
    infra_names = [s for s in services if s.kind == "infra"]

    # --- 1. docker-compose depends_on + environment values pointing to other services
    for cs in info.compose.values():
        src = compose_map.get(cs.name, cs.name)
        for dep in cs.depends_on:
            tgt = compose_map.get(dep, dep)
            kind = _kind_for(svc_by_name[tgt], "depends_on") if tgt in svc_by_name else "depends_on"
            edges.add(src, tgt, kind, f"{cs.file}: depends_on {dep}")
        for key, value in cs.env.items():
            for host, _ in ca.extract_hosts(f'"{value}"') + [(value.split(":")[0], 0)]:
                tgt = _resolve(host, services)
                if tgt:
                    kind = _kind_for(svc_by_name[tgt], "config")
                    edges.add(src, tgt, kind, f"{cs.file}: {key}={value}")

    # --- 2. kubernetes env vars
    for wl in info.k8s.values():
        src = k8s_map.get(wl.name, wl.name)
        for key, value in wl.env.items():
            for host, _ in ca.extract_hosts(f'"{value}"') + [(str(value).split(":")[0], 0)]:
                tgt = _resolve(host, services)
                if tgt:
                    kind = _kind_for(svc_by_name[tgt], "config")
                    edges.add(src, tgt, kind, f"{wl.file}: {key}={value}")

    # env var name -> service it points to (per source service), used to resolve code lookups
    env_targets: dict[str, dict[str, str]] = defaultdict(dict)
    for svc in services:
        for key, value in svc.env.items():
            for host, _ in ca.extract_hosts(f'"{value}"') + [(str(value).split(":")[0], 0)]:
                tgt = _resolve(host, services)
                if tgt and tgt != svc.name:
                    env_targets[svc.name][key] = tgt

    # --- 3. code: endpoints, URLs, env lookups, topics, gRPC, datastores
    publishers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    proto_names = [p.name for p in info.proto_services]
    grpc_owner: dict[str, str] = {}
    code_files = [f for f in files if f.kind == "code" and f.service in svc_by_name]
    units_by_file: dict[str, list[CodeUnit]] = defaultdict(list)
    module_by_file: dict[str, CodeUnit] = {}
    for u in units:
        if u.kind in ("function", "method"):
            units_by_file[u.file_path].append(u)
        elif u.kind == "module":
            module_by_file[u.file_path] = u

    # Express router prefixes: app.use('/api/orders', router) in one file applies to another file
    route_prefix: dict[str, str] = {}
    known_paths = {f.path for f in code_files}
    for f in code_files:
        if f.language not in ("javascript", "typescript", "tsx"):
            continue
        for prefix, module in ca.express_mounts(f.text):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(f.path), module))
            for cand in (base, base + ".js", base + ".ts", base + "/index.js", base + "/index.ts"):
                if cand in known_paths:
                    route_prefix[cand] = prefix

    for f in code_files:
        svc = svc_by_name[f.service]
        text = f.text

        for ep in ca.extract_endpoints(text, f.language):
            ep = {**ep, "file": f.path}
            if f.path in route_prefix and ep["framework"] == "express":
                ep["path"] = (route_prefix[f.path] + ("" if ep["path"] == "/" else ep["path"])) or "/"
            svc.endpoints.append(ep)
            candidates = units_by_file.get(f.path, [])
            owner = next((u for u in candidates if u.start_line <= ep["line"] <= u.end_line), None)
            if owner is None:  # annotation/decorator just above the definition
                owner = next((u for u in candidates if 0 < u.start_line - ep["line"] <= 3), None)
            if owner is None:  # route registered at module level (Express, Go mux)
                owner = module_by_file.get(f.path)
            if owner is not None:
                owner.endpoints.append(ep)

        for host, line in ca.extract_hosts(text):
            tgt = _resolve(host, services)
            if tgt:
                kind = _kind_for(svc_by_name[tgt], "http")
                edges.add(svc.name, tgt, kind, f"{f.path}:{line} references {host}")

        for var, line in ca.extract_env_refs(text):
            tgt = env_targets.get(svc.name, {}).get(var)
            if not tgt:  # naming convention: PAYMENT_SERVICE_ADDR / PAYMENT_URL
                cleaned = var.lower()
                for suffix in ("_addr", "_address", "_url", "_uri", "_host", "_endpoint", "_base_url"):
                    if cleaned.endswith(suffix):
                        tgt = _resolve(cleaned[: -len(suffix)].replace("_", "-"), services)
                        break
            if tgt:
                kind = _kind_for(svc_by_name[tgt], "http")
                edges.add(svc.name, tgt, kind, f"{f.path}:{line} reads env {var}")

        for t in ca.extract_topics(text):
            (publishers if t["direction"] == "pub" else consumers)[t["topic"]].add(svc.name)
            target_list = svc.topics_published if t["direction"] == "pub" else svc.topics_consumed
            if t["topic"] not in target_list:
                target_list.append(t["topic"])

        for store in ca.extract_datastores(text):
            for infra in infra_names:
                if store in infra.name.lower() or store in (infra.image or "").lower():
                    edges.add(svc.name, infra.name, "database", f"{f.path} uses {store}")

        for pname in proto_names:
            if ca.grpc_server_refs(text, pname):
                grpc_owner.setdefault(pname, svc.name)
                if pname not in svc.grpc_services:
                    svc.grpc_services.append(pname)

    for pname in proto_names:
        if pname not in grpc_owner:
            owner = _resolve(pname, services)
            if owner:
                grpc_owner[pname] = owner
                if pname not in svc_by_name[owner].grpc_services:
                    svc_by_name[owner].grpc_services.append(pname)

    for f in code_files:
        for pname, owner in grpc_owner.items():
            if f.service != owner and ca.grpc_client_refs(f.text, pname):
                edges.add(f.service, owner, "grpc", f"{f.path} creates {pname} client")

    # --- 4. async messaging: publisher -> consumer of the same topic
    for topic, pubs in publishers.items():
        for p in pubs:
            for c in consumers.get(topic, set()):
                edges.add(p, c, "async", f"topic '{topic}'")
    broker = next((s for s in infra_names if any(k in s.name.lower() for k in ("kafka", "rabbit", "nats"))), None)
    if broker:
        for topic in set(publishers) | set(consumers):
            for s in publishers.get(topic, set()) | consumers.get(topic, set()):
                edges.add(s, broker.name, "async", f"topic '{topic}'")

    # Prefer the most specific edge kind: drop 'config' / 'depends_on' if http/grpc/async exists
    result = edges.all()
    strong = {(d.source, d.target) for d in result if d.kind in ("http", "grpc", "async")}
    result = [d for d in result if not (d.kind in ("config", "depends_on") and (d.source, d.target) in strong)]

    for svc in services:
        svc.endpoints.sort(key=lambda e: (e.get("file", ""), e.get("line", 0)))
    log.info("Dependency graph: %d edges", len(result))
    return result


def dependency_maps(edges: list[Dependency]) -> tuple[dict[str, list[Dependency]], dict[str, list[Dependency]]]:
    outgoing: dict[str, list[Dependency]] = defaultdict(list)
    incoming: dict[str, list[Dependency]] = defaultdict(list)
    for e in edges:
        outgoing[e.source].append(e)
        incoming[e.target].append(e)
    return outgoing, incoming
