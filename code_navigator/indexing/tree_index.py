"""Vectorless hierarchical index.

    root (system)
     └── service
          └── module (folder)          <- skipped when a service has a single folder
               └── file
                    └── symbol (class / function / config block / doc section)

Every node carries a short summary. Summaries are written bottom-up by the LLM
(SUMMARY_MODE=llm) or assembled from symbols/endpoints (SUMMARY_MODE=heuristic).
At query time an LLM navigates this tree top-down instead of using vector similarity.
"""
from __future__ import annotations

import posixpath
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .. import prompts
from ..llm.providers import LLM, LLMError
from ..models import CodeUnit, Dependency, Service, SourceFile
from ..utils import first_sentence, load_json, log, save_json, sha1, truncate

Progress = Callable[[str, float], None]


class TreeIndex:
    def __init__(self, nodes: dict[str, dict], project: str = ""):
        self.nodes = nodes
        self.project = project

    ROOT = "root"

    def get(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    def root(self) -> dict:
        return self.nodes[self.ROOT]

    def children(self, node_id: str) -> list[dict]:
        node = self.nodes.get(node_id)
        return [self.nodes[c] for c in (node or {}).get("children", []) if c in self.nodes]

    def service_node(self, service: str) -> dict | None:
        return self.nodes.get(f"svc:{service}")

    def services(self) -> list[dict]:
        return self.children(self.ROOT)

    def files_under(self, node_id: str) -> list[dict]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        if node["kind"] == "file":
            return [node]
        out: list[dict] = []
        for c in self.children(node_id):
            if c["kind"] in ("module", "service", "file"):
                out.extend(self.files_under(c["id"]))
        return out

    def symbols_under(self, node_id: str) -> list[dict]:
        out = []
        for f in self.files_under(node_id):
            out.extend(c for c in self.children(f["id"]) if c["kind"] == "symbol")
        return out

    def save(self, path: Path) -> None:
        save_json(path, {"project": self.project, "nodes": self.nodes})

    @classmethod
    def load(cls, path: Path) -> "TreeIndex":
        data = load_json(path, {"project": "", "nodes": {}})
        return cls(data["nodes"], data.get("project", ""))


# ----------------------------------------------------------------- summaries

class _Summarizer:
    def __init__(self, llm: LLM | None, mode: str, cache: dict[str, str], workers: int):
        self.llm = llm
        self.mode = mode if llm is not None and llm.provider != "mock" else "heuristic"
        self.cache = cache
        self.workers = workers
        self.failures = 0

    def run(self, prompt: str, fallback: str, max_tokens: int = 160) -> str:
        if self.mode != "llm":
            return fallback
        key = sha1(prompt)
        if key in self.cache:
            return self.cache[key]
        if self.failures >= 5:  # LLM clearly unavailable - stop trying
            return fallback
        try:
            text = self.llm.complete(prompt, system=prompts.SUMMARY_SYSTEM, max_tokens=max_tokens, retries=1)
            text = " ".join(text.split())
            if not text:
                return fallback
            self.cache[key] = text
            return text
        except LLMError as exc:
            self.failures += 1
            log.warning("Summary generation failed (%s); using heuristic summary", exc)
            return fallback

    def map(self, fn, items: list) -> list:
        if self.mode != "llm" or self.workers <= 1 or len(items) < 2:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fn, items))


def _endpoint_str(endpoints: list[dict], limit: int = 8) -> str:
    if not endpoints:
        return "none"
    eps = [f"{e.get('method', '')} {e.get('path', '')}".strip() for e in endpoints[:limit]]
    more = f" (+{len(endpoints) - limit} more)" if len(endpoints) > limit else ""
    return ", ".join(dict.fromkeys(eps)) + more


def _symbol_summary(u: CodeUnit) -> str:
    s = f"{u.kind} {u.name}"
    if u.docstring:
        s += f": {first_sentence(u.docstring, 160)}"
    elif u.signature and u.kind not in ("doc", "config"):
        s += f" - {truncate(u.signature, 120)}"
    if u.endpoints:
        s += f" [endpoints: {_endpoint_str(u.endpoints, 4)}]"
    return s


def _short_module_name(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[-3:]) if parts else "(root)"


# ----------------------------------------------------------------- builder

def build_tree(project: str, services: list[Service], files: list[SourceFile], units: list[CodeUnit],
               edges: list[Dependency], llm: LLM | None, mode: str, cache: dict[str, str],
               workers: int = 4, progress: Progress | None = None) -> TreeIndex:
    summarizer = _Summarizer(llm, mode, cache, workers)
    file_by_path = {f.path: f for f in files}
    nodes: dict[str, dict] = {}

    def node(id_, kind, name, path, service, parent, meta=None) -> dict:
        n = {"id": id_, "kind": kind, "name": name, "path": path, "service": service,
             "summary": "", "children": [], "parent": parent, "meta": meta or {}}
        nodes[id_] = n
        if parent:
            nodes[parent]["children"].append(id_)
        return n

    node(TreeIndex.ROOT, "system", project, "", "", None)

    units_by_svc_file: dict[str, dict[str, list[CodeUnit]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        units_by_svc_file[u.service][u.file_path].append(u)

    svc_by_name = {s.name: s for s in services}
    ordered = [s for s in services if s.name in units_by_svc_file or s.kind == "infra"]
    for extra in units_by_svc_file:
        if extra not in svc_by_name:
            ordered.append(Service(name=extra, path="", kind="shared"))
            svc_by_name[extra] = ordered[-1]

    file_nodes: list[tuple[dict, list[CodeUnit]]] = []
    module_nodes: list[dict] = []
    for svc in ordered:
        sn = node(f"svc:{svc.name}", "service", svc.name, svc.path, svc.name, TreeIndex.ROOT,
                  {"kind": svc.kind, "language": svc.language,
                   "endpoints": _endpoint_str(svc.endpoints, 6) if svc.endpoints else ""})
        by_file = units_by_svc_file.get(svc.name, {})
        groups: dict[str, list[str]] = defaultdict(list)
        for path in sorted(by_file):
            rel = path[len(svc.path) + 1:] if svc.path and path.startswith(svc.path + "/") else path
            groups[posixpath.dirname(rel)].append(path)
        use_modules = len(groups) > 1
        for module_dir, paths in sorted(groups.items()):
            parent_id = sn["id"]
            if use_modules:
                full_dir = posixpath.join(svc.path, module_dir) if svc.path else module_dir
                mn = node(f"mod:{svc.name}:{module_dir or '.'}", "module",
                          _short_module_name(full_dir) if module_dir else "(service root files)",
                          full_dir, svc.name, sn["id"])
                module_nodes.append(mn)
                parent_id = mn["id"]
            for path in paths:
                file_units = sorted(by_file[path], key=lambda u: u.start_line)
                sf = file_by_path.get(path)
                fn = node(f"file:{svc.name}:{path}", "file", posixpath.basename(path), path, svc.name, parent_id,
                          {"language": sf.language if sf else file_units[0].language})
                for u in file_units:
                    node(f"sym:{u.id}", "symbol", u.name, path, svc.name, fn["id"],
                         {"unit_id": u.id, "kind": u.kind, "start_line": u.start_line, "end_line": u.end_line})
                    nodes[f"sym:{u.id}"]["summary"] = _symbol_summary(u)
                file_nodes.append((fn, file_units))

    # ---- file summaries (the expensive level, parallelised and cached)
    total = len(file_nodes)
    done = [0]

    def summarise_file(item):
        fn, file_units = item
        sf = file_by_path.get(fn["path"])
        symbols = [u for u in file_units if u.kind not in ("module",)]
        endpoints = [e for u in file_units for e in u.endpoints]
        classes = [u.name for u in symbols if u.kind == "class"][:6]
        funcs = [u.name.split(".")[-1] for u in symbols if u.kind in ("function", "method")][:10]
        other = [u.name for u in symbols if u.kind in ("config", "api_spec", "doc")][:6]
        parts = [f"{fn['meta'].get('language', '')} file".strip()]
        if classes:
            parts.append("classes " + ", ".join(classes))
        if funcs:
            parts.append("functions " + ", ".join(funcs))
        if other:
            parts.append(", ".join(other))
        if endpoints:
            parts.append("endpoints " + _endpoint_str(endpoints, 5))
        fallback = "; ".join(parts)
        code = sf.text if sf else "\n\n".join(u.code for u in file_units)
        prompt = prompts.FILE_SUMMARY.format(
            service=fn["service"], path=fn["path"], language=fn["meta"].get("language", ""),
            symbols=", ".join(u.name for u in symbols[:30]) or "none",
            endpoints=_endpoint_str(endpoints), code=truncate(code, 3000),
        )
        summary = summarizer.run(prompt, fallback)
        done[0] += 1
        if progress and (done[0] % 5 == 0 or done[0] == total):
            progress(f"Summarised {done[0]}/{total} files", done[0] / max(total, 1))
        return summary

    for (fn, _), summary in zip(file_nodes, summarizer.map(summarise_file, file_nodes)):
        fn["summary"] = summary

    # ---- module summaries
    def summarise_module(mn):
        kids = [nodes[c] for c in mn["children"]]
        listing = "\n".join(f"- {k['name']}: {k['summary']}" for k in kids[:30])
        fallback = f"{len(kids)} files: " + "; ".join(f"{k['name']} ({truncate(k['summary'], 80)})" for k in kids[:5])
        return summarizer.run(prompts.MODULE_SUMMARY.format(service=mn["service"], path=mn["path"], children=listing),
                              truncate(fallback, 400))

    for mn, summary in zip(module_nodes, summarizer.map(summarise_module, module_nodes)):
        mn["summary"] = summary

    # ---- service summaries
    outgoing: dict[str, list[Dependency]] = defaultdict(list)
    incoming: dict[str, list[Dependency]] = defaultdict(list)
    for e in edges:
        outgoing[e.source].append(e)
        incoming[e.target].append(e)

    service_nodes = [n for n in nodes.values() if n["kind"] == "service"]

    def summarise_service(sn):
        svc = svc_by_name.get(sn["name"])
        out = ", ".join(f"{e.target} ({e.kind})" for e in outgoing.get(sn["name"], [])) or "none"
        inc = ", ".join(f"{e.source} ({e.kind})" for e in incoming.get(sn["name"], [])) or "none"
        if svc and svc.kind == "infra":
            return f"Infrastructure component ({svc.language}) image {svc.image or 'n/a'}; used by: {inc}."
        kids = [nodes[c] for c in sn["children"]]
        listing = "\n".join(f"- {k['name']}: {truncate(k['summary'], 200)}" for k in kids[:25])
        eps = _endpoint_str(svc.endpoints if svc else [], 10)
        fallback = (f"{(svc.language if svc else '')} service. Endpoints: {eps}. Calls: {out}. Called by: {inc}."
                    + (f" Publishes: {', '.join(svc.topics_published)}." if svc and svc.topics_published else "")
                    + (f" Consumes: {', '.join(svc.topics_consumed)}." if svc and svc.topics_consumed else ""))
        prompt = prompts.SERVICE_SUMMARY.format(
            service=sn["name"], language=svc.language if svc else "unknown", endpoints=eps,
            grpc=", ".join(svc.grpc_services) if svc and svc.grpc_services else "none",
            published=", ".join(svc.topics_published) if svc and svc.topics_published else "none",
            consumed=", ".join(svc.topics_consumed) if svc and svc.topics_consumed else "none",
            outgoing=out, incoming=inc, children=listing,
        )
        return summarizer.run(prompt, fallback.strip(), max_tokens=220)

    if progress:
        progress("Summarising services", 0.95)
    for sn, summary in zip(service_nodes, summarizer.map(summarise_service, service_nodes)):
        sn["summary"] = summary
        if sn["name"] in svc_by_name:
            svc_by_name[sn["name"]].summary = summary

    # ---- system summary
    svc_lines = "\n".join(f"- {n['name']} ({n['meta'].get('language')}, {n['meta'].get('kind')}): {truncate(n['summary'], 300)}"
                          for n in service_nodes)
    edge_lines = "\n".join(f"- {e.source} -> {e.target} ({e.kind})" for e in edges[:80]) or "none detected"
    app_names = [n["name"] for n in service_nodes if n["meta"].get("kind") == "app"]
    fallback = (f"System '{project}' with {len(app_names)} application services ({', '.join(app_names)}) "
                f"and {len(edges)} detected dependencies.")
    nodes[TreeIndex.ROOT]["summary"] = summarizer.run(
        prompts.SYSTEM_SUMMARY.format(project=project, services=svc_lines, edges=edge_lines), fallback, max_tokens=300)

    log.info("Tree index: %d nodes (%d services, %d modules, %d files); summaries via %s",
             len(nodes), len(service_nodes), len(module_nodes), len(file_nodes), summarizer.mode)
    return TreeIndex(nodes, project)
