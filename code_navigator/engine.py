"""Query an existing index. Used by the CLI, the MCP server and as a Python library."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .config import Settings, get_settings
from .features.callgraph import DEFAULT_MIN_CONFIDENCE, CallGraph
from .features.docs_generator import generate_service_doc
from .features.graph import architecture_checks, graph_json, impact_analysis, to_dot
from .generation.answer_generator import AnswerGenerator
from .indexing.tree_index import TreeIndex
from .indexing.vector_index import open_vector_store
from .llm import get_embedder, get_llm
from .models import Dependency, Hit, Service
from .retrieval.hybrid_retriever import MODES, HybridRetriever
from .retrieval.tree_navigator import TreeNavigator
from .storage import IndexPaths
from .utils import load_json, load_jsonl, log, truncate


class NotIndexed(FileNotFoundError):
    pass


class Navigator:
    """Open the index of ``repo`` (default: ``<repo>/.code-navigator/``)."""

    def __init__(self, repo: str | Path = ".", settings: Settings | None = None,
                 index_dir: str | Path | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.settings = settings or get_settings(self.repo)
        self.paths = IndexPaths.for_repo(self.repo, index_dir)
        if not self.paths.exists():
            raise NotIndexed(f"No index found at {self.paths.root}. Run: code-navigator index {repo}")
        self.manifest = load_json(self.paths.manifest)
        self.project = self.manifest["project"]
        self.services = [Service.from_dict(d) for d in load_json(self.paths.services, [])]
        self.edges = [Dependency(**d) for d in load_json(self.paths.dependencies, [])]
        self.chunks = {c["id"]: c for c in load_jsonl(self.paths.chunks)}
        self.tree = TreeIndex.load(self.paths.tree)
        self.callgraph = CallGraph(load_json(self.paths.callgraph, {}))

        # Always query with the embedding model that built the index
        emb = self.manifest.get("embedding", {})
        if emb.get("provider") and (emb["provider"], emb["model"]) != (self.settings.embed_provider, self.settings.embed_model):
            log.info("Using index embedder %s:%s (differs from current settings)", emb["provider"], emb["model"])
        self.embedder = get_embedder(self.settings, emb.get("provider"), emb.get("model"))
        vs = self.manifest.get("vector_store") or {}
        self.store = open_vector_store(vs.get("backend") or self.manifest.get("vector_backend", self.settings.vector_backend),
                                       self.paths.vectors, self.settings, self.project, collection=vs.get("collection"))
        self.llm = get_llm(self.settings)
        self.navigator = TreeNavigator(self.tree, self.llm, self.edges, self.settings.nav_max_services,
                                       self.settings.nav_max_children)
        self.retriever = HybridRetriever(self.tree, self.store, self.embedder, self.navigator, self.chunks,
                                         self.edges, self.settings.nav_confidence_threshold)
        self.generator = AnswerGenerator(self.llm, self.settings.max_context_chars)

    def close(self) -> None:
        """Release the vector store (embedded Qdrant allows one open client per folder)."""
        self.store.close()

    # ------------------------------------------------------------------ Q&A
    def ask(self, question: str, mode: str | None = None, top_k: int | None = None,
            expand: int | None = None) -> dict:
        """Retrieve + generate a cited answer. Without an LLM the answer lists the best code."""
        mode = (mode or self.settings.retrieval_mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t = time.time()
        result = self._retrieve(question, mode, top_k, expand)
        answer = self.generator.answer(result)
        answer["timings"]["total"] = round(time.time() - t, 2)
        answer["llm"] = self.llm.describe() if self.settings.llm_enabled else None
        return answer

    def search(self, question: str, mode: str | None = None, top_k: int | None = None,
               include_code: bool = False, max_code_chars: int = 1500, expand: int | None = None) -> dict:
        """Retrieval only (no answer generation)."""
        mode = (mode or self.settings.retrieval_mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        result = self._retrieve(question, mode, top_k, expand)
        hits = []
        for h in result.hits:
            hit = {"chunk_id": h.chunk_id, "score": round(h.score, 5), "found_by": h.found_by, **h.metadata}
            if include_code:
                hit["code"] = truncate(self.chunks.get(h.chunk_id, {}).get("text", ""), max_code_chars)
            hits.append(hit)
        return {
            "question": question, "mode": result.mode, "hits": hits,
            "navigation": result.navigation.to_dict() if result.navigation else None,
            "notes": result.notes, "timings": result.timings,
        }

    def _retrieve(self, question: str, mode: str, top_k: int | None, expand: int | None):
        result = self.retriever.retrieve(question, mode=mode, top_k=top_k or self.settings.top_k)
        n = self.settings.call_expansion if expand is None else expand
        if n > 0 and self.callgraph:
            self._expand_with_calls(result, n)
        return result

    def _expand_with_calls(self, result, limit: int, seeds: int = 3, min_conf: float = 0.6) -> None:
        """Add the functions that the top hits call (and are called by) - code a cited answer
        usually needs but similarity search misses, e.g. the payment handler behind a client."""
        present = {h.chunk_id for h in result.hits}
        present_units = {h.metadata.get("unit_id") for h in result.hits}
        added: list[Hit] = []
        for h in result.hits[:seeds]:
            uid = h.metadata.get("unit_id")
            if not uid:
                continue
            # relation is read as "<hit> <relation> <expanded_from>"
            related = [(n, "called by") for n in self.callgraph.callees(uid, 1, min_conf)]
            related += [(n, "calls") for n in self.callgraph.callers(uid, 1, min_conf)]
            for node, relation in related:
                if len(added) >= limit:
                    break
                if node["id"] in present_units or node["kind"] not in ("function", "method"):
                    continue
                cids = self.retriever.by_unit.get(node["id"], [])
                if not cids or cids[0] in present:
                    continue
                hit = self.retriever._hit(cids[0], h.score * 0.5)
                hit.metadata = {**hit.metadata, "expanded_from": h.metadata.get("name"),
                                "relation": relation if node["kind_of_edge"] != "http" else f"{relation} (http)"}
                hit.found_by = ["call_graph"]
                added.append(hit)
                present.add(cids[0])
                present_units.add(node["id"])
        if added:
            result.hits.extend(added)
            result.notes.append(f"call graph added {len(added)} related function(s): "
                                + ", ".join(a.metadata.get("name", "") for a in added))
            in_scope = {n["name"] for n in result.services}
            for a in added:
                svc = a.metadata.get("service")
                node = self.tree.service_node(svc) if svc and svc not in in_scope else None
                if node is not None:
                    result.services.append(node)
                    in_scope.add(svc)

    # ------------------------------------------------------------------ call graph
    def _find_symbol(self, symbol: str, service: str | None) -> list[str]:
        if not self.callgraph:
            raise ValueError("this index has no call graph - re-index with code-navigator >= 0.2")
        ids = self.callgraph.find(symbol, service)
        if not ids:
            raise KeyError(f"no function or class named '{symbol}'" + (f" in {service}" if service else ""))
        return ids

    def calls(self, symbol: str, service: str | None = None, depth: int = 1,
              min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
        """What a function calls and what calls it (``depth`` levels each way)."""
        ids = self._find_symbol(symbol, service)
        uid = ids[0]
        return {
            "symbol": self.callgraph.describe(uid),
            "related": self.callgraph.related(uid),
            "other_matches": [self.callgraph.describe(i) for i in ids[1:10]],
            "callees": self.callgraph.callees(uid, depth, min_confidence),
            "callers": self.callgraph.callers(uid, depth, min_confidence),
            "entry_points": [e for e in self.callgraph.entry_points if e.get("handler") == uid],
        }

    def call_path(self, source: str, target: str, service: str | None = None,
                  min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
        """Shortest call chain from ``source`` to ``target`` (can cross services over HTTP)."""
        best: list[dict] = []
        for s in self._find_symbol(source, service)[:5]:
            for t in self._find_symbol(target, None)[:5]:
                hops = self.callgraph.path(s, t, min_conf=min_confidence)
                if hops and (not best or len(hops) < len(best)):
                    best = hops
        return {"source": source, "target": target, "found": bool(best), "hops": best}

    def affected(self, symbols: list[str] | None = None, files: list[str] | None = None,
                 git_ref: str | None = None, depth: int = 8,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
        """What may break if these functions, files or the current git diff change: callers
        (transitively, across services over HTTP), subclasses, message consumers, the HTTP
        endpoints and topics that reach them, and the tests among them."""
        if not self.callgraph:
            raise ValueError("this index has no call graph - re-index with a current code-navigator")
        uids: list[str] = []
        notes: list[str] = []
        for s in symbols or []:
            ids = self._find_symbol(s, None)
            uids.append(ids[0])
            if len(ids) > 1:
                notes.append(f"'{s}' matched {len(ids)} symbols; using {self.callgraph.describe(ids[0])['name']} "
                             f"in {self.callgraph.describe(ids[0])['service']}")
        for f in files or []:
            rel = self._relative(f)
            found = self.callgraph.units_at(rel)
            if not found:
                notes.append(f"no indexed functions in {rel}")
            uids += found
        diff_summary = None
        if git_ref is not None:
            changed = git_changed_lines(self.repo, git_ref)
            diff_summary = {p: (len(lines) if lines is not None else "whole file") for p, lines in changed.items()}
            for path, lines in changed.items():
                found = self.callgraph.units_at(path, lines)
                if not found and lines is None:
                    notes.append(f"{path}: new, deleted or not indexed code - nothing to trace")
                uids += found
            if not changed:
                notes.append(f"no changes against {git_ref}")
        uids = list(dict.fromkeys(uids))
        result = self.callgraph.affected(uids, depth=depth, min_conf=min_confidence)
        result["notes"] = notes
        if diff_summary is not None:
            result["diff"] = diff_summary
        return result

    def god_nodes(self, top: int = 10, service: str | None = None,
                  min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[dict]:
        """Most connected functions/classes - the riskiest places to change."""
        if not self.callgraph:
            raise ValueError("this index has no call graph - re-index with a current code-navigator")
        return self.callgraph.god_nodes(top=top, service=service, min_conf=min_confidence)

    def write_html(self, path: str | Path | None = None) -> Path:
        """Write the interactive graph viewer (default: <index>/graph.html)."""
        from .features.graph_html import graph_data, render_html
        out = Path(path).expanduser() if path else self.paths.graph_html
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(graph_data(self.project, self.services, self.edges, self.callgraph,
                                              self.manifest.get("stats"))), encoding="utf-8")
        return out

    def _relative(self, path: str) -> str:
        p = Path(path).expanduser()
        if p.is_absolute():
            try:
                return p.resolve().relative_to(self.repo).as_posix()
            except ValueError:
                return p.as_posix()
        return p.as_posix().lstrip("./") if p.as_posix().startswith("./") else p.as_posix()

    def entry_points(self, service: str | None = None) -> list[dict]:
        out = []
        for e in self.callgraph.entry_points:
            if service and e["service"] != service:
                continue
            out.append({**e, "handler_name": self.callgraph.describe(e["handler"])["name"] if e.get("handler") else None})
        return out

    def get_chunk(self, chunk_id: str) -> dict:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(chunk_id)
        return chunk

    # ------------------------------------------------------------------ architecture
    def overview(self) -> dict:
        return {
            "project": self.project,
            "summary": self.tree.root().get("summary", ""),
            "stats": self.manifest.get("stats", {}),
            "indexed_at": self.manifest.get("indexed_at"),
            "services": [
                {"name": s.name, "kind": s.kind, "language": s.language,
                 "summary": (self.tree.service_node(s.name) or {}).get("summary", s.summary)}
                for s in self.services if s.kind != "shared"
            ],
            "findings": architecture_checks(self.services, self.edges),
        }

    def get_service(self, name: str) -> Service | None:
        return next((s for s in self.services if s.name == name), None)

    def service_info(self, name: str) -> dict:
        svc = self.get_service(name)
        if svc is None:
            raise KeyError(f"unknown service '{name}'. Known: {', '.join(s.name for s in self.services)}")
        node = self.tree.service_node(name)
        files = self.tree.files_under(node["id"]) if node else []
        d = svc.to_dict()
        d["summary"] = (node or {}).get("summary", svc.summary)
        d["file_count"] = len(svc.files)
        d["files"] = [{"path": f["path"], "summary": f["summary"]} for f in files]
        d["depends_on"] = sorted({e.target for e in self.edges if e.source == name})
        d["used_by"] = sorted({e.source for e in self.edges if e.target == name})
        return d

    def dependencies(self, service: str | None = None) -> list[dict]:
        edges = [e for e in self.edges if service is None or service in (e.source, e.target)]
        return [e.to_dict() for e in edges]

    def graph(self, highlight: list[str] | None = None) -> dict:
        data = graph_json(self.services, self.edges)
        data["dot"] = to_dot(self.services, self.edges, set(highlight or []))
        return data

    def impact(self, service: str) -> dict:
        if self.get_service(service) is None:
            raise KeyError(f"unknown service '{service}'. Known: {', '.join(s.name for s in self.services)}")
        data = impact_analysis(service, self.services, self.edges)
        affected = {service} | {d["service"] for d in data["transitive_dependents"]}
        data["dot"] = to_dot(self.services, self.edges, affected)
        return data

    def service_doc(self, service: str) -> str:
        svc = self.get_service(service)
        if svc is None:
            raise KeyError(service)
        return generate_service_doc(svc, self.tree, self.edges, self.llm)

    def tree_view(self, node_id: str = "root") -> dict:
        """One node of the vectorless tree with its children - for step-by-step navigation."""
        node = self.tree.get(node_id)
        if node is None:
            raise KeyError(node_id)
        out = {k: node[k] for k in ("id", "kind", "name", "path", "service", "summary", "meta") if k in node}
        out["parent"] = node.get("parent")
        out["children"] = [
            {k: c[k] for k in ("id", "kind", "name", "path", "summary")} for c in self.tree.children(node_id)
        ]
        if node["kind"] == "symbol":
            unit_id = node.get("meta", {}).get("unit_id")
            out["chunk_ids"] = [cid for cid, c in self.chunks.items()
                                if c.get("metadata", {}).get("unit_id") == unit_id]
            if self.callgraph and unit_id in self.callgraph.nodes:
                out["calls"] = [n["name"] for n in self.callgraph.callees(unit_id)][:15]
                out["called_by"] = [n["name"] for n in self.callgraph.callers(unit_id)][:15]
        return out


def git_changed_lines(repo: Path, ref: str = "HEAD") -> dict[str, set[int] | None]:
    """Changed lines per file (relative to ``repo``) between ``ref`` and the working tree,
    including untracked files. ``None`` means the whole file (new or deleted)."""
    def git(*args: str) -> str:
        res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if res.returncode != 0:
            raise ValueError(f"git {' '.join(args[:2])} failed: {res.stderr.strip() or 'not a git repository?'}")
        return res.stdout

    top = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    prefix = repo.resolve().relative_to(top).as_posix() if repo.resolve() != top else ""

    def rel(path: str) -> str | None:
        if prefix:
            if not path.startswith(prefix + "/"):
                return None
            path = path[len(prefix) + 1:]
        return path

    changed: dict[str, set[int] | None] = {}
    old_path, cur = None, None
    for line in git("diff", "-U0", "--no-color", "--no-ext-diff", ref, "--").splitlines():
        if line.startswith("--- "):
            old_path = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            if line[4:] == "/dev/null":          # deleted file
                cur = None
                r = rel(old_path) if old_path else None
                if r:
                    changed[r] = None
            else:
                cur = rel(line[6:] if line.startswith("+++ b/") else line[4:])
                if cur and old_path is None:
                    changed[cur] = None              # new file
                elif cur:
                    changed.setdefault(cur, set())
        elif line.startswith("@@") and cur and changed.get(cur) is not None:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                changed[cur].update(range(start, start + count) if count else [max(start, 1)])
    for path in git("ls-files", "--others", "--exclude-standard", "--", ".").splitlines():
        r = path if not prefix else (path if not path.startswith(prefix + "/") else path[len(prefix) + 1:])
        changed[r] = None
    return changed
