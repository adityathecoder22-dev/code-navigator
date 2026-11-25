"""Hybrid retrieval = vectorless tree navigation + vector search.

  hybrid     : navigate the tree to find the relevant services/files, then run vector
               search *inside* that scope. Falls back to a global vector search when the
               navigator is unsure. Exact symbol-name matches are added. Results fused by RRF.
  vector     : plain vector search over everything (+ symbol matches).
  vectorless : the navigator goes all the way down to symbols; no embeddings used.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from ..indexing.tree_index import TreeIndex
from ..indexing.vector_index import VectorStore
from ..llm.embeddings import Embedder
from ..models import Dependency, Hit
from .tree_navigator import NavigationResult, TreeNavigator

MODES = ("hybrid", "vector", "vectorless")


@dataclass
class RetrievalResult:
    question: str
    mode: str
    hits: list[Hit]
    navigation: NavigationResult | None = None
    services: list[dict] = field(default_factory=list)     # tree nodes of services in scope
    edges: list[Dependency] = field(default_factory=list)  # dependencies among them
    overview: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(ranked_lists: dict[str, list[Hit]], weights: dict[str, float] | None = None,
                           k: int = 60) -> list[Hit]:
    """score(d) = sum_r  w_r / (k + rank_r(d))"""
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, Hit] = {}
    found_by: dict[str, list[str]] = defaultdict(list)
    for name, hits in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] += w / (k + rank)
            if hit.chunk_id not in best:
                best[hit.chunk_id] = hit
            if name not in found_by[hit.chunk_id]:
                found_by[hit.chunk_id].append(name)
    fused = []
    for cid in sorted(scores, key=scores.get, reverse=True):
        h = best[cid]
        fused.append(Hit(chunk_id=cid, score=round(scores[cid], 5), text=h.text, metadata=h.metadata,
                         found_by=found_by[cid]))
    return fused


_IDENT = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[a-z0-9]+_[a-z0-9_]+)(?:\(\))?")


def code_identifiers(question: str) -> list[str]:
    out = []
    for m in _IDENT.finditer(question):
        ident = (m.group(1) or m.group(2) or "").strip().rstrip("()")
        if ident and ident.lower() not in out:
            out.append(ident.lower())
    return out


class HybridRetriever:
    def __init__(self, tree: TreeIndex, store: VectorStore, embedder: Embedder, navigator: TreeNavigator,
                 chunks: dict[str, dict], edges: list[Dependency], confidence_threshold: float = 0.6):
        self.tree = tree
        self.store = store
        self.embedder = embedder
        self.navigator = navigator
        self.chunks = chunks
        self.edges = edges
        self.threshold = confidence_threshold
        self.by_unit: dict[str, list[str]] = defaultdict(list)
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_file: dict[str, list[str]] = defaultdict(list)
        for cid, c in chunks.items():
            meta = c["metadata"]
            self.by_unit[c["unit_id"]].append(cid)
            self.by_file[meta["file_path"]].append(cid)
            self.by_name[meta["name"].lower()].append(cid)
            self.by_name[meta.get("short_name", "")].append(cid)
            self.by_name[meta["file_path"].split("/")[-1].lower()].append(cid)

    def _hit(self, cid: str, score: float = 0.0) -> Hit:
        c = self.chunks[cid]
        return Hit(chunk_id=cid, score=score, text=c["text"], metadata=c["metadata"])

    def symbol_hits(self, question: str, limit: int = 6) -> list[Hit]:
        hits, seen = [], set()
        for ident in code_identifiers(question):
            for key in (ident, ident.split(".")[-1]):
                for cid in self.by_name.get(key, []):
                    if cid not in seen:
                        seen.add(cid)
                        hits.append(self._hit(cid, 1.0))
        return hits[:limit]

    def retrieve(self, question: str, mode: str = "hybrid", top_k: int = 8) -> RetrievalResult:
        mode = mode if mode in MODES else "hybrid"
        timings: dict[str, float] = {}
        notes: list[str] = []
        lists: dict[str, list[Hit]] = {}
        weights = {"scoped_vector": 1.0, "global_vector": 0.7, "symbol": 1.2, "tree": 1.0}
        nav: NavigationResult | None = None

        symbol = self.symbol_hits(question)
        if symbol:
            lists["symbol"] = symbol

        if mode in ("hybrid", "vectorless"):
            t = time.time()
            nav = self.navigator.navigate(question, depth="symbol" if mode == "vectorless" else "file",
                                          max_symbols=top_k)
            timings["navigation"] = round(time.time() - t, 2)

        if mode == "vectorless":
            tree_hits = []
            for uid in (nav.symbols if nav else []):
                for cid in self.by_unit.get(uid, []):
                    tree_hits.append(self._hit(cid))
            if not tree_hits and nav:  # navigator stopped at file/service level
                for f in nav.files:
                    tree_hits.extend(self._hit(cid) for cid in self.by_file.get(f, [])[:3])
                for svc in nav.whole_services:
                    svc_node = self.tree.service_node(svc)
                    for f in (self.tree.files_under(svc_node["id"]) if svc_node else [])[:top_k]:
                        tree_hits.extend(self._hit(cid) for cid in self.by_file.get(f["path"], [])[:1])
            lists["tree"] = tree_hits
        else:
            t = time.time()
            q_vec = self.embedder.embed_query(question)
            timings["embed_query"] = round(time.time() - t, 2)
            t = time.time()
            use_global = mode == "vector"
            if mode == "hybrid" and nav is not None:
                if nav.files or nav.services:
                    scoped = self.store.query(q_vec, top_k * 2,
                                              services=(nav.whole_services or None) if nav.files else nav.services,
                                              files=nav.files or None)
                    if nav.files and nav.scope != "local":
                        # also allow service-level matches outside the chosen files (configs, other modules)
                        scoped += self.store.query(q_vec, top_k, services=nav.services)
                    lists["scoped_vector"] = _dedupe(scoped)
                if nav.confidence < self.threshold or len(lists.get("scoped_vector", [])) < max(2, top_k // 2):
                    use_global = True
                    notes.append(f"navigator confidence {nav.confidence:.2f} - added global vector search")
            if use_global:
                lists["global_vector"] = self.store.query(q_vec, top_k * 2)
                if mode == "vector":
                    weights["global_vector"] = 1.0
            timings["vector_search"] = round(time.time() - t, 2)

        hits = reciprocal_rank_fusion(lists, weights)[:top_k]

        # context: services in scope + their dependencies
        scope_services = set(nav.services if nav else [])
        scope_services |= {h.metadata.get("service") for h in hits}
        svc_nodes = [n for n in (self.tree.service_node(s) for s in scope_services if s) if n]
        edges = [e for e in self.edges if e.source in scope_services or e.target in scope_services]
        overview = self.tree.root().get("summary", "") if (nav and nav.scope == "architecture") or len(svc_nodes) > 2 else ""
        return RetrievalResult(question=question, mode=mode, hits=hits, navigation=nav, services=svc_nodes,
                               edges=edges, overview=overview, timings=timings, notes=notes)


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in sorted(hits, key=lambda x: -x.score):
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            out.append(h)
    return out
