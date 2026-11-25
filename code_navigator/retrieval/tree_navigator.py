"""Vectorless retrieval: an LLM walks the tree index top-down
(system -> services -> modules -> files [-> symbols]) choosing where to look next.

If the LLM is unavailable or returns invalid JSON, a keyword-overlap scorer takes
over for that step, so navigation always produces a result."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import prompts
from ..indexing.tree_index import TreeIndex
from ..llm.providers import LLM, LLMError
from ..models import Dependency
from ..utils import log, tokenize, truncate

SMALL_SERVICE_FILES = 8      # don't drill into services this small
MAX_FILES_PER_MODULE = 6     # above this, ask the LLM to pick files inside a module
MAX_SYMBOL_OPTIONS = 60


@dataclass
class NavigationResult:
    services: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)          # file paths
    file_nodes: list[str] = field(default_factory=list)     # tree node ids
    symbols: list[str] = field(default_factory=list)        # unit ids (vectorless mode)
    whole_services: list[str] = field(default_factory=list) # services searched without file narrowing
    scope: str = "local"
    confidence: float = 0.0
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _keyword_scores(question: str, options: list[dict]) -> list[float]:
    q = set(tokenize(question))
    scores = []
    for opt in options:
        name_toks = set(tokenize(opt.get("name", "") + " " + opt.get("path", "")))
        body_toks = set(tokenize(opt.get("summary", "")))
        scores.append(3.0 * len(q & name_toks) + 1.0 * len(q & body_toks))
    return scores


class TreeNavigator:
    def __init__(self, tree: TreeIndex, llm: LLM, edges: list[Dependency], max_services: int = 3,
                 max_children: int = 4, max_llm_calls: int = 6):
        self.tree = tree
        self.llm = llm
        self.edges = edges
        self.max_services = max_services
        self.max_children = max_children
        self.max_llm_calls = max_llm_calls

    # ------------------------------------------------------------ LLM choice helper
    def _choose(self, prompt: str, labels: dict[str, dict], question: str, max_select: int,
                result: NavigationResult, step: str) -> tuple[list[dict], dict]:
        data: dict = {}
        if result.llm_calls < self.max_llm_calls and self.llm.provider != "mock":
            try:
                result.llm_calls += 1
                data = self.llm.complete_json(prompt, system=prompts.NAV_SYSTEM, default={}) or {}
            except LLMError as exc:
                log.warning("Navigation LLM call failed: %s", exc)
                data = {}
        raw = data.get("selected") if isinstance(data, dict) else None
        chosen: list[dict] = []
        if isinstance(raw, list):
            by_name = {opt["name"].lower(): opt for opt in labels.values()}
            for item in raw:
                key = str(item).strip()
                opt = labels.get(key) or labels.get(key.upper()) or by_name.get(key.lower())
                if opt and opt not in chosen:
                    chosen.append(opt)
                if len(chosen) >= max_select:
                    break
        method = "llm"
        if not chosen and isinstance(raw, list) and any(str(r).upper() == "ALL" for r in raw):
            chosen = list(labels.values())[:max_select]
        if not chosen:  # keyword fallback
            method = "keyword"
            opts = list(labels.values())
            scores = _keyword_scores(question, opts)
            ranked = sorted(zip(scores, range(len(opts))), key=lambda x: -x[0])
            chosen = [opts[i] for s, i in ranked if s > 0][:max_select]
            if not chosen:
                chosen = opts[:max_select] if len(opts) <= max_select else []
            data = {"confidence": 0.3 if chosen else 0.0, "reason": "keyword match (LLM selection unavailable)"}
        result.trace.append({
            "step": step,
            "options": len(labels),
            "selected": [c["name"] for c in chosen],
            "reason": str(data.get("reason", ""))[:300],
            "method": method,
        })
        return chosen, data

    # ------------------------------------------------------------ main
    def navigate(self, question: str, depth: str = "file", max_symbols: int = 8) -> NavigationResult:
        result = NavigationResult()
        root = self.tree.root()
        services = [s for s in self.tree.services() if s["meta"].get("kind") != "infra"]
        if not services:
            return result

        # Step 1: services
        talks: dict[str, list[str]] = {}
        for e in self.edges:
            talks.setdefault(e.source, []).append(f"{e.target}({e.kind})")
        labels = {}
        lines = []
        for i, s in enumerate(services, start=1):
            label = f"S{i}"
            labels[label] = s
            eps = s["meta"].get("endpoints", "")
            lines.append(f"{label} | {s['name']} | {s['meta'].get('language', '')} | {truncate(s['summary'], 350)}"
                         f"{' | endpoints: ' + eps if eps else ''} | talks to: {', '.join(talks.get(s['name'], [])[:6]) or '-'}")
        prompt = prompts.NAV_SERVICES.format(question=question, overview=truncate(root.get("summary", ""), 800),
                                             options="\n".join(lines), max_select=self.max_services)
        chosen, data = self._choose(prompt, labels, question, self.max_services, result, "services")
        scope = str(data.get("scope", "local")).lower()
        result.scope = scope if scope in ("local", "cross_service", "architecture") else "local"
        result.confidence = _to_float(data.get("confidence"), 0.5)
        result.services = [c["name"] for c in chosen]
        if len(result.services) > 1 and result.scope == "local":
            result.scope = "cross_service"
        if result.scope == "architecture":
            return result  # whole-service context is enough; the vector search stays service-scoped

        # Step 2: drill into each chosen service
        for svc_node in chosen:
            all_files = self.tree.files_under(svc_node["id"])
            if len(all_files) <= SMALL_SERVICE_FILES:
                self._add_files(result, all_files)
                result.trace.append({"step": f"{svc_node['name']}: files", "options": len(all_files),
                                     "selected": [f["name"] for f in all_files], "reason": "small service - all files",
                                     "method": "rule"})
                continue
            children = self.tree.children(svc_node["id"])
            picked = self._pick_children(question, svc_node, children, result)
            if not picked:  # could not narrow down - search the whole service
                result.whole_services.append(svc_node["name"])
                continue
            for child in picked:
                if child["kind"] == "file":
                    self._add_files(result, [child])
                    continue
                files = self.tree.files_under(child["id"])
                if len(files) > MAX_FILES_PER_MODULE:
                    files = self._pick_children(question, child, files, result, max_select=4) or files[:4]
                self._add_files(result, files)

        if depth == "symbol":
            self._pick_symbols(question, result, max_symbols)
        return result

    def _add_files(self, result: NavigationResult, files: list[dict]) -> None:
        for f in files:
            if f["id"] not in result.file_nodes:
                result.file_nodes.append(f["id"])
                if f["path"] not in result.files:
                    result.files.append(f["path"])

    def _pick_children(self, question: str, parent: dict, children: list[dict], result: NavigationResult,
                       max_select: int | None = None) -> list[dict]:
        max_select = max_select or self.max_children
        if len(children) <= max_select:
            return children
        labels, lines = {}, []
        for i, c in enumerate(children[:80], start=1):
            label = f"C{i}"
            labels[label] = c
            lines.append(f"{label} | {c['kind']} | {c['name']} | {truncate(c['summary'], 220)}")
        prompt = prompts.NAV_CHILDREN.format(
            question=question, location=f"{parent['kind']} {parent['name']} ({parent.get('path', '')})",
            location_summary=truncate(parent.get("summary", ""), 400), options="\n".join(lines), max_select=max_select)
        chosen, data = self._choose(prompt, labels, question, max_select, result, f"{parent['name']}: children")
        if data.get("confidence") is not None:
            result.confidence = min(result.confidence or 1.0, _to_float(data["confidence"], 0.5))
        return chosen

    def _pick_symbols(self, question: str, result: NavigationResult, max_symbols: int) -> None:
        symbols = []
        for fid in result.file_nodes:
            symbols.extend(c for c in self.tree.children(fid) if c["kind"] == "symbol")
        if not symbols:
            return
        if len(symbols) > MAX_SYMBOL_OPTIONS:
            scores = _keyword_scores(question, symbols)
            symbols = [s for _, s in sorted(zip(scores, symbols), key=lambda x: -x[0])][:MAX_SYMBOL_OPTIONS]
        if len(symbols) <= max_symbols:
            result.symbols = [s["meta"]["unit_id"] for s in symbols]
            return
        labels, lines = {}, []
        for i, s in enumerate(symbols, start=1):
            label = f"F{i}"
            labels[label] = s
            lines.append(f"{label} | {s['path']} | {s['meta'].get('kind')} | {s['name']} | {truncate(s['summary'], 160)}")
        prompt = prompts.NAV_SYMBOLS.format(question=question, options="\n".join(lines), max_select=max_symbols)
        chosen, _ = self._choose(prompt, labels, question, max_symbols, result, "symbols")
        result.symbols = [c["meta"]["unit_id"] for c in chosen]
