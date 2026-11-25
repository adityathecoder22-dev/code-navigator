"""End-to-end indexing pipeline:

  load repo -> parse configs -> detect services -> parse code -> map dependencies
  -> chunk -> embed + vector store -> build vectorless tree -> save artifacts
"""
from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .. import __version__

from ..config import Settings, get_settings
from ..features.callgraph import CallGraph, build_call_graph
from ..features.graph_html import graph_data, render_html
from ..ingestion.code_parser import parse_code_file
from ..ingestion.config_parser import build_config_units, parse_configs
from ..ingestion.dependency_mapper import build_dependencies
from ..ingestion.repo_loader import load_repository
from ..ingestion.service_detector import detect_services
from ..llm import get_embedder, get_llm
from ..storage import GITIGNORE, IndexPaths
from ..utils import load_json, load_jsonl, log, save_json, save_jsonl, sha1
from .chunker import chunk_units
from .tree_index import build_tree
from .vector_index import LOCAL_DIRS, open_vector_store, redact

ProgressFn = Callable[[str, str, float], None]  # (stage, message, fraction 0..1)


def _noop(stage: str, message: str, fraction: float) -> None:
    log.info("[%s] %s", stage, message)


def _settings_for(source: str, settings: Settings | None, summary_mode: str | None) -> Settings:
    if settings is None:
        local = Path(source).expanduser()
        settings = get_settings(local if local.is_dir() else None)
    if summary_mode:
        settings = settings.with_overrides(summary_mode=summary_mode)
    return settings


def config_signature(settings: Settings, embedder_desc: str | None = None) -> dict:
    """Settings that change the index content - a change forces a rebuild in `update`."""
    llm_on = settings.summary_mode == "llm" and settings.llm_enabled
    return {
        "tool": __version__,
        "embedding": embedder_desc or f"{settings.embed_provider}:{settings.embed_model}",
        "summaries": f"llm:{settings.llm_provider}:{settings.summary_llm_model}" if llm_on else "heuristic",
        "chunk_max_lines": settings.chunk_max_lines,
        "include_tests": settings.include_tests,
        "max_file_bytes": settings.max_file_bytes,
        "vector_backend": settings.vector_backend,
        "vector_location": redact(settings.vector_url or ""),
        "vector_collection": settings.vector_collection or "",
    }


def _file_changes(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(p for p in set(old) & set(new) if old[p] != new[p]),
    }


def update_index(source: str = ".", name: str | None = None, settings: Settings | None = None,
                 summary_mode: str | None = None, progress: ProgressFn | None = None,
                 index_dir: str | Path | None = None, force: bool = False) -> dict:
    """Re-index only if files or index settings changed since the last run. Unchanged chunks
    keep their embeddings and unchanged files keep their LLM summaries, so updates are cheap.
    Returns the manifest, with ``up_to_date: True`` when nothing had to be done."""
    settings = _settings_for(source, settings, summary_mode)
    repo = load_repository(source, settings, name)
    paths = IndexPaths.for_repo(repo.root, index_dir)
    previous = load_json(paths.manifest) if paths.exists() else None
    if previous and not force:
        changes = _file_changes(previous.get("files", {}), {f.path: f.sha1 for f in repo.files})
        if previous.get("config_signature") == config_signature(settings) and not any(changes.values()):
            return {**previous, "up_to_date": True}
    return build_index(source, name=name, settings=settings, progress=progress, index_dir=index_dir, repo=repo)


def build_index(source: str = ".", name: str | None = None, settings: Settings | None = None,
                summary_mode: str | None = None, progress: ProgressFn | None = None,
                index_dir: str | Path | None = None, repo=None) -> dict:
    """Index a repository folder (or git URL) and write the index to <repo>/.code-navigator/
    (or to ``index_dir``). Returns the manifest."""
    settings = _settings_for(source, settings, summary_mode)
    progress = progress or _noop
    t0 = time.time()
    timings: dict[str, float] = {}

    def lap(stage: str, start: float) -> float:
        timings[stage] = round(time.time() - start, 2)
        return time.time()

    # 1. load
    progress("load", f"Loading repository {source}", 0.02)
    t = time.time()
    repo = repo or load_repository(source, settings, name)
    paths = IndexPaths.for_repo(repo.root, index_dir)
    previous = (load_json(paths.manifest) if paths.exists() else None) or {}
    t = lap("load", t)

    # 2. configs + services
    progress("services", "Parsing configuration and detecting services", 0.08)
    info = parse_configs(repo.files)
    services, compose_map, k8s_map = detect_services(repo, info)
    t = lap("services", t)

    # 3. code units
    progress("parse", "Parsing source code", 0.15)
    units = []
    for f in repo.files:
        if f.kind in ("code", "doc"):
            units.extend(parse_code_file(f))
    units.extend(build_config_units(repo.files, info, compose_map, k8s_map))
    t = lap("parse", t)

    # 4. dependencies
    progress("dependencies", "Mapping service dependencies", 0.25)
    edges = build_dependencies(repo.files, units, services, info, compose_map, k8s_map)
    t = lap("dependencies", t)

    # 4b. function-level call graph (inside services + HTTP calls across services)
    progress("callgraph", "Building the call graph", 0.27)
    callgraph = build_call_graph(units, services, repo.files, edges)
    t = lap("callgraph", t)

    # 5. chunks + embeddings
    chunks = chunk_units(units, settings.chunk_max_lines)
    seen_ids: dict[str, int] = {}
    for c in chunks:  # vector stores require unique ids
        if c.id in seen_ids:
            seen_ids[c.id] += 1
            c.id = f"{c.id}~{seen_ids[c.id]}"
        else:
            seen_ids[c.id] = 0
    progress("embed", f"Embedding {len(chunks)} chunks with {settings.embed_provider}:{settings.embed_model}", 0.3)
    embedder = get_embedder(settings)

    def embed_progress(done: int, total: int):
        progress("embed", f"Embedded {done}/{total} chunks", 0.3 + 0.3 * done / max(total, 1))

    # Open the vector store once and reset its collection in place. Deleting the files of a
    # database that is open (ChromaDB caches one client per path) leaves it read-only.
    store = open_vector_store(settings.vector_backend, paths.vectors, settings, repo.name)
    backend_used = store.backend

    # Reuse vectors of chunks whose text did not change (same embedding model only)
    reusable: dict[str, list[float]] = {}
    if previous.get("embedding") == {"provider": embedder.provider, "model": embedder.model}:
        try:
            prev = previous.get("vector_store") or {"backend": previous.get("vector_backend", backend_used)}
            same = prev.get("backend") == backend_used and prev.get("collection", store.collection) == store.collection
            old_store = store if same else open_vector_store(prev["backend"], paths.vectors, settings, repo.name,
                                                               collection=prev.get("collection"))
            old_vectors = old_store.vectors_by_id()
            if old_store is not store:
                old_store.close()
            for row in load_jsonl(paths.chunks):
                vec = old_vectors.get(row["id"])
                if vec is not None:
                    reusable[sha1(row["text"])] = vec
        except Exception as exc:  # a corrupt or foreign store just means re-embedding everything
            log.warning("Could not reuse previous embeddings: %s", exc)
    keys = [sha1(c.text) for c in chunks]
    todo = [i for i, k in enumerate(keys) if k not in reusable]
    fresh = embedder.embed_documents([chunks[i].text for i in todo], progress=embed_progress) if todo else []
    fresh_by_index = dict(zip(todo, fresh))
    vectors = [fresh_by_index[i] if i in fresh_by_index else reusable[keys[i]] for i in range(len(chunks))]
    store.reset()
    if chunks:
        store.add([c.id for c in chunks], vectors, [c.text for c in chunks], [c.metadata for c in chunks])
    for backend, folder in LOCAL_DIRS.items():  # folders left over from a previous embedded backend
        if folder != LOCAL_DIRS.get(backend_used) and (paths.vectors / folder).exists():
            shutil.rmtree(paths.vectors / folder, ignore_errors=True)
    store_info = store.describe()
    store.close()
    t = lap("embed", t)

    # 6. vectorless tree
    cache = load_json(paths.summary_cache, {}) or {}
    llm = (get_llm(settings.with_overrides(llm_model=settings.summary_llm_model))
           if settings.summary_mode == "llm" and settings.llm_enabled else None)
    summary_mode_used = "llm" if llm is not None else "heuristic"
    progress("tree", f"Building vectorless tree index (summaries: {summary_mode_used})", 0.62)

    def tree_progress(message: str, fraction: float):
        progress("tree", message, 0.62 + 0.35 * fraction)

    tree = build_tree(repo.name, services, repo.files, units, edges, llm, summary_mode_used,
                      cache, settings.summary_workers, tree_progress)
    t = lap("tree", t)

    # 7. save
    progress("save", "Saving index", 0.98)
    paths.root.mkdir(parents=True, exist_ok=True)
    save_json(paths.services, [s.to_dict() for s in services])
    save_json(paths.dependencies, [e.to_dict() for e in edges])
    save_jsonl(paths.chunks, [c.to_dict() for c in chunks])
    save_json(paths.callgraph, callgraph)
    try:
        paths.graph_html.write_text(render_html(graph_data(repo.name, services, edges, CallGraph(callgraph))),
                                    encoding="utf-8")
    except Exception as exc:  # the viewer is a convenience; never fail indexing because of it
        log.warning("Could not write graph.html: %s", exc)
    tree.save(paths.tree)
    save_json(paths.summary_cache, cache)
    if not paths.gitignore.exists():
        paths.gitignore.write_text(GITIGNORE, encoding="utf-8")

    langs: dict[str, int] = {}
    for f in repo.files:
        if f.kind == "code":
            langs[f.language] = langs.get(f.language, 0) + 1

    manifest = {
        "project": repo.name,
        "source": source,
        "repo_path": str(repo.root),
        "index_path": str(paths.root),
        "tool": f"code-navigator {__version__}",
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {
            "files": len(repo.files),
            "code_files": len(repo.by_kind("code")),
            "languages": langs,
            "services": len([s for s in services if s.kind == "app"]),
            "infra": len([s for s in services if s.kind == "infra"]),
            "units": len(units),
            "chunks": len(chunks),
            "dependencies": len(edges),
            "tree_nodes": len(tree.nodes),
            "call_edges": callgraph["stats"]["edges"],
            "cross_service_calls": callgraph["stats"]["cross_service_edges"],
            "entry_points": callgraph["stats"]["entry_points"],
        },
        "embedding": {"provider": embedder.provider, "model": embedder.model},
        "vector_backend": backend_used,
        "vector_store": store_info,
        "summary_mode": summary_mode_used,
        "config_signature": config_signature(settings, f"{embedder.provider}:{embedder.model}"),
        "files": {f.path: f.sha1 for f in repo.files},
        "incremental": {
            "previous_index": previous.get("indexed_at"),
            **(_file_changes(previous["files"], {f.path: f.sha1 for f in repo.files}) if previous.get("files")
               else {"added": [], "removed": [], "changed": []}),
            "embeddings_reused": len(chunks) - len(todo),
            "embeddings_computed": len(todo),
        },
        "llm": f"{settings.llm_provider}:{settings.summary_llm_model}" if llm is not None else None,
        "timings_sec": {**timings, "total": round(time.time() - t0, 2)},
    }
    save_json(paths.manifest, manifest)
    progress("done", f"Indexed {repo.name}: {len(chunks)} chunks, {manifest['stats']['services']} services", 1.0)
    return manifest
