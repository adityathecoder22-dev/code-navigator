"""Vector store backends.

    simple    numpy matrix in <index>/vectors/simple       (default, no extra dependency)
    chroma    ChromaDB in <index>/vectors/chroma            pip install "code-navigator[chroma]"
    qdrant    Qdrant, embedded in <index>/vectors/qdrant or a server (CODENAV_VECTOR_URL)
    lancedb   LanceDB in <index>/vectors/lancedb (or CODENAV_VECTOR_URL, e.g. s3://...)
    pgvector  PostgreSQL + pgvector (CODENAV_VECTOR_URL=postgresql://...)

Every backend stores the same thing: chunk id, embedding, chunk text and the chunk metadata
(service, file_path, kind, name, lines ...), and supports search restricted to services/files.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Hit
from ..utils import log


BACKENDS = ("simple", "chroma", "qdrant", "lancedb", "pgvector")
LOCAL_DIRS = {"simple": "simple", "chroma": "chroma", "qdrant": "qdrant", "lancedb": "lancedb"}


class VectorStore(ABC):
    backend = "base"
    location = ""
    collection = "code"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], k: int, services: list[str] | None = None,
              files: list[str] | None = None) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def vectors_by_id(self) -> dict[str, list[float]]:
        """All stored vectors - used to reuse embeddings of unchanged chunks on re-index."""

    def describe(self) -> dict:
        return {"backend": self.backend, "location": redact(self.location), "collection": self.collection}

    def close(self) -> None:
        """Release files/connections (some embedded databases allow one open client per folder)."""


def redact(location: str) -> str:
    """Hide passwords in connection strings before they are written to the manifest."""
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", location or "")


def default_collection(project: str) -> str:
    """Name for a collection/table on a shared server: one per indexed project."""
    return "codenav_" + (re.sub(r"[^a-z0-9_]+", "_", project.lower()).strip("_") or "project")[:50]


def _clean_meta(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v)
    return out


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self, path: Path, collection: str = "code"):
        self.location, self.collection = str(path), collection
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('chromadb is not installed: pip install "code-navigator[chroma]"') from exc
        path.mkdir(parents=True, exist_ok=True)
        try:
            from chromadb.config import Settings as ChromaSettings  # type: ignore
            self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        except Exception:
            self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.col = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, documents, metadatas) -> None:
        batch = 2000
        for i in range(0, len(ids), batch):
            self.col.add(
                ids=ids[i : i + batch],
                embeddings=embeddings[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=[_clean_meta(m) for m in metadatas[i : i + batch]],
            )

    def query(self, embedding, k, services=None, files=None) -> list[Hit]:
        where = None
        clauses = []
        if files:
            clauses.append({"file_path": {"$in": list(files)}})
        if services:
            clauses.append({"service": {"$in": list(services)}})
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$or": clauses}
        n = max(1, min(k, self.count()))
        res = self.col.query(query_embeddings=[embedding], n_results=n, where=where,
                             include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(chunk_id=cid, score=float(1.0 - dist), text=doc, metadata=dict(meta)))
        return hits

    def count(self) -> int:
        return self.col.count()

    def vectors_by_id(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        total, step = self.count(), 2000
        for offset in range(0, total, step):
            res = self.col.get(include=["embeddings"], limit=step, offset=offset)
            for cid, vec in zip(res["ids"], res["embeddings"]):
                out[cid] = list(vec)
        return out


class SimpleVectorStore(VectorStore):
    """Brute-force cosine similarity in numpy; persisted as .npy + .json. Fine up to ~100k chunks."""
    backend = "simple"

    def __init__(self, path: Path):
        self.path = path
        self.location = str(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._vec_file = path / "vectors.npy"
        self._meta_file = path / "records.json"
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metas: list[dict] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        if self._vec_file.exists() and self._meta_file.exists():
            self.matrix = np.load(self._vec_file)
            records = json.loads(self._meta_file.read_text(encoding="utf-8"))
            self.ids, self.docs, self.metas = records["ids"], records["docs"], records["metas"]

    def reset(self) -> None:
        self.ids, self.docs, self.metas = [], [], []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self._save()

    def _save(self) -> None:
        np.save(self._vec_file, self.matrix)
        self._meta_file.write_text(json.dumps({"ids": self.ids, "docs": self.docs, "metas": self.metas}),
                                   encoding="utf-8")

    def add(self, ids, embeddings, documents, metadatas) -> None:
        vecs = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
        self.matrix = vecs if self.matrix.size == 0 else np.vstack([self.matrix, vecs])
        self.ids += list(ids)
        self.docs += list(documents)
        self.metas += [_clean_meta(m) for m in metadatas]
        self._save()

    def query(self, embedding, k, services=None, files=None) -> list[Hit]:
        if not self.ids:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        with np.errstate(all="ignore"):  # numpy 2.0 + macOS Accelerate emits spurious matmul warnings
            scores = self.matrix @ q
        if services or files:
            sset, fset = set(services or []), set(files or [])
            mask = np.array([(m.get("service") in sset) or (m.get("file_path") in fset) for m in self.metas])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [Hit(chunk_id=self.ids[i], score=float(scores[i]), text=self.docs[i], metadata=dict(self.metas[i]))
                for i in order if np.isfinite(scores[i])]

    def count(self) -> int:
        return len(self.ids)

    def vectors_by_id(self) -> dict[str, list[float]]:
        return {cid: self.matrix[i] for i, cid in enumerate(self.ids)} if self.matrix.size else {}


def open_vector_store(backend: str, path: Path, settings: Any = None, project: str = "code",
                      collection: str | None = None) -> VectorStore:
    """Open the store for ``backend``. ``path`` is <index>/vectors; embedded stores live in a
    sub-folder, server stores use CODENAV_VECTOR_URL. Falls back to the simple store (with a
    warning) when a backend's library is missing or its server cannot be reached."""
    backend = (backend or "simple").lower()
    url = getattr(settings, "vector_url", None) or None
    api_key = getattr(settings, "vector_api_key", None) or None
    name = collection or getattr(settings, "vector_collection", None)
    try:
        if backend == "chroma":
            return ChromaVectorStore(path / "chroma")
        if backend == "qdrant":
            from .vector_db import QdrantVectorStore
            return QdrantVectorStore(path / "qdrant", name or (default_collection(project) if url else "code"), url, api_key)
        if backend == "lancedb":
            from .vector_db import LanceVectorStore
            return LanceVectorStore(url or str(path / "lancedb"), name or (default_collection(project) if url else "code"),
                                    api_key)
        if backend == "pgvector":
            from .vector_db import PgVectorStore
            if not url:
                raise RuntimeError("pgvector needs CODENAV_VECTOR_URL=postgresql://user:password@host:5432/db")
            return PgVectorStore(url, name or default_collection(project))
        if backend != "simple":
            log.warning("Unknown vector backend '%s' (choose from %s) - using simple", backend, ", ".join(BACKENDS))
    except RuntimeError as exc:
        log.warning("%s - falling back to the simple vector store", exc)
    return SimpleVectorStore(path / "simple")
