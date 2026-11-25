"""Embedding providers for the vector index."""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import requests

from ..config import Settings
from ..utils import log, tokenize

MAX_EMBED_CHARS = 6000


class Embedder(ABC):
    provider = "base"

    def __init__(self, model: str, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]: ...

    # Some models (nomic-embed-text, e5, bge) expect task prefixes
    def _doc_prefix(self) -> str:
        return "search_document: " if "nomic" in self.model else ""

    def _query_prefix(self) -> str:
        return "search_query: " if "nomic" in self.model else ""

    def embed_documents(self, texts: list[str], progress=None) -> list[list[float]]:
        out: list[list[float]] = []
        prefix = self._doc_prefix()
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + t[:MAX_EMBED_CHARS] for t in texts[i : i + self.batch_size]]
            out.extend(self._embed(batch))
            if progress:
                progress(min(i + self.batch_size, len(texts)), len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix() + text[:MAX_EMBED_CHARS]])[0]

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


class OllamaEmbedder(Embedder):
    provider = "ollama"

    def __init__(self, model: str, base_url: str, batch_size: int = 32, timeout: int = 300):
        super().__init__(model, batch_size)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, texts):
        resp = requests.post(f"{self.base_url}/api/embed",
                             json={"model": self.model, "input": texts, "truncate": True},
                             timeout=self.timeout)
        if resp.status_code == 404:  # older Ollama versions: one text per call
            vectors = []
            for t in texts:
                r = requests.post(f"{self.base_url}/api/embeddings",
                                  json={"model": self.model, "prompt": t}, timeout=self.timeout)
                r.raise_for_status()
                vectors.append(r.json()["embedding"])
            return vectors
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama embed failed ({resp.status_code}): {resp.text[:300]}. "
                               f"Did you run `ollama pull {self.model}`?")
        return resp.json()["embeddings"]


class OpenAIEmbedder(Embedder):
    provider = "openai"

    def __init__(self, model: str, base_url: str, api_key: str | None, batch_size: int = 64, timeout: int = 120):
        super().__init__(model, batch_size)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _embed(self, texts):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(f"{self.base_url}/embeddings", headers=headers,
                             json={"model": self.model, "input": texts}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Embedding API failed ({resp.status_code}): {resp.text[:300]}")
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class SentenceTransformerEmbedder(Embedder):
    provider = "sentence-transformers"

    def __init__(self, model: str, batch_size: int = 32):
        super().__init__(model, batch_size)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install sentence-transformers to use this embedder") from exc
        self._model = SentenceTransformer(model)

    def _embed(self, texts):
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class HashEmbedder(Embedder):
    """Dependency-free 'feature hashing' embedder. Works offline and makes tests
    deterministic; quality is closer to keyword search than semantic search."""
    provider = "hash"

    def __init__(self, dim: int = 512):
        super().__init__(f"hash-{dim}", batch_size=256)
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text, drop_stopwords=True)
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _embed(self, texts):
        return [self._vector(t) for t in texts]


def get_embedder(settings: Settings, provider: str | None = None, model: str | None = None) -> Embedder:
    provider = (provider or settings.embed_provider).lower()
    model = model or settings.embed_model
    if provider == "ollama":
        return OllamaEmbedder(model, settings.embed_base_url or "http://localhost:11434",
                              settings.embed_batch_size)
    if provider == "openai":
        return OpenAIEmbedder(model, settings.embed_base_url or "https://api.openai.com/v1",
                              settings.embed_api_key, settings.embed_batch_size)
    if provider in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(model, settings.embed_batch_size)
    if provider == "hash":
        return HashEmbedder()
    log.warning("Unknown EMBED_PROVIDER '%s', using hash embedder", provider)
    return HashEmbedder()
