"""Configuration.

Every setting is an environment variable prefixed with ``CODENAV_`` so it never clashes
with the variables of the project being indexed. Values are read, highest priority first, from:

1. the process environment
2. ``<repo>/.code-navigator.env``           (per-repository settings)
3. ``~/.config/code-navigator/config.env``  (per-user settings, e.g. your API key)

With no configuration at all, everything runs offline: no LLM, hash embeddings and a numpy
vector store. Add an LLM and an embedding model to get better summaries and answers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

PREFIX = "CODENAV_"
INDEX_DIR_NAME = ".code-navigator"
REPO_ENV_FILE = ".code-navigator.env"
USER_ENV_FILE = Path.home() / ".config" / "code-navigator" / "config.env"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "code-navigator"

LLM_DEFAULTS = {
    "none": {"model": "none", "base_url": ""},
    "ollama": {"model": "qwen2.5-coder:7b", "base_url": "http://localhost:11434"},
    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "anthropic": {"model": "claude-sonnet-5-5", "base_url": "https://api.anthropic.com"},
}

EMBED_DEFAULTS = {
    "hash": {"model": "hash-512", "base_url": ""},
    "ollama": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"},
    "openai": {"model": "text-embedding-3-small", "base_url": "https://api.openai.com/v1"},
    "sentence-transformers": {"model": "sentence-transformers/all-MiniLM-L6-v2", "base_url": ""},
}


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (comments, blank lines, optional quotes, optional 'export')."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value
    return values


class _Source:
    def __init__(self, repo: Path | None):
        self.values: dict[str, str] = {}
        self.values.update(read_env_file(USER_ENV_FILE))
        if repo is not None:
            self.values.update(read_env_file(repo / REPO_ENV_FILE))
        self.values.update({k: v for k, v in os.environ.items() if k.startswith(PREFIX)})
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            if os.environ.get(key):
                self.values.setdefault(key, os.environ[key])

    def get(self, name: str, default: str | None = None) -> str | None:
        value = self.values.get(PREFIX + name)
        return value if value not in (None, "") else default

    def int(self, name: str, default: int) -> int:
        try:
            return int(self.get(name, str(default)))
        except ValueError:
            return default

    def float(self, name: str, default: float) -> float:
        try:
            return float(self.get(name, str(default)))
        except ValueError:
            return default

    def bool(self, name: str, default: bool) -> bool:
        return str(self.get(name, str(default))).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    cache_dir: Path

    llm_provider: str
    llm_model: str
    summary_llm_model: str
    llm_base_url: str
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: int

    embed_provider: str
    embed_model: str
    embed_base_url: str
    embed_api_key: str | None
    embed_batch_size: int

    vector_backend: str
    vector_url: str | None
    vector_api_key: str | None
    vector_collection: str | None

    summary_mode: str
    summary_workers: int

    retrieval_mode: str
    top_k: int
    nav_max_services: int
    nav_max_children: int
    nav_confidence_threshold: float
    max_context_chars: int
    call_expansion: int

    max_file_bytes: int
    chunk_max_lines: int
    include_tests: bool

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider not in ("none", "mock")

    @classmethod
    def load(cls, repo: str | Path | None = None) -> "Settings":
        src = _Source(Path(repo).expanduser().resolve() if repo else None)
        llm_provider = (src.get("LLM_PROVIDER", "none") or "none").lower()
        if llm_provider == "mock":
            llm_provider = "none"
        embed_provider = (src.get("EMBED_PROVIDER", "hash") or "hash").lower()
        llm_def = LLM_DEFAULTS.get(llm_provider, LLM_DEFAULTS["none"])
        emb_def = EMBED_DEFAULTS.get(embed_provider, EMBED_DEFAULTS["hash"])

        # Embeddings reuse the LLM endpoint/key when both use the same provider
        embed_base = src.get("EMBED_BASE_URL")
        if not embed_base and embed_provider == llm_provider:
            embed_base = src.get("LLM_BASE_URL")
        embed_key = src.get("EMBED_API_KEY")
        if not embed_key and embed_provider == llm_provider:
            embed_key = src.get("LLM_API_KEY")

        vendor_key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(llm_provider)
        llm_model = src.get("LLM_MODEL", llm_def["model"])
        llm_on = llm_provider != "none"

        return cls(
            cache_dir=Path(src.get("CACHE_DIR", str(CACHE_DIR))).expanduser(),
            llm_provider=llm_provider,
            llm_model=llm_model,
            # cheaper/faster model for the many index-time summaries
            summary_llm_model=src.get("SUMMARY_LLM_MODEL",
                                      "claude-haiku-5-5" if llm_provider == "anthropic" else llm_model),
            llm_base_url=(src.get("LLM_BASE_URL", llm_def["base_url"]) or "").rstrip("/"),
            llm_api_key=src.get("LLM_API_KEY") or (src.values.get(vendor_key) if vendor_key else None),
            llm_temperature=src.float("LLM_TEMPERATURE", 0.1),
            llm_max_tokens=src.int("LLM_MAX_TOKENS", 1500),
            llm_timeout=src.int("LLM_TIMEOUT", 240),
            embed_provider=embed_provider,
            embed_model=src.get("EMBED_MODEL", emb_def["model"]),
            embed_base_url=(embed_base or emb_def["base_url"]).rstrip("/"),
            embed_api_key=embed_key or (src.values.get("OPENAI_API_KEY") if embed_provider == "openai" else None),
            embed_batch_size=src.int("EMBED_BATCH_SIZE", 32),
            vector_backend=(src.get("VECTOR_BACKEND", "simple") or "simple").lower(),
            vector_url=src.get("VECTOR_URL"),
            vector_api_key=src.get("VECTOR_API_KEY"),
            vector_collection=src.get("VECTOR_COLLECTION"),
            summary_mode=(src.get("SUMMARY_MODE", "llm" if llm_on else "heuristic") or "heuristic").lower(),
            summary_workers=max(1, src.int("SUMMARY_WORKERS", 4)),
            retrieval_mode=(src.get("RETRIEVAL_MODE", "hybrid") or "hybrid").lower(),
            top_k=src.int("TOP_K", 8),
            nav_max_services=src.int("NAV_MAX_SERVICES", 3),
            nav_max_children=src.int("NAV_MAX_CHILDREN", 4),
            nav_confidence_threshold=src.float("NAV_CONFIDENCE_THRESHOLD", 0.6),
            max_context_chars=src.int("MAX_CONTEXT_CHARS", 24000),
            call_expansion=max(0, src.int("CALL_EXPANSION", 4)),
            max_file_bytes=src.int("MAX_FILE_BYTES", 300_000),
            chunk_max_lines=src.int("CHUNK_MAX_LINES", 120),
            include_tests=src.bool("INCLUDE_TESTS", True),
        )

    def with_overrides(self, **kwargs) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def get_settings(repo: str | Path | None = None) -> Settings:
    """Read settings fresh (cheap; keeps tests isolated)."""
    return Settings.load(repo)
