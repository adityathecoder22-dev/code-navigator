"""Core data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceFile:
    """A file discovered in the repository."""
    path: str                 # path relative to repo root (posix style)
    abs_path: str
    kind: str                 # code | config | doc
    language: str             # python, java, go ... or config subtype (compose, k8s, openapi, proto ...)
    size: int
    sha1: str
    text: str = ""
    service: str = "_shared"  # filled in by the service detector

    def to_dict(self, with_text: bool = False) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text", None)
        return d


@dataclass
class CodeUnit:
    """A meaningful piece of code: function, method, class, module header, config block, doc section."""
    id: str
    service: str
    file_path: str
    language: str
    kind: str                 # function | method | class | module | config | api_spec | doc
    name: str
    signature: str
    code: str
    start_line: int
    end_line: int
    parent: str | None = None
    docstring: str = ""
    calls: list[str] = field(default_factory=list)
    endpoints: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Service:
    """A microservice (or infrastructure component such as Kafka/Postgres)."""
    name: str
    path: str                 # directory relative to repo root ("" if unknown)
    kind: str = "app"         # app | infra
    language: str = "unknown"
    detected_by: list[str] = field(default_factory=list)
    image: str | None = None
    ports: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    endpoints: list[dict] = field(default_factory=list)
    grpc_services: list[str] = field(default_factory=list)
    topics_published: list[str] = field(default_factory=list)
    topics_consumed: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Service":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Dependency:
    """A directed edge: `source` service depends on / talks to `target`."""
    source: str
    target: str
    kind: str                 # http | grpc | async | depends_on | config | database
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """The unit stored in the vector index."""
    id: str
    unit_id: str
    text: str                 # text that is embedded (header + code)
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    """A retrieved chunk with its score and the retrievers that found it."""
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]
    found_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
