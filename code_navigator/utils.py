"""Small helpers used across modules."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("code_navigator")

_STOPWORDS = set(
    """a an the is are was were be been of to in on for and or not with by from at as it this that
    these those what which who whom how why where when does do did can could should would will
    i we you they he she my our your their me us them there here into about over under than then
    if else so such any all each some more most other only own same very just also get gets got
    service services code codebase file files function functions class method methods work works
    happen happens explain show tell find""".split()
)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def norm_name(name: str) -> str:
    """Normalise service names for fuzzy matching: 'Order-Service' -> 'orderservice'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def core_name(name: str) -> str:
    """'order-service' -> 'order', 'cartservice' -> 'cart', 'payments-api' -> 'payments'."""
    n = norm_name(name)
    for suffix in ("microservice", "service", "svc", "api", "server", "app"):
        if n.endswith(suffix) and len(n) > len(suffix) + 1:
            return n[: -len(suffix)]
    return n


def names_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    na, nb = norm_name(a), norm_name(b)
    if na == nb:
        return True
    ca, cb = core_name(a), core_name(b)
    return len(ca) >= 3 and ca == cb


def split_identifier(token: str) -> list[str]:
    """camelCase / snake_case / kebab-case -> lowercase words."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", token)
    return [p.lower() for p in parts if p]


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", text):
        for w in split_identifier(raw):
            if len(w) < 2:
                continue
            if drop_stopwords and w in _STOPWORDS:
                continue
            words.append(w)
    return words


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n... [truncated]"


def first_sentence(text: str, max_chars: int = 200) -> str:
    text = " ".join(text.strip().split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return truncate(s, max_chars)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_json(text: str, default: Any = None) -> Any:
    """Pull the first JSON object/array out of an LLM response."""
    if not text:
        return default
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for candidate in (cleaned, text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default


def safe_project_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")
    return name or "project"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stderr,  # stdout stays clean for --json output and the MCP protocol
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
