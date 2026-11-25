"""Turn code units into embedding-ready chunks (one per unit; large units are split)."""
from __future__ import annotations

from ..models import Chunk, CodeUnit

OVERLAP_LINES = 10


def _header(u: CodeUnit, part: str = "") -> str:
    lines = [
        f"Service: {u.service} | File: {u.file_path} | Language: {u.language}",
        f"{u.kind.capitalize()}: {u.name}{part}",
    ]
    if u.signature and u.signature != u.name:
        lines.append(f"Signature: {u.signature}")
    if u.docstring:
        lines.append(f"Description: {u.docstring[:400]}")
    if u.endpoints:
        eps = ", ".join(f"{e['method']} {e['path']}" for e in u.endpoints[:8])
        lines.append(f"Endpoints: {eps}")
    if u.calls:
        lines.append(f"Calls: {', '.join(u.calls[:20])}")
    return "\n".join(lines)


def _meta(u: CodeUnit, start: int, end: int, part: int, parts: int) -> dict:
    return {
        "unit_id": u.id,
        "service": u.service,
        "file_path": u.file_path,
        "language": u.language,
        "kind": u.kind,
        "name": u.name,
        "short_name": u.name.split(".")[-1].lower(),
        "start_line": int(start),
        "end_line": int(end),
        "part": part,
        "parts": parts,
        "endpoints": "; ".join(f"{e['method']} {e['path']}" for e in u.endpoints[:8]),
    }


def chunk_units(units: list[CodeUnit], max_lines: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for u in units:
        code_lines = u.code.splitlines() or [""]
        if len(code_lines) <= max_lines or u.kind in ("class", "config", "api_spec"):
            chunks.append(Chunk(
                id=u.id, unit_id=u.id,
                text=f"{_header(u)}\n\n{u.code}",
                metadata=_meta(u, u.start_line, u.end_line, 1, 1),
            ))
            continue
        step = max_lines - OVERLAP_LINES
        windows = list(range(0, len(code_lines), step))
        for i, offset in enumerate(windows, start=1):
            piece = code_lines[offset : offset + max_lines]
            if not piece:
                continue
            start = u.start_line + offset
            end = start + len(piece) - 1
            chunks.append(Chunk(
                id=f"{u.id}@part{i}", unit_id=u.id,
                text=f"{_header(u, f' (part {i}/{len(windows)})')}\n\n" + "\n".join(piece),
                metadata=_meta(u, start, end, i, len(windows)),
            ))
            if offset + max_lines >= len(code_lines):
                break
    return chunks
