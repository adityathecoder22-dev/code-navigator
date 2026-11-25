"""Build the LLM context from retrieval results and generate a cited answer."""
from __future__ import annotations

import time

from .. import prompts
from ..llm.providers import LLM, LLMError
from ..retrieval.hybrid_retriever import RetrievalResult
from ..utils import truncate


def _strip_header(text: str) -> str:
    """Chunk text = metadata header + blank line + code; keep the code only."""
    return text.split("\n\n", 1)[1] if "\n\n" in text else text


def build_context(result: RetrievalResult, max_chars: int = 24000) -> tuple[str, list[dict]]:
    parts: list[str] = []
    if result.overview:
        parts.append(f"## System overview\n{result.overview}")
    if result.services:
        lines = [f"- {s['name']} ({s['meta'].get('language', '')}): {truncate(s['summary'], 500)}" for s in result.services]
        parts.append("## Services involved\n" + "\n".join(lines))
    if result.edges:
        lines = [f"- {e.source} -> {e.target} [{e.kind}] ({'; '.join(e.evidence[:2])})" for e in result.edges[:30]]
        parts.append("## Service dependencies\n" + "\n".join(lines))

    used = sum(len(p) for p in parts)
    sources: list[dict] = []
    snippets: list[str] = []
    for hit in result.hits:
        m = hit.metadata
        code = _strip_header(hit.text)
        budget = max_chars - used
        if budget < 400:
            break
        code = truncate(code, min(budget - 200, 6000))
        n = len(sources) + 1
        loc = f"{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}"
        snippet = (f"[{n}] service={m.get('service')} file={loc} {m.get('kind')} {m.get('name')}"
                   + (f" endpoints: {m['endpoints']}" if m.get("endpoints") else "")
                   + f"\n```{m.get('language', '')}\n{code}\n```")
        snippets.append(snippet)
        used += len(snippet)
        sources.append({
            "n": n, "service": m.get("service"), "file": m.get("file_path"), "start_line": m.get("start_line"),
            "end_line": m.get("end_line"), "name": m.get("name"), "kind": m.get("kind"),
            "score": hit.score, "found_by": hit.found_by, "code": code, "language": m.get("language"),
        })
    if snippets:
        parts.append("## Code snippets\n" + "\n\n".join(snippets))
    return "\n\n".join(parts), sources


class AnswerGenerator:
    def __init__(self, llm: LLM, max_context_chars: int = 24000):
        self.llm = llm
        self.max_context_chars = max_context_chars

    def answer(self, result: RetrievalResult) -> dict:
        context, sources = build_context(result, self.max_context_chars)
        t = time.time()
        if not sources and not result.services:
            answer = "I could not find relevant code for this question in the index."
        elif self.llm.provider == "mock":
            names = ", ".join(f"[{s['n']}] {s['file']} ({s['name']})" for s in sources)
            answer = ("No LLM configured (CODENAV_LLM_PROVIDER=none). The most relevant code found: " + names
                      if names else "No LLM configured (CODENAV_LLM_PROVIDER=none).")
        else:
            try:
                answer = self.llm.complete(
                    prompts.ANSWER_USER.format(context=context, question=result.question),
                    system=prompts.ANSWER_SYSTEM,
                )
            except LLMError as exc:
                answer = f"The LLM could not be reached ({exc}). Retrieved sources are listed below."
        timings = {**result.timings, "generation": round(time.time() - t, 2)}
        nav = result.navigation
        return {
            "question": result.question,
            "answer": answer,
            "mode": result.mode,
            "sources": sources,
            "services": [s["name"] for s in result.services],
            "dependencies": [e.to_dict() for e in result.edges],
            "navigation": nav.to_dict() if nav else None,
            "notes": result.notes,
            "timings": timings,
        }
