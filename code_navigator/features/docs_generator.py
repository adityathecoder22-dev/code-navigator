"""Generate Markdown documentation for a service from the index."""
from __future__ import annotations

from .. import prompts
from ..indexing.tree_index import TreeIndex
from ..llm.providers import LLM, LLMError
from ..models import Service
from ..utils import truncate
from .graph import impact_analysis


def _facts(service: Service, tree: TreeIndex, edges) -> str:
    impact = impact_analysis(service.name, [service], edges)
    node = tree.service_node(service.name)
    files = tree.files_under(node["id"]) if node else []
    lines = [
        f"Name: {service.name}",
        f"Language: {service.language}",
        f"Location: {service.path or '(repository root)'}",
        f"Summary: {service.summary or (node or {}).get('summary', '')}",
        f"Image: {service.image or 'n/a'}; Ports: {', '.join(service.ports) or 'n/a'}",
        "Endpoints:",
        *[f"  - {e.get('method')} {e.get('path')} ({e.get('file')}:{e.get('line')})" for e in service.endpoints[:40]],
        f"gRPC services: {', '.join(service.grpc_services) or 'none'}",
        f"Publishes topics: {', '.join(service.topics_published) or 'none'}",
        f"Consumes topics: {', '.join(service.topics_consumed) or 'none'}",
        "Depends on:",
        *[f"  - {d['service']} ({d['kind']}): {'; '.join(d['evidence'][:2])}" for d in impact["depends_on"]],
        "Called by:",
        *[f"  - {d['service']} ({d['kind']}): {'; '.join(d['evidence'][:2])}" for d in impact["called_by"]],
        "Environment variables: " + ", ".join(sorted(service.env)[:30]),
        "Files:",
        *[f"  - {f['path']}: {truncate(f['summary'], 200)}" for f in files[:40]],
    ]
    return "\n".join(lines)


def generate_service_doc(service: Service, tree: TreeIndex, edges, llm: LLM) -> str:
    facts = _facts(service, tree, edges)
    if llm.provider != "mock":
        try:
            return llm.complete(prompts.SERVICE_DOC.format(service=service.name, facts=facts), max_tokens=2000)
        except LLMError:
            pass
    # deterministic fallback
    return f"# {service.name}\n\n> Generated without an LLM (facts extracted from code and configuration).\n\n```\n{facts}\n```\n"
