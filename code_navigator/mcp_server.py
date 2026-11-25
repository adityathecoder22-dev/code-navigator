"""Model Context Protocol server (stdio, JSON-RPC 2.0, one message per line). No extra dependencies.

The AI assistant that connects (Claude Code, Cursor, ...) does the reasoning; this server only
serves the index. So the tool itself needs no LLM or API key.

    claude mcp add code-navigator -- code-navigator mcp --repo /path/to/repo
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .storage import IndexPaths
from .utils import load_json, log

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = """code-navigator serves a pre-built index of a microservice codebase.
Suggested workflow:
1. `overview` - system summary, services and architecture findings.
2. Narrow down like an engineer browsing the repo: `browse_tree` from "root" into services,
   modules, files and symbols (each node has a summary), or `search_code` for hybrid search.
3. `get_chunk` to read the full code of a hit; `dependencies` / `impact` for cross-service questions.
4. Follow the code: `calls` (callers and callees of a function), `call_path` (chain from A to B,
   also across services over HTTP and Kafka) and `entry_points` (which function handles each
   endpoint or topic).
5. Before changing code: `affected` (blast radius of a function, file or git diff) and
   `god_nodes` (the most connected, riskiest functions).
Cite files as path:line from the results."""


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


_SERVICE = {"type": "string", "description": "service name, as listed by list_services"}

TOOLS = [
    {"name": "overview",
     "description": "System overview: summary, index stats, every service with its summary, and architecture "
                    "findings (cycles, shared databases, high fan-in/out). Start here.",
     "inputSchema": _schema()},
    {"name": "list_services",
     "description": "All services with language, endpoints, Kafka/RabbitMQ topics and direct dependencies.",
     "inputSchema": _schema()},
    {"name": "get_service",
     "description": "Full details of one service: endpoints (file:line), env vars (secrets masked), topics, "
                    "gRPC services, files with summaries, depends_on and used_by.",
     "inputSchema": _schema({"name": _SERVICE}, ["name"])},
    {"name": "browse_tree",
     "description": "Vectorless navigation. Returns one node of the hierarchical index (system > service > "
                    "module > file > symbol) with its summary and its children's summaries. Start at 'root' and "
                    "follow child ids. Symbol nodes include chunk_ids for get_chunk.",
     "inputSchema": _schema({"node_id": {"type": "string", "default": "root"}})},
    {"name": "search_code",
     "description": "Hybrid code search (tree navigation + vector search + exact symbol match, fused with RRF). "
                    "Returns ranked functions/classes with service, file:line and code.",
     "inputSchema": _schema({
         "query": {"type": "string", "description": "natural-language question or identifier"},
         "mode": {"type": "string", "enum": ["hybrid", "vector", "vectorless"], "default": "hybrid"},
         "top_k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
         "include_code": {"type": "boolean", "default": True},
     }, ["query"])},
    {"name": "get_chunk",
     "description": "Full text of one indexed chunk (metadata header + code) by chunk_id.",
     "inputSchema": _schema({"chunk_id": {"type": "string"}}, ["chunk_id"])},
    {"name": "dependencies",
     "description": "Service dependency edges (http, grpc, async topic, database, ...) with file:line evidence. "
                    "Optionally only edges touching one service.",
     "inputSchema": _schema({"service": _SERVICE})},
    {"name": "impact",
     "description": "Impact analysis: services that call this one directly and transitively (what breaks if it "
                    "changes or goes down), and what it depends on.",
     "inputSchema": _schema({"service": _SERVICE}, ["service"])},
    {"name": "calls",
     "description": "Call graph around one function or method: what it calls and what calls it, with file:line, "
                    "edge kind (calls / instantiates / http to another service) and confidence.",
     "inputSchema": _schema({
         "symbol": {"type": "string", "description": "e.g. place_order, OrderService.place_order, Handler.Charge"},
         "service": _SERVICE,
         "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
     }, ["symbol"])},
    {"name": "call_path",
     "description": "Shortest call chain from one function to another, following calls inside services and HTTP "
                    "calls between them (e.g. api-gateway route -> order-service -> payment-service handler).",
     "inputSchema": _schema({"source": {"type": "string"}, "target": {"type": "string"}, "service": _SERVICE},
                            ["source", "target"])},
    {"name": "entry_points",
     "description": "HTTP endpoints of each service mapped to the function that handles them - where a request "
                    "enters the code.",
     "inputSchema": _schema({"service": _SERVICE})},
    {"name": "affected",
     "description": "Impact of a change: given changed functions/classes, changed files, or a git ref (uses the "
                    "uncommitted diff against it), returns everything that may break - callers transitively, across "
                    "services over HTTP, subclasses, consumers of published messages - plus the endpoints/topics "
                    "reached and the tests among them.",
     "inputSchema": _schema({
         "symbols": {"type": "array", "items": {"type": "string"}},
         "files": {"type": "array", "items": {"type": "string"}},
         "git_ref": {"type": "string", "description": "e.g. HEAD or main"},
         "depth": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
     })},
    {"name": "god_nodes",
     "description": "The most connected functions/classes (many callers, many callees, callers in several services) "
                    "- the riskiest code to change.",
     "inputSchema": _schema({"top": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}, "service": _SERVICE})},
    {"name": "ask",
     "description": "End-to-end answer generated by code-navigator's own configured LLM, with sources. If no LLM "
                    "is configured it returns the ranked sources only - prefer search_code in that case.",
     "inputSchema": _schema({"question": {"type": "string"}}, ["question"])},
]


class Server:
    def __init__(self, repo: str | Path = ".", index_dir: str | Path | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.index_dir = index_dir
        self._nav = None
        self._indexed_at = None

    def nav(self):
        """Open the index lazily and reopen it after a re-index."""
        from .engine import Navigator
        manifest = load_json(IndexPaths.for_repo(self.repo, self.index_dir).manifest) or {}
        if self._nav is None or manifest.get("indexed_at") != self._indexed_at:
            if self._nav is not None:
                self._nav.close()
            self._nav = Navigator(self.repo, index_dir=self.index_dir)
            self._indexed_at = self._nav.manifest.get("indexed_at")
        return self._nav

    # -------------------------------------------------------------- tools
    def call_tool(self, name: str, args: dict):
        nav = self.nav()
        if name == "overview":
            return nav.overview()
        if name == "list_services":
            out = []
            for s in nav.services:
                if s.kind == "shared":
                    continue
                info = nav.service_info(s.name)
                out.append({k: info[k] for k in ("name", "kind", "language", "path", "summary", "depends_on", "used_by")}
                           | {"endpoints": [f"{e.get('method', '')} {e.get('path', '')}".strip() for e in s.endpoints],
                              "topics_published": s.topics_published, "topics_consumed": s.topics_consumed})
            return out
        if name == "get_service":
            return nav.service_info(args["name"])
        if name == "browse_tree":
            return nav.tree_view(args.get("node_id") or "root")
        if name == "search_code":
            return nav.search(args["query"], mode=args.get("mode"), top_k=args.get("top_k"),
                              include_code=args.get("include_code", True))
        if name == "get_chunk":
            return nav.get_chunk(args["chunk_id"])
        if name == "dependencies":
            return nav.dependencies(args.get("service"))
        if name == "impact":
            data = nav.impact(args["service"])
            data.pop("dot", None)
            return data
        if name == "calls":
            return nav.calls(args["symbol"], service=args.get("service"), depth=int(args.get("depth") or 1))
        if name == "call_path":
            return nav.call_path(args["source"], args["target"], service=args.get("service"))
        if name == "entry_points":
            return nav.entry_points(args.get("service"))
        if name == "affected":
            if not (args.get("symbols") or args.get("files") or args.get("git_ref")):
                raise ValueError("give symbols, files or git_ref")
            return nav.affected(args.get("symbols"), args.get("files"), args.get("git_ref"),
                                depth=int(args.get("depth") or 8))
        if name == "god_nodes":
            return nav.god_nodes(top=int(args.get("top") or 10), service=args.get("service"))
        if name == "ask":
            res = nav.ask(args["question"])
            for s in res.get("sources", []):
                s.pop("code", None)
            return res
        raise KeyError(f"unknown tool '{name}'")

    # -------------------------------------------------------------- protocol
    def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification (e.g. notifications/initialized) - no reply
            return None
        try:
            if method == "initialize":
                requested = (msg.get("params") or {}).get("protocolVersion")
                return self._ok(mid, {
                    "protocolVersion": requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "code-navigator", "version": __version__},
                    "instructions": INSTRUCTIONS,
                })
            if method == "ping":
                return self._ok(mid, {})
            if method == "tools/list":
                return self._ok(mid, {"tools": TOOLS})
            if method == "tools/call":
                params = msg.get("params") or {}
                try:
                    data = self.call_tool(params.get("name", ""), params.get("arguments") or {})
                    text, is_error = json.dumps(data, ensure_ascii=False, default=str), False
                except Exception as exc:  # tool errors are reported to the model, not as protocol errors
                    detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
                    text, is_error = f"Error: {detail}", True
                return self._ok(mid, {"content": [{"type": "text", "text": text}], "isError": is_error})
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        except Exception as exc:  # pragma: no cover - defensive
            log.error("MCP request failed: %s", traceback.format_exc())
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}

    @staticmethod
    def _ok(mid, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(repo: str | Path = ".", index_dir: str | Path | None = None, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = Server(repo, index_dir)
    log.info("code-navigator MCP server for %s", server.repo)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = server.handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
