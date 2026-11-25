"""Config, index layout, CLI and MCP server."""
from __future__ import annotations

import io
import json
from pathlib import Path

from code_navigator import config as cfg
from code_navigator.cli import main
from code_navigator.mcp_server import TOOLS, Server, serve


def test_defaults_are_offline(monkeypatch, tmp_path):
    for key in list(__import__("os").environ):
        if key.startswith(cfg.PREFIX):
            monkeypatch.delenv(key)
    monkeypatch.setattr(cfg, "USER_ENV_FILE", tmp_path / "missing.env")
    s = cfg.Settings.load(tmp_path)
    assert (s.llm_provider, s.embed_provider, s.vector_backend, s.summary_mode) == ("none", "hash", "simple", "heuristic")
    assert not s.llm_enabled


def test_repo_env_file_and_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "USER_ENV_FILE", tmp_path / "user.env")
    (tmp_path / "user.env").write_text("CODENAV_TOP_K=3\nCODENAV_LLM_PROVIDER=openai\n")
    (tmp_path / cfg.REPO_ENV_FILE).write_text('# comment\nexport CODENAV_TOP_K="5"\nLLM_PROVIDER=ignored\n')
    monkeypatch.setenv("CODENAV_EMBED_PROVIDER", "ollama")
    s = cfg.Settings.load(tmp_path)
    assert s.top_k == 5                      # repo file beats user file
    assert s.llm_provider == "openai"        # unprefixed LLM_PROVIDER from the project is ignored
    assert s.embed_provider == "ollama"      # environment beats both files
    assert s.summary_mode == "llm"


def test_index_lives_inside_repo(sample_index, sample_repo):
    root = Path(sample_index["index_path"])
    assert root == sample_repo / ".code-navigator"
    for name in ("manifest.json", "services.json", "dependencies.json", "chunks.jsonl", "tree.json", ".gitignore"):
        assert (root / name).exists(), name
    assert sample_index["summary_mode"] == "heuristic" and sample_index["llm"] is None


def test_cli_search_json(sample_index, sample_repo, capsys):
    main(["search", "What does ChargeWithRetry do?", "-C", str(sample_repo), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert any("ChargeWithRetry" in h["name"] for h in data["hits"])


def test_cli_impact_unknown_service(sample_index, sample_repo, capsys):
    try:
        main(["impact", "nope-service", "-C", str(sample_repo)])
    except SystemExit as exc:
        assert exc.code == 2
    assert "unknown service" in capsys.readouterr().err


def _rpc(server: Server, method: str, params: dict | None = None, mid: int = 1) -> dict:
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_mcp_handshake_and_tools(sample_index, sample_repo):
    server = Server(sample_repo)
    init = _rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    assert init["result"]["serverInfo"]["name"] == "code-navigator"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in _rpc(server, "tools/list")["result"]["tools"]}
    assert names == {t["name"] for t in TOOLS}


def _call(server: Server, tool: str, **args):
    res = _rpc(server, "tools/call", {"name": tool, "arguments": args})["result"]
    return res, (json.loads(res["content"][0]["text"]) if not res["isError"] else res["content"][0]["text"])


def test_mcp_tool_calls(sample_index, sample_repo):
    server = Server(sample_repo)
    _, overview = _call(server, "overview")
    assert {s["name"] for s in overview["services"]} >= {"order-service", "payment-service"}

    _, node = _call(server, "browse_tree", node_id="svc:payment-service")
    assert node["kind"] == "service" and node["children"]

    _, hits = _call(server, "search_code", query="ChargeWithRetry", top_k=3)
    assert hits["hits"][0]["code"]
    _, chunk = _call(server, "get_chunk", chunk_id=hits["hits"][0]["chunk_id"])
    assert chunk["text"]

    _, impact = _call(server, "impact", service="payment-service")
    assert impact["direct_dependents"] == ["order-service"]

    res, err = _call(server, "get_service", name="nope")
    assert res["isError"] and "unknown service" in err


def test_mcp_stdio_loop(sample_index, sample_repo):
    stdin = io.StringIO("\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]) + "\n")
    stdout = io.StringIO()
    serve(sample_repo, stdin=stdin, stdout=stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert replies[1]["error"]["code"] == -32700
