"""Edge types, Kafka at function level, impact analysis, god nodes, graph.html, incremental updates."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from code_navigator.engine import Navigator, git_changed_lines
from code_navigator.features.callgraph import CallGraph, build_call_graph, class_bases
from code_navigator.indexing.pipeline import update_index
from code_navigator.indexing.watch import hook_status, install_hooks, snapshot, uninstall_hooks, watch
from code_navigator.mcp_server import Server
from code_navigator.models import CodeUnit, Service, SourceFile


def _edge(engine, source, target, kind=None):
    cg = engine.callgraph
    return next((e for e in cg.edges if cg.nodes[e["source"]]["name"] == source
                 and cg.nodes[e["target"]]["name"] == target and (kind is None or e["kind"] == kind)), None)


# ------------------------------------------------------------------ edge types

def _unit(name, kind, code, file="svc/a.py", lang="python", parent=None, start=1):
    return CodeUnit(id=f"{file}::{name}#{start}", service="svc", file_path=file, language=lang, kind=kind,
                    name=name, signature=code.splitlines()[0], code=code, start_line=start,
                    end_line=start + code.count("\n"), parent=parent)


@pytest.mark.parametrize("lang,code,expected", [
    ("python", "class Repo(Base, Mixin):\n    pass", [("Base", "inherits"), ("Mixin", "inherits")]),
    ("java", "public class A extends B implements C, D<E> {", [("B", "inherits"), ("C", "implements"), ("D", "implements")]),
    ("typescript", "export class A extends B implements I {", [("B", "inherits"), ("I", "implements")]),
    ("csharp", "public class Repo : BaseRepo, IRepo {", [("BaseRepo", "inherits"), ("IRepo", "implements")]),
    ("go", "type Handler struct {\n\tBase\n\t*Logger\n\tname string\n}", [("Base", "inherits"), ("Logger", "inherits")]),
])
def test_class_bases(lang, code, expected):
    assert class_bases(_unit("A", "class", code, lang=lang)) == expected


def test_inheritance_edges_and_inherited_method_resolution():
    base = _unit("Base", "class", "class Base:\n    def save(self):\n        pass", start=1)
    save = _unit("Base.save", "method", "    def save(self):\n        pass", parent="Base", start=2)
    child = _unit("Repo", "class", "class Repo(Base):\n    def put(self):\n        self.save()", start=5)
    put = _unit("Repo.put", "method", "    def put(self):\n        self.save()", parent="Repo", start=6)
    sf = SourceFile(path="svc/a.py", abs_path="", kind="code", language="python", size=1, sha1="x",
                    text="\n".join(["class Base:", "    def save(self):", "        pass", "", "class Repo(Base):",
                                    "    def put(self):", "        self.save()"]), service="svc")
    cg = CallGraph(build_call_graph([base, save, child, put], [Service(name="svc", path="svc")], [sf], []))
    kinds = {(cg.nodes[e["source"]]["name"], cg.nodes[e["target"]]["name"]): (e["kind"], e["resolution"]) for e in cg.edges}
    assert kinds[("Repo", "Base")] == ("inherits", "declaration")
    assert kinds[("Repo.put", "Base.save")] == ("calls", "inherited")
    assert cg.related(child.id)["extends"][0]["name"] == "Base"
    impact = cg.affected([save.id])
    assert {a["name"] for a in impact["affected"]} == {"Repo.put"}


def test_import_edges(engine):
    cg = engine.callgraph
    imports = [(e["source"], cg.nodes[e["target"]]["name"]) for e in cg.edges if e["kind"] == "imports"]
    assert ("file:order-service/app/service.py", "PaymentClient") in imports
    assert any(src == "file:payment-service/main.go" and t.endswith("handler.go") for src, t in imports)
    # imports never show up as calls
    assert all(n["kind_of_edge"] != "imports" for n in engine.calls("PaymentClient")["callers"])


# ------------------------------------------------------------------ kafka

def test_kafka_publisher_to_consumer_edges(engine):
    e = _edge(engine, "OrderEventPublisher.publish", "OrderEventListener.onOrderEvent", "async")
    assert e and e["detail"] == "topic order-events"
    assert _edge(engine, "Publisher.PaymentProcessed", "main", "async")        # Go topic set in the constructor
    consumers = {(e["service"], e["path"]) for e in engine.entry_points() if e["method"] == "CONSUME"}
    assert ("inventory-service", "order-events") in consumers and ("notification-service", "payment-events") in consumers


def test_call_path_over_kafka(engine):
    hops = engine.call_path("cancel_order", "InventoryService.release")["hops"]
    assert [h["kind"] for h in hops].count("async") == 1 and hops[-1]["to"]["name"] == "InventoryService.release"


# ------------------------------------------------------------------ impact + god nodes

def test_affected_by_symbol_crosses_services(engine):
    data = engine.affected(["IdempotencyStore.Put"])
    names = [a["name"] for a in data["affected"]]
    assert names[0] == "Handler.Charge" and "PaymentClient.charge" in names and "create_order" in names
    eps = {(e["service"], e["path"]) for e in data["entry_points"]}
    assert ("payment-service", "/charge") in eps and ("api-gateway", "/api/orders") in eps
    assert ("api-gateway", "/api/orders/:id") not in eps       # inline route that does not reach the change
    assert set(data["cross_service"]) == {"order-service", "api-gateway"}


def test_affected_follows_messages_downstream(engine):
    data = engine.affected(["OrderEventPublisher.publish"])
    by_name = {a["name"]: a for a in data["affected"]}
    assert by_name["OrderEventListener.onOrderEvent"]["edge"] == "async"
    assert ("inventory-service", "order-events") in {(e["service"], e["path"]) for e in data["entry_points"]}


def test_affected_by_file(engine):
    data = engine.affected(files=["payment-service/internal/payment/idempotency.go"])
    assert {c["name"] for c in data["changed"]} >= {"IdempotencyStore.Put", "IdempotencyStore.Get"}


def test_god_nodes(engine):
    rows = engine.god_nodes(top=5)
    assert len(rows) == 5 and rows == sorted(rows, key=lambda r: -r["score"])
    assert all(r["fan_in"] + r["fan_out"] > 0 and r["transitive_dependents"] >= 0 for r in rows)
    multi = [r for r in rows if len(r["caller_services"]) > 1]
    assert multi  # something is called from several services


# ------------------------------------------------------------------ graph.html

def test_graph_html_is_self_contained(sample_index, engine, tmp_path):
    html_path = Path(sample_index["index_path"]) / "graph.html"
    text = html_path.read_text()
    assert "<script src" not in text and "http://" not in text.split("<script>")[1].split("const D")[0]
    data = json.loads(text.split("const D = ", 1)[1].split(";\nconst PALETTE", 1)[0])
    assert {n["n"] for n in data["nodes"]} >= {"OrderService.place_order", "Handler.Charge"}
    assert {e["k"] for e in data["edges"]} >= {"calls", "http", "async", "imports"}
    out = engine.write_html(tmp_path / "g.html")
    assert out.exists() and "</script>" not in out.read_text().split("const D = ", 1)[1].split(";\nconst PALETTE")[0]


# ------------------------------------------------------------------ incremental

@pytest.fixture
def scratch_repo(tmp_path, offline_settings):
    src = Path(__file__).resolve().parent.parent / "examples" / "sample-shop"
    dest = tmp_path / "shop"
    shutil.copytree(src, dest)
    return dest


def test_update_reuses_embeddings_and_skips_when_unchanged(scratch_repo, offline_settings):
    first = update_index(str(scratch_repo), settings=offline_settings, progress=lambda *a: None)
    assert not first.get("up_to_date") and first["incremental"]["embeddings_computed"] > 0
    again = update_index(str(scratch_repo), settings=offline_settings, progress=lambda *a: None)
    assert again.get("up_to_date")
    f = scratch_repo / "payment-service/internal/payment/idempotency.go"
    f.write_text(f.read_text() + "\n// touched\n")
    third = update_index(str(scratch_repo), settings=offline_settings, progress=lambda *a: None)
    inc = third["incremental"]
    assert inc["changed"] == ["payment-service/internal/payment/idempotency.go"]
    assert inc["embeddings_computed"] <= 3 and inc["embeddings_reused"] > 100
    changed_settings = offline_settings.with_overrides(chunk_max_lines=60)
    assert not update_index(str(scratch_repo), settings=changed_settings, progress=lambda *a: None).get("up_to_date")


def test_watch_reports_settled_changes(scratch_repo):
    target = scratch_repo / "order-service/app/models.py"
    seen: list[list[str]] = []
    ticks = iter(range(100))

    def fake_sleep(_):
        if next(ticks) == 0:
            target.write_text(target.read_text() + "\n# edit\n")

    watch(scratch_repo, seen.append, interval=0, max_cycles=3, sleep=fake_sleep)
    assert seen == [["order-service/app/models.py"]]
    assert ".code-navigator/manifest.json" not in snapshot(scratch_repo)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_diff_impact_and_hooks(scratch_repo, offline_settings):
    _git(scratch_repo, "init", "-q")
    _git(scratch_repo, "add", "-A")
    _git(scratch_repo, "commit", "-qm", "init")
    update_index(str(scratch_repo), settings=offline_settings, progress=lambda *a: None)
    clients = scratch_repo / "order-service/app/clients.py"
    clients.write_text(clients.read_text().replace("if r.status_code == 409:", "if r.status_code in (409, 422):"))
    (scratch_repo / "order-service/app/new_module.py").write_text("def x():\n    return 1\n")
    changed = git_changed_lines(scratch_repo, "HEAD")
    assert changed["order-service/app/new_module.py"] is None and changed["order-service/app/clients.py"] == {38}
    data = Navigator(scratch_repo, settings=offline_settings).affected(git_ref="HEAD")
    assert [c["name"] for c in data["changed"]] == ["InventoryClient.reserve"]
    assert "OrderService.place_order" in {a["name"] for a in data["affected"]}

    hook = scratch_repo / ".git/hooks/post-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho existing\n")
    install_hooks(scratch_repo)
    install_hooks(scratch_repo)            # idempotent
    text = hook.read_text()
    assert text.count(">>> code-navigator >>>") == 1 and "echo existing" in text
    assert all(hook_status(scratch_repo).values())
    uninstall_hooks(scratch_repo)
    assert hook.read_text().strip() == "#!/bin/sh\necho existing"
    assert not (scratch_repo / ".git/hooks/post-merge").exists()


def test_mcp_new_tools(sample_index, sample_repo):
    server = Server(sample_repo)

    def call(name, **args):
        res = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": name, "arguments": args}})["result"]
        return json.loads(res["content"][0]["text"])

    assert call("affected", symbols=["Handler.Charge"])["affected"][0]["name"] == "PaymentClient.charge"
    assert len(call("god_nodes", top=3)) == 3


def test_chroma_backend_survives_incremental_updates(scratch_repo, offline_settings):
    pytest.importorskip("chromadb")
    chroma = offline_settings.with_overrides(vector_backend="chroma")
    first = update_index(str(scratch_repo), settings=chroma, progress=lambda *a: None)
    assert first["vector_backend"] == "chroma"
    f = scratch_repo / "payment-service/main.go"
    f.write_text(f.read_text() + "\n// touched\n")
    second = update_index(str(scratch_repo), settings=chroma, progress=lambda *a: None)  # used to fail: read-only DB
    assert second["incremental"]["embeddings_computed"] <= 3
    nav = Navigator(scratch_repo, settings=chroma)
    assert nav.store.count() == second["stats"]["chunks"]
    assert nav.search("ChargeWithRetry", top_k=3)["hits"]
    back = update_index(str(scratch_repo), settings=offline_settings, progress=lambda *a: None)
    assert back["vector_backend"] == "simple" and back["incremental"]["embeddings_computed"] == 0
    assert not (scratch_repo / ".code-navigator/vectors/chroma").exists()


# ------------------------------------------------------------------ vector databases

def _pg_dsn(tmp_path_factory):
    import os
    if os.environ.get("CODENAV_TEST_PG_DSN"):
        return os.environ["CODENAV_TEST_PG_DSN"]
    pgserver = pytest.importorskip("pgserver")
    return pgserver.get_server(str(tmp_path_factory.mktemp("pg"))).get_uri()


@pytest.mark.parametrize("backend,module", [("qdrant", "qdrant_client"), ("lancedb", "lancedb"), ("pgvector", "psycopg")])
def test_vector_database_backends(backend, module, scratch_repo, offline_settings, tmp_path_factory):
    pytest.importorskip(module)
    url = _pg_dsn(tmp_path_factory) if backend == "pgvector" else None
    settings = offline_settings.with_overrides(vector_backend=backend, vector_url=url)
    first = update_index(str(scratch_repo), settings=settings, progress=lambda *a: None)
    assert first["vector_backend"] == backend and first["vector_store"]["backend"] == backend
    nav = Navigator(scratch_repo, settings=settings)
    try:
        assert nav.store.count() == first["stats"]["chunks"]
        vec = nav.embedder.embed_query("charge retry")
        assert {h.metadata["service"] for h in nav.store.query(vec, 5, services=["payment-service"])} == {"payment-service"}
        java = "inventory-service/src/main/java/com/shop/inventory/service/InventoryService.java"
        assert {h.metadata["file_path"] for h in nav.store.query(vec, 3, files=[java])} == {java}
        assert nav.search("What does ChargeWithRetry do?", top_k=3, expand=0)["hits"][0]["name"] == "ProviderGateway.ChargeWithRetry"
    finally:
        nav.close()
    f = scratch_repo / "payment-service/main.go"
    f.write_text(f.read_text() + "\n// touched\n")
    second = update_index(str(scratch_repo), settings=settings, progress=lambda *a: None)
    assert second["incremental"]["embeddings_computed"] <= 3 and second["incremental"]["embeddings_reused"] > 100
    # switching back reuses the vectors (for a server store its URL must still be configured)
    back = update_index(str(scratch_repo), settings=offline_settings.with_overrides(vector_url=url),
                        progress=lambda *a: None)
    assert back["vector_backend"] == "simple" and back["incremental"]["embeddings_computed"] == 0


def test_server_locations_are_redacted_and_named_per_project():
    from code_navigator.indexing.vector_index import default_collection, redact
    assert redact("postgresql://app:s3cr3t@db:5432/x") == "postgresql://app:***@db:5432/x"
    assert redact("http://localhost:6333") == "http://localhost:6333"
    assert default_collection("Sample Shop!") == "codenav_sample_shop"


def test_missing_library_falls_back_to_simple(tmp_path, offline_settings, monkeypatch):
    import builtins
    from code_navigator.indexing.vector_index import open_vector_store
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith(("qdrant_client", "lancedb")):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert open_vector_store("qdrant", tmp_path, offline_settings).backend == "simple"
    assert open_vector_store("lancedb", tmp_path, offline_settings).backend == "simple"
    assert open_vector_store("pgvector", tmp_path, offline_settings).backend == "simple"  # no URL configured
