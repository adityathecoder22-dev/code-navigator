"""Call graph: resolution inside services, HTTP links across services, queries and expansion."""
from __future__ import annotations

import json

from code_navigator.cli import main
from code_navigator.features.callgraph import call_sites, field_types, file_imports
from code_navigator.mcp_server import Server
from code_navigator.models import CodeUnit


def _edge(engine, source: str, target: str) -> dict | None:
    cg = engine.callgraph
    for e in cg.edges:
        if cg.nodes[e["source"]]["name"] == source and cg.nodes[e["target"]]["name"] == target:
            return e
    return None


def test_index_writes_call_graph(sample_index, engine):
    assert sample_index["stats"]["call_edges"] > 20
    assert sample_index["stats"]["cross_service_calls"] >= 4
    assert engine.callgraph.stats["entry_points"] >= 10


def test_python_field_and_self_resolution(engine):
    e = _edge(engine, "OrderService.place_order", "PaymentClient.charge")
    assert e and e["resolution"] == "field_type" and e["line"] == 39
    assert _edge(engine, "OrderService.place_order", "OrderService._release")["resolution"] == "self"
    assert _edge(engine, "OrderService.__init__", "PaymentClient")["kind"] == "instantiates"


def test_module_level_variable_and_attribute_chain(engine):
    # main.py: service = OrderService(); create_order -> service.place_order(); get_order -> service.repo.get()
    assert _edge(engine, "create_order", "OrderService.place_order")["resolution"] == "local_type"
    assert _edge(engine, "get_order", "OrderRepository.get")["resolution"] == "field_type"


def test_go_receiver_struct_fields_and_packages(engine):
    assert _edge(engine, "Handler.Charge", "ProviderGateway.ChargeWithRetry")["resolution"] == "field_type"
    assert _edge(engine, "ProviderGateway.ChargeWithRetry", "ProviderGateway.isOpen")["resolution"] == "self"
    assert _edge(engine, "main", "NewHandler")["resolution"] == "import"


def test_java_injected_field(engine):
    assert _edge(engine, "ProductController.reserve", "InventoryService.reserve")["resolution"] == "field_type"
    assert _edge(engine, "ProductController.reserve", "ProductController.reserve") is None  # signature is not a call


def test_entry_points_map_endpoints_to_handlers(engine):
    eps = {(e["service"], e["method"], e["path"]): e["handler_name"] for e in engine.entry_points()}
    assert eps[("payment-service", "POST", "/charge")] == "Handler.Charge"
    assert eps[("order-service", "POST", "/orders")] == "create_order"
    assert eps[("inventory-service", "POST", "/products/{id}/reserve")] == "ProductController.reserve"


def test_cross_service_http_edges(engine):
    e = _edge(engine, "PaymentClient.charge", "Handler.Charge")
    assert e and e["kind"] == "http" and e["resolution"] == "endpoint_path"
    assert _edge(engine, "InventoryClient.reserve", "ProductController.reserve")["kind"] == "http"


def test_call_path_crosses_services(engine):
    data = engine.call_path("create_order", "ChargeWithRetry")
    names = [h["to"]["name"] for h in data["hops"]]
    assert data["found"] and names[-1] == "ProviderGateway.ChargeWithRetry"
    assert "Handler.Charge" in names and any(h["kind"] == "http" for h in data["hops"])


def test_calls_query_and_depth(engine):
    data = engine.calls("place_order", depth=2)
    callees = {n["name"]: n for n in data["callees"]}
    assert callees["PaymentClient.charge"]["depth"] == 1
    assert callees["Handler.Charge"]["depth"] == 2
    assert {n["name"] for n in data["callers"]} >= {"create_order"}


def test_search_is_expanded_with_called_functions(engine):
    res = engine.search("What happens when a user places an order?", top_k=5)
    added = [h for h in res["hits"] if h["found_by"] == ["call_graph"]]
    assert added and all(h["expanded_from"] for h in added)
    plain = engine.search("What happens when a user places an order?", top_k=5, expand=0)
    assert not any(h["found_by"] == ["call_graph"] for h in plain["hits"])


def test_tree_symbol_shows_calls(engine):
    uid = engine.callgraph.find("OrderService.place_order")[0]
    node = engine.tree_view(f"sym:{uid}")
    assert "PaymentClient.charge" in node["calls"] and "create_order" in node["called_by"]


def test_cli_and_mcp(sample_index, sample_repo, capsys):
    main(["path", "create_order", "ChargeWithRetry", "-C", str(sample_repo), "--json"])
    assert json.loads(capsys.readouterr().out)["found"]
    server = Server(sample_repo)
    res = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "calls", "arguments": {"symbol": "Handler.Charge"}}})["result"]
    data = json.loads(res["content"][0]["text"])
    assert data["entry_points"][0]["path"] == "/charge"
    assert any(n["name"] == "PaymentClient.charge" for n in data["callers"])


def test_call_site_extraction_edge_cases():
    unit = CodeUnit(id="x", service="s", file_path="a.java", language="java", kind="method", name="A.run",
                    signature="public ResponseEntity<Map<String, Object>> run(String id) {",
                    code="public ResponseEntity<Map<String, Object>> run(String id) {\n"
                         "    // helper(ignored)\n"
                         "    Product p = repo.find(id);\n"
                         "    throw new MissingException(id);\n"
                         "}", start_line=10, end_line=14)
    sites = call_sites(unit)
    assert ("repo", "find", 12) in sites and ("new", "MissingException", 13) in sites
    assert not any(name in ("run", "helper") for _, name, _ in sites)
    assert field_types("type H struct {\n\tgw *ProviderGateway\n}", "go", {"ProviderGateway"}) == {"gw": "ProviderGateway"}
    assert file_imports("from .clients import A, B as C\nimport os.path as p", "python") == {"A": "clients", "C": "clients", "p": "path"}
