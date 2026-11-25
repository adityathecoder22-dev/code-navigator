from code_navigator.generation.answer_generator import AnswerGenerator


def test_ask_returns_answer_with_sources(engine):
    res = engine.ask("What happens when a user places an order?")
    assert res["answer"]
    assert res["sources"] and all({"file", "start_line", "code"} <= s.keys() for s in res["sources"])
    assert res["navigation"]["trace"]
    assert "order-service" in res["services"]


def test_answer_generator_with_llm(engine, scripted_llm):
    result = engine.retriever.retrieve("How are payments charged?", mode="hybrid")
    out = AnswerGenerator(scripted_llm([]), 8000).answer(result)
    assert out["answer"] == "Scripted answer [1]."
    assert sum(len(s["code"]) for s in out["sources"]) <= 8000


def test_impact_analysis_is_transitive(engine):
    imp = engine.impact("payment-service")
    assert imp["direct_dependents"] == ["order-service"]
    transitive = {d["service"]: d for d in imp["transitive_dependents"]}
    assert "api-gateway" in transitive and transitive["api-gateway"]["distance"] == 2
    assert "digraph" in imp["dot"]


def test_architecture_findings_detect_shared_database(engine):
    findings = engine.overview()["findings"]
    assert any(f["type"] == "shared_database" and "postgres" in f["detail"] for f in findings)


def test_service_docs_and_tree(engine):
    doc = engine.service_doc("order-service")
    assert "order-service" in doc and "/orders" in doc
    root = engine.tree_view("root")
    assert {c["name"] for c in root["children"]} >= {"order-service", "payment-service"}
    svc = engine.tree_view("svc:inventory-service")
    assert svc["children"]  # modules or files


def test_service_info(engine):
    info = engine.service_info("order-service")
    assert "payment-service" in info["depends_on"]
    assert "api-gateway" in info["used_by"]
    assert info["files"]
