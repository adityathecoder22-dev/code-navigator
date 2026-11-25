from code_navigator.models import Hit
from code_navigator.retrieval.hybrid_retriever import code_identifiers, reciprocal_rank_fusion
from code_navigator.retrieval.tree_navigator import TreeNavigator
from code_navigator.utils import extract_json


def _h(cid):
    return Hit(chunk_id=cid, score=0.0, text=cid, metadata={})


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"a": [_h("x"), _h("y")], "b": [_h("y"), _h("z")]})
    assert fused[0].chunk_id == "y"
    assert set(fused[0].found_by) == {"a", "b"}
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-4


def test_code_identifiers():
    ids = code_identifiers("Where is `ChargeWithRetry` called and what does place_order() or OrderService.place_order do?")
    assert "chargewithretry" in ids and "place_order" in ids and "orderservice.place_order" in ids


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"selected": ["S1"]} hope that helps') == {"selected": ["S1"]}
    assert extract_json("not json", default={}) == {}


def test_all_modes_return_hits(engine):
    for mode in ("hybrid", "vector", "vectorless"):
        res = engine.search("How does the payment service retry charges to the provider?", mode=mode)
        assert res["hits"], mode
        files = {h["file_path"] for h in res["hits"]}
        assert any("payment" in f for f in files), (mode, files)


def test_symbol_lookup_finds_exact_function(engine):
    res = engine.search("What does ChargeWithRetry do?", mode="hybrid")
    assert res["hits"][0]["name"] == "ProviderGateway.ChargeWithRetry"
    assert "symbol" in res["hits"][0]["found_by"]


def test_navigator_uses_llm_choices(engine, scripted_llm):
    llm = scripted_llm(["payment-service", "order-service"])
    nav = TreeNavigator(engine.tree, llm, engine.edges, max_services=3).navigate("What happens when payment fails?")
    assert set(nav.services) == {"order-service", "payment-service"}
    assert nav.trace[0]["method"] == "llm"
    assert nav.confidence == 0.9 and nav.scope == "cross_service"
    assert nav.files and all(f.startswith(("order-service", "payment-service", "docker-compose")) for f in nav.files)


def test_navigator_falls_back_to_keywords_when_llm_gives_nothing(engine, scripted_llm):
    llm = scripted_llm(["no-such-keyword"])
    nav = TreeNavigator(engine.tree, llm, engine.edges).navigate("How does the inventory reserve stock?")
    assert "inventory-service" in nav.services
    assert nav.trace[0]["method"] == "keyword"
    assert nav.confidence < 0.6


def test_service_kept_when_children_cannot_be_chosen(engine, scripted_llm):
    # picks services by name, but the keyword never matches any module/file name
    llm = scripted_llm(["inventory-service"])
    nav = TreeNavigator(engine.tree, llm, engine.edges, max_children=1).navigate("zzz qqq")
    assert "inventory-service" in nav.services
    assert "inventory-service" in nav.whole_services or any(f.startswith("inventory-service") for f in nav.files)
