from code_navigator.models import Dependency, Service
from pathlib import Path

from code_navigator.storage import IndexPaths
from code_navigator.utils import load_json


def _load(sample_index, offline_settings):
    paths = IndexPaths(Path(sample_index["index_path"]))
    services = {s["name"]: Service.from_dict(s) for s in load_json(paths.services)}
    edges = [Dependency(**d) for d in load_json(paths.dependencies)]
    return services, edges


def test_detects_app_and_infra_services(sample_index, offline_settings):
    services, _ = _load(sample_index, offline_settings)
    apps = {n for n, s in services.items() if s.kind == "app"}
    infra = {n for n, s in services.items() if s.kind == "infra"}
    assert apps == {"api-gateway", "order-service", "payment-service", "inventory-service", "notification-service"}
    assert infra == {"postgres", "kafka", "redis"}
    assert services["inventory-service"].language == "java"
    assert services["payment-service"].language == "go"


def test_endpoints_and_topics(sample_index, offline_settings):
    services, _ = _load(sample_index, offline_settings)
    gw = {(e["method"], e["path"]) for e in services["api-gateway"].endpoints}
    assert ("POST", "/api/orders") in gw
    assert "order-events" in services["order-service"].topics_published
    assert "payment-events" in services["payment-service"].topics_published
    assert set(services["notification-service"].topics_consumed) == {"order-events", "payment-events"}


def test_dependency_edges(sample_index, offline_settings):
    _, edges = _load(sample_index, offline_settings)
    pairs = {(e.source, e.target, e.kind) for e in edges}
    expected = {
        ("api-gateway", "order-service", "http"),
        ("api-gateway", "inventory-service", "http"),
        ("order-service", "payment-service", "http"),
        ("order-service", "inventory-service", "http"),
        ("order-service", "notification-service", "async"),
        ("payment-service", "notification-service", "async"),
        ("order-service", "postgres", "database"),
        ("payment-service", "redis", "database"),
    }
    assert expected <= pairs
    assert all(e.evidence for e in edges)
    # no self-loops or edges involving unassigned files
    assert not any(e.source == e.target or "_shared" in (e.source, e.target) for e in edges)
