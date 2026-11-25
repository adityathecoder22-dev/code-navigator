"""Work out which microservices exist and which files belong to each.

Signals, strongest first:
  1. docker-compose services with a `build` context  -> exact directory
  2. Kubernetes workloads / compose services whose name matches a directory name
  3. Directories containing a Dockerfile or build manifest (pom.xml, package.json, go.mod ...)
  4. Fallback: the whole repository is one service
Services that only reference an image (redis, kafka, postgres ...) become `infra` services.
"""
from __future__ import annotations

import posixpath
from collections import Counter

from ..models import Service
from ..utils import log, names_match
from .config_parser import ConfigInfo, mask_env
from .repo_loader import TEST_DIRS, Repository

MARKER_FILES = {
    "dockerfile", "pom.xml", "build.gradle", "build.gradle.kts", "go.mod", "package.json",
    "requirements.txt", "pyproject.toml", "setup.py", "cargo.toml", "gemfile", "composer.json",
}
IGNORED_MARKER_DIRS = TEST_DIRS | {"docs", "doc", "scripts", "hack", "tools", "examples", "deploy",
                                   "deployment", "kubernetes-manifests", "helm", "charts", "protos", "proto"}

INFRA_HINTS = {
    "postgres": "database", "mysql": "database", "mariadb": "database", "mongo": "database",
    "redis": "cache", "memcached": "cache", "kafka": "broker", "zookeeper": "broker",
    "rabbitmq": "broker", "nats": "broker", "elasticsearch": "search", "jaeger": "tracing",
    "zipkin": "tracing", "prometheus": "monitoring", "grafana": "monitoring", "consul": "discovery",
    "eureka": "discovery", "keycloak": "auth", "nginx": "proxy", "traefik": "proxy", "envoy": "proxy",
    "minio": "storage", "localstack": "cloud", "spanner": "database", "cassandra": "database",
}


def _image_base(image: str | None) -> str:
    if not image:
        return ""
    base = image.split("@")[0].split("/")[-1]
    return base.split(":")[0]


def _infra_type(name: str, image: str | None) -> str | None:
    text = f"{name} {_image_base(image)}".lower()
    for hint, kind in INFRA_HINTS.items():
        if hint in text:
            return kind
    return None


def _all_dirs(repo: Repository) -> set[str]:
    dirs: set[str] = set()
    for f in repo.files:
        d = posixpath.dirname(f.path)
        while d:
            dirs.add(d)
            d = posixpath.dirname(d)
    return dirs


def _find_dir_for_name(name: str, dirs: set[str]) -> str | None:
    candidates = [d for d in dirs if names_match(posixpath.basename(d), name)]
    if not candidates:
        return None
    return min(candidates, key=lambda d: (d.count("/"), len(d)))


def _is_under(path: str, root: str) -> bool:
    return root == "" or path == root or path.startswith(root + "/")


def detect_services(repo: Repository, info: ConfigInfo) -> tuple[list[Service], dict[str, str], dict[str, str]]:
    """Returns (services, compose_name->service, k8s_name->service). Also sets SourceFile.service."""
    services: dict[str, Service] = {}
    dirs = _all_dirs(repo)
    compose_map: dict[str, str] = {}
    k8s_map: dict[str, str] = {}

    def add(name: str, path: str, how: str, kind: str = "app") -> Service:
        existing = next((s for s in services.values() if kind == "app" and s.kind == "app" and s.path == path and path), None)
        if existing:
            if how not in existing.detected_by:
                existing.detected_by.append(how)
            return existing
        if name in services:
            svc = services[name]
            if how not in svc.detected_by:
                svc.detected_by.append(how)
            if not svc.path and path:
                svc.path = path
            return svc
        svc = Service(name=name, path=path, kind=kind, detected_by=[how])
        services[name] = svc
        return svc

    # 1. compose
    # `build: .` means "this repository" when it is the only built service, even if the compose file
    # also starts image-only infra (postgres, redis, kafka ...) next to it.
    built = [c for c in info.compose.values() if c.build_context is not None]
    for cs in info.compose.values():
        if cs.build_context is not None and cs.build_context != "":
            svc = add(cs.name, cs.build_context, "docker-compose build")
        elif cs.build_context == "" and len(built) == 1:
            svc = add(cs.name, "", "docker-compose build")
        else:
            infra = _infra_type(cs.name, cs.image)
            match_dir = None if infra else _find_dir_for_name(cs.name, dirs)
            if match_dir is not None:
                svc = add(cs.name, match_dir, "docker-compose name match")
            else:
                svc = add(cs.name, "", "docker-compose image", kind="infra" if infra or cs.image else "app")
        compose_map[cs.name] = svc.name
        svc.image = svc.image or cs.image
        svc.ports = sorted(set(svc.ports) | set(cs.ports))
        svc.env.update(mask_env(cs.env))

    # 2. kubernetes
    for wl in info.k8s.values():
        existing = next((s for s in services.values() if names_match(s.name, wl.name)), None)
        if existing:
            svc = existing
            if "kubernetes" not in svc.detected_by:
                svc.detected_by.append("kubernetes")
        else:
            image_name = _image_base(wl.images[0]) if wl.images else ""
            match_dir = _find_dir_for_name(wl.name, dirs) or (_find_dir_for_name(image_name, dirs) if image_name else None)
            infra = _infra_type(wl.name, wl.images[0] if wl.images else None)
            if match_dir is not None and not infra:
                svc = add(wl.name, match_dir, "kubernetes")
            else:
                svc = add(wl.name, "", "kubernetes image", kind="infra" if infra else "app")
        k8s_map[wl.name] = svc.name
        if wl.images and not svc.image:
            svc.image = wl.images[0]
        svc.ports = sorted(set(svc.ports) | set(wl.ports))
        for k, v in mask_env(wl.env).items():
            svc.env.setdefault(k, v)

    # 3. marker directories not already covered
    claimed = [s.path for s in services.values() if s.kind == "app" and s.path]
    marker_dirs: set[str] = set()
    for f in repo.files:
        if posixpath.basename(f.path).lower() in MARKER_FILES or f.path.endswith(".csproj"):
            d = posixpath.dirname(f.path)
            parts = d.split("/") if d else []
            if any(p.lower() in IGNORED_MARKER_DIRS for p in parts):
                continue
            marker_dirs.add(d)
    if len(marker_dirs) > 1:
        marker_dirs.discard("")  # a root-level manifest in a monorepo is not a service
    root_built = any(s.kind == "app" and s.path == "" and "docker-compose build" in s.detected_by
                     for s in services.values())
    for d in sorted(marker_dirs, key=lambda x: (x.count("/"), x)):
        if any(_is_under(d, c) or _is_under(c, d) for c in claimed if c):
            continue
        if d == "" and root_built:  # root manifest belongs to the `build: .` compose service
            continue
        if any(_is_under(d, m) for m in marker_dirs if m != d and m and _is_under(d, m)):
            continue
        name = posixpath.basename(d) or repo.name
        if name in services:
            name = d.replace("/", "-")
        add(name, d, "build manifest")
        claimed.append(d)

    # 4. fallback
    if not any(s.kind == "app" for s in services.values()):
        add(repo.name, "", "repository root")

    # assign files: deepest matching service directory wins
    app_services = sorted((s for s in services.values() if s.kind == "app"),
                          key=lambda s: len(s.path), reverse=True)
    single_root = len(app_services) == 1 and app_services[0].path == ""
    for f in repo.files:
        owner = "_shared"
        for svc in app_services:
            if (svc.path and _is_under(f.path, svc.path)) or (single_root and svc.path == ""):
                owner = svc.name
                break
        f.service = owner
        if owner in services:
            services[owner].files.append(f.path)

    for svc in services.values():
        langs = Counter(f.language for f in repo.files if f.service == svc.name and f.kind == "code")
        if langs:
            svc.language = langs.most_common(1)[0][0]
        elif svc.kind == "infra":
            svc.language = _infra_type(svc.name, svc.image) or "infra"

    shared = [f for f in repo.files if f.service == "_shared"]
    if shared:
        services["_shared"] = Service(name="_shared", path="", kind="shared", detected_by=["unassigned files"],
                                      files=[f.path for f in shared])

    result = sorted(services.values(), key=lambda s: ({"app": 0, "infra": 1, "shared": 2}.get(s.kind, 3), s.name))
    log.info("Detected %d services: %s", len(result), ", ".join(f"{s.name}({s.kind})" for s in result))
    return result, compose_map, k8s_map
