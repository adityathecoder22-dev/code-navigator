"""Parse deployment and API configuration: docker-compose, Kubernetes, OpenAPI, gRPC .proto."""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

import yaml

from ..models import CodeUnit, SourceFile
from ..utils import log, truncate

SECRET_KEY_PATTERN = re.compile(r"(PASS|SECRET|TOKEN|KEY|CREDENTIAL)", re.IGNORECASE)
HTTP_METHODS = ("get", "post", "put", "delete", "patch", "head", "options")


@dataclass
class ComposeService:
    name: str
    file: str
    build_context: str | None = None   # directory relative to repo root
    image: str | None = None
    ports: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class K8sWorkload:
    name: str
    kind: str
    file: str
    images: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    ports: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class ProtoService:
    name: str
    package: str
    file: str
    rpcs: list[str] = field(default_factory=list)
    code: str = ""
    start_line: int = 1
    end_line: int = 1


@dataclass
class ApiOperation:
    method: str
    path: str
    file: str
    summary: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class ConfigInfo:
    compose: dict[str, ComposeService] = field(default_factory=dict)
    k8s: dict[str, K8sWorkload] = field(default_factory=dict)
    proto_services: list[ProtoService] = field(default_factory=list)
    api_operations: list[ApiOperation] = field(default_factory=list)


def mask_env(env: dict[str, str]) -> dict[str, str]:
    return {k: ("****" if SECRET_KEY_PATTERN.search(k) else str(v)) for k, v in env.items()}


def _env_to_dict(env) -> dict[str, str]:
    if isinstance(env, dict):
        return {str(k): "" if v is None else str(v) for k, v in env.items()}
    out = {}
    if isinstance(env, list):
        for item in env:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
            elif isinstance(item, dict) and "name" in item:  # k8s style
                out[str(item["name"])] = str(item.get("value", item.get("valueFrom", "")))
    return out


def parse_compose(sf: SourceFile, info: ConfigInfo) -> None:
    try:
        data = yaml.safe_load(sf.text) or {}
    except yaml.YAMLError as exc:
        log.warning("Could not parse %s: %s", sf.path, exc)
        return
    services = data.get("services") if isinstance(data, dict) else None
    if not isinstance(services, dict):
        return
    base_dir = posixpath.dirname(sf.path)
    for name, spec in services.items():
        if not isinstance(spec, dict):
            continue
        build = spec.get("build")
        context = None
        if isinstance(build, str):
            context = build
        elif isinstance(build, dict):
            context = build.get("context", ".")
        if context is not None:
            context = posixpath.normpath(posixpath.join(base_dir, context))
            context = "" if context in (".", "") else context
        deps = spec.get("depends_on") or []
        if isinstance(deps, dict):
            deps = list(deps.keys())
        ports = [str(p) for p in (spec.get("ports") or []) if not isinstance(p, dict)]
        info.compose[str(name)] = ComposeService(
            name=str(name), file=sf.path, build_context=context, image=spec.get("image"),
            ports=ports, env=_env_to_dict(spec.get("environment")), depends_on=[str(d) for d in deps],
            raw=spec,
        )


WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "Pod", "ReplicaSet"}


def parse_k8s(sf: SourceFile, info: ConfigInfo) -> None:
    try:
        docs = list(yaml.safe_load_all(sf.text))
    except yaml.YAMLError:
        return  # Helm templates etc.
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("kind") not in WORKLOAD_KINDS:
            continue
        name = (doc.get("metadata") or {}).get("name")
        if not name:
            continue
        spec = doc.get("spec") or {}
        if doc["kind"] == "CronJob":
            spec = ((spec.get("jobTemplate") or {}).get("spec") or {})
        pod = (spec.get("template") or {}).get("spec") or spec
        containers = (pod.get("containers") or []) + (pod.get("initContainers") or [])
        wl = K8sWorkload(name=str(name), kind=doc["kind"], file=sf.path, raw=doc)
        for c in containers:
            if not isinstance(c, dict):
                continue
            if c.get("image"):
                wl.images.append(str(c["image"]))
            wl.env.update(_env_to_dict(c.get("env")))
            for p in c.get("ports") or []:
                if isinstance(p, dict) and p.get("containerPort"):
                    wl.ports.append(str(p["containerPort"]))
        info.k8s[wl.name] = wl


def parse_openapi(sf: SourceFile, info: ConfigInfo) -> None:
    try:
        data = yaml.safe_load(sf.text)  # YAML is a superset of JSON
    except yaml.YAMLError:
        return
    if not isinstance(data, dict):
        return
    for path, ops in (data.get("paths") or {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            info.api_operations.append(ApiOperation(
                method=method.upper(), path=str(path), file=sf.path,
                summary=str(op.get("summary") or op.get("description") or op.get("operationId") or ""),
                raw=op,
            ))


_PROTO_SERVICE = re.compile(r"^\s*service\s+(\w+)\s*\{", re.MULTILINE)
_PROTO_RPC = re.compile(r"\brpc\s+(\w+)\s*\(")


def _block_end(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def parse_proto(sf: SourceFile, info: ConfigInfo) -> None:
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", sf.text, re.MULTILINE)
    package = pkg.group(1) if pkg else ""
    for m in _PROTO_SERVICE.finditer(sf.text):
        end = _block_end(sf.text, sf.text.index("{", m.start()))
        block = sf.text[m.start() : end + 1]
        info.proto_services.append(ProtoService(
            name=m.group(1), package=package, file=sf.path, rpcs=_PROTO_RPC.findall(block), code=block,
            start_line=sf.text.count("\n", 0, m.start()) + 1, end_line=sf.text.count("\n", 0, end) + 1,
        ))


def parse_configs(files: list[SourceFile]) -> ConfigInfo:
    info = ConfigInfo()
    handlers = {"compose": parse_compose, "k8s": parse_k8s, "openapi": parse_openapi, "proto": parse_proto}
    for sf in files:
        if sf.kind == "config" and sf.language in handlers:
            handlers[sf.language](sf, info)
    log.info("Configs: %d compose services, %d k8s workloads, %d proto services, %d OpenAPI operations",
             len(info.compose), len(info.k8s), len(info.proto_services), len(info.api_operations))
    return info


# ---------------------------------------------------------------- config units

def _dump(obj) -> str:
    try:
        return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False)
    except yaml.YAMLError:
        return str(obj)


def build_config_units(files: list[SourceFile], info: ConfigInfo, compose_to_service: dict[str, str],
                       k8s_to_service: dict[str, str]) -> list[CodeUnit]:
    """Turn configuration into searchable units, attached to the right service."""
    units: list[CodeUnit] = []

    for cs in info.compose.values():
        svc = compose_to_service.get(cs.name, cs.name)
        raw = dict(cs.raw)
        if "environment" in raw:
            raw["environment"] = mask_env(_env_to_dict(raw["environment"]))
        units.append(CodeUnit(
            id=f"compose:{cs.file}:{cs.name}", service=svc, file_path=cs.file, language="yaml",
            kind="config", name=f"docker-compose service {cs.name}", signature=f"services.{cs.name}",
            code=truncate(f"{cs.name}:\n{_dump(raw)}", 4000), start_line=1, end_line=1,
        ))

    for wl in info.k8s.values():
        svc = k8s_to_service.get(wl.name, wl.name)
        units.append(CodeUnit(
            id=f"k8s:{wl.file}:{wl.name}", service=svc, file_path=wl.file, language="yaml",
            kind="config", name=f"k8s {wl.kind} {wl.name}", signature=f"{wl.kind}/{wl.name}",
            code=truncate(_dump(wl.raw), 4000), start_line=1, end_line=1,
        ))

    by_path = {f.path: f for f in files}
    for ps in info.proto_services:
        owner = by_path[ps.file].service if ps.file in by_path else "_shared"
        units.append(CodeUnit(
            id=f"proto:{ps.file}:{ps.name}", service=owner, file_path=ps.file, language="proto",
            kind="api_spec", name=f"gRPC service {ps.name}", signature=f"service {ps.name}",
            code=ps.code, start_line=ps.start_line, end_line=ps.end_line,
            endpoints=[{"method": "RPC", "path": f"{ps.name}/{r}", "framework": "grpc"} for r in ps.rpcs],
        ))

    for op in info.api_operations:
        owner = by_path[op.file].service if op.file in by_path else "_shared"
        units.append(CodeUnit(
            id=f"openapi:{op.file}:{op.method}:{op.path}", service=owner, file_path=op.file,
            language="yaml", kind="api_spec", name=f"{op.method} {op.path}",
            signature=f"{op.method} {op.path} - {op.summary}"[:200],
            code=truncate(f"{op.method} {op.path}\n{_dump(op.raw)}", 3000), start_line=1, end_line=1,
            endpoints=[{"method": op.method, "path": op.path, "framework": "openapi"}],
        ))

    for sf in files:
        if sf.kind != "config" or sf.language not in ("dockerfile", "manifest", "app_config"):
            continue
        text = sf.text
        if sf.language == "app_config":
            text = re.sub(r"(?im)^(\s*[\w.-]*(password|secret|token|key)[\w.-]*\s*[:=]\s*).+$", r"\1****", text)
        units.append(CodeUnit(
            id=f"cfg:{sf.path}", service=sf.service, file_path=sf.path, language=sf.language,
            kind="config", name=posixpath.basename(sf.path), signature=sf.path,
            code=truncate(text, 5000), start_line=1, end_line=text.count("\n") + 1,
        ))
    return units
