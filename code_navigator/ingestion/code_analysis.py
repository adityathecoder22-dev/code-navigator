"""Static extraction of microservice signals from source code:
HTTP endpoints, outgoing URLs, environment-variable lookups, message topics, gRPC usage."""
from __future__ import annotations

import re

HTTP_VERBS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

# (framework, regex, method_group, path_group) ; method_group None => taken from match text
_ENDPOINT_RULES = [
    ("spring", re.compile(r'@(Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*|path\s*=\s*)?\{?\s*"([^"]*)"'), 1, 2),
    ("spring", re.compile(r'@(Get|Post|Put|Delete|Patch)Mapping\s*(?:\(\s*\))?\s*$', re.MULTILINE), 1, None),
    ("spring", re.compile(r'@RequestMapping\s*\(\s*(?:value\s*=\s*|path\s*=\s*)?\{?\s*"([^"]*)"[^)]*?(?:method\s*=\s*\{?\s*RequestMethod\.(\w+))?'), 2, 1),
    ("express", re.compile(r'\b(?:app|router|server|api|routes)\.(get|post|put|delete|patch|all)\s*\(\s*[\'"`]([^\'"`]+)'), 1, 2),
    ("python", re.compile(r'@\w+\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)'), 1, 2),
    ("flask", re.compile(r'@\w+\.route\s*\(\s*[\'"]([^\'"]+)[\'"](?:[^)]*methods\s*=\s*\[([^\]]*)\])?'), 2, 1),
    ("go", re.compile(r'\.(?:HandleFunc|Handle)\s*\(\s*"([^"]+)"[^\n]*?(?:\.Methods\(\s*"(\w+)")?'), 2, 1),
    ("go", re.compile(r'\b\w+\.(GET|POST|PUT|DELETE|PATCH)\s*\(\s*"([^"]+)"'), 1, 2),
    ("aspnet", re.compile(r'\[Http(Get|Post|Put|Delete|Patch)(?:\(\s*"([^"]*)"\s*\))?\]'), 1, 2),
    ("aspnet", re.compile(r'\.Map(Get|Post|Put|Delete|Patch)\s*\(\s*"([^"]+)"'), 1, 2),
]

_CLASS_PREFIX = [
    re.compile(r'@RequestMapping\s*\(\s*(?:value\s*=\s*|path\s*=\s*)?\{?\s*"([^"]*)"[^)]*\)\s*(?:\n\s*@\w+(?:\([^)]*\))?)*\s*\n\s*(?:public\s+)?(?:abstract\s+)?class'),
    re.compile(r'\[Route\(\s*"([^"]+)"\s*\)\]\s*(?:\n\s*\[[^\]]+\])*\s*\n\s*public\s+(?:sealed\s+)?class'),
    re.compile(r'APIRouter\s*\([^)]*prefix\s*=\s*[\'"]([^\'"]+)'),
]


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def extract_endpoints(text: str, language: str) -> list[dict]:
    endpoints: list[dict] = []
    prefix = ""
    for rx in _CLASS_PREFIX:
        m = rx.search(text)
        if m:
            prefix = m.group(1).rstrip("/")
            break
    prefix_span = None
    if prefix:
        pm = _CLASS_PREFIX[0].search(text) or _CLASS_PREFIX[1].search(text)
        prefix_span = pm.span() if pm else None

    for framework, rx, mg, pg in _ENDPOINT_RULES:
        for m in rx.finditer(text):
            if prefix_span and prefix_span[0] <= m.start() < prefix_span[1]:
                continue  # this is the class-level prefix itself
            method = (m.group(mg) if mg and m.group(mg) else "ANY") or "ANY"
            path = (m.group(pg) if pg else "") or ""
            if framework == "flask" and mg and m.group(mg):
                method = ",".join(x.strip(" '\"").upper() for x in m.group(mg).split(","))
            if framework == "go" and " " in path:  # Go 1.22 "POST /charge"
                maybe, rest = path.split(" ", 1)
                if maybe.upper() in HTTP_VERBS:
                    method, path = maybe, rest
            method = method.upper()
            if framework in ("spring", "aspnet", "python") and prefix:
                path = prefix + ("/" + path.lstrip("/") if path else "")
            if framework == "aspnet" and path and not path.startswith("/"):
                path = "/" + path
            endpoints.append({"method": method, "path": path or "/", "framework": framework,
                              "line": _line_of(text, m.start())})
    # de-duplicate
    seen, out = set(), []
    for e in endpoints:
        key = (e["method"], e["path"], e["line"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


_REQUIRE = re.compile(r'(?:const|let|var|import)\s+(\w+)\s*(?:=\s*require\(\s*|from\s+)[\'"](\.{1,2}/[^\'"]+)[\'"]')
_MOUNT = re.compile(r'\b(?:app|router|server)\.use\(\s*[\'"`](/[^\'"`]*)[\'"`]\s*,\s*(?:[\w.]+\s*,\s*)*(\w+)\s*\)')


def express_mounts(text: str) -> list[tuple[str, str]]:
    """Express `app.use('/api/orders', auth, ordersRouter)` + `require('./routes/orders')`
    -> [('/api/orders', './routes/orders')]"""
    modules = {m.group(1): m.group(2) for m in _REQUIRE.finditer(text)}
    return [(m.group(1).rstrip("/"), modules[m.group(2)]) for m in _MOUNT.finditer(text) if m.group(2) in modules]


_URL_HOST = re.compile(r'(?:https?|grpc|ws)://([A-Za-z0-9_.\-]+)(?::(\d+))?(/[^\s\'"`)]*)?')
_HOSTPORT = re.compile(r'["\'`]([a-z][a-z0-9\-_]{2,})(?::(\d{2,5}))["\'`/]')


def extract_hosts(text: str) -> list[tuple[str, int]]:
    """Hostnames referenced in URLs or 'host:port' strings, with line numbers."""
    out: list[tuple[str, int]] = []
    for m in _URL_HOST.finditer(text):
        out.append((m.group(1), _line_of(text, m.start())))
    for m in _HOSTPORT.finditer(text):
        out.append((m.group(1), _line_of(text, m.start())))
    return out


_ENV_RULES = [
    re.compile(r'os\.(?:getenv|environ\.get)\(\s*[\'"](\w+)[\'"]'),
    re.compile(r'os\.environ\[\s*[\'"](\w+)[\'"]\s*\]'),
    re.compile(r'process\.env\.(\w+)'),
    re.compile(r'process\.env\[\s*[\'"](\w+)[\'"]\s*\]'),
    re.compile(r'os\.(?:Getenv|LookupEnv)\(\s*"(\w+)"'),
    re.compile(r'mustMapEnv\([^,]+,\s*"(\w+)"'),
    re.compile(r'System\.getenv\(\s*"(\w+)"'),
    re.compile(r'Environment\.GetEnvironmentVariable\(\s*"(\w+)"'),
    re.compile(r'\$\{(\w+)(?::[^}]*)?\}'),
]


def extract_env_refs(text: str) -> list[tuple[str, int]]:
    out = []
    for rx in _ENV_RULES:
        for m in rx.finditer(text):
            out.append((m.group(1), _line_of(text, m.start())))
    return out


_TOPIC_RULES = [
    # (direction, regex)  direction: pub | sub | ctx (decide from surrounding words)
    ("sub", re.compile(r'@KafkaListener\s*\([^)]*topics\s*=\s*\{?\s*"([\w.\-]+)"')),
    ("sub", re.compile(r'@RabbitListener\s*\([^)]*queues\s*=\s*\{?\s*"([\w.\-]+)"')),
    ("pub", re.compile(r'(?:kafkaTemplate|producer|publisher|template|channel|writer|bus)\w*\.(?:send|produce|publish|basic_publish|emit|convertAndSend)\w*\s*\(\s*[\'"]([\w.\-]+)[\'"]', re.IGNORECASE)),
    ("ctx", re.compile(r'\btopic\s*[:=]\s*[\'"]([\w.\-]+)[\'"]', re.IGNORECASE)),
    ("ctx", re.compile(r'\b(?:TOPIC|QUEUE|EXCHANGE)\w*\s*[:=]\s*[\'"]([\w.\-]+)[\'"]')),
]
_PUB_WORDS = re.compile(r"produc|publish|writer|send|emit|outbox", re.IGNORECASE)
_SUB_WORDS = re.compile(r"consum|subscrib|reader|listen|poll|handler", re.IGNORECASE)


_SUB_LISTS = re.compile(r'(?:\.subscribe|KafkaConsumer|\.Subscribe(?:Topics)?)\s*\(\s*([^)]*)\)', re.DOTALL)
_QUOTED_TOPIC = re.compile(r'[\'"]([A-Za-z][\w.\-]*)[\'"]')
_NOT_TOPICS = {"utf-8", "earliest", "latest", "json"}


def extract_topics(text: str) -> list[dict]:
    lines = text.splitlines()
    found: list[dict] = []
    for m in _SUB_LISTS.finditer(text):
        args = m.group(1)
        args = re.split(r"\b(?:bootstrap_servers|group_id|value_deserializer|key_deserializer|auto_offset_reset)\s*=", args)[0]
        for t in _QUOTED_TOPIC.findall(args):
            if t.lower() not in _NOT_TOPICS and ":" not in t:
                found.append({"topic": t, "direction": "sub", "line": _line_of(text, m.start())})
    for direction, rx in _TOPIC_RULES:
        for m in rx.finditer(text):
            line = _line_of(text, m.start())
            d = direction
            if d == "ctx":
                window = "\n".join(lines[max(0, line - 4) : line + 3])
                pub, sub = bool(_PUB_WORDS.search(window)), bool(_SUB_WORDS.search(window))
                if pub == sub:
                    # Constant definitions: decide later from usage elsewhere in the file
                    pub = bool(_PUB_WORDS.search(text)) and not bool(_SUB_WORDS.search(text))
                    sub = bool(_SUB_WORDS.search(text)) and not bool(_PUB_WORDS.search(text))
                    if pub == sub:
                        continue
                d = "pub" if pub else "sub"
            found.append({"topic": m.group(1), "direction": d, "line": line})
    seen, out = set(), []
    for t in found:
        key = (t["topic"], t["direction"])
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def grpc_client_refs(text: str, proto_service: str) -> bool:
    """True if this code creates a gRPC client/stub for `proto_service`."""
    s = re.escape(proto_service)
    pattern = rf"(New{s}Client\b|\b{s}Stub\b|\b{s}Grpc\.new\w*Stub|\b{s}Client\s*\(|\b{s}\.{s}Client\b)"
    return re.search(pattern, text) is not None


def grpc_server_refs(text: str, proto_service: str) -> bool:
    """True if this code implements/registers a gRPC server for `proto_service`."""
    s = re.escape(proto_service)
    pattern = rf"(Register{s}Server\b|add_{s}Servicer_to_server|\b{s}Servicer\b|{s}Grpc\.{s}ImplBase|\b{s}\.{s}Base\b|{s}\.service\b)"
    return re.search(pattern, text) is not None


_DB_HINTS = {
    "postgres": re.compile(r"postgres(ql)?://|psycopg|pgx|jdbc:postgresql|Npgsql", re.IGNORECASE),
    "mysql": re.compile(r"mysql://|jdbc:mysql|pymysql|go-sql-driver/mysql", re.IGNORECASE),
    "mongodb": re.compile(r"mongodb(\+srv)?://|pymongo|mongoose|mongo-driver", re.IGNORECASE),
    "redis": re.compile(r"redis://|\bredis\.|StackExchange\.Redis|go-redis|ioredis", re.IGNORECASE),
}


def extract_datastores(text: str) -> list[str]:
    return [name for name, rx in _DB_HINTS.items() if rx.search(text)]
