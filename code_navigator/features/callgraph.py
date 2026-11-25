"""Function-level call graph.

Built at index time from the parsed code units. Inside a service, each call site is
resolved to the function/method it most likely invokes:

    resolution     confidence  example
    self           0.95        self.helper() / this.helper() / h.Charge() inside a Go method of h
    same_file      0.95        helper() defined in the same file
    field_type     0.90        self.payments.charge()  where  self.payments = PaymentClient()
    local_type     0.85        repo = OrderRepository(); repo.save()
    class_name     0.90        OrderService.create() / PaymentClient()  (instantiation)
    import         0.90        from .clients import _with_retry; _with_retry()  /  payment.NewHandler()
    same_package   0.90        Go: a function of the same package (directory)
    name_match     0.50        the only method with that name in the service
    ambiguous      0.30        one of 2-3 methods with that name (stored, hidden by default)

Across services, HTTP client code is linked to the handler of the endpoint it calls
(resolution ``endpoint_path``), and a function that publishes to a Kafka/RabbitMQ topic is
linked to the functions that consume it (edge kind ``async``, resolution ``topic``).

Structural edges: ``inherits`` / ``implements`` (class declarations, Go struct embedding) and
``imports`` (file -> imported class/function, or file -> file).
Results are heuristics from static analysis: dynamic dispatch, reflection and URLs built at
runtime are not visible.
"""
from __future__ import annotations

import posixpath
import re
from collections import Counter, defaultdict, deque

from ..ingestion.code_analysis import extract_topics
from ..models import CodeUnit, Dependency, Service, SourceFile

CALLER_KINDS = {"function", "method", "module"}
TARGET_KINDS = {"function", "method", "class"}
DEFAULT_MIN_CONFIDENCE = 0.5

_NOT_CALLS = {
    "if", "for", "while", "switch", "catch", "return", "function", "func", "fn", "def", "class", "new",
    "sizeof", "typeof", "await", "async", "elif", "else", "try", "with", "lambda", "super", "assert",
    "throw", "throws", "yield", "match", "case", "when", "using", "lock", "foreach", "select", "go",
    "defer", "not", "and", "or", "in", "is", "print", "raise", "except", "synchronized", "require",
}
# Too generic to resolve by name alone (only resolved through self/field/local/import types)
_GENERIC = {
    "get", "set", "put", "post", "patch", "delete", "add", "append", "remove", "update", "items", "keys",
    "values", "pop", "push", "map", "filter", "reduce", "join", "split", "format", "strip", "replace",
    "lower", "upper", "close", "open", "read", "write", "send", "emit", "on", "then", "json", "dict",
    "copy", "clear", "find", "run", "start", "stop", "call", "apply", "toString", "equals", "hashCode",
    "length", "size", "isEmpty", "contains", "forEach", "of", "parse", "error", "info", "warn", "debug",
    "log", "status", "next", "init", "main", "test", "String", "Error", "Errorf", "Println", "Printf",
}
_DECL_PREFIX = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|func|function|class|fun|fn|interface|struct)\b")
_JAVA_DECL = re.compile(r"\b([A-Za-z_]\w*)(?:<[^()]*>)?(?:\[\])?\s+$")
_DECL_OK_WORDS = {"return", "await", "yield", "new", "not", "and", "or", "in", "throw", "raise", "else",
                  "case", "go", "defer", "if", "elif", "while", "print", "assert", "is", "echo", "then", "do"}
_CALL_RE = re.compile(r"([A-Za-z_][\w]*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\(")
_CHAINED_RE = re.compile(r"\)\s*\.\s*([A-Za-z_]\w*)\s*\(")
_COMMENT = ("//", "#", "*", "/*", "--")
_VERBS = ("get", "post", "put", "delete", "patch")

CONFIDENCE = {"self": 0.95, "same_file": 0.95, "field_type": 0.9, "class_name": 0.9, "import": 0.9,
              "same_package": 0.9, "local_type": 0.85, "inherited": 0.85, "name_match": 0.5, "ambiguous": 0.3}

# edge kinds: what depends on what (used by impact analysis)
CALL_KINDS = ("calls", "instantiates", "http", "async")
STRUCTURAL_KINDS = ("inherits", "implements", "imports")
ALL_KINDS = CALL_KINDS + STRUCTURAL_KINDS
_TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|spec|specs)/|(^|/)test_[^/]*$|_test\.(go|py)$|"
                        r"\.(spec|test)\.[jt]sx?$|Tests?\.(java|cs|kt)$")


def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE.search(path or ""))


def _short(name: str) -> str:
    return name.split(".")[-1]


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0]


# ------------------------------------------------------------------ call-site extraction

def unit_lines(unit: CodeUnit, file_text: str | None = None, covered: set[int] | None = None):
    """(line number, text) of a unit. A module unit is the file's top-level code, so its lines
    come from the file itself, skipping lines that belong to functions and classes."""
    if unit.kind == "module" and file_text is not None:
        return [(i, t) for i, t in enumerate(file_text.splitlines(), start=1) if i not in (covered or set())]
    return [(unit.start_line + i, t) for i, t in enumerate(unit.code.splitlines())]


def call_sites(unit: CodeUnit, lines=None) -> list[tuple[str | None, str, int]]:
    """(receiver, name, line) for every call in a unit's code. Receiver is None for bare calls,
    'new' for constructor calls and '?' when called on an expression result (foo().bar())."""
    out: list[tuple[str | None, str, int]] = []
    in_block_comment = False
    for lineno, raw in (lines if lines is not None else unit_lines(unit)):
        line = raw.strip()
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue
        if line.startswith("/*") and "*/" not in line:
            in_block_comment = True
            continue
        if not line or line.startswith(_COMMENT) or line.startswith(("@", "import ", "from ", "package ")):
            continue
        if _DECL_PREFIX.match(raw):
            body = raw.split("{", 1)[1] if "{" in raw else ""
            if not body:
                continue
            raw = body
        for m in _CALL_RE.finditer(raw):
            start = m.start()
            if start > 0 and (raw[start - 1].isalnum() or raw[start - 1] in "_.$"):
                continue
            dotted = re.sub(r"\s+", "", m.group(1))
            parts = dotted.split(".")
            name, receiver = parts[-1], ".".join(parts[:-1]) or None
            if name in _NOT_CALLS:
                continue
            before = raw[:start]
            decl = _JAVA_DECL.search(before)
            if receiver is None and decl and decl.group(1) not in _DECL_OK_WORDS and not before.rstrip().endswith(("=", "(", ",")):
                continue  # a declaration such as `public void reserve(` or `Product reserve(`
            if receiver is None and re.search(r"\bnew\s+$", before):
                receiver = "new"
            out.append((receiver, name, lineno))
        for m in _CHAINED_RE.finditer(raw):
            out.append(("?", m.group(1), lineno))
    return out


# ------------------------------------------------------------------ type hints

def _go_receiver_var(unit: CodeUnit) -> str | None:
    m = re.match(r"\s*func\s*\(\s*(\w+)\s+\*?\w+", unit.signature or "")
    return m.group(1) if m else None


def field_types(code: str, language: str, known: set[str]) -> dict[str, str]:
    """Field name -> class name, for fields whose type is a class of the same service."""
    pairs: list[tuple[str, str]] = []
    if language == "python":
        pairs += re.findall(r"self\.(\w+)\s*(?::\s*[\w\[\]., |]+)?=\s*([A-Za-z_]\w*)\s*\(", code)
        pairs += [(n, t) for n, t in re.findall(r"^\s+(\w+)\s*:\s*([A-Z]\w*)\s*(?:=|$)", code, re.M)]
    elif language in ("javascript", "typescript", "tsx"):
        pairs += re.findall(r"this\.(\w+)\s*=\s*new\s+(\w+)", code)
        pairs += re.findall(r"(?:private|public|protected|readonly)\s+(?:readonly\s+)?(\w+)\s*:\s*(\w+)", code)
    elif language == "go":
        pairs += [(n, t) for n, t in re.findall(r"^\s*(\w+)\s+\*?(?:\[\])?(?:\w+\.)?(\w+)\s*(?:`[^`]*`)?\s*$", code, re.M)]
        pairs += re.findall(r"\b(\w+)\s*:\s*&?(?:\w+\.)?(?:New)?([A-Z]\w*)\s*[({]", code)
    else:  # java, csharp, kotlin, scala, php, rust ...
        pairs += [(n, t) for t, n in re.findall(
            r"(?:private|protected|public|internal|readonly|final|static|\s)\s*([A-Z]\w*)(?:<[^>]*>)?\s+(\w+)\s*[;=]", code)]
        pairs += re.findall(r"(?:val|var|let)\s+(\w+)\s*:\s*([A-Z]\w*)", code)
        pairs += [(n, t) for t, n in re.findall(r"[(,]\s*(?:final\s+)?([A-Z]\w*)(?:<[^>]*>)?\s+(\w+)\s*(?=[,)])", code)]
    return {n: t for n, t in pairs if t in known}


def local_types(code: str, language: str, known: set[str]) -> dict[str, str]:
    pairs: list[tuple[str, str]] = []
    if language == "python":
        pairs += re.findall(r"\b(\w+)\s*=\s*([A-Z]\w*)\s*\(", code)
        pairs += re.findall(r"\b(\w+)\s*:\s*([A-Z]\w*)\s*[=,)]", code)
    elif language == "go":
        pairs += re.findall(r"\b(\w+)\s*:?=\s*&?(?:\w+\.)?New([A-Z]\w*)\s*\(", code)
        pairs += re.findall(r"\b(\w+)\s*:?=\s*&?(?:\w+\.)?([A-Z]\w*)\s*\{", code)
        pairs += re.findall(r"\b(\w+)\s+\*?(?:\w+\.)?([A-Z]\w*)\s*[,)]", code)  # parameters
    else:
        pairs += re.findall(r"(?:const|let|var|val|auto)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*new\s+(\w+)", code)
        pairs += [(n, t) for t, n in re.findall(r"\b([A-Z]\w*)(?:<[^>]*>)?\s+(\w+)\s*[=;,)]", code)]
        pairs += re.findall(r"\b(\w+)\s*:\s*([A-Z]\w*)\s*[=,)]", code)
    return {n: t for n, t in pairs if t in known}


def file_imports(text: str, language: str) -> dict[str, str]:
    """Imported name/alias -> module hint (file stem or package directory)."""
    imports: dict[str, str] = {}
    if language == "python":
        for mod, names in re.findall(r"^\s*from\s+([.\w]+)\s+import\s+\(?([^)\n]+)", text, re.M):
            hint = mod.strip(".").split(".")[-1] or mod
            for n in names.split(","):
                n = n.strip().strip("()")
                if n:
                    parts = n.split(" as ")
                    imports[parts[-1].strip()] = hint if mod.strip(".") else parts[0].strip()
        for mod, alias in re.findall(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", text, re.M):
            imports[alias or mod.split(".")[0]] = mod.split(".")[-1]
    elif language in ("javascript", "typescript", "tsx"):
        for names, single, mod in re.findall(
                r"(?:const|let|var)\s+(?:\{([^}]+)\}|(\w+))\s*=\s*require\(\s*['\"]([^'\"]+)['\"]", text):
            for n in ([single] if single else [x.split(":")[-1].strip() for x in names.split(",")]):
                if n:
                    imports[n] = _stem(mod)
        for default, names, mod in re.findall(
                r"import\s+(?:(\w+)\s*,?\s*)?(?:\{([^}]+)\})?\s*from\s+['\"]([^'\"]+)['\"]", text):
            for n in [default] + [x.split(" as ")[-1].strip() for x in (names or "").split(",")]:
                if n:
                    imports[n] = _stem(mod)
    elif language == "go":
        block = re.findall(r"import\s*\(([^)]*)\)", text, re.S)
        specs = re.findall(r"^\s*(?:(\w+)\s+)?\"([^\"]+)\"", "\n".join(block), re.M)
        specs += re.findall(r"^import\s+(?:(\w+)\s+)?\"([^\"]+)\"", text, re.M)
        for alias, path in specs:
            imports[alias or path.split("/")[-1]] = path.split("/")[-1]
    else:  # java / kotlin / scala / csharp
        for pkg, name in re.findall(r"^\s*import\s+(?:static\s+)?([\w.]+)\.(\w+)\s*;?", text, re.M):
            imports[name] = name
    return imports


def _clean_type(name: str) -> str:
    name = re.sub(r"<.*>|\[.*\]|\(.*\)", "", name).strip()
    return name.split(".")[-1].strip("*& ")


def class_bases(unit: CodeUnit) -> list[tuple[str, str]]:
    """[(base class name, 'inherits'|'implements')] from a class declaration."""
    lines = unit.code.splitlines()
    head = ""
    for line in lines[:8]:
        if line.strip().startswith(("@", "[", "#", "//", "/*", "*")):
            continue
        head += " " + line.strip()
        if "{" in line or line.rstrip().endswith(":"):
            break
    head = head.split("{", 1)[0]
    lang = unit.language
    out: list[tuple[str, str]] = []
    if lang == "python":
        m = re.search(r"class\s+\w+\s*\(([^)]*)\)", head)
        if m:
            out += [(_clean_type(b), "inherits") for b in m.group(1).split(",") if b.strip() and "=" not in b]
    elif lang == "go":
        m = re.search(r"struct\s*\{", unit.code)
        if m:
            for line in unit.code[m.end():].splitlines():
                mm = re.fullmatch(r"\s*\*?(?:\w+\.)?([A-Z]\w*)\s*(?:`[^`]*`)?\s*", line)
                if mm:
                    out.append((mm.group(1), "inherits"))
    else:
        m = re.search(r"\bextends\s+(.+?)(?=\bimplements\b|\bwith\b|$)", head)
        if m:
            out += [(_clean_type(b), "inherits") for b in re.split(r",(?![^<]*>)", m.group(1)) if b.strip()]
        m = re.search(r"\bimplements\s+(.+)$", head)
        if m:
            out += [(_clean_type(b), "implements") for b in re.split(r",(?![^<]*>)", m.group(1)) if b.strip()]
        out += [(_clean_type(b), "implements") for b in re.findall(r"\bwith\s+([\w.]+)", head)]  # scala traits
        if not out and lang in ("csharp", "kotlin", "swift", "cpp"):
            m = re.search(r"(?:class|struct|object|interface)\s+\w+(?:<[^>]*>)?(?:\([^)]*\))?\s*:\s*(.+)$", head)
            if m:
                for i, b in enumerate(re.split(r",(?![^<]*>)", m.group(1))):
                    name = _clean_type(b)
                    iface = i > 0 or (lang == "csharp" and re.match(r"I[A-Z]", name))
                    out.append((name, "implements" if iface else "inherits"))
    return [(b, k) for b, k in out if b and b[0].isalpha()]


def _matches_hint(unit: CodeUnit, hint: str) -> bool:
    d = posixpath.dirname(unit.file_path)
    return _stem(unit.file_path) == hint or d == hint or d.endswith("/" + hint)


# ------------------------------------------------------------------ builder

class _ServiceIndex:
    def __init__(self, units: list[CodeUnit], files: dict[str, SourceFile]):
        self.units = units
        self.by_short: dict[str, list[CodeUnit]] = defaultdict(list)
        self.methods: dict[tuple[str, str], list[CodeUnit]] = defaultdict(list)
        self.classes: dict[str, list[CodeUnit]] = defaultdict(list)
        self.members: dict[str, list[CodeUnit]] = defaultdict(list)
        for u in units:
            if u.kind in ("function", "method"):
                self.by_short[_short(u.name)].append(u)
                if u.parent:
                    self.methods[(_short(u.parent), _short(u.name))].append(u)
                    self.members[_short(u.parent)].append(u)
            elif u.kind == "class":
                self.classes[_short(u.name)].append(u)
        known = set(self.classes)
        self.known = known
        self.fields: dict[str, dict[str, str]] = {}
        for cls, cls_units in self.classes.items():
            code = "\n".join(c.code for c in cls_units + self.members.get(cls, []))
            self.fields[cls] = field_types(code, cls_units[0].language, known)
        self.imports: dict[str, dict[str, str]] = {}
        self.paths = sorted({u.file_path for u in units})
        for path in self.paths:
            sf = files.get(path)
            self.imports[path] = file_imports(sf.text, sf.language) if sf else {}
        # class -> [(base class unit, kind)] for classes of this service
        self.bases: dict[str, list[tuple[CodeUnit, str]]] = defaultdict(list)
        for cls, cls_units in self.classes.items():
            for cu in cls_units:
                for base, kind in class_bases(cu):
                    cands = self._prefer(self.classes.get(base, []), cu, self.imports.get(cu.file_path, {}).get(base))
                    if cands and cands[0].id != cu.id:
                        self.bases[cls].append((cands[0], kind))

    def _prefer(self, cands: list[CodeUnit], caller: CodeUnit, hint: str | None) -> list[CodeUnit]:
        if len(cands) <= 1:
            return cands
        for test in ((lambda c: hint and _matches_hint(c, hint)), (lambda c: c.file_path == caller.file_path),
                     (lambda c: posixpath.dirname(c.file_path) == posixpath.dirname(caller.file_path))):
            narrowed = [c for c in cands if test(c)]
            if narrowed:
                return narrowed
        return cands

    def _method(self, cls: str, name: str, caller: CodeUnit) -> CodeUnit | None:
        found = self._method_with_origin(cls, name, caller)
        return found[0] if found else None

    def _method_with_origin(self, cls: str, name: str, caller: CodeUnit) -> tuple[CodeUnit, bool] | None:
        """Look the method up on the class, then on its base classes (breadth first)."""
        queue, seen = deque([(_short(cls), False)]), set()
        while queue:
            c, inherited = queue.popleft()
            if c in seen or len(seen) > 20:
                continue
            seen.add(c)
            cands = self._prefer(self.methods.get((c, name), []), caller, None)
            if cands:
                return cands[0], inherited
            queue.extend((_short(b.name), True) for b, _ in self.bases.get(c, []))
        return None

    def resolve(self, caller: CodeUnit, receiver: str | None, name: str,
                locals_: dict[str, str]) -> list[tuple[CodeUnit, str, str]]:
        """-> [(target, edge kind, resolution)]"""
        cls = _short(caller.parent) if caller.parent else None
        imports = self.imports.get(caller.file_path, {})
        self_names = {"self", "this", "cls", "super"}
        go_recv = _go_receiver_var(caller) if caller.language == "go" else None
        if go_recv:
            self_names.add(go_recv)

        if receiver == "new" or (receiver is None and name in self.classes):
            cands = self._prefer(self.classes.get(name, []), caller, imports.get(name))
            return [(cands[0], "instantiates", "class_name")] if cands else []

        if receiver is None:
            same_file = [u for u in self.by_short.get(name, []) if u.file_path == caller.file_path and not u.parent]
            if same_file:
                return [(same_file[0], "calls", "same_file")]
            found = self._method_with_origin(cls, name, caller) if cls else None
            if found is not None:
                return [(found[0], "calls", "inherited" if found[1] else "self")]
            if name in imports:
                cands = [u for u in self.by_short.get(name, []) if _matches_hint(u, imports[name])]
                if cands:
                    return [(cands[0], "calls", "import")]
            if caller.language == "go":
                d = posixpath.dirname(caller.file_path)
                pkg = [u for u in self.by_short.get(name, []) if not u.parent and posixpath.dirname(u.file_path) == d]
                if pkg:
                    return [(pkg[0], "calls", "same_package")]
            return self._by_name(caller, name)

        parts = receiver.split(".")
        head = parts[0]
        target_cls, resolution = self._chain_type(parts, cls, self_names, locals_)
        if target_cls is None and len(parts) == 1 and head in imports:
            hint = imports[head]
            cands = [u for u in self.by_short.get(name, []) if not u.parent and _matches_hint(u, hint)]
            if not cands and name in self.classes:
                cands = [c for c in self.classes[name] if _matches_hint(c, hint)]
                if cands:
                    return [(cands[0], "instantiates", "import")]
            if cands:
                return [(cands[0], "calls", "import")]
        if target_cls:
            found = self._method_with_origin(target_cls, name, caller)
            if found is not None:
                return [(found[0], "calls", "inherited" if found[1] and resolution == "self" else resolution)]
        if receiver == "?" or head in imports:
            return []
        return self._by_name(caller, name)

    def _chain_type(self, parts: list[str], cls: str | None, self_names: set[str],
                    locals_: dict[str, str]) -> tuple[str | None, str]:
        """Type of a receiver chain such as self.payments, service.repo or repo."""
        head, rest = parts[0], parts[1:]
        if head in self_names and cls:
            t, res = cls, "self"
        elif head in locals_:
            t, res = locals_[head], "local_type"
        elif cls and head in self.fields.get(cls, {}):
            t, res = self.fields[cls][head], "field_type"
        elif head in self.classes:
            t, res = head, "class_name"
        else:
            return None, ""
        for attr in rest:
            t = self.fields.get(t, {}).get(attr)
            res = "field_type"
            if t is None:
                return None, ""
        return t, res

    def _by_name(self, caller: CodeUnit, name: str) -> list[tuple[CodeUnit, str, str]]:
        if name in _GENERIC:
            return []
        cands = [u for u in self.by_short.get(name, []) if u.id != caller.id]
        if len(cands) == 1:
            return [(cands[0], "calls", "name_match")]
        if 1 < len(cands) <= 3:
            return [(c, "calls", "ambiguous") for c in cands]
        return []


def _norm_path(p: str) -> str:
    p = re.sub(r"\$\{[^}]*\}|\{[^}]*\}|<[^>]*>|:[A-Za-z_]\w*|%[sd]", "{}", p)
    p = p.split("?", 1)[0].rstrip("/")
    return p or "/"


_STR_RE = re.compile(r"[fFrRbB]?(['\"`])((?:(?!\1).){1,240})\1")


def _cross_service_edges(units_by_service: dict[str, list[CodeUnit]], entry: list[dict],
                         deps: list[Dependency], lines_of) -> list[dict]:
    http_targets: dict[str, set[str]] = defaultdict(set)
    for d in deps:
        if d.kind == "http":
            http_targets[d.source].add(d.target)
    by_service: dict[str, list[dict]] = defaultdict(list)
    for ep in entry:
        if ep.get("handler") and ep.get("path") and _norm_path(ep["path"]) not in ("/", ""):
            by_service[ep["service"]].append({**ep, "norm": _norm_path(ep["path"])})
    edges = []
    for src_svc, targets in http_targets.items():
        eps = [e for t in targets for e in by_service.get(t, [])]
        if not eps:
            continue
        for u in units_by_service.get(src_svc, []):
            if u.kind not in CALLER_KINDS:
                continue
            for lineno, line in lines_of(u):
                for m in _STR_RE.finditer(line):
                    lit = m.group(2)
                    if "/" not in lit:
                        continue
                    path = lit
                    if "://" in path:
                        path = "/" + path.split("://", 1)[1].partition("/")[2]
                    path = _norm_path(path[path.find("/"):] if "/" in path else path)
                    if path == "/":
                        continue
                    verb = next((v.upper() for v in _VERBS if re.search(rf"\b{v}\b", line, re.I)), None)
                    hits = [e for e in eps if path == e["norm"] or (path.endswith(e["norm"]) and e["norm"].count("/") >= 2)]
                    if verb:
                        hits = [e for e in hits if e["method"].upper() in (verb, "ANY", "")] or []
                    for e in hits[:2]:
                        edges.append({"source": u.id, "target": e["handler"], "kind": "http",
                                      "resolution": "endpoint_path", "confidence": 0.8 if verb else 0.6,
                                      "line": lineno, "detail": f"{e['method']} {e['path']} -> {e['service']}"})
    return edges


def _structural_edges(index: dict[str, _ServiceIndex]) -> tuple[list[dict], set[str]]:
    """inherits / implements between classes, imports from files to classes, functions or files."""
    edges: list[dict] = []
    file_nodes: set[str] = set()
    for si in index.values():
        for cls, pairs in si.bases.items():
            for cu in si.classes[cls]:
                for base, kind in pairs:
                    edges.append({"source": cu.id, "target": base.id, "kind": kind, "resolution": "declaration",
                                  "confidence": 0.95, "line": cu.start_line})
        paths = set(si.paths)
        for path, imports in si.imports.items():
            src = f"file:{path}"
            for name, hint in imports.items():
                cands = [u for u in si.classes.get(name, []) + [f for f in si.by_short.get(name, []) if not f.parent]
                         if _matches_hint(u, hint)]
                if cands:
                    edges.append({"source": src, "target": cands[0].id, "kind": "imports", "resolution": "import",
                                  "confidence": 0.95, "line": 1})
                    file_nodes.add(src)
                    continue
                targets = [p for p in paths if p != path and (_stem(p) == hint or posixpath.dirname(p).endswith("/" + hint)
                                                              or posixpath.dirname(p) == hint)]
                for t in targets[:10]:
                    edges.append({"source": src, "target": f"file:{t}", "kind": "imports", "resolution": "import",
                                  "confidence": 0.95, "line": 1})
                    file_nodes.update((src, f"file:{t}"))
    return edges, file_nodes


_PUB_CALL = re.compile(r"\.(?:send|produce|publish|emit|WriteMessages|convertAndSend|basic_publish|Produce|"
                       r"send_and_wait|sendMessage|PublishMessage)\w*\s*\(", re.I)
_CONSUME = re.compile(r"for\s+\w+\s+in\s+[\w.]*consumer|\.poll\s*\(|ReadMessage|FetchMessage|\.consume\s*\(|"
                      r"\.on\(\s*['\"]message|@KafkaListener|@RabbitListener|\.subscribe\s*\(|basic_consume|"
                      r"KafkaConsumer\s*\(|\.run\(\s*\{\s*eachMessage", re.I)


def _is_ctor(u: CodeUnit) -> bool:
    short = _short(u.name)
    return short in ("__init__", "constructor", "init") or short.startswith("New") or (
        u.parent is not None and short == _short(u.parent))


def _async_edges(units_by_service: dict[str, list[CodeUnit]], files: dict[str, SourceFile]) -> tuple[list[dict], list[dict]]:
    """Publisher function -> consumer function for every message topic (Kafka, RabbitMQ, ...)."""
    by_file: dict[str, list[CodeUnit]] = defaultdict(list)
    for us in units_by_service.values():
        for u in us:
            by_file[u.file_path].append(u)
    pubs: dict[str, dict[str, int]] = defaultdict(dict)
    subs: dict[str, dict[str, int]] = defaultdict(dict)
    unit_by_id: dict[str, CodeUnit] = {}
    for path, us in by_file.items():
        sf = files.get(path)
        if sf is None or sf.kind != "code":
            continue
        funcs = [u for u in us if u.kind in ("function", "method")]
        for t in extract_topics(sf.text):
            line, pub = t["line"], t["direction"] == "pub"
            inside = sorted((u for u in funcs if u.start_line <= line <= u.end_line), key=lambda u: u.end_line - u.start_line)
            below = sorted((u for u in funcs if 0 < u.start_line - line <= 3), key=lambda u: u.start_line)
            unit = inside[0] if inside else (below[0] if below else None)
            pattern = _PUB_CALL if pub else _CONSUME
            if unit is None or _is_ctor(unit):
                # topic set in a constant or a constructor: use the functions that actually publish/consume
                cands = [f for f in funcs if pattern.search(f.code) and not _is_ctor(f)]
                if unit is not None and unit.parent:
                    cands = [c for c in cands if c.parent == unit.parent] or cands
                targets = cands or ([unit] if unit else [])
            else:
                targets = [unit]
            for u in targets:
                (pubs if pub else subs)[t["topic"]].setdefault(u.id, line if u is unit else u.start_line)
                unit_by_id[u.id] = u
    edges, entry = [], []
    for topic, consumers in subs.items():
        for cid, line in consumers.items():
            c = unit_by_id[cid]
            entry.append({"service": c.service, "method": "CONSUME", "path": topic, "file": c.file_path,
                          "line": line, "handler": cid, "via": "topic"})
        for pid, line in pubs.get(topic, {}).items():
            for cid in consumers:
                if pid != cid:
                    edges.append({"source": pid, "target": cid, "kind": "async", "resolution": "topic",
                                  "confidence": 0.75, "line": line, "detail": f"topic {topic}"})
    return edges, entry


def _entry_points(services: list[Service], units: list[CodeUnit], index: dict[str, _ServiceIndex],
                  files: dict[str, SourceFile]) -> list[dict]:
    """Map each HTTP endpoint to the function that handles it."""
    by_file: dict[str, list[CodeUnit]] = defaultdict(list)
    for u in units:
        by_file[u.file_path].append(u)
    out = []
    for svc in services:
        si = index.get(svc.name)
        for ep in svc.endpoints:
            f, line = ep.get("file"), ep.get("line")
            if not f or not line or ep.get("framework") in ("grpc", "openapi"):
                continue
            sf = files.get(f)
            text_lines = sf.text.splitlines() if sf else []
            text_line = text_lines[line - 1].strip() if 0 < line <= len(text_lines) else ""
            funcs = [u for u in by_file.get(f, []) if u.kind in ("function", "method")]
            inside = sorted((u for u in funcs if u.start_line <= line <= u.end_line),
                            key=lambda u: u.end_line - u.start_line)
            below = sorted((u for u in funcs if 0 < u.start_line - line <= 3), key=lambda u: u.start_line)
            module = next((u for u in by_file.get(f, []) if u.kind == "module"), None)
            container = inside[0] if inside else module
            inline = "=>" in text_line or bool(re.search(r"\bfunc\s*\(|\bfunction\s*\(|\blambda\b", text_line))
            handler, via = None, ""
            if text_line.startswith(("@", "[")):            # decorator / annotation / attribute
                if inside and inside[0].start_line <= line < inside[0].start_line + 6:
                    handler, via = inside[0], "decorated"
                elif below:
                    handler, via = below[0], "annotated"
            elif inline:                                      # router.post('/x', (req, res) => ...)
                handler, via = container, "inline"
            elif si is not None and container is not None:     # mux.HandleFunc("/x", h.Charge)
                tail = text_line.split(",", 1)[1] if "," in text_line else ""
                refs = re.findall(r"([A-Za-z_][\w.]*)\s*[,)]", tail)
                locals_ = local_types(container.code, container.language, si.known)
                for ref in reversed(refs):
                    parts = ref.split(".")
                    res = si.resolve(container, ".".join(parts[:-1]) or None, parts[-1], locals_)
                    if res:
                        handler, via = res[0][0], "reference"
                        break
            if handler is None and below:
                handler, via = below[0], "annotated"
            out.append({"service": svc.name, "method": ep.get("method", ""), "path": ep.get("path", ""),
                        "file": f, "line": line, "handler": handler.id if handler else None, "via": via})
    return out


def build_call_graph(units: list[CodeUnit], services: list[Service], files: list[SourceFile],
                     deps: list[Dependency]) -> dict:
    file_map = {f.path: f for f in files}
    units_by_service: dict[str, list[CodeUnit]] = defaultdict(list)
    for u in units:
        if u.kind in CALLER_KINDS | TARGET_KINDS:
            units_by_service[u.service].append(u)
    index = {svc: _ServiceIndex(us, file_map) for svc, us in units_by_service.items()}

    merged: dict[tuple[str, str, str], dict] = {}

    def add(edge: dict) -> None:
        key = (edge["source"], edge["target"], edge["kind"])
        prev = merged.get(key)
        if prev is None:
            merged[key] = {**edge, "count": 1}
        else:
            prev["count"] += 1
            if edge["confidence"] > prev["confidence"]:
                prev.update({k: edge[k] for k in ("confidence", "resolution", "line")})

    covered: dict[str, set[int]] = defaultdict(set)
    for u in units:
        if u.kind in ("function", "method", "class"):
            covered[u.file_path].update(range(u.start_line, u.end_line + 1))

    def lines_of(u: CodeUnit):
        sf = file_map.get(u.file_path)
        return unit_lines(u, sf.text if sf else None, covered.get(u.file_path))

    module_types: dict[str, dict[str, str]] = {}
    for svc, si in index.items():
        for u in si.units:
            if u.kind == "module":
                module_types[u.file_path] = local_types(u.code, u.language, si.known)

    unresolved = 0
    for svc, si in index.items():
        for u in si.units:
            if u.kind not in CALLER_KINDS:
                continue
            locals_ = {**module_types.get(u.file_path, {}), **local_types(u.code, u.language, si.known)}
            for receiver, name, line in call_sites(u, lines_of(u)):
                targets = si.resolve(u, receiver, name, locals_)
                if not targets:
                    unresolved += 1
                for target, kind, resolution in targets:
                    if target.id == u.id and resolution in ("name_match", "ambiguous"):
                        continue
                    add({"source": u.id, "target": target.id, "kind": kind, "resolution": resolution,
                         "confidence": CONFIDENCE[resolution], "line": line})

    entry = _entry_points(services, units, index, file_map)
    for e in _cross_service_edges(units_by_service, entry, deps, lines_of):
        add(e)
    async_edges, consumers = _async_edges(units_by_service, file_map)
    for e in async_edges:
        add(e)
    entry += consumers
    structural, file_nodes = _structural_edges(index)
    for e in structural:
        add(e)

    edges = sorted(merged.values(), key=lambda e: (e["source"], -e["confidence"], e["line"]))
    used = {e["source"] for e in edges} | {e["target"] for e in edges} | {e["handler"] for e in entry if e["handler"]}
    nodes = {u.id: {"name": u.name, "kind": u.kind, "service": u.service, "file": u.file_path,
                    "start_line": u.start_line, "end_line": u.end_line}
             for us in units_by_service.values() for u in us if u.id in used or u.kind in ("function", "method")}
    file_service = {f.path: f.service for f in files}
    for fid in sorted(file_nodes):
        path = fid[len("file:"):]
        nodes[fid] = {"name": path, "kind": "file", "service": file_service.get(path, ""), "file": path,
                      "start_line": 1, "end_line": 1}
    edges = [e for e in edges if e["source"] in nodes and e["target"] in nodes]
    svc_of = {k: v["service"] for k, v in nodes.items()}
    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "edge_kinds": dict(Counter(e["kind"] for e in edges)),
        "cross_service_edges": sum(1 for e in edges if svc_of[e["source"]] != svc_of[e["target"]]),
        "low_confidence_edges": sum(1 for e in edges if e["confidence"] < DEFAULT_MIN_CONFIDENCE),
        "entry_points": len(entry),
        "unresolved_calls": unresolved,
    }
    return {"nodes": nodes, "edges": edges, "entry_points": entry, "stats": stats}


# ------------------------------------------------------------------ queries

class CallGraph:
    def __init__(self, data: dict | None):
        data = data or {}
        self.nodes: dict[str, dict] = data.get("nodes", {})
        self.edges: list[dict] = data.get("edges", [])
        self.entry_points: list[dict] = data.get("entry_points", [])
        self.stats: dict = data.get("stats", {})
        self.out: dict[str, list[dict]] = defaultdict(list)
        self.inc: dict[str, list[dict]] = defaultdict(list)
        for e in self.edges:
            self.out[e["source"]].append(e)
            self.inc[e["target"]].append(e)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def find(self, symbol: str, service: str | None = None) -> list[str]:
        """Resolve 'place_order', 'OrderService.place_order', 'Handler.Charge' or a full unit id."""
        if symbol in self.nodes:
            return [symbol]
        s = symbol.strip().rstrip("()").lower()
        pool = [(uid, n) for uid, n in self.nodes.items()
                if n["kind"] != "file" and (service is None or n["service"] == service)]
        for test in ((lambda n: n["name"].lower() == s),
                     (lambda n: n["name"].lower().endswith("." + s)),
                     (lambda n: _short(n["name"]).lower() == _short(s))):
            found = [uid for uid, n in pool if test(n)]
            if found:
                return sorted(found, key=lambda uid: (self.nodes[uid]["kind"] == "class", uid))
        return []

    def describe(self, uid: str) -> dict:
        n = self.nodes.get(uid, {})
        return {"id": uid, "name": n.get("name", uid), "kind": n.get("kind"), "service": n.get("service"),
                "file": n.get("file"), "line": n.get("start_line")}

    def _walk(self, uid: str, direction: str, depth: int, min_conf: float,
              kinds: tuple[str, ...] = CALL_KINDS) -> list[dict]:
        adj = self.out if direction == "out" else self.inc
        seen = {uid}
        result = []
        frontier = deque([(uid, 0)])
        while frontier:
            cur, d = frontier.popleft()
            if d >= depth:
                continue
            for e in adj.get(cur, []):
                if e["confidence"] < min_conf or e["kind"] not in kinds:
                    continue
                nxt = e["target"] if direction == "out" else e["source"]
                if nxt in seen:
                    continue
                seen.add(nxt)
                result.append({**self.describe(nxt), "depth": d + 1, "from": cur, "kind_of_edge": e["kind"],
                               "resolution": e["resolution"], "confidence": e["confidence"],
                               "call_line": e["line"], "call_file": self.nodes.get(e["source"], {}).get("file"),
                               **({"detail": e["detail"]} if e.get("detail") else {})})
                frontier.append((nxt, d + 1))
        return result

    def callees(self, uid: str, depth: int = 1, min_conf: float = DEFAULT_MIN_CONFIDENCE,
                kinds: tuple[str, ...] = CALL_KINDS) -> list[dict]:
        return self._walk(uid, "out", depth, min_conf, kinds)

    def callers(self, uid: str, depth: int = 1, min_conf: float = DEFAULT_MIN_CONFIDENCE,
                kinds: tuple[str, ...] = CALL_KINDS) -> list[dict]:
        return self._walk(uid, "in", depth, min_conf, kinds)

    def related(self, uid: str, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> dict:
        """Structural neighbours: base classes, subclasses, implementations, imports."""
        out: dict[str, list[dict]] = defaultdict(list)
        for e in self.out.get(uid, []):
            if e["kind"] in STRUCTURAL_KINDS and e["confidence"] >= min_conf:
                out[{"inherits": "extends", "implements": "implements", "imports": "imports"}[e["kind"]]].append(
                    self.describe(e["target"]))
        for e in self.inc.get(uid, []):
            if e["kind"] in STRUCTURAL_KINDS and e["confidence"] >= min_conf:
                out[{"inherits": "subclasses", "implements": "implemented_by", "imports": "imported_by"}[e["kind"]]].append(
                    self.describe(e["source"]))
        return dict(out)

    def path(self, src: str, dst: str, max_depth: int = 10, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> list[dict]:
        """Shortest call chain src -> ... -> dst (empty if none)."""
        prev: dict[str, dict | None] = {src: None}
        queue = deque([(src, 0)])
        while queue:
            cur, d = queue.popleft()
            if cur == dst:
                break
            if d >= max_depth:
                continue
            for e in self.out.get(cur, []):
                if e["confidence"] >= min_conf and e["kind"] in CALL_KINDS and e["target"] not in prev:
                    prev[e["target"]] = e
                    queue.append((e["target"], d + 1))
        if dst not in prev:
            return []
        hops = []
        cur = dst
        while prev[cur] is not None:
            e = prev[cur]
            hops.append({"from": self.describe(e["source"]), "to": self.describe(e["target"]), "kind": e["kind"],
                         "resolution": e["resolution"], "confidence": e["confidence"], "line": e["line"],
                         **({"detail": e["detail"]} if e.get("detail") else {})})
            cur = e["source"]
        return list(reversed(hops))

    # -------------------------------------------------------------- impact of a change
    def units_at(self, file: str, lines: set[int] | None = None) -> list[str]:
        """Innermost function/method/class covering each changed line (all of them if lines is None)."""
        units = [(uid, n) for uid, n in self.nodes.items()
                 if n["file"] == file and n["kind"] in ("function", "method", "class", "module")]
        if lines is None:
            return [uid for uid, n in units if n["kind"] != "module"] or [uid for uid, _ in units]
        hit: list[str] = []
        for line in sorted(lines):
            covering = [(n["end_line"] - n["start_line"], uid) for uid, n in units
                        if n["kind"] != "module" and n["start_line"] <= line <= n["end_line"]]
            if not covering:
                covering = [(10 ** 9, uid) for uid, n in units if n["kind"] == "module"]
            if covering:
                uid = min(covering)[1]
                if uid not in hit:
                    hit.append(uid)
        return hit

    def affected(self, uids: list[str], depth: int = 8, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> dict:
        """Everything that may break when ``uids`` change: callers (transitively, also over HTTP),
        subclasses and implementations, and consumers of messages the changed code publishes."""
        dependents: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for e in self.edges:
            if e["confidence"] < min_conf or e["kind"] == "imports":
                continue
            if e["kind"] == "async":
                dependents[e["source"]].append((e["target"], e))   # consumers depend on what is published
            else:
                dependents[e["target"]].append((e["source"], e))   # callers / subclasses depend on the callee
        start = [u for u in uids if u in self.nodes]
        dist = {u: 0 for u in start}
        via: dict[str, tuple[str, dict]] = {}
        queue = deque(start)
        while queue:
            cur = queue.popleft()
            if dist[cur] >= depth:
                continue
            for nxt, e in dependents.get(cur, []):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    via[nxt] = (cur, e)
                    queue.append(nxt)
        affected = []
        for uid, d in sorted(dist.items(), key=lambda kv: (kv[1], kv[0])):
            if d == 0:
                continue
            src, e = via[uid]
            affected.append({**self.describe(uid), "distance": d, "because_of": self.describe(src)["name"],
                             "edge": e["kind"], "confidence": e["confidence"], "test": is_test_file(self.nodes[uid]["file"])})
        hit = set(dist)
        services: dict[str, int] = Counter(a["service"] for a in affected)

        # Several inline route handlers can share one file-level node; only count the routes whose
        # handler code (the lines up to the next route in that file) reaches the changed code.
        inline_lines: dict[str, list[int]] = defaultdict(list)
        for ep in self.entry_points:
            if ep.get("via") == "inline":
                inline_lines[ep["file"]].append(ep["line"])

        def reached(ep: dict) -> bool:
            h = ep.get("handler")
            if h not in hit:
                return False
            if ep.get("via") != "inline" or dist.get(h) == 0:
                return True
            nxt = min([l for l in inline_lines[ep["file"]] if l > ep["line"]] or [10 ** 9])
            lines = [e["line"] for e in self.out.get(h, []) if e["target"] in hit and e["kind"] != "imports"]
            return any(ep["line"] <= l < nxt for l in lines)

        return {
            "changed": [self.describe(u) for u in start],
            "affected": affected,
            "services": dict(services),
            "cross_service": sorted({a["service"] for a in affected} - {c["service"] for c in map(self.describe, start)}),
            "entry_points": [e for e in self.entry_points if reached(e)],
            "tests": [a for a in affected if a["test"]],
        }

    # -------------------------------------------------------------- god nodes
    def god_nodes(self, top: int = 10, service: str | None = None, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> list[dict]:
        """The most connected functions/classes: many callers (fan-in), many callees (fan-out),
        callers in several services. Changes to them have the widest blast radius."""
        fan_in: dict[str, set[str]] = defaultdict(set)
        fan_out: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            if e["kind"] in CALL_KINDS and e["confidence"] >= min_conf:
                fan_out[e["source"]].add(e["target"])
                fan_in[e["target"]].add(e["source"])
        rows = []
        for uid, n in self.nodes.items():
            if n["kind"] in ("file", "module") or (service and n["service"] != service):
                continue
            callers = fan_in.get(uid, set())
            caller_services = {self.nodes[c]["service"] for c in callers if c in self.nodes}
            fi, fo = len(callers), len(fan_out.get(uid, set()))
            if fi + fo == 0:
                continue
            score = 2 * fi + fo + 3 * max(0, len(caller_services - {n["service"]}))
            rows.append({**self.describe(uid), "fan_in": fi, "fan_out": fo, "score": score,
                         "caller_services": sorted(caller_services),
                         "transitive_dependents": 0})
        rows.sort(key=lambda r: (-r["score"], r["name"]))
        rows = rows[:top]
        for r in rows:
            r["transitive_dependents"] = len(self.affected([r["id"]], min_conf=min_conf)["affected"])
        return rows
