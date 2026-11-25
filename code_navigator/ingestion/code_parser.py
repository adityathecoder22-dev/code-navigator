"""Split source files into code units (classes, functions, methods, module headers).

* Python  -> built-in `ast` module (exact)
* Others  -> tree-sitter via `tree-sitter-language-pack` (exact)
* Fallback -> regex + brace matching, if tree-sitter is not installed or fails
* Markdown -> one unit per heading section
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from ..models import CodeUnit, SourceFile
from ..utils import log

CLASS_COMPACT_LINES = 60
MAX_MODULE_LINES = 300

KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "func", "fn", "def", "class", "new",
    "sizeof", "typeof", "await", "async", "elif", "else", "try", "with", "lambda", "print", "super",
    "this", "self", "assert", "throw", "throws", "yield", "match", "case", "when", "using", "lock",
    "foreach", "select", "go", "defer", "make", "len", "append", "range", "panic", "int", "str", "list",
    "dict", "set", "tuple", "string", "bool", "float", "byte", "rune", "error", "and", "or", "not", "in",
}


@dataclass
class _Def:
    kind: str           # class | function | method
    name: str           # qualified, e.g. OrderService.place_order
    start: int          # 0-based first line (includes decorators/annotations)
    end: int            # 0-based last line (inclusive)
    signature: str = ""
    parent: str | None = None
    docstring: str = ""
    calls: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ helpers

def extract_calls(code: str, limit: int = 40) -> list[str]:
    seen: list[str] = []
    for m in re.finditer(r"(?:\b|\.)([A-Za-z_]\w*)\s*\(", code):
        name = m.group(1)
        if name in KEYWORDS or name in seen:
            continue
        seen.append(name)
        if len(seen) >= limit:
            break
    return seen


def leading_comment(lines: list[str], start: int, max_lines: int = 12) -> str:
    """Collect the comment block immediately above `start` (Javadoc, //, #, ///)."""
    out: list[str] = []
    i = start - 1
    while i >= 0 and len(out) < max_lines:
        s = lines[i].strip()
        if s.startswith(("//", "/*", "*", "*/", "#", "///", "--")) and not s.startswith("#include"):
            cleaned = re.sub(r"^(/\*\*?|\*/|\*|///?|#|--)\s?", "", s).rstrip("*/").strip()
            if cleaned and not cleaned.startswith("@"):
                out.append(cleaned)
            i -= 1
        elif s.startswith("@") or s.startswith("["):  # annotations / attributes between comment and def
            i -= 1
        else:
            break
    return " ".join(reversed(out))[:600]


def _compact_class(lines: list[str], d: _Def, members: list[str]) -> str:
    body = lines[d.start : d.end + 1]
    if len(body) <= CLASS_COMPACT_LINES:
        return "\n".join(body)
    head = "\n".join(body[:25])
    member_list = ", ".join(members[:60])
    return f"{head}\n    ...\n    // members: {member_list}\n" + (body[-1] if body else "")


# ------------------------------------------------------------------ python (ast)

def _python_defs(text: str) -> list[_Def]:
    tree = ast.parse(text)
    lines = text.splitlines()
    defs: list[_Def] = []

    def calls_of(node) -> list[str]:
        names: list[str] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if nm and nm not in names and nm not in KEYWORDS:
                    names.append(nm)
        return names[:40]

    def visit(body, parent: str | None):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                end = (getattr(node, "end_lineno", None) or node.lineno) - 1
                qual = f"{parent}.{node.name}" if parent else node.name
                sig_line = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else node.name
                doc = ast.get_docstring(node) or ""
                if isinstance(node, ast.ClassDef):
                    defs.append(_Def("class", qual, start, end, sig_line, parent, doc, []))
                    visit(node.body, qual)
                else:
                    kind = "method" if parent else "function"
                    defs.append(_Def(kind, qual, start, end, sig_line, parent, doc, calls_of(node)))

    visit(tree.body, None)
    return defs


# ------------------------------------------------------------------ tree-sitter

TS_SPEC: dict[str, tuple[set[str], set[str]]] = {
    "java": ({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
             {"method_declaration", "constructor_declaration"}),
    "go": ({"type_spec"}, {"function_declaration", "method_declaration"}),
    "javascript": ({"class_declaration"}, {"function_declaration", "generator_function_declaration", "method_definition"}),
    "typescript": ({"class_declaration", "abstract_class_declaration", "interface_declaration"},
                   {"function_declaration", "generator_function_declaration", "method_definition"}),
    "csharp": ({"class_declaration", "interface_declaration", "struct_declaration", "record_declaration", "enum_declaration"},
               {"method_declaration", "constructor_declaration"}),
    "kotlin": ({"class_declaration", "object_declaration"}, {"function_declaration"}),
    "rust": ({"struct_item", "enum_item", "trait_item", "impl_item"}, {"function_item"}),
    "ruby": ({"class", "module"}, {"method", "singleton_method"}),
    "php": ({"class_declaration", "interface_declaration", "trait_declaration"}, {"function_definition", "method_declaration"}),
    "scala": ({"class_definition", "object_definition", "trait_definition"}, {"function_definition"}),
}
TS_SPEC["tsx"] = TS_SPEC["typescript"]
JS_LIKE = {"javascript", "typescript", "tsx"}
NAME_NODE_TYPES = ("identifier", "type_identifier", "simple_identifier", "constant", "property_identifier", "name")

_parser_cache: dict[str, object] = {}
_ts_disabled = False


def _get_ts_parser(language: str):
    global _ts_disabled
    if _ts_disabled:
        return None
    if language in _parser_cache:
        return _parser_cache[language]
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
    except ImportError:
        log.warning("tree-sitter-language-pack not installed; using regex parsers for non-Python code")
        _ts_disabled = True
        return None
    try:
        parser = get_parser(language)
    except Exception as exc:  # unsupported language or binary issue
        log.debug("No tree-sitter parser for %s: %s", language, exc)
        parser = None
    _parser_cache[language] = parser
    return parser


def _ts_defs(text: str, language: str) -> list[_Def] | None:
    spec = TS_SPEC.get(language)
    parser = _get_ts_parser(language) if spec else None
    if parser is None:
        return None
    src = text.encode("utf-8")
    tree = parser.parse(src)
    class_types, func_types = spec
    lines = text.splitlines()

    def node_text(n) -> str:
        return src[n.start_byte : n.end_byte].decode("utf-8", errors="replace")

    def node_name(n) -> str | None:
        target = n.child_by_field_name("name")
        if target is None and n.type == "impl_item":
            target = n.child_by_field_name("type")
        if target is None:
            for c in n.children:
                if c.type in NAME_NODE_TYPES:
                    target = c
                    break
        return node_text(target).strip() if target is not None else None

    def line_of(n) -> int:
        return n.start_point[0]

    def span(n) -> tuple[int, int]:
        start_node = n
        parent = n.parent
        if parent is not None and parent.type in ("export_statement", "decorated_definition", "type_declaration"):
            start_node = parent
        return start_node.start_point[0], n.end_point[0]

    defs: list[_Def] = []
    stack = [(tree.root_node, None)]
    while stack:
        node, cls = stack.pop()
        t = node.type
        if t in class_types:
            name = node_name(node) or "<anonymous>"
            qual = f"{cls}.{name}" if cls else name
            s, e = span(node)
            defs.append(_Def("class", qual, s, e, lines[line_of(node)].strip() if line_of(node) < len(lines) else qual, cls))
            for child in reversed(node.children):
                stack.append((child, qual))
            continue
        if t in func_types:
            name = node_name(node) or "<anonymous>"
            parent = cls
            if language == "go" and t == "method_declaration":
                recv = node.child_by_field_name("receiver")
                m = re.search(r"(\w+)\s*\)\s*$", node_text(recv)) if recv is not None else None
                parent = m.group(1) if m else parent
            qual = f"{parent}.{name}" if parent else name
            s, e = span(node)
            name_node = node.child_by_field_name("name")
            sig_row = name_node.start_point[0] if name_node is not None else line_of(node)
            defs.append(_Def("method" if parent else "function", qual, s, e,
                             lines[sig_row].strip() if sig_row < len(lines) else qual, parent))
            continue
        if language in JS_LIKE and t == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                name = node_name(node) or "<anonymous>"
                qual = f"{cls}.{name}" if cls else name
                decl = node.parent if node.parent is not None else node
                s, e = span(decl)
                defs.append(_Def("method" if cls else "function", qual, s, e,
                                 lines[line_of(node)].strip() if line_of(node) < len(lines) else qual, cls))
                continue
        for child in reversed(node.children):
            stack.append((child, cls))
    return defs


# ------------------------------------------------------------------ regex fallback

_RX = {
    "go": [
        ("function", re.compile(r"^func\s+(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)?(\w+)\s*[\[(]")),
        ("class", re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\b")),
    ],
    "js": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|\w+\s*=>)")),
        ("method", re.compile(r"^\s+(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*(\w+)\s*\([^)]*\)\s*(?::[^{]+)?\{\s*$")),
    ],
    "jvm": [
        ("class", re.compile(r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|abstract|final|static|sealed|partial|data|open|enum)\s+)*(?:class|interface|enum|record|object|struct)\s+(\w+)")),
        ("method", re.compile(r"^\s+(?:(?:public|private|protected|internal|static|final|async|override|virtual|abstract|synchronized|suspend|open)\s+)+(?:fun\s+)?[\w<>\[\],.?\s]*?\b(\w+)\s*\([^;]*$")),
        ("method", re.compile(r"^\s*(?:(?:private|public|internal|override|suspend|open)\s+)*fun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?(\w+)\s*\(")),
    ],
}
_RX_LANG = {"go": "go", "javascript": "js", "typescript": "js", "tsx": "js",
            "java": "jvm", "csharp": "jvm", "kotlin": "jvm", "scala": "jvm", "php": "jvm", "rust": "go"}
_NOT_METHODS = {"if", "for", "while", "switch", "catch", "return", "else", "new", "throw", "using", "lock", "foreach"}


def _brace_end(lines: list[str], start: int) -> int:
    depth = 0
    opened = False
    for i in range(start, len(lines)):
        line = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//.*$', "", lines[i])
        for ch in line:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i
        if not opened and i - start > 3:
            return start  # expression-bodied / declaration only
    return len(lines) - 1


def _regex_defs(text: str, language: str) -> list[_Def]:
    family = _RX_LANG.get(language)
    if not family:
        return []
    lines = text.splitlines()
    defs: list[_Def] = []
    class_stack: list[_Def] = []
    for i, line in enumerate(lines):
        while class_stack and i > class_stack[-1].end:
            class_stack.pop()
        for kind, rx in _RX[family]:
            m = rx.match(line)
            if not m:
                continue
            if family == "go" and kind == "function":
                recv, name = m.group(1), m.group(2)
                parent = recv
            else:
                name = m.group(1)
                parent = class_stack[-1].name if class_stack and kind != "class" else None
                if kind == "class" and class_stack:
                    parent = class_stack[-1].name
            if name in _NOT_METHODS:
                continue
            if kind == "method" and not parent and family != "js":
                kind = "function"
            if kind == "method" and family == "js" and not parent:
                continue
            end = _brace_end(lines, i)
            qual = f"{parent}.{name}" if parent else name
            start = i
            while start > 0 and lines[start - 1].strip().startswith(("@", "[")):
                start -= 1
            d = _Def("class" if kind == "class" else ("method" if parent else "function"),
                     qual, start, end, line.strip(), parent)
            defs.append(d)
            if kind == "class":
                class_stack.append(d)
            break
    return defs


# ------------------------------------------------------------------ markdown

def parse_markdown(sf: SourceFile) -> list[CodeUnit]:
    lines = sf.text.splitlines()
    heads = [i for i, l in enumerate(lines) if re.match(r"^#{1,3}\s+\S", l)]
    if not heads or heads[0] != 0:
        heads = [0] + heads
    units = []
    for idx, start in enumerate(heads):
        end = (heads[idx + 1] - 1) if idx + 1 < len(heads) else len(lines) - 1
        body = "\n".join(lines[start : end + 1]).strip()
        if len(body) < 40:
            continue
        title = lines[start].lstrip("#").strip() if lines[start].startswith("#") else sf.path
        units.append(CodeUnit(
            id=f"{sf.path}::doc#{start + 1}", service=sf.service, file_path=sf.path, language="markdown",
            kind="doc", name=title[:120], signature=title[:200], code=body,
            start_line=start + 1, end_line=end + 1,
        ))
    return units


# ------------------------------------------------------------------ main entry

def parse_code_file(sf: SourceFile) -> list[CodeUnit]:
    if sf.kind == "doc":
        return parse_markdown(sf)
    if sf.kind != "code":
        return []

    text = sf.text
    lines = text.splitlines()
    defs: list[_Def] | None = None
    parser_used = "regex"
    try:
        if sf.language == "python":
            defs = _python_defs(text)
            parser_used = "ast"
        else:
            defs = _ts_defs(text, sf.language)
            if defs is not None:
                parser_used = "tree-sitter"
    except SyntaxError:
        defs = None
    except Exception as exc:  # never let one file break indexing
        log.debug("Parser failed on %s: %s", sf.path, exc)
        defs = None
    if defs is None:
        defs = _regex_defs(text, sf.language)

    defs.sort(key=lambda d: (d.start, -d.end))
    units: list[CodeUnit] = []
    covered = [False] * len(lines)

    members_by_class: dict[str, list[str]] = {}
    for d in defs:
        if d.parent:
            members_by_class.setdefault(d.parent, []).append(d.name.split(".")[-1])

    for d in defs:
        start, end = max(0, d.start), min(len(lines) - 1, max(d.start, d.end))
        if d.kind == "class":
            code = _compact_class(lines, _Def(d.kind, d.name, start, end), members_by_class.get(d.name, []))
        else:
            code = "\n".join(lines[start : end + 1])
        for i in range(start, end + 1):
            covered[i] = True
        doc = d.docstring or leading_comment(lines, start)
        calls = d.calls if d.calls or d.kind == "class" else extract_calls(code)
        units.append(CodeUnit(
            id=f"{sf.path}::{d.name}#{start + 1}", service=sf.service, file_path=sf.path,
            language=sf.language, kind=d.kind, name=d.name, signature=d.signature[:300],
            code=code, start_line=start + 1, end_line=end + 1, parent=d.parent,
            docstring=doc, calls=calls if d.kind != "class" else [],
        ))

    # Module unit: imports, top-level statements, route registrations, constants
    rest = [lines[i] for i in range(len(lines)) if not covered[i]]
    if sum(1 for l in rest if l.strip()) >= 3:
        module_code = "\n".join(rest[:MAX_MODULE_LINES]).strip()
        units.insert(0, CodeUnit(
            id=f"{sf.path}::<module>", service=sf.service, file_path=sf.path, language=sf.language,
            kind="module", name=sf.path.split("/")[-1], signature=f"module {sf.path}",
            code=module_code, start_line=1, end_line=len(lines), calls=extract_calls(module_code),
            docstring=leading_comment(lines, 0) if lines else "",
        ))
    log.debug("%s: %d units via %s", sf.path, len(units), parser_used)
    return units
