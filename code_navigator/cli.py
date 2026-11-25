"""Command-line interface.

    code-navigator index [PATH]                       build <PATH>/.code-navigator/
    code-navigator status                             what is indexed, with which models
    code-navigator ask "How does checkout work?"      cited answer (LLM) or ranked code (no LLM)
    code-navigator search "retry logic"               retrieval only
    code-navigator services | service NAME            service summaries, endpoints, dependencies
    code-navigator graph [--dot]                      service dependency graph
    code-navigator impact SERVICE                     who breaks if SERVICE changes or fails
    code-navigator tree [NODE_ID]                     browse the vectorless tree
    code-navigator calls SYMBOL [--depth N]           what a function calls / what calls it
    code-navigator path FROM TO                       shortest call chain (can cross services)
    code-navigator entrypoints [--service S]          HTTP endpoints / topics and their handler functions
    code-navigator affected SYMBOL... | --git [REF]   what may break if this changes
    code-navigator god-nodes                          the most connected functions
    code-navigator html [--open]                      interactive graph viewer
    code-navigator update [PATH]                      re-index only what changed
    code-navigator watch [PATH]                       re-index on every save
    code-navigator hook install|uninstall|status      re-index after git commit/merge/checkout
    code-navigator docs SERVICE [-o FILE]             Markdown docs for a service
    code-navigator check                              test LLM / embedding connectivity
    code-navigator mcp                                run the MCP server (stdio) for AI assistants

Query commands use the index of the current directory; pass -C/--repo PATH for another one.
Add --json for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings

# macOS system Python is built with LibreSSL; urllib3 warns about it on every run. Harmless here.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from . import __version__
from .config import INDEX_DIR_NAME, REPO_ENV_FILE, USER_ENV_FILE, get_settings
from .indexing.vector_index import redact
from .utils import setup_logging


def _dump(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _nav(args):
    from .engine import Navigator
    return Navigator(args.repo, index_dir=args.index_dir)


def _print_answer(res: dict) -> None:
    print("\n" + "=" * 80)
    print(res["answer"])
    print("=" * 80)
    nav = res.get("navigation")
    if nav:
        print(f"\nNavigation (scope={nav['scope']}, confidence={nav['confidence']:.2f}, llm calls={nav['llm_calls']}):")
        for step in nav["trace"]:
            print(f"  - {step['step']}: {', '.join(step['selected'][:8])}  [{step['method']}] {step['reason']}")
    print("\nSources:")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['service']} | {s['file']}:{s['start_line']}-{s['end_line']} | {s['name']} "
              f"(found by {', '.join(s['found_by'])})")
    for note in res.get("notes", []):
        print(f"  note: {note}")
    print(f"\nmode={res['mode']}  llm={res.get('llm') or 'none'}  timings={res['timings']}")


# ---------------------------------------------------------------------------- commands

def cmd_index(args) -> None:
    from .indexing.pipeline import build_index
    manifest = build_index(args.path, name=args.name, summary_mode=args.summaries, index_dir=args.out)
    if args.json:
        _dump(manifest)
        return
    s = manifest["stats"]
    print(f"\nIndexed '{manifest['project']}' -> {manifest['index_path']}")
    print(f"  services {s['services']} (+{s['infra']} infra) | files {s['files']} | chunks {s['chunks']} | "
          f"dependencies {s['dependencies']} | tree nodes {s['tree_nodes']}")
    print(f"  call graph: {s.get('call_edges', 0)} edges ({s.get('cross_service_calls', 0)} across services) | "
          f"entry points {s.get('entry_points', 0)}")
    inc = manifest.get("incremental") or {}
    if inc.get("previous_index"):
        n = len(inc.get("changed", [])) + len(inc.get("added", [])) + len(inc.get("removed", []))
        print(f"  incremental: {n} file(s) changed since the last index | embeddings reused {inc['embeddings_reused']}, "
              f"computed {inc['embeddings_computed']}")
    print(f"  viewer: {manifest['index_path']}/graph.html")
    print(f"  summaries: {manifest['summary_mode']}  | llm: {manifest['llm'] or 'none'}  | "
          f"embeddings: {manifest['embedding']['provider']}:{manifest['embedding']['model']}  | "
          f"{manifest['timings_sec']['total']}s")
    print("\nNext: code-navigator ask \"What does this system do?\"" +
          ("" if args.path in (".", "./") else f" -C {args.path}"))


def cmd_update(args) -> None:
    from .indexing.pipeline import update_index
    manifest = update_index(args.path, summary_mode=args.summaries, index_dir=args.out, force=args.force)
    if args.json:
        _dump(manifest)
    elif manifest.get("up_to_date"):
        if not args.quiet:
            print(f"Index of '{manifest['project']}' is up to date ({manifest['indexed_at']}).")
    elif not args.quiet:
        inc = manifest.get("incremental", {})
        print(f"Updated '{manifest['project']}': changed {len(inc.get('changed', []))}, added {len(inc.get('added', []))}, "
              f"removed {len(inc.get('removed', []))} file(s) | embeddings reused {inc.get('embeddings_reused', 0)}, "
              f"computed {inc.get('embeddings_computed', 0)} | {manifest['timings_sec']['total']}s")


def cmd_watch(args) -> None:
    from .indexing.pipeline import update_index
    from .indexing.watch import watch
    manifest = update_index(args.path, index_dir=args.out)
    print(f"Watching {manifest['repo_path']} (every {args.interval}s). Ctrl-C to stop.", flush=True)

    def on_change(paths: list[str]) -> None:
        shown = ", ".join(paths[:3]) + (f" (+{len(paths) - 3} more)" if len(paths) > 3 else "")
        try:
            m = update_index(args.path, index_dir=args.out)
        except Exception as exc:  # keep watching after a failed run
            print(f"  update failed: {exc}", flush=True)
            return
        if m.get("up_to_date"):
            print(f"  {shown}: no indexed content changed", flush=True)
        else:
            inc = m.get("incremental", {})
            print(f"  {shown}: re-indexed in {m['timings_sec']['total']}s "
                  f"(embeddings computed {inc.get('embeddings_computed', 0)})", flush=True)

    try:
        watch(manifest["repo_path"], on_change, interval=args.interval)
    except KeyboardInterrupt:
        print()


def cmd_hook(args) -> None:
    from .indexing.watch import hook_status, install_hooks, uninstall_hooks
    if args.action == "install":
        for path in install_hooks(args.path, args.index_dir):
            print(f"installed {path}")
        print("The index is now refreshed in the background after commit, merge, checkout and rebase.")
    elif args.action == "uninstall":
        removed = uninstall_hooks(args.path)
        print("\n".join(f"removed from {p}" for p in removed) or "no code-navigator hooks installed")
    else:
        for name, on in hook_status(args.path).items():
            print(f"{name:<14} {'installed' if on else '-'}")


def cmd_affected(args) -> None:
    if not args.symbols and not args.file and args.git is None:
        raise ValueError("give one or more symbols, --file PATH or --git [REF]")
    if args.git is not None and not args.no_update:
        from .indexing.pipeline import update_index
        update_index(args.repo, index_dir=args.index_dir)
    data = _nav(args).affected(args.symbols, args.file, args.git, depth=args.depth, min_confidence=args.min_confidence)
    if args.json:
        _dump(data)
        return
    for note in data.get("notes", []):
        print(f"note: {note}")
    if not data["changed"]:
        print("Nothing to analyse.")
        return
    print("Changed:")
    for c in data["changed"]:
        print(f"  * {c['name']}  [{c['service']}]  {c['file']}:{c['line']}")
    if not data["affected"]:
        print("\nNothing else depends on it (within the confidence threshold).")
        return
    print(f"\nMay be affected ({len(data['affected'])}):")
    for a in data["affected"]:
        print(f"  {'  ' * (a['distance'] - 1)}<- {a['name']}  [{a['service']}]  {a['file']}:{a['line']}  "
              f"({a['edge']} {a['because_of']}){'  TEST' if a['test'] else ''}")
    if data["entry_points"]:
        print("\nEntry points reached:")
        for e in data["entry_points"]:
            print(f"  {e['service']:<22} {e['method']:<7} {e['path']}")
    print(f"\nServices: {', '.join(f'{k} ({v})' for k, v in data['services'].items())}")
    if data["cross_service"]:
        print(f"Other services affected: {', '.join(data['cross_service'])}")
    if data["tests"]:
        print(f"Tests to run: {', '.join(t['name'] for t in data['tests'])}")


def cmd_god_nodes(args) -> None:
    rows = _nav(args).god_nodes(top=args.top, service=args.service, min_confidence=args.min_confidence)
    if args.json:
        _dump(rows)
        return
    print(f"{'#':>2}  {'callers':>7} {'callees':>7} {'affected':>8}  function")
    for i, r in enumerate(rows, start=1):
        extra = f"  called from {', '.join(r['caller_services'])}" if len(r["caller_services"]) > 1 else ""
        print(f"{i:>2}  {r['fan_in']:>7} {r['fan_out']:>7} {r['transitive_dependents']:>8}  {r['name']}  [{r['service']}]  "
              f"{r['file']}:{r['line']}{extra}")


def cmd_html(args) -> None:
    path = _nav(args).write_html(args.output)
    print(f"Written {path}")
    if args.open:
        import webbrowser
        webbrowser.open(path.resolve().as_uri())


def cmd_status(args) -> None:
    nav = _nav(args)
    m = nav.manifest
    if args.json:
        _dump(m)
        return
    print(f"Project    : {m['project']}  ({m['repo_path']})")
    print(f"Index      : {m.get('index_path', nav.paths.root)}  indexed {m['indexed_at']}")
    print(f"Stats      : {m['stats']}")
    print(f"Summaries  : {m['summary_mode']} (llm: {m.get('llm') or 'none'})")
    vs = m.get("vector_store") or {}
    print(f"Embeddings : {m['embedding']['provider']}:{m['embedding']['model']}")
    print(f"Vector DB  : {m['vector_backend']}" + (f"  {vs.get('location')}  collection={vs.get('collection')}" if vs else ""))
    print(f"Query LLM  : {nav.llm.describe() if nav.settings.llm_enabled else 'none (offline mode)'}")


def cmd_ask(args) -> None:
    nav = _nav(args)
    if args.question:
        res = nav.ask(args.question, mode=args.mode, top_k=args.top_k, expand=0 if args.no_expand else None)
        _dump(res) if args.json else _print_answer(res)
        return
    print(f"Interactive mode for '{nav.project}'. Ctrl-D to exit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q:
            _print_answer(nav.ask(q, mode=args.mode, top_k=args.top_k, expand=0 if args.no_expand else None))


def cmd_search(args) -> None:
    res = _nav(args).search(args.question, mode=args.mode, top_k=args.top_k, include_code=args.code,
                            expand=0 if args.no_expand else None)
    if args.json:
        _dump(res)
        return
    for i, h in enumerate(res["hits"], start=1):
        how = ", ".join(h["found_by"])
        if h.get("expanded_from"):
            how += f": {h['relation']} {h['expanded_from']}"
        print(f"{i:2}. {h['score']:.4f} {h['service']:<22} {h['file_path']}:{h['start_line']}  {h['name']}  [{how}]")
        if args.code:
            print(textwrap.indent(h["code"], "      ") + "\n")
    for note in res["notes"]:
        print(f"  note: {note}")


def cmd_services(args) -> None:
    nav = _nav(args)
    infos = [nav.service_info(s.name) for s in nav.services if s.kind != "shared"]
    if args.json:
        _dump(infos)
        return
    for info in infos:
        print(f"\n## {info['name']}  [{info['kind']}, {info['language']}]  path={info['path'] or '-'}")
        print(textwrap.fill(info["summary"] or "", 100, initial_indent="   ", subsequent_indent="   "))
        if info["endpoints"]:
            print("   endpoints:", ", ".join(f"{e['method']} {e['path']}" for e in info["endpoints"][:10]))
        if info["depends_on"]:
            print("   depends on:", ", ".join(info["depends_on"]))
        if info["used_by"]:
            print("   used by:", ", ".join(info["used_by"]))


def cmd_service(args) -> None:
    _dump(_nav(args).service_info(args.name))


def cmd_graph(args) -> None:
    g = _nav(args).graph()
    if args.dot:
        print(g["dot"])
    elif args.json:
        g.pop("dot", None)
        _dump(g)
    else:
        for e in g["edges"]:
            print(f"{e['source']:<25} -> {e['target']:<25} {e['kind']:<10} {e['evidence'][0] if e['evidence'] else ''}")


def cmd_impact(args) -> None:
    data = _nav(args).impact(args.service)
    data.pop("dot", None)
    _dump(data)


def cmd_tree(args) -> None:
    node = _nav(args).tree_view(args.node_id)
    if args.json:
        _dump(node)
        return
    print(f"[{node['kind']}] {node['name']}  ({node['id']})")
    if node.get("summary"):
        print(textwrap.fill(node["summary"], 100, initial_indent="  ", subsequent_indent="  "))
    for c in node["children"]:
        print(f"  - [{c['kind']}] {c['id']}\n      {c['summary'][:140]}")


def _node_line(n: dict) -> str:
    edge = f"{n['kind_of_edge']}, {n['resolution']} {n['confidence']:.2f}"
    return f"{n['name']}  [{n['service']}]  {n['file']}:{n['line']}  ({edge}; call at line {n['call_line']})"


def cmd_calls(args) -> None:
    data = _nav(args).calls(args.symbol, service=args.service, depth=args.depth, min_confidence=args.min_confidence)
    if args.json:
        _dump(data)
        return
    sym = data["symbol"]
    print(f"{sym['name']}  [{sym['service']}]  {sym['file']}:{sym['line']}")
    for ep in data["entry_points"]:
        print(f"  entry point: {ep['method']} {ep['path']}")
    for title, key in (("calls", "callees"), ("called by", "callers")):
        print(f"\n{title}:")
        if not data[key]:
            print("  (none found)")
        for n in data[key]:
            print("  " + "  " * (n["depth"] - 1) + ("-> " if key == "callees" else "<- ") + _node_line(n))
    if data["other_matches"]:
        print("\nother matches (use --service or the full name):")
        for m in data["other_matches"]:
            print(f"  {m['name']}  [{m['service']}]  {m['file']}:{m['line']}")


def cmd_path(args) -> None:
    data = _nav(args).call_path(args.source, args.target, service=args.service, min_confidence=args.min_confidence)
    if args.json:
        _dump(data)
        return
    if not data["found"]:
        print(f"No call chain found from {args.source} to {args.target}.")
        return
    first = data["hops"][0]["from"]
    print(f"{first['name']}  [{first['service']}]  {first['file']}:{first['line']}")
    for hop in data["hops"]:
        to = hop["to"]
        print(f"  -> {to['name']}  [{to['service']}]  {to['file']}:{to['line']}   "
              f"({hop['kind']}, {hop['resolution']} {hop['confidence']:.2f}, at line {hop['line']})")


def cmd_entrypoints(args) -> None:
    eps = _nav(args).entry_points(args.service)
    if args.json:
        _dump(eps)
        return
    for e in eps:
        print(f"{e['service']:<22} {e['method']:<6} {e['path']:<32} -> {e['handler_name'] or '?'}  ({e['file']}:{e['line']}, {e['via'] or 'unresolved'})")


def cmd_docs(args) -> None:
    doc = _nav(args).service_doc(args.service)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(doc)
        print(f"Written to {args.output}")
    else:
        print(doc)


def cmd_check(args) -> None:
    from .llm import get_embedder, get_llm
    settings = get_settings(args.repo)
    print(f"code-navigator {__version__}")
    print(f"Config     : env CODENAV_*, {args.repo}/{REPO_ENV_FILE}, {USER_ENV_FILE}")
    print(f"LLM        : {settings.llm_provider}:{settings.llm_model}"
          + (f" @ {settings.llm_base_url}" if settings.llm_enabled else " (offline mode)"))
    print(f"Embeddings : {settings.embed_provider}:{settings.embed_model}"
          + (f" @ {settings.embed_base_url}" if settings.embed_base_url else ""))
    print(f"Vector DB  : {settings.vector_backend}" + (f" @ {redact(settings.vector_url)}" if settings.vector_url else ""))
    try:
        from .ingestion.code_parser import _get_ts_parser
        ts = _get_ts_parser("java") is not None
        print(f"tree-sitter: {'available' if ts else 'not installed (regex fallback; pip install code-navigator[parsers])'}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"tree-sitter: error {exc}")
    ok = True
    if settings.llm_enabled:
        try:
            reply = get_llm(settings).complete("Reply with the single word: pong", max_tokens=10, retries=0)
            print(f"LLM check  : OK -> {reply[:40]!r}")
        except Exception as exc:
            ok = False
            print(f"LLM check  : FAILED -> {exc}")
    try:
        vec = get_embedder(settings).embed_query("hello world")
        print(f"Embed check: OK -> dimension {len(vec)}")
    except Exception as exc:
        ok = False
        print(f"Embed check: FAILED -> {exc}")
    try:
        import tempfile
        from pathlib import Path
        from .indexing.vector_index import open_vector_store
        with tempfile.TemporaryDirectory() as tmp:
            store = open_vector_store(settings.vector_backend, Path(tmp), settings, "check",
                                      collection="codenav_connection_check")
            used = store.backend
            store.close()
        if used != settings.vector_backend:
            ok = False
            print(f"Vector DB  : FAILED -> '{settings.vector_backend}' unavailable, would fall back to '{used}'")
        else:
            print(f"Vector DB  : OK -> {used}")
    except Exception as exc:
        ok = False
        print(f"Vector DB  : FAILED -> {exc}")
    sys.exit(0 if ok else 1)


def cmd_mcp(args) -> None:
    from .mcp_server import serve
    serve(args.repo, index_dir=args.index_dir)


# ---------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-navigator",
        description="Index a microservice codebase and ask questions about it - with or without an LLM.",
    )
    parser.add_argument("--version", action="version", version=f"code-navigator {__version__}")
    parser.add_argument("--log", default="WARNING", help="log level (DEBUG, INFO, WARNING)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def query(name: str, fn, helptext: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("-C", "--repo", default=".", help="indexed repository (default: current directory)")
        p.add_argument("--index-dir", help=f"index location (default: <repo>/{INDEX_DIR_NAME})")
        p.set_defaults(func=fn)
        return p

    p = sub.add_parser("index", help="index a repository folder or git URL")
    p.add_argument("path", nargs="?", default=".", help="folder or git URL (default: current directory)")
    p.add_argument("--name", help="project name (default: folder name)")
    p.add_argument("--summaries", choices=["llm", "heuristic"], help="override the summary mode")
    p.add_argument("--out", help=f"write the index here instead of <repo>/{INDEX_DIR_NAME}")
    p.add_argument("--json", action="store_true", help="print the manifest as JSON")
    p.set_defaults(func=cmd_index, log_default="INFO")

    p = query("status", cmd_status, "show what is indexed")
    p.add_argument("--json", action="store_true")

    for name, fn, helptext in (("ask", cmd_ask, "ask a question (omit it for interactive mode)"),
                               ("search", cmd_search, "retrieval only, no answer generation")):
        p = query(name, fn, helptext)
        p.add_argument("question", nargs="?" if name == "ask" else None)
        p.add_argument("--mode", choices=["hybrid", "vector", "vectorless"])
        p.add_argument("--top-k", type=int)
        p.add_argument("--json", action="store_true")
        p.add_argument("--no-expand", action="store_true", help="do not add related functions from the call graph")
        if name == "search":
            p.add_argument("--code", action="store_true", help="include the code of each hit")

    p = query("services", cmd_services, "list services with summaries")
    p.add_argument("--json", action="store_true")

    p = query("service", cmd_service, "details of one service (JSON)")
    p.add_argument("name")

    p = query("graph", cmd_graph, "print the service dependency graph")
    p.add_argument("--dot", action="store_true", help="Graphviz DOT output")
    p.add_argument("--json", action="store_true")

    p = query("impact", cmd_impact, "impact analysis for a service (JSON)")
    p.add_argument("service")

    p = query("tree", cmd_tree, "browse the vectorless tree")
    p.add_argument("node_id", nargs="?", default="root")
    p.add_argument("--json", action="store_true")

    p = query("calls", cmd_calls, "what a function calls and what calls it")
    p.add_argument("symbol", help="function or method, e.g. place_order or Handler.Charge")
    p.add_argument("--service", help="limit the lookup to one service")
    p.add_argument("--depth", type=int, default=1, help="levels to follow in each direction")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--json", action="store_true")

    p = query("path", cmd_path, "shortest call chain between two functions")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--service", help="service of the source function")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--json", action="store_true")

    p = query("entrypoints", cmd_entrypoints, "HTTP endpoints mapped to their handler functions")
    p.add_argument("--service")
    p.add_argument("--json", action="store_true")

    p = query("affected", cmd_affected, "what may break if functions, files or the git diff change")
    p.add_argument("symbols", nargs="*", help="changed functions/classes, e.g. IdempotencyStore.Put")
    p.add_argument("--file", action="append", help="a changed file (repeatable)")
    p.add_argument("--git", nargs="?", const="HEAD", help="use the uncommitted diff against REF (default HEAD)")
    p.add_argument("--no-update", action="store_true", help="with --git: do not refresh the index first")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--json", action="store_true")

    p = query("god-nodes", cmd_god_nodes, "the most connected functions (widest blast radius)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--service")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--json", action="store_true")

    p = query("html", cmd_html, "write the interactive graph viewer")
    p.add_argument("-o", "--output", help="output file (default: <index>/graph.html)")
    p.add_argument("--open", action="store_true", help="open it in the browser")

    p = sub.add_parser("update", help="re-index only if something changed (reuses embeddings and summaries)")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    p.add_argument("--summaries", choices=["llm", "heuristic"])
    p.add_argument("--out", "--index-dir", dest="out")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("watch", help="keep the index fresh while you edit")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between checks")
    p.add_argument("--out", "--index-dir", dest="out")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("hook", help="install git hooks that refresh the index")
    p.add_argument("action", choices=["install", "uninstall", "status"])
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--index-dir")
    p.set_defaults(func=cmd_hook)

    p = query("docs", cmd_docs, "generate Markdown docs for a service")
    p.add_argument("service")
    p.add_argument("-o", "--output")

    p = sub.add_parser("check", help="check LLM / embedding configuration")
    p.add_argument("-C", "--repo", default=".")
    p.set_defaults(func=cmd_check)

    query("mcp", cmd_mcp, "run the MCP server over stdio (for Claude Code, Cursor, ...)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = args.log if args.log != "WARNING" else getattr(args, "log_default", "WARNING")
    setup_logging(level)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
