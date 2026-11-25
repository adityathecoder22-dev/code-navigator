"""code-navigator: index a microservice codebase into a navigable tree, a vector store and a
service dependency graph, then query it with or without an LLM.

Library use::

    from code_navigator import index_repo, Navigator

    index_repo("path/to/repo")                 # writes path/to/repo/.code-navigator/
    nav = Navigator("path/to/repo")
    nav.search("where is retry logic?")        # works without an LLM
    nav.impact("payment-service")
    nav.ask("What happens when an order is placed?")   # uses an LLM if one is configured
"""
__version__ = "0.4.0"

__all__ = ["__version__", "index_repo", "Navigator", "NotIndexed", "Settings", "get_settings"]


def __getattr__(name):  # lazy imports keep `import code_navigator` cheap
    if name == "index_repo":
        from .indexing.pipeline import build_index
        return build_index
    if name in ("Navigator", "NotIndexed"):
        from . import engine
        return getattr(engine, name)
    if name in ("Settings", "get_settings"):
        from . import config
        return getattr(config, name)
    raise AttributeError(f"module 'code_navigator' has no attribute {name!r}")
