# code-navigator

**Index a microservice codebase once, then ask it questions with or without an LLM.**

`code-navigator` reads a repository (docker-compose, Kubernetes, OpenAPI, `.proto` files and the code of
every service) and builds three things inside it, in `.code-navigator/`:

| Artifact | What it is | Used for |
|---|---|---|
| `tree.json` | A **vectorless tree**: system → service → module → file → symbol, each node with a summary | Navigating like an engineer browsing the repo |
| `vectors/` + `chunks.jsonl` | One chunk per function/class with a metadata header, embedded and stored in a **vector database** (built-in numpy store, ChromaDB, Qdrant, LanceDB or pgvector) | Semantic code search |
| `services.json` + `dependencies.json` | Services, endpoints, topics and a **service dependency graph** (HTTP, gRPC, Kafka/RabbitMQ, databases) with `file:line` evidence on every edge | Cross-service questions, impact analysis |
| `callgraph.json` | A **function-level graph**: calls, HTTP calls and Kafka messages between services, class inheritance, imports, and which function handles each endpoint and topic | "Who calls this?", tracing a request end to end, "what breaks if I change this?" |
| `graph.html` | An **interactive viewer** for both graphs. One file, works offline | Exploring the system in a browser |

Everything except the viewer is plain JSON or NumPy, so you can read it from any language or tool.

code-navigator maps both levels of a system:
- **Services:** which services exist, how they talk to each other, and what breaks when one fails.
- **Functions:** which function calls which, and how one request travels from an API route through several services.

---

## Install

```bash
pip install code-navigator              # core: PyYAML, numpy, requests
pip install "code-navigator[parsers]"   # recommended: tree-sitter parsing for Java, Go, JS/TS, C#, ...
```

From a checkout of this repository:

```bash
pip install -e ".[parsers,dev]"
```

Python 3.9+.

## Quick start

```bash
cd path/to/your-microservices-repo
code-navigator index                         # writes ./.code-navigator/
code-navigator services                      # what services exist, what they call
code-navigator search "where is retry logic?"
code-navigator impact payment-service        # who is affected if it changes or goes down
code-navigator calls place_order --depth 2   # what it calls, what calls it
code-navigator path create_order ChargeWithRetry
code-navigator affected --git                # what your uncommitted change may break
code-navigator god-nodes                     # the riskiest, most connected functions
code-navigator html --open                   # interactive graph in the browser
code-navigator ask "What happens when a user places an order?"
```

Try it on the bundled 5-service demo system (Node, Python, Go and Java, with Kafka, Postgres and Redis):

```bash
code-navigator index examples/sample-shop
code-navigator ask "How does payment-service avoid charging twice?" -C examples/sample-shop
```

`code-navigator index` also accepts a git URL; the clone goes to `~/.cache/code-navigator/repos/`.

---

## Three ways to use the index

### 1. No LLM (the default): offline and deterministic

With no configuration, nothing leaves your machine:
- **Summaries** are built from extracted facts (class and function names, endpoints, dependencies).
- **Navigation** uses a keyword scorer.
- **Embeddings** use feature hashing.
- **`ask`** returns the ranked code with `file:line` instead of a written answer.

The dependency graph, impact analysis, architecture checks, endpoint and topic listings work the same in every mode.

### 2. With an LLM: written summaries and cited answers

Put settings in `<repo>/.code-navigator.env`, in `~/.config/code-navigator/config.env`, or in the
environment (see [`code-navigator.env.example`](code-navigator.env.example)):

```bash
# fully local (Ollama)
CODENAV_LLM_PROVIDER=ollama
CODENAV_LLM_MODEL=qwen2.5-coder:7b
CODENAV_EMBED_PROVIDER=ollama          # ollama pull nomic-embed-text

# or Claude / OpenAI / any OpenAI-compatible server
CODENAV_LLM_PROVIDER=anthropic         # key from ANTHROPIC_API_KEY or CODENAV_LLM_API_KEY
```

Then re-run `code-navigator index`. The LLM is used in three places:
- **Index time:** one summary per file, module and service, plus one for the whole system. Summaries are cached by content hash, so re-indexing only re-summarises what changed.
- **Query time, navigation:** 2–6 short calls to pick services, then modules and files.
- **Query time, answer:** one call that writes the answer with `[n]` citations.

Check the setup with `code-navigator check`.

### 3. Through your AI assistant (MCP): no API key needed

```bash
claude mcp add code-navigator -- code-navigator mcp --repo /path/to/repo
```

Any MCP client works (Claude Code, Cursor, VS Code, ...). The assistant's own model does the reasoning; code-navigator only serves the index.

| Tool | Purpose |
|---|---|
| `overview` | System summary, services, architecture findings. Start here |
| `list_services` / `get_service` | Endpoints, topics, env vars (secrets masked), files, callers and callees |
| `browse_tree` | Walk the vectorless tree from `root` down to symbols, reading summaries |
| `search_code` | Hybrid search: tree navigation, then vector search, then exact symbol match, fused with RRF |
| `get_chunk` | Full code of a result |
| `dependencies` / `impact` | Service edges with evidence; services affected directly and transitively |
| `calls` / `call_path` / `entry_points` | Callers and callees of a function, the call chain from A to B (across services over HTTP and Kafka), and which function handles each endpoint or topic |
| `affected` / `god_nodes` | What a change to functions, files or a git diff may break; the most connected functions |
| `ask` | An end-to-end answer from code-navigator's own LLM, if one is configured |

## Call graph

Edge kinds:

| Kind | Meaning |
|---|---|
| `calls` / `instantiates` | A function calls another function, or creates a class |
| `http` | Client code calls an endpoint of another service; the edge points to that endpoint's handler |
| `async` | A function publishes to a Kafka/RabbitMQ topic; the edge points to each function that consumes it |
| `inherits` / `implements` | Class declarations (`extends`, `implements`, Python bases, C#/Kotlin `:`, Go struct embedding) |
| `imports` | A file imports a class or function, or another file |

Each call site is resolved to the function it most likely invokes. Every edge records how it was
resolved and a confidence score:

| Resolution | Confidence | Example |
|---|---|---|
| `self` | 0.95 | `self.helper()`, `this.helper()`, `h.Charge()` inside a Go method on `h` |
| `same_file` | 0.95 | `helper()` defined in the same file |
| `field_type` | 0.90 | `self.payments.charge()` where `self.payments = PaymentClient()`; Go struct fields; Java injected fields |
| `class_name` / `import` / `same_package` | 0.90 | `PaymentClient()`, `payment.NewHandler()`, `from .clients import _with_retry` |
| `local_type` | 0.85 | `service = OrderService()` … `service.place_order()` |
| `inherited` | 0.85 | `self.save()`, where `save` is defined on a base class |
| `endpoint_path` | 0.80 | `self.http.post("/charge")` → the payment-service handler for `POST /charge` |
| `topic` | 0.75 | `producer.send("order-events")` → `@KafkaListener(topics="order-events")` method |
| `declaration` | 0.95 | `class Repo(Base)`, `class A extends B implements C` |
| `name_match` / `ambiguous` | 0.50 / 0.30 | Only a method name matched. Ambiguous matches are stored but hidden by default |

Real output on the bundled sample:

```
$ code-navigator path orders.js IdempotencyStore.Put -C examples/sample-shop
orders.js  [api-gateway]  api-gateway/src/routes/orders.js:1
  -> create_order  [order-service]  order-service/app/main.py:10   (http, endpoint_path 0.80, at line 11)
  -> OrderService.place_order  [order-service]  order-service/app/service.py:17   (calls, local_type 0.85, at line 13)
  -> PaymentClient.charge  [order-service]  order-service/app/clients.py:57   (calls, field_type 0.90, at line 39)
  -> Handler.Charge  [payment-service]  payment-service/internal/payment/handler.go:35   (http, endpoint_path 0.80, at line 59)
  -> IdempotencyStore.Put  [payment-service]  payment-service/internal/payment/idempotency.go:29   (calls, field_type 0.90, at line 51)
```

**Search expansion.** `search` and `ask` add up to 4 functions that the top results call or are called by,
marked `found_by: call_graph`. So a question about placing an order also brings in `PaymentClient.charge`
and `InventoryClient.reserve`. Turn this off with `--no-expand` or `CODENAV_CALL_EXPANSION=0`.

## What breaks if I change this?

```
$ code-navigator affected IdempotencyStore.Put
Changed:
  * IdempotencyStore.Put  [payment-service]  payment-service/internal/payment/idempotency.go:29

May be affected (5):
  <- Handler.Charge  [payment-service]  payment-service/internal/payment/handler.go:35  (calls IdempotencyStore.Put)
    <- PaymentClient.charge  [order-service]  order-service/app/clients.py:57  (http Handler.Charge)
      <- OrderService.place_order  [order-service]  order-service/app/service.py:17  (calls PaymentClient.charge)
        <- create_order  [order-service]  order-service/app/main.py:10  (calls OrderService.place_order)
          <- orders.js  [api-gateway]  api-gateway/src/routes/orders.js:1  (http create_order)

Entry points reached:
  api-gateway            POST    /api/orders
  order-service          POST    /orders
  payment-service        POST    /charge
```

`affected` follows each edge in the direction that matters:
- **Callers and subclasses** depend on the code they call or extend.
- **Message consumers** depend on what the changed code publishes.

You can give it symbols, `--file PATH`, or `--git [REF]`. With `--git`, it uses your uncommitted diff, including new files, and refreshes the index first. Results list the HTTP endpoints and topics reached and flag the tests among the affected code.

`god-nodes` ranks functions by how connected they are: number of callers, number of callees, and callers from other services. It also shows how much would be affected by changing each one. These are the riskiest places to change.

## Vector databases

The vectorless tree, the call graph and the dependency graph are JSON files. Only the chunk embeddings go
into a vector store, and you can choose which one:

| `CODENAV_VECTOR_BACKEND` | Install | Where the data lives |
|---|---|---|
| `simple` (default) | nothing | numpy files in `.code-navigator/vectors/simple/` |
| `chroma` | `pip install "code-navigator[chroma]"` | ChromaDB in `.code-navigator/vectors/chroma/` |
| `qdrant` | `pip install "code-navigator[qdrant]"` | Embedded in `.code-navigator/vectors/qdrant/`, or a Qdrant server or Qdrant Cloud with `CODENAV_VECTOR_URL=http://host:6333` (+ `CODENAV_VECTOR_API_KEY`) |
| `lancedb` | `pip install "code-navigator[lancedb]"` | `.code-navigator/vectors/lancedb/`, or `CODENAV_VECTOR_URL=s3://…` / LanceDB Cloud |
| `pgvector` | `pip install "code-navigator[pgvector]"` | PostgreSQL with the pgvector extension: `CODENAV_VECTOR_URL=postgresql://user:pass@host:5432/db` |

```bash
CODENAV_VECTOR_BACKEND=pgvector CODENAV_VECTOR_URL=postgresql://app:secret@localhost:5432/codenav \
  code-navigator index .
code-navigator check          # verifies the LLM, the embeddings and the vector database
```

**What each backend stores per chunk:** the id, the embedding, the chunk text, and its metadata (service, file, kind, name, lines). Every backend supports search restricted to services or files, which hybrid retrieval uses after tree navigation.

**Server details:**
- **Naming:** on a shared server, each project gets its own collection or table, named `codenav_<project>`. Override it with `CODENAV_VECTOR_COLLECTION`.
- **pgvector setup:** pgvector creates the `vector` extension if allowed, plus service and file indexes and an HNSW cosine index (pgvector ≥ 0.5).
- **Passwords** are never written to the manifest.

**Notes:**
- **Unreachable server:** indexing stops with a clear error instead of silently using another store.
- **Missing library:** if a backend's client library is not installed, code-navigator falls back to `simple` with a warning.
- **Embedded Qdrant** allows one process per folder. If the MCP server and `watch` or the git hooks run at the same time, use a Qdrant server.
- **Switching backends** reuses the existing embeddings. To read a server store you are leaving, its URL must still be set.
- **Old server data is not deleted:** a collection or table you no longer use stays on the server.

## Interactive graph

Every index also writes `.code-navigator/graph.html`. Open it, or run `code-navigator html --open`.
- **Two views:** services, and functions coloured by service.
- **Navigation:** pan, zoom, drag nodes, and search.
- **Filters:** by service, by edge kind, and by minimum confidence.
- **Details panel:** click a node to see its callers and callees, its entry points, and a **"Show what a change affects"** button.
- **Offline:** it is a single file with no CDN or network access.

## Keeping the index fresh

```bash
code-navigator update          # re-index only if files or settings changed
code-navigator watch           # re-index automatically while you edit
code-navigator hook install    # re-index in the background after commit, merge, checkout and rebase
```

Re-indexing keeps the expensive parts:
- **Embeddings** are cached by chunk content, so only new or changed code is embedded again.
- **LLM summaries** are cached by prompt, so only changed files, modules and services are summarised again.
- **Up-to-date check:** `update` exits immediately when no indexed file and no index setting changed.

`hook uninstall` removes only the lines code-navigator added and leaves your existing hooks alone.

**Limits.** The call graph comes from static analysis. It cannot see dynamic dispatch, reflection,
dependency-injection containers that are configured outside the code, callbacks passed as values, or URLs
assembled at runtime. Calls into libraries are not tracked.

## Python API

```python
from code_navigator import index_repo, Navigator
from code_navigator.indexing.pipeline import update_index

index_repo("path/to/repo")
update_index("path/to/repo")                    # no-op when nothing changed
nav = Navigator("path/to/repo")

nav.overview()["services"]
nav.search("idempotency", include_code=True)["hits"]
nav.impact("inventory-service")["transitive_dependents"]
nav.tree_view("svc:payment-service")["children"]
nav.calls("place_order", depth=2)["callees"]
nav.call_path("create_order", "ChargeWithRetry")["hops"]
nav.entry_points("payment-service")
nav.affected(["IdempotencyStore.Put"])["entry_points"]
nav.affected(git_ref="HEAD")                    # impact of the uncommitted diff
nav.god_nodes(top=10)
nav.write_html("graph.html")
nav.ask("Which services consume order-events?")["answer"]
```

## CLI reference

```
code-navigator index [PATH|GIT_URL] [--name N] [--summaries llm|heuristic] [--out DIR] [--json]
code-navigator status | services | service NAME | graph [--dot|--json] | impact SERVICE
code-navigator ask ["QUESTION"] [--mode hybrid|vector|vectorless] [--top-k K] [--json]
code-navigator search "QUESTION" [--code] [--json]
code-navigator tree [NODE_ID]               # e.g. root, svc:order-service
code-navigator calls SYMBOL [--service S] [--depth N] [--min-confidence X] [--json]
code-navigator path FROM TO [--service S] [--json]
code-navigator entrypoints [--service S] [--json]
code-navigator affected [SYMBOL...] [--file F]... [--git [REF]] [--depth N] [--json]
code-navigator god-nodes [--top N] [--service S] [--json]
code-navigator html [-o FILE] [--open]
code-navigator update [PATH] [--force] [--quiet]
code-navigator watch [PATH] [--interval SECONDS]
code-navigator hook install|uninstall|status [PATH]
code-navigator docs SERVICE [-o FILE]       # Markdown docs for a service
code-navigator check                        # test LLM / embeddings
code-navigator mcp                          # MCP server over stdio
```

Query commands read the index of the current directory. Use `-C/--repo PATH` for another one, or
`--index-dir DIR` if you wrote it elsewhere with `index --out`. `cnav` is a short alias.

## Index files and git

`.code-navigator/` contains a `.gitignore` that ignores everything, so the index is not committed by
accident. Delete that file if your team wants to commit and share the index. To keep the index current,
use `update`, `watch` or `hook install`.

## Development

```bash
pip install -e ".[dev]"
pytest          # fully offline: no LLM, hash embeddings
```

## License

MIT
