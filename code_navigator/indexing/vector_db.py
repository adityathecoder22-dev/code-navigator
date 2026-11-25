"""Vector database integrations: Qdrant, LanceDB and PostgreSQL/pgvector.

Each one stores, per chunk: id, embedding, chunk text and metadata, and supports similarity
search restricted to a set of services and/or files (the scope chosen by the tree navigator).
A missing client library raises RuntimeError (the caller falls back to the simple store); a
database that cannot be reached raises ValueError (indexing stops with a clear message).
"""
from __future__ import annotations

import json
import uuid

from ..models import Hit
from ..utils import log
from .vector_index import VectorStore, _clean_meta

_BATCH = 256
_ID_NS = uuid.UUID("6f1c7a52-6b5e-4c8e-9a1f-3d0d2b6f7c11")


def _scope(services: list[str] | None, files: list[str] | None) -> tuple[list[str], list[str]]:
    return list(services or []), list(files or [])


# ---------------------------------------------------------------------------- Qdrant

class QdrantVectorStore(VectorStore):
    """Qdrant: embedded (a folder, single process) or a server (``CODENAV_VECTOR_URL=http://host:6333``)."""
    backend = "qdrant"

    def __init__(self, path, collection: str, url: str | None = None, api_key: str | None = None):
        try:
            from qdrant_client import QdrantClient, models  # type: ignore
        except ImportError as exc:
            raise RuntimeError('qdrant-client is not installed: pip install "code-navigator[qdrant]"') from exc
        self.models = models
        self.collection = collection
        self.location = url or str(path)
        try:
            if url:
                self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
                self.client.get_collections()          # fail early if the server is unreachable
            else:
                path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(path))
        except Exception as exc:
            raise ValueError(f"cannot open Qdrant at {self.location}: {exc}") from exc

    def _exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection))
        except AttributeError:  # older clients
            return self.collection in {c.name for c in self.client.get_collections().collections}

    def reset(self) -> None:
        if self._exists():
            self.client.delete_collection(self.collection)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        if not ids:
            return
        m = self.models
        if not self._exists():
            self.client.create_collection(self.collection, vectors_config=m.VectorParams(
                size=len(embeddings[0]), distance=m.Distance.COSINE))
            for field in ("service", "file_path"):
                try:
                    self.client.create_payload_index(self.collection, field, m.PayloadSchemaType.KEYWORD)
                except Exception:  # embedded mode has no payload indexes
                    pass
        for i in range(0, len(ids), _BATCH):
            points = [m.PointStruct(id=str(uuid.uuid5(_ID_NS, cid)), vector=[float(x) for x in vec],
                                    payload={**_clean_meta(meta), "chunk_id": cid, "document": doc})
                      for cid, vec, doc, meta in zip(ids[i:i + _BATCH], embeddings[i:i + _BATCH],
                                                     documents[i:i + _BATCH], metadatas[i:i + _BATCH])]
            self.client.upsert(self.collection, points=points, wait=True)

    def query(self, embedding, k, services=None, files=None) -> list[Hit]:
        if not self._exists():
            return []
        m = self.models
        svc, fls = _scope(services, files)
        should = ([m.FieldCondition(key="service", match=m.MatchAny(any=svc))] if svc else []) + \
                 ([m.FieldCondition(key="file_path", match=m.MatchAny(any=fls))] if fls else [])
        flt = m.Filter(should=should) if should else None
        vec = [float(x) for x in embedding]
        if hasattr(self.client, "query_points"):
            points = self.client.query_points(self.collection, query=vec, limit=k, query_filter=flt,
                                              with_payload=True).points
        else:  # qdrant-client < 1.10
            points = self.client.search(self.collection, query_vector=vec, limit=k, query_filter=flt, with_payload=True)
        hits = []
        for p in points:
            payload = dict(p.payload or {})
            cid, doc = payload.pop("chunk_id", str(p.id)), payload.pop("document", "")
            hits.append(Hit(chunk_id=cid, score=float(p.score), text=doc, metadata=payload))
        return hits

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self._exists() else 0

    def vectors_by_id(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if not self._exists():
            return out
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=1000, offset=offset,
                                                with_payload=["chunk_id"], with_vectors=True)
            for p in points:
                out[(p.payload or {}).get("chunk_id", str(p.id))] = list(p.vector)
            if offset is None:
                break
        return out

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------- LanceDB

def _sql_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


class LanceVectorStore(VectorStore):
    """LanceDB: an embedded columnar vector database (a folder, or s3:// / gs:// / db:// URIs)."""
    backend = "lancedb"

    def __init__(self, uri: str, table: str = "code", api_key: str | None = None):
        try:
            import lancedb  # type: ignore
        except ImportError as exc:
            raise RuntimeError('lancedb is not installed: pip install "code-navigator[lancedb]"') from exc
        self.location, self.collection = uri, table
        try:
            self.db = lancedb.connect(uri, api_key=api_key) if api_key else lancedb.connect(uri)
        except Exception as exc:
            raise ValueError(f"cannot open LanceDB at {uri}: {exc}") from exc

    def _names(self) -> list[str]:
        if hasattr(self.db, "list_tables"):
            res = self.db.list_tables()
            return list(getattr(res, "tables", res))
        return list(self.db.table_names())

    def _table(self):
        return self.db.open_table(self.collection) if self.collection in self._names() else None

    def reset(self) -> None:
        if self.collection in self._names():
            self.db.drop_table(self.collection)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        if not ids:
            return
        rows = [{"id": cid, "vector": [float(x) for x in vec], "document": doc,
                 "service": str(meta.get("service", "")), "file_path": str(meta.get("file_path", "")),
                 "metadata": json.dumps(_clean_meta(meta))}
                for cid, vec, doc, meta in zip(ids, embeddings, documents, metadatas)]
        table = self._table()
        if table is None:
            self.db.create_table(self.collection, data=rows)
        else:
            table.add(rows)

    def query(self, embedding, k, services=None, files=None) -> list[Hit]:
        table = self._table()
        if table is None:
            return []
        q = table.search([float(x) for x in embedding])
        q = q.distance_type("cosine") if hasattr(q, "distance_type") else q.metric("cosine")
        svc, fls = _scope(services, files)
        clauses = ([f"service IN ({_sql_list(svc)})"] if svc else []) + ([f"file_path IN ({_sql_list(fls)})"] if fls else [])
        if clauses:
            q = q.where(" OR ".join(clauses), prefilter=True)
        rows = q.limit(k).to_list()
        return [Hit(chunk_id=r["id"], score=1.0 - float(r.get("_distance", 1.0)), text=r.get("document", ""),
                    metadata=json.loads(r.get("metadata") or "{}")) for r in rows]

    def count(self) -> int:
        table = self._table()
        return table.count_rows() if table is not None else 0

    def vectors_by_id(self) -> dict[str, list[float]]:
        table = self._table()
        if table is None:
            return {}
        data = table.to_arrow().select(["id", "vector"]).to_pydict()
        return {cid: list(vec) for cid, vec in zip(data["id"], data["vector"])}


# ---------------------------------------------------------------------------- PostgreSQL + pgvector

class PgVectorStore(VectorStore):
    """PostgreSQL with the pgvector extension. One table per indexed project, HNSW cosine index."""
    backend = "pgvector"

    def __init__(self, dsn: str, table: str):
        try:
            import psycopg  # type: ignore
            self._connect = lambda: psycopg.connect(dsn, autocommit=True)
        except ImportError:
            try:
                import psycopg2  # type: ignore
            except ImportError as exc:
                raise RuntimeError('psycopg is not installed: pip install "code-navigator[pgvector]"') from exc

            def _connect():
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                return conn
            self._connect = _connect
        self.location, self.collection = dsn, table
        self.table = '"' + table.replace('"', "") + '"'
        try:
            self.conn = self._connect()
            self._exec("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            raise ValueError(f"cannot use pgvector at {self.describe()['location']}: {exc}") from exc

    def _exec(self, sql: str, params=None, fetch: bool = False):
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            return cur.fetchall() if fetch else None
        finally:
            cur.close()

    def _exists(self) -> bool:
        return bool(self._exec("SELECT to_regclass(%s)", (self.table,), fetch=True)[0][0])

    @staticmethod
    def _vec(v) -> str:
        return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"

    def reset(self) -> None:
        self._exec(f"DROP TABLE IF EXISTS {self.table}")

    def add(self, ids, embeddings, documents, metadatas) -> None:
        if not ids:
            return
        dim = len(embeddings[0])
        t, base = self.table, self.collection.replace('"', "")[:40]
        self._exec(f"CREATE TABLE IF NOT EXISTS {t} (id text PRIMARY KEY, embedding vector({dim}) NOT NULL, "
                   f"document text, service text, file_path text, metadata jsonb)")
        self._exec(f'CREATE INDEX IF NOT EXISTS "{base}_service" ON {t} (service)')
        self._exec(f'CREATE INDEX IF NOT EXISTS "{base}_file" ON {t} (file_path)')
        try:
            self._exec(f'CREATE INDEX IF NOT EXISTS "{base}_hnsw" ON {t} USING hnsw (embedding vector_cosine_ops)')
        except Exception as exc:  # pgvector < 0.5 has no HNSW; exact search still works
            log.info("pgvector HNSW index not created (%s); using exact search", exc)
        sql = (f"INSERT INTO {t} (id, embedding, document, service, file_path, metadata) "
               f"VALUES (%s, %s::vector, %s, %s, %s, %s::jsonb) ON CONFLICT (id) DO UPDATE SET "
               f"embedding = EXCLUDED.embedding, document = EXCLUDED.document, service = EXCLUDED.service, "
               f"file_path = EXCLUDED.file_path, metadata = EXCLUDED.metadata")
        cur = self.conn.cursor()
        try:
            for i in range(0, len(ids), _BATCH):
                cur.executemany(sql, [(cid, self._vec(vec), doc, str(meta.get("service", "")),
                                       str(meta.get("file_path", "")), json.dumps(_clean_meta(meta)))
                                      for cid, vec, doc, meta in zip(ids[i:i + _BATCH], embeddings[i:i + _BATCH],
                                                                     documents[i:i + _BATCH], metadatas[i:i + _BATCH])])
        finally:
            cur.close()

    def query(self, embedding, k, services=None, files=None) -> list[Hit]:
        if not self._exists():
            return []
        svc, fls = _scope(services, files)
        where, params = "", []
        if svc or fls:
            parts = []
            if svc:
                parts.append("service = ANY(%s)")
                params.append(svc)
            if fls:
                parts.append("file_path = ANY(%s)")
                params.append(fls)
            where = "WHERE " + " OR ".join(parts)
        vec = self._vec(embedding)
        rows = self._exec(f"SELECT id, document, metadata, 1 - (embedding <=> %s::vector) AS score FROM {self.table} "
                          f"{where} ORDER BY embedding <=> %s::vector LIMIT %s", [vec, *params, vec, k], fetch=True)
        hits = []
        for cid, doc, meta, score in rows:
            meta = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            hits.append(Hit(chunk_id=cid, score=float(score), text=doc or "", metadata=meta))
        return hits

    def count(self) -> int:
        return int(self._exec(f"SELECT count(*) FROM {self.table}", fetch=True)[0][0]) if self._exists() else 0

    def vectors_by_id(self) -> dict[str, list[float]]:
        if not self._exists():
            return {}
        rows = self._exec(f"SELECT id, embedding::text FROM {self.table}", fetch=True)
        return {cid: [float(x) for x in text.strip("[]").split(",") if x] for cid, text in rows}

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
