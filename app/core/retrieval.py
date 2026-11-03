"""Tenant-scoped, versioned, persisted hybrid retrieval for small document sets."""

import hashlib
import json
import math
import os
import re
import threading
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.database import Database, digest
from app.core.security import Principal


def tokens(text: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "to",
        "of",
        "for",
        "in",
        "on",
        "and",
        "or",
        "how",
        "what",
        "when",
        "can",
        "could",
        "i",
        "my",
        "we",
        "you",
        "your",
        "our",
        "do",
        "does",
        "this",
        "that",
        "with",
        "be",
        "it",
        "would",
        "should",
    }
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in stop]


class Encoder(Protocol):
    name: str

    def documents(self, texts: list[str]) -> list[list[float]]: ...
    def query(self, text: str) -> list[float]: ...


class HashEncoder:
    name = "lexical-hash-512-v2"

    def documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            row = [0.0] * 512
            for token in tokens(text):
                row[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 512] += 1
            norm = math.sqrt(sum(x * x for x in row)) or 1
            result.append([x / norm for x in row])
        return result

    def query(self, text: str) -> list[float]:
        return self.documents([text])[0]


class SemanticEncoder:
    name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self.name,
                    cache_dir=os.getenv("MODEL_CACHE", "/tmp/fastembed-cache"),
                    threads=2,
                )
        return self._model

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model().passage_embed(texts, batch_size=16)]

    def query(self, text: str) -> list[float]:
        return next(iter(self.model().query_embed(text))).tolist()


def configured_encoder() -> Encoder:
    backend = os.getenv("EMBEDDER", "fastembed")
    if backend == "hash":
        return HashEncoder()
    if backend != "fastembed":
        raise ValueError("EMBEDDER must be fastembed or hash")
    return SemanticEncoder()


class Knowledge:
    def __init__(self, database: Database, encoder: Encoder):
        self.db, self.encoder = database, encoder

    def ingest(
        self,
        principal: Principal,
        source: str,
        title: str,
        entity: str,
        scope: str,
        text: str,
        size: int = 140,
        overlap: int = 25,
    ) -> dict:
        if scope not in principal.scopes:
            raise PermissionError("Cannot write a scope outside your access")
        if not text.strip() or not 20 <= size <= 300 or not 0 <= overlap < size:
            raise ValueError("Empty text or invalid chunk settings")
        parts = text.split()
        if len(parts) > 10000:
            raise ValueError("Document exceeds 10000 words")
        revision = digest([text, title, entity, scope, size, overlap, self.encoder.name])
        with self.db.connect() as db:
            row = db.execute(
                "SELECT revision FROM documents WHERE tenant=? AND source=? AND active=1",
                (principal.tenant, source),
            ).fetchone()
            if row and row[0] == revision:
                return {"source": source, "revision": revision, "replayed": True}
            # An operator cannot downgrade or replace reviewer-only content.
            existing = db.execute(
                "SELECT scope FROM documents WHERE tenant=? AND source=? AND active=1",
                (principal.tenant, source),
            ).fetchone()
            if existing and existing[0] not in principal.scopes:
                raise PermissionError("Source is outside your scope")
        chunks = []
        for start in range(0, len(parts), size - overlap):
            chunks.append((start, " ".join(parts[start : start + size])))
            if start + size >= len(parts):
                break
        vectors = self.encoder.documents([text for _, text in chunks])
        with self.db.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Repeat ACL check under the write lock to prevent a concurrent scope downgrade.
            existing = db.execute(
                "SELECT scope FROM documents WHERE tenant=? AND source=? AND active=1",
                (principal.tenant, source),
            ).fetchone()
            if existing and existing[0] not in principal.scopes:
                raise PermissionError("Source is outside your scope")
            count = db.execute(
                "SELECT count(*) FROM documents WHERE tenant=?", (principal.tenant,)
            ).fetchone()[0]
            if count >= 500:
                raise ValueError("Tenant revision quota reached")
            db.execute(
                "UPDATE documents SET active=0 WHERE tenant=? AND source=?",
                (principal.tenant, source),
            )
            db.execute(
                "INSERT INTO documents(tenant,source,revision,title,entity,scope,active) "
                "VALUES(?,?,?,?,?,?,1) ON CONFLICT(tenant,source,revision) DO UPDATE SET active=1",
                (principal.tenant, source, revision, title, entity, scope),
            )
            for (position, text), vector in zip(chunks, vectors, strict=True):
                cid = digest([source, revision, position])
                db.execute(
                    "INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                    (
                        principal.tenant,
                        cid,
                        source,
                        revision,
                        position,
                        text,
                        json.dumps(vector),
                        self.encoder.name,
                    ),
                )
            self.db.audit(
                db,
                principal.tenant,
                principal.subject,
                "document_ingested",
                source,
                {"revision": revision, "chunks": len(chunks)},
            )
        return {"source": source, "revision": revision, "chunks": len(chunks), "replayed": False}

    def search(
        self,
        principal: Principal,
        query: str,
        entities: list[str] | None = None,
        limit: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        if not 1 <= limit <= 15 or mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError("Invalid retrieval settings")
        clauses = ["c.tenant=?", "d.active=1", "c.model=?"]
        params: list = [principal.tenant, self.encoder.name]
        clauses.append("d.scope IN (" + ",".join("?" for _ in principal.scopes) + ")")
        params.extend(principal.scopes)
        if entities:
            clauses.append("d.entity IN (" + ",".join("?" for _ in entities) + ")")
            params.extend(entities)
        with self.db.connect() as db:
            rows = db.execute(
                "SELECT c.*,d.title,d.entity,d.scope FROM chunks c JOIN documents d "
                "ON c.tenant=d.tenant AND c.source=d.source AND c.revision=d.revision WHERE "
                + " AND ".join(clauses)
                + " ORDER BY c.source,c.position",
                params,
            ).fetchall()
        if not rows or not tokens(query):
            return []
        matrix = np.asarray([json.loads(r["vector"]) for r in rows], dtype=np.float32)
        q = np.asarray(self.encoder.query(query), dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        similarity = matrix @ q
        lexical = BM25Okapi([tokens(r["text"]) for r in rows]).get_scores(tokens(query))
        overlap = [len(set(tokens(query)) & set(tokens(r["text"]))) for r in rows]
        ranks = (
            [similarity]
            if mode == "dense"
            else [lexical]
            if mode == "bm25"
            else [lexical, similarity]
        )
        score = np.zeros(len(rows))
        for values in ranks:
            for rank, index in enumerate(np.argsort(-np.asarray(values), kind="stable")[:40]):
                score[index] += 1 / (60 + rank + 1)

        def eligible(i: int) -> bool:
            if mode == "bm25":
                return overlap[i] > 0
            if mode == "dense":
                return float(similarity[i]) >= (
                    0.15 if self.encoder.name == HashEncoder.name else 0.55
                )
            return overlap[i] > 0 or (
                self.encoder.name != HashEncoder.name and similarity[i] >= 0.65
            )

        candidates = [int(i) for i in np.argsort(-score, kind="stable") if eligible(int(i))]
        result = []
        for i in candidates[:limit]:
            row = rows[i]
            result.append(
                {
                    k: row[k]
                    for k in ["id", "source", "revision", "position", "text", "title", "entity"]
                }
                | {
                    "rrf": float(score[i]),
                    "similarity": float(similarity[i]),
                    "lexical_overlap": overlap[i],
                    "embedding_model": self.encoder.name,
                }
            )
        return result

    def snapshot(self, principal: Principal, entities: list[str], connection=None) -> list[dict]:
        def read(db):
            sql = (
                "SELECT source,revision FROM documents WHERE tenant=? AND active=1 AND scope IN ("
                + ",".join("?" for _ in principal.scopes)
                + ") AND entity IN ("
                + ",".join("?" for _ in entities)
                + ") ORDER BY source"
            )
            return [
                dict(row)
                for row in db.execute(sql, [principal.tenant, *principal.scopes, *entities])
            ]

        if connection is not None:
            return read(connection)
        with self.db.connect() as db:
            return read(db)

    def current(self, db, principal: Principal, references: list[dict]) -> bool:
        for ref in references:
            row = db.execute(
                "SELECT scope,revision FROM documents WHERE tenant=? AND source=? AND active=1",
                (principal.tenant, ref["source"]),
            ).fetchone()
            if (
                not row
                or row["scope"] not in principal.scopes
                or row["revision"] != ref["revision"]
            ):
                return False
        return True


def validate_quotes(claims: list[dict], evidence: list[dict]) -> bool:
    indexed = {e["id"]: e for e in evidence}
    if not claims:
        return False
    for claim in claims:
        ref = indexed.get(claim.get("chunk_id"))
        quote = claim.get("quote")
        text = claim.get("text")
        if (
            not ref
            or not isinstance(quote, str)
            or not isinstance(text, str)
            or not quote.strip()
            or not text.strip()
            or quote not in ref["text"]
            or text not in quote
        ):
            return False
    return True
