import json
import sqlite3
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schema_info(version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO schema_info VALUES(1);
                CREATE TABLE IF NOT EXISTS documents(
                    tenant TEXT, source TEXT, revision TEXT, title TEXT, entity TEXT,
                    scope TEXT, active INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(tenant,source,revision));
                CREATE UNIQUE INDEX IF NOT EXISTS active_source
                    ON documents(tenant,source) WHERE active=1;
                CREATE TABLE IF NOT EXISTS chunks(
                    tenant TEXT, id TEXT, source TEXT, revision TEXT, position INTEGER,
                    text TEXT, vector TEXT, model TEXT,
                    PRIMARY KEY(tenant,id),
                    FOREIGN KEY(tenant,source,revision)
                    REFERENCES documents(tenant,source,revision));
                CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY, tenant TEXT, subject TEXT, action TEXT,
                    target TEXT, details TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS runs(
                    tenant TEXT, id TEXT, creator TEXT, request_key TEXT,
                    input_hash TEXT, input TEXT, status TEXT, proposal TEXT,
                    proposal_hash TEXT, expires REAL, result TEXT, reviewer TEXT,
                    approved INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(tenant,id), UNIQUE(tenant,creator,request_key));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=15000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def audit(db, tenant: str, subject: str, action: str, target: str, details: dict) -> None:
        # Callers pass identifiers/counts only; never keys, rows, documents or model prompts.
        db.execute(
            "INSERT INTO audit(tenant,subject,action,target,details) VALUES(?,?,?,?,?)",
            (tenant, subject, action, target, json.dumps(details)),
        )
