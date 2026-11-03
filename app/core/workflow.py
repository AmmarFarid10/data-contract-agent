"""Durable proposal/decision ledger; all approved effects share its SQLite transaction."""

import json
import threading
import time
from uuid import uuid4

from fastapi import HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.core.database import Database, digest
from app.core.security import Principal, reviewer


class Conflict(ValueError):
    pass


TERMINAL = {"COMPLETED", "APPROVED", "REJECTED", "EXPIRED", "CONFLICT", "FAILED"}


class Workflow:
    def __init__(self, db: Database, builder, effect, clock=time.time):
        self.db, self.effect, self.clock = db, effect, clock
        self.lock = threading.RLock()
        import sqlite3

        self.connection = sqlite3.connect(db.path + ".checkpoints", check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.saver.setup()
        self.graph = builder(self, self.saver)

    def close(self):
        self.connection.close()

    def config(self, tenant: str, run_id: str) -> dict:
        return {"configurable": {"thread_id": tenant + ":" + run_id}, "recursion_limit": 12}

    def row(self, db, principal: Principal, run_id: str):
        row = db.execute(
            "SELECT * FROM runs WHERE tenant=? AND id=?", (principal.tenant, run_id)
        ).fetchone()
        if not row or (row["creator"] != principal.subject and principal.role != "reviewer"):
            raise HTTPException(404, "Run not found")
        return row

    def get(self, principal: Principal, run_id: str) -> dict:
        with self.db.connect() as db:
            row = self.row(db, principal, run_id)
            return {
                "id": row["id"],
                "status": row["status"],
                "creator": row["creator"],
                "proposal_hash": row["proposal_hash"],
                "expires_at": row["expires"],
                "proposal": json.loads(row["proposal"]) if row["proposal"] else None,
                "result": json.loads(row["result"]) if row["result"] else None,
                "reviewer": row["reviewer"],
            }

    def start(self, principal: Principal, payload: dict, key: str) -> dict:
        with self.lock:
            with self.db.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM runs WHERE tenant=? AND creator=? AND request_key=?",
                    (principal.tenant, principal.subject, key),
                ).fetchone()
                if row:
                    if row["input_hash"] != digest(payload):
                        raise HTTPException(409, "Idempotency key already used for different input")
                    run_id, status = row["id"], row["status"]
                else:
                    run_id, status = str(uuid4()), "RUNNING"
                    db.execute(
                        "INSERT INTO runs(tenant,id,creator,request_key,input_hash,input,status) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (
                            principal.tenant,
                            run_id,
                            principal.subject,
                            key,
                            digest(payload),
                            json.dumps(payload),
                            status,
                        ),
                    )
                    self.db.audit(
                        db, principal.tenant, principal.subject, "run_created", run_id, {}
                    )
            if status == "RUNNING":
                cfg = self.config(principal.tenant, run_id)
                snapshot = self.graph.get_state(cfg)
                initial = {
                    "run_id": run_id,
                    "tenant": principal.tenant,
                    "creator": principal.subject,
                    "role": principal.role,
                    "input": payload,
                    "events": [],
                }
                try:
                    self.graph.invoke(None if snapshot.values else initial, cfg)
                except (ValueError, RuntimeError) as exc:
                    # A safe category, never exception text which may contain input/provider output.
                    with self.db.connect() as db:
                        db.execute(
                            "UPDATE runs SET status='FAILED', result=? WHERE tenant=? AND id=? "
                            "AND status='RUNNING'",
                            (json.dumps({"error": type(exc).__name__}), principal.tenant, run_id),
                        )
            return self.get(principal, run_id)

    def offer(self, state: dict, proposal: dict) -> dict:
        hashed = digest(proposal)
        with self.db.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM runs WHERE tenant=? AND id=?", (state["tenant"], state["run_id"])
            ).fetchone()
            if row["proposal_hash"] and row["proposal_hash"] != hashed:
                raise Conflict("A proposal cannot change during checkpoint replay")
            expires = row["expires"] or self.clock() + 900
            if row["status"] == "RUNNING":
                db.execute(
                    "UPDATE runs SET status='AWAITING_APPROVAL',proposal=?,proposal_hash=?,"
                    "expires=? WHERE tenant=? AND id=?",
                    (json.dumps(proposal), hashed, expires, state["tenant"], state["run_id"]),
                )
                self.db.audit(
                    db,
                    state["tenant"],
                    state["creator"],
                    "proposal_created",
                    state["run_id"],
                    {"hash": hashed},
                )
        return {"proposal": proposal, "proposal_hash": hashed, "expires_at": expires}

    def decide(self, principal: Principal, run_id: str, approved: bool, hashed: str) -> dict:
        reviewer(principal)
        with self.lock:
            with self.db.connect() as db:
                row = self.row(db, principal, run_id)
                if row["creator"] == principal.subject:
                    raise HTTPException(403, "A different reviewer must decide")
                if row["proposal_hash"] != hashed:
                    raise HTTPException(409, "Proposal hash changed")
                if row["status"] in TERMINAL:
                    if row["reviewer"] == principal.subject and row["approved"] == int(approved):
                        return self.get(principal, run_id)
                    raise HTTPException(409, "Run already decided")
                if row["status"] != "AWAITING_APPROVAL":
                    raise HTTPException(409, "Run is not awaiting approval")
            cfg = self.config(principal.tenant, run_id)
            # Recover a crash after offer() but before the interrupt checkpoint.
            snapshot = self.graph.get_state(cfg)
            if not snapshot.interrupts:
                self.graph.invoke(None, cfg)
            self.graph.invoke(
                Command(
                    resume={
                        "approved": approved,
                        "hash": hashed,
                        "tenant": principal.tenant,
                        "subject": principal.subject,
                        "role": principal.role,
                    }
                ),
                cfg,
            )
            return self.get(principal, run_id)

    def finish(self, state: dict, decision: dict) -> dict:
        principal = Principal(decision["tenant"], decision["subject"], decision["role"])
        reviewer(principal)
        with self.db.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.row(db, principal, state["run_id"])
            if principal.subject == row["creator"] or decision["hash"] != row["proposal_hash"]:
                raise Conflict("Decision does not match proposal")
            if row["status"] in TERMINAL:
                if row["reviewer"] != principal.subject or row["approved"] != int(
                    decision["approved"]
                ):
                    raise Conflict("Conflicting replay")
                return {"status": row["status"]}
            proposal = json.loads(row["proposal"])
            if digest(proposal) != row["proposal_hash"]:
                raise Conflict("Stored proposal was modified")
            status, result = "REJECTED", {"reason": "Reviewer declined"}
            if self.clock() >= row["expires"]:
                status, result = "EXPIRED", {"reason": "Approval window expired"}
            elif decision["approved"]:
                db.execute("SAVEPOINT effect")
                try:
                    result = self.effect(db, principal, state, proposal)
                    status = "APPROVED"
                    db.execute("RELEASE SAVEPOINT effect")
                except Conflict as exc:
                    db.execute("ROLLBACK TO SAVEPOINT effect")
                    db.execute("RELEASE SAVEPOINT effect")
                    status, result = "CONFLICT", {"reason": str(exc)}
            db.execute(
                "UPDATE runs SET status=?,result=?,reviewer=?,approved=? WHERE tenant=? AND id=?",
                (
                    status,
                    json.dumps(result),
                    principal.subject,
                    int(decision["approved"]),
                    principal.tenant,
                    state["run_id"],
                ),
            )
            self.db.audit(
                db,
                principal.tenant,
                principal.subject,
                "decision_committed",
                state["run_id"],
                {"status": status, "hash": row["proposal_hash"]},
            )
        return {"status": status}
