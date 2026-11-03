from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.core.database import Database, digest
from app.core.model import LocalModel, ModelError
from app.core.retrieval import Knowledge, validate_quotes
from app.core.security import Principal
from app.core.workflow import Conflict, Workflow
from app.quality import analyze, parse, proposed_operations, repair
from app.schemas import Contract, Dataset, RepairAdvice, ToolPlan


class State(TypedDict, total=False):
    run_id: str
    tenant: str
    creator: str
    role: str
    input: dict
    events: list[dict]
    calls: list[dict]
    profile: dict
    source_revision: str
    evidence: list[dict]
    handbook_snapshot: list[dict]
    proposal: dict
    proposal_hash: str
    expires_at: float
    decision: dict
    status: str


class DataQuality:
    def __init__(self, db: Database, knowledge: Knowledge, clock=None, model=None):
        self.db, self.knowledge, self.model = db, knowledge, model or LocalModel()
        with db.connect() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS datasets(tenant TEXT,id TEXT,revision TEXT,text TEXT,active INTEGER,
                PRIMARY KEY(tenant,id,revision));
            CREATE UNIQUE INDEX IF NOT EXISTS active_dataset ON datasets(tenant,id) WHERE active=1;
            CREATE TABLE IF NOT EXISTS artifacts(tenant TEXT,run_id TEXT,text TEXT,hash TEXT,
                PRIMARY KEY(tenant,run_id));
            """)
        options = {"clock": clock} if clock else {}
        self.workflow = Workflow(db, self.build, self.effect, **options)

    def upload(self, principal: Principal, body: Dataset) -> dict:
        header, rows = parse(body.csv_text)
        revision = digest(body.csv_text)
        with self.db.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM datasets WHERE tenant=?", (principal.tenant,)
            ).fetchone()[0]
            if count >= 100:
                raise ValueError("Dataset revision quota reached")
            db.execute(
                "UPDATE datasets SET active=0 WHERE tenant=? AND id=?",
                (principal.tenant, body.dataset_id),
            )
            db.execute(
                "INSERT INTO datasets VALUES(?,?,?,?,1) ON CONFLICT(tenant,id,revision) "
                "DO UPDATE SET active=1",
                (principal.tenant, body.dataset_id, revision, body.csv_text),
            )
            self.db.audit(
                db,
                principal.tenant,
                principal.subject,
                "dataset_uploaded",
                body.dataset_id,
                {"revision": revision, "rows": len(rows), "columns": len(header)},
            )
        return {"dataset_id": body.dataset_id, "revision": revision, "rows": len(rows)}

    def dataset(self, tenant: str, dataset_id: str):
        with self.db.connect() as db:
            row = db.execute(
                "SELECT text,revision FROM datasets WHERE tenant=? AND id=? AND active=1",
                (tenant, dataset_id),
            ).fetchone()
        if not row:
            raise ValueError("Dataset not found")
        return row["text"], row["revision"]

    def build(self, workflow: Workflow, saver):
        def plan(state: State) -> dict:
            names = ["inspect_schema", "profile_records", "retrieve_handbook"]
            expected = [{"name": n, "dataset_id": state["input"]["dataset_id"]} for n in names]
            calls = expected
            if state["input"].get("planner") == "ollama":
                result = self.model.structured(
                    ToolPlan,
                    "Plan these three read tools in dependency order: inspect_schema, profile_records, "
                    "retrieve_handbook. Use only the provided dataset ID. Never propose code execution.",
                    {"dataset_id": state["input"]["dataset_id"], "available_tools": names},
                )
                calls = [c.model_dump() for c in result.calls]
            if calls != expected:
                raise ModelError("Plan attempted an unknown tool, order or dataset")
            return {"calls": calls, "events": [{"step": "plan", "tool_count": len(calls)}]}

        def inspect(state: State) -> dict:
            principal = Principal(state["tenant"], state["creator"], state["role"])
            text, revision = self.dataset(state["tenant"], state["input"]["dataset_id"])
            handbook_snapshot = self.knowledge.snapshot(principal, ["data-quality"])
            contract = Contract.model_validate(state["input"]["contract"])
            profile = analyze(text, contract)

            @tool
            def inspect_schema(dataset_id: str) -> dict:
                """Return validated dataset column names and row count only."""
                if dataset_id != state["input"]["dataset_id"]:
                    raise ValueError("Dataset mismatch")
                return {"columns": profile["columns"], "rows": profile["rows"]}

            @tool
            def profile_records(dataset_id: str) -> dict:
                """Return aggregate contract violations without individual cell values."""
                if dataset_id != state["input"]["dataset_id"]:
                    raise ValueError("Dataset mismatch")
                return profile

            @tool
            def retrieve_handbook(dataset_id: str) -> dict:
                """Retrieve repair policy for the aggregate quality failure codes."""
                if dataset_id != state["input"]["dataset_id"]:
                    raise ValueError("Dataset mismatch")
                question = " ".join(i["code"].replace("_", " ") for i in profile["issues"])
                return {
                    "evidence": self.knowledge.search(
                        principal, question or "quality review", ["data-quality"], limit=5
                    )
                }

            registry = {t.name: t for t in [inspect_schema, profile_records, retrieve_handbook]}
            observations = []
            for call in state["calls"]:
                observations.append(
                    registry[call["name"]].invoke({"dataset_id": call["dataset_id"]})
                )
            return {
                "profile": profile,
                "source_revision": revision,
                "handbook_snapshot": handbook_snapshot,
                "evidence": observations[-1]["evidence"],
                "events": state["events"] + [{"step": c["name"]} for c in state["calls"]],
            }

        def propose(state: State) -> dict:
            text, revision = self.dataset(state["tenant"], state["input"]["dataset_id"])
            if revision != state["source_revision"]:
                raise Conflict("Dataset changed during investigation")
            candidates = proposed_operations(state["profile"])
            authorizations = {
                "trim_whitespace": "Whitespace cells may be trimmed",
                "drop_exact_duplicates": "Exact duplicate rows may be removed",
                "escape_formulas": "Spreadsheet formula cells must be escaped",
            }
            operations = [
                op
                for op in candidates
                if any(
                    authorizations[op].lower() in item["text"].lower() for item in state["evidence"]
                )
            ]
            model_claims = []
            if state["input"].get("planner") == "ollama" and operations:
                advice = self.model.structured(
                    RepairAdvice,
                    "Review aggregate data quality findings against the supplied repair handbook. "
                    "Choose only supported allowed operations. Cite exact policy quotes, with claim text "
                    "copied from each quote. List unresolved issue codes for manual review. Never invent "
                    "replacement cell values or executable transformations.",
                    {
                        "profile": state["profile"],
                        "allowed_operations": operations,
                        "policy_evidence": state["evidence"],
                    },
                )
                model_claims = [claim.model_dump() for claim in advice.claims]
                codes = {item["code"] for item in state["profile"]["issues"]}
                if (
                    not set(advice.operations) <= set(operations)
                    or len(advice.operations) != len(set(advice.operations))
                    or not set(advice.manual_issue_codes) <= codes
                    or not validate_quotes(model_claims, state["evidence"])
                ):
                    raise ModelError("Repair advice was not supported by policy evidence")
                operations = [op for op in operations if op in advice.operations]
            corrected, impact = repair(text, operations)
            remaining = analyze(corrected, Contract.model_validate(state["input"]["contract"]))
            proposal = {
                "dataset_id": state["input"]["dataset_id"],
                "source_revision": revision,
                "operations": operations,
                "model_policy_claims": model_claims,
                "impact": impact,
                "profile": state["profile"],
                "remaining_issues": remaining["issues"],
                "output_hash": digest(corrected),
                "evidence": state["evidence"],
                "handbook_snapshot": state["handbook_snapshot"],
                "events": state["events"],
            }
            return workflow.offer(dict(state), proposal)

        def review(state: State) -> dict:
            decision = interrupt(
                {
                    "proposal_hash": state["proposal_hash"],
                    "expires_at": state["expires_at"],
                    "proposal": state["proposal"],
                }
            )
            return {"decision": decision}

        def commit(state: State) -> dict:
            return workflow.finish(dict(state), state["decision"])

        graph = StateGraph(State)
        for name, fn in [
            ("plan", plan),
            ("inspect", inspect),
            ("propose", propose),
            ("review", review),
            ("commit", commit),
        ]:
            graph.add_node(name, fn)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "inspect")
        graph.add_edge("inspect", "propose")
        graph.add_edge("propose", "review")
        graph.add_edge("review", "commit")
        graph.add_edge("commit", END)
        return graph.compile(checkpointer=saver)

    def effect(self, db, principal: Principal, state: dict, proposal: dict) -> dict:
        row = db.execute(
            "SELECT text,revision FROM datasets WHERE tenant=? AND id=? AND active=1",
            (principal.tenant, proposal["dataset_id"]),
        ).fetchone()
        if not row or row["revision"] != proposal["source_revision"]:
            raise Conflict("Dataset changed after proposal")
        if (
            self.knowledge.snapshot(principal, ["data-quality"], db)
            != proposal["handbook_snapshot"]
        ):
            raise Conflict("Repair handbook changed after proposal")
        text, impact = repair(row["text"], proposal["operations"])
        if digest(text) != proposal["output_hash"] or impact != proposal["impact"]:
            raise Conflict("Repair preview no longer matches output")
        if not proposal["operations"]:
            return {
                "artifact_created": False,
                "reason": "No supported automatic repair",
                "remaining_issues": proposal["remaining_issues"],
            }
        db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?)",
            (principal.tenant, state["run_id"], text, digest(text)),
        )
        return {
            "artifact_created": True,
            "artifact_hash": digest(text),
            "impact": impact,
            "remaining_issues": proposal["remaining_issues"],
            "fully_repaired": not proposal["remaining_issues"],
        }
