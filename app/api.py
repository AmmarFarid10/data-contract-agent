import csv
import io
import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from app.core.database import Database
from app.core.http import Decision, Document, ingest, instrument
from app.core.retrieval import Knowledge, configured_encoder
from app.core.security import Principal, authenticate
from app.quality import count_formula_headers
from app.schemas import Dataset, Investigation
from app.service import DataQuality


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(os.getenv("DATA_PATH", "data/quality.sqlite3"))
    app.state.knowledge = Knowledge(db, configured_encoder())
    app.state.service = DataQuality(db, app.state.knowledge)
    yield
    app.state.service.workflow.close()


app = FastAPI(title="Data Contract Agent", lifespan=lifespan)
instrument(app)
Identity = Annotated[Principal, Depends(authenticate)]
Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=100)]


@app.get("/health")
def health():
    return {"status": "ok", "service": "data-contract-agent"}


@app.post("/documents")
def documents(body: Document, request: Request, principal: Identity):
    return ingest(request.app, principal, body)


@app.post("/datasets", status_code=201)
def datasets(body: Dataset, request: Request, principal: Identity):
    try:
        return request.app.state.service.upload(principal, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/investigations", status_code=201)
def investigate(body: Investigation, request: Request, principal: Identity, key: Key):
    try:
        request.app.state.service.dataset(principal.tenant, body.dataset_id)
    except ValueError:
        raise HTTPException(404, "Dataset not found") from None
    return request.app.state.service.workflow.start(principal, body.model_dump(mode="json"), key)


@app.get("/investigations/{run_id}")
def get(run_id: UUID, request: Request, principal: Identity):
    return request.app.state.service.workflow.get(principal, str(run_id))


@app.post("/investigations/{run_id}/decision")
def decide(run_id: UUID, body: Decision, request: Request, principal: Identity):
    return request.app.state.service.workflow.decide(
        principal, str(run_id), body.approved, body.proposal_hash
    )


@app.get("/investigations/{run_id}/artifact")
def download(run_id: UUID, request: Request, principal: Identity):
    service = request.app.state.service
    service.workflow.get(principal, str(run_id))
    with service.db.connect() as db:
        row = db.execute(
            "SELECT text FROM artifacts WHERE tenant=? AND run_id=?",
            (principal.tenant, str(run_id)),
        ).fetchone()
    if not row:
        raise HTTPException(404, "No approved repair artifact")
    # Serialized repairs can exceed ingestion's byte limit after CSV quoting.
    # Inspect only the stored artifact's header; do not reapply upload limits.
    header = next(csv.reader(io.StringIO(row[0], newline="")), [])
    if count_formula_headers(header):
        raise HTTPException(
            409,
            "CSV download blocked: formula-like column names require manual correction "
            "of the source and contract, followed by a new investigation and approval.",
        )
    return Response(
        row[0],
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="reviewed-data.csv"'},
    )
