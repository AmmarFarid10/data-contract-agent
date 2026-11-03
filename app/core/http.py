import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from starlette.responses import JSONResponse

from app.core.security import Principal


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(StrictModel):
    source: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,100}$")
    title: str = Field(min_length=1, max_length=200)
    entity: str = Field(default="policy", pattern=r"^[a-zA-Z0-9_.-]{1,100}$")
    scope: Literal["team", "reviewers"] = "team"
    text: str = Field(min_length=1, max_length=100000)
    chunk_size: int = Field(default=140, ge=20, le=300)
    overlap: int = Field(default=25, ge=0, le=100)


class Decision(StrictModel):
    approved: StrictBool
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def ingest(app, principal: Principal, body: Document):
    try:
        return app.state.knowledge.ingest(
            principal,
            body.source,
            body.title,
            body.entity,
            body.scope,
            body.text,
            body.chunk_size,
            body.overlap,
        )
    except PermissionError:
        raise HTTPException(403, "Source scope is restricted") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


def instrument(app: FastAPI) -> None:
    @app.middleware("http")
    async def boundary(request: Request, call_next):
        started = time.monotonic()
        trace = str(uuid4())
        limit = 2_000_000
        length = request.headers.get("content-length")
        if length:
            try:
                size = int(length)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
            if size < 0 or size > limit:
                return JSONResponse({"detail": "Request size rejected"}, status_code=413)
        # Count actual bytes, including chunked uploads, before JSON parsing.
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > limit:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        request._body = bytes(content)
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace
        response.headers["Server-Timing"] = f"app;dur={(time.monotonic() - started) * 1000:.1f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response
