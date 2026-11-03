"""Server-configured principals. Request JSON cannot select a tenant or role."""

import json
import os
import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException


@dataclass(frozen=True)
class Principal:
    tenant: str
    subject: str
    role: str

    @property
    def scopes(self) -> list[str]:
        return ["team", "reviewers"] if self.role == "reviewer" else ["team"]


def authenticate(authorization: Annotated[str | None, Header()] = None) -> Principal:
    try:
        configured = json.loads(os.environ.get("AUTH_KEYS_JSON", "[]"))
        if not isinstance(configured, list) or not configured:
            raise ValueError()
        matched = None
        for record in configured:
            key = record["key"]
            if not isinstance(key, str) or len(key) < 24:
                raise ValueError()
            principal = Principal(record["tenant"], record["subject"], record["role"])
            if (
                not principal.tenant
                or not principal.subject
                or principal.role not in {"operator", "reviewer"}
            ):
                raise ValueError()
            if secrets.compare_digest(authorization or "", "Bearer " + key):
                matched = principal
        if matched:
            return matched
    except (ValueError, KeyError, TypeError):
        raise HTTPException(503, "Configure valid AUTH_KEYS_JSON principals") from None
    raise HTTPException(401, "Authentication required")


def reviewer(principal: Principal) -> None:
    if principal.role != "reviewer":
        raise HTTPException(403, "Reviewer role required")
