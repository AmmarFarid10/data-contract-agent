import json
from pathlib import Path

import pytest
from conftest import auth

from app.core.workflow import Workflow

SAMPLE = json.loads((Path(__file__).parent.parent / "examples/scenario.json").read_text())
PREFIX = "/investigations"


def seed(client):
    assert client.post("/documents", json=SAMPLE["document"]).status_code == 200
    csv_text = (Path(__file__).parent.parent / "examples/orders.csv").read_text()
    assert (
        client.post("/datasets", json={"dataset_id": "orders", "csv_text": csv_text}).status_code
        == 201
    )


def start(client, key="run-1", body=None):
    return client.post(
        PREFIX, json=body or SAMPLE["investigation"], headers={"Idempotency-Key": key}
    )


def decide(client, result, approved=True, hashed=None):
    return client.post(
        PREFIX + "/" + result["id"] + "/decision",
        json={"approved": approved, "proposal_hash": hashed or result["proposal_hash"]},
    )


def test_workflow_pauses_and_operator_cannot_approve(client):
    seed(client)
    result = start(client).json()
    assert result["status"] == "AWAITING_APPROVAL"
    assert decide(client, result).status_code == 403
    assert client.get(PREFIX + "/" + result["id"] + "/artifact").status_code == 404


def test_independent_reviewer_approval_is_idempotent(client):
    seed(client)
    result = start(client).json()
    auth(client, "b")
    first = decide(client, result)
    assert first.status_code == 200 and first.json()["status"] == "APPROVED"
    assert decide(client, result).json() == first.json()
    assert client.get(PREFIX + "/" + result["id"] + "/artifact").status_code == 200
    assert decide(client, result, approved=False).status_code == 409


def test_different_tenant_and_different_operator_cannot_read_or_decide(client):
    seed(client)
    result = start(client).json()
    for letter in ["c", "d"]:
        auth(client, letter)
        assert client.get(PREFIX + "/" + result["id"]).status_code == 404
        assert decide(client, result).status_code in {403, 404}


def test_decline_creates_no_effect(client):
    seed(client)
    result = start(client).json()
    auth(client, "b")
    assert decide(client, result, approved=False).json()["status"] == "REJECTED"
    assert client.get(PREFIX + "/" + result["id"] + "/artifact").status_code == 404


def test_expired_approval_creates_no_effect(client):
    seed(client)
    result = start(client).json()
    client.app.state.service.workflow.clock = lambda: result["expires_at"] + 1
    auth(client, "b")
    assert decide(client, result).json()["status"] == "EXPIRED"
    assert client.get(PREFIX + "/" + result["id"] + "/artifact").status_code == 404


def test_stale_hash_is_rejected_without_consuming_decision(client):
    seed(client)
    result = start(client).json()
    auth(client, "b")
    assert decide(client, result, hashed="0" * 64).status_code == 409
    assert decide(client, result).json()["status"] == "APPROVED"


def test_idempotency_key_is_bound_to_payload(client):
    seed(client)
    result = start(client).json()
    assert start(client).json()["id"] == result["id"]
    changed = {**SAMPLE["investigation"], "planner": "ollama"}
    assert start(client, body=changed).status_code == 409


def test_persisted_interrupt_survives_service_recreation(client):
    seed(client)
    result = start(client).json()
    service = client.app.state.service
    service.workflow.close()
    service.workflow = Workflow(service.db, service.build, service.effect)
    auth(client, "b")
    assert decide(client, result).json()["status"] == "APPROVED"


def test_same_person_cannot_approve_own_proposal_even_with_reviewer_role(client):
    seed(client)
    auth(client, "b")
    result = start(client).json()
    assert decide(client, result).status_code == 403


def test_boolean_approval_must_be_a_real_boolean(client):
    seed(client)
    result = start(client).json()
    auth(client, "b")
    response = client.post(
        PREFIX + "/" + result["id"] + "/decision",
        json={"approved": "yes", "proposal_hash": result["proposal_hash"]},
    )
    assert response.status_code == 422


def test_committed_effect_survives_crash_before_graph_checkpoint(client, monkeypatch):
    seed(client)
    result = start(client).json()
    workflow = client.app.state.service.workflow
    original = workflow.finish

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic crash after ledger commit")

    monkeypatch.setattr(workflow, "finish", interrupted)
    auth(client, "b")
    with pytest.raises(RuntimeError):
        decide(client, result)
    monkeypatch.setattr(workflow, "finish", original)
    assert decide(client, result).json()["status"] == "APPROVED"
    assert client.get(PREFIX + "/" + result["id"] + "/artifact").status_code == 200
