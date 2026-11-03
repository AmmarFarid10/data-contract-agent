import pytest


def test_missing_auth_is_rejected(client):
    client.headers.pop("Authorization")
    assert (
        client.post("/documents", json={"source": "x", "title": "X", "text": "data"}).status_code
        == 401
    )


def test_tenant_and_role_cannot_be_selected_by_document_json(client):
    response = client.post(
        "/documents",
        json={"source": "x", "title": "X", "text": "data", "tenant": "other", "role": "reviewer"},
    )
    assert response.status_code == 422


def test_operator_cannot_write_reviewer_only_document(client):
    assert (
        client.post(
            "/documents",
            json={"source": "x", "title": "X", "scope": "reviewers", "text": "restricted"},
        ).status_code
        == 403
    )


@pytest.mark.parametrize("length", ["invalid", "-1", "2000001"])
def test_bad_or_excessive_content_length(client, length):
    response = client.post("/documents", content="{}", headers={"content-length": length})
    assert response.status_code in {400, 413}


def test_trace_header_and_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"


def test_ambiguous_duplicate_api_key_configuration_is_rejected(client, monkeypatch):
    import json

    from conftest import KEYS

    duplicate = [*KEYS, {**KEYS[0], "tenant": "another-tenant", "role": "reviewer"}]
    monkeypatch.setenv("AUTH_KEYS_JSON", json.dumps(duplicate))
    assert (
        client.post("/documents", json={"source": "x", "title": "X", "text": "data"}).status_code
        == 503
    )
