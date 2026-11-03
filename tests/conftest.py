import json

import pytest
from fastapi.testclient import TestClient

from app.api import app

KEYS = [
    {"tenant": "demo", "subject": "alice", "role": "operator", "key": "a" * 32},
    {"tenant": "demo", "subject": "bob", "role": "reviewer", "key": "b" * 32},
    {"tenant": "foreign", "subject": "outsider", "role": "reviewer", "key": "c" * 32},
    {"tenant": "demo", "subject": "charlie", "role": "operator", "key": "d" * 32},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_KEYS_JSON", json.dumps(KEYS))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("EMBEDDER", "hash")
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer " + "a" * 32
        yield client


def auth(client, letter):
    client.headers["Authorization"] = "Bearer " + letter * 32
