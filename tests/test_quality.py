from pathlib import Path

import pytest
from conftest import auth
from test_workflow import SAMPLE, decide, seed, start

from app.quality import analyze, parse, repair
from app.schemas import Contract, ToolPlan


@pytest.mark.parametrize("text", ["id,id\n1,2", "id,value\n1", "id,value\n1,2,3", "\n"])
def test_malformed_or_ambiguous_csv_is_rejected(text):
    with pytest.raises(ValueError):
        parse(text)


def test_nonfinite_numbers_are_type_errors():
    result = analyze(
        "amount\nNaN\ninf\n3\n",
        Contract.model_validate({"columns": [{"name": "amount", "dtype": "number"}]}),
    )
    assert result["issues"] == [{"code": "type_errors", "count": 2, "column": "amount"}]


def test_formula_escape_preserves_numeric_negatives():
    text, impact = repair("v\n-20\n=SUM(A1)\n+HYPERLINK\n", ["escape_formulas"])
    assert "-20" in text and "'=SUM" in text and "'+HYPERLINK" in text
    assert impact["cells_changed"] == 2


@pytest.mark.parametrize("header", ["=1+1", " +SUM(A1)", "\t@SUM(A1)", "-SUM(A1)"])
def test_formula_headers_are_reported_without_renaming_contract_columns(header):
    source = f"{header}\n=2+2\n"
    contract = Contract.model_validate({"columns": [{"name": header}]})
    output, impact = repair(source, ["escape_formulas"])
    result = analyze(output, contract)
    assert result["columns"] == [header]
    assert result["issues"] == [{"code": "spreadsheet_formula_headers", "count": 1, "column": None}]
    assert not result["passes_contract"]
    assert impact["cells_changed"] == 1


@pytest.mark.parametrize("header", ["=1+1", " +SUM(A1)", "\t@SUM(A1)", "-SUM(A1)", "-20"])
def test_formula_header_download_guard_preserves_source_and_approval(client, header):
    seed(client)
    source = f"{header}\n=2+2\n"
    assert (
        client.post("/datasets", json={"dataset_id": "headers", "csv_text": source}).status_code
        == 201
    )
    run = start(
        client, body={"dataset_id": "headers", "contract": {"columns": [{"name": header}]}}
    ).json()
    assert run["status"] == "AWAITING_APPROVAL"
    auth(client, "b")
    approved = decide(client, run).json()
    assert approved["status"] == "APPROVED"
    assert approved["result"]["artifact_created"]
    response = client.get("/investigations/" + run["id"] + "/artifact")
    if header == "-20":
        assert response.status_code == 200
        assert response.text == "-20\n'=2+2\n"
        assert approved["result"]["fully_repaired"]
    else:
        assert response.status_code == 409
        assert "formula-like column names" in response.json()["detail"]
        assert "Content-Disposition" not in response.headers
        assert not approved["result"]["fully_repaired"]
    assert client.app.state.service.dataset("demo", "headers")[0] == source
    assert decide(client, run).json() == approved


@pytest.mark.parametrize("header,status", [("=1+1", 409), ("name", 200)])
def test_download_checks_legacy_artifact_without_reapplying_upload_size_limit(
    client, header, status
):
    seed(client)
    run = start(client).json()
    auth(client, "b")
    assert decide(client, run).json()["status"] == "APPROVED"
    # Simulate a stored artifact from before header validation. CSV quoting and
    # escaping may grow an accepted upload beyond the ingestion byte limit.
    legacy_text = header + "\n" + "x" * 1000001 + "\n"
    with client.app.state.service.db.connect() as db:
        db.execute("UPDATE artifacts SET text=? WHERE run_id=?", (legacy_text, run["id"]))
    response = client.get("/investigations/" + run["id"] + "/artifact")
    assert response.status_code == status
    if status == 200:
        assert response.text == legacy_text
    auth(client, "c")
    assert client.get("/investigations/" + run["id"] + "/artifact").status_code == 404


def test_arbitrary_transformations_cannot_execute_code():
    with pytest.raises(ValueError):
        repair("id\n1", ["eval_python"])


def test_approval_creates_derived_artifact_and_exposes_unresolved_issues(client):
    seed(client)
    result = start(client).json()
    assert result["proposal"]["operations"] == SAMPLE["expected_operations"]
    assert result["proposal"]["remaining_issues"]
    auth(client, "b")
    approved = decide(client, result).json()
    assert approved["result"]["artifact_created"] and not approved["result"]["fully_repaired"]
    output = client.get("/investigations/" + result["id"] + "/artifact").text
    assert output.count("A-1") == 1
    with client.app.state.service.db.connect() as db:
        source = db.execute("SELECT text FROM datasets WHERE active=1").fetchone()[0]
    assert source == (Path(__file__).parent.parent / "examples/orders.csv").read_text()


def test_source_change_invalidates_approval(client):
    seed(client)
    result = start(client).json()
    client.post("/datasets", json={"dataset_id": "orders", "csv_text": "id\nnew source"})
    auth(client, "b")
    assert decide(client, result).json()["status"] == "CONFLICT"
    assert client.get("/investigations/" + result["id"] + "/artifact").status_code == 404


def test_new_handbook_document_invalidates_approval(client):
    seed(client)
    result = start(client).json()
    client.post(
        "/documents",
        json={
            "source": "new-rule",
            "title": "New rule",
            "entity": "data-quality",
            "text": "Quality repairs require additional approval.",
        },
    )
    auth(client, "b")
    assert decide(client, result).json()["status"] == "CONFLICT"


def test_missing_handbook_does_not_authorize_automatic_repairs(client):
    text = (Path(__file__).parent.parent / "examples/orders.csv").read_text()
    client.post("/datasets", json={"dataset_id": "orders", "csv_text": text})
    result = start(client).json()
    assert result["proposal"]["operations"] == []
    auth(client, "b")
    approved = decide(client, result).json()
    assert not approved["result"]["artifact_created"]


def test_model_cannot_change_dataset_target(client, monkeypatch):
    seed(client)
    fake = ToolPlan.model_validate(
        {
            "calls": [
                {"name": name, "dataset_id": "another-dataset"}
                for name in ["inspect_schema", "profile_records", "retrieve_handbook"]
            ]
        }
    )
    monkeypatch.setattr(client.app.state.service.model, "structured", lambda *_: fake)
    result = start(client, body={**SAMPLE["investigation"], "planner": "ollama"}).json()
    assert result["status"] == "FAILED" and result["proposal"] is None


def test_model_planning_and_policy_advice_use_aggregates_not_rows(client, monkeypatch):
    import json

    import httpx

    from app.core.model import LocalModel

    seed(client)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][-1]["content"])
        requests.append(data)
        if payload["format"]["title"] == "ToolPlan":
            answer = {
                "calls": [
                    {"name": name, "dataset_id": "orders"}
                    for name in ["inspect_schema", "profile_records", "retrieve_handbook"]
                ]
            }
        else:
            c = data["policy_evidence"][0]
            answer = {
                "operations": data["allowed_operations"],
                "claims": [{"text": c["text"], "quote": c["text"], "chunk_id": c["id"]}],
                "manual_issue_codes": ["type_errors", "enum_errors"],
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(answer)}})

    client.app.state.service.model = LocalModel(transport=httpx.MockTransport(handler))
    result = start(client, body={**SAMPLE["investigation"], "planner": "ollama"}).json()
    assert result["status"] == "AWAITING_APPROVAL"
    assert len(requests) == 2 and "profile" in requests[1]
    assert "=HYPERLINK" not in json.dumps(requests)
    assert result["proposal"]["model_policy_claims"]
