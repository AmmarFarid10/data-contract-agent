import pytest

from app.core.database import Database
from app.core.retrieval import HashEncoder, Knowledge, validate_quotes
from app.core.security import Principal


@pytest.fixture
def knowledge(tmp_path):
    return Knowledge(Database(str(tmp_path / "knowledge.sqlite3")), HashEncoder())


def principal(tenant="a", role="operator"):
    return Principal(tenant, "alice", role)


def test_retrieval_filters_tenant_entity_and_acl_before_ranking(knowledge):
    knowledge.ingest(
        principal(),
        "visible",
        "Visible",
        "orders",
        "team",
        "Damaged switches require return inspection.",
    )
    knowledge.ingest(
        principal("b"), "foreign", "Foreign", "orders", "team", "Damaged switches SECRET FOREIGN."
    )
    knowledge.ingest(
        principal(role="reviewer"),
        "restricted",
        "Restricted",
        "orders",
        "reviewers",
        "Damaged switches SECRET REVIEWER.",
    )
    knowledge.ingest(
        principal(), "other", "Other", "invoices", "team", "Damaged switches SECRET ENTITY."
    )
    for mode in ["hybrid", "bm25", "dense"]:
        result = knowledge.search(principal(), "damaged switches", ["orders"], mode=mode)
        assert [r["source"] for r in result] == ["visible"]
        assert "SECRET" not in str(result)


def test_document_revisions_are_idempotent_and_old_chunks_are_not_retrieved(knowledge):
    a = knowledge.ingest(
        principal(), "policy", "Title", "orders", "team", "Original return policy for switches"
    )
    b = knowledge.ingest(
        principal(), "policy", "Title", "orders", "team", "Original return policy for switches"
    )
    assert b["replayed"] and b["revision"] == a["revision"]
    c = knowledge.ingest(
        principal(), "policy", "Title", "orders", "team", "Updated return policy for switches"
    )
    assert c["revision"] != a["revision"]
    assert all(
        r["revision"] == c["revision"] for r in knowledge.search(principal(), "return policy")
    )
    with knowledge.db.connect() as db:
        assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 2
        assert not knowledge.current(
            db, principal(), [{"source": "policy", "revision": a["revision"]}]
        )


def test_operator_cannot_downgrade_reviewer_document(knowledge):
    knowledge.ingest(
        principal(role="reviewer"),
        "policy",
        "Title",
        "orders",
        "reviewers",
        "Secret reviewer policy",
    )
    with pytest.raises(PermissionError):
        knowledge.ingest(principal(), "policy", "Title", "orders", "team", "Make this public")


def test_scope_change_reindexes_without_exposing_old_content(knowledge):
    reviewer = principal(role="reviewer")
    knowledge.ingest(reviewer, "policy", "Title", "orders", "team", "Shared return policy")
    knowledge.ingest(reviewer, "policy", "Title", "orders", "reviewers", "Shared return policy")
    assert knowledge.search(principal(), "return policy") == []
    assert knowledge.search(reviewer, "return policy")


@pytest.mark.parametrize("size,overlap,text", [(20, 20, "words"), (19, 0, "words"), (20, 2, " ")])
def test_invalid_chunk_configuration(knowledge, size, overlap, text):
    with pytest.raises(ValueError):
        knowledge.ingest(principal(), "a", "A", "orders", "team", text, size, overlap)


def test_quote_validation_rejects_fabrication():
    evidence = [{"id": "1", "text": "Warranty months: 24."}]
    assert validate_quotes(
        [{"chunk_id": "1", "quote": "Warranty months: 24.", "text": "Warranty months: 24."}],
        evidence,
    )
    assert not validate_quotes(
        [{"chunk_id": "1", "quote": "Warranty months: 24.", "text": "Free lifetime warranty"}],
        evidence,
    )
    assert not validate_quotes(
        [{"chunk_id": "fake", "quote": "Warranty months: 24.", "text": "Warranty months: 24."}],
        evidence,
    )
    assert not validate_quotes([{"chunk_id": "1", "quote": "", "text": ""}], evidence)


def test_database_rolls_back_failed_transaction(knowledge):
    with pytest.raises(RuntimeError):
        with knowledge.db.connect() as db:
            knowledge.db.audit(db, "a", "alice", "test", "id", {})
            raise RuntimeError("synthetic failure")
    with knowledge.db.connect() as db:
        assert db.execute("SELECT count(*) FROM audit").fetchone()[0] == 0
