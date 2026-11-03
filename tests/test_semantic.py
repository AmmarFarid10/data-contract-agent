import os

import numpy as np
import pytest

from app.core.database import Database
from app.core.retrieval import Knowledge, SemanticEncoder
from app.core.security import Principal


@pytest.mark.semantic
@pytest.mark.skipif(
    os.getenv("RUN_SEMANTIC") != "1",
    reason="Set RUN_SEMANTIC=1 to download and run pretrained embeddings",
)
def test_pretrained_embeddings_retrieve_paraphrase_and_persist(tmp_path):
    encoder = SemanticEncoder()
    knowledge = Knowledge(Database(str(tmp_path / "semantic.sqlite3")), encoder)
    principal = Principal("demo", "operator", "operator")
    knowledge.ingest(
        principal,
        "refunds",
        "Refunds",
        "policy",
        "team",
        "Refunds are paid back to the original payment method after inspection.",
    )
    knowledge.ingest(
        principal,
        "delivery",
        "Delivery",
        "policy",
        "team",
        "Delivery tracking is available after parcel dispatch; shipping estimates vary by carrier.",
    )
    result = knowledge.search(
        principal, "How can I get my money back?", ["policy"], limit=2, mode="dense"
    )
    assert result and result[0]["source"] == "refunds"
    with knowledge.db.connect() as db:
        assert (
            db.execute("SELECT count(*) FROM chunks WHERE model=?", (encoder.name,)).fetchone()[0]
            == 2
        )
    assert np.asarray(encoder.query("test")).shape == (384,)
