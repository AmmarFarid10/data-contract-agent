"""Small labeled synthetic retrieval ablation, not a production-quality benchmark."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.database import Database
from app.core.retrieval import HashEncoder, Knowledge, SemanticEncoder
from app.core.security import Principal

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    fixture = ROOT / "examples/vendors.json"
    if not fixture.exists():
        fixture = ROOT / "examples/scenario.json"
    sample = json.loads(fixture.read_text())
    documents = sample.get("documents", [sample.get("document")]) + sample.get("distractors", [])
    cases = sample["retrieval_cases"]
    encoder = SemanticEncoder() if args.semantic else HashEncoder()
    with tempfile.TemporaryDirectory() as temp:
        knowledge = Knowledge(Database(str(Path(temp) / "eval.sqlite3")), encoder)
        operator = Principal("demo", "alice", "operator")
        reviewer = Principal("demo", "bob", "reviewer")
        for doc in documents:
            knowledge.ingest(
                reviewer,
                doc["source"],
                doc["title"],
                doc["entity"],
                doc.get("scope", "team"),
                doc["text"],
            )
        results = []
        for mode in ["bm25", "dense", "hybrid"]:
            hits = []
            reciprocal = []
            for case in cases:
                evidence = knowledge.search(
                    operator, case["question"], [case["entity"]], limit=3, mode=mode
                )
                sources = [e["source"] for e in evidence]
                hit = case["source"] in sources
                hits.append(hit)
                reciprocal.append(1 / (sources.index(case["source"]) + 1) if hit else 0)
            results.append(
                {
                    "mode": mode,
                    "positive_queries": len(cases),
                    "hit_at_3": sum(hits) / len(hits),
                    "mrr_at_3": sum(reciprocal) / len(reciprocal),
                }
            )
        foreign = Principal("foreign", "outsider", "reviewer")
        assert not knowledge.search(foreign, cases[0]["question"], [cases[0]["entity"]])
        output = {
            "embedding_model": encoder.name,
            "documents": len(documents),
            "results": results,
            "cross_tenant_evidence_visible": False,
            "scope": "synthetic retrieval mechanics only",
        }
        print(json.dumps(output, indent=2))
        return output


if __name__ == "__main__":
    main()
