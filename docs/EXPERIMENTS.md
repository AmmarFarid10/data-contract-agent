# Retrieval experiments and failure modes

The committed JSON reports come from `scripts/evaluate.py` and `scripts/evaluate.py --semantic`, with the supplied synthetic documents and labeled questions. They measure source Hit@3 and MRR@3 after domain/entity filtering. They do not measure end-to-end business accuracy, deployed customer outcomes, or general LLM faithfulness. The corpus is intentionally small; thresholds and candidate settings need validation on a representative held-out corpus before deployment.

## What the experiments show
- A lexical/hash baseline is easy to reproduce but misses questions whose wording differs from the source. It is explicitly labeled, and must not be described as a pretrained semantic model.
- The pretrained 384-dimensional BGE small encoder runs on CPU and retrieves paraphrases. Its scores are similarities, not calibrated correctness probabilities.
- Hybrid fusion is not automatically better than dense search. Equal RRF weighting and the conservative hybrid candidate threshold can discard a relevant paraphrase or promote a lexical distractor. Inspect the per-mode JSON rather than claiming a universal winner.
- Dense-only candidates use a 0.55 similarity floor; hybrid admits lexical matches or a stronger semantic score of 0.65. This deliberate conservative setting can trade recall for fewer unrelated excerpts. Tune on independent positive and unanswerable cases, rather than maximizing this tiny fixture.
- Persisting vectors at ingestion avoids repeatedly embedding every document during each request. Query embedding and full vector scans are still latency costs; batching/caching and an ANN index become relevant at larger sizes.
- Retrieval relevance and answer support are different gates. Exact citations/quotes and domain rules reject fabricated evidence, but even a faithfully quoted source can be outdated, wrong or inapplicable.

## Domain safeguards worth demonstrating in an interview
Supplier requirements: missing currency/units and conflicting quotation values remain unknown; a new or changed source during comparison forces a retry.

CSV repair: the agent can propose only a small policy-backed operation set; an independent reviewer sees exact impact and remaining violations. Source changes invalidate approval. A simulated crash after effect commit is replayed safely from the authoritative ledger.

Support replies: customer/order access is tenant-scoped; restricted reviewer notes never enter customer-visible retrieval. Unsupported model claims cause escalation. A changed customer snapshot or policy set invalidates a pending reply, and no external message is sent.

## Model reproducibility
`MODEL_MANIFEST.json` records the actual downloaded files and hashes used in this review. We do not bundle weights. Package locking alone does not pin future hosted model revisions; retain an approved model cache and compare manifests in a controlled deployment, then re-ingest vectors when the encoder changes. Optional live Ollama output was not evaluated here; schema/transport/failure tests use explicit mocks and cannot establish model quality.
