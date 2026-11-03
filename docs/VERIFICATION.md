# Verification — 3 November 2026

Python 3.12.14; `uv sync --frozen --extra dev` from a fresh project environment succeeded.

| Check | Result |
| --- | --- |
| Domain/boundary/provider/workflow tests | 46 passed; semantic test deselected explicitly |
| Actual pretrained embedding test | 1 passed; downloaded BGE small ONNX weights used |
| Ruff | Passed |
| Mypy | Passed with missing third-party imports ignored; not strict-mode certification |
| Dependency audit | No known advisories reported in the locked tested environment |
| End-to-end API demo | Passed against temporary storage with fresh synthetic identities |
| Lexical and pretrained retrieval ablations | Executed; raw JSON is committed in this folder |

No paid LLM API calls were made. Live Ollama generation was unavailable and is not represented as tested: local-model schema, retry, tool-plan and citation paths were tested with explicit mocks. Real pretrained embedding inference was executed, rather than mocked.

A Docker engine was not available in the authoring workspace. The repository's CI builds the image and runs the actual API demo inside the container; hosted run status is authoritative once uploaded. There is no claim of a deployed production service, load benchmark, external business-system integration or real customer result. Fixtures are synthetic.
