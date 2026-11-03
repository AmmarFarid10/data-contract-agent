# Five-minute engineering walkthrough
1. Run `uv run python scripts/demo.py --check` and explain the concrete business input, result and explicit unknown/manual-review states.
2. Show `app/service.py` and the typed schemas: domain policy and verified data constrain model output. For agents, show actual tool calls and the persisted reviewer pause.
3. Compare `docs/LEXICAL_EVALUATION.json` with `docs/SEMANTIC_EVALUATION.json`. Explain why paraphrases need semantic retrieval and why hybrid is not always the winning ranker.
4. Open the regression tests for tenant isolation, stale sources, invented citations and (in the agent repos) approval replay/crash recovery. Explain exactly what each guarantee covers.
5. Discuss the limits before calling it production-ready: one worker, a small corpus, source trust, identity lifecycle, immutable model deployment, retention/backup, live model-quality evaluation and representative load testing.

Describe this as a portfolio reference implementation that you can run, inspect and extend. Do not claim commercial deployment, customer savings, production accuracy, or previous professional ownership that has not occurred. The committed demos, tests and experiment results are evidence of this implementation only.
