# Boundary model
- Server configuration binds API keys to a tenant, subject and operator/reviewer role. Unknown/invalid credentials do not authorize business APIs.
- SQL uses parameters. Tenant and ACL filters apply before retrieval ranking; client-supplied tenant/role fields are rejected by schemas.
- Input IDs never become filesystem paths. Text/CSV uploads and request bodies are bounded; arbitrary URL crawling, SQL and code execution are absent.
- Model output is schema-checked and cannot grant permissions. Unknown tools or changed dataset targets fail closed. Cited claim text must be an exact substring of a retrieved quote.
- Agent projects require a different reviewer from the proposal creator; exact hash, expiry and source freshness are checked at the effect boundary. Vendor comparisons have no business side effect.
- Approved effects stay local. No project sends email, changes payment state or updates an external system.
- Environment keys, documents, CSV cells and model prompts are not application audit fields. Database contents and model caches still need access control and retention.

Do not expose a development instance publicly without TLS, identity lifecycle, rate limits, backup/restore procedures and appropriate monitoring. Public model/source inputs can be wrong or malicious; provenance checks are not a complete prompt-injection or security guarantee.

Optional LangChain tracing must be governed separately: enabling external tracing can export tool inputs/results. No external tracing is configured by this project.
