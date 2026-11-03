"""Print a shell export with fresh local-development keys. Never commit its output."""

import json
import secrets
import shlex

records = [
    {"tenant": "demo", "subject": subject, "role": role, "key": secrets.token_urlsafe(32)}
    for subject, role in [("operator-1", "operator"), ("reviewer-1", "reviewer")]
]
print("export AUTH_KEYS_JSON=" + shlex.quote(json.dumps(records)))
