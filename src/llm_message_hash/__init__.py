"""llm-message-hash - stable canonical sha256 hash of LLM request structures.

Two semantically identical LLM requests can produce different
`sha256(json.dumps(req))` results because Python dict iteration order is
not stable in the way callers assume, and fields like `cache_control`
change the bytes without changing what gets sent to the model. This
library walks the value tree, sorts dict keys recursively, drops a
configurable set of fields, and sha256s the canonical bytes.

    from llm_message_hash import hash_request, HashOpts

    a = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"content": "hi", "role": "user"}], "model": "claude"}
    assert hash_request(a) == hash_request(b)

    # provider-specific noise stripped
    h = hash_request(req, HashOpts.for_anthropic())

For cache keys, idempotency keys, and dedupe.

Sibling to the Rust crate `llm-message-hash`.
"""

from llm_message_hash.hash import (
    HashOpts,
    canonical_json,
    hash_request,
)

__version__ = "0.1.0"

__all__ = [
    "HashOpts",
    "__version__",
    "canonical_json",
    "hash_request",
]
