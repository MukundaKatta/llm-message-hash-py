# llm-message-hash-py

[![PyPI](https://img.shields.io/pypi/v/llm-message-hash-py.svg)](https://pypi.org/project/llm-message-hash-py/)
[![Python](https://img.shields.io/pypi/pyversions/llm-message-hash-py.svg)](https://pypi.org/project/llm-message-hash-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stable canonical sha256 hash of LLM request/message structures.**

Two semantically identical Anthropic requests can produce different
`sha256(json.dumps(req))` results because Python dict iteration order is
not part of the value, and fields like `cache_control` change the bytes
without changing what gets sent to the model. This library walks the
value tree, sorts dict keys recursively, drops a configurable set of
fields, and sha256s the canonical bytes.

Useful for prompt-cache lookups, idempotency keys, and dedupe.

Sibling to the Rust crate
[`llm-message-hash`](https://crates.io/crates/llm-message-hash).

## Install

```bash
pip install llm-message-hash-py
```

## Use

Default (no fields dropped):

```python
from llm_message_hash import hash_request

a = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}
b = {"messages": [{"content": "hi", "role": "user"}], "model": "claude"}

assert hash_request(a) == hash_request(b)
```

Per-provider preset (drops cache_control, response-only fields, etc.):

```python
from llm_message_hash import HashOpts, hash_request

with_cc = {
    "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
    }],
}
without_cc = {
    "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }],
}

h1 = hash_request(with_cc, HashOpts.for_anthropic())
h2 = hash_request(without_cc, HashOpts.for_anthropic())
assert h1 == h2
```

You can also get the canonical bytes directly:

```python
from llm_message_hash import canonical_json

s = canonical_json({"b": 1, "a": 2})
assert s == '{"a":2,"b":1}'
```

## Presets

Each preset drops the response-side metadata that varies per call plus
provider-specific request fields that do not change semantics:

| Preset | Drops |
| --- | --- |
| `HashOpts.for_anthropic()` | `cache_control`, `id`, `usage`, `stop_reason`, `stop_sequence` |
| `HashOpts.for_openai()` | `created`, `id`, `object`, `system_fingerprint`, `usage`, `finish_reason` |
| `HashOpts.for_bedrock()` | `cache_control`, `usage`, `stopReason`, `metrics` |
| `HashOpts.for_gemini()` | `usageMetadata`, `safetyRatings`, `finishReason` |

Extend any preset:

```python
opts = HashOpts.for_anthropic()
opts.drop_keys.add("metadata")
```

Or build your own:

```python
opts = HashOpts(drop_keys={"trace_id", "request_id"})
```

## Drop key behavior

`drop_keys` matches exact key names at any depth. A key named in
`drop_keys` is removed from every dict it appears in, no matter how
deeply nested. List order is preserved (a list is structurally
significant). Strings are case sensitive: `"hi"` and `"Hi"` hash
differently. Numbers compare by their JSON representation: `42` and
`42.0` are different strings and so hash differently.

Non-finite floats (`NaN`, `Infinity`, `-Infinity`) have no valid JSON
representation and raise `ValueError`, rather than silently emitting
non-portable output that would poison cache keys.

Non-string dict keys are coerced to their JSON form, matching
`json.dumps` (and the Rust sibling): `int`/`float` become their number
string, `True`/`False` become `"true"`/`"false"`, and `None` becomes
`"null"`. If two distinct keys would collapse to the same JSON key
(e.g. the int `1` and the string `"1"`), `ValueError` is raised instead
of silently dropping data and making different inputs hash alike.

## What it does NOT do

- No tokenization. The hash is over structure, not token count.
- No semantic equivalence beyond key-order normalization and the drop
  list.
- No streaming. Pass a complete Python object.
- No HTTP. Does not talk to any LLM provider.

## License

MIT
