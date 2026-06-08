"""Canonical JSON + sha256 hashing for LLM request structures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HashOpts:
    """Tunables for `hash_request` / `canonical_json`.

    Attributes:
        drop_keys: Exact dict-key names to drop at any depth. A key in
            `drop_keys` is removed from every dict it appears in. Match
            is case-sensitive and exact on the key string.

    The class method presets (`for_anthropic`, `for_openai`,
    `for_bedrock`, `for_gemini`) mirror the Rust crate's drop sets and
    target the response-side metadata that varies per call plus
    provider-specific request fields that do not change semantics.
    """

    drop_keys: set[str] = field(default_factory=set)

    @classmethod
    def for_anthropic(cls) -> HashOpts:
        """Drops `cache_control`, `id`, `usage`, `stop_reason`, `stop_sequence`.

        - `cache_control`: prompt-cache hint, no semantic effect.
        - `id`: assistant message ids differ per call.
        - `usage`, `stop_reason`, `stop_sequence`: response-side.
        """
        return cls(
            drop_keys={
                "cache_control",
                "id",
                "usage",
                "stop_reason",
                "stop_sequence",
            }
        )

    @classmethod
    def for_openai(cls) -> HashOpts:
        """Drops `created`, `id`, `object`, `system_fingerprint`, `usage`, `finish_reason`."""
        return cls(
            drop_keys={
                "created",
                "id",
                "object",
                "system_fingerprint",
                "usage",
                "finish_reason",
            }
        )

    @classmethod
    def for_bedrock(cls) -> HashOpts:
        """Drops `cache_control`, `usage`, `stopReason` (camelCase), `metrics`."""
        return cls(
            drop_keys={
                "cache_control",
                "usage",
                "stopReason",
                "metrics",
            }
        )

    @classmethod
    def for_gemini(cls) -> HashOpts:
        """Drops `usageMetadata`, `safetyRatings`, `finishReason`."""
        return cls(
            drop_keys={
                "usageMetadata",
                "safetyRatings",
                "finishReason",
            }
        )


def canonical_json(obj: Any, opts: HashOpts | None = None) -> str:
    """Return a deterministic compact JSON string for `obj`.

    Rules:
      - dict keys are sorted recursively (lexicographic on the string)
      - keys in `opts.drop_keys` are removed from every dict at any depth
      - list/tuple order is preserved
      - tuples become arrays (same as `json.dumps`)
      - no whitespace
      - strings use `json.dumps` escaping (so non-ASCII stays escaped
        for byte-stability across Python builds)

    Raises:
        ValueError: if the structure contains a non-finite float
            (`NaN`, `Infinity`, `-Infinity`). These have no valid JSON
            representation, so emitting them would produce non-portable
            output that the Rust sibling crate (and any spec-compliant
            JSON parser) rejects. Failing loudly avoids silently
            poisoning cache/idempotency keys.
    """
    canon = _canonicalize(obj, opts or HashOpts())
    # ensure_ascii=True keeps the byte sequence stable regardless of how
    # the source string was decoded; sort_keys + the recursive walk above
    # is belt-and-suspenders since we already sorted dicts.
    # allow_nan=False rejects NaN/Infinity, which json.dumps would
    # otherwise emit as bare tokens that are not valid JSON.
    return json.dumps(
        canon, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def hash_request(obj: Any, opts: HashOpts | None = None) -> str:
    """Return the lowercase-hex sha256 of `canonical_json(obj, opts)`."""
    payload = canonical_json(obj, opts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---- internals ----


def _json_key(k: Any) -> str:
    """Return the dict-key string exactly as `json.dumps` would emit it.

    `json.dumps` coerces non-string keys before serializing: `bool` ->
    `"true"`/`"false"`, `None` -> `"null"`, `int`/`float` -> their JSON
    number form. We must replicate that coercion here (rather than a bare
    `str(k)`) for two reasons:

      - Correctness: a bare `str()` would emit `"True"`/`"None"`, which is
        not valid JSON and would diverge from the Rust sibling crate.
      - Necessity: once we need to compare keys (for `drop_keys` and
        collision detection) we have to normalize them ourselves, because
        `json.dumps(sort_keys=True)` raises on a dict that mixes string
        and non-string keys.
    """
    if isinstance(k, str):
        return k
    if isinstance(k, bool):
        return "true" if k else "false"
    if k is None:
        return "null"
    if isinstance(k, int | float):
        # reuse json's own number formatting; rejects NaN/Infinity keys too
        return json.dumps(k, allow_nan=False)
    raise TypeError(f"unsupported dict key type for canonical JSON: {type(k).__name__}")


def _canonicalize(value: Any, opts: HashOpts) -> Any:
    """Recursively rebuild `value` with dropped keys removed.

    Dict-key sorting is delegated to `json.dumps(sort_keys=True)`. We
    only need to ensure (a) the structure has no dropped keys at any
    depth, (b) child dicts/lists are rebuilt the same way, and (c) keys
    are coerced to their JSON string form so a dict that mixes string and
    non-string keys still serializes deterministically.

    Raises:
        ValueError: if two distinct keys coerce to the same JSON string
            (e.g. the int `1` and the string `"1"`). Silently collapsing
            them would drop data and make two different inputs hash
            identically, defeating the purpose of the library.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = _json_key(k)
            if key in opts.drop_keys:
                continue
            if key in out:
                raise ValueError(
                    f"dict has colliding keys that map to the same JSON key {key!r}; "
                    "cannot produce a stable canonical form"
                )
            out[key] = _canonicalize(v, opts)
        return out
    if isinstance(value, list | tuple):
        return [_canonicalize(item, opts) for item in value]
    # leaves (str, int, float, bool, None) pass through untouched
    return value
