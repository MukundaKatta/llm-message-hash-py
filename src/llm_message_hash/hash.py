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
    """
    canon = _canonicalize(obj, opts or HashOpts())
    # ensure_ascii=True keeps the byte sequence stable regardless of how
    # the source string was decoded; sort_keys + the recursive walk above
    # is belt-and-suspenders since we already sorted dicts.
    return json.dumps(canon, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def hash_request(obj: Any, opts: HashOpts | None = None) -> str:
    """Return the lowercase-hex sha256 of `canonical_json(obj, opts)`."""
    payload = canonical_json(obj, opts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---- internals ----


def _canonicalize(value: Any, opts: HashOpts) -> Any:
    """Recursively rebuild `value` with dropped keys removed.

    Dict-key sorting is delegated to `json.dumps(sort_keys=True)`. We
    only need to ensure (a) the structure has no dropped keys at any
    depth and (b) child dicts/lists are rebuilt the same way.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            # cast non-string keys to str the same way json.dumps would
            key = k if isinstance(k, str) else str(k)
            if key in opts.drop_keys:
                continue
            out[key] = _canonicalize(v, opts)
        return out
    if isinstance(value, list | tuple):
        return [_canonicalize(item, opts) for item in value]
    # leaves (str, int, float, bool, None) pass through untouched
    return value
