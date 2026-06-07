import math

import pytest

from llm_message_hash import HashOpts, canonical_json, hash_request

# ---------- canonical_json: structure ----------


def test_canonical_json_sorts_top_level_keys():
    assert canonical_json({"b": 1, "a": 2, "c": 3}) == '{"a":2,"b":1,"c":3}'


def test_canonical_json_sorts_nested_keys():
    obj = {"outer": {"z": 1, "a": 2, "m": 3}}
    assert canonical_json(obj) == '{"outer":{"a":2,"m":3,"z":1}}'


def test_canonical_json_preserves_list_order():
    assert canonical_json([3, 1, 2]) == "[3,1,2]"
    assert canonical_json({"xs": [3, 1, 2]}) == '{"xs":[3,1,2]}'


def test_canonical_json_handles_primitives():
    assert canonical_json(None) == "null"
    assert canonical_json(True) == "true"
    assert canonical_json(False) == "false"
    assert canonical_json(42) == "42"
    assert canonical_json(2.5) == "2.5"
    assert canonical_json("hi") == '"hi"'


def test_canonical_json_empty_collections():
    assert canonical_json({}) == "{}"
    assert canonical_json([]) == "[]"


def test_canonical_json_escapes_strings():
    # serde_json-style escaping for double-quote
    assert canonical_json('she said "hi"') == '"she said \\"hi\\""'


def test_canonical_json_utf8_strings_stable():
    # ensure_ascii=True means non-ASCII gets escaped as \uXXXX so the
    # byte sequence is stable across decoding modes
    assert canonical_json("héllo") == '"h\\u00e9llo"'
    assert canonical_json("漢字") == '"\\u6f22\\u5b57"'
    assert canonical_json({"name": "Renée"}) == '{"name":"Ren\\u00e9e"}'


# ---------- hash_request: stability ----------


def test_hash_is_stable_across_key_reordering():
    a = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"content": "hi", "role": "user"}], "model": "claude"}
    assert hash_request(a) == hash_request(b)


def test_hash_is_64_char_lowercase_hex():
    h = hash_request({"x": 1})
    assert len(h) == 64
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


def test_identical_objects_hash_identically():
    a = {"model": "gpt", "messages": [{"role": "system", "content": "be helpful"}]}
    b = {"model": "gpt", "messages": [{"role": "system", "content": "be helpful"}]}
    assert hash_request(a) == hash_request(b)


def test_different_objects_hash_differently():
    assert hash_request({"x": 1}) != hash_request({"x": 2})
    assert hash_request({"x": 1}) != hash_request({"y": 1})
    # case-sensitive on strings
    assert hash_request({"x": "hi"}) != hash_request({"x": "Hi"})
    # 42 vs 42.0 -> different JSON repr
    assert hash_request({"x": 42}) != hash_request({"x": 42.0})


def test_list_order_changes_hash():
    # arrays are structurally significant
    assert hash_request([1, 2, 3]) != hash_request([3, 2, 1])


def test_nested_object_key_reorder_still_stable():
    a = {"outer": {"a": {"b": {"c": 1, "d": 2}}}}
    b = {"outer": {"a": {"b": {"d": 2, "c": 1}}}}
    assert hash_request(a) == hash_request(b)


# ---------- HashOpts: default vs drop ----------


def test_default_opts_keeps_all_keys():
    obj = {"a": 1, "cache_control": {"type": "ephemeral"}, "b": 2}
    assert hash_request(obj) != hash_request({"a": 1, "b": 2})


def test_explicit_drop_keys_removes_field():
    obj = {"a": 1, "cache_control": {"type": "ephemeral"}, "b": 2}
    stripped = {"a": 1, "b": 2}
    opts = HashOpts(drop_keys={"cache_control"})
    assert hash_request(obj, opts) == hash_request(stripped, opts)
    # and equal to no-opts hash of stripped
    assert hash_request(obj, opts) == hash_request(stripped)


def test_drop_keys_applies_at_every_depth():
    obj = {
        "a": 1,
        "wrap": {"id": "noise-1", "inner": {"id": "noise-2", "real": "value"}},
    }
    stripped = {"a": 1, "wrap": {"inner": {"real": "value"}}}
    opts = HashOpts(drop_keys={"id"})
    assert hash_request(obj, opts) == hash_request(stripped)


def test_drop_keys_in_list_of_objects():
    # arrays-of-objects: drop should apply inside each object
    obj = {
        "messages": [
            {"role": "user", "content": "hi", "id": "msg-1"},
            {"role": "user", "content": "bye", "id": "msg-2"},
        ]
    }
    stripped = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
    }
    opts = HashOpts(drop_keys={"id"})
    assert hash_request(obj, opts) == hash_request(stripped)


# ---------- HashOpts presets ----------


def test_anthropic_preset_drops_cache_control_id_usage_stop():
    expected = {
        "cache_control",
        "id",
        "usage",
        "stop_reason",
        "stop_sequence",
    }
    assert HashOpts.for_anthropic().drop_keys == expected


def test_anthropic_preset_strips_cache_control_in_nested_content():
    with_cc = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    }
    without_cc = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    opts = HashOpts.for_anthropic()
    assert hash_request(with_cc, opts) == hash_request(without_cc, opts)


def test_openai_preset_drops_response_metadata():
    expected = {
        "created",
        "id",
        "object",
        "system_fingerprint",
        "usage",
        "finish_reason",
    }
    assert HashOpts.for_openai().drop_keys == expected
    a = {
        "model": "gpt-5.4",
        "id": "chatcmpl-abc",
        "created": 1700000000,
        "object": "chat.completion",
        "system_fingerprint": "fp_xyz",
        "usage": {"prompt_tokens": 5},
        "messages": [{"role": "user", "content": "hi", "finish_reason": "stop"}],
    }
    b = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hi"}],
    }
    opts = HashOpts.for_openai()
    assert hash_request(a, opts) == hash_request(b, opts)


def test_bedrock_preset_drops_camelcase_stopreason_and_metrics():
    expected = {"cache_control", "usage", "stopReason", "metrics"}
    assert HashOpts.for_bedrock().drop_keys == expected
    a = {
        "modelId": "anthropic.claude-sonnet",
        "stopReason": "end_turn",
        "metrics": {"latencyMs": 250},
        "usage": {"inputTokens": 10},
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    b = {
        "modelId": "anthropic.claude-sonnet",
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    opts = HashOpts.for_bedrock()
    assert hash_request(a, opts) == hash_request(b, opts)


def test_gemini_preset_drops_camelcase_response_fields():
    expected = {"usageMetadata", "safetyRatings", "finishReason"}
    assert HashOpts.for_gemini().drop_keys == expected
    a = {
        "model": "gemini-2.0",
        "usageMetadata": {"totalTokenCount": 12},
        "safetyRatings": [{"category": "HARM", "probability": "NEGLIGIBLE"}],
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "hi"}]}}],
    }
    b = {
        "model": "gemini-2.0",
        "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
    }
    opts = HashOpts.for_gemini()
    assert hash_request(a, opts) == hash_request(b, opts)


def test_presets_are_independent_instances():
    # mutating one preset's drop_keys must not leak into another call
    opts1 = HashOpts.for_anthropic()
    opts2 = HashOpts.for_anthropic()
    opts1.drop_keys.add("extra")
    assert "extra" not in opts2.drop_keys


# ---------- HashOpts: extension ----------


def test_extending_preset_with_extra_drop_key():
    opts = HashOpts.for_anthropic()
    opts.drop_keys.add("metadata")
    a = {"messages": [{"role": "user", "content": "hi"}], "metadata": {"user_id": "u1"}}
    b = {"messages": [{"role": "user", "content": "hi"}]}
    assert hash_request(a, opts) == hash_request(b, opts)


def test_drop_keys_only_matches_dict_keys_not_string_values():
    # "id" appears as a string value, not a key. it should not be dropped.
    obj = {"text": "id", "list": ["id", "x"]}
    opts = HashOpts(drop_keys={"id"})
    assert hash_request(obj, opts) == hash_request(obj)


# ---------- canonical_json: drop integration ----------


def test_canonical_json_with_drop_keys_emits_stripped_structure():
    obj = {"a": 1, "drop_me": "noise", "b": 2}
    opts = HashOpts(drop_keys={"drop_me"})
    assert canonical_json(obj, opts) == '{"a":1,"b":2}'


# ---------- canonical_json: non-finite floats ----------


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), math.nan, math.inf, -math.inf],
)
def test_canonical_json_rejects_non_finite_floats(bad):
    # NaN/Infinity have no valid JSON representation. Emitting them
    # silently would produce non-portable output (the Rust sibling crate
    # rejects them) and poison cache keys, so we fail loudly instead.
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


def test_hash_request_rejects_non_finite_floats():
    with pytest.raises(ValueError):
        hash_request({"score": float("inf")})


def test_canonical_json_rejects_non_finite_float_when_nested_in_list():
    with pytest.raises(ValueError):
        canonical_json({"xs": [1, 2, float("nan")]})


def test_canonical_json_allows_normal_floats():
    # finite floats are unaffected by the non-finite guard
    assert canonical_json({"x": 2.5}) == '{"x":2.5}'
    assert canonical_json({"x": 0.0}) == '{"x":0.0}'
    assert canonical_json({"x": -1.5}) == '{"x":-1.5}'
