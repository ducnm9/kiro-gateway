# -*- coding: utf-8 -*-

"""
Unit tests for the DSML tool-call parser (dsml_parser.py).

Tests:
- parse_dsml_tool_calls() envelope extraction (single/multiple/malformed/absent)
- has_complete_dsml_envelope()
- repair_truncated_json() structural repair
- strip_dsml_suffix() leaked-suffix removal
"""

import json

from kiro.dsml_parser import (
    has_complete_dsml_envelope,
    parse_dsml_tool_calls,
    repair_truncated_json,
    strip_dsml_suffix,
)


READ_ENVELOPE = (
    '<tool_calls><invoke name="read">'
    '<parameter name="offset" string="false">95</parameter>'
    '<parameter name="path" string="true">/tmp/protocol.ts</parameter>'
    '</invoke></tool_calls>'
)


class TestParseDsmlToolCalls:
    """Tests for parse_dsml_tool_calls()."""

    def test_single_invoke_mixed_parameter_types(self):
        """
        What it does: Verifies one invoke with int and string parameters is parsed.
        Purpose: Ensure string="false" values become raw JSON and string="true" stay strings.
        """
        results = parse_dsml_tool_calls(READ_ENVELOPE)

        assert len(results) == 1
        call = results[0]
        assert call["type"] == "function"
        assert call["id"].startswith("call_dsml_")
        assert call["function"]["name"] == "read"
        args = json.loads(call["function"]["arguments"])
        assert args == {"offset": 95, "path": "/tmp/protocol.ts"}
        assert isinstance(args["offset"], int)
        assert isinstance(args["path"], str)

    def test_multiple_invokes(self):
        """
        What it does: Verifies multiple invoke elements are all extracted.
        Purpose: Ensure a multi-tool envelope yields one call per invoke.
        """
        text = (
            '<tool_calls>'
            '<invoke name="bash"><parameter name="command" string="true">ls</parameter></invoke>'
            '<invoke name="read"><parameter name="path" string="true">/x</parameter></invoke>'
            '</tool_calls>'
        )

        results = parse_dsml_tool_calls(text)

        assert len(results) == 2
        assert results[0]["function"]["name"] == "bash"
        assert results[1]["function"]["name"] == "read"

    def test_malformed_envelope_returns_empty(self):
        """
        What it does: Verifies a malformed envelope returns an empty list.
        Purpose: Ensure malformed XML is never executed as a tool call.
        """
        results = parse_dsml_tool_calls("<tool_calls><invoke name='read'></tool_calls>")

        assert results == []

    def test_no_envelope_returns_empty(self):
        """
        What it does: Verifies text without a tool_calls envelope returns empty.
        Purpose: Ensure ordinary prose is never treated as a tool call.
        """
        results = parse_dsml_tool_calls("Just some ordinary text here.")

        assert results == []

    def test_false_string_json_array_parameter(self):
        """
        What it does: Verifies a string="false" array parameter parses to a JSON list.
        Purpose: Ensure non-string parameters are parsed as raw JSON structures.
        """
        text = (
            '<tool_calls><invoke name="f">'
            '<parameter name="items" string="false">[1, 2, 3]</parameter>'
            '</invoke></tool_calls>'
        )

        results = parse_dsml_tool_calls(text)

        args = json.loads(results[0]["function"]["arguments"])
        assert args["items"] == [1, 2, 3]


class TestHasCompleteDsmlEnvelope:
    """Tests for has_complete_dsml_envelope()."""

    def test_complete_envelope_true(self):
        """Verifies a complete envelope is detected."""
        assert has_complete_dsml_envelope(READ_ENVELOPE) is True

    def test_fragmented_envelope_false(self):
        """Verifies a fragmented envelope (no closing tag) is not complete."""
        assert has_complete_dsml_envelope('<tool_calls><invoke name="read">') is False

    def test_no_envelope_false(self):
        """Verifies text without any envelope returns False."""
        assert has_complete_dsml_envelope("hello") is False


class TestRepairTruncatedJson:
    """Tests for repair_truncated_json()."""

    def test_missing_closers_repaired(self):
        """
        What it does: Verifies truncated JSON gets its structural closers balanced.
        Purpose: Ensure truncated arguments are recoverable without fabricating values.
        """
        result = repair_truncated_json('{"command":"ls"')

        assert result is not None
        assert json.loads(result) == {"command": "ls"}

    def test_valid_json_passes_through(self):
        """Verifies already-valid JSON is returned unchanged."""
        result = repair_truncated_json('{"a": 1}')

        assert result == '{"a": 1}'

    def test_nested_missing_closers(self):
        """Verifies nested missing closers are balanced correctly."""
        result = repair_truncated_json('{"a":{"b":[1,2')

        assert result is not None
        assert json.loads(result) == {"a": {"b": [1, 2]}}

    def test_unrecoverable_returns_none(self):
        """Verifies non-JSON text returns None."""
        assert repair_truncated_json("not json at all") is None


class TestStripDsmlSuffix:
    """Tests for strip_dsml_suffix()."""

    def test_strips_leaked_dsml_suffix(self):
        """
        What it does: Verifies a leaked DSML suffix is removed from JSON arguments.
        Purpose: Ensure the leaked XML fragment does not corrupt arguments.
        """
        result = strip_dsml_suffix(
            '{"command":"ls"}</parameter></invoke></tool_calls>'
        )

        assert result == '{"command":"ls"}'

    def test_no_suffix_unchanged(self):
        """Verifies text without a suffix is unchanged."""
        assert strip_dsml_suffix('{"a":1}') == '{"a":1}'
