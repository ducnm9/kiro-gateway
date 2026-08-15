# -*- coding: utf-8 -*-

"""DSML tool-call parsing for Command Code responses.

Some Command Code models emit tool calls as a DSML (XML) envelope embedded in
text deltas. This module extracts and repairs those tool calls. Clean-room
implementation based solely on observed wire behavior.
"""

import json
import secrets
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

_DSML_ROOT = "tool_calls"
_DSML_CLOSE = "</tool_calls>"


def parse_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse complete DSML tool_calls envelopes into OpenAI tool_calls.

    Handles one or more ``<invoke name="...">`` elements. Parameter values
    with ``string="true"`` are strings; ``string="false"`` values are parsed
    as raw JSON. Returns an empty list if no complete envelope is present or
    the envelope is malformed (malformed envelopes are surfaced as literal
    text by the caller, never executed).

    Args:
        text: Raw text that may contain DSML envelope(s).

    Returns:
        List of tool-call dicts shaped as
        ``{"id", "type": "function", "function": {"name", "arguments"}}``
        where ``arguments`` is a JSON string.
    """
    results: List[Dict[str, Any]] = []
    search_from = 0
    while True:
        start = text.find(f"<{_DSML_ROOT}", search_from)
        if start == -1:
            break
        end = text.find(_DSML_CLOSE, start)
        if end == -1:
            break
        envelope = text[start:end + len(_DSML_CLOSE)]
        search_from = end + len(_DSML_CLOSE)
        try:
            root = ET.fromstring(envelope)
        except ET.ParseError:
            continue
        if root.tag != _DSML_ROOT:
            continue
        for invoke in root.findall("invoke"):
            name = invoke.get("name", "")
            if not name:
                continue
            args: Dict[str, Any] = {}
            for param in invoke.findall("parameter"):
                key = param.get("name", "")
                is_string = param.get("string", "false") == "true"
                raw_value = (param.text or "").strip()
                if is_string:
                    value: Any = raw_value
                else:
                    try:
                        value = json.loads(raw_value)
                    except (json.JSONDecodeError, ValueError):
                        value = raw_value
                args[key] = value
            call_id = f"call_dsml_{secrets.token_hex(9)}"
            results.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    return results


def has_complete_dsml_envelope(text: str) -> bool:
    """Return True if text contains a complete DSML tool_calls envelope."""
    start = text.find(f"<{_DSML_ROOT}")
    if start == -1:
        return False
    return text.find(_DSML_CLOSE, start) != -1


def strip_dsml_suffix(text: str) -> str:
    """Remove a leaked DSML suffix from otherwise-complete JSON arguments.

    Args:
        text: Raw JSON arguments possibly terminated by a DSML fragment.

    Returns:
        Text with any trailing DSML/XML fragment removed.
    """
    import re
    # Remove any trailing <...>...</...> fragment and surrounding whitespace
    cleaned = re.sub(r"\s*</?[a-zA-Z][^>]*>.*$", "", text, flags=re.DOTALL)
    return cleaned.strip()


def repair_truncated_json(text: str) -> Optional[str]:
    """Repair truncated JSON by balancing structural closers.

    Only appends missing closers — never fabricates field values. Returns a
    valid-JSON string, or None if unrecoverable.

    Args:
        text: Possibly-truncated JSON object text.

    Returns:
        Repaired JSON string or None.
    """
    stripped = strip_dsml_suffix(text)
    if not stripped:
        return None
    # Try as-is first
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        pass
    # Balance braces/brackets and quotes
    s = stripped
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
    if in_string:
        s += '"'
    while stack:
        opener = stack.pop()
        s += "}" if opener == "{" else "]"
    try:
        json.loads(s)
        return s
    except (json.JSONDecodeError, ValueError):
        return None
