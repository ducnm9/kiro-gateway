# -*- coding: utf-8 -*-

"""Command Code SSE stream parsing and non-stream collection.

Clean-room implementation of the Command Code streaming wire format, based
solely on observed behavior. No source was copied from third-party tools.
"""

import json
import re
import secrets
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import httpx
from fastapi import HTTPException

from kiro.dsml_parser import (
    has_complete_dsml_envelope,
    parse_dsml_tool_calls,
    repair_truncated_json,
)
from kiro.streaming_anthropic import (
    format_sse_event,
    generate_message_id,
    generate_thinking_signature,
)
from kiro.streaming_core import FIRST_TOKEN_TIMEOUT, KiroEvent
from kiro.utils import generate_completion_id


_PLAIN_TOOL_CALL_RE = re.compile(
    r"Assistant requested tool\s+([^\s(]+)\s+\(([^)]+)\)\s+with\s+"
    r"(?P<invalid>invalid\s+)?arguments:\s*(?P<args>.*)",
    re.DOTALL,
)


def _structured_tool_call(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a tool_use dict from a structured tool-call event."""
    call_id = data.get("toolCallId") or data.get("id") or ""
    name = data.get("toolName") or data.get("name") or ""
    if not name:
        return None
    input_raw = data.get("input")
    if input_raw is None:
        input_raw = data.get("args")
    if input_raw is None:
        input_raw = data.get("arguments")
    return _build_tool_use(call_id, name, input_raw)


def _build_tool_use(call_id: str, name: str, input_raw: Any) -> Optional[Dict[str, Any]]:
    """Build a tool_use dict with arguments normalized to a JSON string."""
    if isinstance(input_raw, str):
        repaired = repair_truncated_json(input_raw)
        args_str = repaired if repaired is not None else input_raw
    elif isinstance(input_raw, dict):
        args_str = json.dumps(input_raw)
    else:
        args_str = "{}"
    if not call_id:
        call_id = f"call_cc_{secrets.token_hex(9)}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _looks_like_tool_call_text(text: str) -> bool:
    """True if a text-delta chunk is a tool-call line rather than prose."""
    stripped = text.strip()
    return (
        stripped.startswith("Assistant requested tool ")
        or stripped.startswith("<tool_calls")
    )


def _extract_plain_text_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract OpenAI tool calls from plain-text 'Assistant requested tool' lines."""
    results: List[Dict[str, Any]] = []
    for match in _PLAIN_TOOL_CALL_RE.finditer(text):
        if match.group("invalid"):
            continue
        name = match.group(1)
        call_id = match.group(2)
        raw_args = match.group("args").strip()
        repaired = repair_truncated_json(raw_args)
        args_str = repaired if repaired is not None else raw_args
        results.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return results


def _emit_tool_use(tc: Dict[str, Any], seen_calls: Set[Any]) -> bool:
    """Dedupe tool calls; return True if this one should be emitted."""
    call_id = tc.get("id") or ""
    key = call_id or (
        tc.get("function", {}).get("name", ""),
        tc.get("function", {}).get("arguments", ""),
    )
    if key in seen_calls:
        return False
    seen_calls.add(key)
    return True


async def parse_cc_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True,
) -> AsyncGenerator[KiroEvent, None]:
    """Parse a Command Code SSE stream into unified KiroEvent objects.

    The live Command Code stream is NDJSON: one bare JSON object per line
    (no ``data:`` prefix). Each object's ``type`` field names the event. A
    ``data:`` prefix and ``[DONE]`` sentinel are tolerated for robustness.
    Unknown events are ignored; a missing sentinel is fine (the upstream may
    simply close the connection).

    Tool calls arrive in three formats and are reconstructed into
    ``tool_use`` events: structured ``tool-call`` events, buffered
    ``tool-input-*`` fragments, and text-embedded plain-text/DSML calls.

    Args:
        response: The upstream streaming HTTP response.
        first_token_timeout: Accepted for interface parity; unused.
        enable_thinking_parser: Accepted for interface parity; unused.

    Yields:
        KiroEvent objects (content, thinking, tool_use, usage with
        finish_reason).

    Raises:
        HTTPException: On a stream-level ``error`` event.
    """
    tool_buffers: Dict[str, Dict[str, Any]] = {}  # id -> {"name": str, "parts": List[str]}
    seen_calls: Set[Any] = set()  # dedupe by id and by (name, args)
    dsml_buffer = ""

    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        # Live Command Code streams NDJSON (bare JSON per line, no `data:`
        # prefix). Tolerate a `data:` prefix if present.
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = data.get("type", "")

        if event_type == "text-delta":
            text = data.get("text") or data.get("delta") or ""
            # Accumulate for DSML (may be fragmented); emit non-tool text as content
            dsml_buffer += text
            if has_complete_dsml_envelope(dsml_buffer):
                for tc in parse_dsml_tool_calls(dsml_buffer):
                    if _emit_tool_use(tc, seen_calls):
                        yield KiroEvent(type="tool_use", tool_use=tc)
                dsml_buffer = ""
            else:
                # Plain-text "Assistant requested tool ..." lines are tool calls,
                # not content. Emit remaining text as content otherwise.
                for tc in _extract_plain_text_tool_calls(text):
                    if _emit_tool_use(tc, seen_calls):
                        yield KiroEvent(type="tool_use", tool_use=tc)
                if not _looks_like_tool_call_text(text) and text:
                    yield KiroEvent(type="content", content=text)

        elif event_type == "reasoning-delta":
            yield KiroEvent(
                type="thinking",
                thinking_content=data.get("text") or data.get("delta") or "",
            )

        elif event_type == "tool-call":
            tc = _structured_tool_call(data)
            if tc and _emit_tool_use(tc, seen_calls):
                yield KiroEvent(type="tool_use", tool_use=tc)

        elif event_type == "tool-input-start":
            cid = data.get("id") or data.get("toolCallId") or ""
            tool_buffers[cid] = {"name": data.get("toolName") or "", "parts": []}

        elif event_type == "tool-input-delta":
            cid = data.get("id") or data.get("toolCallId") or ""
            buf = tool_buffers.setdefault(cid, {"name": data.get("toolName") or "", "parts": []})
            buf["parts"].append(data.get("delta") or "")
            if data.get("toolName"):
                buf["name"] = data["toolName"]

        elif event_type in ("tool-input-end", "tool-input-available", "tool-error"):
            cid = data.get("id") or data.get("toolCallId") or ""
            buf = tool_buffers.pop(cid, {"name": data.get("toolName") or "", "parts": []})
            name = data.get("toolName") or buf["name"] or ""
            input_raw = data.get("input")
            if input_raw is None:
                input_raw = "".join(buf["parts"]) or None
            tc = _build_tool_use(cid, name, input_raw)
            if tc and name and _emit_tool_use(tc, seen_calls):
                yield KiroEvent(type="tool_use", tool_use=tc)

        elif event_type == "finish":
            usage = data.get("totalUsage") or data.get("usage")
            yield KiroEvent(type="usage", usage=usage, finish_reason=data.get("finishReason"))

        elif event_type == "error":
            raise HTTPException(
                status_code=502,
                detail=f"Command Code stream error: {data.get('error')}",
            )

        # All other event types (finish-step, text-end, reasoning-end) are
        # intentionally ignored in this milestone.

    # Flush any remaining tool buffers and DSML buffer at EOF
    if dsml_buffer and has_complete_dsml_envelope(dsml_buffer):
        for tc in parse_dsml_tool_calls(dsml_buffer):
            if _emit_tool_use(tc, seen_calls):
                yield KiroEvent(type="tool_use", tool_use=tc)
    for cid, buf in tool_buffers.items():
        if not buf["parts"]:
            continue
        input_raw = "".join(buf["parts"])
        tc = _build_tool_use(cid, buf["name"], input_raw)
        if tc and buf["name"] and _emit_tool_use(tc, seen_calls):
            yield KiroEvent(type="tool_use", tool_use=tc)


def _map_finish_reason(raw: Optional[str]) -> str:
    """Map a Command Code finish reason to an OpenAI finish reason."""
    if raw in ("max_tokens", "max_output_tokens", "length"):
        return "length"
    if raw in ("tool-calls", "tool_calls"):
        return "tool_calls"
    if raw == "stop":
        return "stop"
    return raw or "stop"


def _map_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Map Command Code usage to OpenAI usage token counts."""
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("inputTokens", 0) or 0)
    completion_tokens = int(usage.get("outputTokens", 0) or 0)
    total_tokens = int(usage.get("totalTokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def collect_cc_response(response: httpx.Response, model: str) -> Dict[str, Any]:
    """Collect a Command Code stream into an OpenAI chat.completion dict.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Returns:
        An OpenAI ``chat.completion`` response dict.
    """
    content_parts = []
    thinking_parts = []
    tool_calls: List[Dict[str, Any]] = []
    usage = None
    finish_reason = None

    async for event in parse_cc_stream(response):
        if event.type == "content" and event.content:
            content_parts.append(event.content)
        elif event.type == "thinking" and event.thinking_content:
            thinking_parts.append(event.thinking_content)
        elif event.type == "tool_use" and event.tool_use:
            tool_calls.append(event.tool_use)
        elif event.type == "usage":
            if event.usage:
                usage = event.usage
            if event.finish_reason:
                finish_reason = event.finish_reason

    content = "".join(content_parts)
    thinking = "".join(thinking_parts)

    if tool_calls:
        finish_reason = "tool_calls"

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        message["reasoning_content"] = thinking
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": generate_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _map_finish_reason(finish_reason),
            }
        ],
        "usage": _map_usage(usage),
    }


async def stream_cc_to_openai(
    response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Convert a Command Code SSE stream to OpenAI chat.completion chunks.

    Command Code reasoning deltas map to native ``reasoning_content``. Tool
    calls are emitted as an indexed ``tool_calls`` delta chunk. The final
    chunk carries the finish reason (``tool_calls`` when tool calls are
    present) and real usage (from the ``finish`` event). The response is
    always closed on exit.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Yields:
        OpenAI SSE chunk strings ("data: {...}\\n\\n"), ending with "[DONE]".

    Raises:
        HTTPException: On a stream-level error event (from parse_cc_stream).
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = []

    try:
        async for event in parse_cc_stream(response):
            delta: Optional[Dict[str, Any]] = None
            if event.type == "content" and event.content:
                delta = {"content": event.content}
            elif event.type == "thinking" and event.thinking_content:
                delta = {"reasoning_content": event.thinking_content}
            elif event.type == "tool_use" and event.tool_use:
                tool_calls.append(event.tool_use)
                continue
            elif event.type == "usage":
                if event.usage:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason
                continue

            if delta is None:
                continue

            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Emit tool calls as an indexed delta chunk (OpenAI streaming spec)
        if tool_calls:
            indexed_tool_calls = []
            for idx, tc in enumerate(tool_calls):
                func = tc.get("function") or {}
                indexed_tool_calls.append({
                    "index": idx,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": func.get("name") or "",
                        "arguments": func.get("arguments") or "{}",
                    },
                })
            tool_calls_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": indexed_tool_calls},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(tool_calls_chunk, ensure_ascii=False)}\n\n"

        final_finish_reason = "tool_calls" if tool_calls else _map_finish_reason(finish_reason)
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": final_finish_reason,
                }
            ],
            "usage": _map_usage(usage),
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


def _map_stop_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    """Map a Command Code finish reason to an Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if finish_reason in ("max_tokens", "max_output_tokens", "length"):
        return "max_tokens"
    return "end_turn"


def _map_anthropic_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map Command Code usage to Anthropic usage fields."""
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    out = {
        "input_tokens": int(usage.get("inputTokens", 0) or 0),
        "output_tokens": int(usage.get("outputTokens", 0) or 0),
    }
    details = usage.get("inputTokenDetails") or {}
    if isinstance(details, dict):
        cr = details.get("cacheReadTokens")
        cw = details.get("noCacheTokens")  # live API reports cache misses, not writes
        if isinstance(cr, (int, float)):
            out["cache_read_input_tokens"] = int(cr)
        if isinstance(cw, (int, float)):
            out["cache_creation_input_tokens"] = int(cw)
    return out


async def collect_cc_anthropic_response(response: httpx.Response, model: str) -> Dict[str, Any]:
    """Collect a Command Code stream into an Anthropic message response dict."""
    content_parts = []
    thinking_parts = []
    tool_calls = []
    usage = None
    finish_reason = None
    async for event in parse_cc_stream(response):
        if event.type == "content" and event.content:
            content_parts.append(event.content)
        elif event.type == "thinking" and event.thinking_content:
            thinking_parts.append(event.thinking_content)
        elif event.type == "tool_use" and event.tool_use:
            tool_calls.append(event.tool_use)
        elif event.type == "usage":
            if event.usage:
                usage = event.usage
            if event.finish_reason:
                finish_reason = event.finish_reason
    content = "".join(content_parts)
    thinking = "".join(thinking_parts)

    blocks: List[Dict[str, Any]] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking, "signature": generate_thinking_signature()})
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, ValueError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": name, "input": args})

    return {
        "id": generate_message_id(),
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": _map_stop_reason(finish_reason, bool(tool_calls)),
        "stop_sequence": None,
        "usage": _map_anthropic_usage(usage),
    }


async def stream_cc_to_anthropic(response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Convert a Command Code SSE stream to Anthropic SSE events."""
    message_id = generate_message_id()
    # Track block indexes; text/thinking blocks may interleave, each gets a stable index.
    current_block_index = 0
    text_block_index = None
    thinking_block_index = None
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = None
    usage = None

    try:
        # 1. message_start (always first)
        yield format_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                # ponytail: CC reports real input tokens only at the terminal
                # finish event; Anthropic's stream spec only carries output_tokens
                # in message_delta, so stream-time input_tokens are 0. Non-stream
                # is exact. Revisit if CC exposes stream-time usage.
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        async for event in parse_cc_stream(response):
            if event.type == "thinking" and event.thinking_content:
                if thinking_block_index is None:
                    thinking_block_index = current_block_index
                    current_block_index += 1
                    yield format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": thinking_block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": generate_thinking_signature()},
                    })
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": thinking_block_index,
                    "delta": {"type": "thinking_delta", "thinking": event.thinking_content},
                })

            elif event.type == "content" and event.content:
                if text_block_index is None:
                    text_block_index = current_block_index
                    current_block_index += 1
                    yield format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": event.content},
                })

            elif event.type == "tool_use" and event.tool_use:
                tool_calls.append(event.tool_use)
                tc = event.tool_use
                name = tc.get("function", {}).get("name", "")
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, ValueError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                idx = current_block_index
                current_block_index += 1
                yield format_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "tool_use", "id": tc.get("id", ""), "name": name, "input": {}},
                })
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(args, ensure_ascii=False)},
                })
                yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})

            elif event.type == "usage":
                if event.usage:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason

        # Close open thinking/text blocks
        if thinking_block_index is not None:
            yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
        if text_block_index is not None:
            yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

        # Terminal events
        mapped_usage = _map_anthropic_usage(usage)
        yield format_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _map_stop_reason(finish_reason, bool(tool_calls)), "stop_sequence": None},
            "usage": {"output_tokens": mapped_usage["output_tokens"]},
        })
        yield format_sse_event("message_stop", {"type": "message_stop"})
    finally:
        try:
            await response.aclose()
        except Exception:
            pass
