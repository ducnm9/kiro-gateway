# -*- coding: utf-8 -*-

"""Antigravity (Cloud Code Assist) SSE stream parsing and response collection.

Clean-room implementation of the Cloud Code Assist streaming wire format.
Antigravity uses standard SSE (Server-Sent Events) with ``data:`` prefixed
lines containing JSON objects. Each JSON object wraps a ``candidates`` array
with content parts (text, thought, functionCall) and metadata.

This module provides:
- parse_antigravity_stream: Parse SSE into KiroEvent objects
- stream_antigravity_to_openai: Convert stream to OpenAI SSE chunks
- stream_antigravity_to_anthropic: Convert stream to Anthropic SSE events
- collect_antigravity_response: Collect stream into OpenAI non-streaming response
- collect_antigravity_anthropic_response: Collect stream into Anthropic response
"""

import json
import secrets
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import HTTPException

from kiro.streaming_anthropic import (
    format_sse_event,
    generate_message_id,
    generate_thinking_signature,
)
from kiro.streaming_core import KiroEvent
from kiro.utils import generate_completion_id


# ==============================================================================
# Stream Parser
# ==============================================================================


async def parse_antigravity_stream(
    response: httpx.Response,
) -> AsyncGenerator[KiroEvent, None]:
    """Parse an Antigravity SSE stream into unified KiroEvent objects.

    Antigravity streams use standard SSE format: each line starts with
    ``data: `` followed by a JSON object. The stream ends with ``data: [DONE]``
    or by connection close.

    Each JSON chunk has the structure:
    {
      "candidates": [{
        "content": {"parts": [...]},
        "finishReason": "STOP"
      }],
      "usageMetadata": {...}
    }

    Parts can be:
    - {"text": "..."} → content event
    - {"text": "...", "thought": true} → thinking event
    - {"functionCall": {"name": "...", "args": {...}, "id": "..."}} → tool_use event

    Args:
        response: The upstream streaming HTTP response.

    Yields:
        KiroEvent objects (content, thinking, tool_use, usage).

    Raises:
        HTTPException: On a stream-level error event.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue

        # Standard SSE: lines start with "data: "
        if line.startswith("data:"):
            line = line[5:].strip()
        elif not line.startswith("{"):
            # Skip non-data lines (comments, event types, etc.)
            continue

        if line == "[DONE]":
            break

        if not line.startswith("{"):
            continue

        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        # Handle error objects in the stream.
        error = data.get("error")
        if error:
            error_msg = error.get("message", "Unknown Antigravity stream error") if isinstance(error, dict) else str(error)
            raise HTTPException(status_code=502, detail=f"Antigravity stream error: {error_msg}")

        # The response can be nested under "response" or at the top level.
        response_data = data.get("response", data)
        candidates = response_data.get("candidates", [])

        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # Text content (regular or thinking).
                if "text" in part and part["text"] is not None:
                    text = part["text"]
                    is_thinking = part.get("thought") is True

                    if is_thinking:
                        yield KiroEvent(
                            type="thinking",
                            thinking_content=text,
                        )
                    elif text:
                        yield KiroEvent(type="content", content=text)

                # Function calls.
                if "functionCall" in part:
                    fc = part["functionCall"]
                    name = fc.get("name", "")
                    args = fc.get("args", {})
                    call_id = fc.get("id", "") or f"call_ag_{secrets.token_hex(9)}"

                    if name:
                        yield KiroEvent(
                            type="tool_use",
                            tool_use={
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                                },
                            },
                        )

            # Finish reason.
            finish_reason = candidate.get("finishReason")
            if finish_reason:
                yield KiroEvent(
                    type="usage",
                    finish_reason=_map_finish_reason_raw(finish_reason),
                )

        # Usage metadata.
        usage_metadata = response_data.get("usageMetadata")
        if usage_metadata:
            yield KiroEvent(
                type="usage",
                usage=_map_usage_metadata(usage_metadata),
            )


# ==============================================================================
# OpenAI Streaming Output
# ==============================================================================


async def stream_antigravity_to_openai(
    response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Convert an Antigravity SSE stream to OpenAI chat.completion.chunk format.

    Thinking deltas map to ``reasoning_content``. Tool calls are emitted as
    indexed ``tool_calls`` delta chunks. The final chunk carries finish_reason
    and usage.

    Args:
        response: The upstream streaming HTTP response.
        model: The client-supplied model name.

    Yields:
        OpenAI SSE chunk strings ("data: {...}\\n\\n"), ending with "data: [DONE]".
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = []

    try:
        async for event in parse_antigravity_stream(response):
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
            yield f"data: {json.dumps(chunk)}\n\n"

        # Emit tool call chunks.
        for idx, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            tc_delta = {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        },
                    }
                ]
            }
            if first_chunk:
                tc_delta["role"] = "assistant"
                first_chunk = False

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": tc_delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason and usage.
        final_reason = "tool_calls" if tool_calls else (finish_reason or "stop")
        final_chunk: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": final_reason}],
        }
        if usage:
            final_chunk["usage"] = usage
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        await response.aclose()


# ==============================================================================
# OpenAI Non-Streaming Collection
# ==============================================================================


async def collect_antigravity_response(
    response: httpx.Response, model: str
) -> Dict[str, Any]:
    """Collect an Antigravity stream into an OpenAI chat.completion response.

    Args:
        response: The upstream streaming HTTP response.
        model: The client-supplied model name.

    Returns:
        OpenAI chat.completion response dict.
    """
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    async for event in parse_antigravity_stream(response):
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
                "finish_reason": _map_finish_reason(finish_reason, bool(tool_calls)),
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ==============================================================================
# Anthropic Streaming Output
# ==============================================================================


async def stream_antigravity_to_anthropic(
    response: httpx.Response, model: str
) -> AsyncGenerator[str, None]:
    """Convert an Antigravity SSE stream to Anthropic Messages SSE events.

    Emits the full Anthropic streaming protocol: message_start,
    content_block_start, content_block_delta, content_block_stop,
    message_delta, message_stop.

    Args:
        response: The upstream streaming HTTP response.
        model: The client-supplied model name.

    Yields:
        Anthropic SSE event strings.
    """
    message_id = generate_message_id()
    current_block_index = 0
    text_block_index: Optional[int] = None
    thinking_block_index: Optional[int] = None
    tool_calls: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    try:
        # 1. message_start
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
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        async for event in parse_antigravity_stream(response):
            if event.type == "thinking" and event.thinking_content:
                if thinking_block_index is None:
                    thinking_block_index = current_block_index
                    current_block_index += 1
                    yield format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": thinking_block_index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": generate_thinking_signature(),
                        },
                    })
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": thinking_block_index,
                    "delta": {"type": "thinking_delta", "thinking": event.thinking_content},
                })

            elif event.type == "content" and event.content:
                if text_block_index is None:
                    # Close thinking block if open.
                    if thinking_block_index is not None:
                        yield format_sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": thinking_block_index,
                        })
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

            elif event.type == "usage":
                if event.usage:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason

        # Close open blocks.
        if thinking_block_index is not None and text_block_index is None:
            yield format_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": thinking_block_index,
            })
        if text_block_index is not None:
            yield format_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": text_block_index,
            })

        # Emit tool_use blocks.
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, ValueError):
                args = {}
            if not isinstance(args, dict):
                args = {}

            tool_block_index = current_block_index
            current_block_index += 1
            yield format_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": tool_block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{secrets.token_hex(12)}"),
                    "name": name,
                    "input": {},
                },
            })
            yield format_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": tool_block_index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)},
            })
            yield format_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": tool_block_index,
            })

        # message_delta with stop_reason and usage.
        stop_reason = _map_stop_reason_anthropic(finish_reason, bool(tool_calls))
        anthropic_usage = _map_anthropic_usage(usage)
        yield format_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": anthropic_usage.get("output_tokens", 0)},
        })

        # message_stop
        yield format_sse_event("message_stop", {"type": "message_stop"})

    finally:
        await response.aclose()


# ==============================================================================
# Anthropic Non-Streaming Collection
# ==============================================================================


async def collect_antigravity_anthropic_response(
    response: httpx.Response, model: str
) -> Dict[str, Any]:
    """Collect an Antigravity stream into an Anthropic Messages response.

    Args:
        response: The upstream streaming HTTP response.
        model: The client-supplied model name.

    Returns:
        Anthropic Messages response dict.
    """
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    async for event in parse_antigravity_stream(response):
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
        blocks.append({
            "type": "thinking",
            "thinking": thinking,
            "signature": generate_thinking_signature(),
        })
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, ValueError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{secrets.token_hex(12)}"),
            "name": name,
            "input": args,
        })

    return {
        "id": generate_message_id(),
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": _map_stop_reason_anthropic(finish_reason, bool(tool_calls)),
        "stop_sequence": None,
        "usage": _map_anthropic_usage(usage),
    }


# ==============================================================================
# Mapping Helpers
# ==============================================================================


def _map_finish_reason_raw(reason: str) -> str:
    """Map a raw Antigravity finish reason to a normalized string.

    Args:
        reason: Raw finish reason from the API (e.g. "STOP", "MAX_TOKENS").

    Returns:
        Normalized finish reason string.
    """
    reason_upper = reason.upper() if reason else ""
    if reason_upper == "STOP":
        return "stop"
    if reason_upper == "MAX_TOKENS":
        return "length"
    if reason_upper in ("SAFETY", "RECITATION", "OTHER"):
        return "stop"
    return "stop"


def _map_finish_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    """Map finish reason for OpenAI response format.

    Args:
        finish_reason: Normalized finish reason.
        has_tool_calls: Whether tool calls were emitted.

    Returns:
        OpenAI finish_reason string.
    """
    if has_tool_calls:
        return "tool_calls"
    if finish_reason == "length":
        return "length"
    return "stop"


def _map_stop_reason_anthropic(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    """Map finish reason for Anthropic response format.

    Args:
        finish_reason: Normalized finish reason.
        has_tool_calls: Whether tool calls were emitted.

    Returns:
        Anthropic stop_reason string.
    """
    if has_tool_calls:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _map_usage_metadata(usage_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Map Antigravity usageMetadata to OpenAI usage format.

    Args:
        usage_metadata: Raw usage metadata from Antigravity response.

    Returns:
        OpenAI-compatible usage dict.
    """
    prompt_tokens = int(usage_metadata.get("promptTokenCount", 0) or 0)
    cached_tokens = int(usage_metadata.get("cachedContentTokenCount", 0) or 0)
    candidates_tokens = int(usage_metadata.get("candidatesTokenCount", 0) or 0)
    thoughts_tokens = int(usage_metadata.get("thoughtsTokenCount", 0) or 0)
    total_tokens = int(usage_metadata.get("totalTokenCount", 0) or 0)

    completion_tokens = candidates_tokens + thoughts_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
    }


def _map_anthropic_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map usage to Anthropic format.

    Args:
        usage: OpenAI-style usage dict (from _map_usage_metadata).

    Returns:
        Anthropic-compatible usage dict.
    """
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
    }
