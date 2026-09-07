# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""ChatGPT (Codex) Responses API SSE stream parsing and conversion.

Clean-room implementation of the Codex Responses streaming wire format, based
solely on observed behavior. Converts the Codex event stream into unified
``KiroEvent`` objects, then to OpenAI Chat Completions and Anthropic Messages
output (streaming and non-streaming).

The Codex Responses SSE stream is a sequence of ``data: {json}`` lines whose
JSON ``type`` field names the event:
- ``response.output_text.delta``            → content text delta
- ``response.reasoning_summary_text.delta`` → reasoning (thinking) delta
- ``response.reasoning_text.delta``         → reasoning (thinking) delta
- ``response.output_item.done`` (function_call) → a completed tool call
- ``response.completed``                    → usage + finish
- ``response.failed`` / ``error``           → stream error

Some transient errors are delivered inside a 200-OK body. A ``model_at_capacity``
error signals the caller to fail over to another account; this is surfaced via
``CodexSSEAccountFallbackError`` so the route layer can react.
"""

import json
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


# SSE-body error substrings that should trigger account failover (not a retry).
_ACCOUNT_FALLBACK_PATTERNS = ("selected model is at capacity", "model_at_capacity")
# SSE-body error substrings that indicate a transient overload.
_OVERLOADED_PATTERNS = ("server_is_overloaded", "service_unavailable_error")


class CodexSSEAccountFallbackError(Exception):
    """Raised when a 200-OK Codex stream body reports a capacity error.

    Signals the route layer to mark the current account unavailable and fail
    over to another account, mirroring the official Codex client behavior.
    """

    def __init__(self, message: str) -> None:
        """Initialize with the upstream capacity message."""
        super().__init__(message)
        self.message = message


def _extract_stream_error(data: Dict[str, Any]) -> Optional[str]:
    """Extract an error message from a Codex error event payload.

    Args:
        data: A parsed SSE event dict.

    Returns:
        The error message string, or None if not an error payload.
    """
    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        return err["message"]
    if isinstance(err, str) and err:
        return err
    resp = data.get("response")
    if isinstance(resp, dict):
        rerr = resp.get("error")
        if isinstance(rerr, dict) and rerr.get("message"):
            return rerr["message"]
    return None


async def parse_codex_stream(
    response: httpx.Response,
) -> AsyncGenerator[KiroEvent, None]:
    """Parse a Codex Responses SSE stream into unified KiroEvent objects.

    Args:
        response: The upstream streaming HTTP response.

    Yields:
        KiroEvent objects (content, thinking, tool_use, usage with finish_reason).

    Raises:
        CodexSSEAccountFallbackError: On a capacity error inside a 200-OK body.
        HTTPException: On a fatal stream-level error event.
    """
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or line.startswith("event:"):
            # Codex sends both `event:` and `data:` lines; we key off `data:`.
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            if line == "[DONE]":
                break
            continue
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = data.get("type", "")

        if event_type == "response.output_text.delta":
            text = data.get("delta") or ""
            if text:
                yield KiroEvent(type="content", content=text)

        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            text = data.get("delta") or ""
            if text:
                yield KiroEvent(type="thinking", thinking_content=text)

        elif event_type == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                name = item.get("name") or ""
                arguments = item.get("arguments")
                call_id = item.get("call_id") or item.get("id") or ""
                if name:
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments or {})
                    yield KiroEvent(type="tool_use", tool_use={
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments or "{}"},
                    })

        elif event_type == "response.completed":
            resp = data.get("response") or {}
            usage = resp.get("usage")
            status = resp.get("status")
            yield KiroEvent(type="usage", usage=usage, finish_reason=status)

        elif event_type in ("response.failed", "error"):
            message = _extract_stream_error(data) or "Codex stream error"
            lowered = message.lower()
            if any(p in lowered for p in _ACCOUNT_FALLBACK_PATTERNS):
                raise CodexSSEAccountFallbackError(message)
            raise HTTPException(status_code=502, detail=f"Codex stream error: {message}")

        # Other events (response.created, response.output_item.added,
        # response.output_text.done, ...) are intentionally ignored.


def _map_finish_reason(raw: Optional[str]) -> str:
    """Map a Codex response status to an OpenAI finish reason."""
    if raw in ("completed", "stop", None):
        return "stop"
    if raw in ("incomplete", "max_output_tokens", "length"):
        return "length"
    return "stop"


def _map_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Map Codex usage to OpenAI usage token counts."""
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(usage.get("input_tokens", 0) or 0)
    completion = int(usage.get("output_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _map_anthropic_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Map Codex usage to Anthropic usage fields."""
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def _map_stop_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    """Map a Codex response status to an Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if finish_reason in ("incomplete", "max_output_tokens", "length"):
        return "max_tokens"
    return "end_turn"


async def collect_codex_response(response: httpx.Response, model: str) -> Dict[str, Any]:
    """Collect a Codex stream into an OpenAI chat.completion dict.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Returns:
        An OpenAI ``chat.completion`` response dict.
    """
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    usage = None
    finish_reason = None

    async for event in parse_codex_stream(response):
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
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else _map_finish_reason(finish_reason),
        }],
        "usage": _map_usage(usage),
    }


async def stream_codex_to_openai(response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Convert a Codex SSE stream to OpenAI chat.completion.chunk SSE.

    Reasoning deltas map to ``reasoning_content``. Tool calls are emitted as an
    indexed ``tool_calls`` delta chunk. The final chunk carries the finish
    reason and usage. The response is always closed on exit.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Yields:
        OpenAI SSE chunk strings, ending with "[DONE]".

    Raises:
        CodexSSEAccountFallbackError: On a capacity error (account failover).
        HTTPException: On a fatal stream error.
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = []

    try:
        async for event in parse_codex_stream(response):
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

        if tool_calls:
            indexed = []
            for idx, tc in enumerate(tool_calls):
                func = tc.get("function") or {}
                indexed.append({
                    "index": idx,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": func.get("name") or "",
                        "arguments": func.get("arguments") or "{}",
                    },
                })
            tc_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": indexed}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls" if tool_calls else _map_finish_reason(finish_reason),
            }],
            "usage": _map_usage(usage),
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


async def collect_codex_anthropic_response(response: httpx.Response, model: str) -> Dict[str, Any]:
    """Collect a Codex stream into an Anthropic message response dict.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Returns:
        An Anthropic ``message`` response dict.
    """
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    usage = None
    finish_reason = None

    async for event in parse_codex_stream(response):
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


async def stream_codex_to_anthropic(response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Convert a Codex SSE stream to Anthropic Messages SSE events.

    Args:
        response: The upstream streaming HTTP response (must be open).
        model: The client-supplied model name.

    Yields:
        Anthropic SSE event strings.

    Raises:
        CodexSSEAccountFallbackError: On a capacity error (account failover).
        HTTPException: On a fatal stream error.
    """
    message_id = generate_message_id()
    current_block_index = 0
    text_block_index: Optional[int] = None
    thinking_block_index: Optional[int] = None
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = None
    usage = None

    try:
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

        async for event in parse_codex_stream(response):
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

        if thinking_block_index is not None:
            yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
        if text_block_index is not None:
            yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

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
