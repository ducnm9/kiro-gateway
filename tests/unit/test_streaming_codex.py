# -*- coding: utf-8 -*-

"""Unit tests for the Codex (ChatGPT) SSE stream parsing and conversion.

Covers:
- parse_codex_stream: text/reasoning deltas, function_call tool items, usage,
  and error handling (fatal vs. account-fallback capacity errors).
- collect_codex_response / stream_codex_to_openai (OpenAI output).
- collect_codex_anthropic_response / stream_codex_to_anthropic (Anthropic output).
- SSE-body capacity error → CodexSSEAccountFallbackError.

Network is fully isolated via a duck-typed streaming response stub.
"""

import json
from typing import Any, Dict, List

import pytest
from fastapi import HTTPException

from kiro.streaming_codex import (
    parse_codex_stream,
    collect_codex_response,
    collect_codex_anthropic_response,
    stream_codex_to_openai,
    stream_codex_to_anthropic,
    CodexSSEAccountFallbackError,
    _map_finish_reason,
    _map_usage,
    _map_stop_reason,
)


class FakeCodexResponse:
    """Duck-typed SSE response stub with an async line iterator."""

    def __init__(self, lines: List[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.closed = True


def _sse(payload: Dict[str, Any]) -> str:
    """Build a Codex SSE data line from a JSON payload."""
    return "data: " + json.dumps(payload)


def _text_delta(text: str) -> str:
    return _sse({"type": "response.output_text.delta", "delta": text})


def _reasoning_delta(text: str) -> str:
    return _sse({"type": "response.reasoning_summary_text.delta", "delta": text})


def _function_call(name: str, arguments: str, call_id: str = "c1") -> str:
    return _sse({
        "type": "response.output_item.done",
        "item": {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id},
    })


def _completed(usage: Dict[str, Any] = None, status: str = "completed") -> str:
    return _sse({"type": "response.completed", "response": {"status": status, "usage": usage or {}}})


async def _collect_events(lines):
    events = []
    async for event in parse_codex_stream(FakeCodexResponse(lines)):
        events.append(event)
    return events


# =============================================================================
# parse_codex_stream
# =============================================================================

class TestParseCodexStream:
    """Tests for the low-level Codex SSE parser."""

    @pytest.mark.asyncio
    async def test_text_deltas(self):
        """
        What it does: output_text.delta events become content events.
        Purpose: Core text streaming.
        """
        events = await _collect_events([_text_delta("Hello"), _text_delta(" world"), _completed()])
        contents = [e.content for e in events if e.type == "content"]
        assert contents == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_reasoning_deltas(self):
        """
        What it does: reasoning deltas become thinking events.
        Purpose: Extended thinking support.
        """
        events = await _collect_events([_reasoning_delta("thinking..."), _completed()])
        thinking = [e.thinking_content for e in events if e.type == "thinking"]
        assert thinking == ["thinking..."]

    @pytest.mark.asyncio
    async def test_reasoning_text_delta_variant(self):
        """
        What it does: response.reasoning_text.delta is also treated as thinking.
        Purpose: Handle both reasoning event variants.
        """
        line = _sse({"type": "response.reasoning_text.delta", "delta": "r"})
        events = await _collect_events([line, _completed()])
        assert any(e.type == "thinking" and e.thinking_content == "r" for e in events)

    @pytest.mark.asyncio
    async def test_function_call_tool_use(self):
        """
        What it does: A function_call output item becomes a tool_use event.
        Purpose: Tool call extraction.
        """
        events = await _collect_events([_function_call("get_weather", '{"city":"paris"}'), _completed()])
        tools = [e.tool_use for e in events if e.type == "tool_use"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["function"]["arguments"] == '{"city":"paris"}'
        assert tools[0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_function_call_dict_arguments_serialized(self):
        """
        What it does: Non-string arguments are JSON-serialized.
        Purpose: Robustness to argument encoding.
        """
        line = _sse({
            "type": "response.output_item.done",
            "item": {"type": "function_call", "name": "f", "arguments": {"a": 1}, "call_id": "c2"},
        })
        events = await _collect_events([line, _completed()])
        tool = [e.tool_use for e in events if e.type == "tool_use"][0]
        assert json.loads(tool["function"]["arguments"]) == {"a": 1}

    @pytest.mark.asyncio
    async def test_usage_and_finish(self):
        """
        What it does: response.completed yields a usage event with status.
        Purpose: Usage + finish reason extraction.
        """
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        events = await _collect_events([_text_delta("hi"), _completed(usage, "completed")])
        usage_events = [e for e in events if e.type == "usage"]
        assert usage_events and usage_events[0].usage == usage
        assert usage_events[0].finish_reason == "completed"

    @pytest.mark.asyncio
    async def test_done_sentinel_stops(self):
        """
        What it does: A [DONE] sentinel ends parsing.
        Purpose: Respect stream termination.
        """
        events = await _collect_events([_text_delta("a"), "data: [DONE]", _text_delta("b")])
        contents = [e.content for e in events if e.type == "content"]
        assert contents == ["a"]

    @pytest.mark.asyncio
    async def test_event_lines_and_blanks_ignored(self):
        """
        What it does: 'event:' lines, blank lines, and non-JSON are ignored.
        Purpose: Robust SSE parsing.
        """
        lines = ["event: response.output_text.delta", "", "garbage", _text_delta("x"), _completed()]
        events = await _collect_events(lines)
        assert [e.content for e in events if e.type == "content"] == ["x"]

    @pytest.mark.asyncio
    async def test_fatal_error_raises_http_exception(self):
        """
        What it does: A non-capacity error event raises HTTPException.
        Purpose: Surface fatal stream errors.
        """
        line = _sse({"type": "error", "error": {"message": "boom"}})
        with pytest.raises(HTTPException):
            await _collect_events([line])

    @pytest.mark.asyncio
    async def test_capacity_error_raises_account_fallback(self):
        """
        What it does: A model_at_capacity error raises CodexSSEAccountFallbackError.
        Purpose: Signal account failover on 200-OK body capacity errors.
        """
        line = _sse({"type": "response.failed", "response": {
            "error": {"message": "Selected model is at capacity. Please try a different model."}
        }})
        with pytest.raises(CodexSSEAccountFallbackError):
            await _collect_events([line])


# =============================================================================
# Mapping helpers
# =============================================================================

class TestMappingHelpers:
    """Tests for finish-reason / usage / stop-reason mapping."""

    @pytest.mark.parametrize("raw,expected", [
        ("completed", "stop"), ("stop", "stop"), (None, "stop"),
        ("incomplete", "length"), ("max_output_tokens", "length"),
    ])
    def test_map_finish_reason(self, raw, expected):
        assert _map_finish_reason(raw) == expected

    def test_map_usage_totals(self):
        """total_tokens computed when absent."""
        assert _map_usage({"input_tokens": 3, "output_tokens": 4}) == {
            "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
        }

    def test_map_usage_empty(self):
        assert _map_usage(None) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_map_stop_reason_tool_use(self):
        assert _map_stop_reason("completed", True) == "tool_use"

    def test_map_stop_reason_max_tokens(self):
        assert _map_stop_reason("incomplete", False) == "max_tokens"

    def test_map_stop_reason_end_turn(self):
        assert _map_stop_reason("completed", False) == "end_turn"


# =============================================================================
# collect_codex_response (OpenAI non-streaming)
# =============================================================================

class TestCollectCodexResponseOpenAI:
    """Tests for OpenAI non-streaming collection."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        """
        What it does: Collects text + usage into a chat.completion dict.
        Purpose: Non-streaming OpenAI output.
        """
        lines = [_text_delta("Hello"), _text_delta(" there"),
                 _completed({"input_tokens": 2, "output_tokens": 3})]
        result = await collect_codex_response(FakeCodexResponse(lines), "gpt-5.5")
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello there"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 5

    @pytest.mark.asyncio
    async def test_tool_calls_set_finish_reason(self):
        """
        What it does: Tool calls set finish_reason=tool_calls and include tool_calls.
        Purpose: Tool-call response shape.
        """
        lines = [_function_call("f", "{}"), _completed()]
        result = await collect_codex_response(FakeCodexResponse(lines), "gpt-5.5")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"

    @pytest.mark.asyncio
    async def test_reasoning_content_included(self):
        """
        What it does: Reasoning deltas populate reasoning_content.
        Purpose: Expose thinking in non-streaming output.
        """
        lines = [_reasoning_delta("hmm"), _text_delta("answer"), _completed()]
        result = await collect_codex_response(FakeCodexResponse(lines), "gpt-5.5")
        assert result["choices"][0]["message"]["reasoning_content"] == "hmm"


# =============================================================================
# stream_codex_to_openai (OpenAI streaming)
# =============================================================================

async def _collect_sse(gen):
    chunks = []
    async for c in gen:
        chunks.append(c)
    return chunks


class TestStreamCodexToOpenAI:
    """Tests for OpenAI streaming output."""

    @pytest.mark.asyncio
    async def test_streams_text_and_done(self):
        """
        What it does: Text deltas stream as chunks and end with [DONE].
        Purpose: Core streaming contract.
        """
        lines = [_text_delta("Hi"), _completed({"input_tokens": 1, "output_tokens": 1})]
        resp = FakeCodexResponse(lines)
        chunks = await _collect_sse(stream_codex_to_openai(resp, "gpt-5.5"))
        assert chunks[-1] == "data: [DONE]\n\n"
        # First content chunk carries role=assistant.
        first = json.loads(chunks[0][6:])
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["delta"]["content"] == "Hi"
        assert resp.closed is True

    @pytest.mark.asyncio
    async def test_tool_calls_chunk_emitted(self):
        """
        What it does: Tool calls are emitted as an indexed tool_calls delta chunk.
        Purpose: OpenAI streaming tool-call spec.
        """
        lines = [_function_call("f", '{"x":1}'), _completed()]
        chunks = await _collect_sse(stream_codex_to_openai(FakeCodexResponse(lines), "gpt-5.5"))
        joined = "".join(chunks)
        assert "tool_calls" in joined
        # Final chunk finish_reason should be tool_calls.
        final = json.loads([c for c in chunks if c != "data: [DONE]\n\n"][-1][6:])
        assert final["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_capacity_error_propagates(self):
        """
        What it does: A capacity error propagates as CodexSSEAccountFallbackError.
        Purpose: Streaming path also supports account failover.
        """
        line = _sse({"type": "response.failed", "response": {
            "error": {"message": "model_at_capacity"}}})
        with pytest.raises(CodexSSEAccountFallbackError):
            await _collect_sse(stream_codex_to_openai(FakeCodexResponse([line]), "gpt-5.5"))


# =============================================================================
# Anthropic output
# =============================================================================

class TestCodexAnthropic:
    """Tests for Anthropic non-streaming and streaming output."""

    @pytest.mark.asyncio
    async def test_collect_anthropic_text(self):
        """
        What it does: Collects text into an Anthropic message with text block.
        Purpose: Non-streaming Anthropic output.
        """
        lines = [_text_delta("Hi"), _completed({"input_tokens": 2, "output_tokens": 1})]
        result = await collect_codex_anthropic_response(FakeCodexResponse(lines), "gpt-5.5")
        assert result["type"] == "message"
        assert result["content"][0] == {"type": "text", "text": "Hi"}
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 2

    @pytest.mark.asyncio
    async def test_collect_anthropic_tool_use(self):
        """
        What it does: Tool calls become tool_use blocks with parsed input; stop_reason=tool_use.
        Purpose: Anthropic tool-call shape.
        """
        lines = [_function_call("calc", '{"a":1}'), _completed()]
        result = await collect_codex_anthropic_response(FakeCodexResponse(lines), "gpt-5.5")
        tool_block = [b for b in result["content"] if b["type"] == "tool_use"][0]
        assert tool_block["name"] == "calc"
        assert tool_block["input"] == {"a": 1}
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_stream_anthropic_events_sequence(self):
        """
        What it does: Streaming emits message_start ... message_stop with text deltas.
        Purpose: Anthropic streaming contract.
        """
        lines = [_text_delta("Hi"), _completed({"input_tokens": 1, "output_tokens": 1})]
        resp = FakeCodexResponse(lines)
        chunks = await _collect_sse(stream_codex_to_anthropic(resp, "gpt-5.5"))
        joined = "".join(chunks)
        assert "message_start" in joined
        assert "content_block_start" in joined
        assert "text_delta" in joined
        assert "message_delta" in joined
        assert "message_stop" in joined
        assert resp.closed is True

    @pytest.mark.asyncio
    async def test_stream_anthropic_thinking_block(self):
        """
        What it does: Reasoning deltas emit a thinking content block.
        Purpose: Anthropic thinking streaming.
        """
        lines = [_reasoning_delta("hmm"), _text_delta("ok"), _completed()]
        chunks = await _collect_sse(stream_codex_to_anthropic(FakeCodexResponse(lines), "gpt-5.5"))
        joined = "".join(chunks)
        assert "thinking_delta" in joined
        assert "text_delta" in joined
