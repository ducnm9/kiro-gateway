# -*- coding: utf-8 -*-

"""
Unit tests for Command Code request conversion and stream parsing.

Tests:
- build_cc_payload() envelope construction (system/messages/max_tokens)
- parse_cc_stream() SSE event parsing
- collect_cc_response() OpenAI chat.completion assembly
- extract_cc_error_message() error body extraction
"""

import datetime
import json
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from kiro.config import COMMAND_CODE_MAX_TOKENS, COMMAND_CODE_MAX_TOKENS_CAP
from kiro.converters_cc import build_cc_payload
from kiro.models_openai import ChatCompletionRequest, ChatMessage, Tool, ToolFunction
from kiro.streaming_cc import (
    collect_cc_response,
    parse_cc_stream,
    stream_cc_to_openai,
    _map_finish_reason,
    _map_usage,
)
from kiro.upstream_cc import extract_cc_error_message, raise_cc_http_error, _rate_limit_hint
from kiro.upstream_base import resolve_upstream


# =============================================================================
# Helpers
# =============================================================================

class FakeCCResponse:
    """Duck-typed SSE response stub with an async line iterator."""

    def __init__(self, lines, status_code=200, body=b""):
        self.status_code = status_code
        self._lines = lines
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body

    async def aclose(self):
        pass


async def _collect_events(lines):
    """Collect all KiroEvent objects parsed from the given SSE lines."""
    events = []
    async for event in parse_cc_stream(FakeCCResponse(lines)):
        events.append(event)
    return events


def _sse_line(payload: Dict[str, Any]) -> str:
    """Build an SSE data line from a JSON payload dict."""
    return "data: " + json.dumps(payload)


# =============================================================================
# Tests for build_cc_payload()
# =============================================================================

class TestBuildCCPayload:
    """Tests for the Command Code request envelope builder."""

    def test_system_and_developer_join_into_params_system(self):
        """
        What it does: Verifies system/developer messages are concatenated (newline-joined)
            into params.system and excluded from params.messages.
        Purpose: Ensure the CC envelope extracts system context correctly.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="developer", content="Be concise."),
                ChatMessage(role="user", content="Hello"),
            ],
        )

        payload = build_cc_payload(request_data)

        params = payload["params"]
        assert params["system"] == "You are helpful.\nBe concise."
        roles = [m["role"] for m in params["messages"]]
        assert roles == ["user"]

    def test_tool_message_becomes_user_text_part(self):
        """
        What it does: Verifies a tool-role message is converted to a user text part.
        Purpose: Ensure tool results are flattened into user content for CC.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[
                ChatMessage(role="tool", content="result text"),
            ],
        )

        payload = build_cc_payload(request_data)

        params = payload["params"]
        assert params["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "result text"}]}
        ]

    def test_user_content_is_list_of_text_parts(self):
        """
        What it does: Verifies user/assistant content is always a list of text parts.
        Purpose: Ensure the CC content-part wire format is produced.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[
                ChatMessage(role="user", content="Question"),
                ChatMessage(role="assistant", content="Answer"),
            ],
        )

        payload = build_cc_payload(request_data)

        params = payload["params"]
        assert params["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        ]

    def test_model_passed_verbatim(self):
        """
        What it does: Verifies params.model matches the input model verbatim.
        Purpose: Ensure provider-qualified model names (with slash) are preserved.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["model"] == "deepseek/deepseek-v4-pro"

    def test_stream_true_and_tools_empty(self):
        """
        What it does: Verifies params.stream is True and params.tools is empty.
        Purpose: Ensure CC is always streaming and tools are deferred to a later milestone.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        params = payload["params"]
        assert params["stream"] is True
        assert params["tools"] == []

    def test_config_block_and_permission_mode(self):
        """
        What it does: Verifies the fixed config block and permissionMode are present.
        Purpose: Ensure the boilerplate fields required by CC are emitted.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        assert payload["permissionMode"] == "standard"
        config = payload["config"]
        assert config["workingDir"] == "/"
        assert config["date"] == datetime.date.today().isoformat()
        assert config["isGitRepo"] is False
        assert "structure" in config
        assert "recentCommits" in config

    def test_max_tokens_default_when_unset(self):
        """
        What it does: Verifies max_tokens falls back to COMMAND_CODE_MAX_TOKENS when unset.
        Purpose: Ensure a sane default budget is sent.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["max_tokens"] == COMMAND_CODE_MAX_TOKENS

    def test_max_tokens_clamped_to_cap(self):
        """
        What it does: Verifies max_tokens above the cap is clamped.
        Purpose: Ensure the CC hard upper bound is respected.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=COMMAND_CODE_MAX_TOKENS_CAP + 5000,
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["max_tokens"] == COMMAND_CODE_MAX_TOKENS_CAP

    def test_max_tokens_passed_through_when_valid(self):
        """
        What it does: Verifies an explicit valid max_tokens is passed through.
        Purpose: Ensure client budgets are honored within the cap.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=12345,
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["max_tokens"] == 12345

    def test_no_system_key_when_no_system_messages(self):
        """
        What it does: Verifies the system key is omitted when no system/developer messages exist.
        Purpose: Ensure the system field is only present when there is system content.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        assert "system" not in payload["params"]


# =============================================================================
# Tests for parse_cc_stream()
# =============================================================================

class TestParseCCStream:
    """Tests for the Command Code SSE parser."""

    @pytest.mark.asyncio
    async def test_text_delta_yields_content(self):
        """
        What it does: Verifies a text-delta event yields a content KiroEvent.
        Purpose: Ensure content deltas are extracted from the CC stream.
        """
        events = await _collect_events([
            'data: {"type":"text-delta","text":"Hello"}',
        ])

        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_reasoning_delta_yields_thinking(self):
        """
        What it does: Verifies a reasoning-delta event yields a thinking KiroEvent.
        Purpose: Ensure reasoning content is captured.
        """
        events = await _collect_events([
            'data: {"type":"reasoning-delta","text":"thinking..."}',
        ])

        assert len(events) == 1
        assert events[0].type == "thinking"
        assert events[0].thinking_content == "thinking..."

    @pytest.mark.asyncio
    async def test_done_sentinel_stops(self):
        """
        What it does: Verifies the [DONE] sentinel stops parsing.
        Purpose: Ensure parsing halts at the terminator.
        """
        events = await _collect_events([
            'data: {"type":"text-delta","text":"a"}',
            "data: [DONE]",
            'data: {"type":"text-delta","text":"b"}',
        ])

        contents = [e.content for e in events]
        assert contents == ["a"]

    @pytest.mark.asyncio
    async def test_finish_yields_usage_and_finish_reason(self):
        """
        What it does: Verifies a finish event yields usage and finish_reason.
        Purpose: Ensure token usage and finish reason are captured.
        """
        events = await _collect_events([
            'data: {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}',
        ])

        assert len(events) == 1
        assert events[0].type == "usage"
        assert events[0].finish_reason == "stop"
        assert events[0].usage == {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}

    @pytest.mark.asyncio
    async def test_unknown_event_ignored(self):
        """
        What it does: Verifies unknown event types are ignored without error.
        Purpose: Ensure forward-compatibility with future event types.
        """
        events = await _collect_events([
            'data: {"type":"finish-step","name":"x"}',
            'data: {"type":"text-end"}',
            'data: {"type":"some-future-event"}',
            'data: {"type":"text-delta","text":"kept"}',
        ])

        assert len(events) == 1
        assert events[0].content == "kept"

    @pytest.mark.asyncio
    async def test_error_event_raises_http_exception(self):
        """
        What it does: Verifies an error event raises HTTPException 502.
        Purpose: Ensure upstream stream errors surface as gateway errors.
        """
        with pytest.raises(HTTPException) as exc_info:
            await _collect_events([
                'data: {"type":"error","error":"boom"}',
            ])

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_non_data_lines_ignored(self):
        """
        What it does: Verifies non-data lines (comments, blanks) are skipped.
        Purpose: Ensure SSE framing noise does not break parsing.
        """
        events = await _collect_events([
            ": keep-alive",
            "",
            'data: {"type":"text-delta","text":"x"}',
        ])

        assert len(events) == 1
        assert events[0].content == "x"

    @pytest.mark.asyncio
    async def test_bare_ndjson_lines_parsed(self):
        """
        What it does: Verifies bare JSON lines (no `data:` prefix) are parsed.
        Purpose: The live Command Code stream is NDJSON, not data:-prefixed SSE.
        """
        events = await _collect_events([
            '{"type":"text-delta","id":"txt-0","text":"Hello"}',
            '{"type":"reasoning-delta","id":"reasoning-0","text":"think"}',
            '{"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}',
        ])

        assert [e.type for e in events] == ["content", "thinking", "usage"]
        assert events[0].content == "Hello"
        assert events[1].thinking_content == "think"
        assert events[2].finish_reason == "stop"


# =============================================================================
# Tests for collect_cc_response()
# =============================================================================

class TestCollectCCResponse:
    """Tests for assembling an OpenAI chat.completion from a CC stream."""

    @pytest.mark.asyncio
    async def test_collects_content_reasoning_and_usage(self):
        """
        What it does: Verifies content, reasoning, finish_reason, and usage are assembled.
        Purpose: Ensure the full non-streaming response is reconstructed.
        """
        response = FakeCCResponse([
            'data: {"type":"reasoning-delta","text":"Let me think"}',
            'data: {"type":"text-delta","text":"Hello"}',
            'data: {"type":"text-delta","text":" world"}',
            'data: {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}',
        ])

        result = await collect_cc_response(response, "deepseek/deepseek-v4-pro")

        assert result["object"] == "chat.completion"
        assert result["model"] == "deepseek/deepseek-v4-pro"
        message = result["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert message["content"] == "Hello world"
        assert message["reasoning_content"] == "Let me think"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.asyncio
    async def test_max_tokens_finish_reason_maps_to_length(self):
        """
        What it does: Verifies a max_tokens finish reason maps to length.
        Purpose: Ensure CC truncation reason maps to the OpenAI length reason.
        """
        response = FakeCCResponse([
            'data: {"type":"text-delta","text":"x"}',
            'data: {"type":"finish","finishReason":"max_tokens","totalUsage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}',
        ])

        result = await collect_cc_response(response, "deepseek/deepseek-v4-pro")

        assert result["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_usage_total_fallback_to_sum(self):
        """
        What it does: Verifies totalTokens falls back to input+output when zero.
        Purpose: Ensure total tokens are derived when the upstream omits them.
        """
        response = FakeCCResponse([
            'data: {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":7,"outputTokens":3,"totalTokens":0}}',
        ])

        result = await collect_cc_response(response, "deepseek/deepseek-v4-pro")

        assert result["usage"]["total_tokens"] == 10

    @pytest.mark.asyncio
    async def test_no_reasoning_content_when_absent(self):
        """
        What it does: Verifies reasoning_content is omitted when no thinking events arrive.
        Purpose: Ensure the reasoning field is only added when thinking exists.
        """
        response = FakeCCResponse([
            'data: {"type":"text-delta","text":"plain"}',
            'data: {"type":"finish","finishReason":"stop"}',
        ])

        result = await collect_cc_response(response, "deepseek/deepseek-v4-pro")

        assert "reasoning_content" not in result["choices"][0]["message"]


# =============================================================================
# Tests for stream_cc_to_openai()
# =============================================================================

async def _collect_stream_chunks(lines, model="deepseek/deepseek-v4-pro"):
    """Collect all SSE chunk strings yielded by stream_cc_to_openai."""
    chunks = []
    async for chunk in stream_cc_to_openai(FakeCCResponse(lines), model):
        chunks.append(chunk)
    return chunks


def _parse_chunks(chunks):
    """Parse SSE chunk strings into a list of dicts and the "[DONE]" sentinel."""
    parsed = []
    for chunk in chunks:
        assert chunk.startswith("data: ")
        payload = chunk[len("data: "):].strip()
        if payload == "[DONE]":
            parsed.append("[DONE]")
        else:
            parsed.append(json.loads(payload))
    return parsed


class TestStreamCCToOpenAI:
    """Tests for converting a CC stream to OpenAI chunk SSE."""

    @pytest.mark.asyncio
    async def test_text_delta_yields_content_chunk_with_role(self):
        """
        What it does: Verifies text-delta produces a content chunk with role on the
            first chunk only.
        Purpose: Ensure content deltas and the leading assistant role are emitted.
        """
        chunks = await _collect_stream_chunks([
            'data: {"type":"text-delta","text":"Hello"}',
            'data: {"type":"text-delta","text":" world"}',
        ])

        parsed = _parse_chunks(chunks)
        first = parsed[0]
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"] == {"role": "assistant", "content": "Hello"}
        assert first["choices"][0]["finish_reason"] is None

        second = parsed[1]
        assert second["choices"][0]["delta"] == {"content": " world"}
        assert "role" not in second["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_reasoning_delta_yields_reasoning_content(self):
        """
        What it does: Verifies reasoning-delta maps to reasoning_content in the delta.
        Purpose: Ensure thinking content is surfaced as native reasoning_content.
        """
        chunks = await _collect_stream_chunks([
            'data: {"type":"reasoning-delta","text":"thinking..."}',
        ])

        parsed = _parse_chunks(chunks)
        assert parsed[0]["choices"][0]["delta"] == {
            "role": "assistant",
            "reasoning_content": "thinking...",
        }

    @pytest.mark.asyncio
    async def test_finish_chunk_has_finish_reason_and_usage(self):
        """
        What it does: Verifies the finish event produces a final chunk with finish
            reason and mapped usage.
        Purpose: Ensure the terminal chunk carries completion metadata.
        """
        chunks = await _collect_stream_chunks([
            'data: {"type":"text-delta","text":"x"}',
            'data: {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}',
        ])

        parsed = _parse_chunks(chunks)
        final = parsed[-2]  # second-to-last is the finish chunk; last is [DONE]
        assert final["choices"][0]["delta"] == {}
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self):
        """
        What it does: Verifies the stream terminates with the [DONE] sentinel.
        Purpose: Ensure clients can detect stream completion.
        """
        chunks = await _collect_stream_chunks([
            'data: {"type":"text-delta","text":"x"}',
        ])

        parsed = _parse_chunks(chunks)
        assert parsed[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_empty_stream_yields_final_chunk_and_done(self):
        """
        What it does: Verifies an empty stream still yields a final chunk with
            stop and zero usage, then [DONE].
        Purpose: Ensure a valid completion is produced even with no events.
        """
        chunks = await _collect_stream_chunks([])

        parsed = _parse_chunks(chunks)
        assert len(parsed) == 2
        final = parsed[0]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        assert parsed[1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_missing_done_sentinel_tolerated(self):
        """
        What it does: Verifies a missing [DONE] sentinel from the upstream is tolerated.
        Purpose: Ensure the stream still emits the gateway-side [DONE] terminator.
        """
        chunks = await _collect_stream_chunks([
            'data: {"type":"text-delta","text":"x"}',
            'data: {"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}',
        ])

        parsed = _parse_chunks(chunks)
        assert parsed[-1] == "[DONE]"


# =============================================================================
# Tests for finish reason and usage mappers
# =============================================================================

class TestCCMappers:
    """Tests for finish reason and usage mapping helpers."""

    def test_map_finish_reason_variants(self):
        """
        What it does: Verifies finish reason mapping across all known variants.
        Purpose: Ensure each upstream reason maps to a correct OpenAI reason.
        """
        assert _map_finish_reason("stop") == "stop"
        assert _map_finish_reason("tool-calls") == "tool_calls"
        assert _map_finish_reason("tool_calls") == "tool_calls"
        assert _map_finish_reason("max_tokens") == "length"
        assert _map_finish_reason("max_output_tokens") == "length"
        assert _map_finish_reason("length") == "length"
        assert _map_finish_reason(None) == "stop"
        assert _map_finish_reason("custom") == "custom"

    def test_map_usage_none(self):
        """
        What it does: Verifies usage mapping returns zeros for None.
        Purpose: Ensure a missing usage block yields zero counts.
        """
        assert _map_usage(None) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


# =============================================================================
# Tests for extract_cc_error_message()
# =============================================================================

class TestExtractCCErrorMessage:
    """Tests for Command Code error body extraction."""

    def test_nested_error_message(self):
        """
        What it does: Verifies a nested error.message is extracted.
        Purpose: Ensure the common CC error envelope is parsed.
        """
        assert extract_cc_error_message('{"error":{"message":"bad key"}}') == "bad key"

    def test_top_level_message(self):
        """
        What it does: Verifies a top-level message field is extracted.
        Purpose: Ensure alternate error shapes are handled.
        """
        assert extract_cc_error_message('{"message":"nope"}') == "nope"

    def test_unparseable_body_truncated(self):
        """
        What it does: Verifies an unparseable body is returned truncated.
        Purpose: Ensure non-JSON error bodies still surface something useful.
        """
        assert extract_cc_error_message("plain text error") == "plain text error"


# =============================================================================
# Tests for raise_cc_http_error() and _rate_limit_hint()
# =============================================================================

class _FakeErrorResponse:
    """Duck-typed non-200 response for raise_cc_http_error tests."""

    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    async def aread(self):
        return self._body

    async def aclose(self):
        pass


class TestRaiseCCHttpError:
    """Tests for the Command Code error-to-HTTPException mapping."""

    @pytest.mark.asyncio
    async def test_429_with_retry_after_header(self):
        """
        What it does: Verifies a 429 with Retry-After header includes the seconds.
        Purpose: Ensure clients get an actionable retry hint.
        """
        response = _FakeErrorResponse(
            status_code=429,
            body=b'{"message":"too fast"}',
            headers={"Retry-After": "12"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await raise_cc_http_error(response)

        assert exc_info.value.status_code == 429
        assert "retry after 12s" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_429_with_rate_limit_reset_body(self):
        """
        What it does: Verifies a 429 with rateLimit.reset in the body includes the
            reset timestamp.
        Purpose: Ensure the reset time is surfaced when the header is absent.
        """
        response = _FakeErrorResponse(
            status_code=429,
            body=b'{"rateLimit":{"reset": 1700000000}}',
        )

        with pytest.raises(HTTPException) as exc_info:
            await raise_cc_http_error(response)

        assert exc_info.value.status_code == 429
        assert "resets at" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_429_with_no_reset_info(self):
        """
        What it does: Verifies a 429 with no reset info still produces an actionable message.
        Purpose: Ensure the rate-limit branch degrades gracefully.
        """
        response = _FakeErrorResponse(status_code=429, body=b"{}")

        with pytest.raises(HTTPException) as exc_info:
            await raise_cc_http_error(response)

        assert exc_info.value.status_code == 429
        assert "rate limit exceeded" in exc_info.value.detail
        assert "retry later" in exc_info.value.detail

    def test_rate_limit_hint_retry_after(self):
        """Verifies Retry-After header produces a seconds hint."""
        response = _FakeErrorResponse(429, headers={"Retry-After": "30"})
        assert _rate_limit_hint(response, "") == ", retry after 30s"

    def test_rate_limit_hint_reset_unix(self):
        """Verifies rateLimit.reset produces a resets-at hint."""
        response = _FakeErrorResponse(429)
        hint = _rate_limit_hint(response, '{"rateLimit":{"reset": 1700000000}}')
        assert hint.startswith(", resets at ")

    def test_rate_limit_hint_no_info(self):
        """Verifies no reset info yields an empty hint."""
        response = _FakeErrorResponse(429)
        assert _rate_limit_hint(response, "{}") == ""


# =============================================================================
# Tests for routing (Kiro model must not route to CC)
# =============================================================================

class TestCCRouting:
    """Tests confirming Kiro models are not routed to Command Code."""

    def test_bare_model_resolves_to_kiro(self):
        """
        What it does: Verifies a bare Kiro model resolves to the Kiro upstream.
        Purpose: Ensure only provider-qualified models route to Command Code.
        """
        assert resolve_upstream("claude-haiku-4.5") == "kiro"


# =============================================================================
# Tests for tool forwarding in build_cc_payload()
# =============================================================================

class TestBuildCCPayloadTools:
    """Tests for forwarding OpenAI tools to the Command Code envelope."""

    def test_standard_openai_tool_forwarded(self):
        """
        What it does: Verifies a standard OpenAI tool is forwarded with the
            input_schema key (not parameters).
        Purpose: Ensure Command Code receives tools in its flat function form.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[Tool(
                type="function",
                function=ToolFunction(
                    name="bash",
                    description="Run a command",
                    parameters={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                ),
            )],
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["tools"] == [{
            "type": "function",
            "name": "bash",
            "description": "Run a command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        }]

    def test_flat_format_tool_forwarded(self):
        """
        What it does: Verifies a Cursor-style flat tool is forwarded.
        Purpose: Ensure flat-format tools are normalized the same as standard ones.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[Tool(
                name="bash",
                description="Run a command",
                input_schema={"type": "object", "properties": {}},
            )],
        )

        payload = build_cc_payload(request_data)

        tools = payload["params"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "bash"
        assert tools[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_missing_schema_uses_default(self):
        """
        What it does: Verifies a tool with no schema gets the default empty object schema.
        Purpose: Ensure Command Code never receives a None schema.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[Tool(
                type="function",
                function=ToolFunction(name="bash", description="Run a command"),
            )],
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["tools"][0]["input_schema"] == {
            "type": "object",
            "properties": {},
        }

    def test_no_tools_returns_empty_list(self):
        """
        What it does: Verifies a request without tools yields an empty tools list.
        Purpose: Ensure the tools key is always present and empty when no tools.
        """
        request_data = ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        payload = build_cc_payload(request_data)

        assert payload["params"]["tools"] == []


# =============================================================================
# Tests for tool-call reconstruction in parse_cc_stream()
# =============================================================================

class TestParseCCStreamToolCalls:
    """Tests for reconstructing tool calls from the CC stream."""

    @pytest.mark.asyncio
    async def test_structured_tool_call_event(self):
        """
        What it does: Verifies a structured tool-call event yields a tool_use event.
        Purpose: Ensure the object-input tool-call format is reconstructed.
        """
        events = await _collect_events([
            _sse_line({
                "type": "tool-call",
                "toolCallId": "call_x",
                "toolName": "read",
                "input": {"file": "test.go"},
            }),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["id"] == "call_x"
        assert tc["function"]["name"] == "read"
        assert json.loads(tc["function"]["arguments"]) == {"file": "test.go"}

    @pytest.mark.asyncio
    async def test_tool_input_start_delta_end(self):
        """
        What it does: Verifies buffered tool-input fragments assemble into one tool call.
        Purpose: Ensure streamed tool arguments are reconstructed correctly.
        """
        events = await _collect_events([
            _sse_line({"type": "tool-input-start", "id": "call_x", "toolName": "write"}),
            _sse_line({"type": "tool-input-delta", "id": "call_x", "delta": '{"path":'}),
            _sse_line({"type": "tool-input-delta", "id": "call_x", "delta": '"/tmp/x"}'}),
            _sse_line({"type": "tool-input-end", "id": "call_x", "toolName": "write"}),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["id"] == "call_x"
        assert tc["function"]["name"] == "write"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_tool_error_with_preencoded_json_string(self):
        """
        What it does: Verifies a tool-error event with a pre-encoded JSON string
            input is reconstructed.
        Purpose: Ensure string-encoded tool input is normalized to a JSON string.
        """
        events = await _collect_events([
            _sse_line({
                "type": "tool-error",
                "id": "call_x",
                "toolName": "write",
                "input": '{"command":"ls"}',
            }),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["id"] == "call_x"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_plain_text_tool_call(self):
        """
        What it does: Verifies a plain-text tool-call line yields a tool_use and no content.
        Purpose: Ensure the 'Assistant requested tool' format is parsed.
        """
        events = await _collect_events([
            _sse_line({
                "type": "text-delta",
                "text": 'Assistant requested tool bash (call_abc123) with arguments: {"command":"ls /tmp"}',
            }),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["id"] == "call_abc123"
        assert tc["function"]["name"] == "bash"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls /tmp"}

    @pytest.mark.asyncio
    async def test_plain_text_invalid_arguments_ignored(self):
        """
        What it does: Verifies an 'invalid arguments' tool line is not emitted as a tool call.
        Purpose: Ensure error variants are not reconstructed as valid tool calls.
        """
        events = await _collect_events([
            _sse_line({
                "type": "text-delta",
                "text": "Assistant requested tool bash (call_x) with invalid arguments: boom",
            }),
        ])

        assert events == []

    @pytest.mark.asyncio
    async def test_dsml_complete_envelope(self):
        """
        What it does: Verifies a complete DSML envelope yields a tool_use and no content.
        Purpose: Ensure DSML tool calls are reconstructed from text deltas.
        """
        dsml = (
            '<tool_calls><invoke name="read">'
            '<parameter name="path" string="true">/tmp/x</parameter>'
            '</invoke></tool_calls>'
        )
        events = await _collect_events([
            _sse_line({"type": "text-delta", "text": dsml}),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["function"]["name"] == "read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_dsml_fragmented_across_deltas(self):
        """
        What it does: Verifies a DSML envelope fragmented across two deltas is buffered
            and yields one tool_use.
        Purpose: Ensure fragmented DSML is reassembled before parsing.
        """
        part1 = '<tool_calls><invoke name="read">'
        part2 = (
            '<parameter name="path" string="true">/tmp/x</parameter>'
            '</invoke></tool_calls>'
        )
        events = await _collect_events([
            _sse_line({"type": "text-delta", "text": part1}),
            _sse_line({"type": "text-delta", "text": part2}),
        ])

        assert len(events) == 1
        assert events[0].type == "tool_use"
        tc = events[0].tool_use
        assert tc["function"]["name"] == "read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/x"}


# =============================================================================
# Tests for tool_calls in collect_cc_response() and stream_cc_to_openai()
# =============================================================================

class TestCCResponseToolCalls:
    """Tests for tool calls in collected and streamed OpenAI responses."""

    @pytest.mark.asyncio
    async def test_collect_cc_response_includes_tool_calls(self):
        """
        What it does: Verifies collect_cc_response includes tool_calls and overrides
            finish_reason to tool_calls.
        Purpose: Ensure non-streaming tool-call responses are complete.
        """
        response = FakeCCResponse([
            _sse_line({
                "type": "tool-call",
                "toolCallId": "call_x",
                "toolName": "read",
                "input": {"file": "test.go"},
            }),
            _sse_line({"type": "finish", "finishReason": "stop"}),
        ])

        result = await collect_cc_response(response, "deepseek/deepseek-v4-pro")

        message = result["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "read"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_cc_to_openai_emits_tool_calls_chunk(self):
        """
        What it does: Verifies stream_cc_to_openai emits an indexed tool_calls chunk
            and a final chunk with finish_reason tool_calls.
        Purpose: Ensure streaming tool-call responses match the OpenAI spec.
        """
        chunks = await _collect_stream_chunks([
            _sse_line({
                "type": "tool-call",
                "toolCallId": "call_x",
                "toolName": "read",
                "input": {"file": "test.go"},
            }),
        ])

        parsed = _parse_chunks(chunks)
        tool_calls_chunks = [
            c for c in parsed
            if isinstance(c, dict) and "tool_calls" in c["choices"][0]["delta"]
        ]
        assert len(tool_calls_chunks) == 1
        tc = tool_calls_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["id"] == "call_x"
        assert tc["function"]["name"] == "read"

        final = parsed[-2]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
