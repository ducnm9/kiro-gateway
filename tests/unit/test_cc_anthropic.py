# -*- coding: utf-8 -*-

"""
Unit tests for Command Code Anthropic surface parity.

Tests:
- build_cc_payload_anthropic() envelope construction (system/messages/tools)
- collect_cc_anthropic_response() Anthropic message assembly
- stream_cc_to_anthropic() SSE ordering
- _map_stop_reason() and _map_anthropic_usage() helpers
- /v1/messages routing to the Command Code upstream
"""

import json
from typing import Any, Dict

import pytest
from unittest.mock import AsyncMock, Mock, patch

from kiro.config import COMMAND_CODE_MAX_TOKENS_CAP
from kiro.converters_cc import build_cc_payload_anthropic
from kiro.models_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
    Base64ImageSource,
    ImageContentBlock,
    SystemContentBlock,
    TextContentBlock,
    ThinkingContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
    URLImageSource,
)
from kiro.streaming_cc import (
    _map_anthropic_usage,
    _map_stop_reason,
    collect_cc_anthropic_response,
    stream_cc_to_anthropic,
)


# =============================================================================
# Helpers
# =============================================================================

class FakeCCResponse:
    """Duck-typed SSE response stub with an async line iterator."""

    def __init__(self, lines, status_code=200, body=b""):
        self.status_code = status_code
        self._lines = lines
        self._body = body
        self.aclose_called = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body

    async def aclose(self):
        self.aclose_called = True


def _sse_line(payload: Dict[str, Any]) -> str:
    """Build an SSE data line from a JSON payload dict."""
    return "data: " + json.dumps(payload)


def _make_request(**overrides) -> AnthropicMessagesRequest:
    """Build a minimal AnthropicMessagesRequest with defaults."""
    kwargs: Dict[str, Any] = {
        "model": "deepseek/deepseek-v4-pro",
        "messages": [AnthropicMessage(role="user", content="Hi")],
        "max_tokens": 1024,
    }
    kwargs.update(overrides)
    return AnthropicMessagesRequest(**kwargs)


async def _collect_chunks(lines, model="deepseek/deepseek-v4-pro"):
    """Collect all SSE event strings yielded by stream_cc_to_anthropic."""
    chunks = []
    async for chunk in stream_cc_to_anthropic(FakeCCResponse(lines), model):
        chunks.append(chunk)
    return chunks


# =============================================================================
# Tests for build_cc_payload_anthropic()
# =============================================================================

class TestBuildCCPayloadAnthropic:
    """Tests for the Anthropic→Command Code envelope builder."""

    def test_system_string_maps_to_params_system(self):
        """
        What it does: Verifies a string system prompt is placed in params.system.
        Purpose: Ensure the system prompt is forwarded correctly.
        """
        payload = build_cc_payload_anthropic(_make_request(system="You are helpful."))

        assert payload["params"]["system"] == "You are helpful."

    def test_system_list_blocks_joined(self):
        """
        What it does: Verifies a list-of-text-blocks system prompt is joined.
        Purpose: Ensure system prompt caching blocks are flattened.
        """
        payload = build_cc_payload_anthropic(_make_request(
            system=[SystemContentBlock(text="Line A"), SystemContentBlock(text="Line B")],
        ))

        assert payload["params"]["system"] == "Line A\nLine B"

    def test_no_system_omits_key(self):
        """Verifies the system key is omitted when there is no system prompt."""
        payload = build_cc_payload_anthropic(_make_request())

        assert "system" not in payload["params"]

    def test_tool_use_block_flattened_to_text(self):
        """
        What it does: Verifies a tool_use block becomes an assistant text part
            matching the plain-text tool-call format.
        Purpose: Ensure the next turn can parse it back into a tool call.
        """
        msg = AnthropicMessage(
            role="assistant",
            content=[ToolUseContentBlock(id="call_1", name="bash", input={"command": "ls"})],
        )
        payload = build_cc_payload_anthropic(_make_request(messages=[msg]))

        messages = payload["params"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        text = messages[0]["content"][0]["text"]
        assert text.startswith("Assistant requested tool bash (call_1) with arguments: ")
        assert json.loads(text.split("arguments: ", 1)[1]) == {"command": "ls"}

    def test_tool_result_block_flattened_to_user_text(self):
        """
        What it does: Verifies a tool_result block becomes a user text part.
        Purpose: Ensure tool results are forwarded as user content.
        """
        msg = AnthropicMessage(
            role="user",
            content=[ToolResultContentBlock(tool_use_id="call_1", content="file listing")],
        )
        payload = build_cc_payload_anthropic(_make_request(messages=[msg]))

        messages = payload["params"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "Tool result (call_1):\nfile listing"

    def test_image_base64_becomes_image_part(self):
        """
        What it does: Verifies a base64 image block becomes a CC image part.
        Purpose: Ensure image content is forwarded in the CC image shape.
        """
        msg = AnthropicMessage(
            role="user",
            content=[ImageContentBlock(
                source=Base64ImageSource(media_type="image/png", data="aGVsbG8=")
            )],
        )
        payload = build_cc_payload_anthropic(_make_request(messages=[msg]))

        messages = payload["params"]["messages"]
        assert messages[0]["content"] == [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        }]

    def test_image_url_skipped(self):
        """
        What it does: Verifies a URL image is skipped.
        Purpose: Ensure unsupported URL images don't break the envelope.
        """
        msg = AnthropicMessage(
            role="user",
            content=[ImageContentBlock(source=URLImageSource(url="https://example.com/i.png"))],
        )
        payload = build_cc_payload_anthropic(_make_request(messages=[msg]))

        # No text/image parts remain for that message, so it is dropped.
        assert payload["params"]["messages"] == []

    def test_max_tokens_clamped_to_cap(self):
        """Verifies max_tokens is clamped to COMMAND_CODE_MAX_TOKENS_CAP."""
        payload = build_cc_payload_anthropic(
            _make_request(max_tokens=COMMAND_CODE_MAX_TOKENS_CAP + 5000)
        )

        assert payload["params"]["max_tokens"] == COMMAND_CODE_MAX_TOKENS_CAP

    def test_stream_true_and_permission_mode(self):
        """Verifies stream is always True and permissionMode is standard."""
        payload = build_cc_payload_anthropic(_make_request())

        assert payload["params"]["stream"] is True
        assert payload["permissionMode"] == "standard"

    def test_user_tool_forwarded(self):
        """
        What it does: Verifies a user-defined tool is forwarded with input_schema.
        Purpose: Ensure Anthropic tools are mapped to the CC flat function form.
        """
        payload = build_cc_payload_anthropic(_make_request(
            tools=[AnthropicTool(
                name="bash",
                description="Run a command",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            )],
        ))

        assert payload["params"]["tools"] == [{
            "type": "function",
            "name": "bash",
            "description": "Run a command",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }]

    def test_server_side_tool_skipped(self):
        """
        What it does: Verifies a server-side tool (with type) is skipped.
        Purpose: Ensure server-side tools aren't forwarded to CC.
        """
        payload = build_cc_payload_anthropic(_make_request(
            tools=[AnthropicTool(type="web_search_20250305", name="web_search")],
        ))

        assert payload["params"]["tools"] == []


# =============================================================================
# Tests for collect_cc_anthropic_response()
# =============================================================================

class TestCollectCCAnthropicResponse:
    """Tests for assembling an Anthropic message from a CC stream."""

    @pytest.mark.asyncio
    async def test_content_thinking_and_tool_use_blocks(self):
        """
        What it does: Verifies content/thinking/tool_use map to Anthropic blocks.
        Purpose: Ensure the full non-streaming message is reconstructed.
        """
        response = FakeCCResponse([
            _sse_line({"type": "reasoning-delta", "text": "Let me think"}),
            _sse_line({"type": "text-delta", "text": "Hello"}),
            _sse_line({"type": "tool-call", "toolCallId": "call_x", "toolName": "read", "input": {"file": "t.go"}}),
            _sse_line({"type": "finish", "finishReason": "stop"}),
        ])

        result = await collect_cc_anthropic_response(response, "deepseek/deepseek-v4-pro")

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "deepseek/deepseek-v4-pro"
        assert result["stop_reason"] == "tool_use"

        block_types = [b["type"] for b in result["content"]]
        assert block_types == ["thinking", "text", "tool_use"]
        assert result["content"][0]["thinking"] == "Let me think"
        assert result["content"][0]["signature"].startswith("sig_")
        assert result["content"][1]["text"] == "Hello"
        assert result["content"][2]["id"] == "call_x"
        assert result["content"][2]["name"] == "read"
        assert result["content"][2]["input"] == {"file": "t.go"}

    @pytest.mark.asyncio
    async def test_usage_mapping_includes_cache_fields(self):
        """
        What it does: Verifies usage maps cacheRead/noCache token fields.
        Purpose: Ensure Anthropic cache usage fields are populated.
        """
        response = FakeCCResponse([
            _sse_line({
                "type": "finish",
                "finishReason": "stop",
                "totalUsage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15,
                    "inputTokenDetails": {"cacheReadTokens": 3, "noCacheTokens": 4},
                },
            }),
        ])

        result = await collect_cc_anthropic_response(response, "deepseek/deepseek-v4-pro")

        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
        }


class TestCCAnthropicMappers:
    """Tests for the stop reason and usage mapping helpers."""

    def test_map_stop_reason(self):
        """Verifies stop reason mapping across tool-call and truncation cases."""
        assert _map_stop_reason("stop", False) == "end_turn"
        assert _map_stop_reason("length", False) == "max_tokens"
        assert _map_stop_reason("max_tokens", False) == "max_tokens"
        assert _map_stop_reason(None, False) == "end_turn"
        assert _map_stop_reason("stop", True) == "tool_use"

    def test_map_anthropic_usage_none(self):
        """Verifies empty usage returns zero tokens."""
        assert _map_anthropic_usage(None) == {"input_tokens": 0, "output_tokens": 0}


# =============================================================================
# Tests for stream_cc_to_anthropic()
# =============================================================================

class TestStreamCCToAnthropic:
    """Tests for Anthropic SSE streaming from a CC stream."""

    @pytest.mark.asyncio
    async def test_sse_ordering(self):
        """
        What it does: Verifies the SSE event sequence is ordered correctly.
        Purpose: Ensure message_start first, then content blocks, then terminal events.
        """
        chunks = await _collect_chunks([
            _sse_line({"type": "text-delta", "text": "Hello"}),
            _sse_line({"type": "finish", "finishReason": "stop"}),
        ])

        assert "event: message_start" in chunks[0]
        assert "event: content_block_start" in chunks[1]
        assert "event: content_block_delta" in chunks[2]
        assert "event: content_block_stop" in chunks[3]
        assert "event: message_delta" in chunks[-2]
        assert "event: message_stop" in chunks[-1]

    @pytest.mark.asyncio
    async def test_message_delta_stop_reason(self):
        """
        What it does: Verifies message_delta carries the correct stop_reason.
        Purpose: Ensure the terminal stop_reason maps from the CC finish reason.
        """
        chunks = await _collect_chunks([
            _sse_line({"type": "finish", "finishReason": "max_tokens"}),
        ])

        message_delta = chunks[-2]
        assert '"stop_reason": "max_tokens"' in message_delta
        assert '"output_tokens": 0' in message_delta

    @pytest.mark.asyncio
    async def test_response_closed(self):
        """
        What it does: Verifies the upstream response is closed on exit.
        Purpose: Ensure no connection leak.
        """
        response = FakeCCResponse([
            _sse_line({"type": "text-delta", "text": "x"}),
        ])
        async for _ in stream_cc_to_anthropic(response, "deepseek/deepseek-v4-pro"):
            pass

        assert response.aclose_called is True


# =============================================================================
# Tests for /v1/messages routing to Command Code
# =============================================================================

class TestMessagesCommandCode:
    """Tests for routing /v1/messages to the Command Code upstream."""

    @staticmethod
    def _make_cc_backend():
        """Return a fake Command Code backend."""
        backend = Mock()
        backend.base_url = "https://api.commandcode.ai"
        backend.build_headers.return_value = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }
        return backend

    @staticmethod
    def _make_fake_response(lines, status_code=200):
        """Return a duck-typed async response yielding the given SSE lines."""
        async def aiter_lines():
            for line in lines:
                yield line

        response = AsyncMock()
        response.status_code = status_code
        response.aiter_lines = aiter_lines
        response.aread = AsyncMock(return_value=b"")
        return response

    def _post(self, test_client, valid_proxy_api_key, stream=False):
        """POST /v1/messages with the given stream flag."""
        return test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "deepseek/deepseek-v4-pro",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": stream,
            },
        )

    def test_messages_non_streaming_returns_anthropic_message(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """
        What it does: Verifies a non-streaming CC-model request returns an
            Anthropic message dict.
        Purpose: Ensure the M5 non-streaming Anthropic path works end to end.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        test_client.app.state.command_code_backend = self._make_cc_backend()

        fake_response = self._make_fake_response([
            _sse_line({"type": "text-delta", "text": "Hello"}),
            _sse_line({"type": "finish", "finishReason": "stop"}),
        ])
        mock_client = Mock()
        mock_client.build_request = Mock(return_value=Mock())
        mock_client.send = AsyncMock(return_value=fake_response)
        test_client.app.state.http_client = mock_client

        response = self._post(test_client, valid_proxy_api_key, stream=False)

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0]["text"] == "Hello"

    def test_messages_streaming_returns_event_stream(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """
        What it does: Verifies a streaming CC-model request returns an SSE stream.
        Purpose: Ensure the M5 streaming Anthropic path works end to end.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        test_client.app.state.command_code_backend = self._make_cc_backend()

        fake_response = self._make_fake_response([
            _sse_line({"type": "text-delta", "text": "Hello"}),
            _sse_line({"type": "finish", "finishReason": "stop"}),
        ])
        mock_stream_client = AsyncMock()
        mock_stream_client.build_request = Mock(return_value=Mock())
        mock_stream_client.send = AsyncMock(return_value=fake_response)
        mock_stream_client.__aenter__ = AsyncMock(return_value=mock_stream_client)
        mock_stream_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiro.routes_anthropic.httpx.AsyncClient", return_value=mock_stream_client):
            response = self._post(test_client, valid_proxy_api_key, stream=True)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = response.text
        assert "event: message_start" in body
        assert "event: message_stop" in body

    def test_messages_without_backend_returns_503(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """
        What it does: Verifies a CC-model request returns 503 when backend absent.
        Purpose: Ensure a clear error when Command Code is not configured.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        test_client.app.state.command_code_backend = None

        response = self._post(test_client, valid_proxy_api_key, stream=False)

        assert response.status_code == 503
