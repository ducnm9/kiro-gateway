# -*- coding: utf-8 -*-

"""Unit tests for the Codex (ChatGPT) request converters.

Covers OpenAI→Codex and Anthropic→Codex translation:
- input array shape (roles + typed content parts).
- system/developer → instructions (not a system-role item).
- store=false, stream=true always set.
- reasoning.effort normalization + include for encrypted content.
- tools flattened to Responses form; tool_choice validation/drop.
- allowlist stripping of unexpected fields.
- images: data URLs pass through, remote images become input_image parts,
  and inline_remote_images fetches remote URLs to base64 (fail-safe).
- empty input falls back to a placeholder message.
- tool_calls / tool_result flattened to text.
"""

import base64
import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kiro.converters_codex import (
    build_codex_payload,
    build_codex_payload_anthropic,
    inline_remote_images,
    _normalize_effort,
    CODEX_DEFAULT_INSTRUCTIONS,
    _RESPONSES_ALLOWLIST,
)
from kiro.models_openai import ChatCompletionRequest, ChatMessage, Tool, ToolFunction
from kiro.models_anthropic import AnthropicMessagesRequest, AnthropicMessage, AnthropicTool


def _openai_request(**kwargs):
    """Build a ChatCompletionRequest with sane defaults."""
    base = {"model": "gpt-5.5", "messages": [ChatMessage(role="user", content="hello")]}
    base.update(kwargs)
    return ChatCompletionRequest(**base)


# =============================================================================
# OpenAI → Codex
# =============================================================================

class TestBuildCodexPayloadOpenAI:
    """Tests for build_codex_payload (OpenAI → Codex)."""

    def test_basic_shape(self):
        """
        What it does: A simple user message yields a well-formed Codex payload.
        Purpose: Core envelope invariants.
        """
        payload = build_codex_payload(_openai_request())
        assert payload["model"] == "gpt-5.5"
        assert payload["store"] is False
        assert payload["stream"] is True
        assert isinstance(payload["input"], list) and len(payload["input"]) == 1
        item = payload["input"][0]
        assert item["role"] == "user"
        assert item["content"][0] == {"type": "input_text", "text": "hello"}

    def test_system_becomes_instructions(self):
        """
        What it does: system/developer messages become instructions, not input items.
        Purpose: Keep system prompt in the cacheable prefix, no system-role item.
        """
        req = _openai_request(messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="developer", content="use python"),
            ChatMessage(role="user", content="hi"),
        ])
        payload = build_codex_payload(req)
        assert "be terse" in payload["instructions"]
        assert "use python" in payload["instructions"]
        roles = [i["role"] for i in payload["input"]]
        assert "system" not in roles and "developer" not in roles
        assert roles == ["user"]

    def test_default_instructions_when_absent(self):
        """
        What it does: With no system message, default instructions are used.
        Purpose: Codex requires a non-empty instructions field.
        """
        payload = build_codex_payload(_openai_request())
        assert payload["instructions"] == CODEX_DEFAULT_INSTRUCTIONS

    def test_assistant_message_output_text(self):
        """
        What it does: Assistant messages become output_text items.
        Purpose: Correct role/content typing for prior turns.
        """
        req = _openai_request(messages=[
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
            ChatMessage(role="user", content="q2"),
        ])
        payload = build_codex_payload(req)
        asst = [i for i in payload["input"] if i["role"] == "assistant"]
        assert asst and asst[0]["content"][0] == {"type": "output_text", "text": "a"}

    def test_tool_calls_flattened_to_text(self):
        """
        What it does: Assistant tool_calls are flattened into readable text.
        Purpose: Avoid unresolved client-side tool ids under store=false.
        """
        req = _openai_request(messages=[
            ChatMessage(role="user", content="weather?"),
            ChatMessage(role="assistant", content="", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"paris"}'}}
            ]),
        ])
        payload = build_codex_payload(req)
        asst = [i for i in payload["input"] if i["role"] == "assistant"]
        assert asst
        text = asst[0]["content"][0]["text"]
        assert "get_weather" in text and "paris" in text

    def test_tool_result_message_flattened(self):
        """
        What it does: role=tool messages become user input_text with a marker.
        Purpose: Preserve tool output in the conversation for Codex.
        """
        req = _openai_request(messages=[
            ChatMessage(role="tool", content="sunny", tool_call_id="c1"),
        ])
        payload = build_codex_payload(req)
        item = payload["input"][0]
        assert item["role"] == "user"
        assert "tool result c1" in item["content"][0]["text"]
        assert "sunny" in item["content"][0]["text"]

    def test_tools_flattened(self):
        """
        What it does: OpenAI tools become Responses function tools.
        Purpose: Correct tool shape for Codex.
        """
        tool = Tool(type="function", function=ToolFunction(
            name="search", description="web search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}}))
        payload = build_codex_payload(_openai_request(tools=[tool]))
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["name"] == "search"
        assert payload["tools"][0]["parameters"]["properties"]["q"]["type"] == "string"

    def test_tool_choice_valid_kept(self):
        """
        What it does: A tool_choice referencing a known tool is kept.
        Purpose: Honor explicit tool selection.
        """
        tool = Tool(type="function", function=ToolFunction(name="search", parameters={}))
        req = _openai_request(tools=[tool], tool_choice={"type": "function", "function": {"name": "search"}})
        payload = build_codex_payload(req)
        assert payload.get("tool_choice", {}).get("function", {}).get("name") == "search"

    def test_tool_choice_unknown_dropped(self):
        """
        What it does: A tool_choice referencing an unknown tool is dropped.
        Purpose: Prevent upstream rejection.
        """
        tool = Tool(type="function", function=ToolFunction(name="search", parameters={}))
        req = _openai_request(tools=[tool], tool_choice={"type": "function", "function": {"name": "missing"}})
        payload = build_codex_payload(req)
        assert "tool_choice" not in payload

    def test_reasoning_effort_applied(self):
        """
        What it does: reasoning_effort maps into reasoning.effort + include.
        Purpose: Pass through the client's reasoning hint.
        """
        req = _openai_request(reasoning_effort="high")
        payload = build_codex_payload(req)
        assert payload["reasoning"]["effort"] == "high"
        assert payload["include"] == ["reasoning.encrypted_content"]

    def test_allowlist_strips_unexpected_fields(self):
        """
        What it does: Only allowlisted keys appear in the final payload.
        Purpose: Avoid 'routing_unsupported' from stray fields.
        """
        payload = build_codex_payload(_openai_request())
        assert set(payload.keys()) <= _RESPONSES_ALLOWLIST

    def test_image_data_url_passthrough(self):
        """
        What it does: A data: image URL becomes an input_image part unchanged.
        Purpose: Local images require no fetching.
        """
        data_url = "data:image/png;base64,AAAA"
        req = _openai_request(messages=[ChatMessage(role="user", content=[
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])])
        payload = build_codex_payload(req)
        parts = payload["input"][0]["content"]
        assert {"type": "input_image", "image_url": data_url} in parts

    def test_empty_input_falls_back_to_placeholder(self):
        """
        What it does: An empty conversation yields a placeholder user message.
        Purpose: Codex rejects an empty input array.
        """
        # Only a system message → no input items.
        req = _openai_request(messages=[ChatMessage(role="system", content="sys")])
        payload = build_codex_payload(req)
        assert len(payload["input"]) == 1
        assert payload["input"][0]["role"] == "user"


class TestNormalizeEffort:
    """Tests for reasoning-effort normalization."""

    @pytest.mark.parametrize("value,expected", [
        (None, "low"), ("", "low"), ("weird", "low"),
        ("high", "high"), ("HIGH", "high"), ("none", "none"),
        ("ultra", "xhigh"), ("max", "xhigh"), ("minimal", "minimal"),
    ])
    def test_normalize(self, value, expected):
        """Effort values normalize to Codex-accepted levels."""
        assert _normalize_effort(value) == expected

    def test_effort_none_omits_include(self):
        """
        What it does: effort=none does not set include=reasoning.encrypted_content.
        Purpose: Encrypted reasoning only requested when reasoning is active.
        """
        req = _openai_request(reasoning_effort="none")
        payload = build_codex_payload(req)
        assert payload["reasoning"]["effort"] == "none"
        assert "include" not in payload


# =============================================================================
# Anthropic → Codex
# =============================================================================

def _anthropic_request(**kwargs):
    base = {
        "model": "gpt-5.5",
        "max_tokens": 1024,
        "messages": [AnthropicMessage(role="user", content="hello")],
    }
    base.update(kwargs)
    return AnthropicMessagesRequest(**base)


class TestBuildCodexPayloadAnthropic:
    """Tests for build_codex_payload_anthropic (Anthropic → Codex)."""

    def test_basic_shape(self):
        """
        What it does: A simple Anthropic request yields a Codex payload.
        Purpose: Core envelope invariants for the Anthropic path.
        """
        payload = build_codex_payload_anthropic(_anthropic_request())
        assert payload["store"] is False and payload["stream"] is True
        assert payload["input"][0]["role"] == "user"
        assert payload["input"][0]["content"][0]["type"] == "input_text"

    def test_system_prompt_becomes_instructions(self):
        """
        What it does: The Anthropic system prompt maps to instructions.
        Purpose: Consistent system handling with the OpenAI path.
        """
        payload = build_codex_payload_anthropic(_anthropic_request(system="be brief"))
        assert payload["instructions"] == "be brief"

    def test_assistant_output_text(self):
        """
        What it does: Assistant text blocks become output_text.
        Purpose: Correct role typing.
        """
        req = _anthropic_request(messages=[
            AnthropicMessage(role="user", content="q"),
            AnthropicMessage(role="assistant", content=[{"type": "text", "text": "a"}]),
        ])
        payload = build_codex_payload_anthropic(req)
        asst = [i for i in payload["input"] if i["role"] == "assistant"]
        assert asst[0]["content"][0] == {"type": "output_text", "text": "a"}

    def test_base64_image_becomes_data_url(self):
        """
        What it does: A base64 image block becomes an input_image data URL.
        Purpose: Inline local images for Codex.
        """
        req = _anthropic_request(messages=[AnthropicMessage(role="user", content=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ])])
        payload = build_codex_payload_anthropic(req)
        part = payload["input"][0]["content"][0]
        assert part["type"] == "input_image"
        assert part["image_url"] == "data:image/png;base64,AAAA"

    def test_tool_use_and_result_flattened(self):
        """
        What it does: tool_use and tool_result blocks flatten to text.
        Purpose: Preserve tool exchange without server-side ids.
        """
        req = _anthropic_request(messages=[
            AnthropicMessage(role="assistant", content=[
                {"type": "tool_use", "id": "t1", "name": "calc", "input": {"x": 1}},
            ]),
            AnthropicMessage(role="user", content=[
                {"type": "tool_result", "tool_use_id": "t1", "content": "2"},
            ]),
        ])
        payload = build_codex_payload_anthropic(req)
        asst = [i for i in payload["input"] if i["role"] == "assistant"][0]
        assert "calc" in asst["content"][0]["text"]
        usr = [i for i in payload["input"] if i["role"] == "user"][0]
        assert "tool result t1" in usr["content"][0]["text"]

    def test_tools_flattened(self):
        """
        What it does: Anthropic tools become Responses function tools.
        Purpose: Correct tool shape.
        """
        tool = AnthropicTool(name="calc", description="math",
                             input_schema={"type": "object", "properties": {}})
        payload = build_codex_payload_anthropic(_anthropic_request(tools=[tool]))
        assert payload["tools"][0]["name"] == "calc"
        assert payload["tools"][0]["type"] == "function"

    def test_allowlist(self):
        """
        What it does: Only allowlisted keys are emitted.
        Purpose: Avoid stray fields.
        """
        payload = build_codex_payload_anthropic(_anthropic_request())
        assert set(payload.keys()) <= _RESPONSES_ALLOWLIST


# =============================================================================
# Remote image inlining
# =============================================================================

class TestInlineRemoteImages:
    """Tests for inline_remote_images."""

    @pytest.mark.asyncio
    async def test_fetches_remote_image_to_base64(self):
        """
        What it does: A remote image URL is fetched and inlined as base64.
        Purpose: Codex cannot fetch remote images.
        """
        payload = {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/a.png"},
        ]}]}

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNG"
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await inline_remote_images(payload)

        url = result["input"][0]["content"][0]["image_url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64encode(b"\x89PNG").decode() in url

    @pytest.mark.asyncio
    async def test_data_url_untouched(self):
        """
        What it does: A data: URL is not fetched or altered.
        Purpose: No network for already-inline images.
        """
        payload = {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]}]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await inline_remote_images(payload)

        assert result["input"][0]["content"][0]["image_url"] == "data:image/png;base64,AAAA"
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_failure_keeps_original_url(self):
        """
        What it does: A fetch error leaves the original URL in place (fail-safe).
        Purpose: Never corrupt the payload on network failure.
        """
        payload = {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/a.png"},
        ]}]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("no route"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await inline_remote_images(payload)

        assert result["input"][0]["content"][0]["image_url"] == "https://example.com/a.png"

    @pytest.mark.asyncio
    async def test_no_images_noop(self):
        """
        What it does: A payload without images is returned unchanged with no fetch.
        Purpose: Avoid creating an HTTP client when unnecessary.
        """
        payload = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
        with patch("httpx.AsyncClient") as mock_cls:
            result = await inline_remote_images(payload)
        assert result == payload
        mock_cls.assert_not_called()
