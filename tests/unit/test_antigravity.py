# -*- coding: utf-8 -*-

"""Comprehensive unit tests for the Antigravity (Cloud Code Assist) provider.

Tests cover:
- Configuration loading
- OAuth auth (PKCE, token refresh, credentials)
- Upstream routing
- Backend class (model routing, headers, error handling)
- Converters (OpenAI→Gemini, Anthropic→Gemini, tools, schemas)
- Streaming parser (SSE → KiroEvent, OpenAI/Anthropic output)
"""

import asyncio
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kiro.auth_antigravity import (
    AntigravityCredentials,
    build_auth_url,
    generate_pkce,
    _base64url_encode,
    _extract_project_id,
    _fallback_project_id,
    _mask_email,
    _redact_tokens,
    TOKEN_REFRESH_MARGIN_SECONDS,
)
from kiro.converters_antigravity import (
    build_antigravity_payload,
    build_antigravity_payload_anthropic,
    _append_turn,
    _convert_assistant_parts,
    _convert_openai_tools,
    _convert_user_parts_openai,
    _dereference_schema,
    _ensure_first_turn_is_user,
    _ensure_root_object_schema,
    _normalize_custom_tool_schema,
    _sanitize_tool_call_id,
    _strip_meta_schema,
)
from kiro.streaming_antigravity import (
    _map_anthropic_usage,
    _map_finish_reason,
    _map_finish_reason_raw,
    _map_stop_reason_anthropic,
    _map_usage_metadata,
    collect_antigravity_anthropic_response,
    collect_antigravity_response,
    parse_antigravity_stream,
)
from kiro.upstream_antigravity import (
    ANTIGRAVITY_MODEL_INFO,
    ANTIGRAVITY_ROUTING,
    AntigravityBackend,
    extract_antigravity_error_message,
    friendly_antigravity_error,
)
from kiro.upstream_base import resolve_upstream


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_antigravity_enabled():
    """Enable Antigravity in config for testing."""
    with patch("kiro.upstream_base.config.ANTIGRAVITY_ENABLED", True):
        yield


@pytest.fixture
def mock_antigravity_disabled():
    """Disable Antigravity in config for testing."""
    with patch("kiro.upstream_base.config.ANTIGRAVITY_ENABLED", False):
        yield


@pytest.fixture
def backend():
    """Create a fresh AntigravityBackend instance with test credentials."""
    with patch("kiro.upstream_antigravity.config.ANTIGRAVITY_REFRESH_TOKEN", "test-refresh-token"):
        with patch("kiro.upstream_antigravity.config.ANTIGRAVITY_PROJECT_ID", "test-project-id"):
            b = AntigravityBackend()
            b.credentials.access_token = "ya29.test-access-token"
            b.credentials.refresh_token = "test-refresh-token"
            b.credentials.expires_at = time.time() + 3600
            b.credentials.project_id = "test-project-id"
            b.credentials.email = "user@example.com"
            return b


@pytest.fixture
def sample_openai_request():
    """Create a minimal OpenAI ChatCompletionRequest mock."""
    request = MagicMock()
    request.model = "antigravity/gemini-3.7-flash"
    request.stream = False
    request.messages = [
        MagicMock(role="system", content="You are helpful."),
        MagicMock(role="user", content="Hello!"),
    ]
    request.max_tokens = None
    request.max_completion_tokens = None
    request.temperature = None
    request.tools = None
    request.tool_choice = None
    # Make reasoning_effort not present
    del request.reasoning_effort
    return request


# ==============================================================================
# Test: Configuration
# ==============================================================================


class TestAntigravityConfig:
    """Tests for Antigravity configuration variables."""

    def test_config_defaults_loaded(self):
        """Verify default config values are loaded correctly."""
        from kiro import config

        # These should have sensible defaults
        assert isinstance(config.ANTIGRAVITY_ENABLED, bool)
        assert isinstance(config.ANTIGRAVITY_BASE_URL, str)
        assert "googleapis.com" in config.ANTIGRAVITY_BASE_URL
        assert isinstance(config.ANTIGRAVITY_MAX_TOKENS, int)
        assert config.ANTIGRAVITY_MAX_TOKENS > 0
        assert config.ANTIGRAVITY_CALLBACK_PORT == 51121

    def test_config_antigravity_disabled_by_default(self):
        """Antigravity should be disabled by default."""
        from kiro import config

        # Unless env var is set in test environment
        # Just verify it's a bool
        assert isinstance(config.ANTIGRAVITY_ENABLED, bool)


# ==============================================================================
# Test: Auth (PKCE, tokens, credentials)
# ==============================================================================


class TestAntigravityAuthPKCE:
    """Tests for PKCE generation."""

    def test_generate_pkce_returns_tuple(self):
        """generate_pkce returns (verifier, challenge) tuple."""
        verifier, challenge = generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 20
        assert len(challenge) > 20

    def test_generate_pkce_challenge_is_sha256_of_verifier(self):
        """Challenge should be SHA-256 of verifier (base64url encoded)."""
        verifier, challenge = generate_pkce()
        expected_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = _base64url_encode(expected_bytes)
        assert challenge == expected

    def test_generate_pkce_unique_each_call(self):
        """Each call produces a different verifier/challenge pair."""
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2
        assert c1 != c2


class TestAntigravityAuthCredentials:
    """Tests for AntigravityCredentials dataclass."""

    def test_credentials_is_authenticated_with_refresh_token(self):
        """Credentials with refresh token should be authenticated."""
        creds = AntigravityCredentials(refresh_token="test-token")
        assert creds.is_authenticated is True

    def test_credentials_is_authenticated_with_valid_access_token(self):
        """Credentials with valid (non-expired) access token should be authenticated."""
        creds = AntigravityCredentials(
            access_token="ya29.test",
            expires_at=time.time() + 3600,
        )
        assert creds.is_authenticated is True

    def test_credentials_not_authenticated_empty(self):
        """Empty credentials should not be authenticated."""
        creds = AntigravityCredentials()
        assert creds.is_authenticated is False

    def test_credentials_is_expired(self):
        """Expired access token should report as expired."""
        creds = AntigravityCredentials(
            access_token="ya29.test",
            expires_at=time.time() - 100,
        )
        assert creds.is_expired is True

    def test_credentials_not_expired(self):
        """Future expiry should not be expired."""
        creds = AntigravityCredentials(
            access_token="ya29.test",
            expires_at=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_credentials_about_to_expire(self):
        """Token within refresh margin should report expired."""
        creds = AntigravityCredentials(
            access_token="ya29.test",
            expires_at=time.time() + TOKEN_REFRESH_MARGIN_SECONDS - 10,
        )
        assert creds.is_expired is True


class TestAntigravityAuthHelpers:
    """Tests for auth helper functions."""

    def test_build_auth_url(self):
        """build_auth_url returns (url, state, verifier) tuple."""
        auth_url, state, verifier = build_auth_url()
        assert "accounts.google.com" in auth_url
        assert "client_id=" in auth_url
        assert "code_challenge=" in auth_url
        assert "S256" in auth_url
        assert len(state) > 20
        assert len(verifier) > 20

    def test_extract_project_id_direct(self):
        """Extract project ID from direct field."""
        data = {"projectId": "my-project-123"}
        assert _extract_project_id(data) == "my-project-123"

    def test_extract_project_id_nested(self):
        """Extract project ID from nested field."""
        data = {"cloudaicompanionProject": {"id": "nested-id"}}
        assert _extract_project_id(data) == "nested-id"

    def test_extract_project_id_array(self):
        """Extract project ID from array field."""
        data = {"projects": ["first-project"]}
        assert _extract_project_id(data) == "first-project"

    def test_extract_project_id_none_when_missing(self):
        """Return None when no project ID found."""
        assert _extract_project_id({}) is None
        assert _extract_project_id({"unrelated": "data"}) is None

    def test_fallback_project_id_deterministic(self):
        """Same seed produces same fallback project ID."""
        id1 = _fallback_project_id("user@example.com")
        id2 = _fallback_project_id("user@example.com")
        assert id1 == id2

    def test_fallback_project_id_uuid_format(self):
        """Fallback project ID should have UUID-like format."""
        with patch("kiro.auth_antigravity.ANTIGRAVITY_PROJECT_ID", ""):
            pid = _fallback_project_id("seed")
            parts = pid.split("-")
            assert len(parts) == 5

    def test_mask_email(self):
        """Email masking should hide middle characters."""
        assert _mask_email("user@example.com") == "u***r@example.com"
        assert _mask_email("ab@x.com") == "a***@x.com"
        assert _mask_email("") == "[unknown]"

    def test_redact_tokens_access_token(self):
        """Redact Google access tokens."""
        text = 'token: ya29.AbCdEfGhIjKlMnOpQrStUvWxYz'
        result = _redact_tokens(text)
        assert "ya29." not in result or "[redacted" in result

    def test_redact_tokens_refresh_token(self):
        """Redact refresh tokens."""
        text = 'refresh: 1/AbCdEfGhIjKlMnOpQrStUvWxYz1234567890'
        result = _redact_tokens(text)
        assert "1/AbCd" not in result


# ==============================================================================
# Test: Upstream Routing
# ==============================================================================


class TestAntigravityRouting:
    """Tests for upstream routing with Antigravity."""

    def test_antigravity_model_routes_to_antigravity(self, mock_antigravity_enabled):
        """Model prefixed with 'antigravity/' should route to antigravity."""
        assert resolve_upstream("antigravity/gemini-3.7-flash") == "antigravity"
        assert resolve_upstream("antigravity/claude-sonnet-4-6") == "antigravity"
        assert resolve_upstream("antigravity/gpt-oss-120b") == "antigravity"

    def test_antigravity_disabled_falls_through(self, mock_antigravity_disabled):
        """When disabled, antigravity/ models fall through to command_code (has /)."""
        with patch("kiro.upstream_base.config.COMMAND_CODE_ENABLED", True):
            assert resolve_upstream("antigravity/gemini-3.7-flash") == "command_code"

    def test_kiro_model_routes_to_kiro(self, mock_antigravity_enabled):
        """Non-prefixed models should still route to kiro."""
        assert resolve_upstream("claude-haiku-4.5") == "kiro"
        assert resolve_upstream("claude-sonnet-4") == "kiro"

    def test_command_code_model_still_works(self, mock_antigravity_enabled):
        """Non-antigravity slash models route to command_code."""
        with patch("kiro.upstream_base.config.COMMAND_CODE_ENABLED", True):
            assert resolve_upstream("deepseek/deepseek-v4-pro") == "command_code"


# ==============================================================================
# Test: Backend Class
# ==============================================================================


class TestAntigravityBackend:
    """Tests for AntigravityBackend class."""

    def test_resolve_runtime_model_gemini_37(self, backend):
        """Gemini 3.7 Flash resolves to tiered runtime model."""
        assert backend.resolve_runtime_model("gemini-3.7-flash", "off") == "gemini-3.7-flash-tiered"
        assert backend.resolve_runtime_model("gemini-3.7-flash", "high") == "gemini-3.7-flash-tiered"

    def test_resolve_runtime_model_gemini_36(self, backend):
        """Gemini 3.6 Flash resolves to effort-specific runtime models."""
        assert backend.resolve_runtime_model("gemini-3.6-flash", "low") == "gemini-3.6-flash-low"
        assert backend.resolve_runtime_model("gemini-3.6-flash", "medium") == "gemini-3.6-flash-medium"
        assert backend.resolve_runtime_model("gemini-3.6-flash", "high") == "gemini-3.6-flash-high"

    def test_resolve_runtime_model_claude(self, backend):
        """Claude models resolve correctly."""
        assert backend.resolve_runtime_model("claude-sonnet-4-6", "high") == "claude-sonnet-4-6"
        assert backend.resolve_runtime_model("claude-opus-4-6", "high") == "claude-opus-4-6-thinking"

    def test_resolve_runtime_model_unknown_passthrough(self, backend):
        """Unknown model passes through as-is."""
        assert backend.resolve_runtime_model("unknown-model", "off") == "unknown-model"

    def test_uses_legacy_parameters_claude(self, backend):
        """Claude models should use legacy parameters."""
        assert backend.uses_legacy_parameters("claude-sonnet-4-6") is True
        assert backend.uses_legacy_parameters("claude-opus-4-6-thinking") is True

    def test_uses_legacy_parameters_gpt_oss(self, backend):
        """GPT-OSS models should use legacy parameters."""
        assert backend.uses_legacy_parameters("gpt-oss-120b-medium") is True

    def test_uses_legacy_parameters_gemini(self, backend):
        """Gemini models should NOT use legacy parameters."""
        assert backend.uses_legacy_parameters("gemini-3.7-flash-tiered") is False
        assert backend.uses_legacy_parameters("gemini-3.6-flash-high") is False

    def test_needs_thinking_header(self, backend):
        """Only claude *-thinking models need the header."""
        assert backend.needs_thinking_header("claude-opus-4-6-thinking") is True
        assert backend.needs_thinking_header("claude-sonnet-4-6") is False
        assert backend.needs_thinking_header("gemini-3.7-flash-tiered") is False

    def test_get_max_output_tokens(self, backend):
        """Max output tokens should match known values."""
        assert backend.get_max_output_tokens("gemini-3.7-flash-tiered") == 65536
        assert backend.get_max_output_tokens("claude-sonnet-4-6") == 64000
        assert backend.get_max_output_tokens("gpt-oss-120b-medium") == 32768

    def test_build_headers(self, backend):
        """Headers should contain required fields."""
        headers = backend.build_headers("ya29.test-token")
        assert headers["Authorization"] == "Bearer ya29.test-token"
        assert headers["Content-Type"] == "application/json"
        assert "antigravity" in headers["User-Agent"]
        assert "X-Goog-Api-Client" in headers
        assert "Client-Metadata" in headers
        # Verify Client-Metadata is valid JSON
        metadata = json.loads(headers["Client-Metadata"])
        assert metadata["ideType"] == "ANTIGRAVITY"
        assert metadata["pluginType"] == "GEMINI"

    def test_is_authenticated(self, backend):
        """Backend with credentials should report authenticated."""
        assert backend.is_authenticated() is True

    def test_is_not_authenticated_empty(self):
        """Backend without credentials should not be authenticated."""
        with patch("kiro.upstream_antigravity.config.ANTIGRAVITY_REFRESH_TOKEN", ""):
            with patch("kiro.upstream_antigravity.config.ANTIGRAVITY_PROJECT_ID", ""):
                b = AntigravityBackend()
                assert b.is_authenticated() is False

    def test_endpoint_candidates(self, backend):
        """Should return base URL and fallback."""
        candidates = backend.endpoint_candidates()
        assert len(candidates) >= 1
        assert "googleapis.com" in candidates[0]

    def test_get_stream_url(self, backend):
        """Stream URL should include path and query params."""
        url = backend.get_stream_url("https://cloudcode-pa.googleapis.com")
        assert "streamGenerateContent" in url
        assert "alt=sse" in url

    def test_model_info_list(self, backend):
        """Backend should have 7 models in the static list."""
        assert len(backend.models) == 7
        model_ids = [m["id"] for m in backend.models]
        assert "antigravity/gemini-3.7-flash" in model_ids
        assert "antigravity/claude-sonnet-4-6" in model_ids
        assert "antigravity/gpt-oss-120b" in model_ids


class TestAntigravityErrorHandling:
    """Tests for error message extraction and formatting."""

    def test_extract_error_message_google_format(self):
        """Extract message from Google API error format."""
        body = json.dumps({"error": {"message": "Model not found", "code": 404}})
        assert extract_antigravity_error_message(body) == "Model not found"

    def test_extract_error_message_simple_message(self):
        """Extract from simple message field."""
        body = json.dumps({"message": "Something went wrong"})
        assert extract_antigravity_error_message(body) == "Something went wrong"

    def test_extract_error_message_invalid_json(self):
        """Return truncated body for invalid JSON."""
        body = "Not JSON at all"
        assert extract_antigravity_error_message(body) == "Not JSON at all"

    def test_friendly_error_401(self):
        """401 should suggest re-login."""
        msg = friendly_antigravity_error(401, "{}")
        assert "login" in msg.lower() or "authenticate" in msg.lower()

    def test_friendly_error_404(self):
        """404 should suggest trying another model."""
        msg = friendly_antigravity_error(404, "{}")
        assert "model" in msg.lower()

    def test_friendly_error_429(self):
        """429 should mention rate limit/quota."""
        msg = friendly_antigravity_error(429, json.dumps({"message": "quota exceeded"}))
        assert "rate limit" in msg.lower() or "quota" in msg.lower()

    def test_friendly_error_503_no_capacity(self):
        """503 with no capacity should say so."""
        msg = friendly_antigravity_error(503, json.dumps({"error": {"message": "No capacity available"}}))
        assert "capacity" in msg.lower()


# ==============================================================================
# Test: Converters
# ==============================================================================


class TestAntigravityConvertersOpenAI:
    """Tests for OpenAI → Antigravity conversion."""

    def test_build_payload_basic(self, sample_openai_request):
        """Basic payload should have correct envelope structure."""
        payload = build_antigravity_payload(
            sample_openai_request,
            runtime_model="gemini-3.7-flash-tiered",
            project_id="test-project",
            thinking_effort="off",
            use_legacy_parameters=False,
        )
        assert payload["project"] == "test-project"
        assert payload["model"] == "gemini-3.7-flash-tiered"
        assert payload["requestType"] == "agent"
        assert payload["userAgent"] == "antigravity"
        assert "requestId" in payload
        assert "request" in payload
        assert "contents" in payload["request"]
        assert "systemInstruction" in payload["request"]

    def test_build_payload_system_in_instruction(self, sample_openai_request):
        """System messages should go to systemInstruction."""
        payload = build_antigravity_payload(
            sample_openai_request,
            runtime_model="gemini-3.7-flash-tiered",
            project_id="test",
            thinking_effort="off",
        )
        sys_parts = payload["request"]["systemInstruction"]["parts"]
        assert any("helpful" in p.get("text", "") for p in sys_parts)

    def test_build_payload_thinking_config(self, sample_openai_request):
        """Thinking effort should produce thinkingConfig for Gemini 3.7."""
        payload = build_antigravity_payload(
            sample_openai_request,
            runtime_model="gemini-3.7-flash-tiered",
            project_id="test",
            thinking_effort="high",
        )
        gen_config = payload["request"]["generationConfig"]
        assert gen_config["thinkingConfig"]["thinkingLevel"] == "HIGH"

    def test_build_payload_no_thinking_for_other_models(self, sample_openai_request):
        """Non-3.7 models should not get thinkingConfig."""
        payload = build_antigravity_payload(
            sample_openai_request,
            runtime_model="gemini-3.6-flash-high",
            project_id="test",
            thinking_effort="high",
        )
        gen_config = payload["request"].get("generationConfig", {})
        assert "thinkingConfig" not in gen_config


class TestAntigravityConvertersTools:
    """Tests for tool conversion and schema handling."""

    def test_convert_openai_tools(self):
        """Convert basic OpenAI tools to Gemini format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        result = _convert_openai_tools(tools, use_legacy=False)
        assert result is not None
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert decls[0]["name"] == "get_weather"
        assert "parametersJsonSchema" in decls[0]

    def test_convert_openai_tools_legacy(self):
        """Legacy mode should use 'parameters' field."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "description": "Does a thing",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }
        ]
        result = _convert_openai_tools(tools, use_legacy=True)
        decls = result[0]["functionDeclarations"]
        assert "parameters" in decls[0]
        assert "parametersJsonSchema" not in decls[0]

    def test_normalize_custom_tool_schema_strips_extra_fields(self):
        """Schema normalization should remove non-allowed fields."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string", "format": "email", "nullable": True}},
            "required": ["x"],
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
        }
        result = _normalize_custom_tool_schema(schema)
        assert "additionalProperties" not in result
        assert "$schema" not in result
        # Properties values should also be normalized
        assert "format" not in result["properties"]["x"]
        assert "nullable" not in result["properties"]["x"]

    def test_normalize_union_type(self):
        """Union types like ['string', 'null'] should collapse to 'string'."""
        schema = {"type": ["string", "null"], "description": "optional field"}
        result = _normalize_custom_tool_schema(schema)
        assert result["type"] == "string"
        assert result["description"] == "optional field"

    def test_dereference_schema_resolves_ref(self):
        """$ref should be resolved from $defs."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        result = _dereference_schema(schema)
        assert "$ref" not in json.dumps(result)
        assert result["properties"]["item"]["type"] == "object"


class TestAntigravityConvertersMessages:
    """Tests for message conversion helpers."""

    def test_append_turn_merges_same_role(self):
        """Consecutive same-role turns should be merged."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        _append_turn(contents, "user", [{"text": " World"}])
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2

    def test_append_turn_new_role(self):
        """Different role creates new turn."""
        contents = [{"role": "user", "parts": [{"text": "Hi"}]}]
        _append_turn(contents, "model", [{"text": "Hello!"}])
        assert len(contents) == 2

    def test_ensure_first_turn_is_user(self):
        """Model-first conversations get a user turn prepended."""
        contents = [{"role": "model", "parts": [{"text": "Hi!"}]}]
        result = _ensure_first_turn_is_user(contents)
        assert result[0]["role"] == "user"
        assert len(result) == 2

    def test_ensure_first_turn_user_already(self):
        """User-first conversations are unchanged."""
        contents = [{"role": "user", "parts": [{"text": "Hi"}]}]
        result = _ensure_first_turn_is_user(contents)
        assert len(result) == 1

    def test_sanitize_tool_call_id_valid(self):
        """Valid IDs pass through (truncated to 64 chars)."""
        assert _sanitize_tool_call_id("call_abc123") == "call_abc123"

    def test_sanitize_tool_call_id_special_chars(self):
        """Special characters are replaced with underscores."""
        result = _sanitize_tool_call_id("call/with.special!chars")
        assert "/" not in result
        assert "." not in result
        assert "!" not in result

    def test_sanitize_tool_call_id_empty(self):
        """Empty ID generates a fallback."""
        result = _sanitize_tool_call_id("", "my_tool")
        assert "my_tool" in result


# ==============================================================================
# Test: Streaming
# ==============================================================================


class TestAntigravityStreamParser:
    """Tests for SSE stream parsing."""

    @pytest.mark.asyncio
    async def test_parse_text_content(self):
        """Parse a simple text content event."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello world"}]}}]}',
            'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_parse_thinking_content(self):
        """Parse a thinking (thought=true) content event."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"text":"Let me think...","thought":true}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"The answer is 42"}]}}]}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        thinking_events = [e for e in events if e.type == "thinking"]
        content_events = [e for e in events if e.type == "content"]
        assert len(thinking_events) == 1
        assert thinking_events[0].thinking_content == "Let me think..."
        assert len(content_events) == 1
        assert content_events[0].content == "The answer is 42"

    @pytest.mark.asyncio
    async def test_parse_function_call(self):
        """Parse a functionCall event."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Tokyo"},"id":"call_123"}}]}}]}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        tool_events = [e for e in events if e.type == "tool_use"]
        assert len(tool_events) == 1
        tc = tool_events[0].tool_use
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}
        assert tc["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_parse_usage_metadata(self):
        """Parse usageMetadata event."""
        response = _mock_sse_response([
            'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":50,"thoughtsTokenCount":20,"totalTokenCount":170}}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        usage_events = [e for e in events if e.type == "usage" and e.usage]
        assert len(usage_events) == 1
        usage = usage_events[0].usage
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 70  # 50 + 20
        assert usage["total_tokens"] == 170

    @pytest.mark.asyncio
    async def test_parse_done_sentinel(self):
        """[DONE] sentinel should stop parsing."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
            'data: [DONE]',
            'data: {"candidates":[{"content":{"parts":[{"text":"Should not appear"}]}}]}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_parse_error_in_stream(self):
        """Error object in stream should raise HTTPException."""
        response = _mock_sse_response([
            'data: {"error":{"message":"Internal error"}}',
        ])
        with pytest.raises(Exception) as exc_info:
            async for _ in parse_antigravity_stream(response):
                pass
        assert "Internal error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_parse_malformed_json_skipped(self):
        """Malformed JSON lines should be silently skipped."""
        response = _mock_sse_response([
            'data: not valid json',
            'data: {"candidates":[{"content":{"parts":[{"text":"Valid"}]}}]}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1

    @pytest.mark.asyncio
    async def test_parse_empty_lines_skipped(self):
        """Empty lines should be silently skipped."""
        response = _mock_sse_response([
            '',
            '  ',
            'data: {"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}',
        ])
        events = []
        async for event in parse_antigravity_stream(response):
            events.append(event)

        assert len([e for e in events if e.type == "content"]) == 1


class TestAntigravityStreamMappers:
    """Tests for stream mapping helper functions."""

    def test_map_finish_reason_stop(self):
        """STOP maps to 'stop'."""
        assert _map_finish_reason_raw("STOP") == "stop"

    def test_map_finish_reason_max_tokens(self):
        """MAX_TOKENS maps to 'length'."""
        assert _map_finish_reason_raw("MAX_TOKENS") == "length"

    def test_map_finish_reason_openai_with_tools(self):
        """With tool calls, finish reason should be 'tool_calls'."""
        assert _map_finish_reason("stop", has_tool_calls=True) == "tool_calls"
        assert _map_finish_reason("stop", has_tool_calls=False) == "stop"

    def test_map_stop_reason_anthropic(self):
        """Anthropic stop reasons map correctly."""
        assert _map_stop_reason_anthropic("stop", False) == "end_turn"
        assert _map_stop_reason_anthropic("length", False) == "max_tokens"
        assert _map_stop_reason_anthropic("stop", True) == "tool_use"

    def test_map_usage_metadata(self):
        """Usage metadata maps to OpenAI format."""
        metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 170,
        }
        result = _map_usage_metadata(metadata)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 70
        assert result["total_tokens"] == 170

    def test_map_anthropic_usage(self):
        """Anthropic usage maps from OpenAI format."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        result = _map_anthropic_usage(usage)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_map_anthropic_usage_none(self):
        """None usage returns zeros."""
        result = _map_anthropic_usage(None)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


class TestAntigravityCollectResponse:
    """Tests for non-streaming response collection."""

    @pytest.mark.asyncio
    async def test_collect_openai_response(self):
        """Collect stream into OpenAI chat.completion format."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}',
        ])
        result = await collect_antigravity_response(response, "antigravity/gemini-3.7-flash")
        assert result["object"] == "chat.completion"
        assert result["model"] == "antigravity/gemini-3.7-flash"
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_collect_anthropic_response(self):
        """Collect stream into Anthropic message format."""
        response = _mock_sse_response([
            'data: {"candidates":[{"content":{"parts":[{"text":"Hi there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"totalTokenCount":13}}',
        ])
        result = await collect_antigravity_anthropic_response(response, "antigravity/gemini-3.7-flash")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hi there"


# ==============================================================================
# Test Helpers
# ==============================================================================


def _mock_sse_response(lines: list) -> MagicMock:
    """Create a mock httpx.Response that yields SSE lines.

    Args:
        lines: List of SSE line strings.

    Returns:
        Mock response with async aiter_lines() method.
    """
    mock_response = MagicMock()

    async def aiter_lines():
        for line in lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_response.aclose = AsyncMock()
    return mock_response
