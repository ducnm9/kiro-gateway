# -*- coding: utf-8 -*-

"""Unit tests for the Codex (ChatGPT) upstream backend and HTTP client.

Covers:
- CodexBackend.build_headers: identity headers + ChatGPT-Account-ID binding.
- CodexBackend.build_url and static model list.
- Error helpers: message extraction, usage_limit_reached resets_at parsing,
  and raise_codex_http_error status/message mapping.
- CodexHttpClient.request_with_retry: 200 passthrough, 401/403 refresh+retry,
  refresh-failure returns rejection, 429/5xx backoff returns last response,
  network error → HTTPException.

Network is fully isolated; per-request httpx clients are patched.
"""

import datetime
import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException

from kiro.upstream_codex import (
    CodexBackend,
    extract_codex_error_message,
    parse_codex_resets_at_ms,
    raise_codex_http_error,
    CODEX_MODEL_CAPACITY_MESSAGE,
)
from kiro.http_client_codex import CodexHttpClient


# =============================================================================
# CodexBackend
# =============================================================================

class TestCodexBackendHeaders:
    """Tests for header/URL construction."""

    def test_headers_include_identity_and_account_id(self):
        """
        What it does: build_headers includes auth, identity, and ChatGPT-Account-ID.
        Purpose: Ensure requests carry the correct account binding + Codex identity.
        """
        backend = CodexBackend()
        headers = backend.build_headers(
            access_token="tok123", chatgpt_account_id="acc_1", session_id="sess_1"
        )
        assert headers["Authorization"] == "Bearer tok123"
        assert headers["ChatGPT-Account-ID"] == "acc_1"
        assert headers["originator"] == "codex_cli_rs"
        assert headers["User-Agent"].startswith("codex_cli_rs/")
        assert headers["session_id"] == "sess_1"
        assert headers["Accept"] == "text/event-stream"

    def test_headers_omit_account_id_when_absent(self):
        """
        What it does: No ChatGPT-Account-ID header when the id is missing.
        Purpose: Avoid sending an empty/invalid binding header.
        """
        backend = CodexBackend()
        headers = backend.build_headers(access_token="tok", chatgpt_account_id=None)
        assert "ChatGPT-Account-ID" not in headers
        # session_id falls back to "default" when no id/session provided.
        assert headers["session_id"] == "default"

    def test_session_id_falls_back_to_account_id(self):
        """
        What it does: session_id defaults to the account id when not given.
        Purpose: Stable per-account caching key.
        """
        backend = CodexBackend()
        headers = backend.build_headers(access_token="t", chatgpt_account_id="acc_9")
        assert headers["session_id"] == "acc_9"

    def test_build_url_returns_base(self):
        """
        What it does: build_url returns the configured base URL.
        Purpose: Correct endpoint targeting.
        """
        backend = CodexBackend()
        assert backend.build_url() == backend.base_url
        assert "codex" in backend.build_url()

    def test_models_static_registry(self):
        """
        What it does: The backend exposes the static Codex model registry.
        Purpose: Feed /v1/models without a network fetch.
        """
        backend = CodexBackend()
        assert len(backend.models) > 0
        ids = {m["id"] for m in backend.models}
        assert "gpt-5.5" in ids
        for m in backend.models:
            assert m["id"] and m["name"]


# =============================================================================
# Error helpers
# =============================================================================

class TestCodexErrorHelpers:
    """Tests for error message extraction and reset parsing."""

    def test_extract_message_nested_error(self):
        """
        What it does: Extracts error.message from a nested error body.
        Purpose: Surface a useful message to the client.
        """
        body = json.dumps({"error": {"message": "bad things"}})
        assert extract_codex_error_message(body) == "bad things"

    def test_extract_message_top_level(self):
        """
        What it does: Falls back to a top-level message field.
        Purpose: Handle alternate error shapes.
        """
        body = json.dumps({"message": "top level"})
        assert extract_codex_error_message(body) == "top level"

    def test_extract_message_non_json_truncates(self):
        """
        What it does: Non-JSON bodies are returned truncated.
        Purpose: Never crash on malformed bodies.
        """
        body = "x" * 500
        assert extract_codex_error_message(body) == "x" * 200

    def test_parse_resets_at_absolute(self):
        """
        What it does: Parses an absolute resets_at (unix seconds) to future ms.
        Purpose: Precise cooldown from upstream.
        """
        future = datetime.datetime.now(datetime.timezone.utc).timestamp() + 3600
        body = json.dumps({"error": {"type": "usage_limit_reached", "resets_at": future}})
        ms = parse_codex_resets_at_ms(body, 429)
        assert ms is not None and ms > 0

    def test_parse_resets_in_seconds(self):
        """
        What it does: Parses relative resets_in_seconds to an absolute future ms.
        Purpose: Support both reset encodings.
        """
        body = json.dumps({"error": {"type": "usage_limit_reached", "resets_in_seconds": 120}})
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        ms = parse_codex_resets_at_ms(body, 429)
        assert ms is not None and ms > now_ms

    def test_parse_resets_ignored_for_non_429(self):
        """
        What it does: Returns None for non-429 statuses.
        Purpose: resets_at is only meaningful for usage limits.
        """
        body = json.dumps({"error": {"type": "usage_limit_reached", "resets_in_seconds": 120}})
        assert parse_codex_resets_at_ms(body, 500) is None

    def test_parse_resets_ignored_for_other_error_type(self):
        """
        What it does: Returns None when the error type isn't usage_limit_reached.
        Purpose: Avoid misinterpreting unrelated 429s.
        """
        body = json.dumps({"error": {"type": "rate_limited", "resets_in_seconds": 120}})
        assert parse_codex_resets_at_ms(body, 429) is None

    def test_parse_resets_past_time_ignored(self):
        """
        What it does: A past resets_at is ignored.
        Purpose: Don't set a cooldown in the past.
        """
        past = datetime.datetime.now(datetime.timezone.utc).timestamp() - 3600
        body = json.dumps({"error": {"type": "usage_limit_reached", "resets_at": past}})
        assert parse_codex_resets_at_ms(body, 429) is None


class TestRaiseCodexHttpError:
    """Tests for raise_codex_http_error status/message mapping."""

    def _mock_response(self, status_code, body):
        resp = AsyncMock()
        resp.status_code = status_code
        resp.aread = AsyncMock(return_value=body.encode("utf-8"))
        resp.aclose = AsyncMock()
        return resp

    @pytest.mark.asyncio
    async def test_429_maps_with_reset_hint(self):
        """
        What it does: 429 usage_limit_reached raises 429 with a reset hint.
        Purpose: Actionable rate-limit error.
        """
        future = datetime.datetime.now(datetime.timezone.utc).timestamp() + 60
        body = json.dumps({"error": {"type": "usage_limit_reached", "resets_at": future}})
        resp = self._mock_response(429, body)
        with pytest.raises(HTTPException) as exc:
            await raise_codex_http_error(resp)
        assert exc.value.status_code == 429
        assert "resets at" in exc.value.detail

    @pytest.mark.asyncio
    async def test_401_maps_to_auth_message(self):
        """
        What it does: 401 raises 401 with a re-auth message.
        Purpose: Guide the user to fix credentials.
        """
        resp = self._mock_response(401, json.dumps({"error": {"message": "unauthorized"}}))
        with pytest.raises(HTTPException) as exc:
            await raise_codex_http_error(resp)
        assert exc.value.status_code == 401
        assert "Codex" in exc.value.detail

    @pytest.mark.asyncio
    async def test_generic_error_maps_status_and_message(self):
        """
        What it does: A generic error maps status + extracted message.
        Purpose: Transparent error propagation.
        """
        resp = self._mock_response(400, json.dumps({"error": {"message": "bad request"}}))
        with pytest.raises(HTTPException) as exc:
            await raise_codex_http_error(resp)
        assert exc.value.status_code == 400
        assert "bad request" in exc.value.detail


# =============================================================================
# CodexHttpClient
# =============================================================================

def _mock_auth(token="valid_token", account_id="acc_1"):
    """Build a stub CodexAuthManager with async get_access_token/force_refresh."""
    auth = AsyncMock()
    auth.get_access_token = AsyncMock(return_value=token)
    auth.force_refresh = AsyncMock(return_value="refreshed_token")
    auth.chatgpt_account_id = account_id
    return auth


def _mock_response(status_code):
    """Build a mock streaming response with a given status."""
    resp = AsyncMock()
    resp.status_code = status_code
    return resp


class TestCodexHttpClient:
    """Tests for the Codex HTTP client retry/refresh behavior."""

    @pytest.mark.asyncio
    async def test_200_returned_directly(self):
        """
        What it does: A 200 response is returned without retries.
        Purpose: Happy path.
        """
        auth = _mock_auth()
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(return_value=_mock_response(200))
        client.client = mock_http
        client._shared_client = mock_http

        resp = await client.request_with_retry({"model": "gpt-5.5"})
        assert resp.status_code == 200
        auth.get_access_token.assert_awaited()

    @pytest.mark.asyncio
    async def test_401_triggers_refresh_then_retry(self):
        """
        What it does: 401 refreshes the token, then a retry yields 200.
        Purpose: Auto token-refresh recovery.
        """
        auth = _mock_auth()
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(side_effect=[_mock_response(401), _mock_response(200)])
        client.client = mock_http
        client._shared_client = mock_http

        resp = await client.request_with_retry({"model": "gpt-5.5"})
        assert resp.status_code == 200
        auth.force_refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_failure_returns_rejection(self):
        """
        What it does: If refresh fails on 401, the rejection response is returned.
        Purpose: Let the caller mark the account FATAL and fail over.
        """
        auth = _mock_auth()
        auth.force_refresh = AsyncMock(side_effect=httpx.HTTPStatusError(
            "bad", request=Mock(), response=Mock(status_code=400)))
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(return_value=_mock_response(401))
        client.client = mock_http
        client._shared_client = mock_http

        resp = await client.request_with_retry({"model": "gpt-5.5"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_429_backoff_returns_last_response(self):
        """
        What it does: Repeated 429s exhaust retries and return the last 429.
        Purpose: Caller classifies quota exhaustion and fails over.
        """
        auth = _mock_auth()
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(return_value=_mock_response(429))
        client.client = mock_http
        client._shared_client = mock_http

        with patch("kiro.http_client_codex.asyncio.sleep", new=AsyncMock()):
            resp = await client.request_with_retry({"model": "gpt-5.5"})
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_5xx_backoff_returns_last_response(self):
        """
        What it does: Repeated 5xx exhaust retries and return the last 5xx.
        Purpose: Transient upstream errors trigger failover after backoff.
        """
        auth = _mock_auth()
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(return_value=_mock_response(503))
        client.client = mock_http
        client._shared_client = mock_http

        with patch("kiro.http_client_codex.asyncio.sleep", new=AsyncMock()):
            resp = await client.request_with_retry({"model": "gpt-5.5"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_network_error_raises_http_exception(self):
        """
        What it does: A persistent network error raises an HTTPException (502/504).
        Purpose: Surface network failures with a user-friendly code.
        """
        auth = _mock_auth()
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        mock_http = AsyncMock()
        mock_http.build_request = Mock(return_value=Mock())
        mock_http.send = AsyncMock(side_effect=httpx.ConnectError("no route"))
        client.client = mock_http
        client._shared_client = mock_http

        with patch("kiro.http_client_codex.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(HTTPException) as exc:
                await client.request_with_retry({"model": "gpt-5.5"})
        assert exc.value.status_code in (502, 504)

    @pytest.mark.asyncio
    async def test_headers_carry_account_id(self):
        """
        What it does: The outgoing request headers include the account's ChatGPT-Account-ID.
        Purpose: Verify correct per-account binding at the client layer.
        """
        auth = _mock_auth(account_id="acc_bind")
        backend = CodexBackend()
        client = CodexHttpClient(auth, backend)

        captured = {}

        def capture_build_request(method, url, **kwargs):
            captured.update(kwargs.get("headers", {}))
            return Mock()

        mock_http = AsyncMock()
        mock_http.build_request = Mock(side_effect=capture_build_request)
        mock_http.send = AsyncMock(return_value=_mock_response(200))
        client.client = mock_http
        client._shared_client = mock_http

        await client.request_with_retry({"model": "gpt-5.5"})
        assert captured.get("ChatGPT-Account-ID") == "acc_bind"
