# -*- coding: utf-8 -*-

"""Unit tests for the Codex (ChatGPT) authentication manager.

Covers:
- JWT claim decoding + ChatGPT account id backfill (id_token / access_token).
- Token expiry inspection (expired / expiring-soon / valid).
- get_access_token() refresh-on-expiry vs. reuse-when-valid.
- force_refresh() always refreshes.
- Refresh request wire format (form-encoded, grant_type=refresh_token) and
  token/expiry update, including rotated refresh token and new id_token.
- Refresh failure (revoked token) raising, and no-refresh-token guard.
- Thread-safe refresh coalescing under concurrency.
- No token values leaked into logs (masking).

Network is fully isolated: httpx.AsyncClient is patched per test.
"""

import base64
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kiro.auth_codex import (
    CodexAuthManager,
    extract_chatgpt_account_id,
    _decode_jwt_claims,
    _mask_token,
    _parse_expires_at,
)


def _make_jwt(payload: dict) -> str:
    """Build an unsigned JWT (header.payload.signature) for testing.

    Only the payload segment matters for our decoder.
    """
    def b64url(obj) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    header = b64url({"alg": "none", "typ": "JWT"})
    body = b64url(payload)
    return f"{header}.{body}.sig"


def _account_jwt(account_id: str) -> str:
    """JWT carrying the OpenAI auth claim with a chatgpt_account_id."""
    return _make_jwt({
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        "email": "user@example.com",
    })


def _patch_refresh(response_json: dict, status_code: int = 200):
    """Return a patcher for httpx.AsyncClient yielding a controlled response.

    Also returns the mock client so tests can assert on the outgoing request.
    """
    # Note: do NOT spec against httpx.AsyncClient/Response here — the global
    # block_all_network_calls fixture replaces httpx.AsyncClient with a Mock,
    # and you cannot spec a Mock. Plain AsyncMock/Mock objects are sufficient.
    mock_response = AsyncMock()
    mock_response.status_code = status_code
    mock_response.json = Mock(return_value=response_json)
    if status_code == 200:
        mock_response.raise_for_status = Mock()
    else:
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("error", request=Mock(), response=Mock(status_code=status_code))
        )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    # __aexit__ must return False so exceptions raised inside the context
    # (e.g. raise_for_status) propagate instead of being swallowed.
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("kiro.auth_codex.httpx.AsyncClient", return_value=mock_client), mock_client


# =============================================================================
# JWT decoding and account id backfill
# =============================================================================

class TestJwtDecoding:
    """Tests for JWT claim decoding and account id extraction."""

    def test_decode_valid_jwt(self):
        """
        What it does: Decodes a well-formed JWT payload.
        Purpose: Ensure base64url payload without padding is decoded.
        """
        token = _make_jwt({"foo": "bar"})
        claims = _decode_jwt_claims(token)
        assert claims == {"foo": "bar"}

    @pytest.mark.parametrize("bad", [None, "", "notajwt", "a.b", "a.b.c.d", 123, "a.%%%.c"])
    def test_decode_malformed_returns_empty(self, bad):
        """
        What it does: Malformed/absent tokens yield empty claims.
        Purpose: Ensure the decoder never raises on bad input.
        """
        assert _decode_jwt_claims(bad) == {}

    def test_extract_account_id_from_openai_auth_claim(self):
        """
        What it does: Extracts chatgpt_account_id from the OpenAI auth claim.
        Purpose: Primary backfill path.
        """
        token = _account_jwt("acc_primary")
        assert extract_chatgpt_account_id(token) == "acc_primary"

    def test_extract_account_id_top_level_fallback(self):
        """
        What it does: Falls back to a top-level chatgpt_account_id claim.
        Purpose: Support alternative token layouts.
        """
        token = _make_jwt({"chatgpt_account_id": "acc_fallback"})
        assert extract_chatgpt_account_id(token) == "acc_fallback"

    def test_extract_account_id_missing_returns_none(self):
        """
        What it does: Returns None when no account id claim is present.
        Purpose: Avoid guessing an id.
        """
        assert extract_chatgpt_account_id(_make_jwt({"email": "x@y.z"})) is None
        assert extract_chatgpt_account_id(None) is None

    def test_backfill_prefers_explicit_over_jwt(self):
        """
        What it does: Explicit chatgpt_account_id wins over JWT claim.
        Purpose: Respect operator-provided value.
        """
        mgr = CodexAuthManager(
            access_token="at",
            refresh_token="rt",
            id_token=_account_jwt("acc_from_jwt"),
            chatgpt_account_id="acc_explicit",
        )
        assert mgr.chatgpt_account_id == "acc_explicit"

    def test_backfill_from_id_token(self):
        """
        What it does: Backfills account id from id_token when not explicit.
        Purpose: Convenience for credentials that omit the id.
        """
        mgr = CodexAuthManager(
            access_token="at",
            refresh_token="rt",
            id_token=_account_jwt("acc_id_token"),
        )
        assert mgr.chatgpt_account_id == "acc_id_token"

    def test_backfill_from_access_token_when_no_id_token(self):
        """
        What it does: Backfills account id from access_token JWT as last resort.
        Purpose: Some setups only ship an access token.
        """
        mgr = CodexAuthManager(
            access_token=_account_jwt("acc_access"),
            refresh_token="rt",
        )
        assert mgr.chatgpt_account_id == "acc_access"


# =============================================================================
# Expiry inspection
# =============================================================================

class TestExpiry:
    """Tests for is_token_expiring_soon / is_token_expired."""

    def _mgr(self, expires_at):
        return CodexAuthManager(
            access_token="at", refresh_token="rt", chatgpt_account_id="acc",
            expires_at=expires_at, refresh_threshold=600,
        )

    def test_no_expiry_is_expiring_and_expired(self):
        """
        What it does: With no expiry known, both checks return True.
        Purpose: Fail-safe toward refreshing.
        """
        mgr = self._mgr(None)
        assert mgr.is_token_expiring_soon() is True
        assert mgr.is_token_expired() is True

    def test_far_future_not_expiring(self):
        """
        What it does: A token far in the future is neither expiring nor expired.
        Purpose: Reuse valid tokens.
        """
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        mgr = self._mgr(future)
        assert mgr.is_token_expiring_soon() is False
        assert mgr.is_token_expired() is False

    def test_within_threshold_is_expiring_not_expired(self):
        """
        What it does: A token expiring within the lead time is 'expiring soon'.
        Purpose: Trigger proactive refresh before hard expiry.
        """
        soon = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
        mgr = self._mgr(soon)
        assert mgr.is_token_expiring_soon() is True
        assert mgr.is_token_expired() is False

    def test_past_is_expired(self):
        """
        What it does: A past timestamp is fully expired.
        Purpose: Detect hard expiry.
        """
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        mgr = self._mgr(past)
        assert mgr.is_token_expired() is True

    def test_parse_expires_at_z_suffix(self):
        """
        What it does: Parses ISO timestamps with a trailing 'Z'.
        Purpose: Accept common UTC formatting.
        """
        dt = _parse_expires_at("2030-01-01T00:00:00Z")
        assert dt is not None and dt.tzinfo is not None

    @pytest.mark.parametrize("bad", [None, "", "not-a-date", 123])
    def test_parse_expires_at_invalid(self, bad):
        """
        What it does: Invalid expiry values parse to None.
        Purpose: Robustness against malformed credentials.
        """
        assert _parse_expires_at(bad) is None


# =============================================================================
# get_access_token / force_refresh
# =============================================================================

class TestGetAccessToken:
    """Tests for the token retrieval + refresh flow."""

    @pytest.mark.asyncio
    async def test_returns_valid_token_without_refresh(self):
        """
        What it does: Returns the cached token when still valid.
        Purpose: Avoid unnecessary refresh network calls.
        """
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        mgr = CodexAuthManager("valid_at", "rt", chatgpt_account_id="acc", expires_at=future)

        patcher, mock_client = _patch_refresh({"access_token": "should_not_be_used"})
        with patcher:
            token = await mgr.get_access_token()

        assert token == "valid_at"
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_refreshes_when_expiring(self):
        """
        What it does: Refreshes when the token is expiring soon and returns new token.
        Purpose: Core refresh-before-expiry behavior.
        """
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        mgr = CodexAuthManager("old_at", "rt", chatgpt_account_id="acc", expires_at=past)

        patcher, mock_client = _patch_refresh({
            "access_token": "new_at", "refresh_token": "new_rt", "expires_in": 3600
        })
        with patcher:
            token = await mgr.get_access_token()

        assert token == "new_at"
        mock_client.post.assert_awaited_once()
        # Verify wire format: form data, refresh_token grant.
        _, kwargs = mock_client.post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "rt"
        assert kwargs["headers"]["Content-Type"] == "application/x-www-form-urlencoded"

    @pytest.mark.asyncio
    async def test_refresh_rotates_refresh_token_and_expiry(self):
        """
        What it does: A rotated refresh_token and new expiry are stored.
        Purpose: Keep credentials current across refreshes.
        """
        mgr = CodexAuthManager("old_at", "rt", chatgpt_account_id="acc")  # no expiry → refresh

        patcher, _ = _patch_refresh({
            "access_token": "new_at", "refresh_token": "rotated_rt", "expires_in": 3600
        })
        with patcher:
            await mgr.get_access_token()

        assert mgr._refresh_token == "rotated_rt"
        assert mgr.is_token_expired() is False

    @pytest.mark.asyncio
    async def test_refresh_updates_account_id_from_new_id_token(self):
        """
        What it does: A new id_token in the refresh response updates account id.
        Purpose: Track account id changes across refreshes.
        """
        mgr = CodexAuthManager("old_at", "rt", chatgpt_account_id="acc_old")

        patcher, _ = _patch_refresh({
            "access_token": "new_at",
            "id_token": _account_jwt("acc_new"),
            "expires_in": 3600,
        })
        with patcher:
            await mgr.get_access_token()

        assert mgr.chatgpt_account_id == "acc_new"

    @pytest.mark.asyncio
    async def test_force_refresh_always_refreshes(self):
        """
        What it does: force_refresh refreshes even when the token is still valid.
        Purpose: Recover from upstream 401/403 rejections.
        """
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        mgr = CodexAuthManager("valid_at", "rt", chatgpt_account_id="acc", expires_at=future)

        patcher, mock_client = _patch_refresh({"access_token": "forced_at", "expires_in": 3600})
        with patcher:
            token = await mgr.force_refresh()

        assert token == "forced_at"
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_failure_raises(self):
        """
        What it does: A non-200 refresh (revoked token) raises HTTPStatusError.
        Purpose: Surface FATAL auth failure for account-level handling.
        """
        mgr = CodexAuthManager("old_at", "revoked_rt", chatgpt_account_id="acc")

        patcher, _ = _patch_refresh({"error": "invalid_grant"}, status_code=400)
        with patcher:
            with pytest.raises(httpx.HTTPStatusError):
                await mgr.get_access_token()

    @pytest.mark.asyncio
    async def test_missing_refresh_token_raises(self):
        """
        What it does: Refresh without a refresh token raises ValueError.
        Purpose: Guard against misconfigured accounts.
        """
        mgr = CodexAuthManager("old_at", "", chatgpt_account_id="acc")
        with pytest.raises(ValueError):
            await mgr.force_refresh()

    @pytest.mark.asyncio
    async def test_missing_access_token_in_response_raises(self):
        """
        What it does: A 200 response lacking access_token raises ValueError.
        Purpose: Detect malformed upstream responses.
        """
        mgr = CodexAuthManager("old_at", "rt", chatgpt_account_id="acc")
        patcher, _ = _patch_refresh({"refresh_token": "rt2", "expires_in": 3600})
        with patcher:
            with pytest.raises(ValueError):
                await mgr.force_refresh()

    @pytest.mark.asyncio
    async def test_concurrent_refresh_coalesces(self):
        """
        What it does: Concurrent get_access_token calls trigger a single refresh.
        Purpose: Lock prevents duplicate refresh network calls.
        """
        import asyncio
        mgr = CodexAuthManager("old_at", "rt", chatgpt_account_id="acc")  # expiring → refresh

        patcher, mock_client = _patch_refresh({"access_token": "new_at", "expires_in": 3600})
        with patcher:
            results = await asyncio.gather(*[mgr.get_access_token() for _ in range(5)])

        assert all(r == "new_at" for r in results)
        # After the first refresh the token is valid, so only one POST happens.
        assert mock_client.post.await_count == 1


# =============================================================================
# Logging safety
# =============================================================================

class TestLoggingSafety:
    """Tests that token values are never logged verbatim."""

    def test_mask_token(self):
        """
        What it does: Masks long and short tokens and None.
        Purpose: Ensure previews never reveal full tokens.
        """
        assert _mask_token(None) == "<none>"
        assert _mask_token("short") == "****"
        masked = _mask_token("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9secret")
        assert "secret" not in masked
        assert masked.startswith("eyJh")

    @pytest.mark.asyncio
    async def test_refresh_does_not_log_token_values(self):
        """
        What it does: Verifies the refresh flow does not emit raw token strings.
        Purpose: Prevent credential leakage in logs.
        """
        from loguru import logger

        captured = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="DEBUG")
        try:
            mgr = CodexAuthManager("supersecretaccesstoken", "supersecretrefreshtoken",
                                   chatgpt_account_id="acc")
            patcher, _ = _patch_refresh({"access_token": "newsecrettoken", "expires_in": 3600})
            with patcher:
                await mgr.force_refresh()
        finally:
            logger.remove(sink_id)

        joined = "\n".join(captured)
        assert "supersecretrefreshtoken" not in joined
        assert "supersecretaccesstoken" not in joined
        assert "newsecrettoken" not in joined



# =============================================================================
# Protocol conformance
# =============================================================================

class TestProtocolConformance:
    """Tests that auth managers satisfy the UpstreamAuthManager protocol."""

    def test_codex_manager_is_upstream_auth_manager(self):
        """
        What it does: CodexAuthManager satisfies UpstreamAuthManager structurally.
        Purpose: Ensure it can be driven by the Account System.
        """
        from kiro.auth_base import UpstreamAuthManager
        mgr = CodexAuthManager("at", "rt", chatgpt_account_id="acc")
        assert isinstance(mgr, UpstreamAuthManager)
        assert mgr.provider == "chatgpt"

    def test_kiro_manager_is_upstream_auth_manager(self):
        """
        What it does: KiroAuthManager satisfies UpstreamAuthManager structurally.
        Purpose: Ensure the existing manager keeps working under the shared contract.
        """
        from kiro.auth_base import UpstreamAuthManager
        from kiro.auth import KiroAuthManager
        mgr = KiroAuthManager(refresh_token="rt", region="us-east-1")
        assert isinstance(mgr, UpstreamAuthManager)
        assert mgr.provider == "kiro"
