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

"""Authentication manager for the ChatGPT (Codex) upstream.

Manages the OAuth token lifecycle for a single ChatGPT account used against the
Codex backend:

- Holds access/refresh/id tokens plus the ChatGPT account id.
- Refreshes the access token before expiry (or on demand) via the OpenAI OAuth
  token endpoint using the ``refresh_token`` grant (form-encoded).
- Thread-safe refresh using ``asyncio.Lock``.
- Backfills the ChatGPT account id from the id/access token JWT claims when the
  credentials file omits it.

Implements the ``UpstreamAuthManager`` protocol so it can be driven by the
Account System alongside ``KiroAuthManager``.

SECURITY: Token values are NEVER logged. Only masked previews are emitted.
"""

import asyncio
import base64
import binascii
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from kiro.config import (
    CHATGPT_OAUTH_CLIENT_ID,
    CHATGPT_OAUTH_SCOPE,
    CHATGPT_TOKEN_REFRESH_THRESHOLD,
    get_codex_refresh_url,
)


def _mask_token(token: Optional[str]) -> str:
    """Return a masked preview of a token for safe logging.

    Args:
        token: The token to mask (may be None/empty).

    Returns:
        A masked string like ``"eyJh…last4"`` that never reveals the full token.
    """
    if not token:
        return "<none>"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}…{token[-4:]}"


def _decode_jwt_claims(token: Optional[str]) -> Dict[str, Any]:
    """Best-effort decode of a JWT payload without signature verification.

    Used only to read non-sensitive identity claims (e.g. the ChatGPT account id
    and plan type). Never raises: returns an empty dict on any problem.

    Args:
        token: A JWT (id_token or access_token), or None.

    Returns:
        The decoded payload claims as a dict, or an empty dict if not decodable.
    """
    if not token or not isinstance(token, str):
        return {}
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    # JWT uses base64url without padding; restore padding before decoding.
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(decoded.decode("utf-8"))
        return claims if isinstance(claims, dict) else {}
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def extract_chatgpt_account_id(token: Optional[str]) -> Optional[str]:
    """Extract the ChatGPT account id from a JWT's OpenAI auth claim.

    OpenAI embeds account info under the ``https://api.openai.com/auth`` claim,
    e.g. ``{"chatgpt_account_id": "acc_..."}``. Falls back to a few alternative
    claim shapes seen across token versions.

    Args:
        token: A JWT id_token or access_token.

    Returns:
        The ChatGPT account id string, or None if not present/decodable.
    """
    claims = _decode_jwt_claims(token)
    if not claims:
        return None
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        acc = auth.get("chatgpt_account_id") or auth.get("account_id")
        if isinstance(acc, str) and acc:
            return acc
    # Fallbacks for alternative token layouts.
    for key in ("chatgpt_account_id", "account_id"):
        val = claims.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _parse_expires_at(value: Any) -> Optional[datetime]:
    """Parse an ``expiresAt`` value (ISO string) into an aware datetime.

    Args:
        value: The raw ``expiresAt`` field (ISO 8601 string) or None.

    Returns:
        A timezone-aware datetime in UTC, or None if unparseable/absent.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        # Support trailing "Z" (UTC) which fromisoformat rejects before 3.11.
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


class CodexAuthManager:
    """Token lifecycle manager for a single ChatGPT (Codex) account.

    Attributes:
        provider: Provider identifier, always ``"chatgpt"``.
        chatgpt_account_id: The ChatGPT account id used for the
            ``ChatGPT-Account-ID`` request header.
    """

    provider: str = "chatgpt"

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        id_token: Optional[str] = None,
        chatgpt_account_id: Optional[str] = None,
        expires_at: Optional[str] = None,
        client_id: Optional[str] = None,
        token_url: Optional[str] = None,
        scope: Optional[str] = None,
        refresh_threshold: Optional[int] = None,
    ) -> None:
        """Initialize the Codex auth manager for one account.

        Args:
            access_token: Current OAuth access token.
            refresh_token: OAuth refresh token used to renew the access token.
            id_token: Optional OIDC id token (used to backfill the account id).
            chatgpt_account_id: The ChatGPT account id; backfilled from the JWT
                claims of ``id_token``/``access_token`` when omitted.
            expires_at: Optional ISO-8601 access-token expiry timestamp.
            client_id: OAuth client id (defaults to configured value).
            token_url: OAuth token endpoint (defaults to configured value).
            scope: OAuth scope for refresh (defaults to configured value).
            refresh_threshold: Seconds before expiry to trigger refresh
                (defaults to configured value).
        """
        self._access_token: str = access_token
        self._refresh_token: str = refresh_token
        self._id_token: Optional[str] = id_token
        self._expires_at: Optional[datetime] = _parse_expires_at(expires_at)
        self._client_id: str = client_id or CHATGPT_OAUTH_CLIENT_ID
        self._token_url: str = token_url or get_codex_refresh_url()
        self._scope: str = scope or CHATGPT_OAUTH_SCOPE
        self._refresh_threshold: int = (
            refresh_threshold if refresh_threshold is not None else CHATGPT_TOKEN_REFRESH_THRESHOLD
        )
        self._lock = asyncio.Lock()

        # Backfill the account id from JWT claims when not provided explicitly.
        self.chatgpt_account_id: Optional[str] = (
            chatgpt_account_id
            or extract_chatgpt_account_id(id_token)
            or extract_chatgpt_account_id(access_token)
        )

    def is_token_expiring_soon(self) -> bool:
        """Return True if the token is expired or within the refresh lead time.

        When no expiry is known, assume a refresh is needed (fail-safe).

        Returns:
            True if a refresh should happen before the next request.
        """
        if not self._expires_at:
            return True
        threshold = datetime.now(timezone.utc).timestamp() + self._refresh_threshold
        return self._expires_at.timestamp() <= threshold

    def is_token_expired(self) -> bool:
        """Return True if the access token is already fully expired.

        When no expiry is known, assume expired (fail-safe).

        Returns:
            True if the token can no longer be used.
        """
        if not self._expires_at:
            return True
        return datetime.now(timezone.utc) >= self._expires_at

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing before expiry when needed.

        Thread-safe via an ``asyncio.Lock``; concurrent callers coalesce so the
        refresh network call happens at most once per expiry window.

        Returns:
            A valid access token.

        Raises:
            ValueError: If no valid token can be obtained.
        """
        async with self._lock:
            if self._access_token and not self.is_token_expiring_soon():
                return self._access_token
            await self._refresh_token_request()
            if not self._access_token:
                raise ValueError("Failed to obtain Codex access token")
            return self._access_token

    async def force_refresh(self) -> str:
        """Force a token refresh regardless of current expiry.

        Used when the Codex backend rejects the current token (401/403).

        Returns:
            The newly refreshed access token.

        Raises:
            ValueError: If the response lacks an access token.
            httpx.HTTPError: On network/HTTP failure.
        """
        async with self._lock:
            await self._refresh_token_request()
            return self._access_token

    async def _refresh_token_request(self) -> None:
        """Perform the OAuth refresh_token grant against the token endpoint.

        Sends a form-encoded request (``grant_type=refresh_token``) and updates
        the in-memory access/refresh tokens and expiry. The account id is
        re-backfilled from a returned id_token when present.

        Raises:
            ValueError: If refresh token is missing or response has no token.
            httpx.HTTPError: On network/HTTP failure.
        """
        if not self._refresh_token:
            raise ValueError("Codex refresh token is not set")

        logger.info(
            f"Refreshing Codex token (account={self.chatgpt_account_id or '<unknown>'}, "
            f"token={_mask_token(self._access_token)})..."
        )

        data = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "refresh_token": self._refresh_token,
            "scope": self._scope,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self._token_url, data=data, headers=headers)
            if response.status_code != 200:
                # Do not log the body verbatim (may include sensitive material);
                # log status + short error code only.
                error_code = "unknown"
                try:
                    error_code = response.json().get("error", "unknown")
                except (ValueError, json.JSONDecodeError):
                    pass
                logger.error(
                    f"Codex token refresh failed: status={response.status_code}, error={error_code}"
                )
                response.raise_for_status()
            result = response.json()

        new_access_token = result.get("access_token")
        new_refresh_token = result.get("refresh_token")
        new_id_token = result.get("id_token")
        expires_in = result.get("expires_in", 3600)

        if not new_access_token:
            raise ValueError("Codex token response does not contain access_token")

        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        if new_id_token:
            self._id_token = new_id_token
            # Refresh may return a new account id claim; keep it in sync.
            backfilled = extract_chatgpt_account_id(new_id_token)
            if backfilled:
                self.chatgpt_account_id = backfilled

        # Apply a 60s safety buffer, mirroring KiroAuthManager.
        self._expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        logger.info(
            f"Codex token refreshed (account={self.chatgpt_account_id or '<unknown>'}), "
            f"expires: {self._expires_at.isoformat()}"
        )

    @property
    def access_token(self) -> str:
        """Current in-memory access token (may be stale; prefer get_access_token)."""
        return self._access_token
