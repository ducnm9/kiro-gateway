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

"""ChatGPT (Codex) upstream backend.

Clean-room integration with ChatGPT's Codex Responses API, based on observable
API behavior. Only the interop strings required for wire compatibility
(endpoint URL and identity header values) are configured here.

The Codex backend is streaming-only and uses the OpenAI Responses API format.
Unlike the Command Code backend, Codex requires per-account OAuth tokens and a
per-account ``ChatGPT-Account-ID`` header, so header building takes the
account's auth manager and metadata rather than a single static key.
"""

import asyncio
import datetime
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro import config


# Message returned by Codex when the selected model has no capacity.
CODEX_MODEL_CAPACITY_MESSAGE: str = "Selected model is at capacity. Please try a different model."


class CodexBackend:
    """Backend for the ChatGPT (Codex) upstream API.

    Holds configuration and builds per-account request headers/URLs. The model
    list starts from a static fallback (``config.CHATGPT_MODELS``) and, when
    discovery is enabled, is replaced at runtime with the account's live model
    set fetched from the Codex model-list API (``fetch_models``). This lets new
    models and plan upgrades (Plus/Pro) appear automatically.
    """

    name: str = "chatgpt"

    def __init__(self) -> None:
        """Initialize the backend from module configuration."""
        self.base_url: str = config.CHATGPT_BASE_URL
        self.originator: str = config.CHATGPT_ORIGINATOR
        self.user_agent: str = config.CHATGPT_USER_AGENT
        self.client_version: str = config.CHATGPT_CLIENT_VERSION
        self.models_url: str = config.CHATGPT_MODELS_URL
        # Start from the static fallback; discovery may replace this.
        self.models: List[Dict[str, Any]] = self._static_models()

    @staticmethod
    def _static_models() -> List[Dict[str, Any]]:
        """Return the static fallback model list from config."""
        return [
            {"id": m["id"], "name": m.get("name", m["id"])} for m in config.CHATGPT_MODELS
        ]

    def model_ids(self) -> set:
        """Return the set of currently-known Codex model ids (for routing)."""
        return {m["id"] for m in self.models}

    def build_url(self) -> str:
        """Return the Codex Responses API endpoint URL.

        Returns:
            The configured Codex backend URL.
        """
        return self.base_url

    def build_headers(
        self,
        access_token: str,
        chatgpt_account_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build upstream request headers for a Codex request.

        The ``ChatGPT-Account-ID`` header binds the request to a specific
        ChatGPT account and is required when multiple accounts are configured,
        otherwise requests may cross-bind and surface as ``token_invalid``.

        Args:
            access_token: A valid Codex OAuth access token.
            chatgpt_account_id: The ChatGPT account id for the active account.
            session_id: Optional stable session id for prompt caching.

        Returns:
            Header name/value mapping for the Codex Responses endpoint.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": self.originator,
            "User-Agent": self.user_agent,
            "session_id": session_id or chatgpt_account_id or "default",
        }
        if chatgpt_account_id:
            headers["ChatGPT-Account-ID"] = chatgpt_account_id
        return headers

    def _parse_models_response(self, data: Any) -> List[Dict[str, Any]]:
        """Parse the Codex model-list response into [{id, name}] entries.

        Keeps only user-facing chat models: ``visibility == "list"`` and slug not
        in ``config.CHATGPT_MODEL_EXCLUDE`` (internal slugs like ``gpt-reserve``
        and ``codex-auto-review``).

        Args:
            data: The decoded JSON body from the model-list endpoint.

        Returns:
            A list of {"id", "name"} model dicts (possibly empty).
        """
        models: List[Dict[str, Any]] = []
        raw = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return models
        for m in raw:
            if not isinstance(m, dict):
                continue
            slug = m.get("slug")
            if not slug or not isinstance(slug, str):
                continue
            if slug in config.CHATGPT_MODEL_EXCLUDE:
                continue
            # Only surface models the plan lists for interactive use.
            if m.get("visibility") not in (None, "list"):
                continue
            name = m.get("display_name") or slug
            models.append({"id": slug, "name": name})
        return models

    async def fetch_models(self, auth_manager) -> List[Dict[str, Any]]:
        """Fetch the account's available Codex models from the model-list API.

        Authenticates with the given account's token and the ChatGPT-Account-ID
        header, requests ``?client_version=...`` (>= 0.145.0 to see newer
        models), parses and filters the response, and updates ``self.models``.

        On any failure the current model list is left unchanged (fail-safe) and
        the error is logged; the caller keeps whatever list is already set
        (static fallback or a previous successful fetch).

        Args:
            auth_manager: A CodexAuthManager for an account (provides token +
                chatgpt_account_id).

        Returns:
            The updated list of {"id", "name"} models.
        """
        try:
            token = await auth_manager.get_access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent,
                "originator": self.originator,
            }
            account_id = getattr(auth_manager, "chatgpt_account_id", None)
            if account_id:
                headers["ChatGPT-Account-ID"] = account_id
            params = {"client_version": self.client_version}

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self.models_url, headers=headers, params=params)
                if response.status_code != 200:
                    logger.warning(
                        f"Codex model discovery failed: HTTP {response.status_code}; "
                        f"keeping current model list ({len(self.models)})"
                    )
                    return self.models
                data = response.json()

            parsed = self._parse_models_response(data)
            if parsed:
                self.models = parsed
                logger.info(
                    f"Codex model discovery: {len(parsed)} model(s) "
                    f"({', '.join(m['id'] for m in parsed)})"
                )
            else:
                logger.warning(
                    "Codex model discovery returned no usable models; "
                    f"keeping current list ({len(self.models)})"
                )
            return self.models
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning(f"Codex model discovery error ({type(e).__name__}); keeping current list")
            return self.models

    async def refresh_models_periodically(self, auth_manager, interval: float) -> None:
        """Periodically refresh the Codex model list until cancelled.

        A transient failure is logged and ignored so the loop keeps retrying on
        the next interval.

        Args:
            auth_manager: A CodexAuthManager used to authenticate the fetch.
            interval: Seconds between refreshes.

        Returns:
            Never returns; runs until the background task is cancelled.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                await self.fetch_models(auth_manager)
            except Exception as e:  # noqa: BLE001 - loop must survive any error
                logger.warning(f"Codex periodic model refresh failed: {type(e).__name__}")


def extract_codex_error_message(body: str) -> str:
    """Extract a human-readable message from a Codex error body.

    Codex error bodies typically nest the message under ``error.message``.

    Args:
        body: Raw response body text.

    Returns:
        Best-effort error message, or the raw body (truncated) if unparseable.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return err["message"]
        if data.get("message"):
            return data["message"]
    return body[:200]


def parse_codex_resets_at_ms(body: str, status_code: int) -> Optional[int]:
    """Parse a precise reset timestamp (ms) from a Codex 429 error body.

    Codex ``usage_limit_reached`` errors carry either an absolute ``resets_at``
    (unix seconds) or a relative ``resets_in_seconds``. Returns the absolute
    reset time in epoch milliseconds when present and in the future.

    Args:
        body: Raw response body text.
        status_code: HTTP status code (only 429 carries usage-limit resets).

    Returns:
        Reset time in epoch milliseconds, or None if not available/applicable.
    """
    if status_code != 429 or not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict) or err.get("type") != "usage_limit_reached":
        return None

    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    resets_at = err.get("resets_at")
    if isinstance(resets_at, (int, float)) and resets_at > 0:
        ms = int(resets_at * 1000)
        if ms > now_ms:
            return ms
    resets_in = err.get("resets_in_seconds")
    if isinstance(resets_in, (int, float)) and resets_in > 0:
        return now_ms + int(resets_in * 1000)
    return None


async def raise_codex_http_error(response) -> None:
    """Read a Codex error body and raise an HTTPException.

    Args:
        response: The upstream non-200 httpx response.

    Raises:
        HTTPException: Always, with a mapped status code and user message.
    """
    body = (await response.aread()).decode("utf-8", errors="replace")
    await response.aclose()
    message = extract_codex_error_message(body)
    status = response.status_code

    if status == 429:
        reset_ms = parse_codex_resets_at_ms(body, status)
        hint = ""
        if reset_ms:
            reset_dt = datetime.datetime.fromtimestamp(reset_ms / 1000, tz=datetime.timezone.utc)
            hint = f", resets at {reset_dt.isoformat()}"
        raise HTTPException(
            status_code=429,
            detail=f"ChatGPT (Codex) usage limit reached{hint}. Please retry later or use another account.",
        )
    if status in (401, 403):
        detail = (
            "ChatGPT (Codex) token rejected. The account may be signed out or lack "
            "Codex access. Re-authenticate and update the Codex credentials file."
        )
        raise HTTPException(status_code=status, detail=detail)

    raise HTTPException(status_code=status, detail=f"ChatGPT (Codex) error ({status}): {message}")
