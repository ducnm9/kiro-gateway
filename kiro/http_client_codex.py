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

"""HTTP client for the ChatGPT (Codex) API with retry logic.

Thin analogue of ``KiroHttpClient`` keyed on a ``CodexAuthManager``:
- 401/403: refresh the token once and retry.
- 429/5xx: exponential backoff, then return the last response so the caller can
  classify it (and apply account failover).
- Network errors: classified via ``network_errors`` and surfaced as HTTP 502/504.

Codex is streaming-only, so requests are issued with ``stream=True`` using a
per-request client to avoid CLOSE_WAIT leaks (mirrors the Kiro streaming path).
"""

import asyncio
import json
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import MAX_RETRIES, BASE_RETRY_DELAY, STREAMING_READ_TIMEOUT
from kiro.auth_codex import CodexAuthManager
from kiro.upstream_codex import CodexBackend
from kiro.network_errors import classify_network_error, get_short_error_message, NetworkErrorInfo


class CodexHttpClient:
    """HTTP client for the Codex API with retry + token-refresh logic.

    Attributes:
        auth_manager: The Codex auth manager for the active account.
        backend: The CodexBackend (URL + header builder).
    """

    def __init__(
        self,
        auth_manager: CodexAuthManager,
        backend: CodexBackend,
        shared_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the Codex HTTP client.

        Args:
            auth_manager: The account's Codex auth manager.
            backend: The Codex backend used for URL/header construction.
            shared_client: Optional shared httpx client (non-streaming). If None,
                a per-request client is created for each call and owned here.
        """
        self.auth_manager = auth_manager
        self.backend = backend
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """Return the shared client, or create a per-request client.

        Args:
            stream: Whether the client is used for a streaming request.

        Returns:
            An active httpx.AsyncClient.
        """
        if self._shared_client is not None:
            return self._shared_client
        if self.client is None or self.client.is_closed:
            if stream:
                timeout_config = httpx.Timeout(
                    connect=30.0, read=STREAMING_READ_TIMEOUT, write=30.0, pool=30.0
                )
            else:
                timeout_config = httpx.Timeout(timeout=300.0)
            self.client = httpx.AsyncClient(timeout=timeout_config, follow_redirects=True)
        return self.client

    async def close(self) -> None:
        """Close the client if this instance owns it (no-op for shared clients)."""
        if not self._owns_client:
            return
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing Codex HTTP client: {e}")

    async def request_with_retry(
        self,
        payload: dict,
        session_id: Optional[str] = None,
        stream: bool = True,
    ) -> httpx.Response:
        """Send a Codex request with retry + token-refresh logic.

        On 401/403 the token is refreshed once and the request retried. On
        429/5xx an exponential backoff is applied and the last such response is
        returned so the caller can classify it and trigger account failover.

        Args:
            payload: The Codex Responses API request body.
            session_id: Optional stable session id for prompt caching headers.
            stream: Whether to stream the response (Codex is streaming-only).

        Returns:
            An httpx.Response (200, or the last 429/5xx for caller classification).

        Raises:
            HTTPException: On network failure after all retries (502/504).
        """
        client = await self._get_client(stream=stream)
        url = self.backend.build_url()
        last_error_info: Optional[NetworkErrorInfo] = None
        last_response: Optional[httpx.Response] = None

        for attempt in range(MAX_RETRIES):
            try:
                token = await self.auth_manager.get_access_token()
                headers = self.backend.build_headers(
                    access_token=token,
                    chatgpt_account_id=self.auth_manager.chatgpt_account_id,
                    session_id=session_id,
                )
                request_kwargs = {"headers": headers, "content": json.dumps(payload).encode()}

                if stream:
                    # Prevent CLOSE_WAIT connection leak on streaming responses.
                    headers["Connection"] = "close"
                    req = client.build_request("POST", url, **request_kwargs)
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request("POST", url, **request_kwargs)

                if response.status_code == 200:
                    return response

                # 401/403 - token rejected: refresh once and retry.
                if response.status_code in (401, 403):
                    last_response = response
                    logger.warning(
                        f"Codex received {response.status_code}, refreshing token "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    try:
                        await self.auth_manager.force_refresh()
                    except Exception as e:
                        # Refresh failed → cannot recover this account; return the
                        # rejection so the caller marks it FATAL and fails over.
                        logger.error(f"Codex token refresh failed: {type(e).__name__}")
                        return response
                    continue

                # 429 / 5xx - transient: backoff, keep last response for the caller.
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_response = response
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Codex received {response.status_code}, waiting {delay}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Other statuses - return as-is for caller classification.
                return response

            except (httpx.TimeoutException, httpx.RequestError) as e:
                error_info = classify_network_error(e)
                last_error_info = error_info
                short_msg = get_short_error_message(error_info)
                if error_info.is_retryable and attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Codex {short_msg} - waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Codex {short_msg} - no more retries (attempt {attempt + 1}/{MAX_RETRIES})")
                    if not error_info.is_retryable:
                        break

        # Return the last transient response (429/5xx/auth) for caller classification.
        if last_response is not None:
            return last_response

        # Network failure after all retries.
        if last_error_info:
            raise HTTPException(
                status_code=last_error_info.suggested_http_code,
                detail=last_error_info.user_message,
            )
        raise HTTPException(status_code=502, detail=f"Codex request failed after {MAX_RETRIES} attempts.")

    async def __aenter__(self) -> "CodexHttpClient":
        """Async context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the client when exiting context."""
        await self.close()
