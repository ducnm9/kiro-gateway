# -*- coding: utf-8 -*-

"""Command Code upstream backend.

Clean-room implementation of the Command Code (https://commandcode.ai) wire
format, based solely on observable API behavior. No source code was copied
from third-party tools; only the interop strings below (endpoint paths and
header values) are required for wire compatibility.

Command Code is streaming-only and its model IDs are provider-qualified
(e.g. ``deepseek/deepseek-v4-pro``).
"""

import asyncio
import datetime
import json
from typing import Any, Dict, List

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro import config

# Interop constants (required for wire compatibility).
_CHAT_PATH: str = "/alpha/generate"
_MODELS_PATH: str = "/provider/v1/models"


class CommandCodeBackend:
    """Backend for the Command Code upstream API.

    Holds configuration and provides header building and model listing.
    """

    name: str = "command_code"

    def __init__(self) -> None:
        """Initialize the backend from module configuration."""
        self.api_key: str = config.COMMAND_CODE_API_KEY
        self.base_url: str = config.COMMAND_CODE_BASE_URL
        self.version: str = config.COMMAND_CODE_VERSION
        self.environment: str = config.COMMAND_CODE_ENVIRONMENT
        self.models: List[Dict[str, Any]] = []

    def build_headers(self) -> Dict[str, str]:
        """Return the upstream request headers required by Command Code.

        Returns:
            Header name/value mapping for the chat completion endpoint.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-command-code-version": self.version,
            "x-cli-environment": self.environment,
        }

    async def list_models(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Fetch the available models from Command Code.

        The model-list endpoint only requires the Authorization header.

        Args:
            client: Shared HTTP client.

        Returns:
            List of model dicts with keys ``id``, ``name``, ``context_length``.

        Raises:
            httpx.HTTPError: On network or upstream failure.
        """
        url = f"{self.base_url}{_MODELS_PATH}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", [])

    async def refresh_models_periodically(
        self, client: httpx.AsyncClient, interval: float
    ) -> None:
        """Periodically refresh the model list until the task is cancelled.

        A transient refresh failure is logged and ignored so the loop keeps
        retrying on the next interval.

        Args:
            client: Shared HTTP client.
            interval: Seconds between refreshes.

        Returns:
            Never returns; runs until the background task is cancelled.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                self.models = await self.list_models(client)
                logger.info(
                    f"Command Code model list refreshed: {len(self.models)} models"
                )
            except Exception as e:
                logger.warning(f"Command Code model list refresh failed: {e}")


def extract_cc_error_message(body: str) -> str:
    """Extract a human-readable message from a Command Code error body.

    Command Code error bodies nest the message under ``error`` or place it
    at the top level under ``message``.

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
        if isinstance(data.get("error"), dict) and data["error"].get("message"):
            return data["error"]["message"]
        if data.get("message"):
            return data["message"]
    return body[:200]


async def raise_cc_http_error(response: httpx.Response) -> None:
    """Read a Command Code error body and raise an HTTPException.

    Args:
        response: The upstream non-200 response.

    Raises:
        HTTPException: Always, with a mapped status code and user message.
    """
    body = (await response.aread()).decode("utf-8", errors="replace")
    await response.aclose()
    message = extract_cc_error_message(body)
    if response.status_code == 429:
        hint = _rate_limit_hint(response, body)
        raise HTTPException(
            status_code=429,
            detail=f"Command Code rate limit exceeded{hint}. Please retry later.",
        )
    if response.status_code == 401:
        detail = (
            "Command Code key rejected (401). Re-run the Command Code "
            "OAuth flow and update COMMAND_CODE_API_KEY."
        )
    else:
        detail = f"Command Code error ({response.status_code}): {message}"
    raise HTTPException(status_code=response.status_code, detail=detail)


def _rate_limit_hint(response: httpx.Response, body: str) -> str:
    """Build a human-readable rate-limit reset hint for a 429 response.

    Args:
        response: The upstream 429 response.
        body: Raw response body text (already read).

    Returns:
        A hint string like ", retry after 12s" or ", resets at <time>", or
        an empty string if no reset info is available.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            seconds = int(retry_after)
            return f", retry after {seconds}s"
        except (ValueError, TypeError):
            return f", retry after {retry_after}"

    # Try rateLimit.reset (unix seconds) in the body
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        rate_limit = data.get("rateLimit") or {}
        error_obj = data.get("error")
        if isinstance(error_obj, dict):
            rate_limit = rate_limit or error_obj.get("rateLimit") or {}
        reset = rate_limit.get("reset") if isinstance(rate_limit, dict) else None
        if reset:
            try:
                reset_time = datetime.datetime.fromtimestamp(int(reset), tz=datetime.timezone.utc)
                return f", resets at {reset_time.isoformat()}"
            except (ValueError, TypeError, OSError):
                return f", resets at {reset}"
    return ""
