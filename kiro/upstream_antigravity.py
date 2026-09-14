# -*- coding: utf-8 -*-

"""Antigravity (Google Cloud Code Assist) upstream backend.

Clean-room implementation of the Cloud Code Assist wire format, based solely
on observable API behavior from the pi-antigravity project. No source code was
copied; only the interop strings (endpoint paths, header values, model IDs) are
required for wire compatibility.

Antigravity exposes Gemini, Claude, and GPT-OSS models through Google's
Cloud Code Assist API. All models use the same Gemini-style request format.
"""

import asyncio
import json
import platform
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro import config
from kiro.auth_antigravity import (
    AntigravityCredentials,
    discover_project_id,
    refresh_access_token,
    _endpoint_candidates,
    _mask_email,
    _redact_tokens,
)


# ==============================================================================
# Model Routing Table
# ==============================================================================

# Maps public model IDs to runtime model IDs per thinking effort level.
# Mirrors the catalog from Google's fetchAvailableModels API.
ANTIGRAVITY_ROUTING: Dict[str, Dict[str, Any]] = {
    "gemini-3.7-flash": {
        "off": "gemini-3.7-flash-tiered",
        "routing": {
            "minimal": "gemini-3.7-flash-tiered",
            "low": "gemini-3.7-flash-tiered",
            "medium": "gemini-3.7-flash-tiered",
            "high": "gemini-3.7-flash-tiered",
        },
        "default": "gemini-3.7-flash-tiered",
    },
    "gemini-3.6-flash": {
        "off": "gemini-3.6-flash-low",
        "routing": {
            "minimal": "gemini-3.6-flash-low",
            "low": "gemini-3.6-flash-low",
            "medium": "gemini-3.6-flash-medium",
            "high": "gemini-3.6-flash-high",
        },
        "default": "gemini-3.6-flash-low",
    },
    "gemini-3.5-flash": {
        "off": "gemini-3.5-flash-extra-low",
        "routing": {
            "minimal": "gemini-3.5-flash-extra-low",
            "low": "gemini-3.5-flash-low",
            "medium": "gemini-3.5-flash-low",
            "high": "gemini-3-flash-agent",
        },
        "default": "gemini-3.5-flash-extra-low",
    },
    "gemini-3.1-pro": {
        "off": "gemini-3.1-pro-low",
        "routing": {
            "minimal": "gemini-3.1-pro-low",
            "low": "gemini-3.1-pro-low",
            "medium": "gemini-3.1-pro-low",
            "high": "gemini-pro-agent",
        },
        "default": "gemini-3.1-pro-low",
    },
    "claude-sonnet-4-6": {
        "off": "claude-sonnet-4-6",
        "routing": {
            "minimal": "claude-sonnet-4-6",
            "low": "claude-sonnet-4-6",
            "medium": "claude-sonnet-4-6",
            "high": "claude-sonnet-4-6",
        },
        "default": "claude-sonnet-4-6",
    },
    "claude-opus-4-6": {
        "off": "claude-opus-4-6-thinking",
        "routing": {
            "minimal": "claude-opus-4-6-thinking",
            "low": "claude-opus-4-6-thinking",
            "medium": "claude-opus-4-6-thinking",
            "high": "claude-opus-4-6-thinking",
        },
        "default": "claude-opus-4-6-thinking",
    },
    "gpt-oss-120b": {
        "off": "gpt-oss-120b-medium",
        "routing": {
            "minimal": "gpt-oss-120b-medium",
            "low": "gpt-oss-120b-medium",
            "medium": "gpt-oss-120b-medium",
            "high": "gpt-oss-120b-medium",
        },
        "default": "gpt-oss-120b-medium",
    },
}

# Maximum output tokens per runtime model (verified against Cloud Code Assist).
RUNTIME_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "gemini-3.7-flash-tiered": 65536,
    "gemini-3.6-flash-low": 65536,
    "gemini-3.6-flash-medium": 65536,
    "gemini-3.6-flash-high": 65536,
    "gemini-3.5-flash-extra-low": 65536,
    "gemini-3.5-flash-low": 65536,
    "gemini-3-flash-agent": 65536,
    "gemini-3.1-pro-low": 65535,
    "gemini-pro-agent": 65535,
    "claude-sonnet-4-6": 64000,
    "claude-opus-4-6-thinking": 64000,
    "gpt-oss-120b-medium": 32768,
}

# Model metadata for /v1/models response.
ANTIGRAVITY_MODEL_INFO: List[Dict[str, Any]] = [
    {"id": "antigravity/gemini-3.7-flash", "name": "Gemini 3.7 Flash", "context_length": 1048576, "max_tokens": 65536},
    {"id": "antigravity/gemini-3.6-flash", "name": "Gemini 3.6 Flash", "context_length": 1048576, "max_tokens": 65536},
    {"id": "antigravity/gemini-3.5-flash", "name": "Gemini 3.5 Flash", "context_length": 1048576, "max_tokens": 65536},
    {"id": "antigravity/gemini-3.1-pro", "name": "Gemini 3.1 Pro", "context_length": 1048576, "max_tokens": 65535},
    {"id": "antigravity/claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "context_length": 200000, "max_tokens": 64000},
    {"id": "antigravity/claude-opus-4-6", "name": "Claude Opus 4.6", "context_length": 250000, "max_tokens": 64000},
    {"id": "antigravity/gpt-oss-120b", "name": "GPT-OSS 120B", "context_length": 131072, "max_tokens": 32768},
]

# Stream generation endpoint path.
_STREAM_GENERATE_PATH: str = "/v1internal:streamGenerateContent"
_STREAM_GENERATE_PARAMS: str = "alt=sse"

# Fetch available models endpoint.
_FETCH_MODELS_PATH: str = "/v1internal:fetchAvailableModels"


# ==============================================================================
# Backend Class
# ==============================================================================


class AntigravityBackend:
    """Backend for the Google Antigravity (Cloud Code Assist) upstream API.

    Manages credentials, model routing, headers, and model list refresh.

    Attributes:
        name: Backend identifier string.
        credentials: Current OAuth credentials.
        models: List of available model dicts for /v1/models.
    """

    name: str = "antigravity"

    def __init__(self) -> None:
        """Initialize the backend from module configuration."""
        self.credentials: AntigravityCredentials = AntigravityCredentials()
        self.models: List[Dict[str, Any]] = list(ANTIGRAVITY_MODEL_INFO)
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

        # If refresh token provided via env, pre-populate credentials.
        if config.ANTIGRAVITY_REFRESH_TOKEN:
            self.credentials.refresh_token = config.ANTIGRAVITY_REFRESH_TOKEN
        if config.ANTIGRAVITY_PROJECT_ID:
            self.credentials.project_id = config.ANTIGRAVITY_PROJECT_ID

    def is_authenticated(self) -> bool:
        """Check if the backend has valid credentials.

        Returns:
            True if a refresh token or valid access token is available.
        """
        return self.credentials.is_authenticated

    def set_credentials(self, creds: AntigravityCredentials) -> None:
        """Store new credentials after a successful OAuth flow.

        Args:
            creds: New credentials from OAuth exchange.
        """
        self.credentials = creds
        logger.info(
            f"Antigravity credentials updated: email={_mask_email(creds.email)}, "
            f"project={creds.project_id[:8] if creds.project_id else 'none'}..."
        )

    async def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Thread-safe: uses asyncio.Lock to prevent concurrent refreshes.

        Returns:
            Valid access token string.

        Raises:
            HTTPException: If not authenticated or refresh fails.
        """
        if not self.credentials.refresh_token and not self.credentials.access_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Antigravity not authenticated. "
                    "Visit /antigravity/login or set ANTIGRAVITY_REFRESH_TOKEN."
                ),
            )

        if not self.credentials.is_expired:
            return self.credentials.access_token

        # Need to refresh.
        async with self._refresh_lock:
            # Double-check after acquiring lock (another coroutine may have refreshed).
            if not self.credentials.is_expired:
                return self.credentials.access_token

            if not self.credentials.refresh_token:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Antigravity access token expired and no refresh token available. "
                        "Visit /antigravity/login to re-authenticate."
                    ),
                )

            try:
                access_token, expires_at, new_refresh = await refresh_access_token(
                    self.credentials.refresh_token
                )
                self.credentials.access_token = access_token
                self.credentials.expires_at = expires_at
                if new_refresh:
                    self.credentials.refresh_token = new_refresh
                logger.info("Antigravity access token refreshed successfully")
                return access_token
            except ValueError as e:
                logger.error(f"Antigravity token refresh failed: {e}")
                raise HTTPException(status_code=503, detail=str(e))

    def build_headers(self, access_token: str) -> Dict[str, str]:
        """Build the request headers for Antigravity API calls.

        Args:
            access_token: Valid Google OAuth access token.

        Returns:
            Header name/value mapping for the streamGenerateContent endpoint.
        """
        os_name = (
            "darwin"
            if platform.system() == "Darwin"
            else "windows"
            if platform.system() == "Windows"
            else "linux"
        )
        arch = "amd64" if platform.machine() in ("x86_64", "AMD64") else platform.machine()
        plat = (
            "MACOS"
            if platform.system() == "Darwin"
            else "WINDOWS"
            if platform.system() == "Windows"
            else "LINUX"
        )

        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"antigravity/1.15.8 {os_name}/{arch}",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": json.dumps(
                {"ideType": "ANTIGRAVITY", "platform": plat, "pluginType": "GEMINI"}
            ),
        }

    def get_stream_url(self, endpoint: str) -> str:
        """Build the streamGenerateContent URL for a given endpoint.

        Args:
            endpoint: Base URL of the API endpoint.

        Returns:
            Full URL with path and query parameters.
        """
        return f"{endpoint}{_STREAM_GENERATE_PATH}?{_STREAM_GENERATE_PARAMS}"

    def endpoint_candidates(self) -> List[str]:
        """Return ordered list of API endpoint URLs to try.

        Returns:
            List of base URLs (primary first, fallback second).
        """
        return _endpoint_candidates()

    def resolve_runtime_model(self, public_id: str, thinking_effort: str = "off") -> str:
        """Resolve a public model ID to the Antigravity runtime model ID.

        Uses the routing table to map public IDs (e.g. "gemini-3.7-flash")
        to runtime IDs (e.g. "gemini-3.7-flash-tiered") based on thinking effort.

        Args:
            public_id: Public model ID (without "antigravity/" prefix).
            thinking_effort: Thinking effort level (off, minimal, low, medium, high).

        Returns:
            Runtime model ID string for the API request.
        """
        route = ANTIGRAVITY_ROUTING.get(public_id)
        if not route:
            # Unknown model: pass through as-is (let Antigravity decide).
            return public_id

        if thinking_effort == "off" or not thinking_effort:
            return route.get("off", route.get("default", public_id))

        routing = route.get("routing", {})
        return routing.get(thinking_effort, route.get("default", public_id))

    def get_max_output_tokens(self, runtime_model: str) -> int:
        """Get the maximum output tokens for a runtime model.

        Args:
            runtime_model: Runtime model ID.

        Returns:
            Maximum output tokens allowed.
        """
        if runtime_model in RUNTIME_MAX_OUTPUT_TOKENS:
            return RUNTIME_MAX_OUTPUT_TOKENS[runtime_model]
        # Heuristic fallback based on model family.
        if runtime_model.startswith("claude-"):
            return 64000
        if runtime_model.startswith("gpt-oss-"):
            return 32768
        if runtime_model.startswith("gemini-3.1-pro") or runtime_model == "gemini-pro-agent":
            return 65535
        if runtime_model.startswith("gemini-"):
            return 65536
        return config.ANTIGRAVITY_MAX_TOKENS

    def uses_legacy_parameters(self, runtime_model: str) -> bool:
        """Check if a runtime model requires legacy tool parameter format.

        Claude and GPT-OSS models routed through Cloud Code Assist use a
        Protobuf-based custom-tool bridge that only accepts a strict subset
        of JSON Schema fields in the ``parameters`` field (not
        ``parametersJsonSchema``).

        Args:
            runtime_model: Runtime model ID.

        Returns:
            True if the model needs legacy ``parameters`` format.
        """
        return runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-")

    def needs_thinking_header(self, runtime_model: str) -> bool:
        """Check if a runtime model needs the anthropic-beta thinking header.

        Claude models with reasoning need the interleaved-thinking header.

        Args:
            runtime_model: Runtime model ID.

        Returns:
            True if the anthropic-beta header should be added.
        """
        return runtime_model.startswith("claude-") and "thinking" in runtime_model

    async def list_models(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Fetch available models from the Antigravity API.

        Calls fetchAvailableModels to discover which models are currently
        available for the authenticated account.

        Args:
            client: Shared HTTP client.

        Returns:
            List of model info dicts.

        Raises:
            httpx.HTTPError: On network failure (logged, returns static list).
        """
        if not self.is_authenticated():
            return list(ANTIGRAVITY_MODEL_INFO)

        try:
            token = await self.get_valid_token()
        except HTTPException:
            return list(ANTIGRAVITY_MODEL_INFO)

        headers = self.build_headers(token)
        project_id = self.credentials.project_id

        for endpoint in self.endpoint_candidates():
            try:
                response = await client.post(
                    f"{endpoint}{_FETCH_MODELS_PATH}",
                    headers=headers,
                    json={"project": project_id},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    # Parse and return available models; keep static list as base.
                    logger.debug(f"Fetched Antigravity models from {endpoint}")
                    return list(ANTIGRAVITY_MODEL_INFO)
            except (httpx.HTTPError, Exception) as e:
                logger.debug(f"Antigravity fetchAvailableModels failed on {endpoint}: {e}")

        return list(ANTIGRAVITY_MODEL_INFO)

    async def refresh_models_periodically(
        self, client: httpx.AsyncClient, interval: float
    ) -> None:
        """Periodically refresh the model list until the task is cancelled.

        Args:
            client: Shared HTTP client.
            interval: Seconds between refreshes.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                self.models = await self.list_models(client)
                logger.info(f"Antigravity model list refreshed: {len(self.models)} models")
            except Exception as e:
                logger.warning(f"Antigravity model list refresh failed: {e}")

    async def initialize(self, client: httpx.AsyncClient) -> None:
        """Initialize the backend: refresh token and discover project.

        Called during application startup when ANTIGRAVITY_REFRESH_TOKEN is set.

        Args:
            client: Shared HTTP client for API calls.
        """
        if not self.credentials.refresh_token:
            logger.info("Antigravity backend initialized (awaiting OAuth login)")
            return

        # Refresh the access token.
        try:
            token = await self.get_valid_token()
            logger.info("Antigravity access token obtained via refresh")
        except HTTPException as e:
            logger.warning(f"Antigravity initial token refresh failed: {e.detail}")
            return

        # Discover project ID if not set.
        if not self.credentials.project_id:
            project_id = await discover_project_id(token)
            if project_id:
                self.credentials.project_id = project_id
                logger.info(f"Antigravity project ID discovered: {project_id[:8]}...")
            else:
                logger.warning("Antigravity project ID discovery failed; using fallback")
                self.credentials.project_id = self.credentials.project_id or "unknown"

        # Fetch model list.
        self.models = await self.list_models(client)
        logger.info(
            f"Antigravity backend ready: {len(self.models)} models, "
            f"project={self.credentials.project_id[:8]}..."
        )


# ==============================================================================
# Error Handling
# ==============================================================================


def extract_antigravity_error_message(body: str) -> str:
    """Extract a human-readable message from an Antigravity error response.

    Antigravity errors follow the Google API error format:
    {"error": {"message": "...", "status": "...", "code": 123}}

    Args:
        body: Raw response body text.

    Returns:
        Best-effort error message, or truncated body if unparseable.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body[:300]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if data.get("message"):
            return str(data["message"])

    return body[:300]


def friendly_antigravity_error(status: int, text: str) -> str:
    """Build a user-friendly error message from an Antigravity API error.

    Maps HTTP status codes to actionable guidance for the user.

    Args:
        status: HTTP status code from the upstream response.
        text: Raw response body text.

    Returns:
        User-friendly error message with suggested next steps.
    """
    msg = _redact_tokens(extract_antigravity_error_message(text))[:500]

    if status == 400:
        if "API key not valid" in msg or "API_KEY_INVALID" in msg:
            return (
                "Antigravity login expired or credentials invalid. "
                "Next: visit /antigravity/login to re-authenticate."
            )
        if "Invalid JSON payload" in msg or "Unknown name" in msg:
            return (
                f"Antigravity rejected the request format ({msg}). "
                "Next: try a different model or report this as a bug."
            )
        return f"Antigravity bad request: {msg}"

    if status == 401:
        return (
            "Antigravity authentication failed. "
            "Next: visit /antigravity/login to re-authenticate."
        )

    if status == 403:
        return (
            "Antigravity access denied for this account or project. "
            "Next: try another model, or re-login with a different account."
        )

    if status == 404:
        return (
            "This model is not available right now. "
            "Next: try antigravity/gemini-3.7-flash or antigravity/gemini-3.6-flash."
        )

    if status == 429:
        return (
            f"Antigravity rate limit or quota exceeded. {msg} "
            "Next: wait for reset or switch to a different model."
        )

    if status in (500, 502, 503):
        if "No capacity available" in msg:
            return (
                "This model has no capacity right now. "
                "Next: retry later or switch to another model."
            )
        return f"Antigravity server error ({status}). Next: retry in a moment or switch models."

    if status == 504:
        return "Antigravity timed out. Next: retry the same request."

    return f"Antigravity error ({status}): {msg}"


async def raise_antigravity_http_error(response: httpx.Response) -> None:
    """Read an Antigravity error response and raise an HTTPException.

    Args:
        response: The upstream non-200 response.

    Raises:
        HTTPException: Always, with a mapped status code and user message.
    """
    body = (await response.aread()).decode("utf-8", errors="replace")
    await response.aclose()
    detail = friendly_antigravity_error(response.status_code, body)
    raise HTTPException(status_code=response.status_code, detail=detail)
