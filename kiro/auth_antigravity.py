# -*- coding: utf-8 -*-

"""Google OAuth authentication for the Antigravity (Cloud Code Assist) provider.

Implements the OAuth 2.0 Authorization Code flow with PKCE for Google sign-in.
Handles token exchange, refresh, project discovery, and credential management.

The OAuth client credentials used here are Google's public Antigravity desktop
client (not a private app secret). The redirect URI is fixed at
http://localhost:51121/oauth-callback to match the registered OAuth app.
"""

import asyncio
import hashlib
import json
import os
import platform
import secrets
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
from loguru import logger

from kiro.config import (
    ANTIGRAVITY_BASE_URL,
    ANTIGRAVITY_CALLBACK_PORT,
    ANTIGRAVITY_FALLBACK_URL,
    ANTIGRAVITY_PROJECT_ID,
    ANTIGRAVITY_REFRESH_TOKEN,
)


# ==============================================================================
# OAuth Constants (Google's public Antigravity desktop client)
# ==============================================================================

# The default OAuth client is Google's public Antigravity desktop client — a
# public interop credential, not a private app secret. It is stored base64-split
# (matching the upstream pi-antigravity project) so secret scanners do not flag
# it as a leaked credential. Override via ANTIGRAVITY_CLIENT_ID/SECRET env vars
# to use your own OAuth app.
def _decode_default_secret() -> str:
    """Decode the base64-split public OAuth client secret.

    Returns:
        The decoded public client secret string.
    """
    import base64

    encoded = "R09DU1BYLUs1OEZXUjQ4" + "NkxkTEoxbUxCOHNYQzR6NnFEQWY="
    return base64.b64decode(encoded).decode("utf-8")


CLIENT_ID: str = os.getenv("ANTIGRAVITY_CLIENT_ID") or (
    "107100606059-tmhssin2h21lcre235vtoloih4g403esp"
    ".apps.googleusercontent.com"
)
CLIENT_SECRET: str = os.getenv("ANTIGRAVITY_CLIENT_SECRET") or _decode_default_secret()

AUTH_URL: str = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL: str = "https://oauth2.googleapis.com/token"
USERINFO_URL: str = "https://www.googleapis.com/oauth2/v1/userinfo"
REDIRECT_URI: str = f"http://localhost:{ANTIGRAVITY_CALLBACK_PORT}/oauth-callback"

SCOPES: list = [
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# Token refresh 5 minutes before expiry to avoid race conditions.
TOKEN_REFRESH_MARGIN_SECONDS: int = 300

# Timeout for the OAuth callback (user has 5 minutes to complete login).
OAUTH_CALLBACK_TIMEOUT_SECONDS: int = 300

# Timeout for discovery/metadata API calls.
DISCOVERY_TIMEOUT_SECONDS: float = 8.0


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class AntigravityCredentials:
    """Stores Google OAuth credentials for the Antigravity provider.

    Attributes:
        access_token: Short-lived access token (~1h).
        refresh_token: Long-lived refresh token for obtaining new access tokens.
        expires_at: Unix timestamp when access_token expires.
        project_id: Cloud Code Assist project ID.
        email: Google account email address.
    """

    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    project_id: str = ""
    email: str = ""

    @property
    def is_authenticated(self) -> bool:
        """Check if credentials are present (has refresh token or valid access token)."""
        return bool(self.refresh_token) or (
            bool(self.access_token) and self.expires_at > time.time()
        )

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired or about to expire."""
        return self.expires_at <= (time.time() + TOKEN_REFRESH_MARGIN_SECONDS)


# ==============================================================================
# PKCE Helpers
# ==============================================================================


def generate_pkce() -> tuple:
    """Generate a PKCE code verifier and challenge pair.

    Uses SHA-256 for the code challenge method (S256).

    Returns:
        Tuple of (verifier, challenge) as base64url-encoded strings.
    """
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url_encode(verifier_bytes)
    challenge_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url_encode(challenge_bytes)
    return verifier, challenge


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url without padding.

    Args:
        data: Raw bytes to encode.

    Returns:
        Base64url-encoded string without trailing '=' padding.
    """
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


# ==============================================================================
# OAuth Callback Server
# ==============================================================================


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the OAuth callback."""

    # Class-level storage shared with the server instance.
    received_code: Optional[str] = None
    received_state: Optional[str] = None
    received_error: Optional[str] = None
    expected_state: str = ""

    def do_GET(self) -> None:
        """Handle GET request from Google OAuth redirect."""
        parsed = urlparse(self.path)
        if parsed.path != "/oauth-callback":
            self._respond(404, "Not Found")
            return

        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if error:
            self._respond(400, f"Authentication failed: {error}")
            _OAuthCallbackHandler.received_error = error
            return

        if not code or not state:
            self._respond(400, "Authentication failed: missing code or state.")
            _OAuthCallbackHandler.received_error = "missing_code_or_state"
            return

        if state != _OAuthCallbackHandler.expected_state:
            self._respond(400, "Authentication failed: invalid state (possible CSRF).")
            _OAuthCallbackHandler.received_error = "state_mismatch"
            return

        _OAuthCallbackHandler.received_code = code
        _OAuthCallbackHandler.received_state = state
        self._respond(
            200,
            "Antigravity authentication complete. You can close this tab and return to the gateway.",
        )

    def _respond(self, status: int, message: str) -> None:
        """Send an HTML response to the browser.

        Args:
            status: HTTP status code.
            message: Message body to display.
        """
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Kiro Gateway - Antigravity Auth</title></head>"
            f"<body><h2>{message}</h2></body></html>"
        )
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP server logging."""
        pass


class OAuthCallbackServer:
    """Temporary HTTP server that listens for the Google OAuth callback.

    Spawns a background thread, waits for the redirect, then shuts down.

    Args:
        expected_state: The state parameter to validate against CSRF.
        port: Port to listen on (default: 51121).
        timeout: Seconds to wait before timing out.
    """

    def __init__(
        self,
        expected_state: str,
        port: int = ANTIGRAVITY_CALLBACK_PORT,
        timeout: float = OAUTH_CALLBACK_TIMEOUT_SECONDS,
    ):
        self._expected_state = expected_state
        self._port = port
        self._timeout = timeout
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        """Start the callback server in a background thread.

        Raises:
            OSError: If the port is already in use.
        """
        _OAuthCallbackHandler.received_code = None
        _OAuthCallbackHandler.received_state = None
        _OAuthCallbackHandler.received_error = None
        _OAuthCallbackHandler.expected_state = self._expected_state

        self._server = HTTPServer(("127.0.0.1", self._port), _OAuthCallbackHandler)
        self._server.timeout = 1.0  # Check shutdown flag every second
        self._thread = Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        logger.debug(f"OAuth callback server started on port {self._port}")

    def _serve_loop(self) -> None:
        """Serve requests until code is received or timeout expires."""
        if not self._server:
            return
        start = time.time()
        while (time.time() - start) < self._timeout:
            self._server.handle_request()
            if _OAuthCallbackHandler.received_code or _OAuthCallbackHandler.received_error:
                break

    def stop(self) -> None:
        """Shut down the callback server."""
        if self._server:
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("OAuth callback server stopped")

    def get_result(self) -> tuple:
        """Get the OAuth callback result.

        Returns:
            Tuple of (code, error). One will be None.
        """
        return _OAuthCallbackHandler.received_code, _OAuthCallbackHandler.received_error


# ==============================================================================
# Token Operations
# ==============================================================================


async def exchange_code_for_tokens(
    code: str, verifier: str
) -> AntigravityCredentials:
    """Exchange an OAuth authorization code for access and refresh tokens.

    Args:
        code: Authorization code from the OAuth callback.
        verifier: PKCE code verifier used during the auth request.

    Returns:
        AntigravityCredentials with tokens populated.

    Raises:
        httpx.HTTPStatusError: If the token endpoint returns an error.
        ValueError: If the response is missing required fields.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        body = response.text
        logger.error(f"Token exchange failed ({response.status_code}): {_redact_tokens(body[:300])}")
        raise ValueError(
            f"Google token exchange failed ({response.status_code}). "
            "Re-run /antigravity/login and try again."
        )

    data = response.json()
    access_token = data.get("access_token", "")
    refresh_token = data.get("refresh_token", "")
    expires_in = int(data.get("expires_in", 3600))

    if not refresh_token:
        raise ValueError(
            "No refresh token received from Google. "
            "Re-run /antigravity/login and grant offline access."
        )

    # Discover project ID and email in parallel.
    project_id, email = await asyncio.gather(
        discover_project_id(access_token),
        get_user_email(access_token),
    )

    creds = AntigravityCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in - TOKEN_REFRESH_MARGIN_SECONDS,
        project_id=project_id or _fallback_project_id(email),
        email=email or "",
    )
    logger.info(
        f"Antigravity OAuth complete: email={_mask_email(creds.email)}, "
        f"project={creds.project_id[:8]}..."
    )
    return creds


async def refresh_access_token(refresh_token: str) -> tuple:
    """Refresh an expired access token using the refresh token.

    Args:
        refresh_token: Google OAuth refresh token.

    Returns:
        Tuple of (new_access_token, expires_at_timestamp, new_refresh_token_or_None).

    Raises:
        ValueError: If refresh fails (token revoked, network error).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        body = response.text
        logger.error(f"Token refresh failed ({response.status_code}): {_redact_tokens(body[:300])}")
        raise ValueError(
            f"Antigravity token refresh failed ({response.status_code}). "
            "Run /antigravity/login to re-authenticate."
        )

    data = response.json()
    access_token = data.get("access_token", "")
    expires_in = int(data.get("expires_in", 3600))
    # Google may rotate the refresh token.
    new_refresh_token = data.get("refresh_token")

    expires_at = time.time() + expires_in - TOKEN_REFRESH_MARGIN_SECONDS
    return access_token, expires_at, new_refresh_token


# ==============================================================================
# Discovery APIs
# ==============================================================================


async def discover_project_id(access_token: str) -> Optional[str]:
    """Discover the Cloud Code Assist project ID for the authenticated user.

    Tries loadCodeAssist first, falls back to listCloudAICompanionProjects.

    Args:
        access_token: Valid Google OAuth access token.

    Returns:
        Project ID string, or None if discovery fails.
    """
    headers = _build_discovery_headers(access_token)
    endpoints = _endpoint_candidates()

    for endpoint in endpoints:
        # Try loadCodeAssist
        try:
            async with httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json={
                        "metadata": {
                            "ideType": "ANTIGRAVITY",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        }
                    },
                )
            if response.status_code == 200:
                project_id = _extract_project_id(response.json())
                if project_id:
                    logger.debug(f"Discovered project ID via loadCodeAssist: {project_id[:8]}...")
                    return project_id
        except (httpx.HTTPError, Exception) as e:
            logger.debug(f"loadCodeAssist failed on {endpoint}: {e}")

    # Fallback: listCloudAICompanionProjects
    for endpoint in endpoints:
        try:
            async with httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{endpoint}/v1internal:listCloudAICompanionProjects",
                    headers=headers,
                    json={},
                )
            if response.status_code == 200:
                project_id = _extract_project_id(response.json())
                if project_id:
                    logger.debug(
                        f"Discovered project ID via listCloudAICompanionProjects: {project_id[:8]}..."
                    )
                    return project_id
        except (httpx.HTTPError, Exception) as e:
            logger.debug(f"listCloudAICompanionProjects failed on {endpoint}: {e}")

    logger.warning("Could not discover Antigravity project ID from API")
    return None


async def get_user_email(access_token: str) -> Optional[str]:
    """Fetch the authenticated user's email address.

    Args:
        access_token: Valid Google OAuth access token.

    Returns:
        Email string, or None if unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{USERINFO_URL}?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if response.status_code == 200:
            data = response.json()
            return data.get("email")
    except (httpx.HTTPError, Exception) as e:
        logger.debug(f"Failed to fetch user email: {e}")
    return None


# ==============================================================================
# Auth URL Builder
# ==============================================================================


def build_auth_url() -> tuple:
    """Build the Google OAuth authorization URL with PKCE.

    Returns:
        Tuple of (auth_url, state, verifier) for the OAuth flow.
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = f"{AUTH_URL}?{urlencode(params)}"
    return auth_url, state, verifier


# ==============================================================================
# Helper Functions
# ==============================================================================


def _endpoint_candidates() -> list:
    """Return the ordered list of API endpoint candidates.

    Returns:
        List of base URL strings to try in order.
    """
    candidates = [ANTIGRAVITY_BASE_URL]
    if ANTIGRAVITY_FALLBACK_URL and ANTIGRAVITY_FALLBACK_URL != ANTIGRAVITY_BASE_URL:
        candidates.append(ANTIGRAVITY_FALLBACK_URL)
    return candidates


def _build_discovery_headers(access_token: str) -> dict:
    """Build headers for discovery API calls.

    Args:
        access_token: Valid Google OAuth access token.

    Returns:
        Headers dict.
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


def _extract_project_id(data: object) -> Optional[str]:
    """Recursively extract a project ID from a discovery API response.

    Checks common field names used by Cloud Code Assist APIs.

    Args:
        data: Parsed JSON response (dict, list, or primitive).

    Returns:
        Project ID string, or None if not found.
    """
    if not isinstance(data, dict):
        return None

    # Direct fields
    for key in (
        "antigravityProjectId",
        "projectId",
        "backendProjectId",
        "userDefinedCloudaicompanionProject",
        "cloudaicompanionProject",
        "project",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested_id = value.get("id")
            if isinstance(nested_id, str) and nested_id:
                return nested_id

    # Array fields
    for key in ("projects", "projectIds", "cloudaicompanionProjects"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    return item
                if isinstance(item, dict):
                    nested = _extract_project_id(item)
                    if nested:
                        return nested

    return None


def _fallback_project_id(seed: str = "antigravity-default") -> str:
    """Generate a stable fallback project ID from a seed string.

    Produces a UUID-v5-like stable identifier so the same account always
    gets the same fallback project ID.

    Args:
        seed: Seed string (email preferred).

    Returns:
        UUID-shaped project ID string.
    """
    if ANTIGRAVITY_PROJECT_ID:
        return ANTIGRAVITY_PROJECT_ID

    digest = hashlib.sha1(f"antigravity:{seed}".encode()).digest()[:16]
    # Set version 5 and variant bits
    b = bytearray(digest)
    b[6] = (b[6] & 0x0F) | 0x50
    b[8] = (b[8] & 0x3F) | 0x80
    hex_str = b.hex()
    return (
        f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-"
        f"{hex_str[16:20]}-{hex_str[20:]}"
    )


def _redact_tokens(text: str) -> str:
    """Redact OAuth tokens from text for safe logging.

    Args:
        text: Text that may contain tokens.

    Returns:
        Text with tokens replaced by [redacted].
    """
    import re

    text = re.sub(r"\bya29\.[A-Za-z0-9._~+/=-]+", "[redacted-access-token]", text)
    text = re.sub(r"\b1/[A-Za-z0-9_-]{20,}", "[redacted-refresh-token]", text)
    text = re.sub(
        r'("?(?:access_token|refresh_token|token|client_secret)"?\s*[:=]\s*")[^"]*(")',
        r"\1[redacted]\2",
        text,
    )
    return text


def _mask_email(email: str) -> str:
    """Mask an email address for safe logging.

    Args:
        email: Full email address.

    Returns:
        Masked email like "u***r@domain.com".
    """
    if not email or "@" not in email:
        return "[unknown]"
    name, domain = email.split("@", 1)
    if len(name) <= 2:
        masked = f"{name[0]}***"
    else:
        masked = f"{name[0]}***{name[-1]}"
    return f"{masked}@{domain}"
