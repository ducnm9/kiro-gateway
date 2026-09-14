# -*- coding: utf-8 -*-

"""Antigravity OAuth login and status routes.

Provides endpoints for initiating the Google OAuth flow and checking
authentication status. The OAuth callback is handled by a temporary
HTTP server on port 51121 (matching Google's registered redirect URI).

Endpoints:
- GET /antigravity/login: Start OAuth flow, return auth URL
- GET /antigravity/status: Check current authentication state
"""

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from kiro.auth_antigravity import (
    AntigravityCredentials,
    OAuthCallbackServer,
    build_auth_url,
    exchange_code_for_tokens,
    ANTIGRAVITY_CALLBACK_PORT,
    _mask_email,
)
from kiro.config import ANTIGRAVITY_ENABLED


router = APIRouter(prefix="/antigravity", tags=["antigravity"])


@router.get("/login")
async def antigravity_login(request: Request):
    """Initiate the Google OAuth login flow for Antigravity.

    Generates a PKCE-protected auth URL and starts a temporary callback
    server on port 51121 to receive the OAuth redirect.

    The user must open the returned ``auth_url`` in their browser, complete
    Google sign-in, and the callback server will handle the token exchange
    automatically.

    Returns:
        JSON with auth_url, instructions, and callback port.

    Raises:
        HTTPException: 503 if Antigravity is not enabled.
        HTTPException: 409 if the callback port is already in use.
    """
    if not ANTIGRAVITY_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Antigravity is not enabled. Set ANTIGRAVITY_ENABLED=true in .env",
        )

    ag_backend = getattr(request.app.state, "antigravity_backend", None)
    if ag_backend is None:
        raise HTTPException(status_code=503, detail="Antigravity backend not initialized")

    # Generate auth URL with PKCE.
    auth_url, state, verifier = build_auth_url()

    # Start the temporary callback server.
    try:
        callback_server = OAuthCallbackServer(expected_state=state)
        callback_server.start()
    except OSError as e:
        raise HTTPException(
            status_code=409,
            detail=(
                f"OAuth callback port {ANTIGRAVITY_CALLBACK_PORT} is already in use. "
                "Close the process using that port and retry."
            ),
        )

    # Run the OAuth exchange in a background task.
    async def _complete_oauth():
        """Wait for the OAuth callback and exchange code for tokens."""
        try:
            # Wait for the callback (runs in background thread).
            # Poll until code is received or timeout.
            max_wait = 300  # 5 minutes
            poll_interval = 0.5
            elapsed = 0.0
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                code, error = callback_server.get_result()
                if code or error:
                    break

            callback_server.stop()
            code, error = callback_server.get_result()

            if error:
                logger.error(f"Antigravity OAuth failed: {error}")
                return
            if not code:
                logger.warning("Antigravity OAuth timed out (no callback received)")
                return

            # Exchange code for tokens.
            creds = await exchange_code_for_tokens(code, verifier)
            ag_backend.set_credentials(creds)
            logger.info("Antigravity OAuth login completed successfully")

        except Exception as e:
            logger.error(f"Antigravity OAuth exchange failed: {e}")
        finally:
            callback_server.stop()

    # Fire and forget the background task.
    asyncio.create_task(_complete_oauth())

    return JSONResponse(content={
        "status": "login_started",
        "auth_url": auth_url,
        "callback_port": ANTIGRAVITY_CALLBACK_PORT,
        "instructions": (
            "Open the auth_url in your browser to sign in with Google. "
            "After approval, the gateway will automatically receive your credentials. "
            "Check /antigravity/status to confirm authentication."
        ),
    })


@router.get("/status")
async def antigravity_status(request: Request):
    """Check Antigravity authentication status.

    Returns the current authentication state including whether credentials
    are valid, email, project ID, and token expiry.

    Returns:
        JSON with authentication status details.
    """
    if not ANTIGRAVITY_ENABLED:
        return JSONResponse(content={
            "enabled": False,
            "authenticated": False,
            "message": "Antigravity is not enabled. Set ANTIGRAVITY_ENABLED=true in .env",
        })

    ag_backend = getattr(request.app.state, "antigravity_backend", None)
    if ag_backend is None:
        return JSONResponse(content={
            "enabled": True,
            "authenticated": False,
            "message": "Antigravity backend not initialized",
        })

    creds = ag_backend.credentials
    is_auth = ag_backend.is_authenticated()
    is_expired = creds.is_expired if is_auth else True

    response = {
        "enabled": True,
        "authenticated": is_auth,
        "token_valid": is_auth and not is_expired,
        "email": _mask_email(creds.email) if creds.email else None,
        "project_id": creds.project_id[:12] + "..." if creds.project_id else None,
        "expires_at": creds.expires_at if creds.expires_at > 0 else None,
        "expires_in_seconds": max(0, int(creds.expires_at - time.time())) if creds.expires_at > 0 else None,
        "models_available": len(ag_backend.models),
    }

    if not is_auth:
        response["message"] = (
            "Not authenticated. Visit /antigravity/login to sign in with Google, "
            "or set ANTIGRAVITY_REFRESH_TOKEN in .env."
        )
    elif is_expired:
        response["message"] = "Token expired. Will auto-refresh on next request."
    else:
        response["message"] = "Authenticated and ready."

    return JSONResponse(content=response)
