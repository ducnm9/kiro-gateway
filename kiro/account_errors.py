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

"""
Account error classification for failover logic.

Classifies Kiro API errors into two categories:
- FATAL: Error in the request itself → return to client immediately
- RECOVERABLE: Error with the account → try next account

This enables intelligent failover that doesn't waste time retrying
requests that will fail on all accounts.
"""

from enum import Enum
from typing import Optional


class ErrorType(Enum):
    """
    Type of error for failover decision.
    
    FATAL: Error in the request itself (bad payload, context overflow, etc.)
           Should be returned to client immediately without trying other accounts.
    
    RECOVERABLE: Error with the account (expired token, rate limit, quota exceeded)
                Should try next available account.
    """
    FATAL = "fatal"
    RECOVERABLE = "recoverable"


def classify_error(status_code: int, reason: Optional[str]) -> ErrorType:
    """
    Classify Kiro API error for failover decision.
    
    Determines whether an error is account-specific (RECOVERABLE) or
    request-specific (FATAL) based on HTTP status code and error reason.
    
    RECOVERABLE errors (try next account):
    - 400 + INVALID_MODEL_ID: Could be invalid model or insufficient subscription
    - 402: Payment required (monthly quota exceeded, billing issues)
    - 403: Token expired/invalid
    - 429: Rate limit exceeded
    
    FATAL errors (return to client immediately):
    - 400 + CONTENT_LENGTH_EXCEEDS_THRESHOLD: Context overflow
    - 400 + other/null reason: Malformed request
    - 422: Validation error
    - 5xx: Kiro API server error
    
    Args:
        status_code: HTTP status code from Kiro API
        reason: Error reason from Kiro API response (may be None)
    
    Returns:
        ErrorType.RECOVERABLE if should try next account
        ErrorType.FATAL if should return error to client
    
    Examples:
        >>> classify_error(400, "INVALID_MODEL_ID")
        ErrorType.RECOVERABLE
        >>> classify_error(402, "MONTHLY_REQUEST_COUNT")
        ErrorType.RECOVERABLE
        >>> classify_error(403, None)
        ErrorType.RECOVERABLE
        >>> classify_error(429, None)
        ErrorType.RECOVERABLE
        >>> classify_error(400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        ErrorType.FATAL
        >>> classify_error(400, None)
        ErrorType.FATAL
        >>> classify_error(422, None)
        ErrorType.FATAL
        >>> classify_error(500, None)
        ErrorType.FATAL
    """
    # RECOVERABLE: Payment required (quota/billing issues)
    # Kiro API returns 402 for MONTHLY_REQUEST_COUNT
    if status_code == 402:
        return ErrorType.RECOVERABLE
    
    # RECOVERABLE: Token expired/invalid
    if status_code == 403:
        return ErrorType.RECOVERABLE
    
    # RECOVERABLE: Rate limit exceeded
    if status_code == 429:
        return ErrorType.RECOVERABLE
    
    # 400 errors - depends on reason
    if status_code == 400:
        
        # RECOVERABLE: Model not available on this account (subscription level)
        # Different accounts may have different model access based on their subscription
        if reason == "INVALID_MODEL_ID":
            return ErrorType.RECOVERABLE
        
        # FATAL: Context overflow - will fail on all accounts
        if reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
            return ErrorType.FATAL
        
        # FATAL: Generic bad request (malformed payload, validation error)
        # This includes "Improperly formed request" with null/missing reason
        return ErrorType.FATAL
    
    # FATAL: Validation error (malformed request)
    if status_code == 422:
        return ErrorType.FATAL
    
    # FATAL: Server errors (5xx)
    # Note: 503 could be temporary, but we classify as FATAL for simplicity
    # Retrying on different accounts won't help if Kiro API is down
    if 500 <= status_code < 600:
        return ErrorType.FATAL
    
    # Default: treat unknown errors as FATAL to avoid wasting retries
    return ErrorType.FATAL


# Codex (ChatGPT) error reasons/messages that indicate a transient, account-level
# condition and should trigger failover to another account rather than a hard fail.
_CODEX_RECOVERABLE_SUBSTRINGS = (
    "usage_limit_reached",
    "model_at_capacity",
    "selected model is at capacity",
    "server_is_overloaded",
    "service_unavailable_error",
    "rate_limit",
)


def classify_error_codex(status_code: int, reason: Optional[str]) -> ErrorType:
    """
    Classify a ChatGPT (Codex) upstream error for failover decision.

    Codex differs from Kiro in two important ways:
    - Transient server errors (5xx) and capacity/overload conditions are
      RECOVERABLE for Codex (multiple accounts can absorb the load), whereas
      the Kiro classifier treats 5xx as FATAL.
    - Quota exhaustion surfaces as 429 ``usage_limit_reached`` and capacity as
      ``model_at_capacity`` — both RECOVERABLE so failover kicks in.

    RECOVERABLE (try next account):
    - 429 (rate/usage limit), 401/403 (after refresh failed), 5xx.
    - Any error whose reason/message matches a Codex transient substring
      (usage_limit_reached, model_at_capacity, server_is_overloaded, ...).

    FATAL (return to client immediately):
    - 400 / 422 (malformed or invalid request — will fail on every account).

    Args:
        status_code: HTTP status code from the Codex upstream.
        reason: Error reason/message text (may be None).

    Returns:
        ErrorType.RECOVERABLE to try the next account, else ErrorType.FATAL.

    Examples:
        >>> classify_error_codex(429, "usage_limit_reached")
        <ErrorType.RECOVERABLE: 'recoverable'>
        >>> classify_error_codex(200, "Selected model is at capacity")
        <ErrorType.RECOVERABLE: 'recoverable'>
        >>> classify_error_codex(400, "invalid_request")
        <ErrorType.FATAL: 'fatal'>
        >>> classify_error_codex(503, None)
        <ErrorType.RECOVERABLE: 'recoverable'>
    """
    lowered = (reason or "").lower()

    # Transient text signals take priority (may accompany any status, including
    # a 200-OK body carrying a capacity error).
    if any(sub in lowered for sub in _CODEX_RECOVERABLE_SUBSTRINGS):
        return ErrorType.RECOVERABLE

    # Malformed/invalid request — fails on every account.
    if status_code in (400, 422):
        return ErrorType.FATAL

    # Auth rejection (after a refresh attempt), rate limit, and transient server
    # errors are all account-level → try the next account.
    if status_code in (401, 403, 429):
        return ErrorType.RECOVERABLE
    if 500 <= status_code < 600:
        return ErrorType.RECOVERABLE

    # Default: treat unknown Codex errors as RECOVERABLE so a second account
    # gets a chance rather than failing the whole request.
    return ErrorType.RECOVERABLE
