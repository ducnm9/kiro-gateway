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
In-process rate limiting middleware for Kiro Gateway.

Implements a sliding window rate limiter per client IP using a token bucket
algorithm. No external dependencies required.

Features:
- Per-IP rate limiting with configurable limits
- Sliding window counter (not fixed window, avoids burst at window boundaries)
- Automatic cleanup of stale entries to prevent memory leaks
- Configurable via environment variables
- Returns standard 429 Too Many Requests with Retry-After header
"""

import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from loguru import logger

from kiro.config import (
    RATE_LIMIT_RPM,
    RATE_LIMIT_ENABLED,
)


@dataclass
class ClientBucket:
    """
    Token bucket for a single client.
    
    Uses a sliding window approach: tracks request timestamps within
    the current window and counts them.
    
    Attributes:
        timestamps: List of request timestamps within the current window
        last_cleanup: Last time stale timestamps were removed
    """
    timestamps: list = field(default_factory=list)
    last_cleanup: float = 0.0


class RateLimiter:
    """
    In-process sliding window rate limiter.
    
    Tracks requests per client IP within a 60-second window.
    Thread-safe via asyncio.Lock for concurrent request handling.
    
    Attributes:
        max_requests: Maximum requests allowed per window
        window_seconds: Window duration in seconds (default: 60)
    
    Example:
        >>> limiter = RateLimiter(max_requests=60, window_seconds=60)
        >>> allowed, retry_after = limiter.check("192.168.1.1")
        >>> if not allowed:
        ...     return 429, {"Retry-After": str(retry_after)}
    """
    
    # Cleanup stale entries every 5 minutes to prevent memory growth
    _CLEANUP_INTERVAL: float = 300.0
    
    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window per client
            window_seconds: Window duration in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, ClientBucket] = {}
        self._lock = asyncio.Lock()
        self._last_global_cleanup: float = 0.0
    
    async def check(self, client_id: str) -> Tuple[bool, float]:
        """
        Check if a request from client_id is allowed.
        
        Args:
            client_id: Client identifier (typically IP address)
        
        Returns:
            Tuple of (allowed: bool, retry_after: float)
            - allowed: True if request is permitted
            - retry_after: Seconds until next allowed request (0 if allowed)
        """
        async with self._lock:
            now = time.time()
            
            # Periodic cleanup of stale entries
            if now - self._last_global_cleanup > self._CLEANUP_INTERVAL:
                self._cleanup_stale_entries(now)
                self._last_global_cleanup = now
            
            # Get or create bucket for this client
            if client_id not in self._buckets:
                self._buckets[client_id] = ClientBucket()
            
            bucket = self._buckets[client_id]
            window_start = now - self.window_seconds
            
            # Remove timestamps outside the current window
            bucket.timestamps = [ts for ts in bucket.timestamps if ts > window_start]
            
            # Check limit
            if len(bucket.timestamps) >= self.max_requests:
                # Calculate when the oldest request in window will expire
                oldest_in_window = bucket.timestamps[0]
                retry_after = oldest_in_window + self.window_seconds - now
                return False, max(retry_after, 1.0)
            
            # Allow request and record timestamp
            bucket.timestamps.append(now)
            return True, 0.0
    
    def _cleanup_stale_entries(self, now: float) -> None:
        """
        Remove client entries that haven't been seen recently.
        
        Removes entries where all timestamps are outside the window,
        preventing unbounded memory growth from many unique IPs.
        
        Args:
            now: Current timestamp
        """
        window_start = now - self.window_seconds
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if not bucket.timestamps or bucket.timestamps[-1] <= window_start
        ]
        for key in stale_keys:
            del self._buckets[key]
        
        if stale_keys:
            logger.debug(f"Rate limiter: cleaned up {len(stale_keys)} stale client entries")


# Global rate limiter instance
_rate_limiter: RateLimiter = RateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60.0)


# Endpoints that should be rate limited (API endpoints only, not health checks)
_RATE_LIMITED_ENDPOINTS = frozenset({
    "/v1/chat/completions",
    "/v1/messages",
})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting API requests.
    
    Only rate-limits API endpoints (chat completions, messages).
    Health checks and model listing are exempt.
    
    Uses client IP from X-Forwarded-For (if behind reverse proxy) or
    direct connection IP as the rate limit key.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Check rate limit before processing the request.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler
        
        Returns:
            Response from next handler, or 429 if rate limited
        """
        # Skip if rate limiting is disabled
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Only rate-limit API endpoints
        if request.url.path not in _RATE_LIMITED_ENDPOINTS:
            return await call_next(request)
        
        # Get client IP (support reverse proxy via X-Forwarded-For)
        client_ip = _get_client_ip(request)
        
        # Check rate limit
        allowed, retry_after = await _rate_limiter.check(client_ip)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path} "
                f"(limit: {RATE_LIMIT_RPM} rpm, retry_after: {retry_after:.1f}s)"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Rate limit exceeded. Maximum {RATE_LIMIT_RPM} requests per minute. "
                                   f"Please retry after {int(retry_after)} seconds.",
                        "type": "rate_limit_error",
                        "code": 429
                    }
                },
                headers={"Retry-After": str(int(retry_after))}
            )
        
        return await call_next(request)


def _get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Checks X-Forwarded-For header first (for reverse proxy setups),
    then falls back to direct connection IP.
    
    Args:
        request: FastAPI/Starlette request
    
    Returns:
        Client IP address string
    """
    # Check X-Forwarded-For (first IP in chain is the original client)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Fall back to direct connection
    if request.client:
        return request.client.host
    
    return "unknown"
