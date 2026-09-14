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

"""Upstream authentication abstraction.

Defines the minimal token-lifecycle contract that the Account System
(``kiro.account_manager``) depends on, so that multiple upstream providers
(Kiro, ChatGPT/Codex, ...) can share the same multi-account selection,
failover, and Circuit Breaker machinery.

The contract is intentionally small: it covers only what account selection
needs (fetch a valid token, force a refresh, and inspect expiry). Provider
specifics (endpoints, refresh payloads, credential persistence) live in the
concrete implementations (``KiroAuthManager``, ``CodexAuthManager``).

``KiroAuthManager`` already satisfies this Protocol structurally; no changes
to its behavior are required. New auth managers SHOULD implement this Protocol
so they can slot into an ``Account`` transparently.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class UpstreamAuthManager(Protocol):
    """Structural contract for a provider auth/token manager.

    Any object exposing these members can be driven by the Account System.
    This is a structural (duck-typed) Protocol: implementations do NOT need to
    inherit from it, they only need matching members.

    Required members:
        provider: Short provider identifier (e.g. ``"kiro"``, ``"chatgpt"``).
    """

    provider: str

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing it if necessary.

        Implementations MUST be safe to call concurrently (guard refresh with
        an ``asyncio.Lock``) and MUST refresh before expiry when needed.

        Returns:
            A valid access token string.

        Raises:
            ValueError: If a valid token cannot be obtained.
        """
        ...

    async def force_refresh(self) -> str:
        """Force a token refresh regardless of current expiry.

        Used when the upstream rejects the current token (e.g. HTTP 401/403).

        Returns:
            The newly refreshed access token.

        Raises:
            Exception: If the refresh request fails.
        """
        ...

    def is_token_expiring_soon(self) -> bool:
        """Report whether the token is expired or within the refresh lead time.

        Returns:
            True if a refresh should be performed before the next request.
        """
        ...

    def is_token_expired(self) -> bool:
        """Report whether the token is already fully expired.

        Returns:
            True if the token can no longer be used.
        """
        ...
