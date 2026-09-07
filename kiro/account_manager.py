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
Unified Account System for Kiro Gateway.

Manages multiple Kiro accounts with intelligent failover, sticky behavior,
and circuit breaker pattern for reliability.

Key features:
- Lazy initialization (only first working account at startup)
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- Probabilistic retry for "dead" accounts
- TTL-based model cache refresh (only when using account)
- Atomic state persistence
"""

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger

from kiro.auth import KiroAuthManager, AuthType
from kiro.auth_base import UpstreamAuthManager
from kiro.auth_codex import CodexAuthManager
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver, normalize_model_name
from kiro.config import (
    HIDDEN_MODELS,
    MODEL_ALIASES,
    HIDDEN_FROM_LIST,
    ACCOUNT_RECOVERY_TIMEOUT,
    ACCOUNT_MAX_BACKOFF_MULTIPLIER,
    ACCOUNT_PROBABILISTIC_RETRY_CHANCE,
    ACCOUNT_CACHE_TTL,
    STATE_SAVE_INTERVAL_SECONDS,
    FALLBACK_MODELS,
    CHATGPT_MODELS,
    CHATGPT_STRATEGY,
    CHATGPT_STICKY_LIMIT,
)

# Default provider assigned to credential entries that omit a "provider" field.
# Kept as "kiro" for full backward compatibility with existing credentials.json.
DEFAULT_PROVIDER: str = "kiro"
from kiro.utils import get_kiro_headers
from kiro.account_errors import ErrorType
from kiro.http_client import KiroHttpClient


def _is_runtime_endpoint(auth_manager: KiroAuthManager) -> bool:
    """
    Check if auth manager uses runtime endpoint that doesn't provide /ListAvailableModels.
    
    Runtime endpoint pattern: https://runtime.{region}.kiro.dev
    Old endpoint pattern: https://q.{region}.amazonaws.com
    
    Runtime endpoint does not provide /ListAvailableModels API (AWS limitation).
    
    Args:
        auth_manager: KiroAuthManager instance
    
    Returns:
        True if using runtime endpoint, False otherwise
    
    Examples:
        >>> auth_manager.api_host = "https://runtime.us-east-1.kiro.dev"
        >>> _is_runtime_endpoint(auth_manager)
        True
        >>> auth_manager.api_host = "https://runtime.eu-central-1.kiro.dev"
        >>> _is_runtime_endpoint(auth_manager)
        True
        >>> auth_manager.api_host = "https://q.us-east-1.amazonaws.com"
        >>> _is_runtime_endpoint(auth_manager)
        False
    """
    return "://runtime." in auth_manager.api_host


def _format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "30s", "5m", "2h", "1d")
    
    Examples:
        >>> _format_duration(30)
        '30s'
        >>> _format_duration(300)
        '5m'
        >>> _format_duration(7200)
        '2h'
        >>> _format_duration(86400)
        '1d'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    else:
        return f"{int(seconds / 86400)}d"


@dataclass
class AccountStats:
    """
    Statistics for account usage.
    
    Tracks request counts for monitoring and future web UI.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


@dataclass
class Account:
    """
    Complete account entity with all dependencies.
    
    Represents a single upstream account (Kiro or ChatGPT/Codex) with its
    authentication, model cache, resolver, and runtime state.
    
    Attributes:
        id: Unique identifier (path to credentials file, or a derived id)
        provider: Upstream provider this account belongs to ("kiro"/"chatgpt").
        auth_manager: Authentication manager (lazy initialized). Any object
            satisfying the UpstreamAuthManager protocol.
        model_cache: Model metadata cache (lazy initialized)
        model_resolver: Model resolver (lazy initialized)
        failures: Consecutive failure count (for Circuit Breaker)
        last_failure_time: Timestamp of last failure
        models_cached_at: Timestamp of last model cache update
        consecutive_use_count: Consecutive successful uses on this account
            since it became the sticky account (drives round-robin rotation).
        account_meta: Provider-specific metadata (e.g. Codex chatgptAccountId).
        stats: Usage statistics
    """
    id: str
    provider: str = DEFAULT_PROVIDER
    auth_manager: Optional[UpstreamAuthManager] = None
    model_cache: Optional[ModelInfoCache] = None
    model_resolver: Optional[ModelResolver] = None
    failures: int = 0
    last_failure_time: float = 0.0
    models_cached_at: float = 0.0
    consecutive_use_count: int = 0
    account_meta: Dict = field(default_factory=dict)
    stats: AccountStats = field(default_factory=AccountStats)


@dataclass
class ModelAccountList:
    """
    List of accounts for a specific model.
    
    Attributes:
        accounts: List of account IDs that have this model
    
    Note: next_index removed - now using global _current_account_index
    """
    accounts: List[str] = field(default_factory=list)


class AccountManager:
    """
    Manages multiple Kiro accounts with intelligent failover.
    
    Responsibilities:
    - Load credentials from credentials.json
    - Lazy initialization of accounts
    - Select next available account (Circuit Breaker + Sticky)
    - Track statistics and failures
    - Persist state to state.json
    
    Example:
        >>> manager = AccountManager("credentials.json", "state.json")
        >>> await manager.load_credentials()
        >>> await manager.load_state()
        >>> account = await manager.get_next_account("claude-opus-4.5")
        >>> await manager.report_success(account.id, "claude-opus-4.5")
    """
    
    def __init__(self, credentials_file: str, state_file: str):
        """
        Initialize AccountManager.
        
        Args:
            credentials_file: Path to credentials.json
            state_file: Path to state.json
        """
        self._credentials_file = credentials_file
        self._state_file = state_file
        self._accounts: Dict[str, Account] = {}
        self._model_to_accounts: Dict[str, ModelAccountList] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._credentials_config: List[Dict] = []
        # Per-provider GLOBAL sticky index. Each provider (kiro, chatgpt, ...)
        # tracks its own current-account position so failover/rotation for one
        # provider never disturbs another. Defaults to 0 for any provider.
        self._current_index_by_provider: Dict[str, int] = {}

    @property
    def _current_account_index(self) -> int:
        """Backward-compatible accessor for the kiro provider's sticky index.

        Preserves the pre-multi-provider attribute name used by existing tests
        and any legacy callers. Reads/writes the kiro slot of the per-provider
        index map.
        """
        return self._current_index_by_provider.get(DEFAULT_PROVIDER, 0)

    @_current_account_index.setter
    def _current_account_index(self, value: int) -> None:
        self._current_index_by_provider[DEFAULT_PROVIDER] = value

    async def load_credentials(self) -> None:
        """
        Load credentials from credentials.json.
        
        Validates each entry and creates Account objects.
        Invalid entries are skipped with warnings.
        Folders are scanned for credential files.
        """
        creds_path = Path(self._credentials_file).expanduser()
        
        if not creds_path.exists():
            logger.warning(f"Credentials file not found: {self._credentials_file}")
            return
        
        try:
            with open(creds_path, 'r', encoding='utf-8') as f:
                self._credentials_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return
        
        # Process each credential entry
        for entry in self._credentials_config:
            cred_type = entry.get("type")
            path = entry.get("path")
            enabled = entry.get("enabled", True)
            
            if not enabled:
                continue
            
            # Validate required fields based on type
            if not cred_type:
                logger.warning(f"Invalid credential entry (missing type): {entry}")
                continue
            
            # For json/sqlite types, path is required
            if cred_type in ("json", "sqlite") and not path:
                logger.warning(f"Invalid credential entry (type={cred_type} requires path): {entry}")
                continue
            
            # For refresh_token type, refresh_token field is required
            if cred_type == "refresh_token" and not entry.get("refresh_token"):
                logger.warning(f"Invalid credential entry (type=refresh_token requires refresh_token field): {entry}")
                continue
            
            # Handle refresh_token type (no path processing needed)
            if cred_type == "refresh_token":
                # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                token = entry.get('refresh_token', '')
                token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                account_id = f"refresh_token_{token_hash}"
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
                continue  # Skip path processing for refresh_token
            
            # Handle folder scanning for json/sqlite types
            expanded_path = Path(path).expanduser()
            if expanded_path.is_dir():
                logger.info(f"Scanning folder for credentials: {path}")
                for file_path in expanded_path.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    # Validate file before adding as account
                    account_id = str(file_path.resolve())
                    is_valid = False
                    
                    # Try JSON validation
                    if cred_type == "json":
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Valid if has refreshToken or clientId
                                if 'refreshToken' in data or 'clientId' in data:
                                    is_valid = True
                        except Exception as e:
                            logger.warning(f"Invalid JSON credentials file {file_path.name}: {e}")
                    
                    # Try SQLite validation
                    elif cred_type == "sqlite":
                        try:
                            import sqlite3
                            conn = sqlite3.connect(str(file_path))
                            cursor = conn.cursor()
                            # Check if auth_kv table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_kv'")
                            if cursor.fetchone():
                                is_valid = True
                            conn.close()
                        except Exception as e:
                            logger.warning(f"Invalid SQLite database file {file_path.name}: {e}")
                    
                    if is_valid:
                        self._accounts[account_id] = Account(id=account_id)
                        logger.debug(f"Added account from folder: {account_id}")
                    else:
                        logger.warning(f"Skipping invalid credentials file: {file_path.name}")
            elif expanded_path.is_file() or cred_type == "refresh_token":
                # Single file or refresh_token type
                if cred_type == "refresh_token":
                    # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                    token = entry.get('refresh_token', '')
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    account_id = f"refresh_token_{token_hash}"
                else:
                    account_id = str(expanded_path.resolve())
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
            else:
                logger.warning(f"Credential path not found: {path}")

        # Load ChatGPT (Codex) accounts from their dedicated credentials file
        # (only when the feature is enabled). Kiro behavior is unaffected.
        self._load_codex_credentials()

        logger.info(f"Loaded {len(self._accounts)} account(s) from credentials")

    def _load_codex_credentials(self) -> None:
        """Load ChatGPT (Codex) accounts from the Codex credentials file.

        Reads ``CHATGPT_CREDENTIALS_FILE`` (a JSON list of account entries) when
        ``CHATGPT_ENABLED`` is true. Each valid entry becomes an ``Account`` with
        ``provider="chatgpt"``. Its OAuth tokens and ChatGPT account id are kept
        in ``account_meta`` and consumed later by ``_initialize_account``.

        Invalid or disabled entries are skipped with a warning. Tokens are never
        logged. No-op (and no error) when the feature is disabled or the file is
        absent, preserving native Kiro-only behavior.
        """
        # Import inside the method so config toggles are re-read on reload and so
        # test suites that reload config see current values.
        from kiro.config import CHATGPT_ENABLED, CHATGPT_CREDENTIALS_FILE

        if not CHATGPT_ENABLED:
            return

        creds_path = Path(CHATGPT_CREDENTIALS_FILE).expanduser()
        if not creds_path.exists():
            logger.warning(
                f"CHATGPT_ENABLED is true but Codex credentials file not found: "
                f"{CHATGPT_CREDENTIALS_FILE}"
            )
            return

        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load Codex credentials: {e}")
            return

        if not isinstance(entries, list):
            logger.error("Codex credentials file must contain a JSON list of accounts")
            return

        loaded = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-object Codex credential entry at index {index}")
                continue
            if not entry.get("enabled", True):
                continue

            access_token = entry.get("accessToken")
            refresh_token = entry.get("refreshToken")
            if not access_token or not refresh_token:
                logger.warning(
                    f"Skipping Codex credential entry at index {index}: "
                    f"requires both accessToken and refreshToken"
                )
                continue

            # Derive a deterministic, non-sensitive account id. Prefer the
            # ChatGPT account id; otherwise hash the refresh token.
            chatgpt_account_id = entry.get("chatgptAccountId")
            if chatgpt_account_id:
                account_id = f"chatgpt_{chatgpt_account_id}"
            else:
                token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()[:16]
                account_id = f"chatgpt_{token_hash}"

            if account_id in self._accounts:
                logger.warning(f"Duplicate Codex account id, skipping: {account_id}")
                continue

            self._accounts[account_id] = Account(
                id=account_id,
                provider="chatgpt",
                account_meta={
                    "accessToken": access_token,
                    "refreshToken": refresh_token,
                    "idToken": entry.get("idToken"),
                    "chatgptAccountId": chatgpt_account_id,
                    "expiresAt": entry.get("expiresAt"),
                },
            )
            loaded += 1
            logger.debug(f"Added Codex account: {account_id}")

        logger.info(f"Loaded {loaded} ChatGPT (Codex) account(s)")

    async def load_state(self) -> None:
        """
        Load runtime state from state.json.
        
        Restores model_to_accounts mapping and account runtime state.
        Creates empty state if file doesn't exist.
        """
        state_path = Path(self._state_file)
        
        if not state_path.exists():
            logger.debug("State file not found, starting with empty state")
            return
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            # Restore per-provider sticky indices. Prefer the new map; fall back
            # to the legacy single "current_account_index" (assigned to kiro) for
            # state files written by older gateway versions.
            index_map = state_data.get("current_index_by_provider")
            if isinstance(index_map, dict) and index_map:
                self._current_index_by_provider = {
                    str(k): int(v) for k, v in index_map.items()
                }
            else:
                legacy_index = state_data.get("current_account_index", 0)
                self._current_index_by_provider = {DEFAULT_PROVIDER: int(legacy_index)}
            
            # Restore model_to_accounts mapping (without next_index)
            for model, data in state_data.get("model_to_accounts", {}).items():
                self._model_to_accounts[model] = ModelAccountList(
                    accounts=data.get("accounts", [])
                )
            
            # Restore account runtime state
            for account_id, data in state_data.get("accounts", {}).items():
                if account_id in self._accounts:
                    account = self._accounts[account_id]
                    account.failures = data.get("failures", 0)
                    account.last_failure_time = data.get("last_failure_time", 0.0)
                    account.models_cached_at = data.get("models_cached_at", 0.0)
                    account.consecutive_use_count = data.get("consecutive_use_count", 0)
                    
                    stats_data = data.get("stats", {})
                    account.stats = AccountStats(
                        total_requests=stats_data.get("total_requests", 0),
                        successful_requests=stats_data.get("successful_requests", 0),
                        failed_requests=stats_data.get("failed_requests", 0)
                    )
            
            logger.info(f"Loaded state: {len(self._model_to_accounts)} model mappings, {len(self._accounts)} accounts")
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self) -> None:
        """
        Save runtime state to state.json atomically.
        
        Uses tmp file + rename for atomic write.
        """
        state_data = {
            # Kept for backward compatibility with older gateway versions that
            # read a single global index (maps to the kiro provider slot).
            "current_account_index": self._current_index_by_provider.get(DEFAULT_PROVIDER, 0),
            # Per-provider sticky indices (new format).
            "current_index_by_provider": dict(self._current_index_by_provider),
            "accounts": {
                account_id: {
                    "provider": account.provider,
                    "failures": account.failures,
                    "last_failure_time": account.last_failure_time,
                    "models_cached_at": account.models_cached_at,
                    "consecutive_use_count": account.consecutive_use_count,
                    "stats": {
                        "total_requests": account.stats.total_requests,
                        "successful_requests": account.stats.successful_requests,
                        "failed_requests": account.stats.failed_requests
                    }
                }
                for account_id, account in self._accounts.items()
            },
            "model_to_accounts": {
                model: {
                    "accounts": mal.accounts
                }
                for model, mal in self._model_to_accounts.items()
            }
        }
        
        state_path = Path(self._state_file)
        tmp_path = state_path.with_suffix('.json.tmp')
        
        try:
            # Write with restrictive permissions (owner-only read/write)
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            tmp_path.replace(state_path)
            logger.debug("State saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    async def save_state_periodically(self) -> None:
        """
        Background task for periodic state saving.
        
        Saves state every STATE_SAVE_INTERVAL_SECONDS if dirty flag is set.
        The dirty check is performed inside the lock to prevent race conditions
        where a concurrent write sets dirty=True between our check and reset.
        """
        while True:
            await asyncio.sleep(STATE_SAVE_INTERVAL_SECONDS)
            
            async with self._lock:
                if self._dirty:
                    await self._save_state()
                    self._dirty = False
    
    async def _initialize_codex_account(self, account_id: str) -> bool:
        """Initialize a ChatGPT (Codex) account (lazy initialization).

        Builds a ``CodexAuthManager`` from the account's stored OAuth tokens,
        validates them by fetching a valid access token (refreshing if needed),
        and populates the model cache/resolver from the static Codex registry
        (``CHATGPT_MODELS``) — Codex has no /ListAvailableModels endpoint.

        Args:
            account_id: The Codex account id to initialize.

        Returns:
            True on success, False otherwise.
        """
        account = self._accounts.get(account_id)
        if not account:
            return False

        try:
            meta = account.account_meta or {}
            auth_manager = CodexAuthManager(
                access_token=meta.get("accessToken", ""),
                refresh_token=meta.get("refreshToken", ""),
                id_token=meta.get("idToken"),
                chatgpt_account_id=meta.get("chatgptAccountId"),
                expires_at=meta.get("expiresAt"),
            )

            # Verify credentials work (refreshes if expiring). Raises on failure.
            await auth_manager.get_access_token()

            # Keep the (possibly backfilled) ChatGPT account id in sync so the
            # request layer can set the ChatGPT-Account-ID header correctly.
            account.account_meta["chatgptAccountId"] = auth_manager.chatgpt_account_id

            # Build model cache/resolver from the static Codex registry.
            models_list = [{"modelId": m["id"]} for m in CHATGPT_MODELS]
            model_cache = ModelInfoCache()
            await model_cache.update(models_list)

            model_resolver = ModelResolver(
                cache=model_cache,
                hidden_models={},
                aliases={},
                hidden_from_list=[],
            )

            account.auth_manager = auth_manager
            account.model_cache = model_cache
            account.model_resolver = model_resolver
            account.models_cached_at = time.time()

            # Register models in the model→accounts map for dynamic learning.
            for model in model_resolver.get_available_models():
                if model not in self._model_to_accounts:
                    self._model_to_accounts[model] = ModelAccountList()
                if account_id not in self._model_to_accounts[model].accounts:
                    self._model_to_accounts[model].accounts.append(account_id)

            logger.info(
                f"Initialized Codex account: {account_id} "
                f"({len(model_resolver.get_available_models())} models)"
            )
            self._dirty = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Codex account {account_id}: {e}")
            return False

    async def _initialize_account(self, account_id: str) -> bool:
        """
        Initialize account (lazy initialization).
        
        Creates auth_manager, fetches models, creates cache and resolver.
        
        Args:
            account_id: Account ID to initialize
        
        Returns:
            True if successful, False otherwise
        """
        account = self._accounts.get(account_id)
        if not account:
            return False

        # ChatGPT (Codex) accounts use a dedicated init path: a CodexAuthManager
        # plus the static Codex model registry (no /ListAvailableModels call).
        if account.provider == "chatgpt":
            return await self._initialize_codex_account(account_id)

        try:
            # Find credentials config for this account
            creds_config = None
            for entry in self._credentials_config:
                path = entry.get("path", "")
                expanded_path = Path(path).expanduser()
                
                if entry.get("type") == "refresh_token":
                    # Match by deterministic hash for refresh_token type
                    token = entry.get('refresh_token', '')
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    if account_id == f"refresh_token_{token_hash}":
                        creds_config = entry
                        break
                elif str(expanded_path.resolve()) == account_id or (expanded_path.is_dir() and account_id.startswith(str(expanded_path.resolve()) + os.sep)):
                    creds_config = entry
                    break
            
            if not creds_config:
                logger.error(f"No credentials config found for account: {account_id}")
                return False
            
            # Create KiroAuthManager based on type
            cred_type = creds_config.get("type")
            if cred_type == "json":
                auth_manager = KiroAuthManager(
                    creds_file=account_id,
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            elif cred_type == "sqlite":
                auth_manager = KiroAuthManager(
                    sqlite_db=account_id,
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            elif cred_type == "refresh_token":
                auth_manager = KiroAuthManager(
                    refresh_token=creds_config.get("refresh_token"),
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            else:
                logger.error(f"Unknown credential type: {cred_type}")
                return False
            
            # Get token to verify credentials
            token = await auth_manager.get_access_token()
            
            # Determine if we should fetch models or use static list
            if _is_runtime_endpoint(auth_manager):
                # New runtime endpoint does not provide /ListAvailableModels (AWS limitation)
                # Use static list without attempting request
                logger.debug(f"Account {account_id}: Using static model list for runtime.kiro.dev endpoint")
                models_list = FALLBACK_MODELS
            else:
                # Old endpoint - attempt to fetch dynamic model list
                # Fetch models list with retry + fallback
                params = {"origin": "AI_EDITOR"}
                if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                    params["profileArn"] = auth_manager.profile_arn
                
                list_models_url = f"{auth_manager.q_host}/ListAvailableModels"
                
                # Use KiroHttpClient for retry logic (3 attempts with exponential backoff)
                http_client = KiroHttpClient(auth_manager, shared_client=None)
                
                try:
                    response = await http_client.request_with_retry(
                        method="GET",
                        url=list_models_url,
                        json_data=None,
                        params=params,
                        stream=False
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        models_list = data.get("models", [])
                    else:
                        # Shouldn't happen (retry handles non-200), but keep for safety
                        raise Exception(f"HTTP {response.status_code}")
                
                except Exception as e:
                    # All retries exhausted - use fallback
                    logger.error(f"Failed to fetch models for {account_id} after retries: {e}")
                    logger.warning("Using pre-configured fallback models. Models will be refreshed on next TTL cycle when network recovers.")
                    models_list = FALLBACK_MODELS
                
                finally:
                    await http_client.close()
            
            # Create model cache and update
            model_cache = ModelInfoCache()
            await model_cache.update(models_list)
            
            # Add hidden models
            for display_name, internal_id in HIDDEN_MODELS.items():
                model_cache.add_hidden_model(display_name, internal_id)
            
            # Create model resolver
            model_resolver = ModelResolver(
                cache=model_cache,
                hidden_models=HIDDEN_MODELS,
                aliases=MODEL_ALIASES,
                hidden_from_list=HIDDEN_FROM_LIST
            )
            
            # Update account
            account.auth_manager = auth_manager
            account.model_cache = model_cache
            account.model_resolver = model_resolver
            account.models_cached_at = time.time()
            
            # Update model_to_accounts mapping
            available_models = model_resolver.get_available_models()
            for model in available_models:
                if model not in self._model_to_accounts:
                    self._model_to_accounts[model] = ModelAccountList()
                if account_id not in self._model_to_accounts[model].accounts:
                    self._model_to_accounts[model].accounts.append(account_id)
            
            logger.info(f"Initialized account: {account_id} ({len(available_models)} models)")
            self._dirty = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize account {account_id}: {e}")
            return False
    
    async def _refresh_account_models(self, account_id: str) -> None:
        """
        Refresh model cache for account (TTL refresh).
        
        Args:
            account_id: Account ID to refresh
        """
        account = self._accounts.get(account_id)
        if not account or not account.auth_manager:
            return
        
        # Check if using runtime endpoint (no dynamic model list available)
        if _is_runtime_endpoint(account.auth_manager):
            # Runtime endpoint does not provide /ListAvailableModels
            # Use static list and update cache timestamp
            logger.debug(f"Account {account_id}: Skipping model refresh for runtime.kiro.dev endpoint (using static list)")
            await account.model_cache.update(FALLBACK_MODELS)
            account.models_cached_at = time.time()
            self._dirty = True
            return
        
        # Old endpoint - attempt to fetch dynamic model list
        # Use KiroHttpClient for retry logic
        http_client = KiroHttpClient(account.auth_manager, shared_client=None)
        
        try:
            params = {"origin": "AI_EDITOR"}
            if account.auth_manager.auth_type == AuthType.KIRO_DESKTOP and account.auth_manager.profile_arn:
                params["profileArn"] = account.auth_manager.profile_arn
            
            list_models_url = f"{account.auth_manager.q_host}/ListAvailableModels"
            
            response = await http_client.request_with_retry(
                method="GET",
                url=list_models_url,
                json_data=None,
                params=params,
                stream=False
            )
            
            if response.status_code == 200:
                data = response.json()
                models_list = data.get("models", [])
                await account.model_cache.update(models_list)
                account.models_cached_at = time.time()
                
                # Update model_to_accounts mapping (new models may have appeared)
                available_models = account.model_resolver.get_available_models()
                for model in available_models:
                    if model not in self._model_to_accounts:
                        self._model_to_accounts[model] = ModelAccountList()
                    if account_id not in self._model_to_accounts[model].accounts:
                        self._model_to_accounts[model].accounts.append(account_id)
                
                logger.debug(f"Refreshed models for {account_id}")
                self._dirty = True
        
        except Exception as e:
            # All retries exhausted - keep using stale cache
            logger.warning(f"Failed to refresh models for {account_id} after retries: {e}")
        
        finally:
            await http_client.close()
    
    async def get_next_account(
        self,
        model: str,
        exclude_accounts: Optional[set] = None,
        provider: str = DEFAULT_PROVIDER,
    ) -> Optional[Account]:
        """
        Get next available account for model (Circuit Breaker + Sticky).
        
        Implements:
        - Sticky behavior (prefer successful account)
        - Circuit Breaker with exponential backoff
        - Probabilistic retry for "dead" accounts (10%)
        - TTL-based model cache refresh
        - Exclusion of already-tried accounts in current failover loop
        - Per-provider isolation: only accounts of ``provider`` are considered,
          and each provider keeps its own sticky index.
        - Round-robin rotation (ChatGPT only): when the provider strategy is
          "round-robin", the sticky account is advanced after it has been used
          ``sticky_limit`` consecutive times. "fill-first" never advances
          proactively (rotation happens only on failure via ``exclude_accounts``).
        
        Args:
            model: Model name (will be normalized)
            exclude_accounts: Set of account IDs to exclude (already tried in current failover loop)
            provider: Upstream provider to select an account for ("kiro"/"chatgpt").
        
        Returns:
            Account object or None if no accounts available
        """
        async with self._lock:
            # Accounts belonging to the requested provider, in stable insertion order.
            provider_account_ids = [
                aid for aid, acc in self._accounts.items() if acc.provider == provider
            ]

            if not provider_account_ids:
                return None

            # Special case: single account - bypass Circuit Breaker
            # Circuit Breaker is meaningless for single account - user should see real API errors
            # instead of generic "Account unavailable" after cooldown kicks in
            if len(provider_account_ids) == 1:
                account_id = provider_account_ids[0]
                account = self._accounts[account_id]
                
                # Skip if already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    return None
                
                # Lazy initialization if needed
                if account.auth_manager is None:
                    success = await self._initialize_account(account_id)
                    if not success:
                        return None
                
                # Check TTL and refresh if needed
                if account.models_cached_at > 0:
                    age = time.time() - account.models_cached_at
                    if age > ACCOUNT_CACHE_TTL:
                        try:
                            await self._refresh_account_models(account_id)
                        except Exception as e:
                            logger.warning(f"Failed to refresh models for {account_id}: {e}")
                
                # Always return single account (ignore cooldown/failures)
                # No model validation - let API decide (gateway, not gatekeeper)
                return account
            
            # Multi-account logic: per-provider GLOBAL sticky
            normalized_model = normalize_model_name(model)

            # Round-robin advance (ChatGPT only): if the current sticky account
            # has reached its consecutive-use limit, move the start index forward
            # so a different account is preferred first this round.
            start_index = self._current_index_by_provider.get(provider, 0) % len(provider_account_ids)
            if provider == "chatgpt" and CHATGPT_STRATEGY == "round-robin":
                current_sticky = self._accounts[provider_account_ids[start_index]]
                sticky_limit = max(1, CHATGPT_STICKY_LIMIT)
                if current_sticky.consecutive_use_count >= sticky_limit:
                    current_sticky.consecutive_use_count = 0
                    start_index = (start_index + 1) % len(provider_account_ids)
                    self._current_index_by_provider[provider] = start_index
                    self._dirty = True

            # Iterate over this provider's accounts starting from the sticky index
            for i in range(len(provider_account_ids)):
                current_index = (start_index + i) % len(provider_account_ids)
                account_id = provider_account_ids[current_index]
                account = self._accounts[account_id]
                
                # Skip accounts already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    continue
                
                # Check Circuit Breaker (Half-Open state with exponential backoff)
                if account.failures > 0:
                    time_since_failure = time.time() - account.last_failure_time
                    
                    # Exponential backoff: base * 2^(failures - 1), capped at MAX_MULTIPLIER
                    # 1 failure: 60s, 2: 120s, 3: 240s, ..., 12+: 86400s (1 day cap)
                    backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                    effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                    
                    if time_since_failure < effective_timeout:
                        # Probabilistic retry (10% chance)
                        if random.random() > ACCOUNT_PROBABILISTIC_RETRY_CHANCE:
                            continue
                        else:
                            logger.info(f"Probabilistic retry for broken account {account_id}")
                    else:
                        # Half-Open: recovery timeout passed
                        logger.info(f"Half-Open state for {account_id} (recovery timeout passed, effective={effective_timeout}s)")
                
                # Lazy initialization
                if account.auth_manager is None:
                    success = await self._initialize_account(account_id)
                    if not success:
                        account.failures += 1
                        self._dirty = True
                        continue
                
                # Check TTL and refresh if needed
                if account.models_cached_at > 0:
                    age = time.time() - account.models_cached_at
                    if age > ACCOUNT_CACHE_TTL:
                        try:
                            await self._refresh_account_models(account_id)
                        except Exception as e:
                            logger.warning(f"Failed to refresh models for {account_id}: {e}")
                # # Check if model is available on this account
                # available_models = account.model_resolver.get_available_models()
                # if normalized_model not in available_models:
                #     continue
                
                # No model validation - let Kiro API decide (gateway, not gatekeeper)
                # Account is suitable!
                return account
            
            # All accounts unavailable
            return None
    
    async def report_success(self, account_id: str, model: str) -> None:
        """
        Report successful request (reset failures, update stats, sticky, dynamic learning).
        
        Args:
            account_id: Account ID
            model: Model name
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Reset failures
            if account.failures > 0:
                account.failures = 0
                self._dirty = True
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.successful_requests += 1
            self._dirty = True
            
            # Dynamic learning: add model to mapping if successful
            # This allows system to learn about new models not in FALLBACK_MODELS
            normalized_model = normalize_model_name(model)
            if normalized_model not in self._model_to_accounts:
                self._model_to_accounts[normalized_model] = ModelAccountList()
                logger.debug(f"Dynamic learning: discovered new model '{normalized_model}'")
            if account_id not in self._model_to_accounts[normalized_model].accounts:
                self._model_to_accounts[normalized_model].accounts.append(account_id)
                logger.debug(f"Dynamic learning: model '{normalized_model}' works on account {account_id}")
                self._dirty = True
            
            # PER-PROVIDER STICKY: point this provider's sticky index at the
            # account that just succeeded, indexing within the provider's own
            # account list (get_next_account iterates that same list).
            provider = account.provider
            provider_account_ids = [
                aid for aid, acc in self._accounts.items() if acc.provider == provider
            ]
            try:
                successful_index = provider_account_ids.index(account_id)
                if self._current_index_by_provider.get(provider, 0) != successful_index:
                    self._current_index_by_provider[provider] = successful_index
                    self._dirty = True
            except ValueError:
                pass

            # Round-robin bookkeeping: count consecutive successful uses of the
            # sticky account so get_next_account can rotate after the limit.
            account.consecutive_use_count += 1
            self._dirty = True
    
    async def report_failure(
        self,
        account_id: str,
        model: str,
        error_type: ErrorType,
        status_code: int,
        reason: Optional[str]
    ) -> None:
        """
        Report failed request (update failures, stats, failover).
        
        Args:
            account_id: Account ID
            model: Model name
            error_type: Error classification (FATAL or RECOVERABLE)
            status_code: HTTP status code
            reason: Error reason from Kiro API
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Special case: INVALID_MODEL_ID is discovery process, not account failure
            # Account is healthy, model is just not available on this account
            # Log for user visibility but don't penalize account statistics
            if reason == "INVALID_MODEL_ID":
                account.stats.total_requests += 1
                self._dirty = True
                logger.warning(
                    f"Model '{model}' not available on account {account_id}: "
                    f"status={status_code}, reason={reason}"
                )
                return
            
            # Update failure count (only for RECOVERABLE)
            if error_type == ErrorType.RECOVERABLE:
                account.failures += 1
                account.last_failure_time = time.time()
                # Reset the round-robin usage counter: a failed account should not
                # carry a stale consecutive-use count when it recovers.
                account.consecutive_use_count = 0
                self._dirty = True
                
                # Calculate backoff for logging
                backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                logger.warning(
                    f"Account {account_id} failure #{account.failures}: "
                    f"status={status_code}, reason={reason}, "
                    f"cooldown={_format_duration(effective_timeout)}"
                )
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.failed_requests += 1
            self._dirty = True
            
            # GLOBAL STICKY: Do NOT change _current_account_index on failure
            # It only changes on success (GLOBAL sticky behavior)
            # Failover happens through exclude_accounts in get_next_account()
    
    def get_first_account(self) -> Account:
        """
        Get first initialized account (for legacy mode).
        
        Returns:
            First initialized account
        
        Raises:
            RuntimeError: If no initialized accounts available
        """
        for account in self._accounts.values():
            if account.auth_manager is not None:
                return account
        raise RuntimeError("No initialized accounts available")
    
    def get_all_available_models(self) -> List[str]:
        """
        Collect unique models from all initialized accounts.
        
        Used by /v1/models endpoint in account system to show
        all available models across all accounts.
        
        Returns:
            Sorted list of unique model IDs
        """
        all_models = set()
        for account in self._accounts.values():
            if account.model_resolver:
                all_models.update(account.model_resolver.get_available_models())
        return sorted(all_models)
