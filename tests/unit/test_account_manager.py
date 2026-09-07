# -*- coding: utf-8 -*-

"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- TTL-based model cache refresh
- State persistence
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from kiro.account_manager import (
    Account,
    AccountStats,
    ModelAccountList,
    AccountManager,
    _format_duration
)
from kiro.account_errors import ErrorType
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """
    
    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.
        
        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")
        
        # Act
        account = Account(id="/test/path.json")
        
        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")
        
        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)
    
    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.
        
        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")
        
        # Act
        stats = AccountStats()
        
        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.
        
        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.
        
        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.
        
        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")
    
    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.
        
        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.
        
        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")
        
        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1  # Only valid file loaded
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.
        
        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.
        
        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.
        
        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.
        
        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.
        
        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.
        
        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))
        
        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) > 0
    
    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test restoration of global current_account_index.
        
        What it does: Restores sticky index from state
        Purpose: Verify global sticky behavior persistence
        """
        print("\n=== Test: load_state restores current_account_index ===")
        
        # Arrange
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Current account index: {manager._current_account_index}")
        
        assert manager._current_account_index == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.
        
        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")
        
        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")
        
        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).
        
        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")
        
        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())
        
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")
        
        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
    
    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).
        
        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) == 0
        assert manager._current_account_index == 0
    
    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.
        
        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        
        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_initialize_account_json_success(self, tmp_path, mock_list_models_response):
        """
        Test successful account initialization with type=json.
        
        What it does: Initializes account with JSON credentials
        Purpose: Verify complete initialization flow
        """
        print("\n=== Test: initialize_account with JSON ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client for ListAvailableModels
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True
        assert manager._accounts[account_id].auth_manager is not None
        assert manager._accounts[account_id].model_cache is not None
        assert manager._accounts[account_id].model_resolver is not None
    
    @pytest.mark.asyncio
    async def test_initialize_account_fetch_models_fallback(self, tmp_path):
        """
        Test fallback to FALLBACK_MODELS when API fails.
        
        What it does: Initializes account when ListAvailableModels fails
        Purpose: Verify fallback mechanism
        """
        print("\n=== Test: initialize_account with fallback models ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client to fail
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_client.request_with_retry = AsyncMock(side_effect=Exception("Network error"))
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True  # Should succeed with fallback
        assert manager._accounts[account_id].model_cache is not None


class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.
        
        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()
        
        # Act
        account = await manager.get_next_account("claude-opus-4.5")
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.
        
        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures
        manager._accounts[account_id].failures = 5
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0
    
    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.
        
        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.
        
        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1
    
    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.
        
        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """
    
    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.
        
        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager._save_state()
        
        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()
        
        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.
        
        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        account = manager.get_first_account()
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None
    
    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.
        
        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.
        
        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        models = manager.get_all_available_models()
        
        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """
    
    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"
    
    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"
    
    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"
    
    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"



# =============================================================================
# Multi-provider (ChatGPT/Codex) Account System tests
# =============================================================================

def _make_manager(tmp_path):
    """Create an AccountManager with temp file paths (no credentials loaded)."""
    return AccountManager(
        credentials_file=str(tmp_path / "creds.json"),
        state_file=str(tmp_path / "state.json"),
    )


def _inject_account(manager, account_id, provider, *, models=("gpt-5.5",)):
    """Inject a pre-initialized account so get_next_account skips lazy init.

    The auth_manager/model_resolver are lightweight stubs — no network.
    """
    auth = MagicMock()
    auth.provider = provider
    resolver = MagicMock()
    resolver.get_available_models.return_value = list(models)
    acc = Account(
        id=account_id,
        provider=provider,
        auth_manager=auth,
        model_cache=MagicMock(),
        model_resolver=resolver,
        models_cached_at=time.time(),
    )
    manager._accounts[account_id] = acc
    return acc


class TestProviderFiltering:
    """get_next_account must only ever return accounts of the requested provider."""

    @pytest.mark.asyncio
    async def test_selects_only_requested_provider(self, tmp_path):
        """
        What it does: With mixed kiro+chatgpt accounts, requesting chatgpt returns a chatgpt account.
        Purpose: Provider isolation in selection.
        """
        manager = _make_manager(tmp_path)
        _inject_account(manager, "kiro_a", "kiro", models=("claude-sonnet-4.5",))
        _inject_account(manager, "kiro_b", "kiro", models=("claude-sonnet-4.5",))
        _inject_account(manager, "chatgpt_1", "chatgpt")
        _inject_account(manager, "chatgpt_2", "chatgpt")

        acc = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        assert acc is not None
        assert acc.provider == "chatgpt"

    @pytest.mark.asyncio
    async def test_kiro_selection_unaffected_by_codex_accounts(self, tmp_path):
        """
        What it does: Requesting kiro never returns a chatgpt account.
        Purpose: No cross-provider leakage; Kiro path unchanged.
        """
        manager = _make_manager(tmp_path)
        _inject_account(manager, "kiro_a", "kiro", models=("claude-sonnet-4.5",))
        _inject_account(manager, "chatgpt_1", "chatgpt")
        _inject_account(manager, "chatgpt_2", "chatgpt")

        acc = await manager.get_next_account("claude-sonnet-4.5", provider="kiro")
        assert acc is not None
        assert acc.provider == "kiro"
        assert acc.id == "kiro_a"

    @pytest.mark.asyncio
    async def test_no_accounts_for_provider_returns_none(self, tmp_path):
        """
        What it does: Requesting a provider with no accounts returns None.
        Purpose: Graceful handling when a provider isn't configured.
        """
        manager = _make_manager(tmp_path)
        _inject_account(manager, "kiro_a", "kiro", models=("claude-sonnet-4.5",))

        acc = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        assert acc is None


class TestCodexSingleAccountBypass:
    """A single Codex account bypasses the Circuit Breaker (like Kiro)."""

    @pytest.mark.asyncio
    async def test_single_codex_account_ignores_failures(self, tmp_path):
        """
        What it does: One chatgpt account is returned even with failures set.
        Purpose: Single-account bypass so users see real upstream errors.
        """
        manager = _make_manager(tmp_path)
        acc = _inject_account(manager, "chatgpt_1", "chatgpt")
        acc.failures = 10
        acc.last_failure_time = time.time()
        # Add a kiro account too, to prove the count is per-provider.
        _inject_account(manager, "kiro_a", "kiro", models=("claude-sonnet-4.5",))

        result = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        assert result is acc


class TestCodexFillFirst:
    """Fill-first: stay on one account until it fails, then move to the next."""

    @pytest.mark.asyncio
    async def test_fill_first_sticks_then_fails_over(self, tmp_path, monkeypatch):
        """
        What it does: fill-first returns acc1 repeatedly; excluding it returns acc2.
        Purpose: Drain-one-then-next behavior driven by exclude_accounts.
        """
        monkeypatch.setattr("kiro.account_manager.CHATGPT_STRATEGY", "fill-first")
        manager = _make_manager(tmp_path)
        _inject_account(manager, "chatgpt_1", "chatgpt")
        _inject_account(manager, "chatgpt_2", "chatgpt")

        # Repeated calls stick to the same (first) account.
        first = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        again = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        assert first.id == again.id == "chatgpt_1"

        # Excluding the drained account fails over to the other one.
        nxt = await manager.get_next_account(
            "gpt-5.5", exclude_accounts={"chatgpt_1"}, provider="chatgpt"
        )
        assert nxt.id == "chatgpt_2"


class TestCodexRoundRobin:
    """Round-robin: rotate accounts after CHATGPT_STICKY_LIMIT successes."""

    @pytest.mark.asyncio
    async def test_round_robin_rotates_after_limit(self, tmp_path, monkeypatch):
        """
        What it does: With sticky limit 1, consecutive successes rotate accounts.
        Purpose: Verify round-robin advances the sticky index per the limit.
        """
        monkeypatch.setattr("kiro.account_manager.CHATGPT_STRATEGY", "round-robin")
        monkeypatch.setattr("kiro.account_manager.CHATGPT_STICKY_LIMIT", 1)
        manager = _make_manager(tmp_path)
        _inject_account(manager, "chatgpt_1", "chatgpt")
        _inject_account(manager, "chatgpt_2", "chatgpt")

        # First selection → acc1; report success (count → 1, reaches limit).
        a1 = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        await manager.report_success(a1.id, "gpt-5.5")
        # Next selection should rotate to acc2.
        a2 = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        await manager.report_success(a2.id, "gpt-5.5")
        # And rotate back to acc1.
        a3 = await manager.get_next_account("gpt-5.5", provider="chatgpt")

        assert a1.id != a2.id
        assert a3.id == a1.id

    @pytest.mark.asyncio
    async def test_round_robin_respects_sticky_limit_of_two(self, tmp_path, monkeypatch):
        """
        What it does: With sticky limit 2, the same account serves 2 requests then rotates.
        Purpose: Verify the limit is honored, not hardcoded to 1.
        """
        monkeypatch.setattr("kiro.account_manager.CHATGPT_STRATEGY", "round-robin")
        monkeypatch.setattr("kiro.account_manager.CHATGPT_STICKY_LIMIT", 2)
        manager = _make_manager(tmp_path)
        _inject_account(manager, "chatgpt_1", "chatgpt")
        _inject_account(manager, "chatgpt_2", "chatgpt")

        s1 = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        await manager.report_success(s1.id, "gpt-5.5")
        s2 = await manager.get_next_account("gpt-5.5", provider="chatgpt")
        await manager.report_success(s2.id, "gpt-5.5")
        s3 = await manager.get_next_account("gpt-5.5", provider="chatgpt")

        assert s1.id == s2.id  # served twice
        assert s3.id != s1.id  # rotated on the third


class TestCodexReportFailureResetsRoundRobin:
    """A RECOVERABLE failure resets the account's consecutive-use counter."""

    @pytest.mark.asyncio
    async def test_failure_resets_consecutive_count(self, tmp_path):
        """
        What it does: report_failure(RECOVERABLE) zeroes consecutive_use_count.
        Purpose: A recovered account should not carry a stale rotation count.
        """
        manager = _make_manager(tmp_path)
        acc = _inject_account(manager, "chatgpt_1", "chatgpt")
        acc.consecutive_use_count = 5

        await manager.report_failure("chatgpt_1", "gpt-5.5", ErrorType.RECOVERABLE, 429, "usage_limit_reached")

        assert acc.consecutive_use_count == 0
        assert acc.failures == 1


class TestPerProviderStateMigration:
    """State persistence: per-provider index + backward-compatible legacy int."""

    @pytest.mark.asyncio
    async def test_legacy_index_maps_to_kiro(self, tmp_path):
        """
        What it does: An old state file with current_account_index loads into the kiro slot.
        Purpose: Backward compatibility with pre-multi-provider state.json.
        """
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "current_account_index": 3,
            "model_to_accounts": {},
            "accounts": {},
        }))
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file),
        )
        await manager.load_state()

        assert manager._current_index_by_provider.get("kiro") == 3
        # Backward-compatible property still reads the kiro slot.
        assert manager._current_account_index == 3

    @pytest.mark.asyncio
    async def test_new_index_map_loads(self, tmp_path):
        """
        What it does: A new state file with current_index_by_provider loads as-is.
        Purpose: Round-trip of the new per-provider index format.
        """
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "current_index_by_provider": {"kiro": 1, "chatgpt": 2},
            "model_to_accounts": {},
            "accounts": {},
        }))
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file),
        )
        await manager.load_state()

        assert manager._current_index_by_provider["kiro"] == 1
        assert manager._current_index_by_provider["chatgpt"] == 2

    @pytest.mark.asyncio
    async def test_save_state_includes_provider_and_indices(self, tmp_path):
        """
        What it does: _save_state writes provider, consecutive_use_count, and both index forms.
        Purpose: Ensure persisted schema carries the new fields.
        """
        manager = _make_manager(tmp_path)
        acc = _inject_account(manager, "chatgpt_1", "chatgpt")
        acc.consecutive_use_count = 2
        manager._current_index_by_provider = {"kiro": 0, "chatgpt": 0}

        await manager._save_state()

        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["accounts"]["chatgpt_1"]["provider"] == "chatgpt"
        assert saved["accounts"]["chatgpt_1"]["consecutive_use_count"] == 2
        assert "current_index_by_provider" in saved
        assert "current_account_index" in saved  # legacy key retained


class TestCodexCredentialLoading:
    """Loading ChatGPT accounts from the dedicated Codex credentials file."""

    @pytest.mark.asyncio
    async def test_disabled_does_not_load_codex(self, tmp_path, monkeypatch):
        """
        What it does: With CHATGPT_ENABLED false, no Codex accounts are loaded.
        Purpose: Opt-in; no effect when disabled.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)
        codex_file = tmp_path / "chatgpt_credentials.json"
        codex_file.write_text(json.dumps([
            {"provider": "chatgpt", "accessToken": "at", "refreshToken": "rt", "chatgptAccountId": "acc_1"},
        ]))
        monkeypatch.setattr("kiro.config.CHATGPT_CREDENTIALS_FILE", str(codex_file))

        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps([]))
        manager = AccountManager(credentials_file=str(creds_file), state_file=str(tmp_path / "state.json"))
        await manager.load_credentials()

        assert not any(a.provider == "chatgpt" for a in manager._accounts.values())

    @pytest.mark.asyncio
    async def test_enabled_loads_codex_accounts(self, tmp_path, monkeypatch):
        """
        What it does: With CHATGPT_ENABLED true, valid Codex accounts are loaded.
        Purpose: Core multi-account credential loading.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        codex_file = tmp_path / "chatgpt_credentials.json"
        codex_file.write_text(json.dumps([
            {"provider": "chatgpt", "accessToken": "at1", "refreshToken": "rt1", "chatgptAccountId": "acc_1"},
            {"provider": "chatgpt", "accessToken": "at2", "refreshToken": "rt2", "chatgptAccountId": "acc_2"},
        ]))
        monkeypatch.setattr("kiro.config.CHATGPT_CREDENTIALS_FILE", str(codex_file))

        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps([]))
        manager = AccountManager(credentials_file=str(creds_file), state_file=str(tmp_path / "state.json"))
        await manager.load_credentials()

        codex_accounts = [a for a in manager._accounts.values() if a.provider == "chatgpt"]
        assert len(codex_accounts) == 2
        meta_ids = {a.account_meta.get("chatgptAccountId") for a in codex_accounts}
        assert meta_ids == {"acc_1", "acc_2"}

    @pytest.mark.asyncio
    async def test_skips_disabled_and_incomplete_entries(self, tmp_path, monkeypatch):
        """
        What it does: enabled=false and entries missing tokens are skipped.
        Purpose: Robust credential validation.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        codex_file = tmp_path / "chatgpt_credentials.json"
        codex_file.write_text(json.dumps([
            {"provider": "chatgpt", "accessToken": "at1", "refreshToken": "rt1", "chatgptAccountId": "acc_ok"},
            {"provider": "chatgpt", "enabled": False, "accessToken": "at2", "refreshToken": "rt2", "chatgptAccountId": "acc_disabled"},
            {"provider": "chatgpt", "accessToken": "at3", "chatgptAccountId": "acc_no_refresh"},
            {"provider": "chatgpt", "refreshToken": "rt4", "chatgptAccountId": "acc_no_access"},
        ]))
        monkeypatch.setattr("kiro.config.CHATGPT_CREDENTIALS_FILE", str(codex_file))

        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps([]))
        manager = AccountManager(credentials_file=str(creds_file), state_file=str(tmp_path / "state.json"))
        await manager.load_credentials()

        codex_accounts = [a for a in manager._accounts.values() if a.provider == "chatgpt"]
        assert len(codex_accounts) == 1
        assert codex_accounts[0].account_meta["chatgptAccountId"] == "acc_ok"

    @pytest.mark.asyncio
    async def test_missing_file_when_enabled_is_graceful(self, tmp_path, monkeypatch):
        """
        What it does: Enabled but no Codex file → no crash, no Codex accounts.
        Purpose: Fail-safe when misconfigured.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_CREDENTIALS_FILE", str(tmp_path / "does_not_exist.json"))

        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps([]))
        manager = AccountManager(credentials_file=str(creds_file), state_file=str(tmp_path / "state.json"))
        await manager.load_credentials()

        assert not any(a.provider == "chatgpt" for a in manager._accounts.values())


class TestCodexAccountInitialization:
    """Lazy initialization of a Codex account builds a CodexAuthManager + static models."""

    @pytest.mark.asyncio
    async def test_initialize_codex_account(self, tmp_path, monkeypatch):
        """
        What it does: _initialize_codex_account sets auth_manager, cache, resolver.
        Purpose: Codex accounts init from static registry without network model fetch.
        """
        manager = _make_manager(tmp_path)
        # Inject an uninitialized codex account with token metadata.
        manager._accounts["chatgpt_1"] = Account(
            id="chatgpt_1",
            provider="chatgpt",
            account_meta={
                "accessToken": "at", "refreshToken": "rt",
                "idToken": None, "chatgptAccountId": "acc_1", "expiresAt": None,
            },
        )

        # Stub CodexAuthManager.get_access_token so no network occurs.
        async def fake_get_token(self):
            return "valid_token"
        monkeypatch.setattr("kiro.auth_codex.CodexAuthManager.get_access_token", fake_get_token)

        ok = await manager._initialize_account("chatgpt_1")

        assert ok is True
        acc = manager._accounts["chatgpt_1"]
        assert acc.auth_manager is not None
        assert acc.auth_manager.provider == "chatgpt"
        assert acc.model_resolver is not None
        # Static Codex models registered.
        assert "gpt-5.5" in acc.model_resolver.get_available_models()

    @pytest.mark.asyncio
    async def test_initialize_codex_account_auth_failure(self, tmp_path, monkeypatch):
        """
        What it does: A refresh failure during init returns False.
        Purpose: Bad credentials don't crash; account stays uninitialized.
        """
        manager = _make_manager(tmp_path)
        manager._accounts["chatgpt_bad"] = Account(
            id="chatgpt_bad",
            provider="chatgpt",
            account_meta={"accessToken": "", "refreshToken": "", "chatgptAccountId": "acc_bad"},
        )

        async def fail_get_token(self):
            raise ValueError("no token")
        monkeypatch.setattr("kiro.auth_codex.CodexAuthManager.get_access_token", fail_get_token)

        ok = await manager._initialize_account("chatgpt_bad")
        assert ok is False
        assert manager._accounts["chatgpt_bad"].auth_manager is None
