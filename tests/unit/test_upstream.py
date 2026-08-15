# -*- coding: utf-8 -*-

"""
Unit tests for upstream backend routing (upstream_base.py and upstream_cc.py).

Tests:
- resolve_upstream() routing decision (enabled/disabled, slash vs bare model)
- CommandCodeBackend header building
- CommandCodeBackend model listing (success and error propagation)
"""

import asyncio
import httpx
import pytest
from unittest.mock import AsyncMock, Mock

from kiro.upstream_base import resolve_upstream
from kiro.upstream_cc import CommandCodeBackend


# =============================================================================
# Tests for resolve_upstream()
# =============================================================================

class TestResolveUpstream:
    """Tests for the upstream routing decision."""

    def test_resolve_upstream_disabled_returns_kiro_for_slash_model(self, monkeypatch):
        """
        What it does: Verifies a slash model routes to Kiro when Command Code is disabled.
        Purpose: Ensure zero behavior change when COMMAND_CODE_ENABLED is false.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)

        result = resolve_upstream("deepseek/deepseek-v4-pro")

        assert result == "kiro"

    def test_resolve_upstream_bare_model_returns_kiro(self, monkeypatch):
        """
        What it does: Verifies a bare model routes to Kiro even when enabled.
        Purpose: Ensure only provider-qualified models go to Command Code.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)

        result = resolve_upstream("claude-haiku-4.5")

        assert result == "kiro"

    def test_resolve_upstream_slash_model_returns_command_code(self, monkeypatch):
        """
        What it does: Verifies a slash model routes to Command Code when enabled.
        Purpose: Ensure provider-qualified models are routed to the CC upstream.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)

        result = resolve_upstream("deepseek/deepseek-v4-pro")

        assert result == "command_code"


# =============================================================================
# Tests for CommandCodeBackend
# =============================================================================

class TestCommandCodeBackend:
    """Tests for the Command Code backend."""

    def test_command_code_backend_build_headers(self, monkeypatch):
        """
        What it does: Verifies build_headers() returns the exact required headers.
        Purpose: Ensure wire-compatible header values are produced.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_API_KEY", "test-key")
        monkeypatch.setattr("kiro.config.COMMAND_CODE_BASE_URL", "https://api.commandcode.ai")
        monkeypatch.setattr("kiro.config.COMMAND_CODE_VERSION", "0.24.1")
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENVIRONMENT", "production")

        backend = CommandCodeBackend()

        headers = backend.build_headers()

        assert headers == {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
            "x-command-code-version": "0.24.1",
            "x-cli-environment": "production",
        }

    @pytest.mark.asyncio
    async def test_command_code_backend_list_models_success(self, monkeypatch):
        """
        What it does: Verifies list_models() fetches and returns the model data list.
        Purpose: Ensure the model-list endpoint is called with the correct URL and auth.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_API_KEY", "test-key")
        monkeypatch.setattr("kiro.config.COMMAND_CODE_BASE_URL", "https://api.commandcode.ai")

        backend = CommandCodeBackend()

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "deepseek/deepseek-v4-pro",
                    "name": "DeepSeek V4 Pro",
                    "context_length": 200000,
                }
            ],
        }

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=mock_response)

        models = await backend.list_models(mock_client)

        assert models == mock_response.json.return_value["data"]
        mock_client.get.assert_awaited_once_with(
            "https://api.commandcode.ai/provider/v1/models",
            headers={"Authorization": "Bearer test-key"},
        )

    @pytest.mark.asyncio
    async def test_command_code_backend_list_models_raises(self, monkeypatch):
        """
        What it does: Verifies list_models() propagates upstream HTTP errors.
        Purpose: Ensure failures surface to the caller rather than being swallowed.
        """
        monkeypatch.setattr("kiro.config.COMMAND_CODE_API_KEY", "test-key")
        monkeypatch.setattr("kiro.config.COMMAND_CODE_BASE_URL", "https://api.commandcode.ai")

        backend = CommandCodeBackend()

        request = httpx.Request("GET", "https://api.commandcode.ai/provider/v1/models")
        response = httpx.Response(500, request=request)
        error = httpx.HTTPStatusError("Internal Server Error", request=request, response=response)

        mock_client = Mock()
        mock_client.get = AsyncMock(side_effect=error)

        with pytest.raises(httpx.HTTPStatusError):
            await backend.list_models(mock_client)

    @pytest.mark.asyncio
    async def test_refresh_models_periodically_updates_models(self):
        """
        What it does: Verifies the periodic refresh re-fetches and updates models.
        Purpose: Ensure /v1/models stays current without a restart.
        """
        backend = CommandCodeBackend()
        backend.models = []
        backend.list_models = AsyncMock(return_value=[{"id": "new/model"}])

        task = asyncio.create_task(
            backend.refresh_models_periodically(Mock(), 0.01)
        )
        await asyncio.sleep(0.06)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert backend.models == [{"id": "new/model"}]
        assert backend.list_models.await_count >= 1

    @pytest.mark.asyncio
    async def test_refresh_models_periodically_survives_failure(self):
        """
        What it does: Verifies a transient refresh failure does not kill the loop.
        Purpose: Ensure the background task keeps retrying after an error.
        """
        backend = CommandCodeBackend()
        backend.models = []
        calls = 0

        async def fake_list_models(client):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("boom")
            return [{"id": "m2"}]

        backend.list_models = fake_list_models

        task = asyncio.create_task(
            backend.refresh_models_periodically(Mock(), 0.01)
        )
        await asyncio.sleep(0.06)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert backend.models == [{"id": "m2"}]
        assert calls >= 2
