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


class TestResolveUpstreamChatGPT:
    """Tests for ChatGPT (Codex) routing in resolve_upstream."""

    def test_codex_model_routes_to_chatgpt_when_enabled(self, monkeypatch):
        """
        What it does: A registered Codex model id routes to "chatgpt" when enabled.
        Purpose: Core Codex routing.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5", "gpt-5.4"})

        assert resolve_upstream("gpt-5.5") == "chatgpt"

    def test_codex_model_routes_to_kiro_when_disabled(self, monkeypatch):
        """
        What it does: A Codex model routes to Kiro when CHATGPT_ENABLED is false.
        Purpose: Opt-in; no Codex path when disabled (backward compat).
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})

        assert resolve_upstream("gpt-5.5") == "kiro"

    def test_unknown_bare_model_not_codex(self, monkeypatch):
        """
        What it does: A bare model not in the Codex registry does not route to chatgpt.
        Purpose: Only registered Codex ids are matched.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})

        assert resolve_upstream("claude-haiku-4.5") == "kiro"

    def test_codex_takes_priority_over_command_code(self, monkeypatch):
        """
        What it does: A Codex model routes to chatgpt even with Command Code enabled.
        Purpose: Codex check runs first; no collision (Codex ids are bare names).
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)

        assert resolve_upstream("gpt-5.5") == "chatgpt"

    def test_command_code_model_unaffected_by_codex(self, monkeypatch):
        """
        What it does: A slash model still routes to command_code with Codex enabled.
        Purpose: No collision between Codex (bare) and Command Code (qualified).
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)

        assert resolve_upstream("deepseek/deepseek-v4-pro") == "command_code"

    def test_dynamic_model_ids_are_honored(self, monkeypatch):
        """
        What it does: Routing reflects a runtime update of CHATGPT_MODEL_IDS.
        Purpose: Discovery updates config.CHATGPT_MODEL_IDS at runtime; resolve_upstream
            reads it live, so newly-discovered models route to Codex without a restart.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        # Simulate discovery replacing the static set with the live plan's models.
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.6-luna", "gpt-5.6-terra"})

        assert resolve_upstream("gpt-5.6-luna") == "chatgpt"
        assert resolve_upstream("gpt-5.6-terra") == "chatgpt"
        # A model no longer in the discovered set falls through to Kiro.
        assert resolve_upstream("gpt-5.5") == "kiro"


class TestResolveUpstreamKiroDisabled:
    """Tests for KIRO_ENABLED gating in resolve_upstream."""

    def test_unmatched_model_returns_kiro_when_enabled(self, monkeypatch):
        """
        What it does: An unmatched bare model routes to Kiro when KIRO_ENABLED is true.
        Purpose: Backward-compatible default behavior (Kiro is the fallback).
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", True)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        assert resolve_upstream("claude-sonnet-4.6") == "kiro"

    def test_unmatched_model_returns_kiro_disabled_when_off(self, monkeypatch):
        """
        What it does: An unmatched model returns "kiro_disabled" when Kiro is off.
        Purpose: Option A — never silently reroute; caller turns this into an error.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        assert resolve_upstream("claude-sonnet-4.6") == "kiro_disabled"

    def test_codex_model_still_routes_when_kiro_disabled(self, monkeypatch):
        """
        What it does: A Codex model still routes to chatgpt when Kiro is off.
        Purpose: Disabling Kiro must not break other enabled upstreams.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})

        assert resolve_upstream("gpt-5.5") == "chatgpt"

    def test_command_code_model_still_routes_when_kiro_disabled(self, monkeypatch):
        """
        What it does: A slash model still routes to command_code when Kiro is off.
        Purpose: Disabling Kiro must not break Command Code routing.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)

        assert resolve_upstream("deepseek/deepseek-v4-pro") == "command_code"

    def test_unmatched_model_kiro_disabled_but_other_upstream_enabled(self, monkeypatch):
        """
        What it does: An unmatched model still returns "kiro_disabled" even when
                      another upstream is enabled.
        Purpose: We never reroute a user's chosen model to a different upstream.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        # A bare model (no slash) that is not a Codex id cannot go to CC either.
        assert resolve_upstream("claude-sonnet-4.6") == "kiro_disabled"


class TestEnabledUpstreams:
    """Tests for the enabled_upstreams() helper."""

    def test_all_enabled(self, monkeypatch):
        """All three flags on → all three names in precedence order."""
        from kiro.upstream_base import enabled_upstreams
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", True)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)

        assert enabled_upstreams() == ["kiro", "command_code", "chatgpt"]

    def test_none_enabled(self, monkeypatch):
        """All flags off → empty list."""
        from kiro.upstream_base import enabled_upstreams
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        assert enabled_upstreams() == []

    def test_only_chatgpt(self, monkeypatch):
        """Only ChatGPT on → single-element list."""
        from kiro.upstream_base import enabled_upstreams
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)

        assert enabled_upstreams() == ["chatgpt"]


class TestKiroDisabledErrorMessage:
    """Tests for the kiro_disabled_error_message() helper."""

    def test_message_lists_enabled_upstreams(self, monkeypatch):
        """
        What it does: Message names the requested model and enabled upstream(s).
        Purpose: Actionable, user-friendly error (project UX requirement).
        """
        from kiro.upstream_base import kiro_disabled_error_message
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        msg = kiro_disabled_error_message("claude-sonnet-4.6")

        assert "claude-sonnet-4.6" in msg
        assert "command_code" in msg
        assert "KIRO_ENABLED" in msg

    def test_message_when_no_upstream_enabled(self, monkeypatch):
        """
        What it does: Message guides the user to enable an upstream when none are.
        Purpose: Covers the degenerate all-disabled case.
        """
        from kiro.upstream_base import kiro_disabled_error_message
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        msg = kiro_disabled_error_message("some-model")

        assert "some-model" in msg
        assert "No upstream is enabled" in msg


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
