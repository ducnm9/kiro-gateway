# -*- coding: utf-8 -*-

"""End-to-end integration tests for the ChatGPT (Codex) multi-account flow.

These tests drive the real route handlers through the REAL AccountManager
(loaded from a Codex credentials file with two accounts) and the real
CodexBackend + converters + streaming translation. Only two seams are mocked:
- CodexAuthManager.get_access_token (so account init needs no network), and
- CodexHttpClient.request_with_retry (so the upstream response is scripted).

This verifies the true multi-account machinery end to end:
- fill-first failover (acc1 429 → acc2 200),
- round-robin distribution across two accounts,
- per-account ChatGPT-Account-ID binding,
- all-accounts-unavailable behavior,
- both OpenAI and Anthropic surfaces, streaming and non-streaming.

Network is fully isolated.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kiro.account_manager import AccountManager
from kiro.upstream_codex import CodexBackend
from kiro.models_openai import ChatCompletionRequest, ChatMessage
from kiro.models_anthropic import AnthropicMessagesRequest, AnthropicMessage
from kiro.routes_openai import _handle_chatgpt_completion
from kiro.routes_anthropic import _handle_chatgpt_completion_anthropic


# =============================================================================
# Helpers
# =============================================================================

async def _make_manager(tmp_path, monkeypatch, num_accounts=2, strategy="fill-first", sticky=3):
    """Build a real AccountManager with N initialized Codex accounts.

    Codex credentials are written to a temp file; get_access_token is stubbed so
    lazy init succeeds without network. Returns the manager with all Codex
    accounts already initialized (so get_next_account skips lazy init).
    """
    codex_file = tmp_path / "chatgpt_credentials.json"
    entries = [
        {"provider": "chatgpt", "accessToken": f"at{i}", "refreshToken": f"rt{i}",
         "chatgptAccountId": f"acc_{i}"}
        for i in range(num_accounts)
    ]
    codex_file.write_text(json.dumps(entries))

    monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
    monkeypatch.setattr("kiro.config.CHATGPT_CREDENTIALS_FILE", str(codex_file))
    monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})
    monkeypatch.setattr("kiro.account_manager.CHATGPT_STRATEGY", strategy)
    monkeypatch.setattr("kiro.account_manager.CHATGPT_STICKY_LIMIT", sticky)

    # Stub Codex token retrieval so init/use needs no network.
    async def fake_get_token(self):
        return "valid_token"
    monkeypatch.setattr("kiro.auth_codex.CodexAuthManager.get_access_token", fake_get_token)

    manager = AccountManager(
        credentials_file=str(tmp_path / "credentials.json"),
        state_file=str(tmp_path / "state.json"),
    )
    # Only a Codex credentials file — no Kiro accounts needed for these tests.
    (tmp_path / "credentials.json").write_text(json.dumps([]))
    await manager.load_credentials()

    # Eagerly initialize all Codex accounts.
    for aid in list(manager._accounts.keys()):
        await manager._initialize_account(aid)

    return manager


def _make_request(manager):
    """Build a fake FastAPI Request whose app.state has the manager + backend."""
    state = SimpleNamespace(account_manager=manager, codex_backend=CodexBackend())
    app = SimpleNamespace(state=state)
    return SimpleNamespace(app=app)


def _fake_response(lines, status_code=200, body=b""):
    async def aiter_lines():
        for line in lines:
            yield line

    resp = AsyncMock()
    resp.status_code = status_code
    resp.aiter_lines = aiter_lines
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()
    return resp


def _codex_ok(text="ok", usage=None):
    lines = [json.dumps({"type": "response.output_text.delta", "delta": text})]
    lines = ["data: " + line for line in lines]
    lines.append("data: " + json.dumps({"type": "response.completed", "response": {
        "status": "completed", "usage": usage or {"input_tokens": 1, "output_tokens": 1}}}))
    return _fake_response(lines)


def _codex_429():
    return _fake_response([], status_code=429, body=b'{"error":{"type":"usage_limit_reached"}}')


def _patch_client(responses, sessions_seen):
    """Patch both route modules' CodexHttpClient to return scripted responses.

    Records the session_id used per call into ``sessions_seen``.
    """
    state = {"i": 0}

    def make_client(auth_manager, backend, shared_client=None):
        client = AsyncMock()

        async def request_with_retry(payload, session_id=None, stream=True):
            sessions_seen.append(session_id)
            idx = min(state["i"], len(responses) - 1)
            state["i"] += 1
            return responses[idx]

        client.request_with_retry = request_with_retry
        client.close = AsyncMock()
        return client

    return patch.multiple(
        "kiro.routes_openai", CodexHttpClient=make_client
    ), patch.multiple(
        "kiro.routes_anthropic", CodexHttpClient=make_client
    )


async def _read_stream(response):
    """Drain a StreamingResponse body into a string."""
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return "".join(chunks)


def _openai_req(stream=False):
    return ChatCompletionRequest(
        model="gpt-5.5", stream=stream,
        messages=[ChatMessage(role="user", content="hi")],
    )


def _anthropic_req(stream=False):
    return AnthropicMessagesRequest(
        model="gpt-5.5", max_tokens=100, stream=stream,
        messages=[AnthropicMessage(role="user", content="hi")],
    )


# =============================================================================
# OpenAI surface
# =============================================================================

class TestChatGPTFlowOpenAI:
    """End-to-end Codex flow on the OpenAI surface."""

    @pytest.mark.asyncio
    async def test_non_streaming_success(self, tmp_path, monkeypatch):
        """
        What it does: A non-streaming request succeeds on the first account.
        Purpose: Full stack (manager → converter → client → collect) works.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("hello")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion(req, _openai_req(stream=False))
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_fill_first_failover(self, tmp_path, monkeypatch):
        """
        What it does: acc_0 returns 429; the flow fails over to acc_1 (200).
        Purpose: Real multi-account fill-first failover on quota exhaustion.
        """
        manager = await _make_manager(tmp_path, monkeypatch, strategy="fill-first")
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_429(), _codex_ok("recovered")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion(req, _openai_req(stream=False))
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "recovered"
        # Two accounts were tried; the first (acc_0) then the second (acc_1).
        assert sessions[0] == "acc_0"
        assert sessions[1] == "acc_1"

    @pytest.mark.asyncio
    async def test_streaming_success(self, tmp_path, monkeypatch):
        """
        What it does: A streaming request returns an OpenAI event-stream.
        Purpose: Full streaming translation path.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("streamed")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion(req, _openai_req(stream=True))
            body = await _read_stream(resp)
        assert '"content": "streamed"' in body
        assert "data: [DONE]" in body

    @pytest.mark.asyncio
    async def test_round_robin_distributes(self, tmp_path, monkeypatch):
        """
        What it does: With round-robin + sticky=1, two sequential requests use
            different accounts.
        Purpose: Verify real round-robin rotation across accounts.
        """
        manager = await _make_manager(tmp_path, monkeypatch, strategy="round-robin", sticky=1)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("a"), _codex_ok("b")], sessions)
        with po, pa:
            await _handle_chatgpt_completion(req, _openai_req(stream=False))
            await _handle_chatgpt_completion(req, _openai_req(stream=False))
        # The two requests landed on different accounts.
        assert sessions[0] != sessions[1]
        assert set(sessions) == {"acc_0", "acc_1"}

    @pytest.mark.asyncio
    async def test_all_accounts_unavailable(self, tmp_path, monkeypatch):
        """
        What it does: Both accounts 429 → the request ultimately fails.
        Purpose: Exhaustion surfaces an error after failover.
        """
        from fastapi import HTTPException
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_429(), _codex_429()], sessions)
        with po, pa:
            with pytest.raises(HTTPException):
                await _handle_chatgpt_completion(req, _openai_req(stream=False))


# =============================================================================
# Anthropic surface
# =============================================================================

class TestChatGPTFlowAnthropic:
    """End-to-end Codex flow on the Anthropic surface."""

    @pytest.mark.asyncio
    async def test_non_streaming_success(self, tmp_path, monkeypatch):
        """
        What it does: A non-streaming Anthropic request succeeds.
        Purpose: Full stack on the Anthropic surface.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("hi-anthropic")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion_anthropic(req, _anthropic_req(stream=False))
        body = json.loads(resp.body)
        assert body["type"] == "message"
        assert body["content"][0]["text"] == "hi-anthropic"

    @pytest.mark.asyncio
    async def test_failover(self, tmp_path, monkeypatch):
        """
        What it does: acc_0 429 → fail over to acc_1 (200) on Anthropic.
        Purpose: Multi-account failover on the Anthropic surface.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_429(), _codex_ok("ok2")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion_anthropic(req, _anthropic_req(stream=False))
        body = json.loads(resp.body)
        assert body["content"][0]["text"] == "ok2"
        assert sessions[:2] == ["acc_0", "acc_1"]

    @pytest.mark.asyncio
    async def test_streaming_success(self, tmp_path, monkeypatch):
        """
        What it does: A streaming Anthropic request returns SSE events.
        Purpose: Full streaming translation on the Anthropic surface.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("streamed-a")], sessions)
        with po, pa:
            resp = await _handle_chatgpt_completion_anthropic(req, _anthropic_req(stream=True))
            body = await _read_stream(resp)
        assert "message_start" in body
        assert "text_delta" in body
        assert "message_stop" in body

    @pytest.mark.asyncio
    async def test_account_id_binding(self, tmp_path, monkeypatch):
        """
        What it does: The session id passed upstream is the selected account's id.
        Purpose: Verify per-account ChatGPT-Account-ID binding end to end.
        """
        manager = await _make_manager(tmp_path, monkeypatch)
        req = _make_request(manager)
        sessions = []
        po, pa = _patch_client([_codex_ok("x")], sessions)
        with po, pa:
            await _handle_chatgpt_completion_anthropic(req, _anthropic_req(stream=False))
        # The first (and only) account used is acc_0.
        assert sessions[0] == "acc_0"
