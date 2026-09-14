# -*- coding: utf-8 -*-

"""Route-level tests for the ChatGPT (Codex) upstream on the Anthropic API.

Symmetric to test_routes_codex_openai.py, but for /v1/messages:
- non-streaming returns an Anthropic message assembled from the Codex stream.
- streaming returns a text/event-stream with Anthropic SSE events.
- multi-account failover: first account 429 → second account 200.
- FATAL 400 returned immediately without failover.
- disabled feature → not routed to Codex.

Network is isolated: CodexHttpClient is patched with a stub.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kiro.account_errors import ErrorType


class _StubAccount:
    def __init__(self, account_id, chatgpt_account_id):
        self.id = account_id
        self.provider = "chatgpt"
        self.auth_manager = Mock()
        self.auth_manager.chatgpt_account_id = chatgpt_account_id
        self.account_meta = {"chatgptAccountId": chatgpt_account_id}


class _StubManager:
    """Minimal AccountManager stub serving Codex accounts in order."""

    def __init__(self, accounts):
        self._accounts = {a.id: a for a in accounts}
        self.successes = []
        self.failures = []
        self._order = list(self._accounts.keys())

    async def get_next_account(self, model, exclude_accounts=None, provider="kiro"):
        exclude = exclude_accounts or set()
        for aid in self._order:
            acc = self._accounts[aid]
            if acc.provider != provider or aid in exclude:
                continue
            return acc
        return None

    async def report_success(self, account_id, model):
        self.successes.append(account_id)

    async def report_failure(self, account_id, model, error_type, status_code, reason):
        self.failures.append((account_id, status_code))

    async def _save_state(self):
        return None


def _fake_stream_response(lines, status_code=200, body=b""):
    async def aiter_lines():
        for line in lines:
            yield line

    resp = AsyncMock()
    resp.status_code = status_code
    resp.aiter_lines = aiter_lines
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()
    return resp


def _json_line(payload):
    return "data: " + json.dumps(payload)


def _codex_sse(text_lines, usage=None):
    lines = [_json_line({"type": "response.output_text.delta", "delta": t}) for t in text_lines]
    lines.append(_json_line({"type": "response.completed", "response": {
        "status": "completed", "usage": usage or {"input_tokens": 1, "output_tokens": 1}}}))
    return lines


def _patch_codex_client(responses):
    call_state = {"i": 0}

    def make_client(auth_manager, backend, shared_client=None):
        client = AsyncMock()

        async def request_with_retry(payload, session_id=None, stream=True):
            idx = min(call_state["i"], len(responses) - 1)
            call_state["i"] += 1
            client.last_session_id = session_id
            return responses[idx]

        client.request_with_retry = request_with_retry
        client.close = AsyncMock()
        make_client.last_client = client
        return client

    return patch("kiro.routes_anthropic.CodexHttpClient", side_effect=make_client)


def _enable_codex(test_client, monkeypatch, accounts):
    from kiro.upstream_codex import CodexBackend
    monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
    monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})
    manager = _StubManager(accounts)
    test_client.app.state.account_manager = manager
    test_client.app.state.codex_backend = CodexBackend()
    return manager


def _headers(api_key):
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


class TestMessagesCodex:
    """Tests for routing /v1/messages to the Codex upstream."""

    def test_non_streaming_returns_message(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A non-streaming Codex request returns an Anthropic message.
        Purpose: End-to-end non-streaming Codex path (Anthropic).
        """
        manager = _enable_codex(test_client, monkeypatch, [_StubAccount("chatgpt_1", "acc_1")])
        resp = _fake_stream_response(_codex_sse(["Hello"], {"input_tokens": 3, "output_tokens": 2}))

        with _patch_codex_client([resp]):
            r = test_client.post(
                "/v1/messages",
                headers=_headers(valid_proxy_api_key),
                json={"model": "gpt-5.5", "max_tokens": 100,
                      "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0] == {"type": "text", "text": "Hello"}
        assert manager.successes == ["chatgpt_1"]

    def test_streaming_returns_event_stream(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A streaming Codex request returns Anthropic SSE events.
        Purpose: End-to-end streaming Codex path (Anthropic).
        """
        _enable_codex(test_client, monkeypatch, [_StubAccount("chatgpt_1", "acc_1")])
        resp = _fake_stream_response(_codex_sse(["Hi"]))

        with _patch_codex_client([resp]):
            r = test_client.post(
                "/v1/messages",
                headers=_headers(valid_proxy_api_key),
                json={"model": "gpt-5.5", "max_tokens": 100,
                      "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "message_start" in r.text
        assert "text_delta" in r.text
        assert "message_stop" in r.text

    def test_failover_first_429_then_second_200(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: Account #1 429 → fail over to account #2 (200).
        Purpose: Multi-account failover on the Anthropic path.
        """
        manager = _enable_codex(test_client, monkeypatch, [
            _StubAccount("chatgpt_1", "acc_1"),
            _StubAccount("chatgpt_2", "acc_2"),
        ])
        err = _fake_stream_response([], status_code=429, body=b'{"error":{"type":"usage_limit_reached"}}')
        ok = _fake_stream_response(_codex_sse(["ok"]))

        with _patch_codex_client([err, ok]):
            r = test_client.post(
                "/v1/messages",
                headers=_headers(valid_proxy_api_key),
                json={"model": "gpt-5.5", "max_tokens": 100,
                      "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 200
        assert r.json()["content"][0]["text"] == "ok"
        assert ("chatgpt_1", 429) in manager.failures
        assert manager.successes == ["chatgpt_2"]

    def test_fatal_400_returned_immediately(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A 400 from Codex is returned without failover.
        Purpose: FATAL errors are not retried on other accounts.
        """
        manager = _enable_codex(test_client, monkeypatch, [
            _StubAccount("chatgpt_1", "acc_1"),
            _StubAccount("chatgpt_2", "acc_2"),
        ])
        bad = _fake_stream_response([], status_code=400, body=b'{"error":{"message":"bad request"}}')

        with _patch_codex_client([bad]):
            r = test_client.post(
                "/v1/messages",
                headers=_headers(valid_proxy_api_key),
                json={"model": "gpt-5.5", "max_tokens": 100,
                      "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 400
        assert manager.failures == []

    def test_disabled_not_routed_to_codex(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: With CHATGPT_ENABLED false, a Codex model is not routed to Codex.
        Purpose: Opt-in; disabled falls through to the Kiro path.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})

        with patch("kiro.routes_anthropic.CodexHttpClient") as codex_cls:
            test_client.post(
                "/v1/messages",
                headers=_headers(valid_proxy_api_key),
                json={"model": "gpt-5.5", "max_tokens": 100,
                      "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
            codex_cls.assert_not_called()


# =============================================================================
# KIRO_ENABLED gating (Anthropic API)
# =============================================================================

class TestKiroDisabledAnthropic:
    """Tests for the KIRO_ENABLED flag on the Anthropic /v1/messages endpoint."""

    def test_unmatched_model_returns_400_when_kiro_disabled(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """
        What it does: /v1/messages returns 400 for an unmatched model when
                      KIRO_ENABLED is false (option A: clear error, no reroute).
        Purpose: Symmetric to the OpenAI endpoint — both APIs behave the same.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        r = test_client.post(
            "/v1/messages",
            headers=_headers(valid_proxy_api_key),
            json={
                "model": "claude-sonnet-4.6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "claude-sonnet-4.6" in detail

    def test_error_when_kiro_disabled_does_not_reroute(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """
        What it does: An unmatched model still returns 400 even when Command Code
                      is enabled.
        Purpose: We never silently reroute a user's chosen model on Anthropic API.
        """
        monkeypatch.setattr("kiro.config.KIRO_ENABLED", False)
        monkeypatch.setattr("kiro.config.COMMAND_CODE_ENABLED", True)
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)

        r = test_client.post(
            "/v1/messages",
            headers=_headers(valid_proxy_api_key),
            json={
                "model": "claude-sonnet-4.6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert r.status_code == 400
        assert "command_code" in r.json()["detail"]
