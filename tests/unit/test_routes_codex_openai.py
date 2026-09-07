# -*- coding: utf-8 -*-

"""Route-level tests for the ChatGPT (Codex) upstream on the OpenAI API.

Covers /v1/chat/completions routing to Codex and /v1/models merge:
- non-streaming returns a chat.completion assembled from the Codex stream.
- streaming returns a text/event-stream ending with [DONE].
- multi-account failover: first account 429 → second account 200.
- the ChatGPT-Account-ID binding uses the selected account (session id).
- all-accounts-unavailable → error.
- disabled feature → not routed to Codex.
- /v1/models includes Codex models with owned_by="openai".

Network is isolated: CodexHttpClient is patched with a stub.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kiro.account_errors import ErrorType


# =============================================================================
# Helpers: a stub AccountManager and Codex accounts
# =============================================================================

class _StubAccount:
    def __init__(self, account_id, chatgpt_account_id):
        self.id = account_id
        self.provider = "chatgpt"
        self.auth_manager = Mock()
        self.auth_manager.chatgpt_account_id = chatgpt_account_id
        self.account_meta = {"chatgptAccountId": chatgpt_account_id}


class _StubManager:
    """Minimal AccountManager stub for route tests.

    Serves accounts in order, honoring exclude_accounts. Records success/failure.
    """

    def __init__(self, accounts):
        self._accounts = {a.id: a for a in accounts}
        self.successes = []
        self.failures = []
        # A scripted selection order can be set; defaults to insertion order.
        self._order = list(self._accounts.keys())

    async def get_next_account(self, model, exclude_accounts=None, provider="kiro"):
        exclude = exclude_accounts or set()
        for aid in self._order:
            acc = self._accounts[aid]
            if acc.provider != provider:
                continue
            if aid in exclude:
                continue
            return acc
        return None

    async def report_success(self, account_id, model):
        self.successes.append(account_id)

    async def report_failure(self, account_id, model, error_type, status_code, reason):
        self.failures.append((account_id, status_code))

    async def _save_state(self):
        # No-op: satisfies the app lifespan shutdown which calls this on teardown.
        return None


def _fake_stream_response(lines, status_code=200, body=b""):
    """Duck-typed async streaming response with aiter_lines/aread/aclose."""
    async def aiter_lines():
        for line in lines:
            yield line

    resp = AsyncMock()
    resp.status_code = status_code
    resp.aiter_lines = aiter_lines
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()
    return resp


def _codex_sse(text_lines, usage=None):
    """Build a minimal Codex SSE line list producing the given text."""
    lines = [json_line({"type": "response.output_text.delta", "delta": t}) for t in text_lines]
    lines.append(json_line({"type": "response.completed", "response": {
        "status": "completed", "usage": usage or {"input_tokens": 1, "output_tokens": 1}}}))
    return lines


def json_line(payload):
    return "data: " + json.dumps(payload)


def _patch_codex_client(responses):
    """Patch CodexHttpClient so request_with_retry returns scripted responses.

    Args:
        responses: a list of fake responses returned in order across accounts.
    """
    call_state = {"i": 0}

    def make_client(auth_manager, backend, shared_client=None):
        client = AsyncMock()

        async def request_with_retry(payload, session_id=None, stream=True):
            idx = min(call_state["i"], len(responses) - 1)
            call_state["i"] += 1
            resp = responses[idx]
            # Record the session id used (for account-binding assertions).
            client.last_session_id = session_id
            return resp

        client.request_with_retry = request_with_retry
        client.close = AsyncMock()
        make_client.last_client = client
        return client

    return patch("kiro.routes_openai.CodexHttpClient", side_effect=make_client), call_state


def _enable_codex(test_client, monkeypatch, accounts):
    """Enable Codex on config, install a stub manager + real backend on app.state."""
    from kiro.upstream_codex import CodexBackend
    monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", True)
    monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5", "gpt-5.4"})
    manager = _StubManager(accounts)
    test_client.app.state.account_manager = manager
    test_client.app.state.codex_backend = CodexBackend()
    return manager


# =============================================================================
# /v1/chat/completions routing to Codex
# =============================================================================

class TestChatCompletionsCodex:
    """Tests for routing /v1/chat/completions to the Codex upstream."""

    def test_non_streaming_returns_chat_completion(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A non-streaming Codex request returns a chat.completion.
        Purpose: End-to-end non-streaming Codex path.
        """
        manager = _enable_codex(test_client, monkeypatch, [_StubAccount("chatgpt_1", "acc_1")])
        resp = _fake_stream_response(_codex_sse(["Hello", " world"], {"input_tokens": 3, "output_tokens": 2}))
        patcher, _ = _patch_codex_client([resp])

        with patcher:
            r = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello world"
        assert body["usage"]["total_tokens"] == 5
        assert manager.successes == ["chatgpt_1"]

    def test_streaming_returns_event_stream(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A streaming Codex request returns an event-stream with [DONE].
        Purpose: End-to-end streaming Codex path.
        """
        _enable_codex(test_client, monkeypatch, [_StubAccount("chatgpt_1", "acc_1")])
        resp = _fake_stream_response(_codex_sse(["Hi"]))
        patcher, _ = _patch_codex_client([resp])

        with patcher:
            r = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "data: [DONE]" in r.text
        assert '"content": "Hi"' in r.text

    def test_failover_first_429_then_second_200(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: Account #1 returns 429; the handler fails over to account #2 (200).
        Purpose: Core multi-account fill-first failover on quota exhaustion.
        """
        manager = _enable_codex(test_client, monkeypatch, [
            _StubAccount("chatgpt_1", "acc_1"),
            _StubAccount("chatgpt_2", "acc_2"),
        ])
        err_429 = _fake_stream_response([], status_code=429,
                                        body=b'{"error":{"type":"usage_limit_reached"}}')
        ok = _fake_stream_response(_codex_sse(["ok"]))
        patcher, _ = _patch_codex_client([err_429, ok])

        with patcher:
            r = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"
        # Account #1 was marked failed (429), account #2 succeeded.
        assert ("chatgpt_1", 429) in manager.failures
        assert manager.successes == ["chatgpt_2"]

    def test_account_id_binding_uses_selected_account(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: The session id passed to the client is the selected account's id.
        Purpose: Verify per-account binding (ChatGPT-Account-ID) at the route layer.
        """
        _enable_codex(test_client, monkeypatch, [_StubAccount("chatgpt_1", "acc_bind")])
        resp = _fake_stream_response(_codex_sse(["x"]))
        patcher, _ = _patch_codex_client([resp])

        with patcher as p:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
            # side_effect factory recorded the last client it built.
            assert p.side_effect.last_client.last_session_id == "acc_bind"

    def test_all_accounts_unavailable_returns_error(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: When all Codex accounts 429, the request fails after failover.
        Purpose: Exhaustion path surfaces an error rather than hanging.
        """
        _enable_codex(test_client, monkeypatch, [
            _StubAccount("chatgpt_1", "acc_1"),
            _StubAccount("chatgpt_2", "acc_2"),
        ])
        err = _fake_stream_response([], status_code=429,
                                    body=b'{"error":{"type":"usage_limit_reached"}}')
        patcher, _ = _patch_codex_client([err, err])

        with patcher:
            r = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code in (429, 503)

    def test_fatal_400_returned_immediately(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: A 400 from Codex is returned to the client without failover.
        Purpose: FATAL errors are not retried on other accounts.
        """
        manager = _enable_codex(test_client, monkeypatch, [
            _StubAccount("chatgpt_1", "acc_1"),
            _StubAccount("chatgpt_2", "acc_2"),
        ])
        bad = _fake_stream_response([], status_code=400, body=b'{"error":{"message":"bad request"}}')
        patcher, _ = _patch_codex_client([bad])

        with patcher:
            r = test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        assert r.status_code == 400
        # No account was marked failed for a FATAL error.
        assert manager.failures == []

    def test_disabled_not_routed_to_codex(self, test_client, valid_proxy_api_key, monkeypatch):
        """
        What it does: With CHATGPT_ENABLED false, a Codex model id is NOT routed to Codex.
        Purpose: Opt-in; disabled falls through to the normal (Kiro) path.
        """
        monkeypatch.setattr("kiro.config.CHATGPT_ENABLED", False)
        monkeypatch.setattr("kiro.config.CHATGPT_MODEL_IDS", {"gpt-5.5"})
        # If it were routed to Codex, this would raise (no codex_backend); instead it
        # goes down the Kiro path. We assert it does NOT hit the Codex handler by
        # ensuring CodexHttpClient is never constructed.
        with patch("kiro.routes_openai.CodexHttpClient") as codex_cls:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
            codex_cls.assert_not_called()


# =============================================================================
# /v1/models merge
# =============================================================================

class TestModelsMergeCodex:
    """Tests for merging Codex models into /v1/models."""

    def test_models_include_codex_when_backend_present(self, test_client, valid_proxy_api_key):
        """
        What it does: Codex models appear in /v1/models with owned_by=openai.
        Purpose: Expose Codex models to clients.
        """
        from kiro.upstream_codex import CodexBackend
        test_client.app.state.codex_backend = CodexBackend()

        r = test_client.get("/v1/models", headers={"Authorization": f"Bearer {valid_proxy_api_key}"})
        assert r.status_code == 200
        data = r.json()["data"]
        codex_models = [m for m in data if m["owned_by"] == "openai"]
        assert codex_models
        assert any(m["id"] == "gpt-5.5" for m in codex_models)

    def test_models_absent_when_no_codex_backend(self, test_client, valid_proxy_api_key):
        """
        What it does: No Codex models when the backend is absent.
        Purpose: Backward-compat when Codex is disabled.
        """
        test_client.app.state.codex_backend = None
        r = test_client.get("/v1/models", headers={"Authorization": f"Bearer {valid_proxy_api_key}"})
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert "gpt-5.5" not in ids
