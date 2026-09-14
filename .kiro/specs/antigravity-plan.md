# Implementation Plan: Google Antigravity Provider

Ref: #[[file:.kiro/specs/antigravity-provider.md]]

---

## Task Breakdown (Ordered)

### Phase 1: Foundation

#### Task 1.1 — Config Variables
- **File**: `kiro/config.py`
- **Action**: Add `ANTIGRAVITY_*` env vars after the Command Code section (~line 600)
- **Variables**: ENABLED, REFRESH_TOKEN, PROJECT_ID, BASE_URL, FALLBACK_URL, MAX_TOKENS, MAX_TOKENS_CAP, MODEL_REFRESH_INTERVAL, CALLBACK_PORT, DEFAULT_MODEL
- **Done when**: All vars load from env with sensible defaults

#### Task 1.2 — OAuth & Token Management
- **File**: `kiro/auth_antigravity.py` (NEW)
- **Action**: Implement Google OAuth PKCE flow + token refresh
- **Components**:
  - `AntigravityCredentials` dataclass (access_token, refresh_token, expires_at, project_id, email)
  - `generate_pkce()` → (verifier, challenge)
  - `start_oauth_callback_server(state, verifier)` → temporary server on port 51121
  - `exchange_code_for_tokens(code, verifier)` → tokens
  - `refresh_access_token(refresh_token)` → new access_token
  - `discover_project_id(access_token)` → project_id (via loadCodeAssist + listCloudAICompanionProjects)
  - `get_user_email(access_token)` → email
  - Constants: CLIENT_ID, CLIENT_SECRET, AUTH_URL, TOKEN_URL, SCOPES, REDIRECT_URI
- **Done when**: Can generate auth URL, handle callback, exchange+refresh tokens

#### Task 1.3 — Backend Class
- **File**: `kiro/upstream_antigravity.py` (NEW)
- **Action**: Implement AntigravityBackend class
- **Components**:
  - `AntigravityBackend`:
    - `__init__()` — config, credentials, asyncio.Lock for refresh
    - `build_headers()` → full header set
    - `get_valid_token()` → auto-refresh if < 5min to expiry
    - `resolve_runtime_model(public_id, thinking_effort)` → runtime model ID
    - `endpoint_candidates()` → [base_url, fallback_url]
    - `list_models(client)` → fetch from /v1internal:fetchAvailableModels
    - `refresh_models_periodically(client, interval)` → background loop
    - `is_authenticated()` → bool
    - `set_credentials(creds)` → store after OAuth
  - `ANTIGRAVITY_ROUTING` dict — public ID → runtime IDs per effort
  - `RUNTIME_MAX_OUTPUT_TOKENS` dict
  - `raise_antigravity_http_error(response)` → HTTPException
  - `extract_antigravity_error_message(body)` → str
  - `friendly_antigravity_error(status, text)` → user-friendly message
- **Done when**: Backend can build headers, route models, handle errors

#### Task 1.4 — Upstream Routing
- **File**: `kiro/upstream_base.py`
- **Action**: Add antigravity routing condition BEFORE command_code check
- **Logic**: `if ANTIGRAVITY_ENABLED and raw_model.startswith("antigravity/")` → `"antigravity"`
- **Done when**: `resolve_upstream("antigravity/gemini-3.7-flash")` returns `"antigravity"`

---

### Phase 2: Request Conversion

#### Task 2.1 — Converter Module
- **File**: `kiro/converters_antigravity.py` (NEW)
- **Action**: Build Gemini wire format from OpenAI/Anthropic requests
- **Functions**:
  - `build_antigravity_payload(request: ChatCompletionRequest, runtime_model: str, project_id: str, thinking_effort: str) -> Dict`
  - `build_antigravity_payload_anthropic(request: AnthropicMessagesRequest, runtime_model: str, project_id: str, thinking_effort: str) -> Dict`
  - `_convert_openai_messages(messages) -> List[GeminiContent]`
  - `_convert_anthropic_messages(messages) -> List[GeminiContent]`
  - `_convert_tools(tools, use_legacy: bool) -> List[Dict]`
  - `_dereference_schema(schema) -> Dict` — resolve $ref
  - `_normalize_custom_tool_schema(schema) -> Dict` — allowlist for Claude/GPT-OSS
  - `_build_generation_config(max_tokens, temperature, runtime_model, thinking_effort) -> Dict`
  - `_ensure_first_turn_is_user(contents) -> List[GeminiContent]`
  - `_map_tool_choice(tool_choice) -> str` — auto/none/any/validated
- **Key logic**:
  - System messages → systemInstruction.parts
  - Image base64 → inlineData part
  - tool_use blocks → functionCall part
  - tool_result → functionResponse part (user role)
  - Gemini 3.7 gets thinkingConfig in generationConfig
  - Other models get different runtime IDs per effort
  - Claude/GPT-OSS: `parameters` (strict allowlist), Gemini: `parametersJsonSchema`
- **Done when**: Can convert any OpenAI/Anthropic request to valid Gemini payload

---

### Phase 3: Response Streaming

#### Task 3.1 — Stream Parser
- **File**: `kiro/streaming_antigravity.py` (NEW)
- **Action**: Parse Antigravity SSE stream into KiroEvent objects
- **Functions**:
  - `parse_antigravity_stream(response) -> AsyncGenerator[KiroEvent, None]`
    - Read `data:` lines
    - Parse JSON from each SSE event
    - Handle `[DONE]` sentinel
    - Extract: text parts → content, thought parts → thinking, functionCall → tool_use
    - Extract: usageMetadata → usage, finishReason → finish_reason
    - Handle error objects in stream
    - Sanitize tool call IDs (alphanumeric + underscore + dash only, max 64 chars)

#### Task 3.2 — OpenAI SSE Output
- **Functions**:
  - `stream_antigravity_to_openai(response, model) -> AsyncGenerator[str, None]`
    - Emit OpenAI `chat.completion.chunk` format
    - reasoning_content for thinking, content for text
    - tool_calls delta chunks
    - Final chunk with finish_reason + usage
  - `collect_antigravity_response(response, model) -> Dict`
    - Collect full stream → single OpenAI `chat.completion` response

#### Task 3.3 — Anthropic SSE Output
- **Functions**:
  - `stream_antigravity_to_anthropic(response, model) -> AsyncGenerator[str, None]`
    - Emit Anthropic message_start, content_block_start/delta/stop, message_delta, message_stop
    - Thinking blocks, text blocks, tool_use blocks
  - `collect_antigravity_anthropic_response(response, model) -> Dict`
    - Collect full stream → single Anthropic message response
- **Done when**: Can parse any Antigravity stream and output correct OpenAI/Anthropic format

---

### Phase 4: Route Handlers

#### Task 4.1 — OpenAI Route Handler
- **File**: `kiro/routes_openai.py`
- **Action**: Add `_handle_antigravity_completion()` + routing + model list
- **Changes**:
  - Import new modules
  - Add `_handle_antigravity_completion(request, request_data) -> Response`:
    - Get backend from app.state
    - Strip `antigravity/` prefix from model name
    - Determine thinking effort (from model name or request params)
    - Resolve runtime model
    - Get valid token + build headers
    - Build payload via converter
    - Streaming: per-request httpx.AsyncClient, try endpoint candidates
    - Non-streaming: shared client, collect response
    - Error handling with retry on 403/404
  - In `chat_completions()`: add `if resolve_upstream(...) == "antigravity":` check
  - In `get_models()`: merge antigravity model list

#### Task 4.2 — Anthropic Route Handler
- **File**: `kiro/routes_anthropic.py`
- **Action**: Mirror of Task 4.1 for Anthropic API
- **Changes**:
  - Import new modules
  - Add `_handle_antigravity_completion_anthropic(request, request_data) -> Response`
  - In `messages()`: add routing check
- **Done when**: Both APIs route to Antigravity correctly, stream/non-stream

#### Task 4.3 — Auth Routes
- **File**: `main.py` (or `kiro/routes_antigravity_auth.py`)
- **Action**: Add OAuth login/status endpoints
- **Endpoints**:
  - `GET /antigravity/login` — start OAuth flow, return {auth_url, state}
  - `GET /antigravity/status` — return {authenticated, email, project_id, expires_at}
- **Logic**:
  - `/login` generates PKCE, spawns temp callback server on port 51121, returns auth_url
  - Callback server handles redirect, exchanges tokens, stores creds, shuts down
  - `/status` reads current auth state from backend
- **Done when**: User can complete full login flow via browser

---

### Phase 5: Startup & Integration

#### Task 5.1 — Lifespan Integration
- **File**: `main.py`
- **Action**: Initialize AntigravityBackend in app lifespan
- **Logic**:
  - If `ANTIGRAVITY_ENABLED`:
    - Create `AntigravityBackend()` instance
    - If `ANTIGRAVITY_REFRESH_TOKEN` set → refresh token immediately
    - If authenticated → discover project ID, fetch models
    - Store in `app.state.antigravity_backend`
    - Start periodic model refresh task (if interval > 0)
  - On shutdown: cancel refresh task
- **Also**:
  - Register auth router
  - Import ANTIGRAVITY_ENABLED in config imports
- **Done when**: Server starts with Antigravity backend ready (if configured)

#### Task 5.2 — Config Template
- **File**: `.env.example`
- **Action**: Add Antigravity section with documented variables
- **Done when**: Users know what to configure

---

### Phase 6: Testing

#### Task 6.1 — Unit Tests
- **File**: `tests/unit/test_antigravity.py` (NEW)
- **Test classes**:
  - `TestAntigravityConfig` — env var loading, defaults
  - `TestAntigravityAuth` — PKCE generation, token refresh mock, credential storage, expiry check
  - `TestAntigravityRouting` — upstream_base routes correctly, model→runtime resolution
  - `TestAntigravityConvertersOpenAI` — messages, tools, images, system prompt, first-turn fix, thinking effort
  - `TestAntigravityConvertersAnthropic` — same but from Anthropic format
  - `TestAntigravityStreamParser` — SSE parsing, thinking, tool calls, usage, errors, empty response
  - `TestAntigravityStreamOpenAI` — correct OpenAI chunk format
  - `TestAntigravityStreamAnthropic` — correct Anthropic event format
  - `TestAntigravityBackend` — headers, error messages, rate limit hints
  - `TestAntigravityToolSchema` — dereference, normalize, allowlist
- **Coverage target**: All logical branches, error scenarios, edge cases
- **Done when**: `pytest tests/unit/test_antigravity.py -v` passes with high coverage

---

## Dependency Order

```
Task 1.1 (config)
    ↓
Task 1.2 (auth) ←── Task 1.3 (backend) depends on auth
    ↓                    ↓
Task 1.4 (routing)      │
    ↓                    ↓
Task 2.1 (converters) ←─┘
    ↓
Task 3.1-3.3 (streaming)
    ↓
Task 4.1-4.3 (routes) ←── depends on converters + streaming + backend
    ↓
Task 5.1-5.2 (startup)
    ↓
Task 6.1 (tests) ←── can be written in parallel with implementation
```

---

## Estimated Effort per Task

| Task | Effort | LOC |
|------|--------|-----|
| 1.1 Config | 20 min | ~30 |
| 1.2 Auth | 2-3h | ~250 |
| 1.3 Backend | 2-3h | ~300 |
| 1.4 Routing | 10 min | ~10 |
| 2.1 Converters | 3-4h | ~400 |
| 3.1-3.3 Streaming | 3-4h | ~450 |
| 4.1-4.3 Routes | 2-3h | ~200 |
| 5.1-5.2 Startup | 1h | ~60 |
| 6.1 Tests | 4-5h | ~600 |
| **Total** | **~18-24h** | **~2,300 LOC** |

---

## Implementation Notes

1. **Follow Command Code pattern exactly** — same structure, same conventions, just different wire format
2. **Per-request httpx client for streaming** — critical to avoid CLOSE_WAIT leaks
3. **Try endpoint candidates in order** — primary first, fallback if 403/404/5xx
4. **Strip `antigravity/` prefix** before using model name internally
5. **Google requires first turn = user** — auto-prepend `{"role": "user", "parts": [{"text": "Hello"}]}` if missing
6. **Thinking effort mapping**: If OpenAI request has no explicit thinking param, use model's default (off/low)
7. **Claude/GPT-OSS through Antigravity** need `anthropic-beta: interleaved-thinking-2025-05-14` header for reasoning
8. **All network calls mocked in tests** — use existing `block_all_network_calls` fixture
