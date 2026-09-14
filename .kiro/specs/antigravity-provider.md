# Spec: Google Antigravity Provider Integration

## Overview

Add Google Antigravity (Cloud Code Assist) as a third upstream provider in Kiro Gateway, alongside Kiro and Command Code. This enables users to access Gemini, Claude, and GPT-OSS models through Google's Cloud Code Assist API using a single gateway.

## Background

- **pi-antigravity** (https://github.com/Rahularya01/pi-antigravity) is a Pi Coding Agent extension that connects to Google's Cloud Code Assist / Antigravity API.
- The API is undocumented but has been reverse-engineered by the pi-antigravity project.
- The gateway will replicate the wire-level protocol (endpoints, headers, request/response format) from pi-antigravity's observable behavior.

## Goals

1. Users can access Antigravity models via OpenAI-compatible and Anthropic-compatible APIs
2. OAuth login flow works when gateway runs on localhost (same machine as user's browser)
3. Fallback to env-var refresh token for remote/Docker deployments
4. Automatic token refresh (Google tokens expire ~1h)
5. Support all 7 public models: Gemini 3.7/3.6/3.5 Flash, Gemini 3.1 Pro, Claude Sonnet 4.6, Claude Opus 4.6, GPT-OSS 120B
6. Streaming and non-streaming for both OpenAI and Anthropic API surfaces

## Non-Goals

- Building a standalone CLI for Antigravity auth (out of scope)
- Supporting Google Workspace enterprise-specific auth flows
- Implementing quota management UI (just pass errors through)

---

## Architecture

### Upstream Routing

```
Client Request (model: "antigravity/gemini-3.7-flash")
    │
    ▼
upstream_base.resolve_upstream(raw_model)
    │
    ├── contains "/"  and starts with "antigravity/" → "antigravity"
    ├── contains "/"  (other)                        → "command_code"
    └── no "/"                                       → "kiro"
```

### Request Flow

```
OpenAI/Anthropic Request
    │
    ▼
routes_openai.py / routes_anthropic.py
    │ (routing decision)
    ▼
_handle_antigravity_completion()
    │
    ├── converters_antigravity.py  → Build Gemini wire format request
    │
    ▼
POST https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse
    │
    ▼
streaming_antigravity.py → Parse SSE → KiroEvent → OpenAI/Anthropic SSE chunks
```

### Auth Flow

```
┌───────────────────────────── OAUTH FLOW ─────────────────────────────┐
│                                                                       │
│  1. GET /antigravity/login                                           │
│     → Generate PKCE (verifier + challenge) + random state            │
│     → Store {verifier, state} in app.state (pending_auth)            │
│     → Return {auth_url: "https://accounts.google.com/...", state}    │
│                                                                       │
│  2. User opens auth_url in browser → Google consent screen           │
│                                                                       │
│  3. Google redirects → http://localhost:51121/oauth-callback          │
│     (temporary server listening only during auth)                     │
│                                                                       │
│  4. Temp server receives code + state:                               │
│     → Verify state matches pending_auth                              │
│     → Exchange code for tokens (access_token + refresh_token)        │
│     → Discover projectId via /v1internal:loadCodeAssist              │
│     → Get user email via Google userinfo API                         │
│     → Store credentials in AntigravityBackend                        │
│     → Shutdown temp server                                           │
│     → Return HTML "Success! Close this tab."                         │
│                                                                       │
│  5. Subsequent requests use stored access_token                      │
│     → Auto-refresh when expired (using refresh_token)                │
│                                                                       │
│  FALLBACK (Docker/remote):                                           │
│     Set ANTIGRAVITY_REFRESH_TOKEN in .env                            │
│     → Gateway refreshes on startup                                   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### OAuth Constants (from pi-antigravity public client)

```
Client ID:     <public Antigravity desktop OAuth client ID>
Client Secret: <public Antigravity desktop OAuth client secret>
Redirect URI:  http://localhost:51121/oauth-callback
Auth URL:      https://accounts.google.com/o/oauth2/v2/auth
Token URL:     https://oauth2.googleapis.com/token
Scopes:        aicode, cloud-platform, userinfo.email, userinfo.profile, cclog, experimentsandconfigs
```

### Wire Format

**Endpoint**: `POST {base_url}/v1internal:streamGenerateContent?alt=sse`
**Fallback URLs**: `https://cloudcode-pa.googleapis.com`, `https://daily-cloudcode-pa.sandbox.googleapis.com`

**Request envelope**:
```json
{
  "project": "<project-id>",
  "model": "<runtime-model-id>",
  "request": {
    "contents": [
      {"role": "user", "parts": [{"text": "Hello"}]},
      {"role": "model", "parts": [{"text": "Hi!"}]}
    ],
    "systemInstruction": {
      "role": "user",
      "parts": [{"text": "System prompt here"}]
    },
    "generationConfig": {
      "maxOutputTokens": 65536,
      "temperature": 0.7,
      "thinkingConfig": {"thinkingLevel": "HIGH"}
    },
    "tools": [{
      "functionDeclarations": [{
        "name": "get_weather",
        "description": "Get weather",
        "parametersJsonSchema": {"type": "object", "properties": {...}}
      }]
    }],
    "toolConfig": {
      "functionCallingConfig": {"mode": "AUTO"}
    }
  },
  "requestType": "agent",
  "userAgent": "antigravity",
  "requestId": "<uuid>"
}
```

**Response SSE format** (each line: `data: <json>`):
```json
{
  "candidates": [{
    "content": {
      "parts": [
        {"text": "response text"},
        {"text": "thinking...", "thought": true, "thoughtSignature": "sig"},
        {"functionCall": {"name": "tool", "args": {...}, "id": "call_123"}}
      ]
    },
    "finishReason": "STOP"
  }],
  "usageMetadata": {
    "promptTokenCount": 100,
    "candidatesTokenCount": 50,
    "thoughtsTokenCount": 20,
    "cachedContentTokenCount": 10,
    "totalTokenCount": 180
  }
}
```

### Model Routing Table

| Public Model ID | Default Runtime | Thinking Routing |
|---|---|---|
| `antigravity/gemini-3.7-flash` | `gemini-3.7-flash-tiered` | thinkingConfig: LOW/MEDIUM/HIGH |
| `antigravity/gemini-3.6-flash` | `gemini-3.6-flash-low` | low→`-low`, medium→`-medium`, high→`-high` |
| `antigravity/gemini-3.5-flash` | `gemini-3.5-flash-extra-low` | low→`-extra-low`, medium→`-low`, high→`gemini-3-flash-agent` |
| `antigravity/gemini-3.1-pro` | `gemini-3.1-pro-low` | low→`-low`, high→`gemini-pro-agent` |
| `antigravity/claude-sonnet-4-6` | `claude-sonnet-4-6` | all→`claude-sonnet-4-6` |
| `antigravity/claude-opus-4-6` | `claude-opus-4-6-thinking` | all→`claude-opus-4-6-thinking` |
| `antigravity/gpt-oss-120b` | `gpt-oss-120b-medium` | all→`gpt-oss-120b-medium` |

### Headers

```
Authorization: Bearer <access_token>
Content-Type: application/json
Accept: text/event-stream
User-Agent: antigravity/1.15.8 <os>/<arch>
X-Goog-Api-Client: google-cloud-sdk vscode_cloudshelleditor/0.1
Client-Metadata: {"ideType":"ANTIGRAVITY","platform":"<MACOS|LINUX|WINDOWS>","pluginType":"GEMINI"}
```

### Tool Schema Handling

- **Gemini models**: Use `parametersJsonSchema` field (full JSON Schema support)
- **Claude/GPT-OSS models**: Use `parameters` field (Protobuf subset: only type, description, properties, required, items, enum allowed)
- Detection: model runtime ID starts with `claude-` or `gpt-oss-` → use legacy `parameters`

---

## File Plan

### New Files

| File | Purpose |
|------|---------|
| `kiro/upstream_antigravity.py` | Backend class, OAuth auth, token refresh, model discovery, error handling |
| `kiro/converters_antigravity.py` | OpenAI/Anthropic → Gemini wire format conversion |
| `kiro/streaming_antigravity.py` | Parse Antigravity SSE → KiroEvent, emit OpenAI/Anthropic SSE |
| `kiro/auth_antigravity.py` | OAuth flow: login endpoint, callback server, token exchange, refresh |
| `tests/unit/test_antigravity.py` | Comprehensive unit tests |

### Modified Files

| File | Changes |
|------|---------|
| `kiro/config.py` | Add `ANTIGRAVITY_*` configuration variables |
| `kiro/upstream_base.py` | Add `"antigravity"` routing condition |
| `kiro/routes_openai.py` | Add `_handle_antigravity_completion()`, merge models in `/v1/models` |
| `kiro/routes_anthropic.py` | Add `_handle_antigravity_completion_anthropic()` |
| `main.py` | Init backend in lifespan, register auth routes, periodic model refresh |
| `.env.example` | Add Antigravity config section |

---

## Implementation Tasks

### Phase 1: Foundation (Config + Auth + Backend)

**Task 1.1: Configuration** (`kiro/config.py`)
- Add environment variables:
  - `ANTIGRAVITY_ENABLED` (bool, default false)
  - `ANTIGRAVITY_REFRESH_TOKEN` (str, optional — fallback for Docker)
  - `ANTIGRAVITY_PROJECT_ID` (str, optional — override project discovery)
  - `ANTIGRAVITY_BASE_URL` (str, default `https://cloudcode-pa.googleapis.com`)
  - `ANTIGRAVITY_FALLBACK_URL` (str, default `https://daily-cloudcode-pa.sandbox.googleapis.com`)
  - `ANTIGRAVITY_MAX_TOKENS` (int, default 65536)
  - `ANTIGRAVITY_MODEL_REFRESH_INTERVAL` (int, default 3600)
  - `ANTIGRAVITY_CALLBACK_PORT` (int, default 51121)
  - `ANTIGRAVITY_DEFAULT_MODEL` (str, default `gemini-3.7-flash`)

**Task 1.2: OAuth Module** (`kiro/auth_antigravity.py`)
- Google OAuth PKCE flow (code_verifier + code_challenge)
- Temporary HTTP server on port 51121 for callback
- Token exchange (code → access_token + refresh_token)
- Token refresh (refresh_token → new access_token)
- Project ID discovery via `/v1internal:loadCodeAssist`
- User email discovery via Google userinfo API
- Credential storage dataclass: `AntigravityCredentials(access_token, refresh_token, expires_at, project_id, email)`
- Thread-safe refresh with asyncio.Lock

**Task 1.3: Backend Class** (`kiro/upstream_antigravity.py`)
- `AntigravityBackend` class:
  - `name = "antigravity"`
  - `__init__()` — load config, init credentials
  - `build_headers()` → auth + metadata headers
  - `get_valid_token()` → auto-refresh if expired
  - `discover_project_id()` → call loadCodeAssist API
  - `list_models()` → call fetchAvailableModels API
  - `refresh_models_periodically()` → background task
  - `resolve_runtime_model(public_id, thinking_effort)` → routing table
- Error handling: `raise_antigravity_http_error()`, `extract_antigravity_error_message()`
- Rate limit hint parsing (429 responses)

**Task 1.4: Routing** (`kiro/upstream_base.py`)
- Add condition: `if ANTIGRAVITY_ENABLED and raw_model.startswith("antigravity/")` → return `"antigravity"`

### Phase 2: Converters

**Task 2.1: OpenAI → Gemini Converter** (`kiro/converters_antigravity.py`)
- `build_antigravity_payload(request: ChatCompletionRequest, runtime_model: str, project_id: str) -> Dict`
- Convert messages: user→user, assistant→model, system→systemInstruction, tool→functionResponse
- Handle image content (base64 inlineData)
- Convert tools: JSON Schema → functionDeclarations (parametersJsonSchema vs parameters based on model)
- Map tool_choice to Gemini toolConfig modes
- Thinking effort → generationConfig.thinkingConfig
- Enforce first-turn-must-be-user constraint
- Dereference $ref in JSON schemas
- Strip unsupported schema fields for Claude/GPT-OSS models

**Task 2.2: Anthropic → Gemini Converter** (`kiro/converters_antigravity.py`)
- `build_antigravity_payload_anthropic(request: AnthropicMessagesRequest, runtime_model: str, project_id: str) -> Dict`
- Convert Anthropic content blocks (text, image, tool_use, tool_result) → Gemini parts
- Extract system prompt → systemInstruction
- Map Anthropic tools → functionDeclarations

### Phase 3: Streaming

**Task 3.1: Stream Parser** (`kiro/streaming_antigravity.py`)
- `parse_antigravity_stream(response) -> AsyncGenerator[KiroEvent, None]`
  - Parse SSE `data:` lines
  - Handle `[DONE]` sentinel
  - Map `part.text` → `KiroEvent(type="content")`
  - Map `part.thought=true` → `KiroEvent(type="thinking")`
  - Map `part.functionCall` → `KiroEvent(type="tool_use")`
  - Map `usageMetadata` → `KiroEvent(type="usage")`
  - Map `finishReason` → finish
  - Handle `error` objects in stream

**Task 3.2: OpenAI SSE Output** (`kiro/streaming_antigravity.py`)
- `stream_antigravity_to_openai(response, model) -> AsyncGenerator[str, None]`
- `collect_antigravity_response(response, model) -> Dict` (non-streaming)

**Task 3.3: Anthropic SSE Output** (`kiro/streaming_antigravity.py`)
- `stream_antigravity_to_anthropic(response, model) -> AsyncGenerator[str, None]`
- `collect_antigravity_anthropic_response(response, model) -> Dict` (non-streaming)

### Phase 4: Route Integration

**Task 4.1: OpenAI Routes** (`kiro/routes_openai.py`)
- `_handle_antigravity_completion(request, request_data) -> Response`
- Add routing check in `chat_completions()`: `if resolve_upstream(...) == "antigravity"`
- Merge antigravity models in `get_models()`
- Per-request client for streaming, shared client for non-streaming

**Task 4.2: Anthropic Routes** (`kiro/routes_anthropic.py`)
- `_handle_antigravity_completion_anthropic(request, request_data) -> Response`
- Add routing check in `messages()`

**Task 4.3: Auth Routes** (`main.py` or new router)
- `GET /antigravity/login` → initiate OAuth, return auth_url
- `GET /antigravity/status` → return auth state {authenticated, email, expires_at}
- Temporary callback server on port 51121

### Phase 5: Startup & Integration

**Task 5.1: Lifespan** (`main.py`)
- Init `AntigravityBackend` if enabled
- If `ANTIGRAVITY_REFRESH_TOKEN` set → refresh token on startup
- Discover project ID
- Fetch available models
- Start periodic model refresh task
- Register auth router
- Graceful shutdown (cancel refresh task)

**Task 5.2: Config Template** (`.env.example`)
- Add Antigravity section with all variables documented

### Phase 6: Tests

**Task 6.1: Unit Tests** (`tests/unit/test_antigravity.py`)
- Auth: token refresh, PKCE generation, credential storage, expired token detection
- Converters: OpenAI→Gemini, Anthropic→Gemini, tool schema handling, image conversion, first-turn constraint
- Streaming: SSE parsing, thinking blocks, function calls, usage metadata, error events, finish reasons
- Backend: model routing table, header building, error message extraction, rate limit hints
- Edge cases: empty responses, malformed SSE, missing fields, concurrent refresh

---

## Configuration Example

```bash
# ============================================
# Antigravity (Google Cloud Code Assist)
# ============================================
# Enable Antigravity as an upstream provider
ANTIGRAVITY_ENABLED=false

# Google OAuth refresh token (for Docker/remote deployments)
# Get this by running the OAuth flow on a local machine first
ANTIGRAVITY_REFRESH_TOKEN=""

# Override project ID (skip automatic discovery)
ANTIGRAVITY_PROJECT_ID=""

# API endpoints (usually no need to change)
ANTIGRAVITY_BASE_URL="https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_FALLBACK_URL="https://daily-cloudcode-pa.sandbox.googleapis.com"

# Generation defaults
ANTIGRAVITY_MAX_TOKENS=65536

# Model list refresh interval (seconds, 0 to disable)
ANTIGRAVITY_MODEL_REFRESH_INTERVAL=3600

# OAuth callback port (must be 51121 for Google's registered redirect URI)
ANTIGRAVITY_CALLBACK_PORT=51121
```

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Google changes internal API | Monitor pi-antigravity releases; modular design allows quick fixes |
| Quota exhaustion (shared pool) | Clear 429 error messages with reset time; suggest model switching |
| Token refresh fails | Retry with backoff; clear error message directing to re-login |
| Port 51121 conflict | Check port availability before OAuth; clear error if occupied |
| Remote gateway can't do OAuth | Fallback to `ANTIGRAVITY_REFRESH_TOKEN` env var |
| Claude/GPT-OSS tool schema errors | Strict schema normalization (allowlist approach from pi-antigravity) |
| First turn not user (400 error) | Auto-prepend minimal user message |

---

## Success Criteria

1. `GET /v1/models` includes antigravity models when enabled
2. `POST /v1/chat/completions` with `model: "antigravity/gemini-3.7-flash"` returns valid streaming response
3. `POST /v1/messages` (Anthropic) with antigravity model works
4. OAuth flow: `/antigravity/login` → browser → callback → authenticated
5. Token auto-refresh works (no manual intervention after initial login)
6. All 7 models accessible and functional
7. Thinking blocks correctly parsed and forwarded
8. Tool calls work (function calling)
9. Tests pass with full network isolation
10. Error messages are actionable and user-friendly
