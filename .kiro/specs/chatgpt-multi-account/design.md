# Design — ChatGPT (Codex) Multi-Account Integration

## Overview

Thêm ChatGPT làm upstream provider thứ ba (sau Kiro, Command Code) dùng backend Codex OAuth. Thiết kế bám 2 abstraction có sẵn:

1. **Provider routing seam** (`upstream_base.py`) + **per-provider module set** (`upstream_*.py`, `converters_*.py`, `streaming_*.py`) — pattern đã chín, Command Code là template.
2. **Multi-account machinery** (`account_manager.py`) — logic sticky/round-robin/circuit-breaker/state đúng thứ ta cần, nhưng đang **hardcode Kiro**.

Quyết định cốt lõi: **tổng quát hóa tầng auth để AccountManager dùng lại được cho Codex**, thay vì viết lại một `CodexAccountManager` song song (tránh nhân đôi logic circuit-breaker/state — đúng triết lý "systems over patches"). Command Code không cần điều này (1 key tĩnh), nhưng Codex cần multi-account nên đây là chỗ đầu tư kiến trúc hợp lý.

---

## Architecture

### Sơ đồ luồng định tuyến

```
Client (OpenAI / Anthropic)
   │
   ▼
routes_openai.py / routes_anthropic.py
   │  resolve_upstream(raw_model)
   ├── "kiro"          → AccountManager (Kiro accounts) → KiroHttpClient → Kiro API
   ├── "command_code"  → CommandCodeBackend (single key) → httpx → Command Code
   └── "chatgpt"       → AccountManager (Codex accounts) → CodexHttpClient → Codex backend   ◀── MỚI
```

### Quyết định thiết kế: tổng quát hóa AccountManager

Vấn đề: `AccountManager` hiện coi mỗi `Account` gắn cứng `KiroAuthManager`, `ModelResolver`, `/ListAvailableModels`. Ta cần cùng thuật toán chọn account cho Codex.

Phương án đã chọn — **Provider abstraction ở tầng auth + AccountManager tham số hóa theo provider**:

- Định nghĩa một giao thức `UpstreamAuthManager` (Protocol/ABC) với interface tối thiểu mà AccountManager cần: `get_access_token()`, `force_refresh()`, `is_token_expiring_soon()`, `is_token_expired()`, và metadata provider.
- `KiroAuthManager` đã có sẵn các method này → chỉ cần khai báo tuân theo protocol (không đổi hành vi).
- Thêm `CodexAuthManager` mới implement cùng protocol.
- `Account` dataclass: `auth_manager` đổi type hint sang `UpstreamAuthManager` (union), thêm field `provider: str` và `account_meta: dict` (chứa `chatgptAccountId` cho Codex).
- `AccountManager` nhận thêm khái niệm **provider** khi nạp credentials: mỗi entry credentials có field `provider` (default `"kiro"` để tương thích ngược). `_initialize_account` phân nhánh theo `provider` để dựng đúng auth manager và lấy model list đúng nguồn (Kiro: `/ListAvailableModels`; Codex: registry tĩnh).
- Model selection (`get_next_account`, `report_success/failure`, circuit breaker, state persistence) **giữ nguyên** — provider-agnostic sẵn.

Lý do không tách class riêng: thuật toán sticky + backoff + probabilistic retry + `state.json` schema là phần khó và đã test kỹ; nhân đôi sẽ tạo nợ kỹ thuật và lệch hành vi. Chỉ phần "dựng auth manager + lấy model list" là provider-specific và đã được `_initialize_account` cô lập.

> Ghi chú tương thích: một `AccountManager` instance có thể chứa nhiều provider. Nhưng vì routing đã tách theo `resolve_upstream`, đơn giản nhất là **lọc account theo provider** trong `get_next_account` (thêm tham số `provider`), để sticky index và failover chỉ xoay trong nhóm account cùng provider. Kiro request chỉ chọn account Kiro; ChatGPT request chỉ chọn account Codex.

Thay thế cho tương lai: nếu cần cô lập hoàn toàn, có thể chạy 2 instance AccountManager (một cho Kiro, một cho Codex). Bản này chọn **1 instance, lọc theo provider** vì ít thay đổi wiring `main.py`/routes nhất và tái dùng state file.

---

## Components and Interfaces

### 1. `kiro/config.py` — thêm block `CHATGPT_*`
Mirror block Command Code:
```python
CHATGPT_ENABLED: bool          # default False
CHATGPT_BASE_URL: str          # "https://chatgpt.com/backend-api/codex/responses"
CHATGPT_OAUTH_CLIENT_ID: str   # "app_EMoamEEZ73f0CkXaXp7hrann"
CHATGPT_OAUTH_TOKEN_URL: str   # "https://auth.openai.com/oauth/token"
CHATGPT_CREDENTIALS_FILE: str  # "chatgpt_credentials.json"
CHATGPT_STRATEGY: str          # "fill-first" | "round-robin" (default fill-first)
CHATGPT_STICKY_LIMIT: int      # N request/account cho round-robin (default 3)
CHATGPT_MODEL_REFRESH_INTERVAL: int
CHATGPT_ORIGINATOR: str        # "codex_cli_rs"
CHATGPT_USER_AGENT: str        # "codex_cli_rs/<ver>"
# Registry model tĩnh (list id/name) — có thể override qua env JSON.
CHATGPT_MODELS: list[dict]
```
Thêm helper: `get_codex_refresh_url()`, `get_codex_responses_url()`.

### 2. `kiro/auth_codex.py` (MỚI) — `CodexAuthManager`
Implement `UpstreamAuthManager`. Trách nhiệm:
- Giữ `access_token`, `refresh_token`, `id_token`, `expires_at`, `chatgpt_account_id`.
- `get_access_token()`: trả token còn hạn; nếu sắp hết hạn → `force_refresh()`.
- `force_refresh()`: POST form tới `CHATGPT_OAUTH_TOKEN_URL` (`grant_type=refresh_token`, `client_id`, `refresh_token`, `scope`), cập nhật token + `expires_at`. Thread-safe qua `asyncio.Lock`.
- `is_token_expiring_soon()` / `is_token_expired()`: theo `expires_at` với lead time.
- Trích `chatgpt_account_id` từ `id_token`/`access_token` (JWT claim) nếu credentials thiếu (backfill), mirror `extractCodexAccountInfo` của 9router.
- KHÔNG log token (mask).

### 3. `kiro/auth_base.py` (MỚI, nhẹ) — `UpstreamAuthManager` Protocol
Khai báo interface tối thiểu; `KiroAuthManager` và `CodexAuthManager` cùng tuân theo. Chỉ là type contract, không đổi hành vi Kiro.

### 4. `kiro/account_manager.py` — tổng quát hóa (sửa, không viết lại)
- `Account`: `auth_manager: Optional[UpstreamAuthManager]`, thêm `provider: str = "kiro"`, `account_meta: dict = {}`.
- `load_credentials()`: đọc thêm field `provider` mỗi entry (default `"kiro"`). Với `provider="chatgpt"`, entry mang token trực tiếp hoặc trỏ file; nạp danh sách account Codex.
- `_initialize_account()`: phân nhánh `if provider == "chatgpt": dựng CodexAuthManager + model list = CHATGPT_MODELS (tĩnh)`. Nhánh Kiro giữ nguyên.
- `get_next_account(model, exclude_accounts, provider="kiro")`: lọc `self._accounts` theo `provider` trước khi chạy vòng chọn. Sticky index tách theo provider (dùng dict `_current_index_by_provider` thay cho 1 int — migrate an toàn từ state cũ).
- `report_success/failure`: không đổi logic; thao tác theo account_id nên provider-agnostic.
- State persistence: thêm `provider` vào mỗi account state; `current_account_index` → `current_account_index_by_provider` (đọc ngược tương thích: nếu gặp int cũ, gán cho `"kiro"`).

> Chiến lược round-robin: hiện AccountManager là GLOBAL sticky (fill-first). Để hỗ trợ round-robin theo `CHATGPT_STRATEGY`, thêm một biến đếm `consecutive_use_count` theo provider; khi đạt `CHATGPT_STICKY_LIMIT` thì advance sticky index sang account kế khả dụng. Fill-first = sticky limit vô hạn (không advance chủ động, chỉ advance khi lỗi). Round-robin = advance sau N. Điều này mirror `stickyRoundRobinLimit` của 9router.

### 5. `kiro/upstream_base.py` — thêm nhánh `chatgpt`
```python
def resolve_upstream(raw_model: str) -> str:
    if config.CHATGPT_ENABLED and _is_codex_model(raw_model):
        return "chatgpt"
    if config.COMMAND_CODE_ENABLED and "/" in raw_model:
        return "command_code"
    return "kiro"
```
`_is_codex_model(raw_model)`: khớp với tập id trong `CHATGPT_MODELS` (và/hoặc prefix cấu hình). Thứ tự kiểm tra đảm bảo không xung đột với Command Code (`/`) — Codex model là bare name như `gpt-5.5`.

### 6. `kiro/upstream_codex.py` (MỚI) — `CodexBackend`
Mirror `CommandCodeBackend` nhưng auth qua account:
- `build_headers(auth_manager, account_meta)`: `Authorization: Bearer <token>`, `originator`, `User-Agent`, `session_id`, và `ChatGPT-Account-ID: <chatgptAccountId>`.
- `build_url()`: base `CHATGPT_BASE_URL`.
- Model list: tĩnh từ `CHATGPT_MODELS`.
- Error helpers: `extract_codex_error_message()`, `raise_codex_http_error()`, parse `usage_limit_reached` → `resets_at` cho cooldown.

### 7. `kiro/converters_codex.py` (MỚI)
- `build_codex_payload(request_data: ChatCompletionRequest) -> dict` (OpenAI → Codex Responses).
- `build_codex_payload_anthropic(request_data: AnthropicMessagesRequest) -> dict` (Anthropic → Codex Responses).
- Áp dụng transform mirror executor Codex 9router: `input` array, `instructions` default, `store=false`, `stream=true`, convert system→developer, strip server ids, flatten tools, reasoning.effort, allowlist filter, xóa field không hỗ trợ.

### 8. `kiro/streaming_codex.py` (MỚI)
- `stream_codex_to_openai(response, model)` / `collect_codex_response(...)`.
- `stream_codex_to_anthropic(response, model)` / `collect_codex_anthropic_response(...)`.
- Parse SSE của Codex Responses API (`response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, reasoning...). Phát hiện lỗi trong body 200-OK (`model_at_capacity`) để báo failover.

### 9. `kiro/http_client_codex.py` (MỚI, mỏng) — hoặc tổng quát hóa `KiroHttpClient`
Client retry keyed theo `CodexAuthManager`: 401/403 → refresh + retry; 429/5xx → backoff; per-request client cho streaming (tránh CLOSE_WAIT). Giữ mỏng, tái dùng `network_errors.classify_network_error`.

### 10. `kiro/account_errors.py` — thêm phân loại Codex
Mở rộng `classify_error` để nhận diện reason/status Codex: `usage_limit_reached`, `model_at_capacity`, `server_is_overloaded` → RECOVERABLE; 400/422 → FATAL. Giữ nhánh Kiro nguyên vẹn (phân nhánh theo provider hoặc theo tập reason).

### 11. Routes — `routes_openai.py` & `routes_anthropic.py`
- Sau `resolve_upstream`, thêm nhánh `== "chatgpt"` → `_handle_chatgpt_completion(...)` / `_handle_chatgpt_completion_anthropic(...)`.
- Handler Codex **dùng lại vòng failover account** giống nhánh Kiro (gọi `get_next_account(..., provider="chatgpt")`, `report_success/failure`), nhưng build payload/headers/stream bằng module Codex. Đây là điểm khác Command Code (CC không có vòng account).
- `/v1/models`: merge `CHATGPT_MODELS` với `owned_by="openai"` khi enabled.

### 12. `main.py` (lifespan)
- Khi `CHATGPT_ENABLED`: nạp Codex credentials vào cùng `AccountManager` (append entries provider=chatgpt) hoặc gắn `app.state.codex_backend`. Khởi tạo lazy giữ nguyên.
- Không cần background model refresh nếu dùng registry tĩnh (interval=0).

---

## Data Models

### File credentials Codex — `chatgpt_credentials.json`
```json
[
  {
    "provider": "chatgpt",
    "enabled": true,
    "accessToken": "...",
    "refreshToken": "...",
    "idToken": "...",
    "chatgptAccountId": "acc_...",
    "expiresAt": "2026-01-01T00:00:00Z"
  },
  {
    "provider": "chatgpt",
    "enabled": true,
    "accessToken": "...",
    "refreshToken": "...",
    "chatgptAccountId": "acc_..."
  }
]
```
- `chatgptAccountId` bắt buộc để set header đúng; nếu thiếu, backfill từ JWT claim của `idToken`/`accessToken`.
- Có thể hợp nhất vào `credentials.json` chung bằng field `provider`, hoặc dùng file riêng `CHATGPT_CREDENTIALS_FILE`. Bản đầu: file riêng (đơn giản, tách bạch, ít rủi ro hồi quy Kiro).

### Codex Responses request (rút gọn, sau transform)
```json
{
  "model": "gpt-5.5",
  "input": [{ "type": "message", "role": "developer", "content": [...] }],
  "instructions": "<default codex instructions>",
  "tools": [{ "type": "function", "name": "...", "parameters": {...} }],
  "reasoning": { "effort": "medium", "summary": "auto" },
  "store": false,
  "stream": true
}
```

---

## Error Handling

| Tình huống | Phân loại | Hành vi |
|---|---|---|
| 429 `usage_limit_reached` | RECOVERABLE | cooldown theo `resets_at`; failover account |
| `model_at_capacity` (kể cả trong body 200-OK) | RECOVERABLE | failover account |
| `server_is_overloaded`/5xx | RECOVERABLE | retry rồi failover |
| 401/403 | thử refresh 1 lần → nếu vẫn lỗi: FATAL cho account, failover nếu multi | |
| 400/422 body hỏng | FATAL | trả client ngay |
| refresh token bị thu hồi | FATAL cho account | failover; nếu single account → trả lỗi thật |
| tất cả account cạn | 503 | kèm reset gần nhất (human-readable) |

Nguyên tắc: fail-open cho lỗi tạm thời (failover), fail-fast cho lỗi request. Không log token. Thông điệp client actionable.

---

## Testing Strategy

Theo Requirement 8 và triết lý paranoid testing:

- **Unit** (đặt vào `tests/unit/`, file mới khi là module mới):
  - `test_upstream_base.py` (mở rộng): routing chatgpt bật/tắt, khớp/không khớp, không xung đột Command Code/Kiro.
  - `test_auth_codex.py` (mới): refresh thành công, token hết hạn, refresh fail, mask log, backfill chatgptAccountId từ JWT, thread-safe refresh.
  - `test_account_manager.py` (mở rộng): nạp provider chatgpt, lọc theo provider, sticky/round-robin theo `CHATGPT_STICKY_LIMIT`, fill-first fallback, cooldown, all-unavailable, single-account bypass, migrate state cũ (int → dict).
  - `test_converters_codex.py` (mới): OpenAI→Codex, Anthropic→Codex, strip field, system→developer, strip server ids, flatten tools, tool_choice invalid, reasoning effort, allowlist.
  - `test_streaming_codex.py` (mới): Codex SSE → OpenAI + Anthropic, streaming + non-streaming, phát hiện lỗi trong body 200-OK.
  - `test_account_errors.py` (mở rộng): phân loại Codex FATAL/RECOVERABLE.
- **Integration** (`tests/integration/`):
  - `test_chatgpt_flow.py` (mới): full failover 2 account (acc1 429 → acc2 thành công), round-robin phân phối, header `ChatGPT-Account-ID` đúng theo account, disabled = no-op, cả OpenAI + Anthropic + streaming/non-streaming.
- **Isolation**: dùng `block_all_network_calls`; mock token refresh + upstream responses. Không gọi mạng thật.

---

## Migration & Backward Compatibility

1. `CHATGPT_ENABLED` default false → không code path nào chạy; Kiro/Command Code không đổi.
2. `Account.provider` default `"kiro"` → credentials.json cũ nạp y như trước.
3. State cũ có `current_account_index: int` → đọc ngược: gán cho provider `"kiro"` trong dict mới `current_account_index_by_provider`.
4. Type hint đổi `KiroAuthManager` → `UpstreamAuthManager` là superset; không phá runtime.
5. `.env.example` + `chatgpt_credentials.json.example` được thêm mới, không sửa file người dùng.
