# Implementation Plan — ChatGPT (Codex) Multi-Account Integration

Mỗi task viết code + test đi kèm (paranoid testing, AGENTS.md rule 7 & 10: cả OpenAI + Anthropic, cả streaming + non-streaming). Đánh dấu [x] khi hoàn thành.

- [ ] 1. Config foundation cho ChatGPT provider
  - Thêm block `CHATGPT_*` vào `kiro/config.py` (mirror block `COMMAND_CODE_*`): `CHATGPT_ENABLED` (default False), `CHATGPT_BASE_URL`, `CHATGPT_OAUTH_CLIENT_ID`, `CHATGPT_OAUTH_TOKEN_URL`, `CHATGPT_CREDENTIALS_FILE`, `CHATGPT_STRATEGY`, `CHATGPT_STICKY_LIMIT`, `CHATGPT_ORIGINATOR`, `CHATGPT_USER_AGENT`, `CHATGPT_MODELS` (registry tĩnh), helper `get_codex_refresh_url()`/`get_codex_responses_url()`.
  - Cập nhật `.env.example` với các biến `CHATGPT_*` kèm chú thích và cảnh báo rủi ro ToS.
  - Thêm `chatgpt_credentials.json.example` với schema account.
  - Test: mở rộng `tests/unit/test_config.py` — default values, đọc env, enabled/disabled.
  - _Requirements: 7.1, 7.3, 7.4, 9.1, 9.2_

- [ ] 2. Auth abstraction + CodexAuthManager
  - [ ] 2.1 Tạo `kiro/auth_base.py` với Protocol/ABC `UpstreamAuthManager` (interface tối thiểu AccountManager cần: `get_access_token`, `force_refresh`, `is_token_expiring_soon`, `is_token_expired`, metadata provider). Khai báo `KiroAuthManager` tuân theo protocol (không đổi hành vi).
    - Test: `tests/unit/test_auth_manager.py` — xác nhận KiroAuthManager vẫn thỏa protocol (không hồi quy).
    - _Requirements: 2.1_
  - [ ] 2.2 Tạo `kiro/auth_codex.py` — `CodexAuthManager`: giữ token/refresh_token/expires_at/chatgpt_account_id; `get_access_token()` refresh-before-expiry; `force_refresh()` POST form tới token URL, thread-safe (asyncio.Lock); mask log; backfill `chatgpt_account_id` từ JWT claim.
    - Test: `tests/unit/test_auth_codex.py` (mới) — refresh thành công, sắp hết hạn tự refresh, refresh fail (revoked), không log token, backfill account id từ id_token và access_token, refresh đồng thời chỉ gọi 1 lần (lock).
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 2.7_

- [ ] 3. Tổng quát hóa AccountManager theo provider
  - [ ] 3.1 Sửa `Account` dataclass: `auth_manager: Optional[UpstreamAuthManager]`, thêm `provider: str = "kiro"`, `account_meta: dict`. Cập nhật state persistence để lưu/đọc `provider`; đổi `current_account_index: int` → `current_account_index_by_provider: dict` với đọc-ngược tương thích (int cũ → gán `"kiro"`).
    - Test: `tests/unit/test_account_manager.py` — migrate state cũ, round-trip save/load có provider.
    - _Requirements: 3.8, 7.2_
  - [ ] 3.2 `load_credentials()` đọc field `provider` (default kiro); nạp account Codex từ `CHATGPT_CREDENTIALS_FILE` với provider=chatgpt.
    - Test: nạp mixed providers, entry disabled bị bỏ, thiếu token bị bỏ + cảnh báo.
    - _Requirements: 3.1, 7.4_
  - [ ] 3.3 `_initialize_account()` phân nhánh theo provider: chatgpt → dựng `CodexAuthManager` + model list = `CHATGPT_MODELS` (tĩnh, không gọi `/ListAvailableModels`). Nhánh Kiro giữ nguyên.
    - Test: khởi tạo account Codex không chạm mạng Kiro; account Kiro không hồi quy.
    - _Requirements: 3.1, 5.3_
  - [ ] 3.4 `get_next_account(model, exclude_accounts, provider="kiro")` lọc account theo provider; sticky index tách theo provider; thêm `consecutive_use_count` để round-robin theo `CHATGPT_STICKY_LIMIT` (fill-first = không advance chủ động; round-robin = advance sau N). `report_success/failure` cập nhật đúng nhóm provider.
    - Test: fill-first bám 1 account tới khi lỗi; round-robin advance sau N request; cooldown/all-unavailable; single-account bypass circuit breaker; failover không lẫn account khác provider.
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.9_

- [ ] 4. Codex backend + HTTP client
  - [ ] 4.1 Tạo `kiro/upstream_codex.py` — `CodexBackend`: `build_headers(auth_manager, account_meta)` (Authorization Bearer, originator, User-Agent, session_id, `ChatGPT-Account-ID`), `build_url()`, model list tĩnh, error helpers `extract_codex_error_message`/`raise_codex_http_error` + parse `usage_limit_reached`→`resets_at`.
    - Test: `tests/unit/test_upstream_codex.py` (mới) — header đúng gồm ChatGPT-Account-ID theo account_meta; parse resets_at; error message extraction.
    - _Requirements: 2.2, 3.7, 6.1_
  - [ ] 4.2 Tạo `kiro/http_client_codex.py` (mỏng) — retry client keyed theo CodexAuthManager: 401/403 refresh+retry, 429/5xx backoff, per-request client cho streaming; dùng `classify_network_error`.
    - Test: `tests/unit/test_http_client.py` (mở rộng) hoặc file mới — 403→refresh→retry, 429 backoff, network error phân loại.
    - _Requirements: 2.3, 6.1_

- [ ] 5. Converters Codex (OpenAI + Anthropic)
  - [ ] 5.1 Tạo `kiro/converters_codex.py::build_codex_payload` (OpenAI→Codex Responses): input array, instructions default, store=false, stream=true, system→developer, strip server ids, flatten tools + drop tool_choice invalid, reasoning.effort, allowlist filter, xóa field không hỗ trợ.
    - Test: `tests/unit/test_converters_codex.py` (mới) — mọi transform, edge case body rỗng, tool_choice tham chiếu tool không tồn tại, allowlist strip.
    - _Requirements: 4.1, 4.5, 4.6_
  - [ ] 5.2 `build_codex_payload_anthropic` (Anthropic→Codex Responses).
    - Test: cùng file — Anthropic messages/system/tools → Codex; edge cases.
    - _Requirements: 4.2, 4.5_
  - [ ] 5.3 Xử lý ảnh remote: inline base64 trước khi gửi, hoặc trả lỗi rõ ràng nếu ngoài phạm vi bản đầu.
    - Test: ảnh data: giữ nguyên; ảnh remote → inline (mock fetch) hoặc lỗi actionable.
    - _Requirements: 4.7_

- [ ] 6. Streaming Codex (OpenAI + Anthropic, streaming + non-streaming)
  - [ ] 6.1 Tạo `kiro/streaming_codex.py::stream_codex_to_openai` + `collect_codex_response`: parse Codex Responses SSE (`output_text.delta`, `function_call_arguments.delta`, `completed`, reasoning) → OpenAI SSE + object.
    - Test: `tests/unit/test_streaming_codex.py` (mới) — text delta, tool call, usage, completed; non-streaming collect.
    - _Requirements: 4.3_
  - [ ] 6.2 `stream_codex_to_anthropic` + `collect_codex_anthropic_response`.
    - Test: cùng file — sang Anthropic SSE + object.
    - _Requirements: 4.4_
  - [ ] 6.3 Phát hiện lỗi trong body 200-OK (`model_at_capacity`/`server_is_overloaded`) → tín hiệu failover/retry.
    - Test: body 200 chứa lỗi capacity → báo account fallback; overloaded → retry.
    - _Requirements: 6.4_

- [ ] 7. Phân loại lỗi Codex
  - Mở rộng `kiro/account_errors.py::classify_error` cho reason/status Codex: `usage_limit_reached`/`model_at_capacity`/`server_is_overloaded`/5xx → RECOVERABLE; 400/422 → FATAL; 401/403-sau-refresh-fail → FATAL cho account. Không hồi quy phân loại Kiro.
  - Test: `tests/unit/test_account_errors.py` (mở rộng) — mọi mã/reason Codex, không đổi hành vi Kiro.
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [ ] 8. Routing seam
  - Cập nhật `kiro/upstream_base.py::resolve_upstream`: thêm nhánh `chatgpt` (khi enabled + `_is_codex_model(raw_model)`) ĐỨNG TRƯỚC nhánh command_code; giữ Kiro làm default.
  - Test: `tests/unit/test_upstream_base.py` (mở rộng/mới) — enabled/disabled, khớp model Codex, không xung đột model `/` của Command Code, bare Kiro model vẫn về kiro.
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 9. Route handlers OpenAI (streaming + non-streaming) + failover account
  - Thêm `_handle_chatgpt_completion` vào `kiro/routes_openai.py`, dispatch ngay sau `resolve_upstream`. Handler DÙNG LẠI vòng failover account (gọi `get_next_account(..., provider="chatgpt")`, `report_success/report_failure`) nhưng build payload/headers/stream bằng module Codex; per-request client cho streaming.
  - Merge `CHATGPT_MODELS` vào `/v1/models` với `owned_by="openai"` khi enabled.
  - Test: `tests/unit/test_routes_openai.py` (mở rộng) — routing tới Codex, streaming + non-streaming, /v1/models chứa/không chứa model Codex, failover khi 429, header account id đúng.
  - _Requirements: 1.5, 3.2, 3.4, 3.7, 5.1, 5.2_

- [ ] 10. Route handlers Anthropic (đối xứng)
  - Thêm `_handle_chatgpt_completion_anthropic` vào `kiro/routes_anthropic.py`, dispatch sau `resolve_upstream`, cùng vòng failover account.
  - Test: `tests/unit/test_routes_anthropic.py` (mở rộng) — đối xứng với OpenAI: routing, streaming + non-streaming, failover, header account id.
  - _Requirements: 1.5, 3.2, 3.4, 3.7_

- [ ] 11. Wiring main.py lifespan
  - Khi `CHATGPT_ENABLED`: nạp Codex credentials vào `AccountManager` (append entries provider=chatgpt) và/hoặc gắn `app.state.codex_backend`; lazy init giữ nguyên; interval refresh model = 0 (registry tĩnh).
  - Test: `tests/unit/test_main_lifespan.py` (mở rộng) — enabled nạp account Codex; disabled không nạp; không hồi quy khởi tạo Kiro.
  - _Requirements: 7.2_

- [ ] 12. Integration end-to-end
  - Tạo `tests/integration/test_chatgpt_flow.py`: full failover (acc1 429 → acc2 200), round-robin phân phối theo sticky limit, header `ChatGPT-Account-ID` khớp account đang dùng, all-unavailable → 503 + reset time, disabled = no-op. Chạy cho cả OpenAI + Anthropic, streaming + non-streaming. Cô lập mạng hoàn toàn.
  - _Requirements: 3.1–3.9, 8.1, 8.2, 8.3_

- [ ] 13. Tài liệu & hoàn thiện
  - Cập nhật README (mục provider mới + cảnh báo rủi ro ToS + hướng dẫn lấy token + schema credentials + chọn strategy).
  - Cập nhật `tests/README.md` mô tả các file test mới.
  - Chạy toàn bộ `pytest -v`; sửa mọi hồi quy; dọn file tạm.
  - _Requirements: 8.4, 9.1, 9.2_
```