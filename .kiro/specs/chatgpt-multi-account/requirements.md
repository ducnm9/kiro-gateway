# Requirements — ChatGPT (Codex) Multi-Account Integration

## Introduction

Kiro Gateway hiện chỉ proxy tới Kiro (Amazon Q) và Command Code. Feature này thêm **ChatGPT/OpenAI làm upstream provider mới** thông qua backend Codex OAuth (`https://chatgpt.com/backend-api/codex/responses`), với hỗ trợ **nhiều tài khoản ChatGPT** để tối đa hóa quota.

Người dùng mục tiêu có 2 tài khoản ChatGPT (ví dụ ChatGPT Go/Plus) và muốn:
- Kết nối cả 2 tài khoản vào gateway.
- **Round-robin** giữa các tài khoản (chia tải), HOẶC
- **Fill-first fallback**: dùng cạn tài khoản 1, khi hết quota / bị rate-limit thì tự động chuyển sang tài khoản 2.

Feature phải nhất quán với AGENTS.md: triển khai cho **cả OpenAI và Anthropic API**, **cả streaming và non-streaming**, có **test bao phủ toàn diện**, code tiếng Anh, type hints, docstrings, và tuân theo triết lý "systems over patches" (tái sử dụng/ tổng quát hóa hạ tầng account sẵn có thay vì vá tạm).

### Bối cảnh kiến trúc (đã khảo sát)
- `kiro/upstream_base.py::resolve_upstream()` là seam định tuyến provider theo tên model (raw, trước normalize).
- `kiro/upstream_cc.py` + `converters_cc.py` + `streaming_cc.py` là template provider tự chứa (Command Code) — nhưng dùng **1 API key tĩnh, KHÔNG dùng AccountManager**.
- `kiro/account_manager.py` chứa toàn bộ logic multi-account (GLOBAL sticky round-robin + Circuit Breaker + exponential backoff + probabilistic retry + state persistence) nhưng **hardcode cho Kiro** (`KiroAuthManager`, `/ListAvailableModels`).
- `kiro/auth.py::KiroAuthManager` **không có provider abstraction** — cần một token manager OAuth mới cho Codex.
- `kiro/config.py` có block config Command Code làm template cho config provider mới.

### Ranh giới không thuộc phạm vi (Out of scope)
- Không xây UI/dashboard (kiro-gateway không có UI; cấu hình qua env + file credentials).
- Không tự động chạy OAuth login flow qua trình duyệt trong gateway. Gateway **nhận token đã có sẵn** (accessToken/refreshToken) do người dùng nạp vào file credentials, giống cách Command Code nhận key. (Việc lấy token ban đầu do người dùng thực hiện bằng công cụ ngoài; xem Design.)
- Không phá vỡ hành vi mặc định: khi `CHATGPT_ENABLED=false`, gateway hoạt động y hệt hiện tại.

---

## Glossary
- **Codex backend**: endpoint nội bộ ChatGPT dùng bởi Codex CLI (`chatgpt.com/backend-api/codex/responses`), định dạng OpenAI Responses API, streaming-only.
- **Codex account**: một tài khoản ChatGPT có `accessToken`/`refreshToken` + `chatgptAccountId`.
- **Fill-first**: chiến lược ưu tiên dùng một account đến khi lỗi/hết quota rồi mới chuyển account kế (chính là hành vi GLOBAL sticky hiện có của AccountManager).
- **Round-robin**: luân phiên account sau mỗi N request.

---

## Requirements

### Requirement 1 — Định tuyến request tới ChatGPT provider
**User Story:** Là người dùng, tôi muốn gửi request với model ChatGPT (ví dụ `gpt-5.5`) và gateway tự định tuyến tới Codex backend, để dùng được ChatGPT qua gateway.

#### Acceptance Criteria
1. WHEN `CHATGPT_ENABLED=true` AND model name khớp tập model Codex (theo prefix/registry cấu hình) THEN hệ thống SHALL định tuyến request tới upstream `chatgpt`.
2. WHEN `CHATGPT_ENABLED=false` THEN hệ thống SHALL KHÔNG bao giờ định tuyến tới `chatgpt`, giữ nguyên hành vi Kiro/Command Code hiện tại.
3. Việc định tuyến SHALL xảy ra trên **raw model name** trước khi normalize, trong `resolve_upstream()`.
4. WHEN một model không thuộc Codex và không có `/` (Command Code) THEN hệ thống SHALL định tuyến tới Kiro như hiện tại (không hồi quy).
5. Định tuyến SHALL hoạt động đồng nhất ở cả `/v1/chat/completions` (OpenAI) và `/v1/messages` (Anthropic).

### Requirement 2 — Xác thực & làm mới token cho Codex
**User Story:** Là người dùng, tôi muốn gateway tự dùng và làm mới token OAuth ChatGPT, để không phải đăng nhập lại thủ công.

#### Acceptance Criteria
1. Hệ thống SHALL cung cấp một token manager cho Codex (mirror interface của `KiroAuthManager`: `get_access_token()`, phát hiện hết hạn, refresh trước hạn).
2. WHEN access token hết hạn hoặc sắp hết hạn THEN hệ thống SHALL refresh bằng `refreshToken` qua `https://auth.openai.com/oauth/token` (grant_type=refresh_token, encoding=form) trước khi gọi upstream.
3. WHEN upstream trả 401/403 THEN hệ thống SHALL thử refresh token một lần rồi retry request đó.
4. Refresh SHALL thread-safe (dùng khóa async) để tránh refresh đồng thời trên cùng account.
5. WHEN refresh thất bại (refresh token bị thu hồi) THEN hệ thống SHALL phân loại lỗi là FATAL cho account đó (không lặp vô hạn) và, ở chế độ multi-account, chuyển sang account khác.
6. Token đã refresh SHALL được ghi lại (in-memory tối thiểu; tùy chọn ghi bền vào file credentials) để không refresh lại mỗi request.
7. Hệ thống SHALL KHÔNG bao giờ ghi log giá trị token/refresh token (chỉ log dạng masked).

### Requirement 3 — Multi-account: round-robin và fill-first fallback
**User Story:** Là người dùng có 2 tài khoản ChatGPT, tôi muốn gateway luân phiên hoặc dùng cạn acc1 rồi sang acc2, để gộp quota và không gián đoạn.

#### Acceptance Criteria
1. Hệ thống SHALL nạp nhiều Codex account từ file credentials (một provider entry chứa danh sách account, mỗi account có token + `chatgptAccountId`).
2. WHEN chiến lược là fill-first (mặc định) THEN hệ thống SHALL bám một account (GLOBAL sticky) cho tới khi account đó gặp lỗi RECOVERABLE (quota/429/5xx) rồi mới chuyển account kế theo thứ tự.
3. WHEN chiến lược là round-robin THEN hệ thống SHALL luân phiên account sau mỗi N request (N cấu hình được), tôn trọng account đang trong cooldown thì bỏ qua.
4. WHEN một account gặp lỗi quota/rate-limit (429, `usage_limit_reached`) THEN hệ thống SHALL đánh dấu account đó không khả dụng theo cooldown (ưu tiên dùng `resets_at`/`resets_in_seconds` nếu upstream cung cấp; nếu không, dùng exponential backoff của Circuit Breaker).
5. WHEN tất cả account Codex đều không khả dụng THEN hệ thống SHALL trả lỗi 503 kèm thời điểm reset gần nhất (human-readable).
6. WHEN request thành công trên một account THEN hệ thống SHALL reset trạng thái lỗi của account đó và cập nhật sticky index về account đó.
7. Mỗi request tới Codex SHALL gắn header `ChatGPT-Account-ID` đúng với `chatgptAccountId` của account đang dùng, để không cross-bind sai tài khoản.
8. Trạng thái account (failures, cooldown, stats, sticky index) SHALL được persist qua restart (theo cơ chế `state.json` hiện có).
9. WHEN chỉ có 1 account Codex THEN hệ thống SHALL bỏ qua Circuit Breaker cho provider đó và trả lỗi upstream thật (nhất quán với hành vi single-account hiện tại của Kiro).

### Requirement 4 — Chuyển đổi định dạng request/response (Codex Responses API)
**User Story:** Là người dùng dùng CLI OpenAI hoặc Anthropic, tôi muốn request của tôi được dịch đúng sang định dạng Codex Responses API và ngược lại, để nhận kết quả chuẩn.

#### Acceptance Criteria
1. Hệ thống SHALL dịch OpenAI Chat Completions → Codex Responses API request (input array, instructions, tools flat, `store=false`, `stream=true`, reasoning.effort, allowlist field).
2. Hệ thống SHALL dịch Anthropic Messages → Codex Responses API request.
3. Hệ thống SHALL dịch Codex Responses SSE stream → OpenAI Chat Completions SSE (streaming) VÀ → object non-streaming.
4. Hệ thống SHALL dịch Codex Responses SSE stream → Anthropic Messages SSE (streaming) VÀ → object non-streaming.
5. Hệ thống SHALL strip các field không được Codex chấp nhận (temperature, top_p, max_tokens, previous_response_id, v.v.) và convert `role=system` → `role=developer`.
6. Hệ thống SHALL loại các server-generated item id (`rs_`/`fc_`/`resp_`/`msg_`) khi `store=false`.
7. WHEN request có ảnh URL từ xa THEN hệ thống SHALL inline ảnh thành base64 trước khi gửi (Codex không fetch ảnh từ xa) — hoặc, nếu ngoài phạm vi bản đầu, SHALL trả lỗi rõ ràng thay vì gửi request hỏng.

### Requirement 5 — Danh sách model & endpoint /v1/models
**User Story:** Là người dùng, tôi muốn thấy model ChatGPT khả dụng trong `/v1/models`, để biết dùng tên model nào.

#### Acceptance Criteria
1. WHEN `CHATGPT_ENABLED=true` THEN `/v1/models` SHALL bao gồm các model Codex với `owned_by="openai"`.
2. WHEN `CHATGPT_ENABLED=false` THEN `/v1/models` SHALL KHÔNG chứa model Codex.
3. Danh sách model Codex SHALL cấu hình được (registry tĩnh trong config, có thể mở rộng qua env).

### Requirement 6 — Phân loại lỗi Codex
**User Story:** Là người dùng, tôi muốn lỗi Codex được phân loại đúng (FATAL vs RECOVERABLE), để failover chỉ xảy ra khi hợp lý.

#### Acceptance Criteria
1. Hệ thống SHALL phân loại 429/`usage_limit_reached`/`model_at_capacity`/5xx/`server_is_overloaded` là RECOVERABLE (kích hoạt failover account).
2. Hệ thống SHALL phân loại 400/422 (request hỏng) là FATAL (trả về client ngay, không failover).
3. Hệ thống SHALL phân loại 401/403 sau khi refresh token thất bại là FATAL cho account (chuyển account khác nếu multi-account).
4. WHEN 200 OK nhưng body SSE chứa lỗi `model_at_capacity` THEN hệ thống SHALL coi là RECOVERABLE và failover account (mirror hành vi executor Codex của 9router).
5. Thông điệp lỗi trả về client SHALL actionable và không lộ chi tiết nội bộ/token.

### Requirement 7 — Cấu hình & tương thích ngược
**User Story:** Là người vận hành, tôi muốn bật/tắt ChatGPT provider qua config với default an toàn, để không ảnh hưởng cài đặt hiện tại.

#### Acceptance Criteria
1. Hệ thống SHALL thêm block config `CHATGPT_*` (mirror `COMMAND_CODE_*`): `CHATGPT_ENABLED` (default false), endpoint base URL, OAuth clientId/tokenUrl, đường dẫn file credentials Codex, model refresh interval, sticky/round-robin settings.
2. WHEN `CHATGPT_ENABLED=false` (default) THEN toàn bộ code path Codex SHALL không hoạt động và không có hồi quy nào với Kiro/Command Code.
3. `.env.example` SHALL được cập nhật với các biến `CHATGPT_*` kèm chú thích.
4. File credentials Codex SHALL có schema rõ ràng (JSON list các account, mỗi account: `accessToken`, `refreshToken`, `chatgptAccountId`, `enabled`) và có file `.example`.
5. Cấu hình SHALL cho phép chọn chiến lược (fill-first | round-robin) cho Codex độc lập với Kiro.

### Requirement 8 — Kiểm thử toàn diện (Paranoid Testing)
**User Story:** Là maintainer, tôi muốn test bao phủ mọi nhánh và tình huống lỗi, theo triết lý AGENTS.md.

#### Acceptance Criteria
1. Test SHALL cô lập mạng hoàn toàn (dùng fixture `block_all_network_calls`); không gọi mạng thật.
2. Test SHALL bao phủ: routing (bật/tắt, khớp/không khớp model), token refresh (thành công, hết hạn, refresh fail), multi-account (round-robin, fill-first, cooldown, all-unavailable, single-account bypass), converters (OpenAI↔Codex, Anthropic↔Codex), streaming + non-streaming (cả 2 API), phân loại lỗi (FATAL/RECOVERABLE, SSE-body error), header `ChatGPT-Account-ID`, và tương thích ngược khi disabled.
3. Test SHALL bao gồm edge case: token thiếu, body rỗng, tool_choice tham chiếu tool không tồn tại, ảnh remote, quota reset timing.
4. Test mới SHALL đặt trong file `test_*.py` phù hợp theo `tests/README.md`, tránh tạo file thừa.

### Requirement 9 — Rủi ro & minh bạch
**User Story:** Là người dùng, tôi muốn được cảnh báo rõ về rủi ro tài khoản khi dùng Codex backend, để ra quyết định có hiểu biết.

#### Acceptance Criteria
1. Tài liệu (README/.env.example) SHALL ghi rõ cảnh báo: dùng Codex backend ngoài Codex CLI chính thức có thể vi phạm ToS OpenAI và tiềm ẩn rủi ro hạn chế/khóa tài khoản.
2. Feature SHALL opt-in (default tắt) để người dùng chủ động bật.
