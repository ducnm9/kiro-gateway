# Podman Guide — Kiro Gateway

Hướng dẫn chạy Kiro Gateway bằng Podman (rootless). Được tạo sau khi debug thực tế
(2026-09-03): credentials bị Kiro IDE vô hiệu hóa, lỗi `localhost`/IPv6, mount read-only.

> **Nhanh nhất:** chỉ cần chạy `./podman-run.sh` — nó tự build image (nếu chưa có),
> đọc key từ `.env`, tạo lại container với đúng cấu hình đang dùng và khởi động.
> **Mặc định dùng chế độ SQLite** (session kiro-cli tự làm mới token, không bao giờ stale).

---

## 1. Build image

```bash
cd ~/Sites/kiro-gateway
podman build -t kiro-gateway .
```

## 2. Chạy container (lệnh tương đương `./podman-run.sh`)

```bash
podman rm -f kiro-gateway 2>/dev/null   # xóa container cũ nếu có

# --- Chế độ SQLite (MẶC ĐỊNH — khuyến nghị) ---
podman run -d \
  --name kiro-gateway \
  --userns=keep-id:uid=999,gid=999 \
  -p 0.0.0.0:8000:8000 \
  -e PROXY_API_KEY="<PROXY_API_KEY từ .env>" \
  -e KIRO_CLI_DB_FILE=/home/kiro/.local/share/kiro-cli/data.sqlite3 \
  -e COMMAND_CODE_ENABLED=true \
  -e COMMAND_CODE_API_KEY="<COMMAND_CODE_API_KEY từ .env>" \
  -v ~/.local/share/kiro-cli:/home/kiro/.local/share/kiro-cli \
  -v ~/Sites/kiro-gateway/debug_logs:/app/debug_logs \
  --restart unless-stopped \
  localhost/kiro-gateway:latest
```

```bash
# --- Chế độ credentials file (nếu không dùng SQLite) ---
podman run -d \
  --name kiro-gateway \
  --userns=keep-id:uid=999,gid=999 \
  -p 0.0.0.0:8000:8000 \
  -e PROXY_API_KEY="<PROXY_API_KEY từ .env>" \
  -e KIRO_CREDS_FILE=/home/kiro/.aws/sso/cache/kiro-auth-token.json \
  -e COMMAND_CODE_ENABLED=true \
  -e COMMAND_CODE_API_KEY="<COMMAND_CODE_API_KEY từ .env>" \
  -v ~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro \
  -v ~/Sites/kiro-gateway/debug_logs:/app/debug_logs \
  --restart unless-stopped \
  localhost/kiro-gateway:latest
```

**⚠️ `--userns=keep-id:uid=999,gid=999` là BẮT BUỘC.** Container chạy user `kiro`
(uid 999). Không có flag này, uid 999 bị map sang subuid → **không đọc được file
credentials (`Permission denied`) → "Failed to initialize any account"**.

**Giải thích flags quan trọng:**
| Flag | Ý nghĩa |
|------|---------|
| `-p 0.0.0.0:8000:8000` | Bind **IPv4** trên mọi interface (xem mục 5.1) |
| `KIRO_CLI_DB_FILE` | Chế độ SQLite (mặc định). Gateway đọc + tự lưu token vào DB kiro-cli — session luôn fresh |
| `KIRO_CREDS_FILE` | Chế độ credentials file (thay vì SQLite) |
| `COMMAND_CODE_*` | Bật backend Command Code (67 models, tên dạng `provider/model`) |
| `CHATGPT_ENABLED` + mount `chatgpt_credentials.json` | Bật ChatGPT (Codex) upstream; `./podman-run.sh` tự mount file credentials khi `CHATGPT_ENABLED=true` trong `.env`. Model tự discover từ API (Go: `gpt-5.6-luna`/`gpt-5.6-terra`/`gpt-5.4-mini`) |
| `--restart unless-stopped` | Tự động chạy lại sau reboot |

## 3. Kiểm tra sau khi chạy

```bash
# Cấu hình quan trọng: dùng 127.0.0.1, KHÔNG dùng localhost (xem 5.1)
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $(grep -E '^PROXY_API_KEY=' .env | cut -d= -f2)"

# Test Kiro (OpenAI format)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $(grep -E '^PROXY_API_KEY=' .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'

# Test Command Code (model có dấu "/")
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $(grep -E '^PROXY_API_KEY=' .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

## 4. Các lệnh quản lý thường dùng

```bash
podman logs --tail 50 kiro-gateway       # xem log
podman logs -f kiro-gateway              # log theo thời gian thực
podman restart kiro-gateway              # khởi động lại (giữ cấu hình)
podman stop kiro-gateway && podman rm kiro-gateway   # xóa hẳn
podman exec -it kiro-gateway sh          # vào container
```

## 5. Troubleshooting (kinh nghiệm thực tế)

### 5.1 `localhost:8000` không kết nối được — phải dùng `127.0.0.1`

Podman rootless dùng **pasta** làm port forwarder, chỉ bind IPv4 (`0.0.0.0:8000`).
Máy bạn resolve `localhost` → `::1` (IPv6) trước → **connection refused**.

**Luôn dùng** `http://127.0.0.1:8000` (hoặc IP LAN như `192.168.7.200`), không dùng `localhost`.
Mọi client AI (Cursor, Claude Code...): base URL = `http://127.0.0.1:8000`.

Muốn `localhost` hoạt động thì bind thêm IPv6:
```bash
# thêm flag này vào podman run (hoặc mục ports trong docker-compose)
-p "[::]:8000:8000"
```

### 5.2 HTTP 504 "Streaming failed after 3 attempts" trên Kiro = credentials chết

Nguyên nhân đã gặp: **Kiro IDE trên host đăng nhập lại → vô hiệu hóa session cũ và
XÓA luôn `~/.aws/sso/cache/kiro-auth-token.json`**. Gateway còn giữ refresh token
cũ trong memory → OIDC refresh "thành công" nhưng token mới bị Kiro từ chối **403**
→ hết lượt retry → 504.

**Fix:** tạo lại credentials từ session kiro-cli còn sống trong SQLite:

```bash
# 1. Kiểm tra session kiro-cli còn hạn không (xem expires_at)
sqlite3 ~/.local/share/kiro-cli/data.sqlite3 \
  "SELECT value FROM auth_kv WHERE key='kirocli:odic:token';" | python3 -m json.tool | grep expires_at

# 2. Tạo lại file credentials cho gateway (chạy script dưới đây)
#    Lưu ý: expiresAt phải để dạng Z + tối đa 6 chữ số thập phân
#    (Python 3.10 từ chối microsecond 9 chữ số)
```

```python
# regenerate-kiro-creds.py  (chạy trên HOST, không cần container)
import json, os, sqlite3, hashlib

db = os.path.expanduser("~/.local/share/kiro-cli/data.sqlite3")
cache = os.path.expanduser("~/.aws/sso/cache")
con = sqlite3.connect(db)
rows = dict(con.execute("SELECT key, value FROM auth_kv").fetchall())
state = dict(con.execute("SELECT key, value FROM state").fetchall())
tok = json.loads(rows["kirocli:odic:token"])
reg = json.loads(rows["kirocli:odic:device-registration"])
arn = json.loads(state.get("api.codewhisperer.profile", "{}")).get("arn", "")

client_id_hash = hashlib.md5(reg["client_id"].encode()).hexdigest()
with open(os.path.join(cache, f"{client_id_hash}.json"), "w") as f:
    json.dump({"clientId": reg["client_id"], "clientSecret": reg["client_secret"]}, f, indent=2)
with open(os.path.join(cache, "kiro-auth-token.json"), "w") as f:
    json.dump({
        "accessToken": tok["access_token"],
        "refreshToken": tok["refresh_token"],
        "expiresAt": tok["expires_at"].replace("+00:00", "Z")[:26] + "Z",
        "clientIdHash": client_id_hash,
        "authMethod": "IdC",
        "provider": "Enterprise",
        "region": "us-east-1",
        "profileArn": arn,
    }, f, indent=2)
print("OK ->", cache)
```

```bash
# 3. Restart gateway để nạp credentials mới
podman restart kiro-gateway
```

> Nếu `kirocli:odic:token` cũng hết hạn/hỏng → chạy `kiro-cli login` (Identity Center)
> trên host để tạo session mới, rồi lặp lại bước 2.

### 5.3 Mount credentials read-only (`:ro`) = token không lưu lại

Chỉ áp dụng cho **chế độ credentials file**: gateway refresh token OK nhưng
**không ghi được** file (lỗi `[Errno 30] Read-only file system`) → mỗi lần restart
đọc file cũ; nếu refresh token trong file đã bị xoay vòng bởi kiro-cli, lại gặp 403.

✅ **Đã xử lý:** script giờ mặc định dùng **chế độ SQLite** (`KIRO_CLI_DB_FILE`) —
gateway đọc token từ DB kiro-cli và **tự ghi token mới ngược vào DB** (write-back
Read-Merge-Write, `SQLITE_READONLY` mặc định tắt). Không cần làm gì thêm: chạy
`./podman-run.sh` là xong.

Nếu vẫn muốn dùng chế độ file: bỏ `:ro` khỏi mount cache để gateway tự lưu token.

### 5.4 HTTP 401 "Invalid or missing API Key"

Sai/missing `PROXY_API_KEY` khi gọi. Kiểm tra:
```bash
grep -E '^PROXY_API_KEY=' .env   # key trong file
podman exec kiro-gateway printenv PROXY_API_KEY   # key trong container (phải khớp)
```

### 5.5 Command Code: 403 `MODEL_NOT_IN_PLAN`

Model không nằm trong plan (VD `google/gemini-3.8-flash` cần plan GOAT+).
Xem danh sách model khả dụng trước khi dùng:
```bash
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer <key>" | python3 -m json.tool
```

### 5.6 Debug sâu

```bash
# Chạy container với debug mode (lưu request/response lỗi vào debug_logs/)
podman run -d ... -e DEBUG_MODE=errors localhost/kiro-gateway:latest
ls ~/Sites/kiro-gateway/debug_logs/
```

---

*Xem thêm: `README.md` (tài liệu chính), `docker-compose.yml` (cấu hình compose tương đương).*