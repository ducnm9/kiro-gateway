#!/usr/bin/env bash
# podman-run.sh — Build + run Kiro Gateway container (rootless podman).
# Usage: ./podman-run.sh
# Reads keys from .env (never hardcodes secrets). See PODMAN.md for details.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="localhost/kiro-gateway:latest"
CONTAINER="kiro-gateway"
CREDS_MOUNT_SRC="$HOME/.aws/sso/cache"
CREDS_FILE_DEFAULT="/home/kiro/.aws/sso/cache/kiro-auth-token.json"
DEBUG_LOGS_SRC="$(pwd)/debug_logs"
CLI_DATA_SRC="$HOME/.local/share/kiro-cli"

# --- Load helpers ----------------------------------------------------------

# Read a single KEY="value" line from .env (value may be quoted).
env_value() {
    local key="$1"
    grep -E "^[[:space:]]*${key}=" .env 2>/dev/null | head -1 | sed -E "s/^[[:space:]]*${key}=//" | tr -d '"' | tr -d "'"
}

have_env() {
    local key="$1"
    local val
    val="$(env_value "$key")"
    [ -n "$val" ]
}

# --- Validate prerequisites ------------------------------------------------

[ -f .env ] || { echo "ERROR: .env not found in $(pwd)" >&2; exit 1; }

PROXY_API_KEY="${PROXY_API_KEY:-$(env_value PROXY_API_KEY)}"
[ -n "$PROXY_API_KEY" ] || { echo "ERROR: PROXY_API_KEY not set in .env" >&2; exit 1; }

COMMAND_CODE_KEY="$(env_value COMMAND_CODE_API_KEY)"
COMMAND_CODE_ENABLED=false
if [ -n "$COMMAND_CODE_KEY" ]; then
    COMMAND_CODE_ENABLED=true
fi

# ChatGPT (Codex) upstream: opt-in via CHATGPT_ENABLED=true in .env.
# Requires a chatgpt_credentials.json file (JSON list of account tokens),
# mounted read-write so the gateway can persist refreshed tokens.
CHATGPT_ENABLED="$(env_value CHATGPT_ENABLED)"
CHATGPT_CREDS_SRC="$(pwd)/chatgpt_credentials.json"
CHATGPT_CREDS_DST="/app/chatgpt_credentials.json"

# --- Build image if missing ------------------------------------------------

if ! podman image exists "$IMAGE"; then
    echo ">> Building image $IMAGE ..."
    podman build -t kiro-gateway .
else
    echo ">> Image $IMAGE already exists (use 'podman build -t kiro-gateway .' to rebuild)"
fi

# --- Remove old container ----------------------------------------------------

if podman container exists "$CONTAINER"; then
    echo ">> Removing existing container '$CONTAINER' ..."
    podman rm -f "$CONTAINER" >/dev/null
fi

# --- Assemble run arguments ---------------------------------------------------

RUN_ARGS=(
    --name "$CONTAINER"
    --userns=keep-id:uid=999,gid=999
    -p 0.0.0.0:8000:8000
    -e "PROXY_API_KEY=$PROXY_API_KEY"
    -e "COMMAND_CODE_ENABLED=$COMMAND_CODE_ENABLED"
)

if [ "$COMMAND_CODE_ENABLED" = "true" ]; then
    RUN_ARGS+=(-e "COMMAND_CODE_API_KEY=$COMMAND_CODE_KEY")
fi

# ChatGPT (Codex): mount the credentials file + enable the upstream when requested.
if [ "$CHATGPT_ENABLED" = "true" ]; then
    if [ ! -f "$CHATGPT_CREDS_SRC" ]; then
        echo "ERROR: CHATGPT_ENABLED=true but $CHATGPT_CREDS_SRC not found." >&2
        echo "       Create it (see chatgpt_credentials.json.example) or run" >&2
        echo "       scripts/import_codex_auth.py first." >&2
        exit 1
    fi
    RUN_ARGS+=(
        -e "CHATGPT_ENABLED=true"
        -e "CHATGPT_CREDENTIALS_FILE=$CHATGPT_CREDS_DST"
        -v "$CHATGPT_CREDS_SRC:$CHATGPT_CREDS_DST"
    )
    echo ">> ChatGPT (Codex): enabled, mounting $CHATGPT_CREDS_SRC"
fi

# Auth backend: kiro-cli SQLite (default, keeps tokens fresh) or credentials file.
# Force file mode with: KIRO_CREDS_FILE=<path> ./podman-run.sh
if [ -n "${KIRO_CLI_DB_FILE:-}" ] || [ -f "$CLI_DATA_SRC/data.sqlite3" ]; then
    SQLITE_DB="${KIRO_CLI_DB_FILE:-/home/kiro/.local/share/kiro-cli/data.sqlite3}"
    # Remove an explicit "file mode override" set by the user
    KIRO_CREDS_FILE="${KIRO_CREDS_FILE:-}"
    if [ "${KIRO_CREDS_FILE:-unset}" != "unset" ]; then
        echo "WARNING: KIRO_CREDS_FILE is set but SQLite DB exists; using SQLite mode." >&2
        echo "         Unset KIRO_CREDS_FILE or remove the SQLite DB to force file mode." >&2
    fi
    RUN_ARGS+=(
        -e "KIRO_CLI_DB_FILE=$SQLITE_DB"
        -v "$CLI_DATA_SRC:/home/kiro/.local/share/kiro-cli"
    )
    echo ">> Auth: kiro-cli SQLite mode ($SQLITE_DB)"
else
    [ -f "$CREDS_MOUNT_SRC/kiro-auth-token.json" ] || {
        echo "WARNING: $CREDS_MOUNT_SRC/kiro-auth-token.json not found." >&2
        echo "         Generate it first — see PODMAN.md section 5.2." >&2
    }
    KIRO_CREDS_FILE="${KIRO_CREDS_FILE:-$CREDS_FILE_DEFAULT}"
    RUN_ARGS+=(
        -e "KIRO_CREDS_FILE=$KIRO_CREDS_FILE"
        -v "$CREDS_MOUNT_SRC:/home/kiro/.aws/sso/cache:ro"
    )
    echo ">> Auth: credentials file mode ($KIRO_CREDS_FILE)"
fi

RUN_ARGS+=(
    -v "$DEBUG_LOGS_SRC:/app/debug_logs"
    --restart unless-stopped
    "$IMAGE"
)

# --- Start ---------------------------------------------------------------------

echo ">> Starting container '$CONTAINER' ..."
podman run -d "${RUN_ARGS[@]}" >/dev/null

echo ">> Done. Container '$CONTAINER' is running:"
podman ps --filter "name=$CONTAINER" --format "status={{.Status}}  ports={{.Ports}}"

echo
echo ">> Verify:"
echo "   curl http://127.0.0.1:8000/health"
echo "   curl http://127.0.0.1:8000/v1/models -H \"Authorization: Bearer <PROXY_API_KEY>\""
echo
echo "   NOTE: use 127.0.0.1, NOT localhost (pasta binds IPv4 only — see PODMAN.md 5.1)"