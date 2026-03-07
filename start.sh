#!/bin/bash
# Local development startup script for Sleep BCI API
# Starts Redis (if REDIS_URL is configured) then uvicorn

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Defaults
HOST="0.0.0.0"
PORT="8000"
REDIS_URL=""

# --- Step 1: Load .env ---
if [ ! -f "$ENV_FILE" ]; then
    echo "WARNING: .env not found at $ENV_FILE — using defaults (in-memory job store)"
else
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip blank lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            REDIS_URL) REDIS_URL="$value" ;;
            HOST)      HOST="$value" ;;
            PORT)      PORT="$value" ;;
        esac
    done < "$ENV_FILE"
fi

# --- Step 2: Validate REDIS_URL ---
if [ -z "$REDIS_URL" ]; then
    echo "WARNING: REDIS_URL is not set — jobs will use in-memory store (lost on restart)"
else
    if [[ "$REDIS_URL" != redis://* && "$REDIS_URL" != rediss://* ]]; then
        echo "ERROR: REDIS_URL is malformed: '$REDIS_URL'"
        echo "       Must start with redis:// or rediss://"
        exit 1
    fi

    # --- Step 3: Check/start Redis ---
    if redis-cli -u "$REDIS_URL" ping 2>/dev/null | grep -q "PONG"; then
        echo "Redis is already running at $REDIS_URL"
    else
        echo "Redis not reachable at $REDIS_URL — attempting to start redis-server..."
        if ! command -v redis-server &>/dev/null; then
            echo "ERROR: redis-server not found. Install it with:"
            echo "  macOS:   brew install redis"
            echo "  Ubuntu:  sudo apt-get install redis-server"
            exit 1
        fi
        redis-server --daemonize yes
        # Give it a moment to start
        sleep 1
        if redis-cli -u "$REDIS_URL" ping 2>/dev/null | grep -q "PONG"; then
            echo "Redis started successfully"
        else
            echo "ERROR: redis-server started but cannot reach $REDIS_URL"
            echo "       Check that the host/port in REDIS_URL matches the server config"
            exit 1
        fi
    fi
fi

# --- Step 4: Start uvicorn ---
echo ""
echo "Starting Sleep BCI API on http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""

exec uvicorn src.sleep_bci.api.app:app --reload --host "$HOST" --port "$PORT"
