#!/bin/bash
set -e

# Default: normal TUI
TUI_SCRIPT="elin_tui.py"

# Load API keys from .env
ENV_FILE="$HOME/elin-project/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Optional argument handling:
if [[ $# -gt 0 ]]; then
    if [[ "$1" =~ ^-s([0-9]+)$ ]]; then
        TUI_SCRIPT="elin_tui${BASH_REMATCH[1]}.py"
    else
        echo "Usage: $0 [-sN]"
        echo "Example: $0 -s1  -> runs elin_tui1.py"
        exit 1
    fi
fi

cleanup() {
    echo "Cleaning up processes..."
    if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    if [[ -n "${BRIDGE_PID:-}" ]]; then kill "$BRIDGE_PID" 2>/dev/null || true; fi
    cd ~/elin-project/searxng-docker && docker compose down || true
}
trap cleanup EXIT INT TERM

cd ~/elin-project/searxng-docker
docker compose up -d

cd ~/elin-project

# Fixed: Removed 'on' from --flash-attn, and added trailing '&' to background it
~/elin-project/llama.cpp/build/bin/llama-server \
    -m ~/models/qwen/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8081 \
    --ctx-size 65536 \
    -ngl 12 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --mmap \
    --reasoning auto \
    --chat-template-kwargs '{"enable_thinking":true, "preserve_thinking":true}' >/dev/null 2>&1 &

SERVER_PID=$!

echo "Waiting for elin's model to load..."
while [[ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8081/v1/models)" != "200" ]]; do
    sleep 2
done
echo "Model loaded successfully!"

python3 telegram_bridge.py > telegram.log 2>&1 &
BRIDGE_PID=$!

# Run TUI in the foreground so you can interact with it
python3 "$TUI_SCRIPT"
