#!/bin/bash

cd ~/elin-project/searxng-docker && docker compose up -d
cd ~/elin-project

~/elin-project/llama.cpp/build/bin/llama-server \
    -m ~/models/qwen/Qwen3.5-9B-Q5_K_M.gguf \
    --port 8081 \
    --ctx-size 16384 \
    -ngl 99 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --reasoning auto \
    --chat-template-kwargs '{"enable_thinking":true, "preserve_thinking":true}'  >/dev/null 2>&1 &

SERVER_PID=$!

echo "waiting for elin's model to load..."
while ! curl -s http://localhost:8081/v1/models > /dev/null; do
    sleep 2
done

python3 telegram_bridge.py > telegram.log 2>&1 &
BRIDGE_PID=$!

python3 elin.py

kill $SERVER_PID
kill $BRIDGE_PID
cd ~/elin-project/searxng-docker && docker compose down
