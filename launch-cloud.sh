#!/bin/bash

ENV_FILE="$HOME/elin-project/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "No .env file found. Create one with your GROQ_API_KEY:"
    echo "  echo 'GROQ_API_KEY=\"gsk_your_key\"' > ~/elin-project/.env"
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "gsk_your_key_here" ]; then
    echo "Set your GROQ_API_KEY in ~/elin-project/.env"
    exit 1
fi

cd ~/elin-project/searxng-docker && docker compose up -d
cd ~/elin-project

python3 telegram_bridge.py > telegram.log 2>&1 &
BRIDGE_PID=$!

echo "starting elin in cloud mode..."

sudo GROQ_API_KEY="$GROQ_API_KEY" ELIN_MODE="cloud" python3 elin.py

kill $BRIDGE_PID
cd ~/elin-project/searxng-docker && docker compose down
