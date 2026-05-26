#!/bin/bash

ENV_FILE="$HOME/elin-project/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "No .env file found. Create one with your GITHUB_TOKEN:"
    echo "  echo 'GITHUB_TOKEN=\"github_pat_...\"' > ~/elin-project/.env"
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

if [ -z "$GITHUB_TOKEN" ] || [ "$GITHUB_TOKEN" = "github_pat_your_token_here" ]; then
    echo "Set your GITHUB_TOKEN in ~/elin-project/.env"
    echo "Get one at https://github.com/settings/tokens (fine-grained, models:read scope)"
    exit 1
fi

cd ~/elin-project/searxng-docker && docker compose up -d
cd ~/elin-project

python3 telegram_bridge.py > telegram.log 2>&1 &
BRIDGE_PID=$!

echo "starting elin in github models cloud mode..."

ELIN_MODE="github" python3 elin.py

kill $BRIDGE_PID
cd ~/elin-project/searxng-docker && docker compose down
