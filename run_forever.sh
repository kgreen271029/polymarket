#!/usr/bin/env bash
# run_forever.sh — Keeps the trading bot running 24/7.
# Restarts automatically on crash with exponential backoff.
# Usage: bash run_forever.sh
#   or:  nohup bash run_forever.sh >> logs/restart.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "[LAUNCHER] Starting bot at $(date)" | tee -a logs/restart.log

RESTART_DELAY=5
MAX_DELAY=300   # cap backoff at 5 minutes
CONSECUTIVE_CRASHES=0
CRASH_RESET_SECONDS=3600  # reset crash counter if uptime > 1h

while true; do
    START_TS=$(date +%s)
    echo "[LAUNCHER] Launching main.py at $(date)" | tee -a logs/restart.log

    python main.py || true   # never let a non-zero exit kill the wrapper

    EXIT_CODE=$?
    END_TS=$(date +%s)
    UPTIME=$(( END_TS - START_TS ))

    echo "[LAUNCHER] Bot exited (code=$EXIT_CODE, uptime=${UPTIME}s) at $(date)" \
        | tee -a logs/restart.log

    if [ "$UPTIME" -gt "$CRASH_RESET_SECONDS" ]; then
        # Ran for over an hour — reset backoff
        CONSECUTIVE_CRASHES=0
        RESTART_DELAY=5
    else
        CONSECUTIVE_CRASHES=$(( CONSECUTIVE_CRASHES + 1 ))
        RESTART_DELAY=$(( RESTART_DELAY * 2 ))
        if [ "$RESTART_DELAY" -gt "$MAX_DELAY" ]; then
            RESTART_DELAY=$MAX_DELAY
        fi
    fi

    echo "[LAUNCHER] Restarting in ${RESTART_DELAY}s (crash #${CONSECUTIVE_CRASHES})..." \
        | tee -a logs/restart.log
    sleep "$RESTART_DELAY"
done
