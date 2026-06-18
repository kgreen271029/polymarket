#!/usr/bin/env bash
# run_forever.sh — Keeps the trading bot running 24/7.
# Restarts automatically on crash with exponential backoff.
# Usage: bash run_forever.sh
#   or:  nohup bash run_forever.sh >> logs/restart.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

log() { echo "[LAUNCHER] $*" | tee -a logs/restart.log; }

log "Starting bot at $(date)"

RESTART_DELAY=5
MAX_DELAY=300         # cap backoff at 5 minutes
CONSECUTIVE_CRASHES=0
CRASH_RESET_SECONDS=3600  # reset crash counter if uptime > 1h

while true; do
    START_TS=$(date +%s)
    log "Launching main.py at $(date)"

    # Capture real exit code without letting a non-zero exit kill this wrapper
    set +e
    python main.py
    EXIT_CODE=$?
    set -e

    END_TS=$(date +%s)
    UPTIME=$(( END_TS - START_TS ))

    log "Bot exited (code=$EXIT_CODE, uptime=${UPTIME}s) at $(date)"

    if [ "$UPTIME" -gt "$CRASH_RESET_SECONDS" ]; then
        # Ran for over an hour — treat as healthy restart, reset backoff
        CONSECUTIVE_CRASHES=0
        RESTART_DELAY=5
        log "Uptime > 1h — resetting backoff to ${RESTART_DELAY}s"
    else
        CONSECUTIVE_CRASHES=$(( CONSECUTIVE_CRASHES + 1 ))
        RESTART_DELAY=$(( RESTART_DELAY * 2 ))
        if [ "$RESTART_DELAY" -gt "$MAX_DELAY" ]; then
            RESTART_DELAY=$MAX_DELAY
        fi
    fi

    log "Restarting in ${RESTART_DELAY}s (crash #${CONSECUTIVE_CRASHES}, exit=$EXIT_CODE)..."
    sleep "$RESTART_DELAY"
done
