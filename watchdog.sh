#!/bin/bash
# Trading Bot Watchdog - keeps bot running all day

BOT_CMD="python main.py --dry-run"
PID_FILE="bot.pid"
LOG_FILE="trading_bot.log"
WATCHDOG_LOG="watchdog.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $WATCHDOG_LOG
}

keep_alive() {
    log_msg "Starting Trading Bot watchdog..."

    while true; do
        # Check if market is closed (after 4 PM ET or on weekends)
        HOUR=$(date +%H)
        DOW=$(date +%w)

        # Market closed after 4 PM ET (20:00 UTC)
        if [ "$HOUR" -ge "20" ] || [ "$DOW" = "0" ] || [ "$DOW" = "6" ]; then
            log_msg "Market closed - stopping bot"
            if [ -f "$PID_FILE" ]; then
                kill $(cat "$PID_FILE") 2>/dev/null || true
            fi
            log_msg "Watchdog sleeping until market open"
            sleep 3600
            continue
        fi

        # Check if bot is running
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ! kill -0 "$PID" 2>/dev/null; then
                log_msg "Bot process ($PID) died, restarting..."
                rm -f "$PID_FILE"
            else
                # Bot is running, just sleep
                sleep 30
                continue
            fi
        fi

        # Start the bot
        log_msg "Starting: $BOT_CMD"
        nohup $BOT_CMD >> $LOG_FILE 2>&1 &
        NEW_PID=$!
        echo $NEW_PID > "$PID_FILE"
        log_msg "Bot started with PID: $NEW_PID"

        sleep 10
    done
}

keep_alive
