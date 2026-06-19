#!/bin/bash
# Real-time trading bot monitoring dashboard

clear

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🤖 TRADING BOT LIVE MONITORING DASHBOARD            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           🤖 TRADING BOT LIVE MONITORING DASHBOARD            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # System Status
    echo "⏰ Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Process Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 PROCESS STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f bot.pid ] && kill -0 $(cat bot.pid) 2>/dev/null; then
        BOT_PID=$(cat bot.pid)
        BOT_MEM=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $6 "KB"}')
        BOT_CPU=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $3 "%"}')
        echo "✅ Trading Bot: RUNNING (PID: $BOT_PID)"
        echo "   Memory: $BOT_MEM | CPU: $BOT_CPU"
    else
        echo "❌ Trading Bot: STOPPED"
    fi

    if [ -f watchdog.pid ] && kill -0 $(cat watchdog.pid) 2>/dev/null; then
        WD_PID=$(cat watchdog.pid)
        echo "✅ Watchdog:    RUNNING (PID: $WD_PID)"
    else
        echo "❌ Watchdog:    STOPPED"
    fi

    echo ""

    # Log Analysis
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 LATEST TRADING ACTIVITY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f trading_bot.log ]; then
        echo "Last 8 log entries:"
        tail -8 trading_bot.log | sed 's/^/  /'
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 STATISTICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f trading_bot.log ]; then
        ANALYSIS_COUNT=$(grep -c "analyze_and_trade" trading_bot.log)
        BUY_ATTEMPTS=$(grep -c "BUY" trading_bot.log)
        SELL_ATTEMPTS=$(grep -c "SELL" trading_bot.log)
        ERRORS=$(grep -c "ERROR" trading_bot.log)

        echo "Total Analyses: $ANALYSIS_COUNT"
        echo "Buy Signals:    $BUY_ATTEMPTS"
        echo "Sell Signals:   $SELL_ATTEMPTS"
        echo "Errors:         $ERRORS"
    fi

    echo ""
    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)"
    sleep 10
done
