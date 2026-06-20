"""Market scanning — evaluate watchlist stocks and manage existing positions."""
import logging
from bot import robinhood, llm, risk, telegram
from bot.config import WATCHLIST, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger(__name__)


def run_scan(dry_run: bool, starting_equity: float) -> dict:
    """
    One full scan cycle:
    1. Check risk limits
    2. Review existing positions (stop loss / take profit / LLM sell signal)
    3. Scan watchlist for new buy opportunities

    Returns summary dict with actions taken.
    """
    account = robinhood.get_account_info()
    equity = account["equity"]
    buying_power = account["buying_power"]
    positions = robinhood.get_positions()

    logger.info(
        f"Scan start | Equity=${equity:.2f} | Cash=${buying_power:.2f} | "
        f"Positions={len(positions)}"
    )

    actions_taken = []

    # --- Check daily loss limit ---
    if risk.check_daily_loss_limit(equity, starting_equity):
        msg = f"Daily loss limit reached. Halting trading. Equity=${equity:.2f}"
        logger.warning(msg)
        telegram.send(f"🛑 {msg}")
        return {"halted": True, "reason": "daily_loss_limit", "actions": actions_taken}

    # --- Manage existing positions ---
    for pos in positions:
        symbol = pos["symbol"]
        try:
            _manage_position(pos, account, dry_run, actions_taken)
        except Exception as e:
            logger.error(f"Error managing position {symbol}: {e}")

    # Refresh positions after potential sells
    positions = robinhood.get_positions()
    account = robinhood.get_account_info()
    buying_power = account["buying_power"]

    # --- Scan watchlist for buys ---
    if not risk.can_open_position(positions):
        logger.info("Max positions reached, skipping watchlist scan")
        return {"halted": False, "actions": actions_taken}

    if buying_power < 1.0:
        logger.info("Insufficient buying power, skipping watchlist scan")
        return {"halted": False, "actions": actions_taken}

    already_held = {p["symbol"] for p in positions}

    for symbol in WATCHLIST:
        if symbol in already_held:
            continue
        if not risk.can_open_position(positions):
            break
        try:
            result = _evaluate_buy(symbol, account, buying_power, dry_run)
            if result:
                actions_taken.append(result)
                positions = robinhood.get_positions()
                account = robinhood.get_account_info()
                buying_power = account["buying_power"]
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {e}")

    return {"halted": False, "actions": actions_taken}


def _manage_position(pos: dict, account: dict, dry_run: bool, actions: list):
    symbol = pos["symbol"]
    qty = pos["qty"]

    # Hard stop loss / take profit checks first
    if risk.should_stop_loss(pos, STOP_LOSS_PCT):
        order = robinhood.place_market_sell(symbol, qty, dry_run=dry_run)
        msg = f"Stop loss hit on {symbol} ({pos['pnl_pct']:.1f}%)"
        logger.info(msg)
        telegram.send_trade_alert("SELL", symbol, qty * pos["current_price"], pos["current_price"], msg)
        actions.append({"action": "SELL", "symbol": symbol, "reason": "stop_loss", "order": order})
        return

    if risk.should_take_profit(pos, TAKE_PROFIT_PCT):
        order = robinhood.place_market_sell(symbol, qty, dry_run=dry_run)
        msg = f"Take profit hit on {symbol} ({pos['pnl_pct']:.1f}%)"
        logger.info(msg)
        telegram.send_trade_alert("SELL", symbol, qty * pos["current_price"], pos["current_price"], msg)
        actions.append({"action": "SELL", "symbol": symbol, "reason": "take_profit", "order": order})
        return

    # Ask LLM if we should sell
    candles = robinhood.get_historical(symbol)
    news = robinhood.get_news(symbol)
    rec = llm.analyze_stock(
        symbol=symbol,
        current_price=pos["current_price"],
        candles=candles,
        news=news,
        existing_position=pos,
        account_equity=account["equity"],
    )

    if rec["action"] == "SELL" and rec["confidence"] >= 60:
        order = robinhood.place_market_sell(symbol, qty, dry_run=dry_run)
        telegram.send_trade_alert(
            "SELL", symbol, qty * pos["current_price"],
            pos["current_price"], rec["reasoning"]
        )
        actions.append({"action": "SELL", "symbol": symbol, "reason": "llm", "rec": rec, "order": order})


def _evaluate_buy(symbol: str, account: dict, buying_power: float, dry_run: bool) -> dict | None:
    candles = robinhood.get_historical(symbol)
    if not candles:
        return None

    quote = robinhood.get_quote(symbol)
    price = quote["price"]
    if price <= 0:
        return None

    news = robinhood.get_news(symbol)
    rec = llm.analyze_stock(
        symbol=symbol,
        current_price=price,
        candles=candles,
        news=news,
        account_equity=account["equity"],
    )

    if rec["action"] != "BUY":
        return None

    dollars = risk.calc_position_size(account["equity"], buying_power, rec["confidence"])
    if dollars < 1.0:
        logger.info(f"Skipping {symbol}: position size too small (${dollars:.2f})")
        return None

    order = robinhood.place_market_buy(symbol, dollars, dry_run=dry_run)
    telegram.send_trade_alert("BUY", symbol, dollars, price, rec["reasoning"])
    return {"action": "BUY", "symbol": symbol, "dollars": dollars, "rec": rec, "order": order}
