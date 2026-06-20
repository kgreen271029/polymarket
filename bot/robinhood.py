import logging
import robin_stocks.robinhood as rh
from bot.config import ROBINHOOD_MCP_TOKEN, ROBINHOOD_REFRESH_TOKEN, ROBINHOOD_CLIENT_ID

logger = logging.getLogger(__name__)


def login():
    """Authenticate with Robinhood using stored OAuth tokens."""
    try:
        rh.authentication.login(
            username=None,
            password=None,
            expiresIn=86400,
            scope="internal",
            by_sms=False,
            store_session=False,
            mfa_code=None,
            pickle_name="",
        )
        # Inject the tokens directly after the session is created
        rh.helper.set_login_state(True)
        rh.helper.update_session("Authorization", f"Bearer {ROBINHOOD_MCP_TOKEN}")
        logger.info("Robinhood login successful (token auth)")
        return True
    except Exception as e:
        logger.error(f"Robinhood login failed: {e}")
        return False


def get_account_info() -> dict:
    """Return cash, equity, and buying power."""
    try:
        profile = rh.profiles.load_portfolio_profile()
        account = rh.profiles.load_account_profile()
        return {
            "equity": float(profile.get("equity", 0)),
            "cash": float(account.get("cash", 0)),
            "buying_power": float(account.get("buying_power", 0)),
            "extended_hours_equity": float(profile.get("extended_hours_equity") or 0),
        }
    except Exception as e:
        logger.error(f"get_account_info failed: {e}")
        return {"equity": 0, "cash": 0, "buying_power": 0, "extended_hours_equity": 0}


def get_positions() -> list[dict]:
    """Return all open positions with symbol, qty, avg cost, current price, P&L."""
    try:
        positions = rh.account.get_open_stock_positions()
        results = []
        for pos in positions:
            instrument_url = pos.get("instrument")
            symbol = rh.stocks.get_symbol_by_url(instrument_url)
            qty = float(pos.get("quantity", 0))
            avg_cost = float(pos.get("average_buy_price", 0))
            quotes = rh.stocks.get_latest_price(symbol)
            current_price = float(quotes[0]) if quotes else avg_cost
            pnl = (current_price - avg_cost) * qty
            pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost else 0
            results.append({
                "symbol": symbol,
                "qty": qty,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        return results
    except Exception as e:
        logger.error(f"get_positions failed: {e}")
        return []


def get_quote(symbol: str) -> dict:
    """Get latest price + basic quote info for a symbol."""
    try:
        q = rh.stocks.get_quotes(symbol)[0]
        return {
            "symbol": symbol,
            "price": float(q.get("last_trade_price") or q.get("last_extended_hours_trade_price") or 0),
            "ask": float(q.get("ask_price") or 0),
            "bid": float(q.get("bid_price") or 0),
            "volume": int(float(q.get("last_trade_size") or 0)),
        }
    except Exception as e:
        logger.error(f"get_quote({symbol}) failed: {e}")
        return {"symbol": symbol, "price": 0, "ask": 0, "bid": 0, "volume": 0}


def get_historical(symbol: str, interval: str = "5minute", span: str = "day") -> list[dict]:
    """Return OHLCV candles as list of dicts."""
    try:
        data = rh.stocks.get_stock_historicals(symbol, interval=interval, span=span)
        return data or []
    except Exception as e:
        logger.error(f"get_historical({symbol}) failed: {e}")
        return []


def get_news(symbol: str) -> list[str]:
    """Return up to 5 recent news headlines for a symbol."""
    try:
        news = rh.stocks.get_news(symbol)
        return [n.get("title", "") for n in (news or [])[:5]]
    except Exception as e:
        logger.warning(f"get_news({symbol}) failed: {e}")
        return []


def place_market_buy(symbol: str, dollars: float, dry_run: bool = False) -> dict:
    """Buy $dollars worth of symbol at market. Returns order dict."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}BUY ${dollars:.2f} of {symbol}")
    if dry_run:
        return {"state": "dry_run", "symbol": symbol, "dollars": dollars}
    try:
        order = rh.orders.order_buy_fractional_by_price(
            symbol, dollars, timeInForce="gfd", extendedHours=False
        )
        return order or {}
    except Exception as e:
        logger.error(f"place_market_buy({symbol}, ${dollars}) failed: {e}")
        return {}


def place_market_sell(symbol: str, qty: float, dry_run: bool = False) -> dict:
    """Sell qty shares of symbol at market. Returns order dict."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}SELL {qty} shares of {symbol}")
    if dry_run:
        return {"state": "dry_run", "symbol": symbol, "qty": qty}
    try:
        order = rh.orders.order_sell_fractional_by_quantity(
            symbol, qty, timeInForce="gfd", extendedHours=False
        )
        return order or {}
    except Exception as e:
        logger.error(f"place_market_sell({symbol}, {qty} shares) failed: {e}")
        return {}


def get_todays_orders() -> list[dict]:
    """Return all orders placed today."""
    try:
        return rh.orders.get_all_stock_orders() or []
    except Exception as e:
        logger.error(f"get_todays_orders failed: {e}")
        return []
