"""
strategies/short_scanner.py — Bearish short-selling signal scanner.

Robinhood allows short selling with $2k+ margin account (no $25k minimum).
PDT rule eliminated June 4, 2026. Shorts are valid swing trades.

Short setup criteria (Minervini Stage 3/4 breakdown + mean reversion):
  1. RSI Mean Reversion Short: RSI > 75 + price >5% above 20-SMA + MACD bearish
  2. Stage-3 Distribution: price < 50-SMA (after being above it) + volume spike
  3. Death Cross Breakdown: 50-SMA crosses below 200-SMA + close below 50-SMA
  4. Overbought Extension: BB% > 0.95 (price at top of band) + RSI > 72

Stop: 1.5×ATR above entry (shorts have unlimited risk so tight stops are critical)
Target: 2.5×ATR below entry (1.67:1 R:R minimum)
Max short position: 15% of cash, max hold 7 days (shorts can gap up violently)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer
from analysis.filters import (
    earnings_blackout,
    sector_momentum_ok,
    volatility_regime_ok,
)
from analysis.regime import detect_regime

# Liquid large-caps only — must be shortable on Robinhood
SHORT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA",
    "AMD", "PLTR", "SPY", "QQQ",
]

LOOP_INTERVAL  = 900   # 15 min (same as SwingTrader)
MAX_SHORTS     = 1     # never hold more than 1 short at a time
SHORT_ATR_STOP = 1.5   # tight — shorts can gap up
SHORT_ATR_TGT  = 2.5   # 1.67:1 R:R
MAX_HOLD_DAYS  = 7     # shorts are more time-sensitive than longs
MAX_POSITION_PCT = 0.15  # 15% of cash per short


class ShortScanner(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        robinhood_broker,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._rh = robinhood_broker
        self._portfolio = portfolio_tracker

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._market_open():
            return []

        # Only short in bear or choppy regimes — never fight a bull market
        regime = await asyncio.to_thread(detect_regime)
        if regime.name == "bull" and regime.size_mult >= 1.0:
            logger.debug("[Short] Full bull regime — no shorts")
            return []

        # Volatility check: skip in extremely high vol (shorts gap up more)
        regime_ok, reason = await asyncio.to_thread(volatility_regime_ok)

        signals: list[TradeSignal] = []
        open_shorts = sum(
            1 for pos in self._portfolio.positions.values()
            if getattr(pos, "side", "") == "short"
        )
        if open_shorts >= MAX_SHORTS:
            return []

        cash = self._portfolio.get_available_capital()
        if cash < 10.0:
            return []

        open_syms = {p.split(":")[0] for p in self._portfolio.positions}

        for symbol in SHORT_WATCHLIST:
            if symbol in open_syms:
                continue
            try:
                # Earnings blackout — critical for shorts (earnings can gap +20%)
                blocked, _ = await asyncio.to_thread(earnings_blackout, symbol, window_days=21)
                if blocked:
                    logger.debug("[Short] {} near earnings — skip", symbol)
                    continue

                df = await self._market_data.get_bars(symbol, "1Day", limit=60)
                if df is None or len(df) < 30:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                if not s:
                    continue

                price      = s.get("latest_close", 0)
                rsi        = s.get("rsi", 50)
                sma20      = s.get("sma_20", price)
                sma50      = s.get("sma_50", price)
                atr        = s.get("atr", price * 0.02)
                bb_pct     = s.get("bb_pct", 0.5)
                macd_dir   = s.get("macd_direction", "neutral")
                vol_ratio  = s.get("volume_ratio", 1.0)
                trend      = s.get("trend", "sideways")
                pct_vs_sma = s.get("price_vs_sma20_pct", 0)

                if not price or not atr:
                    continue

                # ── Setup 1: RSI Mean Reversion Short ────────────────────
                # Classic overbought + overextended; mean reversion back to SMA20
                rsi_short = (
                    rsi > 75
                    and pct_vs_sma > 5.0         # >5% above 20-SMA
                    and macd_dir == "bearish"     # MACD histogram turning down
                    and vol_ratio < 1.5           # fading volume on extension
                )

                # ── Setup 2: Stage-3 Distribution Breakdown ───────────────
                # Price was above 50-SMA, now breaking below it on high volume
                stage3_short = (
                    price < sma50                 # fell below 50-day
                    and pct_vs_sma < -1.5         # below 20-SMA too
                    and vol_ratio > 1.8           # volume confirming breakdown
                    and trend == "downtrend"
                    and rsi < 50                  # momentum fading
                )

                # ── Setup 3: BB Upper Band Rejection ─────────────────────
                # Price spiked to upper BB but MACD diverging (classic exhaustion)
                bb_rejection = (
                    bb_pct > 0.92                 # at/above upper BB
                    and rsi > 72
                    and macd_dir == "bearish"
                )

                short_ok = rsi_short or stage3_short or bb_rejection
                if not short_ok:
                    continue

                setup_name = (
                    "RSI_mean_rev" if rsi_short else
                    "stage3_breakdown" if stage3_short else "bb_rejection"
                )

                # Stop above entry + 1.5×ATR (tight — short risk is unlimited)
                stop   = round(price + SHORT_ATR_STOP * atr, 2)
                target = round(price - SHORT_ATR_TGT * atr, 2)

                # Size: 15% of cash, capped by stop distance
                risk_per_share = stop - price
                if risk_per_share <= 0 or target >= price:
                    continue

                max_dollar = cash * MAX_POSITION_PCT
                qty = round(min(max_dollar / price, max_dollar / risk_per_share), 6)
                if qty * price < 1.0:
                    continue

                # Require at least 1.5:1 R:R
                reward = price - target
                if reward / risk_per_share < 1.5:
                    continue

                logger.info("[Short] {} — {} | RSI={:.0f} pct_sma={:.1f}% setup={}",
                            symbol, "SHORT", rsi, pct_vs_sma, setup_name)

                signals.append(TradeSignal(
                    symbol=symbol,
                    side="sell",          # "sell" = short entry on Robinhood
                    asset_class="stock",
                    strategy_name=f"{self.name}:{setup_name}",
                    entry_price=price,
                    stop_price=stop,
                    take_profit=target,
                    confidence="Medium",
                    reasoning=(
                        f"Short {setup_name}: RSI={rsi:.0f}, "
                        f"pct_sma={pct_vs_sma:+.1f}%, vol={vol_ratio:.1f}x, "
                        f"R:R={reward/risk_per_share:.1f}x"
                    ),
                    qty=qty,
                ))

                if signals:
                    break  # take one short at a time

            except Exception as e:
                logger.error("[Short] {} error: {}", symbol, e)

        return signals

    async def _check_short_exits(self) -> None:
        """Exit short positions on stop, target, or max hold."""
        if not self._market_open():
            return
        for key, pos in list(self._portfolio.positions.items()):
            if getattr(pos, "side", "") != "short":
                continue
            sym = pos.symbol
            try:
                price = await self._rh.get_quote_price(sym)
                if not price:
                    continue
                price = float(price)

                hold_days = (datetime.now(tz=timezone.utc) - pos.opened_at).days
                reason = None

                if pos.stop_loss and price >= pos.stop_loss:
                    reason = f"short stop hit @ ${price:.2f} (stop=${pos.stop_loss:.2f})"
                elif pos.take_profit and price <= pos.take_profit:
                    reason = f"short target hit @ ${price:.2f}"
                elif hold_days >= MAX_HOLD_DAYS:
                    reason = f"short max hold ({hold_days}d) exit @ ${price:.2f}"

                if reason:
                    logger.info("[Short] COVER {} — {}", sym, reason)
                    await self._signal_bus.put(TradeSignal(
                        symbol=sym,
                        side="buy",     # cover = buy to close short
                        asset_class="stock",
                        strategy_name=self.name,
                        entry_price=price,
                        stop_price=None,
                        take_profit=None,
                        confidence="High",
                        reasoning=reason,
                        qty=pos.qty,
                    ))
            except Exception as e:
                logger.error("[Short] exit check {}: {}", sym, e)

    async def should_exit(self, position) -> bool:
        return False  # handled by _check_short_exits

    async def run(self) -> None:
        self._running = True
        logger.info("[Short] Started — scanning for bearish setups every {}s", LOOP_INTERVAL)
        while self._running:
            try:
                await self._check_short_exits()
                signals = await self.generate_signals()
                for sig in signals:
                    await self._signal_bus.put(sig)
                    logger.info("[Short] → {} {} @ ${:.2f} stop=${:.2f} target=${:.2f}",
                                sig.side.upper(), sig.symbol, sig.entry_price,
                                sig.stop_price or 0, sig.take_profit or 0)
            except Exception as exc:
                logger.error("[Short] run error: {}", exc)
            await asyncio.sleep(LOOP_INTERVAL)
