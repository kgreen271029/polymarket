"""strategies/swing_trader.py — Multi-day stock swing trades via Robinhood MCP (PDT-safe)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer
from analysis.filters import (
    earnings_blackout,
    sector_momentum_ok,
    volatility_regime_ok,
    kelly_position_size,
    atr_trailing_stop,
    bearish_divergence,
    top_sectors,
    symbol_in_top_sectors,
)
from analysis.metrics import vix_kelly_fraction, get_adaptive_stats
from analysis.multifactor import chandelier_exit, score_symbol, volatility_contraction
from analysis.regime import detect_regime

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ",
    "AMZN", "META", "GOOGL", "AMD", "COIN", "MSTR",
    "ARM", "JPM", "GS", "PLTR",
]
LOOP_INTERVAL = 900


class SwingTrader(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        robinhood_broker,
        ai_analyzer,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._rh = robinhood_broker
        self._ai = ai_analyzer
        self._portfolio = portfolio_tracker
        self._top_sectors: list[str] = []
        self._sector_refresh_count = 0

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._market_open():
            return []

        cash = self._portfolio.get_available_capital()
        if cash < 5.0:
            return []   # not enough cash to open anything

        # Refresh top sectors every 4 loops (~1 hour)
        self._sector_refresh_count += 1
        if self._sector_refresh_count % 4 == 1:
            self._top_sectors = await asyncio.to_thread(top_sectors, 3)
            logger.info("[SwingTrader] Top sectors: {}", self._top_sectors)

        # Volatility regime check (skip choppy days)
        regime_ok, regime_reason = await asyncio.to_thread(volatility_regime_ok)
        if not regime_ok:
            logger.info("[SwingTrader] Skipping — {}", regime_reason)
            return []

        # Master regime gate — scales size and can halt new entries
        regime = await asyncio.to_thread(detect_regime)
        if not regime.tradeable:
            logger.info("[SwingTrader] Regime {} — standing aside ({})", regime.name, regime.detail)
            return []
        logger.info("[SwingTrader] Regime: {} (size x{:.1f}) — {}",
                    regime.name, regime.size_mult, regime.detail)

        # VIX-based Kelly fraction + adaptive win-rate stats
        kelly_frac, vix_note = await asyncio.to_thread(vix_kelly_fraction)
        if kelly_frac <= 0:
            logger.info("[SwingTrader] {} — no new trades", vix_note)
            return []
        # Combine VIX Kelly fraction with regime size multiplier
        kelly_frac *= regime.size_mult
        stats = get_adaptive_stats()
        if stats["adaptive"]:
            logger.info("[SwingTrader] Adaptive stats: win_rate={:.0%} R:R={:.2f} over {} trades",
                        stats["win_rate"], stats["avg_win_loss_ratio"], stats["n_trades"])

        signals: list[TradeSignal] = []
        open_syms = {p.split(":")[0] for p in self._portfolio.positions}
        open_stocks = sum(1 for p in self._portfolio.positions.values()
                         if getattr(p, "asset_class", "") == "stock")
        if open_stocks >= 2:
            return signals

        for symbol in WATCHLIST:
            if symbol in open_syms:
                continue
            try:
                # — Sector momentum filter —
                sec_ok, sec_reason = await asyncio.to_thread(sector_momentum_ok, symbol)
                if not sec_ok:
                    logger.debug("[SwingTrader] {} skipped: {}", symbol, sec_reason)
                    continue

                # — Sector rotation: prefer top-performing sectors —
                if self._top_sectors and not symbol_in_top_sectors(symbol, self._top_sectors):
                    logger.debug("[SwingTrader] {} not in top sectors", symbol)
                    continue

                # — Earnings blackout —
                blocked, earn_reason = await asyncio.to_thread(earnings_blackout, symbol)
                if blocked:
                    logger.info("[SwingTrader] {} blocked: {}", symbol, earn_reason)
                    continue

                df = await self._market_data.get_bars(symbol, "1Day", limit=60)
                if df is None or len(df) < 30:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                if not s:
                    continue

                price        = s.get("latest_close", 0)
                rsi          = s.get("rsi", 50)
                sma50        = s.get("sma_50", price)
                volume_ratio = s.get("volume_ratio", 1.0)
                pct_vs_sma   = s.get("price_vs_sma20_pct", 0)
                atr          = s.get("atr", price * 0.02)

                if not (
                    price > 0
                    and price > sma50                  # Stage-2: above 50-day trend
                    and -1.0 < pct_vs_sma < 7.0        # allow slight pullback or extension
                    and 35 <= rsi <= 70                 # expanded RSI sweet spot
                    and volume_ratio > 1.0              # any above-average volume day
                ):
                    continue

                # VCP filter: compute volatility contraction score and pass to AI.
                # We don't hard-block on VCP alone — expanding vol still allows
                # trend/breakout entries; the AI will reduce confidence accordingly.
                vcp_score = 0.3
                if len(df) >= 55:
                    try:
                        vcp_score = volatility_contraction(df)
                    except Exception:
                        pass
                if vcp_score < 0.3:
                    logger.debug("[SwingTrader] {} VCP very weak ({:.2f}) — skip", symbol, vcp_score)
                    continue

                # ATR-based stop and Kelly sizing (use 2.2x ATR — research optimal vs 2.0)
                stop   = round(price - 2.2 * atr, 2)
                target = round(price + 3.3 * atr, 2)  # 1.5:1 R:R minimum
                dollar_size = kelly_position_size(
                    account_cash=cash,
                    entry_price=price,
                    stop_price=stop,
                    win_rate=stats["win_rate"],
                    avg_win_loss_ratio=stats["avg_win_loss_ratio"],
                    max_pct=kelly_frac,
                )
                qty = round(dollar_size / price, 6) if price > 0 else 0
                if qty <= 0:
                    continue

                from analysis.ai_analyzer import AnalysisContext
                context = AnalysisContext(
                    symbol=symbol,
                    asset_class="stock",
                    strategy_name=self.name,
                    proposed_action="BUY",
                    signal_summary={**s, "sector_ok": sec_reason, "regime": regime.detail,
                                    "vcp_score": round(vcp_score, 2)},
                    news_headlines=[],
                    available_capital=cash,
                    open_position_count=len(self._portfolio.positions),
                    daily_pnl_pct=self._portfolio.daily_pnl_pct,
                )
                decision = await self._ai.analyze(context)

                if decision.recommendation == "BUY":
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side="buy",
                        asset_class="stock",
                        strategy_name=self.name,
                        entry_price=price,
                        stop_price=decision.stop_price or stop,
                        take_profit=decision.take_profit or target,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning,
                        qty=qty,
                    ))

            except Exception as e:
                logger.error("[SwingTrader] Error on {}: {}", symbol, e)

        return signals

    async def should_exit(self, position) -> bool:
        try:
            df = await self._market_data.get_bars(position.symbol, "1Day", limit=40)
            if df is None or len(df) < 20:
                return False
            s = TechnicalAnalyzer.build_signal_summary(df)
            rsi = s.get("rsi", 50)
            pct_vs_sma = s.get("price_vs_sma20_pct", 0)
            # Bearish divergence check using full RSI series from the df
            rsi_series   = TechnicalAnalyzer.rsi_series(df)
            price_series = df["close"]
            div = bearish_divergence(price_series, rsi_series)
            return rsi > 72 or pct_vs_sma < -2.5 or div
        except Exception:
            return False

    async def _check_exits(self) -> None:
        """Monitor open positions against stop/target/trailing stop/max hold."""
        if not self._market_open():
            return
        for key, pos in list(self._portfolio.positions.items()):
            if getattr(pos, "asset_class", "") != "stock":
                continue
            sym = pos.symbol
            try:
                price = await self._rh.get_quote_price(sym)
                if price is None:
                    continue
                price = float(price)

                # Update highest price tracker for trailing stop
                if price > pos.highest_price:
                    pos.highest_price = price

                # Chandelier exit: ATR-based trailing stop from recent highs
                trail_stop = 0.0
                try:
                    df = await self._market_data.get_bars(sym, "1Day", limit=30)
                    if df is not None and len(df) >= 22:
                        trail_stop = chandelier_exit(df, atr_mult=2.5)
                except Exception:
                    pass
                if trail_stop <= 0:
                    trail_stop = atr_trailing_stop(
                        pos.entry_price, pos.entry_price * 0.02, pos.highest_price
                    )
                effective_stop = max(pos.stop_loss or 0, trail_stop)

                # Max holding period: cut losers after 10 calendar days
                hold_days = (datetime.now(tz=pos.opened_at.tzinfo) - pos.opened_at).days
                max_hold_exit = hold_days >= 10 and price < pos.entry_price

                reason = None
                if price <= effective_stop:
                    reason = f"stop/trail hit @ ${price:.2f} (stop=${effective_stop:.2f})"
                elif pos.take_profit and price >= pos.take_profit:
                    reason = f"target hit @ ${price:.2f}"
                elif max_hold_exit:
                    reason = f"max hold ({hold_days}d) exit @ ${price:.2f}"
                elif await self.should_exit(pos):
                    reason = f"technical exit @ ${price:.2f}"

                if reason:
                    logger.info("[SwingTrader] EXIT {} — {}", sym, reason)
                    await self._signal_bus.put(TradeSignal(
                        symbol=sym,
                        side="sell",
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
                logger.error("[SwingTrader] Exit check error {}: {}", sym, e)

    async def run(self, market_just_opened: "asyncio.Event | None" = None) -> None:
        self._running = True
        logger.info("[SwingTrader] Started (interval={}s)", LOOP_INTERVAL)
        while self._running:
            try:
                await self._check_exits()
                signals = await self.generate_signals()
                for sig in signals:
                    await self._signal_bus.put(sig)
                    logger.info("[SwingTrader] → {} {} @ ${:.2f} qty={:.4f}",
                                sig.side.upper(), sig.symbol, sig.entry_price, sig.qty)
            except Exception as exc:
                logger.error("[SwingTrader] run error: {}", exc)

            # Sleep in 60s chunks so we wake up quickly when market opens
            remaining = LOOP_INTERVAL
            while remaining > 0 and self._running:
                chunk = min(60, remaining)
                if market_just_opened is not None:
                    try:
                        await asyncio.wait_for(market_just_opened.wait(), timeout=float(chunk))
                        logger.info("[SwingTrader] Market-open signal — immediate scan")
                        break  # re-scan immediately
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(chunk)
                remaining -= chunk
