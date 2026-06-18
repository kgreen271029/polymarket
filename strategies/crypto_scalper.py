"""strategies/crypto_scalper.py — Pure rule-based crypto scalping (no Claude, low latency)."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]
LOOP_INTERVAL = 60
MAX_CONCURRENT = 3
STOP_PCT = 0.008      # 0.8% stop
TARGET_PCT = 0.018    # 1.8% target → 2.25:1 R:R
MIN_BARS = 35


class CryptoScalper(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._portfolio = portfolio_tracker

    async def generate_signals(self) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        open_symbols = {p.split(":")[0] for p in self._portfolio.positions}
        open_crypto = sum(
            1 for p in self._portfolio.positions.values()
            if p.asset_class == "crypto"
        )
        if open_crypto >= MAX_CONCURRENT:
            return signals

        for symbol in WATCHLIST:
            norm = symbol.replace("/", "")
            if norm in open_symbols or symbol in open_symbols:
                continue
            try:
                df = await self._market_data.get_bars(symbol, "1Min", limit=60)
                if df is None or len(df) < MIN_BARS:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                if not s:
                    continue

                rsi = s.get("rsi", 50)
                macd_hist = s.get("macd_hist", 0)
                volume_spike = s.get("volume_spike", False)
                bb_pct = s.get("bb_pct", 0.5)
                price = s.get("latest_close", 0)

                if (
                    rsi < 36
                    and macd_hist > 0
                    and volume_spike
                    and bb_pct < 0.15
                    and price > 0
                ):
                    stop = round(price * (1 - STOP_PCT), 6)
                    target = round(price * (1 + TARGET_PCT), 6)
                    cash = self._portfolio.get_available_capital()
                    max_dollars = min(15.0, cash * 0.15)  # $15 per scalp
                    qty = round(max_dollars / price, 6) if price > 0 else 0
                    if qty <= 0:
                        continue
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side="buy",
                        asset_class="crypto",
                        strategy_name=self.name,
                        entry_price=price,
                        stop_price=stop,
                        take_profit=target,
                        confidence="Medium",
                        reasoning=f"RSI={rsi:.1f} oversold, MACD turning up, vol spike, BB lower",
                        qty=qty,
                    ))
            except Exception as e:
                logger.error("[CryptoScalper] Error scanning {}: {}", symbol, e)

        return signals

    async def should_exit(self, position: object) -> bool:
        try:
            df = await self._market_data.get_bars(position.symbol, "1Min", limit=30)  # type: ignore[attr-defined]
            if df is None or len(df) < 10:
                return False
            s = TechnicalAnalyzer.build_signal_summary(df)
            rsi = s.get("rsi", 50)
            bb_pct = s.get("bb_pct", 0.5)
            return rsi > 65 or bb_pct > 0.92
        except Exception:
            return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
