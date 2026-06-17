"""analysis/technical.py — Technical indicator computation (pure pandas/numpy, no pandas_ta)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalAnalyzer:
    """
    Computes technical indicators from OHLCV DataFrames.
    Expected columns (case-sensitive): open, high, low, close, volume
    """

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 2:
            return 50.0
        delta = df["close"].diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        val = rsi_series.iloc[-1]
        return 50.0 if pd.isna(val) else float(val)

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        if len(df) < slow + signal:
            return 0.0, 0.0, 0.0
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        def _last(s: pd.Series) -> float:
            v = s.iloc[-1]
            return 0.0 if pd.isna(v) else float(v)

        return _last(macd_line), _last(signal_line), _last(histogram)

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[float, float, float]:
        close_last = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        if len(df) < period:
            return close_last, close_last, close_last
        mid = df["close"].rolling(period).mean()
        sd = df["close"].rolling(period).std(ddof=0)
        upper = mid + std * sd
        lower = mid - std * sd

        def _last(s: pd.Series) -> float:
            v = s.iloc[-1]
            return close_last if pd.isna(v) else float(v)

        return _last(upper), _last(mid), _last(lower)

    @staticmethod
    def vwap(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        total_vol = df["volume"].sum()
        if total_vol == 0:
            return float(df["close"].iloc[-1])
        return float((typical * df["volume"]).sum() / total_vol)

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20) -> float:
        close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        if len(df) < period:
            return close
        val = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
        return close if pd.isna(val) else float(val)

    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20) -> float:
        close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        if len(df) < period:
            return close
        val = df["close"].rolling(period).mean().iloc[-1]
        return close if pd.isna(val) else float(val)

    @staticmethod
    def detect_volume_spike(
        df: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 2.0,
    ) -> bool:
        if len(df) < lookback + 1:
            return False
        recent_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-(lookback + 1):-1].mean())
        return avg_vol > 0 and recent_vol / avg_vol >= threshold

    @staticmethod
    def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> bool:
        if len(df) < lookback + 1:
            return False
        latest_close = float(df["close"].iloc[-1])
        prior_high = float(df["close"].iloc[-(lookback + 1):-1].max())
        return latest_close > prior_high

    @staticmethod
    def build_signal_summary(df: pd.DataFrame) -> dict:
        if len(df) < 30:
            return {}
        try:
            latest_close = float(df["close"].iloc[-1])

            rsi_val = TechnicalAnalyzer.rsi(df)
            macd_line, signal_line, macd_hist = TechnicalAnalyzer.macd(df)
            macd_direction = "bullish" if macd_hist > 0 else "bearish" if macd_hist < 0 else "neutral"

            bb_upper, bb_mid, bb_lower = TechnicalAnalyzer.bollinger_bands(df)
            band_width = bb_upper - bb_lower
            bb_pct = (latest_close - bb_lower) / band_width if band_width > 0 else 0.5

            vwap_val = TechnicalAnalyzer.vwap(df)
            vwap_delta_pct = (latest_close - vwap_val) / vwap_val * 100.0 if vwap_val else 0.0

            volume_spike = TechnicalAnalyzer.detect_volume_spike(df)
            latest_vol = float(df["volume"].iloc[-1])
            mean_vol_20 = float(df["volume"].iloc[-21:-1].mean())
            volume_ratio = (latest_vol / mean_vol_20) if mean_vol_20 > 0 else 1.0

            breakout_20 = TechnicalAnalyzer.detect_breakout(df)
            sma_20 = TechnicalAnalyzer.sma(df, 20)
            ema_20 = TechnicalAnalyzer.ema(df, 20)
            price_vs_sma20_pct = (latest_close - sma_20) / sma_20 * 100.0 if sma_20 else 0.0

            above_ema = latest_close > ema_20
            if above_ema and macd_direction == "bullish":
                trend = "uptrend"
            elif not above_ema and macd_direction == "bearish":
                trend = "downtrend"
            else:
                trend = "sideways"

            return {
                "rsi": rsi_val,
                "macd_hist": macd_hist,
                "macd_direction": macd_direction,
                "bb_pct": bb_pct,
                "vwap_delta_pct": vwap_delta_pct,
                "volume_spike": volume_spike,
                "volume_ratio": volume_ratio,
                "breakout_20": breakout_20,
                "sma_20": sma_20,
                "price_vs_sma20_pct": price_vs_sma20_pct,
                "trend": trend,
                "latest_close": latest_close,
            }
        except Exception as exc:
            logger.error("build_signal_summary failed: {}", exc)
            return {}
