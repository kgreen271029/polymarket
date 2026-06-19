"""
Pure-Python technical indicators. No external dependencies.
All functions take a list of floats (usually closing prices) and return
either a single latest value or a list aligned to the input.
"""


def sma(values, period):
    """Simple Moving Average (latest value)."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values, period):
    """Exponential Moving Average as a full series."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    # seed with SMA of first `period` values
    ema = [sum(values[:period]) / period]
    for price in values[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def ema(values, period):
    """Latest EMA value."""
    series = ema_series(values, period)
    return series[-1] if series else None


def rsi(values, period=14):
    """Relative Strength Index (latest value), 0-100."""
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    # Wilder's smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values, fast=12, slow=26, signal=9):
    """
    MACD. Returns (macd_line, signal_line, histogram) latest values,
    or (None, None, None) if not enough data.
    """
    if len(values) < slow + signal:
        return None, None, None
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    # align tails
    n = min(len(fast_ema), len(slow_ema))
    macd_line = [fast_ema[-n + i] - slow_ema[-n + i] for i in range(n)]
    signal_series = ema_series(macd_line, signal)
    if not signal_series:
        return None, None, None
    macd_val = macd_line[-1]
    signal_val = signal_series[-1]
    return macd_val, signal_val, macd_val - signal_val


def bollinger(values, period=20, num_std=2):
    """Bollinger Bands. Returns (upper, mid, lower) latest, or (None,)*3."""
    if len(values) < period:
        return None, None, None
    window = values[-period:]
    mid = sum(window) / period
    variance = sum((v - mid) ** 2 for v in window) / period
    std = variance ** 0.5
    return mid + num_std * std, mid, mid - num_std * std


def atr(highs, lows, closes, period=14):
    """Average True Range (latest) — used for volatility-based stops."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        trs.append(max(high_low, high_close, low_close))
    if len(trs) < period:
        return None
    # Wilder's smoothing
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def pct_change(values, lookback=1):
    """Percent change over the last `lookback` bars."""
    if len(values) < lookback + 1:
        return 0.0
    old = values[-lookback - 1]
    new = values[-1]
    return ((new - old) / old) * 100 if old else 0.0
