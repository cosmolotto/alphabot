"""
==============================================================
  TECHNICAL INDICATORS
  Pure NumPy/Pandas — no TA-Lib dependency needed
==============================================================
"""

from typing import Dict
import numpy as np
import pandas as pd
from typing import Tuple, Dict


# ──────────────────────────────────────────────────────────────
#  TREND INDICATORS
# ──────────────────────────────────────────────────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


def macd(
    series: pd.Series, fast=12, slow=26, signal=9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def supertrend(
    df: pd.DataFrame, period=10, multiplier=3.0
) -> Tuple[pd.Series, pd.Series]:
    """Supertrend indicator — excellent for trend following."""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + (multiplier * atr_val)
    lower = hl2 - (multiplier * atr_val)

    supertrend_vals = [True] * len(df)
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        final_upper.iloc[i] = (
            upper.iloc[i]
            if upper.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            lower.iloc[i]
            if lower.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
            else final_lower.iloc[i - 1]
        )
        if supertrend_vals[i - 1] and df["close"].iloc[i] < final_lower.iloc[i]:
            supertrend_vals[i] = False
        elif not supertrend_vals[i - 1] and df["close"].iloc[i] > final_upper.iloc[i]:
            supertrend_vals[i] = True
        else:
            supertrend_vals[i] = supertrend_vals[i - 1]

    trend = pd.Series(supertrend_vals, index=df.index)
    st_line = pd.Series(
        [
            final_lower.iloc[i] if supertrend_vals[i] else final_upper.iloc[i]
            for i in range(len(df))
        ],
        index=df.index,
    )
    return st_line, trend


def ichimoku(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku Cloud — complete system."""
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(
        26
    )
    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


# ──────────────────────────────────────────────────────────────
#  OSCILLATORS
# ──────────────────────────────────────────────────────────────


def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, k=14, d=3, smooth=3) -> Tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min)
    k_line = k_raw.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def cci(df: pd.DataFrame, period=20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad)


def williams_r(df: pd.DataFrame, period=14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min)


def mfi(df: pd.DataFrame, period=14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = tp * df["volume"]
    positive_mf = raw_mf.where(tp > tp.shift(1), 0)
    negative_mf = raw_mf.where(tp < tp.shift(1), 0)
    pmf = positive_mf.rolling(period).sum()
    nmf = negative_mf.rolling(period).sum()
    mf_ratio = pmf / nmf.replace(0, np.nan)
    return 100 - (100 / (1 + mf_ratio))


# ──────────────────────────────────────────────────────────────
#  VOLATILITY INDICATORS
# ──────────────────────────────────────────────────────────────


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, period=20, std=2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


def keltner_channel(
    df: pd.DataFrame, period=20, multiplier=2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["close"], period)
    atr_v = atr(df, period)
    upper = mid + multiplier * atr_v
    lower = mid - multiplier * atr_v
    return upper, mid, lower


def squeeze_momentum(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """LazyBear Squeeze Momentum Indicator."""
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"])
    kc_upper, kc_mid, kc_lower = keltner_channel(df)
    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

    # Momentum value
    highest_high = df["high"].rolling(20).max()
    lowest_low = df["low"].rolling(20).min()
    delta = df["close"] - ((highest_high + lowest_low) / 2 + sma(df["close"], 20)) / 2
    momentum = delta.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    return momentum, sqz_on


# ──────────────────────────────────────────────────────────────
#  VOLUME INDICATORS
# ──────────────────────────────────────────────────────────────


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff())
    return (sign * df["volume"]).fillna(0).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def cmf(df: pd.DataFrame, period=20) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    ).replace(0, np.nan)
    return (clv * df["volume"]).rolling(period).sum() / df["volume"].rolling(
        period
    ).sum()


# ──────────────────────────────────────────────────────────────
#  TREND STRENGTH
# ──────────────────────────────────────────────────────────────


def adx(df: pd.DataFrame, period=14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index — trend strength."""
    tr_val = atr(df, period)
    up_move = df["high"] - df["high"].shift()
    dn_move = df["low"].shift() - df["low"]

    pdm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    ndm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)

    pdm_smooth = pdm.ewm(com=period - 1, adjust=False).mean()
    ndm_smooth = ndm.ewm(com=period - 1, adjust=False).mean()
    atr_smooth = tr_val

    pdi = 100 * pdm_smooth / atr_smooth.replace(0, np.nan)
    ndi = 100 * ndm_smooth / atr_smooth.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx_val = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_val, pdi, ndi


def pivot_points(df: pd.DataFrame) -> Dict:
    """Classic pivot points for support/resistance."""
    pivot = (df["high"] + df["low"] + df["close"]) / 3
    r1 = 2 * pivot - df["low"]
    r2 = pivot + (df["high"] - df["low"])
    s1 = 2 * pivot - df["high"]
    s2 = pivot - (df["high"] - df["low"])
    return {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2}
