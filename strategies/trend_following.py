"""
==============================================================
  TREND FOLLOWING STRATEGY

  Core logic: Ride strong trends using:
    - Supertrend direction
    - Hull MA crossover
    - ADX for trend strength filter
    - ATR trailing stops
==============================================================
"""

import pandas as pd
import numpy as np
from core.indicators import hull_ma, atr, adx, supertrend, ema, rsi
from .multi_indicator import Signal


class TrendFollowingStrategy:
    def __init__(self, params):
        self.p = params

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < 100:
            return Signal(
                symbol,
                "HOLD",
                0.0,
                float(df["close"].iloc[-1]),
                0.0,
                0.0,
                "Insufficient data",
                {},
            )

        close = df["close"]
        price = float(close.iloc[-1])

        hma_fast = hull_ma(close, 9)
        hma_slow = hull_ma(close, 21)
        st_line, st_bull = supertrend(df, period=10, multiplier=3.0)
        adx_v, pdi, ndi = adx(df, 14)
        rsi_v = rsi(close, 14)
        atr_v = atr(df, 14)
        ema200 = ema(close, 200)

        hf, hs = float(hma_fast.iloc[-1]), float(hma_slow.iloc[-1])
        hf_p, hs_p = float(hma_fast.iloc[-2]), float(hma_slow.iloc[-2])
        adx_v_ = float(adx_v.iloc[-1])
        pdi_, ndi_ = float(pdi.iloc[-1]), float(ndi.iloc[-1])
        st_bull_ = bool(st_bull.iloc[-1])
        rsi_ = float(rsi_v.iloc[-1])
        atr_ = float(atr_v.iloc[-1])
        ema200_ = float(ema200.iloc[-1])

        vals = dict(
            hma_fast=hf,
            hma_slow=hs,
            adx=adx_v_,
            pdi=pdi_,
            ndi=ndi_,
            st_bull=st_bull_,
            rsi=rsi_,
            atr=atr_,
        )

        trend_strong = adx_v_ > self.p.adx_threshold

        # Bullish crossover (HMA fast crosses above slow)
        hma_bull_cross = hf_p <= hs_p and hf > hs
        # Bearish crossover
        hma_bear_cross = hf_p >= hs_p and hf < hs

        # Long condition
        long_ok = (
            (hma_bull_cross or (hf > hs and price > ema200_))
            and st_bull_
            and trend_strong
            and pdi_ > ndi_
            and rsi_ < 75
        )

        # Short condition
        short_ok = (
            (hma_bear_cross or (hf < hs and price < ema200_))
            and not st_bull_
            and trend_strong
            and ndi_ > pdi_
            and rsi_ > 25
        )

        if long_ok:
            sl = float(st_line.iloc[-1]) - 0.001 * price
            tp = price + 3.0 * atr_
            conf = min((adx_v_ - 20) / 40, 1.0)
            return Signal(
                symbol,
                "BUY",
                conf,
                price,
                sl,
                tp,
                f"HMA bullish | Supertrend up | ADX {adx_v_:.0f}",
                vals,
            )

        if short_ok:
            sl = float(st_line.iloc[-1]) + 0.001 * price
            tp = price - 3.0 * atr_
            conf = min((adx_v_ - 20) / 40, 1.0)
            return Signal(
                symbol,
                "SELL",
                conf,
                price,
                sl,
                tp,
                f"HMA bearish | Supertrend down | ADX {adx_v_:.0f}",
                vals,
            )

        return Signal(
            symbol,
            "HOLD",
            0.0,
            price,
            0.0,
            0.0,
            f"No trend signal (ADX={adx_v_:.1f})",
            vals,
        )
