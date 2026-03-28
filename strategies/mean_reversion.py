"""
==============================================================
  MEAN REVERSION STRATEGY

  Trades when price deviates significantly from mean.
  Best in ranging, sideways markets.

  Uses: Bollinger Band squeeze exits, RSI extremes,
        Stochastic confirmation, MFI divergence
==============================================================
"""

import pandas as pd
import numpy as np
from core.indicators import (
    rsi,
    bollinger_bands,
    stochastic,
    mfi,
    atr,
    adx,
    ema,
    williams_r,
)
from .multi_indicator import Signal


class MeanReversionStrategy:
    def __init__(self, params):
        self.p = params

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < 60:
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

        rsi_v = rsi(close, 14)
        bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
        bb_u2, _, bb_l2 = bollinger_bands(close, 20, 2.5)  # Extreme bands
        k_line, d_line = stochastic(df)
        mfi_v = mfi(df, 14)
        atr_v = atr(df, 14)
        adx_v, pdi, ndi = adx(df, 14)
        ema50 = ema(close, 50)
        wr = williams_r(df, 14)

        rsi_ = float(rsi_v.iloc[-1])
        bb_u_ = float(bb_u.iloc[-1])
        bb_l_ = float(bb_l.iloc[-1])
        bb_u2_ = float(bb_u2.iloc[-1])
        bb_l2_ = float(bb_l2.iloc[-1])
        bb_m_ = float(bb_m.iloc[-1])
        k_ = float(k_line.iloc[-1])
        d_ = float(d_line.iloc[-1])
        mfi_ = float(mfi_v.iloc[-1])
        adx_ = float(adx_v.iloc[-1])
        wr_ = float(wr.iloc[-1])
        atr_ = float(atr_v.iloc[-1])
        ema50_ = float(ema50.iloc[-1])

        # Mean reversion works best in sideways (low ADX)
        trending = adx_ > 30

        bb_pct = (price - bb_l_) / (bb_u_ - bb_l_ + 1e-9)

        vals = dict(
            rsi=rsi_,
            bb_pct=bb_pct,
            stoch_k=k_,
            mfi=mfi_,
            adx=adx_,
            wr=wr_,
            price=price,
            bb_mid=bb_m_,
        )

        # ── Long Signal ──────────────────────────────────
        long_conditions = [
            price < bb_l_,  # Below lower BB
            rsi_ < 30,  # RSI oversold
            k_ < 20,  # Stochastic oversold
            # MFI oversold (capitulation)
            mfi_ < 20,
            wr_ < -80,  # Williams R oversold
        ]
        long_score = sum(long_conditions)

        # ── Short Signal ─────────────────────────────────
        short_conditions = [
            price > bb_u_,  # Above upper BB
            rsi_ > 70,  # RSI overbought
            k_ > 80,  # Stochastic overbought
            mfi_ > 80,  # MFI overbought
            wr_ > -20,  # Williams R overbought
        ]
        short_score = sum(short_conditions)

        # Don't fight a strong trend
        if trending:
            long_score = max(0, long_score - 1)
            short_score = max(0, short_score - 1)

        if long_score >= 3:
            sl = min(price - 1.5 * atr_, bb_l2_ - atr_ * 0.5)
            tp = bb_m_  # Target: return to mean
            conf = long_score / 5
            return Signal(
                symbol,
                "BUY",
                conf,
                price,
                sl,
                tp,
                f"MeanRev long ({long_score}/5 signals)",
                vals,
            )

        if short_score >= 3:
            sl = max(price + 1.5 * atr_, bb_u2_ + atr_ * 0.5)
            tp = bb_m_
            conf = short_score / 5
            return Signal(
                symbol,
                "SELL",
                conf,
                price,
                sl,
                tp,
                f"MeanRev short ({short_score}/5 signals)",
                vals,
            )

        return Signal(
            symbol,
            "HOLD",
            0.0,
            price,
            0.0,
            0.0,
            f"No reversal (BB%={bb_pct:.0%} RSI={rsi_:.0f})",
            vals,
        )
