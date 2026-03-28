"""
==============================================================
  MULTI-INDICATOR CONFLUENCE STRATEGY

  Signal requires agreement from:
    1. Trend Direction  (EMA stack + ADX)
    2. Momentum         (RSI + MACD)
    3. Volume           (Volume MA filter)
    4. Volatility       (Bollinger Band position)

  Higher confluence → higher confidence trades.
==============================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.indicators import (
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    adx,
    stochastic,
    obv,
    vwap,
    cmf,
    supertrend,
)


@dataclass
class Signal:
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 – 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    indicators: dict


class MultiIndicatorStrategy:
    """
    Confluence-based strategy that combines multiple indicators.
    Only generates a BUY when ≥4 bullish conditions align.
    Only generates a SELL when ≥4 bearish conditions align.
    """

    def __init__(self, params):
        self.p = params

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < 250:
            return Signal(
                symbol,
                "HOLD",
                0.0,
                df["close"].iloc[-1],
                0.0,
                0.0,
                "Insufficient data",
                {},
            )

        close = df["close"]
        price = float(close.iloc[-1])

        # ── Trend indicators ──────────────────────────
        ema9 = ema(close, self.p.ema_fast)
        ema21 = ema(close, self.p.ema_slow)
        ema200 = ema(close, self.p.ema_trend)
        adx_v, pdi, ndi = adx(df, self.p.adx_period)
        st_line, st_bull = supertrend(df)

        # ── Momentum ──────────────────────────────────
        rsi_v = rsi(close, self.p.rsi_period)
        macd_l, macd_sig, macd_hist = macd(
            close, self.p.macd_fast, self.p.macd_slow, self.p.macd_signal
        )
        k_line, d_line = stochastic(
            df, self.p.stoch_k, self.p.stoch_d, self.p.stoch_smooth
        )

        # ── Volatility ────────────────────────────────
        bb_upper, bb_mid, bb_lower = bollinger_bands(
            close, self.p.bb_period, self.p.bb_std
        )
        atr_v = atr(df, self.p.atr_period)

        # ── Volume ────────────────────────────────────
        vol_ma = df["volume"].rolling(self.p.volume_ma_period).mean()
        vol_ratio = df["volume"].iloc[-1] / (vol_ma.iloc[-1] + 1e-9)
        obv_ma = ema(obv(df), 20)
        cmf_v = cmf(df)

        # ── VWAP ─────────────────────────────────────
        vwap_v = vwap(df)

        # Latest values
        i = -1
        vals = {
            "price": price,
            "ema9": float(ema9.iloc[i]),
            "ema21": float(ema21.iloc[i]),
            "ema200": float(ema200.iloc[i]),
            "adx": float(adx_v.iloc[i]),
            "pdi": float(pdi.iloc[i]),
            "ndi": float(ndi.iloc[i]),
            "rsi": float(rsi_v.iloc[i]),
            "macd": float(macd_l.iloc[i]),
            "macd_sig": float(macd_sig.iloc[i]),
            "macd_hist": float(macd_hist.iloc[i]),
            "stoch_k": float(k_line.iloc[i]),
            "stoch_d": float(d_line.iloc[i]),
            "bb_upper": float(bb_upper.iloc[i]),
            "bb_mid": float(bb_mid.iloc[i]),
            "bb_lower": float(bb_lower.iloc[i]),
            "atr": float(atr_v.iloc[i]),
            "vol_ratio": float(vol_ratio),
            "cmf": float(cmf_v.iloc[i]),
            "vwap": float(vwap_v.iloc[i]),
            "st_bull": bool(st_bull.iloc[i]),
        }

        # ── Score conditions ───────────────────────────
        bull_score = 0
        bear_score = 0
        reasons_bull = []
        reasons_bear = []

        # 1. EMA Alignment
        if vals["ema9"] > vals["ema21"] > vals["ema200"]:
            bull_score += 2
            reasons_bull.append("EMA bullish stack")
        elif vals["ema9"] < vals["ema21"] < vals["ema200"]:
            bear_score += 2
            reasons_bear.append("EMA bearish stack")

        # 2. ADX trend strength
        if vals["adx"] > self.p.adx_threshold:
            if vals["pdi"] > vals["ndi"]:
                bull_score += 1
                reasons_bull.append(f"ADX {vals['adx']:.1f} bullish")
            else:
                bear_score += 1
                reasons_bear.append(f"ADX {vals['adx']:.1f} bearish")

        # 3. Supertrend
        if vals["st_bull"]:
            bull_score += 1
            reasons_bull.append("Supertrend bullish")
        else:
            bear_score += 1
            reasons_bear.append("Supertrend bearish")

        # 4. RSI
        if vals["rsi"] < self.p.rsi_oversold and vals["rsi"] > 20:
            bull_score += 2
            reasons_bull.append(f"RSI oversold {vals['rsi']:.1f}")
        elif vals["rsi"] > self.p.rsi_overbought and vals["rsi"] < 80:
            bear_score += 2
            reasons_bear.append(f"RSI overbought {vals['rsi']:.1f}")
        elif 40 < vals["rsi"] < 60:
            pass  # Neutral
        elif vals["rsi"] > 55:
            bull_score += 0.5
        elif vals["rsi"] < 45:
            bear_score += 0.5

        # 5. MACD crossover
        prev_hist = float(macd_hist.iloc[-2])
        if prev_hist < 0 and vals["macd_hist"] > 0:
            bull_score += 2
            reasons_bull.append("MACD bullish crossover")
        elif prev_hist > 0 and vals["macd_hist"] < 0:
            bear_score += 2
            reasons_bear.append("MACD bearish crossover")
        elif vals["macd_hist"] > 0:
            bull_score += 0.5
        else:
            bear_score += 0.5

        # 6. Stochastic
        prev_k = float(k_line.iloc[-2])
        if vals["stoch_k"] < 20 and vals["stoch_k"] > prev_k:
            bull_score += 1.5
            reasons_bull.append(f"Stoch oversold + rising {vals['stoch_k']:.1f}")
        elif vals["stoch_k"] > 80 and vals["stoch_k"] < prev_k:
            bear_score += 1.5
            reasons_bear.append(f"Stoch overbought + falling {vals['stoch_k']:.1f}")

        # 7. Bollinger Band position
        bb_pct = (price - vals["bb_lower"]) / (
            vals["bb_upper"] - vals["bb_lower"] + 1e-9
        )
        if bb_pct < 0.2:
            bull_score += 1.5
            reasons_bull.append(f"Near BB lower {bb_pct:.2%}")
        elif bb_pct > 0.8:
            bear_score += 1.5
            reasons_bear.append(f"Near BB upper {bb_pct:.2%}")

        # 8. Volume confirmation (REQUIRED)
        vol_confirmed = vol_ratio >= self.p.min_volume_ratio
        if vol_confirmed:
            bull_score += 0.5
            bear_score += 0.5
        else:
            bull_score *= 0.6  # Penalize low volume signals
            bear_score *= 0.6

        # 9. VWAP position
        if price > vals["vwap"]:
            bull_score += 0.5
            reasons_bull.append("Above VWAP")
        else:
            bear_score += 0.5
            reasons_bear.append("Below VWAP")

        # 10. CMF (Chaikin Money Flow)
        if vals["cmf"] > 0.1:
            bull_score += 1
            reasons_bull.append(f"CMF positive {vals['cmf']:.2f}")
        elif vals["cmf"] < -0.1:
            bear_score += 1
            reasons_bear.append(f"CMF negative {vals['cmf']:.2f}")

        # ── ATR-based stop loss & take profit ─────────
        atr_sl = vals["atr"] * self.p.atr_multiplier
        stop_buy = price - atr_sl
        tp_buy = price + (atr_sl * 2.5)  # 2.5:1 reward/risk
        stop_sell = price + atr_sl
        tp_sell = price - (atr_sl * 2.5)

        max_score = 12.0

        # ── Make decision ──────────────────────────────
        if bull_score >= 6 and bull_score > bear_score * 1.5:
            confidence = min(bull_score / max_score, 1.0)
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=price,
                stop_loss=max(stop_buy, price * 0.97),
                take_profit=tp_buy,
                reason=" | ".join(reasons_bull[:4]),
                indicators=vals,
            )

        elif bear_score >= 6 and bear_score > bull_score * 1.5:
            confidence = min(bear_score / max_score, 1.0)
            return Signal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                price=price,
                stop_loss=min(stop_sell, price * 1.03),
                take_profit=tp_sell,
                reason=" | ".join(reasons_bear[:4]),
                indicators=vals,
            )

        return Signal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            price=price,
            stop_loss=0.0,
            take_profit=0.0,
            reason=f"No confluence (bull={bull_score:.1f}, bear={bear_score:.1f})",
            indicators=vals,
        )
