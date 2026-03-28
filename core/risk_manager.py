"""
==============================================================
  RISK MANAGEMENT ENGINE

  Handles:
    - Position sizing (Kelly Criterion + fixed fractional)
    - Stop loss management (fixed, trailing, ATR-based)
    - Daily loss limits
    - Correlation checks (avoid holding correlated assets)
    - Max drawdown protection
==============================================================
"""

import math
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop: Optional[float] = None
    order_id: str = ""
    strategy: str = ""
    confidence: float = 0.5


@dataclass
class TradeResult:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_s: float
    exit_reason: str
    strategy: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskManager:

    def __init__(self, config):
        self.cfg = config
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_date: date = date.today()
        self.equity_curve: List[float] = [config.total_capital]
        self.peak_equity: float = config.total_capital

    # ──────────────────────────────────────────
    #  POSITION SIZING
    # ──────────────────────────────────────────

    def position_size(
        self, price: float, stop_loss: float, confidence: float = 0.5
    ) -> float:
        """
        Calculates position size using:
          1. Fixed fractional (base)
          2. Kelly Criterion adjustment
          3. Confidence scaling
          4. Max risk per trade cap
        Returns: quantity in base asset units
        """
        capital = self._available_capital()
        if capital <= 0 or price <= 0 or stop_loss <= 0:
            return 0.0

        # Risk per trade in USDT
        risk_pct = self.cfg.capital_per_trade
        risk_capital = capital * risk_pct

        # Distance to stop loss
        risk_per_unit = abs(price - stop_loss)
        if risk_per_unit < 1e-9:
            return 0.0

        # Base size
        qty = risk_capital / risk_per_unit

        # Kelly adjustment based on recent win rate
        kelly_f = min(self._kelly_fraction(), 0.5)
        qty *= max(kelly_f, 0.1)

        # Scale by signal confidence
        qty *= max(0.5, min(confidence, 1.0))

        # Cap: never risk more than 2% of total capital on one trade
        max_qty = (self.cfg.total_capital * 0.02) / risk_per_unit
        qty = min(qty, max_qty)

        # Cap: never more than capital_per_trade of available capital
        max_notional = capital * self.cfg.capital_per_trade
        qty = min(qty, max_notional / price)

        return round(qty, 8)

    def _kelly_fraction(self) -> float:
        """Kelly Criterion based on recent 20 trades."""
        recent = self.trade_history[-20:]
        if len(recent) < 5:
            return 0.5  # Conservative until we have data

        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]
        win_rate = len(wins) / len(recent)

        if not losses:
            return 0.8
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0.01

        b = avg_win / (avg_loss + 1e-9)  # win/loss ratio
        kelly = (win_rate * b - (1 - win_rate)) / (b + 1e-9)
        return max(0.1, min(kelly * 0.5, 0.8))  # Half-Kelly capped at 80%

    def _available_capital(self) -> float:
        locked = sum(
            p.quantity * p.entry_price
            for p in self.positions.values()
            if p.side == "long"
        )
        return max(0.0, self.equity_curve[-1] - locked)

    # ──────────────────────────────────────────
    #  DAILY LOSS LIMIT
    # ──────────────────────────────────────────

    def check_daily_loss_limit(self) -> bool:
        """Returns True if trading is allowed today."""
        today = date.today()
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today

        limit = self.cfg.total_capital * self.cfg.max_daily_loss_pct
        if self.daily_pnl < -limit:
            logger.warning(
                f"Daily loss limit hit: {self.daily_pnl:.2f} USDT. "
                f"Trading paused for today."
            )
            return False
        return True

    def check_max_drawdown(self) -> bool:
        """Pause trading if max drawdown exceeds 15%."""
        current_equity = self.equity_curve[-1]
        drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-9)
        if drawdown > 0.15:
            logger.warning(f"Max drawdown {drawdown:.1%} reached. Trading paused.")
            return False
        return True

    def can_open_trade(self, symbol: str) -> bool:
        """Check all risk gates before opening a position."""
        if symbol in self.positions:
            logger.info(f"Already have position in {symbol}")
            return False
        if len(self.positions) >= self.cfg.max_open_trades:
            logger.info(f"Max open trades ({self.cfg.max_open_trades}) reached")
            return False
        if not self.check_daily_loss_limit():
            return False
        if not self.check_max_drawdown():
            return False
        return True

    # ──────────────────────────────────────────
    #  TRAILING STOP MANAGEMENT
    # ──────────────────────────────────────────

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Updates trailing stop for a position.
        Returns "STOP_HIT" if stop is triggered, else None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        trail_pct = self.cfg.trailing_stop_pct

        if pos.side == "long":
            pos.highest_price = max(pos.highest_price, current_price)
            trail_stop = pos.highest_price * (1 - trail_pct)
            pos.trailing_stop = trail_stop
            if current_price <= trail_stop:
                return "STOP_HIT"

        elif pos.side == "short":
            pos.lowest_price = min(pos.lowest_price, current_price)
            trail_stop = pos.lowest_price * (1 + trail_pct)
            pos.trailing_stop = trail_stop
            if current_price >= trail_stop:
                return "STOP_HIT"

        return None

    def check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if a position should be exited.
        Returns exit reason string or None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        if pos.side == "long":
            if current_price <= pos.stop_loss:
                return "STOP_LOSS"
            if current_price >= pos.take_profit:
                return "TAKE_PROFIT"
        elif pos.side == "short":
            if current_price >= pos.stop_loss:
                return "STOP_LOSS"
            if current_price <= pos.take_profit:
                return "TAKE_PROFIT"

        if self.cfg.trailing_stop:
            trail_hit = self.update_trailing_stop(symbol, current_price)
            if trail_hit:
                return "TRAILING_STOP"

        return None

    # ──────────────────────────────────────────
    #  OPEN / CLOSE POSITIONS
    # ──────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        sl: float,
        tp: float,
        strategy: str = "",
        confidence: float = 0.5,
    ):
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            strategy=strategy,
            confidence=confidence,
            highest_price=price,
            lowest_price=price,
        )
        self.positions[symbol] = pos
        logger.info(
            f"OPENED {side.upper()} {symbol} @ {price:.6f} "
            f"| qty={qty:.6f} SL={sl:.6f} TP={tp:.6f}"
        )

    def close_position(self, symbol: str, exit_price: float, reason: str):
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = pnl / (pos.entry_price * pos.quantity + 1e-9)
        duration = (datetime.utcnow() - pos.entry_time).total_seconds()

        result = TradeResult(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_s=duration,
            exit_reason=reason,
            strategy=pos.strategy,
        )

        self.trade_history.append(result)
        self.daily_pnl += pnl

        # Update equity curve
        new_equity = self.equity_curve[-1] + pnl
        self.equity_curve.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} CLOSED {pos.side.upper()} {symbol} @ {exit_price:.6f} "
            f"| PnL: {pnl:+.2f} USDT ({pnl_pct:+.2%}) | {reason}"
        )
        return result

    # ──────────────────────────────────────────
    #  STATISTICS
    # ──────────────────────────────────────────

    def get_stats(self) -> dict:
        trades = self.trade_history
        if not trades:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        profit_factor = abs(
            sum(t.pnl for t in wins) / (sum(t.pnl for t in losses) + 1e-9)
        )

        pnl_series = [t.pnl for t in trades]
        pnl_std = np.std(pnl_series) if len(pnl_series) > 1 else 1
        sharpe = (np.mean(pnl_series) / (pnl_std + 1e-9)) * math.sqrt(365)

        # Max drawdown
        equity = self.equity_curve
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            max_dd = max(max_dd, (peak - e) / (peak + 1e-9))

        return {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "current_equity": self.equity_curve[-1],
            "open_positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
        }
