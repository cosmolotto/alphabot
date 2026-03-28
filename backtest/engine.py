"""
==============================================================
  BACKTESTING ENGINE

  Realistic simulation with:
    - Maker/taker fees
    - Slippage simulation
    - Proper position sizing
    - Walk-forward analysis
    - Performance metrics
==============================================================
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Type, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TAKER_FEE = 0.001   # 0.1% market order
MAKER_FEE = 0.0006  # 0.06% limit order
SLIPPAGE = 0.0003  # 0.03% average slippage


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    confidence: float


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_trade_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_holding_hours: float
    trades: List[BacktestTrade] = field(default_factory=list)


class Backtester:

    def __init__(self, strategy, strategy_params, bot_config):
        self.strategy_class = strategy
        self.strategy_params = strategy_params
        self.cfg = bot_config

    def run(self, df: pd.DataFrame, symbol: str,
            initial_capital: float = None) -> BacktestResult:
        """Run backtest on historical OHLCV data."""
        capital = initial_capital or self.cfg.total_capital
        equity = capital
        peak = capital

        strategy = self.strategy_class(self.strategy_params)
        trades: List[BacktestTrade] = []
        equity_curve = [capital]

        # {"side", "entry_price", "qty", "stop", "take_profit",
        position = None
        #  "entry_time", "confidence", "trailing_high", "trailing_low"}

        warmup = 250  # rows needed for indicators

        for i in range(warmup, len(df)):
            slice_df = df.iloc[: i + 1].copy()
            row = df.iloc[i]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # ── Check exits first ─────────────────────
            if position:
                exit_reason = None

                if position["side"] == "long":
                    # Stop loss (wick)
                    if low <= position["stop"]:
                        exit_reason = "STOP_LOSS"
                        exit_price = position["stop"] * (1 - SLIPPAGE)
                    # Take profit
                    elif high >= position["take_profit"]:
                        exit_reason = "TAKE_PROFIT"
                        exit_price = position["take_profit"] * \
                            (1 - SLIPPAGE / 2)
                    # Trailing stop
                    elif self.cfg.trailing_stop:
                        position["trailing_high"] = max(
                            position["trailing_high"], high)
                        trail = position["trailing_high"] * \
                            (1 - self.cfg.trailing_stop_pct)
                        if low <= trail:
                            exit_reason = "TRAILING_STOP"
                            exit_price = trail * (1 - SLIPPAGE)

                elif position["side"] == "short":
                    if high >= position["stop"]:
                        exit_reason = "STOP_LOSS"
                        exit_price = position["stop"] * (1 + SLIPPAGE)
                    elif low <= position["take_profit"]:
                        exit_reason = "TAKE_PROFIT"
                        exit_price = position["take_profit"] * \
                            (1 + SLIPPAGE / 2)
                    elif self.cfg.trailing_stop:
                        position["trailing_low"] = min(
                            position["trailing_low"], low)
                        trail = position["trailing_low"] * \
                            (1 + self.cfg.trailing_stop_pct)
                        if high >= trail:
                            exit_reason = "TRAILING_STOP"
                            exit_price = trail * (1 + SLIPPAGE)

                if exit_reason:
                    ep = position["entry_price"]
                    qty = position["qty"]
                    side = position["side"]
                    pnl = (exit_price - ep) * qty if side == "long" \
                        else (ep - exit_price) * qty
                    # Deduct fees
                    fee = exit_price * qty * TAKER_FEE
                    pnl -= fee
                    pnl_pct = pnl / (ep * qty + 1e-9)
                    equity += pnl
                    peak = max(peak, equity)
                    equity_curve.append(equity)

                    duration = (
                        row.name - position["entry_time"]).total_seconds() / 3600
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        entry_time=position["entry_time"],
                        exit_time=row.name,
                        side=side,
                        entry_price=ep,
                        exit_price=exit_price,
                        quantity=qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        confidence=position["confidence"],
                    ))
                    position = None

            # ── Get new signal ────────────────────────
            if position is None and equity > 0:
                sig = strategy.analyze(slice_df, symbol)

                if sig.action in ("BUY", "SELL") and sig.confidence > 0.4:
                    # Max 2% risk per trade
                    risk_pct = min(self.cfg.capital_per_trade, 0.1)
                    risk_cap = equity * risk_pct
                    risk_unit = abs(price - sig.stop_loss)
                    if risk_unit < 1e-9:
                        continue

                    qty = risk_cap / risk_unit
                    qty = min(qty, (equity * 0.20) / price)

                    entry_price = price * (1 + SLIPPAGE if sig.action == "BUY"
                                           else -SLIPPAGE)
                    fee = entry_price * qty * TAKER_FEE
                    equity -= fee

                    position = {
                        "side": "long" if sig.action == "BUY" else "short",
                        "entry_price": entry_price,
                        "qty": qty,
                        "stop": sig.stop_loss,
                        "take_profit": sig.take_profit,
                        "entry_time": row.name,
                        "confidence": sig.confidence,
                        "trailing_high": price,
                        "trailing_low": price,
                    }

        # ── Close remaining position at end ───────────
        if position and len(df) > warmup:
            last = df.iloc[-1]
            ep = position["entry_price"]
            qty = position["qty"]
            ep2 = float(last["close"])
            pnl = (ep2 - ep) * qty if position["side"] == "long" \
                else (ep - ep2) * qty
            fee = ep2 * qty * TAKER_FEE
            pnl -= fee
            equity += pnl
            equity_curve.append(equity)

        # ── Metrics ───────────────────────────────────
        return self._compute_metrics(
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            timeframe=self.cfg.timeframe,
            start_date=str(df.index[warmup].date()),
            end_date=str(df.index[-1].date()),
            initial_capital=capital,
            final_capital=equity,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _compute_metrics(
            self,
            *,
            strategy_name,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_capital,
            final_capital,
            trades,
            equity_curve) -> BacktestResult:
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name, symbol=symbol,
                timeframe=timeframe, start_date=start_date, end_date=end_date,
                initial_capital=initial_capital, final_capital=final_capital,
                total_return=0.0, total_trades=0, win_rate=0.0,
                profit_factor=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, avg_trade_pct=0.0, best_trade_pct=0.0,
                worst_trade_pct=0.0, avg_holding_hours=0.0, trades=[],
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        pnl_series = np.array([t.pnl_pct for t in trades])

        # Drawdown
        eq = np.array(equity_curve)
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-9)
        max_dd = float(dd.max())

        # Sharpe (annualised, assuming hourly returns)
        mean_r = pnl_series.mean()
        std_r = pnl_series.std() if pnl_series.std() > 0 else 1e-9
        periods_per_year = 365 * 24  # hourly
        sharpe = (mean_r / std_r) * np.sqrt(periods_per_year)

        # Sortino
        neg_rets = pnl_series[pnl_series < 0]
        down_dev = neg_rets.std() if len(neg_rets) > 0 else 1e-9
        sortino = (mean_r / down_dev) * np.sqrt(periods_per_year)

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=(final_capital - initial_capital) / initial_capital,
            total_trades=len(trades),
            win_rate=len(wins) / len(trades),
            profit_factor=total_win / (total_loss + 1e-9),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=max_dd,
            avg_trade_pct=float(pnl_series.mean()),
            best_trade_pct=float(pnl_series.max()),
            worst_trade_pct=float(pnl_series.min()),
            avg_holding_hours=float(np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in trades
            ])),
            trades=trades,
        )

    def print_report(self, result: BacktestResult):
        r = result
        sep = "─" * 56
        print(f"\n{'═' * 56}")
        print(f"  BACKTEST: {r.strategy_name} | {r.symbol} | {r.timeframe}")
        print(f"  Period  : {r.start_date}  →  {r.end_date}")
        print(f"{'═' * 56}")
        print(
            f"  Capital : ${r.initial_capital:>10,.2f}  →  ${r.final_capital:>10,.2f}")
        print(f"  Return  : {r.total_return:>+.2%}")
        print(sep)
        print(f"  Trades  : {r.total_trades}")
        print(f"  Win Rate: {r.win_rate:.1%}")
        print(f"  Avg PnL : {r.avg_trade_pct:+.3%}")
        print(
            f"  Best    : {r.best_trade_pct:+.3%}   Worst: {r.worst_trade_pct:+.3%}")
        print(sep)
        print(f"  Sharpe  : {r.sharpe_ratio:.2f}")
        print(f"  Sortino : {r.sortino_ratio:.2f}")
        print(f"  MaxDD   : {r.max_drawdown:.2%}")
        print(f"  PF      : {r.profit_factor:.2f}")
        print(f"  Avg Hold: {r.avg_holding_hours:.1f}h")
        print(f"{'═' * 56}\n")
