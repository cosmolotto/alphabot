#
# MAIN BOT ORCHESTRATOR

#

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Optional

from config import BOT_CONFIG, STRATEGY_PARAMS
from core.exchange import ExchangeConnector
from core.risk_manager import RiskManager
from core.indicators import *
from strategies.multi_indicator import MultiIndicatorStrategy, Signal
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy

logger = logging.getLogger(__name__)

STRATEGIES = {
    "multi_indicator": MultiIndicatorStrategy,
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
}


class TradingBot:

    def __init__(self, config=None, params=None):
        self.cfg = config or BOT_CONFIG
        self.params = params or STRATEGY_PARAMS
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # Components
        # # MAIN BOT ORCHESTRATOR

        #

        self.exchange = ExchangeConnector(
            self.cfg.exchange,
            paper_trading=self.cfg.paper_trading,
        )
        self.risk = RiskManager(self.cfg)
        self.strategy = STRATEGIES[self.cfg.strategy](self.params)

        # State
        self.last_signals: Dict[str, Signal] = {}
        self.candle_cache: Dict[str, object] = {}
        self.start_time = datetime.utcnow()

        logger.info(
            f"🤖 Bot initialized | Exchange: {self.cfg.exchange} "
            f"| Strategy: {self.cfg.strategy} "
            f"| {'PAPER' if self.cfg.paper_trading else '🔴 LIVE'}"
        )

    # ──────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────

    def start(self):
        self.running = True
        logger.info("▶️  Bot started")
        while self.running:
            try:
                self._cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
            time.sleep(self.cfg.loop_interval)

    def start_async(self):
        # Start bot in background thread
        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()
        logger.info("🔄 Bot running in background thread")

    def stop(self):
        self.running = False
        logger.info("⏹️  Bot stopping...")

    # ──────────────────────────────────────────
    #  CYCLE
    # ──────────────────────────────────────────

    def _cycle(self):
        cycle_start = time.time()
        logger.debug(f"Cycle start | Open positions: {len(self.risk.positions)}")

        for symbol in self.cfg.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        stats = self.risk.get_stats()
        elapsed = time.time() - cycle_start
        logger.info(
            f"📊 Cycle done ({elapsed:.1f}s) | "
            f"Positions: {stats['open_positions']} | "
            f"Total PnL: {stats['total_pnl']:+.2f} USDT | "
            f"Daily: {stats['daily_pnl']:+.2f} | "
            f"Win Rate: {stats.get('win_rate', 0):.1%}"
        )

    def _process_symbol(self, symbol: str):
        # 1. Fetch candles
        df = self.exchange.fetch_ohlcv(symbol, self.cfg.timeframe, limit=500)
        if df.empty:
            return

        self.candle_cache[symbol] = df
        price = float(df["close"].iloc[-1])

        # 2. Check exits for existing positions
        if symbol in self.risk.positions:
            exit_reason = self.risk.check_exits(symbol, price)
            if exit_reason:
                order = self.exchange.place_market_order(
                    symbol,
                    "sell" if self.risk.positions[symbol].side == "long" else "buy",
                    self.risk.positions[symbol].quantity,
                )
                if order or self.cfg.paper_trading:
                    result = self.risk.close_position(symbol, price, exit_reason)
                    self._notify_trade_close(symbol, result)
            return  # Don't look for new signals while in a position on this symbol

        # 3. Risk gate
        if not self.risk.can_open_trade(symbol):
            return

        # 4. Generate signal
        signal = self.strategy.analyze(df, symbol)
        self.last_signals[symbol] = signal

        # 5. Act on signal
        if signal.action == "HOLD" or signal.confidence < 0.45:
            return

        # 6. Position size
        qty = self.risk.position_size(
            price=signal.price,
            stop_loss=signal.stop_loss,
            confidence=signal.confidence,
        )

        min_qty = self.exchange.get_min_order_size(symbol)
        if qty < min_qty:
            logger.debug(f"Quantity too small for {symbol}: {qty:.8f}")
            return

        # 7. Place order
        order_side = "buy" if signal.action == "BUY" else "sell"
        order = self.exchange.place_market_order(symbol, order_side, qty)

        if order or self.cfg.paper_trading:
            side = "long" if signal.action == "BUY" else "short"
            self.risk.open_position(
                symbol=symbol,
                side=side,
                price=signal.price,
                qty=qty,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                strategy=self.cfg.strategy,
                confidence=signal.confidence,
            )
            self._notify_trade_open(signal, qty)

    # ──────────────────────────────────────────
    #  NOTIFICATIONS
    # ──────────────────────────────────────────

    def _notify_trade_open(self, signal: Signal, qty: float):
        msg = (
            f"🟢 NEW TRADE\n"
            f"Symbol: {signal.symbol}\n"
            f"Action: {signal.action}\n"
            f"Price:  {signal.price:.6f}\n"
            f"Qty:    {qty:.6f}\n"
            f"SL:     {signal.stop_loss:.6f}\n"
            f"TP:     {signal.take_profit:.6f}\n"
            f"Conf:   {signal.confidence:.1%}\n"
            f"Reason: {signal.reason}"
        )
        logger.info(msg)
        self._send_telegram(msg)

    def _notify_trade_close(self, symbol: str, result):
        if result:
            emoji = "✅ PROFIT" if result.pnl > 0 else "❌ LOSS"
            msg = (
                f"{emoji}\n"
                f"Symbol: {symbol}\n"
                f"PnL:    {result.pnl:+.2f} USDT ({result.pnl_pct:+.2%})\n"
                f"Reason: {result.exit_reason}"
            )
            logger.info(msg)
            self._send_telegram(msg)

    def _send_telegram(self, message: str):
        if not self.cfg.enable_telegram or not self.cfg.telegram_token:
            return
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage"
            data = {"chat_id": self.cfg.telegram_chat_id, "text": message}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    # ──────────────────────────────────────────
    #  STATUS
    # ──────────────────────────────────────────

    def get_status(self) -> dict:
        stats = self.risk.get_stats()
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            **stats,
            "running": self.running,
            "uptime_s": uptime,
            "exchange": self.cfg.exchange,
            "strategy": self.cfg.strategy,
            "paper": self.cfg.paper_trading,
            "positions": {
                s: {
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                }
                for s, p in self.risk.positions.items()
            },
            "signals": {
                s: {
                    "action": sig.action,
                    "confidence": sig.confidence,
                    "price": sig.price,
                    "reason": sig.reason,
                }
                for s, sig in self.last_signals.items()
            },
        }
