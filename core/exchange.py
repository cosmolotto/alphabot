"""
==============================================================
  EXCHANGE CONNECTOR

  Multi-exchange support via ccxt.
  Handles: data fetching, order placement, balance checks.
  Supports: Binance, Bybit, OKX, KuCoin, Kraken + more
==============================================================
"""

import time
import logging
import ccxt
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from config import EXCHANGE_CONFIGS, BOT_CONFIG

logger = logging.getLogger(__name__)


class ExchangeConnector:

    def __init__(self, exchange_name: str = "binance", paper_trading: bool = True):
        self.name = exchange_name
        self.paper_trading = paper_trading
        self._exchange = None
        self._init_exchange()

    def _init_exchange(self):
        cfg = EXCHANGE_CONFIGS.get(self.name, {})
        exchange_class = getattr(ccxt, self.name.replace("_futures", ""), None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self.name}' not found in ccxt")

        self._exchange = exchange_class(
            {
                "apiKey": cfg.get("apiKey", ""),
                "secret": cfg.get("secret", ""),
                "password": cfg.get("password", ""),
                "enableRateLimit": True,
                "options": cfg.get("options", {}),
            }
        )

        if cfg.get("sandbox"):
            self._exchange.set_sandbox_mode(True)

        if not self.paper_trading and cfg.get("apiKey"):
            logger.info(f"✅ Connected to {self.name} (LIVE trading)")
        elif not self.paper_trading:
            logger.warning(f"⚠️  {self.name}: No API key. Market data only.")
        else:
            logger.info(f"📝 Paper trading mode — {self.name}")

    # ──────────────────────────────────────────
    #  MARKET DATA
    # ──────────────────────────────────────────

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles. Returns DataFrame with columns:
        timestamp, open, high, low, close, volume
        """
        for attempt in range(3):
            try:
                raw = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not raw:
                    return pd.DataFrame()
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df.dropna(inplace=True)
                return df
            except ccxt.RateLimitExceeded:
                wait = 2**attempt
                logger.warning(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                logger.error(f"Network error fetching {symbol}: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break
        return pd.DataFrame()

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self._exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Ticker error {symbol}: {e}")
            return None

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        try:
            return self._exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Orderbook error {symbol}: {e}")
            return None

    def get_balance(self) -> Dict[str, float]:
        if self.paper_trading:
            return {"USDT": BOT_CONFIG.total_capital}
        try:
            balance = self._exchange.fetch_balance()
            return {
                currency: float(info["free"])
                for currency, info in balance.items()
                if isinstance(info, dict) and float(info.get("free", 0)) > 0
            }
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return {}

    # ──────────────────────────────────────────
    #  ORDER MANAGEMENT
    # ──────────────────────────────────────────

    def place_market_order(
        self, symbol: str, side: str, amount: float
    ) -> Optional[Dict]:
        """
        Place a market order.
        side: "buy" or "sell"
        amount: quantity in base asset
        """
        if self.paper_trading:
            ticker = self.fetch_ticker(symbol)
            price = float(ticker["last"]) if ticker else 0.0
            logger.info(f"[PAPER] {side.upper()} {amount:.6f} {symbol} @ ~{price:.4f}")
            return {
                "id": f"paper_{int(time.time())}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "closed",
                "paper": True,
            }

        try:
            order = self._exchange.create_market_order(symbol, side, amount)
            logger.info(
                f"✅ Market {side.upper()} {amount:.6f} {symbol} "
                f"| OrderID: {order['id']}"
            )
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for {side} {symbol}: {e}")
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order {symbol}: {e}")
        except Exception as e:
            logger.error(f"Order error {symbol}: {e}")
        return None

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Optional[Dict]:
        if self.paper_trading:
            logger.info(
                f"[PAPER] LIMIT {side.upper()} {amount:.6f} {symbol} @ {price:.4f}"
            )
            return {
                "id": f"paper_lim_{int(time.time())}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "open",
                "paper": True,
            }

        try:
            order = self._exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"✅ Limit {side.upper()} {amount:.6f} {symbol} @ {price:.4f}")
            return order
        except Exception as e:
            logger.error(f"Limit order error: {e}")
        return None

    def cancel_order(self, order_id: str, symbol: str):
        if self.paper_trading:
            logger.info(f"[PAPER] Cancel order {order_id}")
            return True
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
        return False

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        if self.paper_trading:
            return []
        try:
            return self._exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Fetch orders error: {e}")
            return []

    # ──────────────────────────────────────────
    #  MARKET INFO
    # ──────────────────────────────────────────

    def get_min_order_size(self, symbol: str) -> float:
        try:
            markets = self._exchange.load_markets()
            market = markets.get(symbol, {})
            limits = market.get("limits", {})
            return float(limits.get("amount", {}).get("min", 0.001))
        except BaseException:
            return 0.001

    def get_price_precision(self, symbol: str) -> int:
        try:
            markets = self._exchange.load_markets()
            market = markets.get(symbol, {})
            return int(market.get("precision", {}).get("price", 2))
        except BaseException:
            return 2

    def get_symbols(self) -> List[str]:
        try:
            markets = self._exchange.load_markets()
            return [
                s
                for s in markets
                if "/USDT" in s
                and markets[s].get("active")
                and markets[s].get("type") == "spot"
            ]
        except BaseException:
            return []
