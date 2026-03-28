"""
==============================================================
  CRYPTO TRADING BOT - Configuration
  Supports: Binance, Bybit, OKX, KuCoin, Coinbase, Kraken
==============================================================
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

# ──────────────────────────────────────────────
#  EXCHANGE CREDENTIALS (use .env or set directly)
# ──────────────────────────────────────────────
EXCHANGE_CONFIGS = {
    "binance": {
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_SECRET", ""),
        "sandbox": False,  # Set True to use Binance Testnet
        "options": {"defaultType": "spot"},
    },
    "binance_futures": {
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_SECRET", ""),
        "sandbox": False,
        "options": {"defaultType": "future"},
    },
    "bybit": {
        "apiKey": os.getenv("BYBIT_API_KEY", ""),
        "secret": os.getenv("BYBIT_SECRET", ""),
        "sandbox": False,
    },
    "okx": {
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", ""),
        "sandbox": False,
    },
    "kucoin": {
        "apiKey": os.getenv("KUCOIN_API_KEY", ""),
        "secret": os.getenv("KUCOIN_SECRET", ""),
        "password": os.getenv("KUCOIN_PASSPHRASE", ""),
    },
    "kraken": {
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_SECRET", ""),
    },
}

# ──────────────────────────────────────────────
#  BOT SETTINGS
# ──────────────────────────────────────────────


@dataclass
class BotConfig:
    # Exchange to use
    exchange: str = "binance"

    # Trading pairs to monitor & trade
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
        ]
    )

    # Primary timeframe for signals
    timeframe: str = "1h"  # 1m 5m 15m 1h 4h 1d

    # Strategy to use
    # Options: "multi_indicator", "trend_following", "mean_reversion",
    #          "momentum", "scalping", "ml_hybrid"
    strategy: str = "multi_indicator"

    # Capital allocation
    base_currency: str = "USDT"
    total_capital: float = 1000.0  # Total USDT to trade with
    capital_per_trade: float = 0.10  # 10% of capital per trade

    # Risk management
    max_open_trades: int = 5
    max_daily_loss_pct: float = 0.05  # Stop trading if -5% in a day
    stop_loss_pct: float = 0.025  # 2.5% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5% trailing

    # Paper trading (RECOMMENDED for first run)
    paper_trading: bool = True

    # Loop interval (seconds between bot cycles)
    loop_interval: int = 60

    # Dashboard
    dashboard_port: int = 8080
    dashboard_host: str = "127.0.0.1"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    # Notifications (optional)
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_telegram: bool = False


# ──────────────────────────────────────────────
#  STRATEGY PARAMETERS
# ──────────────────────────────────────────────
@dataclass
class StrategyParams:
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # EMA
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 200

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # ATR (for dynamic stops)
    atr_period: int = 14
    atr_multiplier: float = 2.0

    # Volume filter
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.2  # Price action must have 1.2x avg volume

    # ADX (trend strength)
    adx_period: int = 14
    adx_threshold: float = 25.0  # Only trade when trend is strong


BOT_CONFIG = BotConfig()
STRATEGY_PARAMS = StrategyParams()
