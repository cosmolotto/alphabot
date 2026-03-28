#!/usr/bin/env python3
"""
==============================================================
  CRYPTO TRADING BOT — Main Entry Point

  Usage:
    python main.py              # Start bot + dashboard
    python main.py --backtest   # Run backtest on BTC/USDT
    python main.py --paper      # Force paper trading
    python main.py --live       # Enable live trading (dangerous!)
    python main.py --symbol BTC/USDT ETH/USDT  # Custom symbols
==============================================================
"""

from dashboard.app import create_dashboard
from bot import TradingBot
from config import BOT_CONFIG, STRATEGY_PARAMS
import sys
import argparse
import logging
import threading
from pathlib import Path

# ── Setup ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO", to_file: bool = True):
    handlers = [logging.StreamHandler(sys.stdout)]
    if to_file:
        Path("logs").mkdir(exist_ok=True)
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler("logs/bot.log", maxBytes=10_000_000, backupCount=5)
        )
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)


def run_backtest(strategy_name: str = "multi_indicator"):
    """Run a full backtest and print results."""
    from backtest.engine import Backtester
    from strategies.multi_indicator import MultiIndicatorStrategy
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from core.exchange import ExchangeConnector

    STRAT_MAP = {
        "multi_indicator": MultiIndicatorStrategy,
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
    }

    strategy_cls = STRAT_MAP.get(strategy_name, MultiIndicatorStrategy)
    exchange = ExchangeConnector("binance", paper_trading=True)
    backtester = Backtester(strategy_cls, STRATEGY_PARAMS, BOT_CONFIG)

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    for symbol in symbols:
        print(f"\n⏳ Fetching 10000 candles for {symbol}...")
        df = exchange.fetch_ohlcv(symbol, BOT_CONFIG.timeframe, limit=10000)
        if df.empty:
            print(f"  No data for {symbol}")
            continue

        result = backtester.run(df, symbol, initial_capital=10000.0)
        backtester.print_report(result)


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest instead of live bot"
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument(
        "--live", action="store_true", help="Enable live trading (USE WITH CAUTION)"
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["multi_indicator", "trend_following", "mean_reversion"],
        help="Override strategy",
    )
    parser.add_argument(
        "--exchange", default=None, help="Override exchange (e.g. binance, bybit, okx)"
    )
    parser.add_argument(
        "--symbol", nargs="+", default=None, help="Override trading symbols"
    )
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Disable web dashboard"
    )
    args = parser.parse_args()

    setup_logging(BOT_CONFIG.log_level, BOT_CONFIG.log_to_file)
    log = logging.getLogger(__name__)

    # Apply overrides
    if args.paper:
        BOT_CONFIG.paper_trading = True
    if args.live:
        print(
            "\n⚠️  LIVE TRADING ENABLED — real funds will be used!\n"
            "   Press Ctrl+C to cancel or ENTER to continue..."
        )
        input()
        BOT_CONFIG.paper_trading = False
    if args.strategy:
        BOT_CONFIG.strategy = args.strategy
    if args.exchange:
        BOT_CONFIG.exchange = args.exchange
    if args.symbol:
        BOT_CONFIG.symbols = args.symbol
    if args.port:
        BOT_CONFIG.dashboard_port = args.port

    # Backtest mode
    if args.backtest:
        run_backtest(BOT_CONFIG.strategy)
        return

    # Print banner
    mode = "📝 PAPER" if BOT_CONFIG.paper_trading else "🔴 LIVE"
    print(
        f"""
╔══════════════════════════════════════════╗
║         CRYPTO TRADING BOT v1.0          ║
╠══════════════════════════════════════════╣
║  Mode     : {mode:<30}║
║  Exchange : {BOT_CONFIG.exchange:<30}║
║  Strategy : {BOT_CONFIG.strategy:<30}║
║  Capital  : ${BOT_CONFIG.total_capital:<29,.0f}║
║  Symbols  : {len(BOT_CONFIG.symbols)} pairs{' '*24}║
╠══════════════════════════════════════════╣
║  Dashboard: http://127.0.0.1:{BOT_CONFIG.dashboard_port:<12}║
╚══════════════════════════════════════════╝
"""
    )

    # Create bot
    bot = TradingBot(BOT_CONFIG, STRATEGY_PARAMS)

    # Dashboard thread
    if not args.no_dashboard:
        dash_app = create_dashboard(bot)
        dash_thread = threading.Thread(
            target=lambda: dash_app.run(
                host=BOT_CONFIG.dashboard_host,
                port=BOT_CONFIG.dashboard_port,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        )
        dash_thread.start()
        log.info(
            f"🌐 Dashboard running at "
            f"http://{BOT_CONFIG.dashboard_host}:{BOT_CONFIG.dashboard_port}"
        )

    # Start bot
    try:
        bot.start()  # blocks
    except KeyboardInterrupt:
        log.info("⚡ Shutdown signal received")
        bot.stop()
        stats = bot.risk.get_stats()
        print(f"\n📊 Final Stats:")
        print(f"   Trades  : {stats['total_trades']}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"   Total PnL: {stats['total_pnl']:+.2f} USDT")
        print(f"   Max DD  : {stats.get('max_drawdown', 0):.2%}")
        print(f"   Sharpe  : {stats.get('sharpe_ratio', 0):.2f}")


if __name__ == "__main__":
    main()

parser.add_argument("--rsi", type=int, default=14)
parser.add_argument("--ema", type=int, default=50)
parser.add_argument("--stoploss", type=float, default=2.0)
parser.add_argument("--takeprofit", type=float, default=4.0)
