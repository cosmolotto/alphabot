# 🤖 Crypto Trading Bot

A professional, multi-exchange crypto trading bot using official exchange APIs.
**100% legal — uses official Binance/Bybit/OKX REST APIs.**

---

## ⚡ Quick Start (MacBook M4)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your exchange API keys

# 4. Run in PAPER TRADING mode (no real money)
python main.py --paper

# 5. Open dashboard
open http://127.0.0.1:8080
```

---

## 🏗️ Architecture

```
trading_bot/
├── main.py               ← Entry point
├── bot.py                ← Main trading loop
├── config.py             ← All configuration
├── requirements.txt
├── .env.example
├── core/
│   ├── exchange.py       ← ccxt multi-exchange connector
│   ├── indicators.py     ← 20+ technical indicators (pure NumPy)
│   └── risk_manager.py   ← Position sizing, stops, drawdown control
├── strategies/
│   ├── multi_indicator.py  ← Confluence strategy (default)
│   ├── trend_following.py  ← HMA + Supertrend
│   └── mean_reversion.py   ← BB + RSI extremes
├── backtest/
│   └── engine.py         ← Realistic backtester with fees + slippage
└── dashboard/
    └── app.py            ← Flask live dashboard
```

---

## 🎯 Strategies

### 1. Multi-Indicator Confluence (Default, Recommended)
Requires **≥4 bullish signals** to align before buying:
- EMA 9/21/200 alignment
- ADX trend strength > 25
- Supertrend direction
- RSI oversold/overbought zones
- MACD crossover
- Stochastic extremes
- Bollinger Band position
- Volume confirmation (1.2x average)
- VWAP position
- Chaikin Money Flow

**Best for:** All market conditions. Most conservative and accurate.

### 2. Trend Following
- Hull MA crossover (9/21)
- Supertrend direction
- ADX trend strength filter
- EMA 200 macro trend

**Best for:** Strong trending markets (BTC bull runs, alt seasons)

### 3. Mean Reversion
- Bollinger Band extremes
- RSI + Stochastic + MFI triple confluence
- Williams %R confirmation

**Best for:** Sideways, ranging markets

---

## ⚙️ Configuration

Edit `config.py`:

```python
BOT_CONFIG.symbols         = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
BOT_CONFIG.timeframe       = "1h"    # 1m, 5m, 15m, 1h, 4h, 1d
BOT_CONFIG.total_capital   = 1000    # USDT to trade with
BOT_CONFIG.capital_per_trade = 0.10  # 10% per trade
BOT_CONFIG.stop_loss_pct   = 0.025   # 2.5% stop loss
BOT_CONFIG.take_profit_pct = 0.05    # 5% take profit
BOT_CONFIG.max_open_trades = 5       # Max simultaneous positions
BOT_CONFIG.paper_trading   = True    # False = real money
```

---

## 📊 Risk Management

The bot has multiple layers of protection:
1. **Stop Loss** — Fixed % or ATR-based
2. **Trailing Stop** — Follows price, locks in profits
3. **Take Profit** — Automatic profit taking
4. **Daily Loss Limit** — Stops trading if -5% in one day
5. **Max Drawdown** — Pauses if equity drops 15% from peak
6. **Position Sizing** — Kelly Criterion + confidence scaling
7. **Max Open Trades** — Never overexposes the portfolio
8. **Volume Filter** — Requires above-average volume

---

## 📈 Backtesting

```bash
# Backtest all strategies on BTC/USDT, ETH/USDT, BNB/USDT
python main.py --backtest

# Backtest specific strategy
python main.py --backtest --strategy trend_following
```

Output includes:
- Total return
- Win rate
- Sharpe & Sortino ratios
- Max drawdown
- Profit factor
- Best/worst trade

---

## 🔐 Exchange API Setup

### Binance (Recommended)
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create API key → Enable **Spot & Margin Trading**
3. **IMPORTANT:** Disable futures, withdrawals (bot doesn't need them)
4. Whitelist your IP address
5. Add to `.env`

### Bybit / OKX / KuCoin
Same process — each exchange has API management in account settings.
Enable **Read + Trade** only. Never enable **Withdraw**.

---

## 🚨 Critical Risk Warnings

> ⚠️ **Crypto trading is highly risky. You can lose all your capital.**
>
> - ALWAYS start with paper trading first
> - Never trade more than you can afford to lose
> - Past backtest performance does NOT guarantee future results
> - Markets can behave unexpectedly — monitor the bot regularly
> - No bot can guarantee profits

---

## 💡 Tips for Best Results

1. **Start with paper trading** for at least 2-4 weeks
2. **Use the `1h` timeframe** — less noise than 5m/15m
3. **Trade BTC, ETH, BNB first** — most liquid, tightest spreads
4. **Review the dashboard daily** — check win rate and drawdown
5. **Run backtests** before changing strategy parameters
6. **Keep API key permissions minimal** — Read + Trade only, no withdrawal

---

## 📬 Telegram Notifications

Get alerts when trades open/close:
1. Create a bot via @BotFather on Telegram
2. Get your Chat ID via @userinfobot
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Enable in config: `BOT_CONFIG.enable_telegram = True`

---

## 🛠️ CLI Options

```
python main.py --paper              Paper trading mode
python main.py --live               Live trading (real money!)
python main.py --backtest           Run backtest
python main.py --strategy X        Set strategy
python main.py --exchange bybit    Use Bybit
python main.py --symbol BTC/USDT ETH/USDT   Custom symbols
python main.py --port 9090         Custom dashboard port
python main.py --no-dashboard      Headless mode
```

---

*Built with ccxt, pandas, numpy, flask.*
*Legally compliant with Binance, Bybit, OKX, KuCoin, Kraken Terms of Service.*
*Using only official public REST APIs.*
