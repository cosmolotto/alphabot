"""
AlphaBot Reddit Marketing Bot
Monitors algo-trading and crypto subreddits.

Setup:
  pip install praw
  Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
"""

import os
import praw
import time
import json
import random
from datetime import datetime

APP_URL = "https://cosmolotto.gumroad.com/l/alphabot"
APP_NAME = "AlphaBot"
GITHUB_URL = "https://github.com/cosmolotto/alphabot"

MONITOR_SUBS = [
    "algotrading",
    "CryptoCurrency",
    "binance",
    "Trading",
    "CryptoMarkets",
    "Python",
]

PROMO_SUBS = [
    "SideProject",
    "algotrading",
    "Python",
    "indiehackers",
]

TRIGGER_KEYWORDS = [
    "trading bot",
    "algo trading",
    "algorithmic trading",
    "binance bot",
    "crypto bot",
    "python trading",
    "automated trading",
    "backtesting",
    "backtest strategy",
    "trading strategy python",
    "quantitative trading",
    "mean reversion",
    "momentum trading",
]

HELPFUL_REPLIES = [
    """Good question on algo trading. The biggest pitfall most beginners hit is overfitting — you get an amazing backtest and then it falls apart on live data.

Things that actually help:
1. Walk-forward testing (not just backtesting on the same window you optimized on)
2. Out-of-sample validation
3. Monte Carlo simulation to stress test edge cases

I built a bot called {app} ({url}) that does all three automatically before running live. Uses 5 pre-built strategies (MACD, RSI, Bollinger, Mean Reversion, Momentum) and runs 4 validation passes. Paper trading mode included so you can verify before going live.

Source is also on GitHub: {github}""",

    """The hardest part of building a trading bot is not the strategy — it's the risk management and knowing when to trust the backtest.

Common mistakes:
- Backtesting on the full dataset you used to optimize (look-ahead bias)
- Not accounting for fees and slippage
- Position sizing that's too aggressive

I open-sourced my bot at {github} — it includes a Walk-Forward analyzer, Monte Carlo stress testing, and a live trading bridge to Binance with proper stop-loss handling. You can also get the full packaged version at {url} if you want the pre-configured setup.

Happy to answer questions on the validation approach.""",

    """For Binance bots specifically, the key things that separate working bots from ones that bleed:

1. Never trade without a validated edge — Monte Carlo on your strategy, not just backtesting
2. Risk per trade: 0.5-1% max, never more
3. Drawdown limits: kill switch if you're down 10% from peak
4. Fees matter: Binance spot is 0.1%, futures 0.02-0.04%. Model these in your backtest.

I built {app} ({url}) which includes all of this plus 5 tested strategies. The Calmar ratio validation is what I'm most happy with — it forces a good risk/return profile before the bot goes live.""",
]

WEEKLY_POST_TITLE = [
    "Open-sourced my Python Binance trading bot after 6 months of backtesting — includes Walk-Forward & Monte Carlo",
    "[Side Project] AlphaBot — Python trading bot with 4-layer validation before live trading",
    "I built a crypto trading bot that refuses to run unless it passes Walk-Forward + Monte Carlo tests",
]

WEEKLY_POST_BODY = """**The problem:** 99% of trading bots are overfit. They backtest beautifully and lose money live.

**What I built:** AlphaBot — a Python trading bot for Binance that runs 4 validation layers before a single trade goes live:

1. **Walk-Forward Analysis** — tests across rolling windows, not just historical data
2. **Monte Carlo Simulation** — stress tests 1,000+ random scenarios
3. **Rolling Window Validation** — ensures edge is consistent over time
4. **Calmar Ratio Check** — enforces minimum risk-adjusted returns

**5 built-in strategies:**
- MACD Crossover
- RSI Mean Reversion
- Bollinger Band Breakout
- EMA Momentum
- Volume-weighted Momentum

**Also includes:**
- Paper trading mode (test before real money)
- Binance API integration with proper rate limiting
- Risk management: position sizing, stop-loss, daily loss limits
- Live dashboard with P&L tracking

**GitHub (free):** {github}
**Packaged kit with setup guide ($49):** {url}

Happy to answer questions about the validation approach or strategy logic.

---
*Built in Python, uses CCXT for exchange connectivity, SQLite for trade logging.*"""


def get_reddit_client():
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=f"{APP_NAME}/1.0 by u/{os.environ['REDDIT_USERNAME']}",
    )


def already_replied(post_id, log_file="replied_ab.json"):
    try:
        with open(log_file) as f:
            return post_id in json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def log_reply(post_id, log_file="replied_ab.json"):
    try:
        with open(log_file) as f:
            replied = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        replied = {}
    replied[post_id] = datetime.utcnow().isoformat()
    with open(log_file, "w") as f:
        json.dump(replied, f, indent=2)


def is_relevant(post):
    text = (post.title + " " + (post.selftext or "")).lower()
    return any(kw in text for kw in TRIGGER_KEYWORDS)


def monitor_and_reply(reddit, dry_run=False):
    replied_count = 0
    for sub_name in MONITOR_SUBS:
        sub = reddit.subreddit(sub_name)
        print(f"Scanning r/{sub_name}...")
        try:
            for post in sub.new(limit=25):
                if already_replied(post.id):
                    continue
                if not is_relevant(post):
                    continue
                if (time.time() - post.created_utc) / 3600 > 6:
                    continue
                reply = random.choice(HELPFUL_REPLIES).format(
                    app=APP_NAME, url=APP_URL, github=GITHUB_URL
                )
                print(f"\n[r/{sub_name}] {post.title[:80]}")
                if not dry_run:
                    post.reply(reply)
                    log_reply(post.id)
                    replied_count += 1
                    time.sleep(30)
                else:
                    print(f"[DRY RUN] Would reply")
                    log_reply(post.id)
        except Exception as e:
            print(f"  Error in r/{sub_name}: {e}")
    print(f"\nReplied to {replied_count} posts.")
    return replied_count


def post_weekly_promo(reddit, dry_run=False):
    log_file = "promo_log_ab.json"
    try:
        with open(log_file) as f:
            promo_log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        promo_log = {}
    week = datetime.utcnow().strftime("%Y-W%W")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    posted = 0
    for sub_name in PROMO_SUBS:
        key = f"{sub_name}:{week}"
        if key in promo_log:
            print(f"Already posted to r/{sub_name} this week.")
            continue
        title = random.choice(WEEKLY_POST_TITLE)
        body = WEEKLY_POST_BODY.format(url=APP_URL, github=GITHUB_URL)
        print(f"\nPosting to r/{sub_name}...")
        if not dry_run:
            try:
                reddit.subreddit(sub_name).submit(title, selftext=body)
                promo_log[key] = today
                with open(log_file, "w") as f:
                    json.dump(promo_log, f, indent=2)
                posted += 1
                time.sleep(60)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"[DRY RUN]")
            promo_log[key] = today
            with open(log_file, "w") as f:
                json.dump(promo_log, f, indent=2)
    print(f"\nPosted to {posted} subreddits.")
    return posted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["monitor", "promo", "both"], default="monitor")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reddit = get_reddit_client()
    print(f"Logged in as: {reddit.user.me()}")
    if args.mode in ("monitor", "both"):
        monitor_and_reply(reddit, dry_run=args.dry_run)
    if args.mode in ("promo", "both"):
        post_weekly_promo(reddit, dry_run=args.dry_run)
