"""
AlphaBot Twitter/X Marketing Bot
Posts daily algo-trading tips and weekly promos.

Setup:
  pip install tweepy
  Set env vars: AB_TWITTER_API_KEY, AB_TWITTER_API_SECRET,
                AB_TWITTER_ACCESS_TOKEN, AB_TWITTER_ACCESS_SECRET
"""

import os
import random
import tweepy

APP_URL = "https://cosmolotto.gumroad.com/l/alphabot"
GITHUB_URL = "https://github.com/cosmolotto/alphabot"

DAILY_TIPS = [
    "Walk-forward analysis > backtesting on full history. If your strategy can't survive rolling out-of-sample windows, it won't survive live markets. Test on what the model never saw.",
    "The #1 reason trading bots fail live: look-ahead bias. Your backtest uses data the bot couldn't have known at trade time. Walk-forward testing is the fix. 📊",
    "Monte Carlo simulation on your trading strategy: run 1,000 random orderings of your trade results. If the strategy survives the worst 5% of simulations, it has a real edge.",
    "Position sizing rule for algo traders: risk 0.5–1% of account per trade. Sounds boring. After 500 trades, it's the difference between still trading and blown account.",
    "Calmar Ratio = annualized return / max drawdown. A ratio above 1.0 means you earn more than you lose in your worst streak. Set this as a hard filter before going live.",
    "Binance bot fees add up fast: 0.1% per trade × 2 (entry + exit) = 0.2% per round trip. That's $200 per $100k traded. Model every trade with real fees or your backtest is fiction.",
    "The best trading strategies are boring: mean reversion, momentum, breakout. The edge isn't the idea — it's the validation, risk management, and discipline to let it run.",
    "A kill switch is not optional. If your bot hits 10% drawdown from peak, stop trading and review. Markets change. No strategy works forever without monitoring.",
    "Backtesting tip: use at least 3 years of data across different market regimes (bull, bear, sideways). A strategy that only works in one regime will fail when the regime shifts.",
    "Why most retail algo traders fail: they optimize strategy parameters on the full dataset, then wonder why live results don't match. Reserve 30% of data for out-of-sample testing — always.",
    "RSI divergence is one of the highest signal-to-noise setups in crypto: price makes new low, RSI makes higher low. Not 100% reliable, but combined with volume confirmation it's solid.",
    "Paper trading isn't optional — it's mandatory. Run every new strategy for at least 30 days in paper mode before risking real capital. The market will teach you what backtests can't.",
]

PROMO_TWEETS = [
    f"I built a Python trading bot that won't run unless it passes:\n\n✅ Walk-Forward Analysis\n✅ Monte Carlo (1,000+ scenarios)\n✅ Rolling Window Validation\n✅ Calmar Ratio Check\n\nMost bots skip all of this. That's why they fail.\n\n→ {APP_URL}",
    f"AlphaBot: 5 built-in strategies, 4 validation layers, paper trading mode.\n\nMACD · RSI · Bollinger · Mean Reversion · Momentum\n\nBuilt in Python, connects to Binance, open-sourced.\n\nGitHub (free): {GITHUB_URL}\nPackaged kit: {APP_URL}",
    f"The hard part of algo trading isn't writing the strategy.\n\nIt's knowing when to trust the backtest.\n\nAlphaBot solves this with Walk-Forward + Monte Carlo validation before a single live trade.\n\n{APP_URL}",
]


def post_daily_tip(client, dry_run=False):
    tip = random.choice(DAILY_TIPS)
    print(f"Posting tip:\n{tip[:100]}...")
    if not dry_run:
        client.create_tweet(text=tip)
        print("Posted!")
    else:
        print("[DRY RUN] Would post tip.")


def post_promo(client, dry_run=False):
    tweet = random.choice(PROMO_TWEETS)
    print(f"Posting promo:\n{tweet[:100]}...")
    if not dry_run:
        client.create_tweet(text=tweet)
        print("Posted!")
    else:
        print("[DRY RUN] Would post promo.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tip", "promo"], default="tip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = tweepy.Client(
        consumer_key=os.environ["AB_TWITTER_API_KEY"],
        consumer_secret=os.environ["AB_TWITTER_API_SECRET"],
        access_token=os.environ["AB_TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["AB_TWITTER_ACCESS_SECRET"],
    )

    if args.mode == "tip":
        post_daily_tip(client, dry_run=args.dry_run)
    elif args.mode == "promo":
        post_promo(client, dry_run=args.dry_run)
