"""
AlphaBot — Hacker News "Show HN" submission helper.
Generates the Show HN post and opens the submission URL in browser.

Usage:
  python hn_post.py --open
  python hn_post.py --print
"""

import argparse
import urllib.parse
import webbrowser

APP_URL = "https://cosmolotto.gumroad.com/l/alphabot"
GITHUB_URL = "https://github.com/cosmolotto/alphabot"

TITLE = "Show HN: AlphaBot – Python trading bot that won't run unless it passes Walk-Forward + Monte Carlo"

TEXT = """Most retail algorithmic trading bots share a common problem: they backtest beautifully and then lose money live. The reason is almost always the same — the backtest was run on the same data used to optimize the parameters.

I built AlphaBot to force myself to solve this before trading real money.

The bot requires four validation passes before it runs live:

1. Walk-Forward Analysis — tests the strategy across rolling out-of-sample windows. If it can't generalize to data it's never seen, it doesn't trade.
2. Monte Carlo Simulation — runs 1,000+ random orderings of the trade history to stress-test the edge. You see the worst-case equity curve, not just the average.
3. Rolling Window Validation — checks that the edge is consistent over time, not just good in one regime.
4. Calmar Ratio Check — enforces minimum risk-adjusted returns before live deployment.

Technical details:
- Python 3.11+, CCXT for exchange connectivity (Binance supported)
- 5 built-in strategies: MACD Crossover, RSI Mean Reversion, Bollinger Band Breakout, EMA Momentum, Volume-weighted Momentum
- Paper trading mode — simulate against real market data without risking capital
- Risk management: position sizing (0.5-1% per trade), daily loss limits, kill switch at 10% drawdown from peak
- SQLite for trade logging and performance tracking

What I've learned running this live:
- The validation layers catch ~70% of strategies I would have otherwise deployed. Most fail Walk-Forward.
- Fees destroy more edges than people think. Model them exactly — don't round down.
- Paper trading for 30 days before live is non-negotiable. The bot behaved differently than I expected in volatile conditions.

GitHub (free, MIT): {github}
Packaged kit with setup guide: {url}

Happy to discuss the Walk-Forward implementation, Monte Carlo approach, or risk management logic.""".format(
    github=GITHUB_URL, url=APP_URL
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="Open HN submission in browser")
    parser.add_argument("--print", action="store_true", help="Print the post text")
    args = parser.parse_args()

    params = urllib.parse.urlencode({"title": TITLE, "url": APP_URL, "text": TEXT})
    hn_url = f"https://news.ycombinator.com/submitlink?{params}"

    if args.print or not args.open:
        print("=== HACKER NEWS SHOW HN ===")
        print(f"Title: {TITLE}")
        print(f"URL: {APP_URL}")
        print(f"\nText:\n{TEXT}")
        print(f"\nSubmission URL:\n{hn_url}")

    if args.open:
        print("Opening HN submission in browser...")
        webbrowser.open(hn_url)


if __name__ == "__main__":
    main()
