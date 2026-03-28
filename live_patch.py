import json, os, sys

print("ALPHABOT LIVE SETUP - BINANCE CONFIGURATION")

cfg = {}
if os.path.exists('bot_config.json'):
    with open('bot_config.json') as f:
        cfg = json.load(f)

print("Capital in USDT (e.g. 20): ", end='')
capital = float(input().strip() or '20')

print("Binance API Key: ", end='')
api_key = input().strip()

print("Binance Secret Key: ", end='')
api_secret = input().strip()

if not api_key or not api_secret:
    print("Keys required")
    sys.exit(1)

cfg.update({
    'capital': capital,
    'api_key': api_key,
    'api_secret': api_secret,
    'exchange': 'binance',
    'risk_per_trade': 0.20 if capital < 20 else 0.015,
    'max_open_trades': 1 if capital < 20 else 5,
    'max_daily_loss_pct': 0.15,
    'max_drawdown_pct': 0.20,
    'min_confidence': 70.0,
    'trailing_pct': 0.015,
    'use_btc_filter': True,
    'use_mtf': True,
    'use_time_filter': True,
    'use_trust': True,
    'min_momentum': 35.0,
})

with open('bot_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)

print("Saved! Capital: ${:.0f} | Risk: {:.0f}% | Max trades: {}".format(
    capital, cfg['risk_per_trade']*100, cfg['max_open_trades']))
print("Now run: python alphabot.py --live --bot --port 8081")
