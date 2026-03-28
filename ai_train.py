import random, subprocess, re, json

best_score = -999
best = None

strategies = ["trend_following","mean_reversion","multi_indicator"]
coins = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT"]

for i in range(1000000):

    strat = random.choice(strategies)
    coin = random.choice(coins)

    rsi = random.randint(10,50)
    ema = random.randint(20,200)
    sl = round(random.uniform(0.5,5),2)
    tp = round(random.uniform(1,10),2)

    cmd = [
        "python3","main.py",
        "--backtest",
        "--strategy",strat,
        "--symbol",coin
    ]

    result = subprocess.run(cmd,capture_output=True,text=True)

    m = re.search(r"Return\s*:\s*(-?\d+\.?\d*)",result.stdout)
    score = float(m.group(1)) if m else -999

    print(f"TEST {i} | {strat} | {coin} | RSI {rsi} EMA {ema} SL {sl} TP {tp} | Return {score}")

    if score > best_score:
        best_score = score
        best = {
            "strategy":strat,
            "coin":coin,
            "rsi":rsi,
            "ema":ema,
            "stoploss":sl,
            "takeprofit":tp,
            "score":score
        }

        with open("trained_strategy.json","w") as f:
            json.dump(best,f,indent=2)

        print("🔥 NEW BEST:",best)

