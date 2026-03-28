import random, subprocess, re, json

best_score = -999
best_params = None

coins = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT"]
strategies = ["trend_following","mean_reversion","multi_indicator"]

def run_test(strategy, coin):
    cmd = ["python3","main.py","--backtest","--strategy",strategy,"--symbol",coin]
    result = subprocess.run(cmd,capture_output=True,text=True)
    output = result.stdout

    m = re.search(r"Return\s*:\s*(-?\d+\.?\d*)",output)
    score = float(m.group(1)) if m else -999
    return score

for i in range(1000000):

    strategy = random.choice(strategies)
    coin = random.choice(coins)

    score = run_test(strategy,coin)

    print(f"TEST {i} | {strategy} | {coin} | Return {score}")

    if score > best_score:
        best_score = score
        best_params = {"strategy":strategy,"coin":coin}

        with open("best_strategy.json","w") as f:
            json.dump({"score":best_score,"params":best_params},f,indent=2)

        print("🔥 NEW BEST:",best_score,best_params)
