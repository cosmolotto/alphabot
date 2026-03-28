import subprocess,re

best=-999
best_params=None

for i in range(100):

    print(f"\nTEST {i+1}")

    cmd="python3 main.py --backtest --strategy trend_following --symbol BTC/USDT"

    result=subprocess.run(cmd,shell=True,capture_output=True,text=True)

    output=result.stdout+result.stderr

    m=re.search(r"Return\s*:\s*(-?\d+\.?\d*)",output)

    score=float(m.group(1)) if m else -999

    print("Return:",score)

    if score>best:
        best=score
        best_params=i
        print("🔥 NEW BEST:",best)

print("\nBEST RESULT:",best)
