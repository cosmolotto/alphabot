#!/usr/bin/env python3
"""
STRATEGY OPTIMIZER  —  v2  (Fixed overtrading + fee drag)

Root cause of all-red results:
  - Old signals fired every 5 candles → 30%+ fee drag
  - No trend filter → traded against the market
  - SL too tight → stopped out constantly

This version:
  - Requires 5+ conditions to align (confluence)
  - Only trades WITH the 200 EMA trend direction
  - Minimum candles between trades (cooldown)
  - ATR-based stops (adapts to volatility)
  - Targets 1 trade per 20-40 candles
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n):
    return s.rolling(n).mean()

def _rsi(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _macd(s, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    return m, _ema(m, sig), m - _ema(m, sig)

def _bb(s, n=20, std=2.0):
    mid   = _sma(s, n)
    sigma = s.rolling(n).std()
    return mid + std*sigma, mid, mid - std*sigma

def _atr(df, n=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(com=n-1, adjust=False).mean()

def _adx(df, n=14):
    atr_v = _atr(df, n)
    up = df['high'] - df['high'].shift()
    dn = df['low'].shift() - df['low']
    pdm = up.where((up > dn) & (up > 0), 0).ewm(com=n-1, adjust=False).mean()
    ndm = dn.where((dn > up) & (dn > 0), 0).ewm(com=n-1, adjust=False).mean()
    pdi = 100 * pdm / (atr_v + 1e-9)
    ndi = 100 * ndm / (atr_v + 1e-9)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(com=n-1, adjust=False).mean(), pdi, ndi

def _stoch(df, k=14, d=3):
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    ks = 100 * (df['close'] - lo) / (hi - lo + 1e-9)
    return ks.rolling(d).mean(), ks.rolling(d).mean().rolling(d).mean()

# ─────────────────────────────────────────────────────
#  STRATEGIES  (high-confluence, low trade frequency)
# ─────────────────────────────────────────────────────

def signal_trend(df, rsi_period, ema_fast, ema_slow, adx_thresh):
    """
    Trend following with 6-condition confluence.
    Only fires when EVERYTHING aligns — low frequency, high quality.
    """
    close  = df['close']
    rsi    = _rsi(close, rsi_period)
    ef     = _ema(close, ema_fast)
    es     = _ema(close, ema_slow)
    e200   = _ema(close, 200)
    adx_v, pdi, ndi = _adx(df, 14)
    macd_l, macd_sig, macd_h = _macd(close)
    vol_ma = df['volume'].rolling(20).mean()
    vol_ok = df['volume'] > vol_ma * 1.2

    # EMA slopes (trend must be accelerating)
    ef_rising  = ef > ef.shift(4)
    es_rising  = es > es.shift(8)
    ef_falling = ef < ef.shift(4)
    es_falling = es < es.shift(8)

    # LONG: need all 6
    long_sig = (
        (ef > es)            # fast above slow
        & ef_rising          # fast EMA rising
        & es_rising          # slow EMA rising
        & (close > e200)     # above 200 EMA (macro uptrend)
        & (rsi > 50)         # RSI bullish
        & (rsi < 75)         # not overbought
        & (adx_v > adx_thresh)  # trend is strong
        & (pdi > ndi)        # +DI above -DI
        & (macd_h > 0)       # MACD histogram positive
        & vol_ok             # volume confirms
    )

    # SHORT: need all 6
    short_sig = (
        (ef < es)
        & ef_falling
        & es_falling
        & (close < e200)
        & (rsi < 50)
        & (rsi > 25)
        & (adx_v > adx_thresh)
        & (ndi > pdi)
        & (macd_h < 0)
        & vol_ok
    )

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig


def signal_mean_rev(df, rsi_period, bb_period, adx_max):
    """
    Mean reversion — only trade when market is RANGING (low ADX).
    Requires price at extreme BB + RSI + Stochastic all agree.
    """
    close  = df['close']
    rsi    = _rsi(close, rsi_period)
    bb_u, bb_m, bb_l = _bb(close, bb_period, 2.0)
    bb_u2, _, bb_l2  = _bb(close, bb_period, 2.5)  # extreme band
    adx_v, pdi, ndi  = _adx(df, 14)
    k, d = _stoch(df, 14, 3)
    e50  = _ema(close, 50)
    vol_ma = df['volume'].rolling(20).mean()
    vol_ok = df['volume'] > vol_ma * 1.1

    ranging = adx_v < adx_max   # Only trade when not trending

    # LONG: price below lower BB + RSI oversold + stoch oversold + ranging
    long_sig = (
        (close < bb_l)       # below lower band
        & (close > bb_l2)    # but not at extreme crash level
        & (rsi < 35)         # RSI oversold
        & (k < 25)           # stochastic oversold
        & ranging            # market is ranging
        & vol_ok
    )

    # SHORT: price above upper BB + RSI overbought + ranging
    short_sig = (
        (close > bb_u)
        & (close < bb_u2)
        & (rsi > 65)
        & (k > 75)
        & ranging
        & vol_ok
    )

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig


def signal_breakout(df, lookback, vol_mult, adx_thresh):
    """
    Breakout strategy — buy when price breaks above recent high with volume.
    Very selective, fires only on genuine breakouts.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    e200   = _ema(close, 200)
    adx_v, pdi, ndi = _adx(df, 14)
    rsi    = _rsi(close, 14)
    vol_ma = df['volume'].rolling(20).mean()

    # Recent high/low (excluding current candle)
    prev_high = high.shift(1).rolling(lookback).max()
    prev_low  = low.shift(1).rolling(lookback).min()

    vol_spike = df['volume'] > vol_ma * vol_mult

    # LONG breakout
    long_sig = (
        (close > prev_high)     # break above resistance
        & vol_spike             # with high volume
        & (close > e200)        # in uptrend
        & (adx_v > adx_thresh)  # trend strengthening
        & (rsi > 50)
        & (rsi < 80)
        & (pdi > ndi)
    )

    # SHORT breakdown
    short_sig = (
        (close < prev_low)
        & vol_spike
        & (close < e200)
        & (adx_v > adx_thresh)
        & (rsi < 50)
        & (rsi > 20)
        & (ndi > pdi)
    )

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig


# ─────────────────────────────────────────────────────
#  BACKTESTER  (ATR-based stops + cooldown period)
# ─────────────────────────────────────────────────────

TAKER_FEE = 0.001    # 0.1%
SLIPPAGE   = 0.0003  # 0.03%

def backtest(df, signal_fn, atr_sl_mult, atr_tp_mult,
             capital=1000.0, cooldown=5):
    """
    atr_sl_mult: stop loss = entry ± ATR * multiplier
    atr_tp_mult: take profit = entry ± ATR * multiplier
    cooldown: minimum candles between trades
    """
    if len(df) < 250:
        return capital, 0.0, 0, 0.0

    try:
        signals = signal_fn(df)
    except Exception as e:
        return capital, -100.0, 0, 0.0

    atr_series = _atr(df, 14)

    equity      = capital
    n_trades    = 0
    n_wins      = 0
    position    = None
    last_exit   = -cooldown   # candle index of last exit

    warmup = 220

    for i in range(warmup, len(df)):
        if equity <= 0:
            break

        price = float(df['close'].iloc[i])
        high  = float(df['high'].iloc[i])
        low   = float(df['low'].iloc[i])
        atr_v = float(atr_series.iloc[i])

        # ── Check exits ──────────────────────────
        if position:
            exit_price = None
            won = False

            if position['side'] == 'long':
                if low  <= position['stop']:
                    exit_price = position['stop'] * (1 - SLIPPAGE)
                elif high >= position['take']:
                    exit_price = position['take']
                    won = True
            else:
                if high >= position['stop']:
                    exit_price = position['stop'] * (1 + SLIPPAGE)
                elif low  <= position['take']:
                    exit_price = position['take']
                    won = True

            if exit_price:
                qty = position['qty']
                ep  = position['entry']
                if position['side'] == 'long':
                    pnl = (exit_price - ep) * qty
                else:
                    pnl = (ep - exit_price) * qty
                pnl    -= exit_price * qty * TAKER_FEE
                equity += pnl
                n_trades += 1
                if won:
                    n_wins += 1
                position  = None
                last_exit = i

        # ── New entry (respect cooldown) ──────────
        if position is None and (i - last_exit) >= cooldown and equity > 10:
            sig = int(signals.iloc[i])
            if sig == 0:
                continue

            entry = price * (1 + SLIPPAGE if sig == 1 else -SLIPPAGE)

            sl_dist = atr_v * atr_sl_mult
            tp_dist = atr_v * atr_tp_mult

            if sig == 1:
                stop_price = entry - sl_dist
                take_price = entry + tp_dist
            else:
                stop_price = entry + sl_dist
                take_price = entry - tp_dist

            # Risk 2% of equity per trade — hard capped
            dist = abs(entry - stop_price)
            if dist < 1e-9 or dist > entry * 0.20:
                # Skip if stop is unreasonably far (> 20% away)
                continue

            max_risk_usdt = min(equity * 0.02, capital * 0.05)  # never risk more than 5% of original capital
            qty = min(
                max_risk_usdt / dist,
                (equity * 0.15) / entry    # max 15% of capital in one position
            )

            equity  -= entry * qty * TAKER_FEE
            position = {
                'side':  'long' if sig == 1 else 'short',
                'entry': entry,
                'stop':  stop_price,
                'take':  take_price,
                'qty':   qty,
            }

    # Close open position at last price
    if position:
        last  = float(df['close'].iloc[-1])
        qty   = position['qty']
        ep    = position['entry']
        pnl   = (last - ep)*qty if position['side']=='long' else (ep-last)*qty
        pnl  -= last * qty * TAKER_FEE
        equity += pnl
        n_trades += 1

    total_return = (equity - capital) / capital * 100
    win_rate     = n_wins / n_trades if n_trades > 0 else 0.0
    return equity, total_return, n_trades, win_rate


# ─────────────────────────────────────────────────────
#  DATA FETCHER
# ─────────────────────────────────────────────────────

def fetch_data(symbol, timeframe='4h', limit=1000):
    """4h candles = less noise, better signals than 1h"""
    try:
        import ccxt
        ex = ccxt.binance({'enableRateLimit': True})
        raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        print(f"  ✅ {symbol}: {len(df)} x {timeframe} candles")
        return df
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────
#  OPTIMIZER
# ─────────────────────────────────────────────────────

def run_optimization(symbols=None, timeframe='4h', output_file='trained_strategy.json'):
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']

    print("\n" + "═"*75)
    print("  STRATEGY OPTIMIZER  v2  —  High Confluence, Low Frequency")
    print("═"*75)

    # Fetch data
    print(f"\n📥 Fetching {timeframe} candles (less noise than 1h)...")
    data = {}
    for sym in symbols:
        df = fetch_data(sym, timeframe)
        if not df.empty:
            data[sym] = df
        time.sleep(0.8)

    if not data:
        print("❌ No data. Check internet + ccxt installed.")
        return

    # ── Parameter grids ───────────────────────────────
    # ATR multipliers for SL/TP
    atr_sl = [1.5, 2.0, 2.5, 3.0]
    atr_tp = [3.0, 4.0, 5.0, 6.0]   # always TP > SL for positive R:R
    cooldowns = [3, 5, 8]

    all_results = []
    test_num    = 0

    hdr = f"\n{'#':<5} {'Strategy':<16} {'Symbol':<12} {'SL_ATR':<8} {'TP_ATR':<8} {'CD':<5} {'Return':>9} {'Trades':>7} {'Win%':>7} {'R:R':>6}"
    print(hdr)
    print("─" * 85)

    for sym, df in data.items():

        # ── TREND FOLLOWING ───────────────────────────
        rsi_vals = [14, 20]
        ema_pairs = [(12,50), (21,100), (21,200), (50,200)]
        adx_vals  = [20, 25]

        for (ema_f, ema_s), rsi_p, adx_t, sl_m, tp_m, cd in itertools.product(
                ema_pairs, rsi_vals, adx_vals, atr_sl, atr_tp, cooldowns):
            if tp_m <= sl_m:
                continue    # enforce positive R:R
            test_num += 1
            fn = lambda d, r=rsi_p, ef=ema_f, es=ema_s, a=adx_t: signal_trend(d,r,ef,es,a)
            eq, ret, trades, wr = backtest(df, fn, sl_m, tp_m, cooldown=cd)
            rr = tp_m / sl_m
            result = dict(strategy='trend_following', symbol=sym,
                          params=dict(rsi=rsi_p, ema_fast=ema_f, ema_slow=ema_s,
                                      adx_thresh=adx_t, atr_sl=sl_m, atr_tp=tp_m, cooldown=cd),
                          return_pct=round(ret,2), n_trades=trades,
                          win_rate=round(wr,3), final_equity=round(eq,2),
                          rr_ratio=round(rr,2))
            all_results.append(result)
            bar = "🟢" if ret>5 else ("🟡" if ret>-10 else "🔴")
            extra = f"EMA{ema_f}/{ema_s} RSI{rsi_p} ADX>{adx_t}"
            print(f"{test_num:<5} {'trend':<16} {sym:<12} {sl_m:<8} {tp_m:<8} {cd:<5} {bar}{ret:>7.1f}%  {trades:>5}  {wr:>6.1%}  {rr:>5.1f}x")

        # ── MEAN REVERSION ────────────────────────────
        rsi_vals2 = [10, 14]
        bb_vals   = [20, 25]
        adx_max   = [22, 28]

        for rsi_p, bb_p, adx_m, sl_m, tp_m, cd in itertools.product(
                rsi_vals2, bb_vals, adx_max, atr_sl, atr_tp, cooldowns):
            if tp_m <= sl_m:
                continue
            test_num += 1
            fn = lambda d, r=rsi_p, b=bb_p, a=adx_m: signal_mean_rev(d,r,b,a)
            eq, ret, trades, wr = backtest(df, fn, sl_m, tp_m, cooldown=cd)
            rr = tp_m / sl_m
            result = dict(strategy='mean_reversion', symbol=sym,
                          params=dict(rsi=rsi_p, bb_period=bb_p,
                                      adx_max=adx_m, atr_sl=sl_m, atr_tp=tp_m, cooldown=cd),
                          return_pct=round(ret,2), n_trades=trades,
                          win_rate=round(wr,3), final_equity=round(eq,2),
                          rr_ratio=round(rr,2))
            all_results.append(result)
            bar = "🟢" if ret>5 else ("🟡" if ret>-10 else "🔴")
            print(f"{test_num:<5} {'mean_rev':<16} {sym:<12} {sl_m:<8} {tp_m:<8} {cd:<5} {bar}{ret:>7.1f}%  {trades:>5}  {wr:>6.1%}  {rr:>5.1f}x")

        # ── BREAKOUT ──────────────────────────────────
        lookbacks  = [10, 20, 30]
        vol_mults  = [1.5, 2.0, 2.5]
        adx_vals3  = [20, 25]

        for lb, vm, adx_t, sl_m, tp_m, cd in itertools.product(
                lookbacks, vol_mults, adx_vals3, atr_sl, atr_tp, cooldowns):
            if tp_m <= sl_m:
                continue
            test_num += 1
            fn = lambda d, l=lb, v=vm, a=adx_t: signal_breakout(d,l,v,a)
            eq, ret, trades, wr = backtest(df, fn, sl_m, tp_m, cooldown=cd)
            rr = tp_m / sl_m
            result = dict(strategy='breakout', symbol=sym,
                          params=dict(lookback=lb, vol_mult=vm,
                                      adx_thresh=adx_t, atr_sl=sl_m, atr_tp=tp_m, cooldown=cd),
                          return_pct=round(ret,2), n_trades=trades,
                          win_rate=round(wr,3), final_equity=round(eq,2),
                          rr_ratio=round(rr,2))
            all_results.append(result)
            bar = "🟢" if ret>5 else ("🟡" if ret>-10 else "🔴")
            print(f"{test_num:<5} {'breakout':<16} {sym:<12} {sl_m:<8} {tp_m:<8} {cd:<5} {bar}{ret:>7.1f}%  {trades:>5}  {wr:>6.1%}  {rr:>5.1f}x")

        time.sleep(0.2)

    # ── Sort & save ───────────────────────────────────
    all_results.sort(key=lambda x: (x['return_pct'], x['win_rate']), reverse=True)

    output = {
        'generated_at':  pd.Timestamp.now().isoformat(),
        'timeframe':     timeframe,
        'total_tests':   len(all_results),
        'profitable':    sum(1 for r in all_results if r['return_pct'] > 0),
        'top_20':        all_results[:20],
        'all_results':   all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Print top 15 ──────────────────────────────────
    print("\n" + "═"*85)
    print("  TOP 15 CONFIGURATIONS")
    print("═"*85)
    print(f"{'#':<4} {'Strategy':<16} {'Symbol':<12} {'Return':>9} {'Trades':>7} {'Win%':>7} {'R:R':>6}  Params")
    print("─"*85)

    for i, r in enumerate(all_results[:15], 1):
        medal = "🥇" if i==1 else ("🥈" if i==2 else ("🥉" if i==3 else f"  {i} "))
        params_str = " ".join(f"{k}={v}" for k,v in r['params'].items())
        bar = "🟢" if r['return_pct']>0 else ("🟡" if r['return_pct']>-10 else "🔴")
        print(f"{medal:<4} {r['strategy']:<16} {r['symbol']:<12} "
              f"{bar}{r['return_pct']:>7.1f}%  {r['n_trades']:>5}  "
              f"{r['win_rate']:>6.1%}  {r['rr_ratio']:>5.1f}x  {params_str}")

    n_profitable = sum(1 for r in all_results if r['return_pct'] > 0)
    print(f"\n📊 {len(all_results)} tests | {n_profitable} profitable | "
          f"Best: {all_results[0]['return_pct']:+.1f}%")
    print(f"💾 Saved → {output_file}")

    # Save best as simple config
    best = all_results[0]
    with open('best_strategy.json', 'w') as f:
        json.dump(best, f, indent=2)
    print(f"⭐ Best config → best_strategy.json")
    print(f"   {best['strategy']} on {best['symbol']} | "
          f"Return: {best['return_pct']:+.1f}% | "
          f"Win rate: {best['win_rate']:.1%} | "
          f"Trades: {best['n_trades']}")

    return all_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='+',
                   default=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'])
    p.add_argument('--timeframe', default='4h',
                   help='4h recommended (less noise than 1h)')
    p.add_argument('--output', default='trained_strategy.json')
    args = p.parse_args()

    run_optimization(
        symbols=args.symbols,
        timeframe=args.timeframe,
        output_file=args.output,
    )
