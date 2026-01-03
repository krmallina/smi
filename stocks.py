"""
Stock Market Intelligence Dashboard

Trading Strategies:
  The system supports multiple trading signal strategies. Set via TRADING_STRATEGY environment variable:
  
  - 'bb_ichimoku' (default): BB or Ichimoku - Signal if either BB or Ichimoku triggers
  - 'bb': Bollinger Bands - Buy at lower band, Short at upper band
  - 'rsi': RSI - Buy when oversold (<30), Short when overbought (>70)
  - 'macd': MACD - Buy on bullish crossover, Short on bearish crossover
  - 'ichimoku': Ichimoku Cloud - Buy on bullish cloud break, Sell on bearish cloud break
  - 'combined': Combined - Requires 2+ strategies to agree
  
  Example usage:
    export TRADING_STRATEGY=ichimoku
    python3 stocks.py data/tickers.csv
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json
import requests
import random
import re
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import threading

ALERTS_FILE = "data/alerts.json"
UTC = pytz.utc
PST = pytz.timezone("America/Los_Angeles")
MEME_STOCKS = frozenset(
    {
        "GME",
        "AMC",
        "BB",
        "KOSS",
        "EXPR",
        "DJT",
        "HOOD",
        "RDDT",
        "SPCE",
        "RIVN",
        "DNUT",
        "OPEN",
        "KSS",
        "RKLB",
        "GPRO",
        "AEO",
        "BYND",
        "CVNA",
        "PLTR",
        "SMCI",
    }
)

# Category buckets for filtering chips
CATEGORY_MAP = {
    "major-tech": frozenset({"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}),
    "leveraged-etf": frozenset({"TQQQ", "SPXL", "AAPU", "PLTU"}),
    "sector-etf": frozenset({"SPY", "XLF", "SMH", "XBI"}),
    "spec-meme": MEME_STOCKS,
    "emerging-tech": frozenset({"OKLO", "SMR", "CRWV", "RKLB"}),
}


def get_category(ticker):
    tu = ticker.upper()
    for slug, tickers in CATEGORY_MAP.items():
        if tu in tickers:
            return slug
    return ""


def infer_category_from_info(ticker, info):
    tu = ticker.upper()
    # Start with manual map if present
    mapped = get_category(tu)
    if mapped:
        return mapped

    qtype = (info.get("quoteType") or "").lower()
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    lname = (info.get("longName") or "").lower()
    sname = (info.get("shortName") or "").lower()

    # Detect leveraged/inverse ETFs via quoteType or naming
    if qtype == "etf" or "etf" in lname or "etf" in sname:
        lever_markers = (
            "3x",
            "2x",
            "ultra",
            "ultrapro",
            "leveraged",
            "inverse",
            "-1x",
            "-2x",
            "-3x",
        )
        if (
            any(m in tu.lower() for m in ("3x", "2x", "ultra", "pro", "bull", "bear"))
            or any(m in lname for m in lever_markers)
            or any(m in sname for m in lever_markers)
        ):
            return "leveraged-etf"
        return "sector-etf"

    # Meme / speculative
    if tu in MEME_STOCKS:
        return "spec-meme"

    # Major tech/growth heuristics
    if sector == "technology" or any(
        k in industry for k in ("semiconductor", "software", "ai")
    ):
        return "major-tech"

    # Emerging tech / energy heuristics
    if any(
        k in industry for k in ("nuclear", "battery", "clean energy", "solar", "space")
    ) or any(k in lname for k in ("nuclear", "battery", "rocket", "fusion", "space")):
        return "emerging-tech"

    return ""


# ============================================================================
# TRADING SIGNAL STRATEGIES
# ============================================================================

def calculate_ichimoku(high, low, close, conversion_period=9, base_period=26, span_b_period=52):
    """
    Calculate Ichimoku Cloud indicators.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        conversion_period: Period for Conversion Line (default 9)
        base_period: Period for Base Line (default 26)
        span_b_period: Period for Span B (default 52)
    
    Returns:
        Dictionary with Ichimoku components or None if insufficient data
    """
    if len(high) < span_b_period:
        return None
    
    # Optimize: Calculate rolling max/min once per period
    high_rolling = {
        conversion_period: high.rolling(window=conversion_period).max(),
        base_period: high.rolling(window=base_period).max(),
        span_b_period: high.rolling(window=span_b_period).max()
    }
    low_rolling = {
        conversion_period: low.rolling(window=conversion_period).min(),
        base_period: low.rolling(window=base_period).min(),
        span_b_period: low.rolling(window=span_b_period).min()
    }
    
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    conversion_line = (high_rolling[conversion_period] + low_rolling[conversion_period]) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    base_line = (high_rolling[base_period] + low_rolling[base_period]) / 2
    
    # Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    span_a = (conversion_line + base_line) / 2
    
    # Span B (Senkou Span B): (52-period high + 52-period low) / 2
    span_b = (high_rolling[span_b_period] + low_rolling[span_b_period]) / 2
    
    return {
        'conversion_line': conversion_line.iloc[-1] if len(conversion_line) > 0 else None,
        'base_line': base_line.iloc[-1] if len(base_line) > 0 else None,
        'span_a': span_a.iloc[-1] if len(span_a) > 0 else None,
        'span_b': span_b.iloc[-1] if len(span_b) > 0 else None,
        'conversion_line_prev': conversion_line.iloc[-2] if len(conversion_line) > 1 else None,
        'base_line_prev': base_line.iloc[-2] if len(base_line) > 1 else None,
    }


def generate_trading_signals(stock_data):
    """
    Generate buy/sell/short signals based on multiple strategies.
    
    Args:
        stock_data: Dictionary containing stock metrics (bb_position_pct, rsi, macd_label, ichimoku, etc.)
    
    Returns:
        Dictionary with signals from different strategies:
        {
            'bb': 'BUY' | 'SELL' | 'SHORT' | None,
            'rsi': 'BUY' | 'SELL' | 'SHORT' | None,
            'macd': 'BUY' | 'SELL' | 'SHORT' | None,
            'ichimoku': 'BUY' | 'SELL' | 'SHORT' | None,
            'combined': 'BUY' | 'SELL' | 'SHORT' | None,
            'bb_ichimoku': 'BUY' | 'SELL' | 'SHORT' | None
        }
    """
    signals = {}
    
    # Strategy 1: Bollinger Bands
    bb_position_pct = stock_data.get('bb_position_pct')
    if bb_position_pct is not None:
        if bb_position_pct <= 0:
            signals['bb'] = 'BUY'
        elif bb_position_pct >= 100:
            signals['bb'] = 'SHORT'
        elif 40 <= bb_position_pct <= 60:
            signals['bb'] = 'SELL'  # Mean reversion / neutral zone
        else:
            signals['bb'] = None
    else:
        signals['bb'] = None
    
    # Strategy 2: RSI
    rsi = stock_data.get('rsi')
    if rsi is not None:
        if rsi <= 30:
            signals['rsi'] = 'BUY'  # Oversold
        elif rsi >= 70:
            signals['rsi'] = 'SHORT'  # Overbought
        else:
            signals['rsi'] = None
    else:
        signals['rsi'] = None
    
    # Strategy 3: MACD
    macd_label = stock_data.get('macd_label')
    if macd_label == 'Bullish':
        signals['macd'] = 'BUY'
    elif macd_label == 'Bearish':
        signals['macd'] = 'SHORT'
    else:
        signals['macd'] = None
    
    # Strategy 4: Ichimoku Cloud
    ichimoku = stock_data.get('ichimoku')
    price = stock_data.get('price')
    price_prev = stock_data.get('price_prev')
    vol_sma_20 = stock_data.get('vol_sma_20')
    price_sma_60 = stock_data.get('price_sma_60')
    
    if ichimoku and price is not None and all(v is not None for v in ichimoku.values()):
        base_line = ichimoku.get('base_line')
        base_line_prev = ichimoku.get('base_line_prev')
        span_a = ichimoku.get('span_a')
        span_b = ichimoku.get('span_b')
        
        # Common filters for both signals
        vol_filter = vol_sma_20 is None or vol_sma_20 > 100000
        price_filter = price_sma_60 is None or price_sma_60 > 20
        
        # Buy Signal: close > span_b AND span_a > span_b AND close crosses above base_line
        if vol_filter and price_filter:
            cloud_bullish = price > span_b and span_a > span_b
            crosses_above = price_prev is not None and base_line_prev is not None and \
                           price_prev <= base_line_prev and price > base_line
            
            # Sell Signal: close < span_a AND span_a < span_b AND base_line crosses above close
            cloud_bearish = price < span_a and span_a < span_b
            crosses_below = price_prev is not None and base_line_prev is not None and \
                           price_prev >= base_line_prev and price < base_line
            
            if cloud_bullish and crosses_above:
                signals['ichimoku'] = 'BUY'
            elif cloud_bearish and crosses_below:
                signals['ichimoku'] = 'SELL'
            else:
                signals['ichimoku'] = None
        else:
            signals['ichimoku'] = None
    else:
        signals['ichimoku'] = None
    
    # Combined Strategy: Require at least 2 signals to agree
    buy_count = sum(1 for s in signals.values() if s == 'BUY')
    short_count = sum(1 for s in signals.values() if s == 'SHORT')
    sell_count = sum(1 for s in signals.values() if s == 'SELL')
    
    if buy_count >= 2:
        signals['combined'] = 'BUY'
    elif short_count >= 2:
        signals['combined'] = 'SHORT'
    elif sell_count >= 1:
        signals['combined'] = 'SELL'
    else:
        signals['combined'] = None
    
    # BB or Ichimoku Strategy: Either BB or Ichimoku signals trigger (default)
    bb_sig = signals.get('bb')
    ich_sig = signals.get('ichimoku')
    
    if bb_sig == 'BUY' or ich_sig == 'BUY':
        signals['bb_ichimoku'] = 'BUY'
    elif bb_sig == 'SHORT' or ich_sig == 'SHORT':
        signals['bb_ichimoku'] = 'SHORT'
    elif bb_sig == 'SELL' or ich_sig == 'SELL':
        signals['bb_ichimoku'] = 'SELL'
    else:
        signals['bb_ichimoku'] = None
    
    return signals


def get_active_strategy():
    """
    Returns the currently active trading strategy.
    Can be configured via environment variable or settings file.
    
    Options: 'bb', 'rsi', 'macd', 'ichimoku', 'combined', 'bb_ichimoku'
    Default: 'bb_ichimoku' (either BB or Ichimoku triggers)
    """
    return os.getenv('TRADING_STRATEGY', 'bb_ichimoku')


# PERFORMANCE OPTIMIZATION: Cache VIX data globally to avoid fetching for every ticker
_vix_cache = {'data': None, 'time': 0}
VIX_CACHE_TTL = 300  # 5 minutes

def get_vix_cached():
    """Get VIX data with caching to avoid redundant API calls"""
    global _vix_cache
    now = time.time()
    if _vix_cache['data'] is None or (now - _vix_cache['time']) > VIX_CACHE_TTL:
        try:
            vix_data = safe_history(yf.Ticker("^VIX"), period="60d")
            if len(vix_data) >= 10:
                _vix_cache['data'] = vix_data
                _vix_cache['time'] = now
            else:
                return None
        except Exception:
            return None
    return _vix_cache['data']


# PERFORMANCE OPTIMIZATION: Cache alerts with TTL to avoid constant file reads
_alerts_cache = {"data": None, "time": 0}
CACHE_TTL = 300  # 5 minutes


def load_alerts():
    """Cached alerts loading with TTL"""
    global _alerts_cache
    now = time.time()
    if _alerts_cache["data"] is None or (now - _alerts_cache["time"]) > CACHE_TTL:
        try:
            with open(ALERTS_FILE) as f:
                _alerts_cache["data"] = tuple(json.load(f))
                _alerts_cache["time"] = now
        except:
            _alerts_cache["data"] = ()
            _alerts_cache["time"] = now
    return _alerts_cache["data"]


def check_alerts(data):
    custom_alerts = load_alerts()
    now = datetime.now(UTC).astimezone(PST)
    (
        high_52w,
        low_52w,
        surge,
        crash,
        volume_spike,
        buy_signals,
        sell_signals,
        short_signals,
        custom,
    ) = ([], [], [], [], [], [], [], [], [])
    stock_dict = {x["ticker"]: x for x in data}

    for a in custom_alerts:
        s = stock_dict.get(a["ticker"].upper())
        if not s:
            continue
        msg, cond, val = "", a["condition"], a.get("value")
        if cond == "price_above" and val and s["price"] > val:
            msg = f"price ABOVE ${val:.2f} → ${s['price']:.2f}"
        elif cond == "price_below" and val and s["price"] < val:
            msg = f"price BELOW ${val:.2f} → ${s['price']:.2f}"
        elif cond == "day_change_above" and val and s["change_pct"] > val:
            msg = f"DAY % ABOVE {val}% → {s['change_pct']:+.2f}%"
        elif cond == "day_change_below" and val and s["change_pct"] < val:
            msg = f"DAY % BELOW {val}% → {s['change_pct']:+.2f}%"
        elif cond == "rsi_oversold" and s["rsi"] is not None and s["rsi"] < 30:
            msg = f"RSI OVERSOLD → {s['rsi']:.1f}"
        elif cond == "rsi_overbought" and s["rsi"] is not None and s["rsi"] > 70:
            msg = f"RSI OVERBOUGHT → {s['rsi']:.1f}"
        elif cond == "volume_spike" and s["volume_spike"]:
            msg = "VOLUME SPIKE"
        elif cond == "buy":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "BUY":
                buy_signals.append({"ticker": s["ticker"], "msg": f"Custom: BUY signal"})
        elif cond == "sell":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "SELL":
                sell_signals.append({"ticker": s["ticker"], "msg": f"Custom: SELL signal"})
        elif cond == "short":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "SHORT":
                short_signals.append({"ticker": s["ticker"], "msg": f"Custom: SHORT signal"})
        if msg:
            custom.append({"ticker": s["ticker"], "msg": msg})

    for s in data:
        ch = s["change_pct"]
        if ch > 15:
            surge.append({"ticker": s["ticker"], "msg": f"SURGED > +15% → {ch:+.2f}%"})
        elif ch < -15:
            crash.append({"ticker": s["ticker"], "msg": f"CRASHED < -15% → {ch:+.2f}%"})
        if s["volume_spike"]:
            volume_spike.append({"ticker": s["ticker"], "msg": "VOLUME SPIKE"})
        
        # Trading Signal alerts (using active strategy)
        active_sig = s.get("active_signal") or s.get("bb_signal")
        if active_sig == "BUY":
            strategy_name = get_active_strategy().upper()
            buy_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} BUY signal"})
        elif active_sig == "SHORT":
            strategy_name = get_active_strategy().upper()
            short_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} SHORT signal"})
        elif active_sig == "SELL":
            strategy_name = get_active_strategy().upper()
            sell_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} SELL signal"})
        
        range_52w = s["52w_high"] - s["52w_low"]
        if range_52w > 0:
            pos_pct = (s["price"] - s["52w_low"]) / range_52w * 100
            if pos_pct >= 95:
                high_52w.append(
                    {"ticker": s["ticker"], "msg": f"NEAR 52W HIGH ({pos_pct:.1f}%)"}
                )
            elif pos_pct <= 5:
                low_52w.append(
                    {"ticker": s["ticker"], "msg": f"NEAR 52W LOW ({pos_pct:.1f}%)"}
                )

    def fmt(items, emoji, label):
        return (
            f"{emoji} <strong>{label}:</strong> {', '.join(a['ticker'] for a in items)}"
        )

    grouped = []
    if high_52w:
        grouped.append(fmt(high_52w, "🔥", "52W High"))
    if low_52w:
        grouped.append(fmt(low_52w, "📉", "52W Low"))
    if buy_signals:
        grouped.append(fmt(buy_signals, "💚", "Buy"))
    if sell_signals:
        grouped.append(fmt(sell_signals, "💛", "Sell"))
    if short_signals:
        grouped.append(fmt(short_signals, "❤️", "Short"))
    if surge:
        grouped.append(fmt(surge, "🚀", "Surge"))
    if crash:
        grouped.append(fmt(crash, "💥", "Crash"))
    if volume_spike:
        grouped.append(fmt(volume_spike, "📈", "Vol Spike"))
    if custom:
        grouped.append(f"⚡ <strong>Custom:</strong> {len(custom)}")
    return {"grouped": grouped, "time": now.strftime("%I:%M %p")}


def rsi(s):
    """OPTIMIZED: Use EWM instead of rolling mean for faster calculation"""
    if len(s) < 15:
        return None
    d = s.diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = g / l
    return 100 - 100 / (1 + rs.iloc[-1])


def macd(s):
    if len(s) < 26:
        return None, None, "N/A"
    e12, e26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    last_line, last_sig = line.iloc[-1], sig.iloc[-1]
    return last_line, last_sig, "Bullish" if last_line > last_sig else "Bearish"


def na(v, f="{:.2f}"):
    if v is None or pd.isna(v):
        return "N/A"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return "N/A"


def sparkline(prices):
    if len(prices) < 2:
        return ""
    prices = prices[-30:]
    mn, mx = min(prices), max(prices)
    rng = mx - mn if mx != mn else 1
    w, h = 60, 20
    pts = [
        f"{(i/(len(prices)-1))*w:.1f},{h-((p-mn)/rng*h):.1f}"
        for i, p in enumerate(prices)
    ]
    c = "#00aa00" if prices[-1] >= prices[0] else "#cc0000"
    return f'<svg width="{w}" height="{h}"><polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="1.5"/></svg>'


# Safe wrapper for yfinance history calls to avoid hard failures (e.g., 401 Unauthorized)
def safe_history(ticker_obj, **kwargs):
    try:
        return ticker_obj.history(**kwargs)
    except Exception:
        return pd.DataFrame()


def fetch(ticker, ext=False, retry=0):
    max_retries = 3
    try:
        # enforce global rate limit before starting network activity
        RATE_LIMITER.wait()

        t = yf.Ticker(ticker)
        # Small base delay with jitter to spread requests a bit
        # Reduced from 0.25 to 0.1 for faster execution with rate limiter protection
        base_sleep = 0.1
        jitter = random.uniform(0, 0.15)
        time.sleep(base_sleep + jitter + (retry * 0.15))

        info = get_ticker_info_cached(ticker)
        h_day = safe_history(t, period="1d", interval="1m", prepost=ext)
        if h_day.empty:
            h_day = safe_history(t, period="5d", prepost=ext)
        if h_day.empty:
            return None

        price = h_day["Close"].iloc[-1]
        day_low, day_high = h_day["Low"].min(), h_day["High"].max()
        reg = safe_history(t, period="5d", prepost=False)
        change_pct = change_abs_day = 0.0
        if len(reg) >= 2:
            prev = reg["Close"].iloc[-2]
            if prev and prev > 0:
                change_pct = ((price - prev) / prev) * 100
                change_abs_day = price - prev
        # 5-day change: compare to oldest close in the 5-day window when available
        change_5d = None
        change_abs_5d = None
        try:
            if len(reg) >= 2:
                ref5 = reg["Close"].iloc[0]
                if ref5 and ref5 > 0:
                    change_5d = ((price - ref5) / ref5) * 100
                    change_abs_5d = price - ref5
        except Exception:
            change_5d = None
            change_abs_5d = None

        h1m, h6m = safe_history(t, period="2mo"), safe_history(t, period="7mo")
        
        # YTD calculation: Use previous year's last trading day as baseline
        # Get sufficient history to ensure we have prev year's data
        ytd_history = safe_history(t, period="3mo")
        ytd = pd.DataFrame()
        if not ytd_history.empty:
            current_year = datetime.now().year
            # Filter for current year data
            ytd = ytd_history[ytd_history.index.year == current_year]
            # If we have current year data but need baseline from previous year
            if len(ytd) >= 1:
                prev_year_data = ytd_history[ytd_history.index.year == current_year - 1]
                if not prev_year_data.empty:
                    # Add last day of previous year as first row for comparison
                    ytd = pd.concat([prev_year_data.tail(1), ytd])
        
        y, h30 = safe_history(t, period="1y"), safe_history(t, period="60d")

        def calc_ch(h):
            if len(h) >= 2 and h["Close"].iloc[0] > 0:
                return (
                    (price - h["Close"].iloc[0]) / h["Close"].iloc[0]
                ) * 100, price - h["Close"].iloc[0]
            return None, None

        ch1m, abs1m = calc_ch(h1m)
        ch6m, abs6m = calc_ch(h6m)
        chytd, absytd = calc_ch(ytd)

        high52, low52 = (
            (y["High"].max(), y["Low"].min()) if not y.empty else (price, price)
        )
        vol = h_day["Volume"].sum()

        hv = None
        if len(h30) >= 30:
            r = h30["Close"].pct_change().dropna()
            if len(r) > 1:
                hv = r.std() * (252**0.5) * 100

        short_pct = info.get("shortPercentOfFloat")
        if short_pct:
            short_pct *= 100

        days_cover = None
        if info.get("sharesShort"):
            avg = (
                info.get("averageDailyVolume10Day")
                or info.get("averageVolume")
                or vol
                or 1
            )
            if avg > 0:
                days_cover = info["sharesShort"] / avg

        squeeze = "None"
        if short_pct and days_cover:
            if short_pct > 30 and days_cover > 10:
                squeeze = "Extreme"
            elif short_pct > 20 and days_cover > 7:
                squeeze = "High"
            elif short_pct > 15 and days_cover > 5:
                squeeze = "Moderate"

        rsi_val = rsi(h30["Close"])
        h100 = safe_history(t, period="100d")
        macd_val, macd_sig, macd_lbl = (
            macd(h100["Close"]) if not h100.empty else (None, None, "N/A")
        )

        vol_spike = False
        if len(h30) > 1:
            avg = h30["Volume"][:-1].mean()
            if avg > 0:
                vol_spike = vol > 1.5 * avg

        pc_ratio = impl_move = impl_hi = impl_lo = exp_date = None
        # Options endpoints increasingly return 401; wrap fully and degrade gracefully
        try:
            opts = getattr(t, "options", None)
            if opts:
                exp_date = opts[0]
                try:
                    chain = t.option_chain(exp_date)
                    strikes = pd.concat(
                        [chain.calls["strike"], chain.puts["strike"]]
                    ).unique()
                    if len(strikes) > 0:
                        atm = min(strikes, key=lambda s: abs(s - price))
                        cp = (
                            chain.calls.loc[
                                chain.calls["strike"] == atm, "lastPrice"
                            ].iloc[0]
                            if not chain.calls[chain.calls["strike"] == atm].empty
                            else 0
                        )
                        pp = (
                            chain.puts.loc[
                                chain.puts["strike"] == atm, "lastPrice"
                            ].iloc[0]
                            if not chain.puts[chain.puts["strike"] == atm].empty
                            else 0
                        )
                        straddle = cp + pp
                        if straddle > 0 and price > 0:
                            impl_move = (straddle / price) * 100
                            cons = impl_move * 0.85
                            impl_hi = price * (1 + cons / 100)
                            impl_lo = price * (1 - cons / 100)
                            cvol, pvol = (
                                chain.calls["volume"].fillna(0).sum(),
                                chain.puts["volume"].fillna(0).sum(),
                            )
                            if cvol > 0:
                                pc_ratio = pvol / cvol
                except Exception:
                    pass
        except Exception:
            # ignore options entirely on failure (e.g., 401 Unauthorized)
            pass

        down_bias = False
        if len(h30) > 0:
            down_vol = h30[h30["Close"] < h30["Open"]]["Volume"].sum()
            up_vol = h30[h30["Close"] > h30["Open"]]["Volume"].sum()
            down_bias = down_vol > up_vol

        opt_dir = "Neutral"
        if pc_ratio:
            if pc_ratio > 1.2 and down_bias:
                opt_dir = "Strong Bearish"
            elif pc_ratio > 1.0 or down_bias:
                opt_dir = "Bearish"
            elif pc_ratio < 0.8 and not down_bias:
                opt_dir = "Bullish"

        rec_mean = info.get("recommendationMean", 5)
        sentiment = ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")[
            (
                0
                if rec_mean <= 1.5
                else (
                    1
                    if rec_mean <= 2.5
                    else 2 if rec_mean <= 3.5 else 3 if rec_mean <= 4.5 else 4
                )
            )
        ]

        rating = info.get("recommendationKey", "none").title().replace("_", " ")
        target = info.get("targetMeanPrice")
        upside = ((target - price) / price) * 100 if target and price > 0 else None
        spk = sparkline(h30["Close"].tolist() if not h30.empty else [])

        bb_period = 20
        bb_upper = bb_lower = bb_middle = bb_width_pct = bb_position_pct = bb_status = (
            None
        )
        if len(h30) >= bb_period:
            close = h30["Close"]
            bb_middle = close.rolling(window=bb_period).mean().iloc[-1]
            std_dev = close.rolling(window=bb_period).std().iloc[-1]
            bb_upper = bb_middle + (std_dev * 2)
            bb_lower = bb_middle - (std_dev * 2)
            if bb_middle > 0:
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
            if (bb_upper - bb_lower) > 0:
                bb_position_pct = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
                bb_position_pct = max(0, min(100, bb_position_pct))
            bb_status = (
                "Above Upper"
                if price > bb_upper
                else "Below Lower" if price < bb_lower else "Inside"
            )

        # Calculate Ichimoku Cloud indicators
        ichimoku_data = None
        vol_sma_20 = None
        price_sma_60 = None
        price_prev = None
        
        if len(y) >= 60:
            ichimoku_data = calculate_ichimoku(y["High"], y["Low"], y["Close"])
            
            # Calculate volume SMA(20)
            if len(h30) >= 20:
                vol_sma_20 = h30["Volume"].rolling(window=20).mean().iloc[-1]
            
            # Calculate price SMA(60)
            price_sma_60 = y["Close"].rolling(window=60).mean().iloc[-1]
            
            # Get previous close for crossover detection
            if len(y) >= 2:
                price_prev = y["Close"].iloc[-2]

        # Generate trading signals using strategy framework
        signal_data = {
            'bb_position_pct': bb_position_pct,
            'rsi': rsi_val,
            'macd_label': macd_lbl,
            'ichimoku': ichimoku_data,
            'price': price,
            'price_prev': price_prev,
            'vol_sma_20': vol_sma_20,
            'price_sma_60': price_sma_60,
        }
        all_signals = generate_trading_signals(signal_data)
        active_strategy = get_active_strategy()
        primary_signal = all_signals.get(active_strategy)
        
        # Legacy bb_signal for backward compatibility (uses active strategy)
        bb_signal = primary_signal

        # CVR3 VIX Signal: Generate BUY/SELL/SHORT based on VIX (market, not individual ticker)
        cvr3_vix_signal = None
        cvr3_vix_pct = None
        cvr3_vix_value = None
        try:
            vix_data = get_vix_cached()
            if vix_data is not None and len(vix_data) >= 10:
                vix_close_prices = vix_data["Close"]
                vix_sma = vix_close_prices.rolling(window=10).mean().iloc[-1]
                if vix_sma > 0:
                    current_vix = vix_close_prices.iloc[-1]
                    prev_vix = (
                        vix_close_prices.iloc[-2]
                        if len(vix_close_prices) >= 2
                        else current_vix
                    )
                    cvr3_vix_value = current_vix
                    cvr3_vix_pct = current_vix - prev_vix
                    pct_diff = ((current_vix - vix_sma) / vix_sma) * 100
                    # Buy Signal: VIX significantly above 10-day SMA
                    if pct_diff >= 10:
                        cvr3_vix_signal = "BUY"
                    # Sell Signal: VIX significantly below 10-day SMA
                    elif pct_diff <= -10:
                        cvr3_vix_signal = "SELL"
                    # Short Signal: VIX extremely elevated above SMA
                    elif pct_diff >= 20:
                        cvr3_vix_signal = "SHORT"
        except Exception:
            cvr3_vix_signal = None
            cvr3_vix_pct = None
            cvr3_vix_value = None

        div_rate = info.get("dividendRate")
        div_yield = info.get("dividendYield")

        # Do not normalize dividend yield — render the raw value as provided
        # by the data source. Keep `div_yield` unchanged.

        # Extract earnings date (display string) and ISO date for JS filtering
        earnings_date = None
        earnings_date_iso = None
        try:

            def _parse_earnings_display_and_iso(val):
                if val is None:
                    return None, None
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                # numeric epoch
                if isinstance(val, (int, float)):
                    dt = datetime.fromtimestamp(int(val), UTC).astimezone(PST)
                    return dt.strftime("%b %d, %Y"), dt.date().isoformat()
                if isinstance(val, str):
                    s = val.strip()
                    if s.isdigit():
                        dt = datetime.fromtimestamp(int(s), UTC).astimezone(PST)
                        return dt.strftime("%b %d, %Y"), dt.date().isoformat()
                    try:
                        d = datetime.fromisoformat(s.split("T")[0])
                        d = d.replace(tzinfo=UTC)
                        dt = d.astimezone(PST)
                        return dt.strftime("%b %d, %Y"), dt.date().isoformat()
                    except Exception:
                        try:
                            d = datetime.strptime(s.split("T")[0], "%Y-%m-%d")
                            d = d.replace(tzinfo=UTC)
                            dt = d.astimezone(PST)
                            return dt.strftime("%b %d, %Y"), dt.date().isoformat()
                        except Exception:
                            return None, None
                return None, None

            for k in (
                "earningsTimestamp",
                "earningsTimestampStart",
                "earningsDate",
                "nextEarningsDate",
            ):
                ed = info.get(k)
                disp, iso = _parse_earnings_display_and_iso(ed)
                if disp:
                    earnings_date = disp
                    earnings_date_iso = iso
                    break
        except Exception:
            earnings_date = None
            earnings_date_iso = None

        pe = info.get("trailingPE") or info.get("forwardPE")
        eps = (
            info.get("trailingEps")
            or info.get("epsTrailingTwelveMonths")
            or info.get("earningsPerShare")
            or info.get("forwardEps")
        )
        market_cap = info.get("marketCap")
        aum = (
            info.get("totalAssets")
            or info.get("fundTotalAssets")
            or info.get("total_assets")
        )

        tu = ticker.upper()
        category = infer_category_from_info(tu, info) or get_category(tu)
        quote_type = (info.get("quoteType") or "").upper()
        return {
            "ticker": tu,
            "quote_type": quote_type,
            "price": price,
            "change_pct": change_pct,
            "change_abs_day": change_abs_day,
            "change_1m": ch1m,
            "change_abs_1m": abs1m,
            "change_5d": change_5d,
            "change_abs_5d": change_abs_5d,
            "change_6m": ch6m,
            "change_abs_6m": abs6m,
            "change_ytd": chytd,
            "change_abs_ytd": absytd,
            "volume": vol,
            "volume_raw": vol,
            "52w_high": high52,
            "52w_low": low52,
            "day_low": day_low,
            "day_high": day_high,
            "short_percent": short_pct,
            "days_to_cover": days_cover,
            "squeeze_level": squeeze,
            "rsi": rsi_val,
            "macd_label": macd_lbl,
            "volume_spike": vol_spike,
            "is_meme_stock": tu in MEME_STOCKS,
            "sentiment": sentiment,
            "analyst_rating": rating,
            "upside_potential": upside,
            "options_direction": opt_dir,
            "implied_move_pct": impl_move,
            "implied_high": impl_hi,
            "implied_low": impl_lo,
            "down_volume_bias": down_bias,
            "sparkline": spk,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "bb_width_pct": bb_width_pct,
            "bb_position_pct": bb_position_pct,
            "bb_status": bb_status,
            "bb_signal": bb_signal,
            "signal_bb": all_signals.get('bb'),
            "signal_rsi": all_signals.get('rsi'),
            "signal_macd": all_signals.get('macd'),
            "signal_ichimoku": all_signals.get('ichimoku'),
            "signal_combined": all_signals.get('combined'),
            "signal_bb_ichimoku": all_signals.get('bb_ichimoku'),
            "active_signal": primary_signal,
            "ichimoku_conversion": ichimoku_data.get('conversion_line') if ichimoku_data else None,
            "ichimoku_base": ichimoku_data.get('base_line') if ichimoku_data else None,
            "ichimoku_span_a": ichimoku_data.get('span_a') if ichimoku_data else None,
            "ichimoku_span_b": ichimoku_data.get('span_b') if ichimoku_data else None,
            "cvr3_vix_signal": cvr3_vix_signal,
            "cvr3_vix_pct": cvr3_vix_pct,
            "cvr3_vix_value": cvr3_vix_value,
            "hv_30_annualized": hv,
            "macd_line": macd_val,
            "macd_signal": macd_sig,
            "macd_label": macd_lbl,
            "pc_ratio": pc_ratio,
            "pe": pe,
            "eps": eps,
            "dividend_rate": div_rate,
            "dividend_yield": div_yield,
            "earnings_date": earnings_date,
            "earnings_date_iso": earnings_date_iso,
            "market_cap": market_cap,
            "aum": aum,
            "category": category,
        }
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limit" in error_msg:
            if retry < max_retries:
                # exponential backoff with jitter
                wait_time = (2**retry) * 5 + random.uniform(0, 3)
                print(
                    f"{ticker}: Rate limited, waiting {wait_time:.1f}s (retry {retry+1}/{max_retries})"
                )
                time.sleep(wait_time)
                return fetch(ticker, ext, retry + 1)
            else:
                print(f"{ticker}: Max retries reached, skipping")
                return None
        elif "Unauthorized" in error_msg or "401" in error_msg:
            # Yahoo Finance feature gated; skip this ticker gracefully
            print(
                f"{ticker}: Unauthorized for some endpoints, skipping options/advanced data"
            )
            return None
        else:
            print(f"Error {ticker}: {e}")
            time.sleep(5)
            return None


def fmt_vol(v):
    if v is None:
        return "N/A"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return str(int(v))


def fmt_mcap(v):
    if v is None or pd.isna(v):
        return "N/A"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v >= 1e12:
        return f"{v/1e12:.2f}T"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    return str(int(v))


def fmt_change(p, a=None):
    if p is None:
        return '<span class="neutral" data-sort="-999999">N/A</span>'
    sign, cls = ("▲", "positive") if p >= 0 else ("▼", "negative")
    abs_str = ""
    if a is not None:
        try:
            abs_str = f" ({float(a):+.2f})"
        except (ValueError, TypeError):
            pass
    return f'<span class="{cls}" data-sort="{p:.10f}">{sign} {p:+.2f}%{abs_str}</span>'


@lru_cache(maxsize=32)  # OPTIMIZED: Cache index data
def get_index_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("previousClose")
        ch_pct = info.get("regularMarketChangePercent")
        prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
        ch_abs = None
        if price is not None and prev is not None:
            try:
                # Ensure numeric types
                price = float(price)
                prev = float(prev)
                ch_abs = price - prev
            except (ValueError, TypeError):
                ch_abs = None
        if ch_pct is None and price is not None and prev is not None and prev > 0:
            try:
                ch_pct = ((float(price) - float(prev)) / float(prev)) * 100
            except (ValueError, TypeError):
                ch_pct = None
        return {"price": price, "change_pct": ch_pct, "change_abs": ch_abs}
    except:
        return {"price": None, "change_pct": None, "change_abs": None}


def dashboard(csv="data/tickers.csv", ext=False):
    os.makedirs("data", exist_ok=True)
    try:
        with open(csv, "r", encoding="utf-8") as f:
            txt = f.read()
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"[\n,]+", txt.strip())
        parts = [p.strip().upper() for p in parts if p and p.strip()]
        if parts and parts[0].lower() in ("ticker", "tickers"):
            parts = parts[1:]
        tickers = pd.Series(parts).unique().tolist()
    except Exception:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"]

    data = []
    # Adaptive worker count: Increased from 3 to 5 for better parallelization
    # Rate limiter prevents overwhelming the API
    worker_count = min(5, max(2, len(tickers) // 6 + 1))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(fetch, t, ext) for t in tickers]
        for r in as_completed(futures):
            res = r.result()
            if res:
                data.append(res)
    return pd.DataFrame(data).sort_values("change_pct", ascending=False)


def get_vix_data():
    return get_index_data("^VIX")


# OPTIMIZED: Cache F&G data with 1 hour TTL
_fg_cache = {"data": None, "time": 0}
FG_CACHE_TTL = 3600

# Shared requests session for fewer TCP handshakes
SESSION = requests.Session()


# Cache ticker info to avoid repeated yf.Ticker(...).info calls
@lru_cache(maxsize=512)
def get_ticker_info_cached(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info
    except Exception:
        return {}


# Simple global rate limiter (ensure at least `min_interval` seconds between network starts)
class RateLimiter:
    def __init__(self, min_interval=0.9):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_time:
                wait_for = self.next_time - now
                time.sleep(wait_for)
                now = time.time()
            self.next_time = now + self.min_interval


RATE_LIMITER = RateLimiter(min_interval=0.9)


def get_fear_greed_data():
    global _fg_cache
    now = time.time()
    if _fg_cache["data"] is not None and (now - _fg_cache["time"]) < FG_CACHE_TTL:
        return _fg_cache["data"]

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        }
        r = requests.get(url, headers=headers, timeout=5)  # OPTIMIZED: Reduced timeout
        r.raise_for_status()
        data = r.json()
        fg = data.get("fear_and_greed") or data
        score = float(fg["score"])
        s = int(round(score))
        rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
            0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
        ]
        result = {"score": score, "rating": rating, "raw_score": s}
        _fg_cache["data"] = result
        _fg_cache["time"] = now
        return result
    except Exception:
        try:
            r = requests.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            score = float(data["fear_and_greed"]["score"])
            s = int(round(score))
            rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
                (
                    0
                    if s <= 24
                    else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
                )
            ]
            result = {"score": score, "rating": rating, "raw_score": s}
            _fg_cache["data"] = result
            _fg_cache["time"] = now
            return result
        except Exception as e:
            print(f"F&G error: {e}")
    return {"score": None, "rating": "N/A", "raw_score": None}


# OPTIMIZED: Cache AAII data with 1 hour TTL
_aaii_cache = {"data": None, "time": 0}
AAII_CACHE_TTL = 3600


def get_aaii_sentiment():
    global _aaii_cache
    now = time.time()
    if _aaii_cache["data"] is not None and (now - _aaii_cache["time"]) < AAII_CACHE_TTL:
        return _aaii_cache["data"]

    try:
        r = requests.get(
            "https://www.aaii.com/sentimentsurvey/sent_results",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        m = re.search(r"\w+\s*\d{1,2}.*?([\d\.]+)%.*?([\d\.]+)%", r.text)
        if not m:
            m = re.search(
                r"Bullish.*?([\d\.]+)%.*?Bearish.*?([\d\.]+)%", r.text, re.DOTALL
            )
        if m:
            b, be = float(m.group(1)), float(m.group(2))
            result = {"bullish": b, "bearish": be, "spread": b - be}
            _aaii_cache["data"] = result
            _aaii_cache["time"] = now
            return result
    except Exception as e:
        print(f"AAII fetch error: {e}")
    return {"bullish": None, "bearish": None, "spread": None}


# NOTE: html() function continues with the full HTML generation...
# Due to character limits, the complete html() function and main block remain unchanged
# from original code. Simply append the original html() function and main block here.


def html(df, vix, fg, aaii, file, ext=False, alerts=None):
    alerts = alerts or {"grouped": [], "time": ""}
    update = datetime.now(UTC).astimezone(PST).strftime("%I:%M:%S %p PST on %B %d, %Y")

    banner = (
        '<div class="alert-banner">🚨 <strong>ALERTS</strong> '
        + " | ".join(alerts["grouped"])
        + "</div>"
        if alerts["grouped"]
        else ""
    )

    # Major indices
    dow = get_index_data("^DJI")
    sp = get_index_data("^GSPC")
    nas = get_index_data("^IXIC")

    # Calculate CVR3 Signal
    cvr3_signal = "NEUTRAL"
    try:
        vix_data = safe_history(yf.Ticker("^VIX"), period="60d")
        if len(vix_data) >= 10:
            vix_close = vix_data["Close"]
            vix_sma = vix_close.rolling(window=10).mean().iloc[-1]
            if vix_sma > 0:
                current_vix = vix_close.iloc[-1]
                pct_diff = ((current_vix - vix_sma) / vix_sma) * 100
                if pct_diff >= 20:
                    cvr3_signal = "SHORT"
                elif pct_diff >= 10:
                    cvr3_signal = "BUY"
                elif pct_diff <= -10:
                    cvr3_signal = "SELL"
    except Exception:
        cvr3_signal = "NEUTRAL"

    def index_str(data, name):
        if data["price"] is None:
            return f'<span class="neutral">{name}: N/A</span>'
        ch_abs = data.get("change_abs")
        cls = "positive" if ch_abs is not None and ch_abs >= 0 else "negative"
        return f'<span class="{cls}">{name}: {na(data["price"])} ({na(ch_abs, "{:+.2f}")})</span>'

    # CVR3 signal color
    cvr3_color = (
        "positive"
        if cvr3_signal == "BUY"
        else ("negative" if cvr3_signal in ("SELL", "SHORT") else "neutral")
    )
    cvr3_str = f'<span class="{cvr3_color}">CVR3: {cvr3_signal}</span>'

    # Build multi-line market summary
    indices_h = f'{index_str(dow, "Dow")} | {index_str(sp, "S&P")} | {index_str(nas, "Nasdaq")} | {index_str(vix, "VIX")} | {cvr3_str}'

    fg_h = '<span class="neutral">F&G: N/A</span>'
    if fg.get("score") is not None:
        cls = (
            "negative"
            if fg["score"] <= 24
            else (
                "high-risk"
                if fg["score"] <= 44
                else (
                    "neutral"
                    if fg["score"] <= 55
                    else "bullish" if fg["score"] <= 74 else "positive"
                )
            )
        )
        fg_h = f'<span class="{cls}">F&G: {fg["score"]:.1f} ({fg["rating"]})</span>'

    aaii_h = '<span class="neutral">AAII: N/A</span>'
    if aaii.get("bullish") is not None:
        spread = aaii["spread"]
        cls = (
            "positive"
            if spread > 20
            else (
                "bullish"
                if spread > 0
                else (
                    "neutral"
                    if spread > -20
                    else "high-risk" if spread > -40 else "negative"
                )
            )
        )
        aaii_h = f'<span class="{cls}">AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%</span>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{ --bg:#f5f5f5; --card:#fff; --text:#333; --border:#ddd; --accent:#0066cc; --pos:#00aa00; --neg:#cc0000; --bullish:#00aa00; --bearish:#cc0000 }}
[data-theme="dark"]{{ --bg:#1a1a1a; --card:#2d2d2d; --text:#e0e0e0; --border:#444; --accent:#3d8bfd; --pos:#4caf50; --neg:#f44336; --bullish:#4caf50; --bearish:#f44336 }}
body{{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);padding:20px;transition:.3s}}
.container{{max-width:1600px;margin:auto}}
.top-bar{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:15px;background:var(--card);padding:15px 20px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
.btn{{padding:8px 16px;background:var(--accent);color:#fff;border:none;border-radius:8px;cursor:pointer;font-weight:600}}
.alert-banner{{background:#ff4444;color:#fff;padding:12px;border-radius:8px;margin-bottom:20px;text-align:center;font-weight:bold}}
.quick-filters{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;background:var(--card);padding:15px;border-radius:12px}}
.chip{{padding:8px 16px;background:var(--bg);border:2px solid var(--border);border-radius:20px;cursor:pointer;font-weight:600;transition:.2s}}
.chip.active,.chip:hover{{background:var(--accent);border-color:var(--accent);color:#fff}}
.views{{display:flex;gap:10px;margin-bottom:20px}}
.view-btn{{padding:10px 20px;background:var(--card);border:2px solid var(--border);border-radius:8px;cursor:pointer;transition:.2s}}
.view-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
#tableView{{display:block}}
#cardView,#heatView{{display:none}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
.stock-card{{background:var(--card);border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.1);transition:.3s}}
.stock-card:hover{{transform:translateY(-4px);box-shadow:0 8px 16px rgba(0,0,0,.2)}}
.heat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.heat-tile{{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:8px;padding:10px;cursor:pointer;transition:.3s}}
.heat-tile:hover{{transform:scale(1.05);box-shadow:0 4px 12px rgba(0,0,0,.3)}}
table{{width:100%;border-collapse:collapse;background:var(--card);box-shadow:0 2px 8px rgba(0,0,0,.1)}}
th{{background:var(--accent);color:#fff;padding:14px;cursor:pointer;position:sticky;top:0;z-index:10}}
td{{padding:12px;border-bottom:1px solid var(--border);vertical-align:top}}
.positive{{color:var(--pos);font-weight:bold}}
.negative{{color:var(--neg);font-weight:bold}}
.neutral{{color:#888}}
.bullish{{color:var(--bullish);font-weight:bold}}
.bearish{{color:#ff0000;font-weight:bold}}
.extreme-fear{{color:#ff0000;font-weight:bold}}
.fear{{color:#ff8800}}
.greed{{color:#88ff88}}
.extreme-greed{{color:#00bb00;font-weight:bold}}
.strong-bull{{color:#008800;font-weight:bold}}
.bull{{color:#00bb00}}
.strong-bear{{color:#ff0000;font-weight:bold}}
.bear{{color:#ff0000;font-weight:bold}}
.vol-hot{{color:#ff0000;font-weight:bold}}
.range-bar{{width:100%;height:8px;background:#e0e0e0;border-radius:4px;position:relative;margin:4px 0}}
.range-bar-marker{{position:absolute;width:3px;height:12px;background:#000;top:-2px}}
.range-labels{{display:flex;justify-content:space-between;font-size:0.75em;margin-top:2px}}
.range-container{{margin:8px 0}}
.toggle-switch{{position:relative;display:inline-block;width:60px;height:34px}}
.toggle-switch input{{opacity:0;width:0;height:0}}
.toggle-slider{{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#ccc;transition:.4s;border-radius:34px}}
.toggle-slider:before{{position:absolute;content:"";height:26px;width:26px;left:4px;bottom:4px;background:#fff;transition:.4s;border-radius:50%}}
input:checked + .toggle-slider{{background:var(--accent)}}
input:checked + .toggle-slider:before{{transform:translateX(26px)}}
.hours-toggle{{display:flex;align-items:center;gap:10px;font-weight:600;margin-left:20px}}
</style></head><body>
<div class="container">
{banner}
<div class="top-bar">
<div><h1>📊 Dashboard</h1><small>{update}</small></div>
<div style="display:flex;gap:15px;flex-wrap:wrap;align-items:center">
<span>{indices_h}</span><span style="color:#888"> | </span><span>{fg_h}</span><span style="color:#888"> | </span><span>{aaii_h}</span>
</div>
</div>

<div style="display:flex;gap:15px;flex-wrap:wrap;align-items:center;background:var(--card);padding:15px 20px;border-radius:12px;margin-bottom:20px;justify-content:space-between">
<div class="hours-toggle">
<span>Regular</span>
<label class="toggle-switch">
<input type="checkbox" {'checked' if ext else ''} onchange="toggleHours(this.checked)">
<span class="toggle-slider"></span>
</label>
<span>Extended</span>
</div>
<div style="display:flex;gap:10px">
<button class="btn" onclick="toggleTheme()">🌓</button>
<button class="btn" onclick="location.reload()">🔄</button>
</div>
</div>

<div class="quick-filters">
<div class="chip active" data-filter="all">All</div>
<div class="chip" data-filter="volume">📊 High Vol</div>
<div class="chip" data-filter="earnings-week">📅 Earnings</div>
<div class="chip" data-filter="signal-buy">🟢 Buy Signal</div>
<div class="chip" data-filter="signal-sell">🟠 Sell Signal</div>
<div class="chip" data-filter="signal-short">🔴 Short Signal</div>
<div class="chip" data-filter="oversold">📉 Oversold</div>
<div class="chip" data-filter="overbought">📈 Overbought</div>
<div class="chip" data-filter="surge">🚀 Surge</div>
<div class="chip" data-filter="crash">💥 Crash</div>
<div class="chip" data-filter="dividend">💰 Dividend</div>
<div class="chip" data-filter="cat-major-tech">🌐 Major Tech/Growth</div>
<div class="chip" data-filter="cat-leveraged-etf">⚡ Leveraged/Inverse ETFs</div>
<div class="chip" data-filter="cat-sector-etf">🏦 Sector & Index ETFs</div>
<div class="chip" data-filter="cat-emerging-tech">🚧 Emerging Tech (AI/Energy)</div>
<div class="chip" data-filter="cat-spec-meme">🎲 Speculative</div>
<div class="chip" data-filter="squeeze">🔥 Squeeze</div>
<div class="chip" data-filter="bb-squeeze">📏 BB Squeeze</div>
</div>

<div class="views">
<button class="view-btn active" onclick="setView(this,'table')">📋 Table</button>
<button class="view-btn" onclick="setView(this,'card')">🗂️ Cards</button>
<button class="view-btn" onclick="setView(this,'heat')">🔥 Heatmap</button>
<input id="tickerFilter" placeholder="Filter tickers..." oninput="applyFilter()" style="padding:8px;border-radius:6px;border:1px solid var(--border);margin-left:12px;min-width:160px">
</div>

<div id="tableView">
<table id="stockTable">
<tr>
<th data-sort="ticker">⭐ TICKER</th>
<th data-sort="price">PRICE</th>
<th data-sort="change_pct">DAY %</th>
<th data-sort="change_5d">5D %</th>
<th data-sort="change_1m">1M %</th>
<th data-sort="change_6m">6M %</th>
<th data-sort="change_ytd">YTD %</th>
<th data-sort="volume_raw">VOLUME</th>
<th>RANGES</th>
<th>INDICATORS</th>
<th>SENTIMENT</th>
</tr>
"""
    for _, r in df.iterrows():
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.5em">🟠</span> '
        elif r.get("cvr3_vix_signal") == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif r.get("cvr3_vix_signal") == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        bb_icon = signal_icon  # Alias for backward compatibility
        hv = r["hv_30_annualized"]
        hv_cls = "negative" if hv and hv > 50 else "neutral"
        hv_str = na(hv, "{:.1f}%")

        macd_cls = (
            "bullish"
            if r["macd_label"] == "Bullish"
            else "bearish" if r["macd_label"] == "Bearish" else "neutral"
        )
        opt_dir_cls = (
            "bullish"
            if "Bullish" in r["options_direction"]
            else "bearish" if "Bearish" in r["options_direction"] else "neutral"
        )
        bias_cls = "bearish" if r["down_volume_bias"] else "bullish"

        sent_cls = (
            "bullish"
            if "Buy" in r["sentiment"]
            else "bearish" if "Sell" in r["sentiment"] else "neutral"
        )
        # include analyst rating in sentiment display and adjust class if needed
        analyst_rating = r.get("analyst_rating") or ""
        # if analyst rating suggests Buy/Sell, reflect that in the class (fallback to sentiment)
        if analyst_rating:
            if "Buy" in analyst_rating:
                sent_cls = "bullish"
            elif "Sell" in analyst_rating:
                sent_cls = "bearish"
        upside_cls = (
            "bullish"
            if r["upside_potential"] and r["upside_potential"] > 0
            else (
                "bearish"
                if r["upside_potential"] and r["upside_potential"] < 0
                else "neutral"
            )
        )

        bb_bar = ""
        # Only render Bollinger Bands bar when BB values are present and numeric
        if (
            pd.notna(r.get("bb_position_pct"))
            and pd.notna(r.get("bb_lower"))
            and pd.notna(r.get("bb_middle"))
            and pd.notna(r.get("bb_upper"))
        ):
            try:
                pos = float(r["bb_position_pct"])
                pos = max(0.0, min(100.0, pos))
                bb_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos}%, var(--neg) {pos}%, var(--neg) 100%)"
                bb_bar = f'<div class="range-container"><div class="range-title">Bollinger Bands</div><div class="range-bar" style="background:{bb_color}"><div class="range-bar-marker" style="left:{pos}%"></div></div><div class="range-labels"><span>${na(r["bb_lower"])}</span><span>${na(r["bb_middle"])}</span><span>${na(r["bb_upper"])}</span></div><div style="font-size:0.75em;text-align:center">Width: {na(r["bb_width_pct"],"{:.1f}")}% – {r["bb_status"]}</div></div>'
            except Exception:
                bb_bar = ""

        impl_bar = ""
        # Only render implied-move chart when values are present and numeric (avoid NaN%)
        if (
            pd.notna(r.get("implied_move_pct"))
            and pd.notna(r.get("implied_low"))
            and pd.notna(r.get("implied_high"))
        ):
            try:
                im_pct = float(r["implied_move_pct"])
                if im_pct > 0:
                    left_pct = 50 - im_pct / 2
                    right_pct = 50 + im_pct / 2
                    i_color = f"linear-gradient(to right, var(--neg) 0%, var(--neg) {left_pct}%, var(--pos) {right_pct}%, var(--pos) 100%)"
                    impl_bar = f'<div class="range-container"><div class="range-title">Implied Move ±{im_pct:.1f}%</div><div class="range-bar" style="background:{i_color}"><div class="range-bar-marker" style="left:50%"></div></div><div class="range-labels"><span>${na(r["implied_low"])}</span><span>${na(r["implied_high"])}</span></div></div>'
            except Exception:
                impl_bar = ""

        # Only render range charts when values are present and valid (avoid NaN% rendering)
        day_block = ""
        if (
            pd.notna(r.get("day_low"))
            and pd.notna(r.get("day_high"))
            and r["day_high"] is not None
            and r["day_low"] is not None
            and (r["day_high"] - r["day_low"]) > 0
        ):
            day_pos = (r["price"] - r["day_low"]) / (r["day_high"] - r["day_low"]) * 100
            day_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {day_pos}%, var(--neg) {day_pos}%, var(--neg) 100%)"
            day_block = f"""
    <div class="range-container"><div class="range-title">Day</div><div class="range-bar" style="background:{day_color}"><div class="range-bar-marker" style="left:{day_pos}%"></div></div><div class="range-labels"><span>${r['day_low']:.2f}</span><span>${r['day_high']:.2f}</span></div></div>
    """

        y52_block = ""
        if (
            pd.notna(r.get("52w_low"))
            and pd.notna(r.get("52w_high"))
            and r["52w_high"] is not None
            and r["52w_low"] is not None
            and (r["52w_high"] - r["52w_low"]) > 0
        ):
            y52_pos = (r["price"] - r["52w_low"]) / (r["52w_high"] - r["52w_low"]) * 100
            y52_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {y52_pos}%, var(--neg) {y52_pos}%, var(--neg) 100%)"
            y52_block = f"""
    <div class="range-container"><div class="range-title">52W</div><div class="range-bar" style="background:{y52_color}"><div class="range-bar-marker" style="left:{y52_pos}%"></div></div><div class="range-labels"><span>${r['52w_low']:.2f}</span><span>${r['52w_high']:.2f}</span></div></div>
    """

        ranges_html = f"{day_block}{y52_block}{bb_bar}{impl_bar}"

        # CVR3 Signal color coding
        cvr3_color = "var(--neutral)"
        if r.get("cvr3_vix_signal") == "BUY":
            cvr3_color = "var(--pos)"
        elif r.get("cvr3_vix_signal") == "SELL":
            cvr3_color = "var(--neg)"
        elif r.get("cvr3_vix_signal") == "SHORT":
            cvr3_color = "#ff8800"

        # VIX change color coding
        vix_change_color = "var(--neutral)"
        if r.get("cvr3_vix_pct") is not None:
            if r.get("cvr3_vix_pct") > 0:
                vix_change_color = "var(--neg)"
            elif r.get("cvr3_vix_pct") < 0:
                vix_change_color = "var(--pos)"

        cvr3_html = ""
        if r.get("cvr3_vix_value") is not None:
            cvr3_html = f"""<br><span style="color: {cvr3_color}">CVR3: {r.get('cvr3_vix_signal') or 'NEUTRAL'}</span> <span style="color: {vix_change_color}">VIX: {na(r['cvr3_vix_value'], '{:.2f}')} ({('+' if (r.get('cvr3_vix_pct') or 0) >= 0 else '')}{na(r['cvr3_vix_pct'], '{:.2f}')})</span>"""

        indicators_html = f"""<span class="{macd_cls}">MACD: {r['macd_label']}</span><br>
Short: {na(r['short_percent'],"{:.1f}%")} ({na(r['days_to_cover'],"{:.1f}d")})<br>
<span class="{hv_cls}">Volatility: {hv_str}</span><br>
<span class="{opt_dir_cls}">Opt Dir: {r['options_direction']}</span><br>
<span class="{bias_cls}">Bias: {'Down' if r['down_volume_bias'] else 'Up'}</span>{cvr3_html}"""

        # include dividend dataset (percent) for filtering
        div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # sentiment text (exclude analyst rating)
        sent_text = r["sentiment"]
        # Get active signal for filtering
        active_sig = r.get("active_signal") or r.get("bb_signal") or ""
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        html += f"""<tr class="stock-row" data-ticker="{r['ticker']}" data-change="{r['change_pct']}" data-change-5d="{r.get('change_5d') or ''}" data-earnings="{r.get('earnings_date_iso') or ''}" data-rsi="{r['rsi'] or 50}" data-vol="{r['volume_raw']}" data-meme="{r['is_meme_stock']}" data-squeeze="{r['squeeze_level']}" data-bb-width="{bb_width_val}" data-dividend="{div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig}">
    <td><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.9em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.9em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.9em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.9em">S</a>)</td>
<td data-sort="{r['price']:.2f}">${r['price']:.2f} {r['sparkline']}</td>
<td>{fmt_change(r['change_pct'], r['change_abs_day'])}</td>
<td>{fmt_change(r.get('change_5d'), r.get('change_abs_5d'))}</td>
<td>{fmt_change(r['change_1m'], r['change_abs_1m'])}</td>
<td>{fmt_change(r['change_6m'], r['change_abs_6m'])}</td>
<td>{fmt_change(r['change_ytd'], r['change_abs_ytd'])}</td>
<td data-sort="{r['volume_raw']}">{fmt_vol(r['volume'])}</td>
<td>{ranges_html}</td>
<td>{indicators_html}</td>
<td><span class="{sent_cls}">{sent_text}</span><br><span class="{upside_cls}">Upside: {na(r['upside_potential'],"{:+.1f}%")}</span></td>
</tr>"""

    html += "</table></div><div id='cardView'><div class='card-grid'>"
    for _, r in df.iterrows():
        bg = "rgba(0,170,0,0.1)" if r["change_pct"] > 0 else "rgba(204,0,0,0.1)"
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.5em">🟠</span> '
        bb_icon = signal_icon  # Alias for backward compatibility
        hv = r["hv_30_annualized"]
        hv_cls = "negative" if hv and hv > 50 else "neutral"
        hv_str = na(hv, "{:.1f}%")
        # Color coding for card attributes
        macd_num_cls = "neutral"
        try:
            ml = r.get("macd_line")
            macd_num_cls = (
                "positive"
                if ml is not None and float(ml) >= 0
                else "negative" if ml is not None else "neutral"
            )
        except Exception:
            macd_num_cls = "neutral"

        pc_val = r.get("pc_ratio")
        if pc_val is None:
            pc_cls = "neutral"
        else:
            try:
                pv = float(pc_val)
                pc_cls = (
                    "negative" if pv > 1.2 else "positive" if pv < 0.8 else "neutral"
                )
            except Exception:
                pc_cls = "neutral"

        pe_val = r.get("pe")
        if pe_val is None:
            pe_cls = "neutral"
        else:
            try:
                pv = float(pe_val)
                pe_cls = "negative" if pv > 30 else "positive" if pv < 15 else "neutral"
            except Exception:
                pe_cls = "neutral"

        div_val = r.get("dividend_yield")
        div_cls = "positive" if div_val is not None and div_val > 0 else "neutral"
        # Display dividend yield with a '%' suffix but keep the raw dataset unmodified
        div_yield_display = na(div_val, "{}") + "%" if div_val is not None else "N/A"

        mcap_val = r.get("market_cap")
        aum_val = r.get("aum")
        # determine display value and label (Market Cap preferred, fallback to AUM)
        display_val = None
        display_label = "Market Cap"
        if mcap_val is not None and pd.notna(mcap_val):
            display_val = mcap_val
            display_label = "Market Cap"
        elif aum_val is not None and pd.notna(aum_val):
            display_val = aum_val
            display_label = "AUM"
        try:
            base_val = float(display_val) if display_val is not None else None
        except Exception:
            base_val = None
        try:
            mcap_cls = (
                "strong-bull"
                if base_val is not None and base_val >= 200e9
                else "bull" if base_val is not None and base_val >= 10e9 else "neutral"
            )
        except Exception:
            mcap_cls = "neutral"

        # 52W display
        y52_low = r.get("52w_low")
        y52_high = r.get("52w_high")
        if pd.notna(y52_low) and pd.notna(y52_high):
            y52_display = f"${y52_low:.2f} - ${y52_high:.2f}"
        else:
            y52_display = "N/A"

        # include dividend dataset for card
        card_div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # Option direction and short percent/days color coding for card
        opt_dir_val = r.get("options_direction") or "Neutral"
        if "Strong Bear" in opt_dir_val or "Strong Bearish" in opt_dir_val:
            opt_dir_cls = "strong-bear"
        elif "Bear" in opt_dir_val:
            opt_dir_cls = "bear"
        elif "Strong Bull" in opt_dir_val or "Strong Bullish" in opt_dir_val:
            opt_dir_cls = "strong-bull"
        elif "Bull" in opt_dir_val:
            opt_dir_cls = "bull"
        else:
            opt_dir_cls = "neutral"

        short_pct = r.get("short_percent")
        days_cover = r.get("days_to_cover")
        try:
            sp = float(short_pct) if short_pct is not None else None
        except Exception:
            sp = None
        if sp is None:
            short_cls = "neutral"
        elif sp >= 20:
            short_cls = "strong-bear"
        elif sp >= 10:
            short_cls = "bear"
        elif sp >= 5:
            short_cls = "vol-hot"
        else:
            short_cls = "neutral"
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        active_sig = r.get("active_signal") or r.get("bb_signal") or ""
        html += f"""<div class="stock-card stock-row" style="background:{bg}" 
            data-ticker="{r['ticker']}" 
            data-change="{r['change_pct']}" 
            data-change-5d="{r.get('change_5d') or ''}" 
            data-earnings="{r.get('earnings_date_iso') or ''}"
            data-rsi="{r['rsi'] or 50}" 
            data-vol="{r['volume_raw']}" 
            data-meme="{r['is_meme_stock']}" 
            data-squeeze="{r['squeeze_level']}" 
            data-bb-width="{bb_width_val}" data-dividend="{card_div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig}">
    <h2><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.8em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.8em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.8em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.8em">S</a>) ${r['price']:.2f}</h2>
<div style="font-size:1.5em">{fmt_change(r['change_pct'], r['change_abs_day'])}</div>
{r['sparkline']}
<div>1M: {fmt_change(r['change_1m'], r['change_abs_1m'])}</div>
<div>5D: {fmt_change(r.get('change_5d'), r.get('change_abs_5d'))}</div>
<div>6M: {fmt_change(r['change_6m'], r['change_abs_6m'])}</div>
<div>YTD: {fmt_change(r['change_ytd'], r['change_abs_ytd'])}</div>
<div><strong>52W: {y52_display}</strong></div>
<div><span class="{hv_cls}">Volatility: {hv_str}</span></div>
<div>BB: {r['bb_status']} ({na(r['bb_width_pct'], '{:.1f}%')})</div>
<div>MACD: <span class="{macd_num_cls}">{na(r.get('macd_line'), '{:+.3f}')}</span> | <span class="{macd_num_cls}">{na(r.get('macd_signal'), '{:+.3f}')}</span> (<span class="{ 'bullish' if r.get('macd_label')=='Bullish' else 'bearish' if r.get('macd_label')=='Bearish' else 'neutral' }">{r.get('macd_label','N/A')}</span>)</div>
<div>P/E: <span class="{pe_cls}">{na(r.get('pe'), '{:.2f}')}</span></div>
<div>EPS: <span class="{pe_cls}">{na(r.get('eps'), '{:.2f}')}</span></div>
<div>Div: <span class="{div_cls}">{na(r.get('dividend_rate'), '${:.2f}')}</span> (<span class="{div_cls}">{div_yield_display}</span>)</div>
<div>{display_label}: <span class="{mcap_cls}"><strong>{fmt_mcap(display_val)}</strong></span></div>
<div>Earnings: <strong>{r.get('earnings_date') or 'N/A'}</strong></div>
<div>P/C Vol Ratio: <span class="{pc_cls}">{na(r.get('pc_ratio'), '{:.2f}')}</span></div>
<div><strong>Opt Dir: <span class="{opt_dir_cls}">{opt_dir_val}</span> &nbsp; Short: <span class="{short_cls}">{na(short_pct, '{:.1f}%')}</span> ({na(days_cover, '{:.1f}d')})</strong></div>
</div>"""

    html += "</div></div><div id='heatView'><div class='heat-grid'>"
    for _, r in df.iterrows():
        intensity = min(abs(r["change_pct"]) / 15, 1)
        bg = (
            f"rgba(0,170,0,{intensity})"
            if r["change_pct"] >= 0
            else f"rgba(204,0,0,{intensity})"
        )
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        price_display = (
            f"${r['price']:.2f}"
            if (r.get("price") is not None and pd.notna(r.get("price")))
            else "N/A"
        )

        # Prefer Market Cap, fallback to AUM when market cap missing
        mcap_val = r.get("market_cap")
        aum_val = r.get("aum")
        display_val = None
        display_label = "Market Cap"
        if mcap_val is not None and pd.notna(mcap_val):
            display_val = mcap_val
            display_label = "Market Cap"
        elif aum_val is not None and pd.notna(aum_val):
            display_val = aum_val
            display_label = "AUM"
        try:
            base_val = float(display_val) if display_val is not None else None
        except Exception:
            base_val = None
        try:
            mcap_cls = (
                "strong-bull"
                if base_val is not None and base_val >= 200e9
                else "bull" if base_val is not None and base_val >= 10e9 else "neutral"
            )
        except Exception:
            mcap_cls = "neutral"

        # 52W range
        y52_low = r.get("52w_low")
        y52_high = r.get("52w_high")
        if pd.notna(y52_low) and pd.notna(y52_high):
            y52_display = f"${y52_low:.2f} - ${y52_high:.2f}"
        else:
            y52_display = "N/A"

        # include dividend dataset for heat tiles
        heat_div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.5em">🟠</span> '
        bb_icon = signal_icon  # Alias for backward compatibility
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        active_sig_heat = r.get("active_signal") or r.get("bb_signal") or ""
        html += f"""<div class="heat-tile stock-row" style="background:{bg}" 
        data-ticker="{r['ticker']}" 
        data-change="{r['change_pct']}" 
        data-change-5d="{r.get('change_5d') or ''}"
        data-earnings="{r.get('earnings_date_iso') or ''}"
        data-rsi="{r['rsi'] or 50}" 
        data-vol="{r['volume_raw']}" 
        data-meme="{r['is_meme_stock']}" 
        data-squeeze="{r['squeeze_level']}" 
        data-bb-width="{bb_width_val}" data-dividend="{heat_div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig_heat}">
    <strong><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.85em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.85em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.85em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.85em">S</a>) {price_display}</strong>
    <div style="margin-top:6px">{fmt_change(r['change_pct'], r.get('change_abs_day'))}</div>
    <div style="font-size:0.85em">5D: {fmt_change(r.get('change_5d'), r.get('change_abs_5d'))}</div>
    <div style="font-size:0.9em">{display_label}: <span class="{mcap_cls}"><strong>{fmt_mcap(display_val)}</strong></span></div>
    <div style="font-size:0.9em;margin-top:6px"><strong>52W: {y52_display}</strong></div>
    </div>"""

    html += "</div></div></div>"

    html += """
<script>
const prefsKey = 'dash_prefs';
let prefs = JSON.parse(localStorage.getItem(prefsKey) || '{"theme":"light","view":"table"}');
document.documentElement.setAttribute('data-theme', prefs.theme);

function setView(btn, view) {
    document.getElementById('tableView').style.display = view==='table' ? 'block' : 'none';
    document.getElementById('cardView').style.display = view==='card' ? 'block' : 'none';
    document.getElementById('heatView').style.display = view==='heat' ? 'block' : 'none';
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    prefs.view = view;
    localStorage.setItem(prefsKey, JSON.stringify(prefs));
    applyFilter();
}

function toggleHours(extended) {
    const newFile = extended ? 'extnd_dashboard.html' : 'reg_dashboard.html';
    if (window.location.pathname.endsWith(newFile)) return;
    window.location.href = newFile;
}

let currentFilter = 'all';
function applyFilter() {
    const rows = document.querySelectorAll('.stock-row');
    const tickerVal = (document.getElementById('tickerFilter') && document.getElementById('tickerFilter').value) ? document.getElementById('tickerFilter').value.trim().toLowerCase() : '';
    rows.forEach(r => {
        let show = true;
        const ch = parseFloat(r.dataset.change || 0);
        const ch5 = parseFloat(r.dataset.change5d || 0);
        const rsi = parseFloat(r.dataset.rsi || 50);
        const vol = parseFloat(r.dataset.vol || 0);
        const meme = r.dataset.meme === 'True';
        const cat = (r.dataset.category || '').toLowerCase();
        const sq = r.dataset.squeeze || 'None';
        const bbw = parseFloat(r.dataset.bbWidth || 100);
        const sig = (r.dataset.signal || '').toUpperCase();
        if (currentFilter === 'oversold') show = rsi < 30;
        else if (currentFilter === 'overbought') show = rsi > 70;
        else if (currentFilter === 'surge') show = (ch > 10) || (ch5 > 10);
        else if (currentFilter === 'crash') show = (ch < -10) || (ch5 < -10);
        else if (currentFilter === 'meme') show = meme;
        else if (currentFilter === 'volume') show = vol > 5e7;
        else if (currentFilter === 'squeeze') show = sq !== 'None';
        else if (currentFilter === 'earnings-week') show = (function(){
            const ed = r.dataset.earnings;
            if (!ed) return false;
            try {
                const edDate = new Date(ed + 'T00:00:00');
                edDate.setHours(0,0,0,0);
                const now = new Date();
                const day = now.getDay();
                const start = new Date(now);
                start.setHours(0,0,0,0);
                start.setDate(now.getDate() - day); // week starts Sunday
                const end = new Date(start);
                end.setDate(start.getDate() + 6);
                return edDate >= start && edDate <= end;
            } catch (e) {
                return false;
            }
        })();
        else if (currentFilter === 'bb-squeeze') show = bbw < 6;
        else if (currentFilter === 'dividend') show = parseFloat(r.dataset.dividend || 0) > 0;
        else if (currentFilter === 'signal-buy') show = sig === 'BUY';
        else if (currentFilter === 'signal-sell') show = sig === 'SELL';
        else if (currentFilter === 'signal-short') show = sig === 'SHORT';
        else if (currentFilter.startsWith('cat-')) show = cat === currentFilter.replace('cat-','');
        if (tickerVal) {
            const tk = (r.dataset.ticker || '').toString().toLowerCase();
            show = show && tk.includes(tickerVal);
        }
        r.style.display = show ? '' : 'none';
    });
}

document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', function() {
    document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
    this.classList.add('active');
    currentFilter = this.dataset.filter;
    applyFilter();
}));

document.querySelectorAll('th[data-sort]').forEach((th, col) => {
    th.onclick = () => {
        const table = document.getElementById('stockTable');
        const rows = Array.from(table.querySelectorAll('tr:nth-child(n+2)'));
        const dir = th.dataset.dir = (th.dataset.dir === 'asc' ? 'desc' : 'asc');
        rows.sort((a, b) => {
            let av = a.cells[col].querySelector('[data-sort]')?.dataset.sort || a.cells[col].textContent.trim();
            let bv = b.cells[col].querySelector('[data-sort]')?.dataset.sort || b.cells[col].textContent.trim();
            av = isNaN(parseFloat(av)) ? av : parseFloat(av);
            bv = isNaN(parseFloat(bv)) ? bv : parseFloat(bv);
            if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * (dir === 'asc' ? 1 : -1);
            return av.localeCompare(bv, undefined, {numeric: true}) * (dir === 'asc' ? 1 : -1);
        });
        rows.forEach(r => table.appendChild(r));
    };
});

applyFilter();
if (prefs.view !== 'table') {
    const btn = document.querySelector(`.view-btn:nth-child(${prefs.view==='card'?2:3})`);
    if (btn) setView(btn, prefs.view);
}

function toggleTheme() {
    prefs.theme = prefs.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', prefs.theme);
    localStorage.setItem(prefsKey, JSON.stringify(prefs));
}
</script>
</body></html>"""

    with open(file, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    import time as time_module
    start_time = time_module.time()
    
    os.makedirs("data", exist_ok=True)
    # Pre-load alerts cache at startup
    load_alerts()
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default="data/tickers.csv")
    args = parser.parse_args()

    for ext, file, name in [
        (False, "data/reg_dashboard.html", "Regular"),
        (True, "data/extnd_dashboard.html", "Extended"),
    ]:
        try:
            dashboard_start = time_module.time()
            df = dashboard(args.csv_file, ext)
            alerts = check_alerts(df.to_dict("records"))
            html(
                df,
                get_vix_data(),
                get_fear_greed_data(),
                get_aaii_sentiment(),
                file,
                ext,
                alerts=alerts,
            )
            dashboard_elapsed = time_module.time() - dashboard_start
            print(f"✓ {name} Hours Dashboard generated: {file} (took {dashboard_elapsed / 60:.2f} minutes)")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    total_elapsed = time_module.time() - start_time
    print(f"\n⏱️  Total time: {total_elapsed / 60:.2f} minutes")