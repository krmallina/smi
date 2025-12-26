import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json
import requests
import re
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

ALERTS_FILE = 'data/alerts.json'
UTC = pytz.utc
PST = pytz.timezone('America/Los_Angeles')
MEME_STOCKS = frozenset({'GME','AMC','BB','KOSS','EXPR','DJT','HOOD','RDDT','SPCE','RIVN','DNUT','OPEN','KSS','RKLB','GPRO','AEO','BYND','CVNA','PLTR','SMCI'})

@lru_cache(maxsize=1)
def load_alerts():
    try:
        with open(ALERTS_FILE) as f:
            return tuple(json.load(f))
    except:
        return ()

def check_alerts(data):
    custom_alerts = load_alerts()
    now = datetime.now(UTC).astimezone(PST)
    high_52w, low_52w, surge, crash, volume_spike, custom = [], [], [], [], [], []
    stock_dict = {x['ticker']: x for x in data}

    for a in custom_alerts:
        s = stock_dict.get(a['ticker'].upper())
        if not s: continue
        msg, cond, val = "", a['condition'], a.get('value')
        if cond == "price_above" and val and s['price'] > val: msg = f"price ABOVE ${val:.2f} → ${s['price']:.2f}"
        elif cond == "price_below" and val and s['price'] < val: msg = f"price BELOW ${val:.2f} → ${s['price']:.2f}"
        elif cond == "day_change_above" and val and s['change_pct'] > val: msg = f"DAY % ABOVE {val}% → {s['change_pct']:+.2f}%"
        elif cond == "day_change_below" and val and s['change_pct'] < val: msg = f"DAY % BELOW {val}% → {s['change_pct']:+.2f}%"
        elif cond == "rsi_oversold" and s['rsi'] is not None and s['rsi'] < 30: msg = f"RSI OVERSOLD → {s['rsi']:.1f}"
        elif cond == "rsi_overbought" and s['rsi'] is not None and s['rsi'] > 70: msg = f"RSI OVERBOUGHT → {s['rsi']:.1f}"
        elif cond == "volume_spike" and s['volume_spike']: msg = "VOLUME SPIKE"
        if msg: custom.append({'ticker': s['ticker'], 'msg': msg})

    for s in data:
        ch = s['change_pct']
        if ch > 15: surge.append({'ticker': s['ticker'], 'msg': f"SURGED > +15% → {ch:+.2f}%"})
        elif ch < -15: crash.append({'ticker': s['ticker'], 'msg': f"CRASHED < -15% → {ch:+.2f}%"})
        if s['volume_spike']: volume_spike.append({'ticker': s['ticker'], 'msg': "VOLUME SPIKE"})
        range_52w = s['52w_high'] - s['52w_low']
        if range_52w > 0:
            pos_pct = (s['price'] - s['52w_low']) / range_52w * 100
            if pos_pct >= 95: high_52w.append({'ticker': s['ticker'], 'msg': f"NEAR 52W HIGH ({pos_pct:.1f}%)"})
            elif pos_pct <= 5: low_52w.append({'ticker': s['ticker'], 'msg': f"NEAR 52W LOW ({pos_pct:.1f}%)"})

    def fmt(items, emoji, label):
        return f"{emoji} <strong>{label}:</strong> {', '.join(a['ticker'] for a in items)}"
    
    grouped = []
    if high_52w: grouped.append(fmt(high_52w, "🔥", "52W High"))
    if low_52w: grouped.append(fmt(low_52w, "📉", "52W Low"))
    if surge: grouped.append(fmt(surge, "🚀", "Surge"))
    if crash: grouped.append(fmt(crash, "💥", "Crash"))
    if volume_spike: grouped.append(fmt(volume_spike, "📈", "Vol Spike"))
    if custom: grouped.append(f"⚡ <strong>Custom:</strong> {len(custom)}")
    return {'grouped': grouped, 'time': now.strftime('%I:%M %p')}

def rsi(s):
    if len(s) < 15: return None
    d = s.diff()
    g, l = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    return (100 - 100 / (1 + g / l)).iloc[-1]

def macd(s):
    if len(s) < 26: return None, None, "N/A"
    e12, e26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    line, sig = e12 - e26, (e12 - e26).ewm(span=9, adjust=False).mean()
    return line.iloc[-1], sig.iloc[-1], "Bullish" if line.iloc[-1] > sig.iloc[-1] else "Bearish"

def na(v, f="{:.2f}"):
    return "N/A" if v is None or pd.isna(v) else f.format(v)

def sparkline(prices):
    if len(prices) < 2: return ""
    prices = prices[-30:]
    mn, mx = min(prices), max(prices)
    rng = mx - mn if mx != mn else 1
    w, h = 60, 20
    pts = [f"{(i/(len(prices)-1))*w:.1f},{h-((p-mn)/rng*h):.1f}" for i,p in enumerate(prices)]
    c = "#00aa00" if prices[-1] >= prices[0] else "#cc0000"
    return f'<svg width="{w}" height="{h}"><polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="1.5"/></svg>'

def fetch(ticker, ext=False):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        h_day = t.history(period="1d", interval="1m", prepost=ext)
        if h_day.empty: h_day = t.history(period="5d", prepost=ext)
        if h_day.empty: return None
        
        price = h_day['Close'].iloc[-1]
        day_low, day_high = h_day['Low'].min(), h_day['High'].max()
        reg = t.history(period="5d", prepost=False)
        change_pct = change_abs_day = 0.0
        if len(reg) >= 2:
            prev = reg['Close'].iloc[-2]
            if prev and prev > 0:
                change_pct = ((price - prev) / prev) * 100
                change_abs_day = price - prev

        h1m, h6m = t.history(period="2mo"), t.history(period="7mo")
        ytd = t.history(start=datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d'))
        y, h30 = t.history(period="1y"), t.history(period="60d")
        
        def calc_ch(h):
            if len(h) >= 2 and h['Close'].iloc[0] > 0:
                return ((price - h['Close'].iloc[0]) / h['Close'].iloc[0]) * 100, price - h['Close'].iloc[0]
            return None, None
        
        ch1m, abs1m = calc_ch(h1m)
        ch6m, abs6m = calc_ch(h6m)
        chytd, absytd = calc_ch(ytd)
        
        high52, low52 = (y['High'].max(), y['Low'].min()) if not y.empty else (price, price)
        vol = h_day['Volume'].sum()
        
        hv = None
        if len(h30) >= 30:
            r = h30['Close'].pct_change().dropna()
            if len(r) > 1: hv = r.std() * (252**0.5) * 100
        
        short_pct = info.get('shortPercentOfFloat')
        if short_pct: short_pct *= 100
        
        days_cover = None
        if info.get('sharesShort'):
            avg = info.get('averageDailyVolume10Day') or info.get('averageVolume') or vol or 1
            if avg > 0: days_cover = info['sharesShort'] / avg
        
        squeeze = "None"
        if short_pct and days_cover:
            if short_pct > 30 and days_cover > 10: squeeze = "Extreme"
            elif short_pct > 20 and days_cover > 7: squeeze = "High"
            elif short_pct > 15 and days_cover > 5: squeeze = "Moderate"
        
        rsi_val = rsi(h30['Close'])
        macd_val, macd_sig, macd_lbl = macd(t.history(period="100d")['Close'])
        
        vol_spike = False
        if len(h30) > 1:
            avg = h30['Volume'][:-1].mean()
            if avg > 0: vol_spike = vol > 1.5 * avg
        
        pc_ratio = impl_move = impl_hi = impl_lo = exp_date = None
        if t.options:
            exp_date = t.options[0]
            try:
                chain = t.option_chain(exp_date)
                strikes = pd.concat([chain.calls['strike'], chain.puts['strike']]).unique()
                if len(strikes) > 0:
                    atm = min(strikes, key=lambda s: abs(s - price))
                    cp = chain.calls.loc[chain.calls['strike'] == atm, 'lastPrice'].iloc[0] if not chain.calls[chain.calls['strike'] == atm].empty else 0
                    pp = chain.puts.loc[chain.puts['strike'] == atm, 'lastPrice'].iloc[0] if not chain.puts[chain.puts['strike'] == atm].empty else 0
                    straddle = cp + pp
                    if straddle > 0:
                        impl_move = (straddle / price) * 100
                        cons = impl_move * 0.85
                        impl_hi = price * (1 + cons/100)
                        impl_lo = price * (1 - cons/100)
                        cvol, pvol = chain.calls['volume'].fillna(0).sum(), chain.puts['volume'].fillna(0).sum()
                        if cvol > 0: pc_ratio = pvol / cvol
            except: pass
        
        down_bias = (h30[h30['Close'] < h30['Open']]['Volume'].sum() > h30[h30['Close'] > h30['Open']]['Volume'].sum()) if len(h30) > 0 else False
        opt_dir = "Neutral"
        if pc_ratio:
            if pc_ratio > 1.2 and down_bias: opt_dir = "Strong Bearish"
            elif pc_ratio > 1.0 or down_bias: opt_dir = "Bearish"
            elif pc_ratio < 0.8 and not down_bias: opt_dir = "Bullish"
        
        sentiment = ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")[
            0 if info.get('recommendationMean', 5) <= 1.5 else 1 if info.get('recommendationMean', 5) <= 2.5 else 2 if info.get('recommendationMean', 5) <= 3.5 else 3 if info.get('recommendationMean', 5) <= 4.5 else 4
        ]
        
        rating = info.get('recommendationKey', 'none').title().replace('_', ' ')
        target = info.get('targetMeanPrice')
        upside = ((target - price) / price) * 100 if target and price > 0 else None
        spk = sparkline(h30['Close'].tolist() if not h30.empty else [])
        
        # Bollinger Bands
        bb_period = 20
        bb_upper = bb_lower = bb_middle = bb_width_pct = bb_position_pct = bb_status = None
        if len(h30) >= bb_period:
            close = h30['Close']
            bb_middle = close.rolling(window=bb_period).mean().iloc[-1]
            std_dev = close.rolling(window=bb_period).std().iloc[-1]
            bb_upper = bb_middle + (std_dev * 2)
            bb_lower = bb_middle - (std_dev * 2)
            if bb_middle > 0:
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
            if (bb_upper - bb_lower) > 0:
                bb_position_pct = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
                bb_position_pct = max(0, min(100, bb_position_pct))
            bb_status = "Above Upper" if price > bb_upper else "Below Lower" if price < bb_lower else "Inside"
        
        time.sleep(1.5)
        tu = ticker.upper()
        return {
            'ticker': tu, 'price': price, 'change_pct': change_pct, 'change_abs_day': change_abs_day,
            'change_1m': ch1m, 'change_abs_1m': abs1m, 'change_6m': ch6m, 'change_abs_6m': abs6m,
            'change_ytd': chytd, 'change_abs_ytd': absytd, 'volume': vol, 'volume_raw': vol,
            '52w_high': high52, '52w_low': low52, 'day_low': day_low, 'day_high': day_high,
            'short_percent': short_pct, 'days_to_cover': days_cover,
            'squeeze_level': squeeze, 'rsi': rsi_val,
            'macd_label': macd_lbl, 'volume_spike': vol_spike,
            'is_meme_stock': tu in MEME_STOCKS, 'sentiment': sentiment,
            'analyst_rating': rating, 'upside_potential': upside,
            'options_direction': opt_dir, 'implied_move_pct': impl_move,
            'implied_high': impl_hi, 'implied_low': impl_lo,
            'down_volume_bias': down_bias, 'sparkline': spk,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle,
            'bb_width_pct': bb_width_pct, 'bb_position_pct': bb_position_pct, 'bb_status': bb_status,
            'hv_30_annualized': hv,
        }
    except Exception as e:
        print(f"Error {ticker}: {e}")
        time.sleep(20)
        return None

def fmt_vol(v):
    if v is None: return "N/A"
    if v >= 1e9: return f"{v/1e9:.1f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return str(int(v))

def fmt_change(p, a=None):
    if p is None: return '<span class="neutral">N/A</span>'
    sign, cls = ("▲", "positive") if p >= 0 else ("▼", "negative")
    abs_str = f' ({a:+.2f})' if a is not None else ''
    return f'<span class="{cls}" data-sort="{p:.10f}">{sign} {p:+.2f}%{abs_str}</span>'

def get_index_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        ch_pct = info.get('regularMarketChangePercent')
        prev = info.get('regularMarketPreviousClose') or info.get('previousClose')
        ch_abs = None
        if price is not None and prev is not None:
            try:
                ch_abs = price - prev
            except Exception:
                ch_abs = None
        if ch_pct is None and price is not None and prev is not None and prev > 0:
            ch_pct = ((price - prev) / prev) * 100
        return {'price': price, 'change_pct': ch_pct, 'change_abs': ch_abs}
    except:
        return {'price': None, 'change_pct': None, 'change_abs': None}

def dashboard(csv='data/tickers.csv', ext=False):
    os.makedirs('data', exist_ok=True)
    try:
        # Support single-line CSV (comma-separated) or multi-line (one ticker per line).
        with open(csv, 'r', encoding='utf-8') as f:
            txt = f.read()
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        parts = re.split(r'[\n,]+', txt.strip())
        parts = [p.strip().upper() for p in parts if p and p.strip()]
        # Drop header if present
        if parts and parts[0].lower() in ('ticker', 'tickers'):
            parts = parts[1:]
        tickers = pd.Series(parts).unique().tolist()
    except Exception:
        tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','SPY']
    
    data = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(fetch, t, ext) for t in tickers]
        for r in as_completed(futures):
            res = r.result()
            if res: data.append(res)
    return pd.DataFrame(data).sort_values('change_pct', ascending=False)

def get_vix_data():
    return get_index_data("^VIX")

def get_fear_greed_data():
    try:
        # Try date-specific endpoint first (more reliable)
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Referer': 'https://www.cnn.com/markets/fear-and-greed'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        fg = data.get('fear_and_greed') or data
        score = float(fg['score'])
        s = int(round(score))
        rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
            0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
        ]
        return {'score': score, 'rating': rating, 'raw_score': s}
    except Exception:
        try:
            # Fallback to generic endpoint
            r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                             headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            r.raise_for_status()
            data = r.json()
            score = float(data['fear_and_greed']['score'])
            s = int(round(score))
            rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
                0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
            ]
            return {'score': score, 'rating': rating, 'raw_score': s}
        except Exception as e:
            print(f"F&G error: {e}")
    return {'score': None, 'rating': "N/A", 'raw_score': None}

def get_aaii_sentiment():
    try:
        r = requests.get("https://www.aaii.com/sentimentsurvey/sent_results", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        # Try a couple of regex patterns to extract Bullish / Bearish percentages
        m = re.search(r'\w+\s*\d{1,2}.*?([\d\.]+)%.*?([\d\.]+)%', r.text)
        if not m:
            m = re.search(r'Bullish.*?([\d\.]+)%.*?Bearish.*?([\d\.]+)%', r.text, re.DOTALL)
        if m:
            b, be = float(m.group(1)), float(m.group(2))
            return {'bullish': b, 'bearish': be, 'spread': b - be}
    except Exception as e:
        print(f"AAII fetch error: {e}")
    return {'bullish': None, 'bearish': None, 'spread': None}

def html(df, vix, fg, aaii, file, ext=False, alerts=None):
    alerts = alerts or {'grouped': [], 'time': ''}
    update = datetime.now(UTC).astimezone(PST).strftime('%I:%M:%S %p PST on %B %d, %Y')
    
    banner = '<div class="alert-banner">🚨 <strong>ALERTS</strong> ' + " | ".join(alerts['grouped']) + '</div>' if alerts['grouped'] else ""
    
    # Major indices
    dow = get_index_data("^DJI")
    sp = get_index_data("^GSPC")
    nas = get_index_data("^IXIC")
    
    def index_str(data, name):
        if data['price'] is None:
            return f'<span class="neutral">{name}: N/A</span>'
        ch_abs = data.get('change_abs')
        cls = "positive" if ch_abs is not None and ch_abs >= 0 else "negative"
        return f'<span class="{cls}">{name}: {na(data["price"])} ({na(ch_abs, "{:+.2f}")})</span>'
    
    indices_h = f"{index_str(dow, 'Dow')} {index_str(sp, 'S&P')} {index_str(nas, 'Nasdaq')} {index_str(vix, 'VIX')}"
    
    fg_h = '<span class="neutral">F&G: N/A</span>'
    if fg.get('score') is not None:
        cls = "negative" if fg['score'] <= 24 else "high-risk" if fg['score'] <= 44 else "neutral" if fg['score'] <= 55 else "bullish" if fg['score'] <= 74 else "positive"
        fg_h = f'<span class="{cls}">F&G: {fg["score"]:.1f} ({fg["rating"]})</span>'
    
    aaii_h = '<span class="neutral">AAII: N/A</span>'
    if aaii.get('bullish') is not None:
        spread = aaii['spread']
        cls = "positive" if spread > 20 else "bullish" if spread > 0 else "neutral" if spread > -20 else "high-risk" if spread > -40 else "negative"
        aaii_h = f'<span class="{cls}">AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%</span>'
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Enhanced Dashboard</title>
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
.bearish{{color:var(--bearish);font-weight:bold}}
.extreme-fear{{color:#ff0000;font-weight:bold}}
.fear{{color:#ff8800}}
.greed{{color:#88ff88}}
.extreme-greed{{color:#00bb00;font-weight:bold}}
.strong-bull{{color:#008800;font-weight:bold}}
.bull{{color:#00bb00}}
.strong-bear{{color:#ff0000;font-weight:bold}}
.bear{{color:#ff8800}}
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
<div><h1>📊 Enhanced Dashboard</h1><small>{update}</small></div>
<div style="display:flex;gap:15px;flex-wrap:wrap;align-items:center">
<span>{indices_h}</span><span>{fg_h}</span><span>{aaii_h}</span>
<div class="hours-toggle">
<span>Regular</span>
<label class="toggle-switch">
<input type="checkbox" {'checked' if ext else ''} onchange="toggleHours(this.checked)">
<span class="toggle-slider"></span>
</label>
<span>Extended</span>
</div>
    <button class="btn" onclick="toggleTheme()">🌓</button>
    <button class="btn" onclick="location.reload()">🔄</button>
</div>
</div>

<div class="quick-filters">
<div class="chip active" data-filter="all">All</div>
<div class="chip" data-filter="oversold">📉 Oversold</div>
<div class="chip" data-filter="overbought">📈 Overbought</div>
<div class="chip" data-filter="surge">🚀 Surge</div>
<div class="chip" data-filter="crash">💥 Crash</div>
<div class="chip" data-filter="meme">🎮 Meme</div>
<div class="chip" data-filter="volume">📊 High Vol</div>
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
        bb_width_val = r['bb_width_pct'] if r['bb_width_pct'] is not None else 100
        hv = r['hv_30_annualized']
        hv_cls = "vol-hot" if hv and hv > 50 else "neutral"
        hv_str = na(hv, '{:.1f}%')
        
        macd_cls = "bullish" if r['macd_label'] == "Bullish" else "bearish" if r['macd_label'] == "Bearish" else "neutral"
        opt_dir_cls = "bullish" if "Bullish" in r['options_direction'] else "bearish" if "Bearish" in r['options_direction'] else "neutral"
        bias_cls = "bearish" if r['down_volume_bias'] else "bullish"
        
        sent_cls = "bullish" if "Buy" in r['sentiment'] else "bearish" if "Sell" in r['sentiment'] else "neutral"
        upside_cls = "bullish" if r['upside_potential'] and r['upside_potential'] > 0 else "bearish" if r['upside_potential'] and r['upside_potential'] < 0 else "neutral"
        
        bb_bar = ""
        # Only render Bollinger Bands bar when BB values are present and numeric
        if pd.notna(r.get('bb_position_pct')) and pd.notna(r.get('bb_lower')) and pd.notna(r.get('bb_middle')) and pd.notna(r.get('bb_upper')):
            try:
                pos = float(r['bb_position_pct'])
                pos = max(0.0, min(100.0, pos))
                bb_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos}%, var(--neg) {pos}%, var(--neg) 100%)"
                bb_bar = f'<div class="range-container"><div class="range-title">Bollinger Bands</div><div class="range-bar" style="background:{bb_color}"><div class="range-bar-marker" style="left:{pos}%"></div></div><div class="range-labels"><span>${na(r["bb_lower"])}</span><span>${na(r["bb_middle"])}</span><span>${na(r["bb_upper"])}</span></div><div style="font-size:0.75em;text-align:center">Width: {na(r["bb_width_pct"],"{:.1f}")}% – {r["bb_status"]}</div></div>'
            except Exception:
                bb_bar = ""
        
        impl_bar = ""
        # Only render implied-move chart when values are present and numeric (avoid NaN%)
        if pd.notna(r.get('implied_move_pct')) and pd.notna(r.get('implied_low')) and pd.notna(r.get('implied_high')):
            try:
                im_pct = float(r['implied_move_pct'])
                if im_pct > 0:
                    left_pct = 50 - im_pct/2
                    right_pct = 50 + im_pct/2
                    i_color = f"linear-gradient(to right, var(--neg) 0%, var(--neg) {left_pct}%, var(--pos) {right_pct}%, var(--pos) 100%)"
                    impl_bar = f'<div class="range-container"><div class="range-title">Implied Move ±{im_pct:.1f}%</div><div class="range-bar" style="background:{i_color}"><div class="range-bar-marker" style="left:50%"></div></div><div class="range-labels"><span>${na(r["implied_low"])}</span><span>${na(r["implied_high"])}</span></div></div>'
            except Exception:
                impl_bar = ""
        
        # Only render range charts when values are present and valid (avoid NaN% rendering)
        day_block = ''
        if pd.notna(r.get('day_low')) and pd.notna(r.get('day_high')) and r['day_high'] is not None and r['day_low'] is not None and (r['day_high'] - r['day_low']) > 0:
            day_pos = ((r['price'] - r['day_low']) / (r['day_high'] - r['day_low']) * 100)
            day_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {day_pos}%, var(--neg) {day_pos}%, var(--neg) 100%)"
            day_block = f"""
    <div class="range-container"><div class="range-title">Day</div><div class="range-bar" style="background:{day_color}"><div class="range-bar-marker" style="left:{day_pos}%"></div></div><div class="range-labels"><span>${r['day_low']:.2f}</span><span>${r['day_high']:.2f}</span></div></div>
    """

        y52_block = ''
        if pd.notna(r.get('52w_low')) and pd.notna(r.get('52w_high')) and r['52w_high'] is not None and r['52w_low'] is not None and (r['52w_high'] - r['52w_low']) > 0:
            y52_pos = ((r['price'] - r['52w_low']) / (r['52w_high'] - r['52w_low']) * 100)
            y52_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {y52_pos}%, var(--neg) {y52_pos}%, var(--neg) 100%)"
            y52_block = f"""
    <div class="range-container"><div class="range-title">52W</div><div class="range-bar" style="background:{y52_color}"><div class="range-bar-marker" style="left:{y52_pos}%"></div></div><div class="range-labels"><span>${r['52w_low']:.2f}</span><span>${r['52w_high']:.2f}</span></div></div>
    """

        ranges_html = f"{day_block}{y52_block}{bb_bar}{impl_bar}"
        
        indicators_html = f'''<span class="{macd_cls}">MACD: {r['macd_label']}</span><br>
Short: {na(r['short_percent'],"{:.1f}%")} ({na(r['days_to_cover'],"{:.1f}d")})<br>
<span class="{hv_cls}">Volatility: {hv_str}</span><br>
<span class="{opt_dir_cls}">Opt Dir: {r['options_direction']}</span><br>
<span class="{bias_cls}">Bias: {'Down' if r['down_volume_bias'] else 'Up'}</span>'''
        
        html += f'''<tr class="stock-row" data-ticker="{r['ticker']}" data-change="{r['change_pct']}" data-rsi="{r['rsi'] or 50}" data-vol="{r['volume_raw']}" data-meme="{r['is_meme_stock']}" data-squeeze="{r['squeeze_level']}" data-bb-width="{bb_width_val}">
    <td><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{r['ticker']}</a> <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.9em;margin-left:6px">(FZ)</a></td>
<td data-sort="{r['price']:.2f}">${r['price']:.2f} {r['sparkline']}</td>
<td>{fmt_change(r['change_pct'], r['change_abs_day'])}</td>
<td>{fmt_change(r['change_1m'], r['change_abs_1m'])}</td>
<td>{fmt_change(r['change_6m'], r['change_abs_6m'])}</td>
<td>{fmt_change(r['change_ytd'], r['change_abs_ytd'])}</td>
<td data-sort="{r['volume_raw']}">{fmt_vol(r['volume'])}</td>
<td>{ranges_html}</td>
<td>{indicators_html}</td>
<td><span class="{sent_cls}">{r['sentiment']}</span><br><span class="{upside_cls}">Upside: {na(r['upside_potential'],"{:+.1f}%")}</span></td>
</tr>'''
    
    html += "</table></div><div id='cardView'><div class='card-grid'>"
    for _, r in df.iterrows():
        bg = "rgba(0,170,0,0.1)" if r['change_pct'] > 0 else "rgba(204,0,0,0.1)"
        bb_width_val = r['bb_width_pct'] if r['bb_width_pct'] is not None else 100
        hv = r['hv_30_annualized']
        hv_cls = "vol-hot" if hv and hv > 50 else "neutral"
        hv_str = na(hv, '{:.1f}%')

        html += f'''<div class="stock-card stock-row" style="background:{bg}" 
            data-ticker="{r['ticker']}" 
            data-change="{r['change_pct']}" 
            data-rsi="{r['rsi'] or 50}" 
            data-vol="{r['volume_raw']}" 
            data-meme="{r['is_meme_stock']}" 
            data-squeeze="{r['squeeze_level']}" 
            data-bb-width="{bb_width_val}">
    <h2><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{r['ticker']}</a> <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.8em;margin-left:6px">(FZ)</a> ${r['price']:.2f}</h2>
<div style="font-size:1.5em">{fmt_change(r['change_pct'], r['change_abs_day'])}</div>
<div>1M: {fmt_change(r['change_1m'], r['change_abs_1m'])}</div>
<div>6M: {fmt_change(r['change_6m'], r['change_abs_6m'])}</div>
<div>YTD: {fmt_change(r['change_ytd'], r['change_abs_ytd'])}</div>
<div><span class="{hv_cls}">Volatility: {hv_str}</span></div>
<div>BB: {r['bb_status']} ({na(r['bb_width_pct'], '{:.1f}%')})</div>
{r['sparkline']}
</div>'''
    
    html += "</div></div><div id='heatView'><div class='heat-grid'>"
    for _, r in df.iterrows():
        intensity = min(abs(r['change_pct']) / 15, 1)
        bg = f"rgba(0,170,0,{intensity})" if r['change_pct'] >= 0 else f"rgba(204,0,0,{intensity})"
        bb_width_val = r['bb_width_pct'] if r['bb_width_pct'] is not None else 100
        price_display = f"${r['price']:.2f}" if (r.get('price') is not None and pd.notna(r.get('price'))) else 'N/A'
        html += f'''<div class="heat-tile stock-row" style="background:{bg}" 
        onclick="window.open('https://www.barchart.com/stocks/quotes/{r['ticker']}', '_blank')"
        data-ticker="{r['ticker']}" 
        data-change="{r['change_pct']}" 
        data-rsi="{r['rsi'] or 50}" 
        data-vol="{r['volume_raw']}" 
        data-meme="{r['is_meme_stock']}" 
        data-squeeze="{r['squeeze_level']}" 
        data-bb-width="{bb_width_val}">
    <strong><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{r['ticker']}</a> <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.85em;margin-left:6px">(FZ)</a> {price_display}</strong>
    <div style="margin-top:6px">{fmt_change(r['change_pct'], r.get('change_abs_day'))}</div>
    </div>'''
    
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
        const rsi = parseFloat(r.dataset.rsi || 50);
        const vol = parseFloat(r.dataset.vol || 0);
        const meme = r.dataset.meme === 'True';
        const sq = r.dataset.squeeze || 'None';
        const bbw = parseFloat(r.dataset.bbWidth || 100);
        if (currentFilter === 'oversold') show = rsi < 30;
        else if (currentFilter === 'overbought') show = rsi > 70;
        else if (currentFilter === 'surge') show = ch > 10;
        else if (currentFilter === 'crash') show = ch < -10;
        else if (currentFilter === 'meme') show = meme;
        else if (currentFilter === 'volume') show = vol > 5e7;
        else if (currentFilter === 'squeeze') show = sq !== 'None';
        else if (currentFilter === 'bb-squeeze') show = bbw < 6;
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

    with open(file, 'w', encoding='utf-8') as f:
        f.write(html)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', nargs='?', default='data/tickers.csv')
    args = parser.parse_args()

    for ext, file, name in [(False, 'data/reg_dashboard.html', 'Regular'), (True, 'data/extnd_dashboard.html', 'Extended')]:
        try:
            df = dashboard(args.csv_file, ext)
            alerts = check_alerts(df.to_dict('records'))
            html(df, get_vix_data(), get_fear_greed_data(), get_aaii_sentiment(), file, ext, alerts=alerts)
            print(f"✓ {name} Hours Dashboard generated: {file}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")