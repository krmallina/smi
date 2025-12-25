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
        
        ma50 = y['Close'].rolling(50).mean().iloc[-1] if len(y) >= 50 else None
        ma200 = y['Close'].rolling(200).mean().iloc[-1] if len(y) >= 200 else None
        death_cross = ma50 and ma200 and ma50 < ma200
        
        rsi_val = rsi(h30['Close'])
        rsi_lbl = "Oversold" if rsi_val and rsi_val < 30 else "Overbought" if rsi_val and rsi_val > 70 else "Neutral"
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
                        impl_hi, impl_lo = price * (1 + cons/100), price * (1 - cons/100)
                        cvol, pvol = chain.calls['volume'].fillna(0).sum(), chain.puts['volume'].fillna(0).sum()
                        if cvol > 0: pc_ratio = pvol / cvol
            except: pass
        
        down_bias = (h30[h30['Close'] < h30['Open']]['Volume'].sum() > h30[h30['Close'] > h30['Open']]['Volume'].sum()) if len(h30) > 0 else False
        opt_dir = "Neutral"
        if pc_ratio:
            if pc_ratio > 1.2 and death_cross and down_bias: opt_dir = "Strong Bearish"
            elif pc_ratio > 1.0 or death_cross or down_bias: opt_dir = "Bearish"
            elif pc_ratio < 0.8 and not death_cross and not down_bias: opt_dir = "Bullish"
        
        beta, pe = info.get('beta'), info.get('trailingPE')
        rec = info.get('recommendationMean')
        sentiment = ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")[
            0 if rec and rec <= 1.5 else 1 if rec and rec <= 2.5 else 2 if rec and rec <= 3.5 else 3 if rec and rec <= 4.5 else 4
        ] if rec else "N/A"
        
        rating = info.get('recommendationKey', 'none').title().replace('_', ' ')
        target = info.get('targetMeanPrice')
        upside = ((target - price) / price) * 100 if target and price > 0 else None
        spk = sparkline(h30['Close'].tolist() if not h30.empty else [])
        
        time.sleep(1.5)
        tu = ticker.upper()
        return {
            'ticker': tu, 'price': price, 'change_pct': change_pct, 'change_abs_day': change_abs_day,
            'change_1m': ch1m, 'change_abs_1m': abs1m, 'change_6m': ch6m, 'change_abs_6m': abs6m,
            'change_ytd': chytd, 'change_abs_ytd': absytd, 'volume': vol, 'volume_raw': vol,
            '52w_high': high52, '52w_low': low52, 'day_low': day_low, 'day_high': day_high,
            'beta': beta, 'pe': pe, 'short_percent': short_pct, 'days_to_cover': days_cover,
            'squeeze_level': squeeze, 'death_cross': death_cross, 'rsi': rsi_val, 'rsi_label': rsi_lbl,
            'macd': macd_val, 'macd_signal': macd_sig, 'macd_label': macd_lbl, 'volume_spike': vol_spike,
            'hv_30_annualized': hv, 'is_meme_stock': tu in MEME_STOCKS, 'sentiment': sentiment,
            'analyst_rating': rating, 'upside_potential': upside, 'put_call_vol_ratio': pc_ratio,
            'down_volume_bias': down_bias, 'options_direction': opt_dir, 'implied_move_pct': impl_move,
            'implied_high': impl_hi, 'implied_low': impl_lo, 'exp_date_used': exp_date, 'sparkline': spk,
            'links': f"https://finance.yahoo.com/quote/{tu} https://www.barchart.com/stocks/quotes/{tu} https://www.tradingview.com/chart/?symbol={tu} https://finviz.com/quote.ashx?t={tu} https://www.zacks.com/stock/quote/{tu}"
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

def fmt_change(p, a):
    if p is None: return '<span class="neutral" data-sort="0">N/A</span>'
    sign, cls = ("▲", "positive") if p >= 0 else ("▼", "negative")
    abs_h = f' ({a:+.2f})' if a else ''
    return f'<span class="{cls}" data-sort="{p:.10f}">{sign} {p:+.2f}%{abs_h}</span>'

def dashboard(csv='data/tickers.csv', ext=False):
    os.makedirs('data', exist_ok=True)
    try:
        tickers = pd.unique(pd.read_csv(csv).iloc[:,0]).tolist()
    except:
        tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','SPY']
    
    data = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in as_completed([ex.submit(fetch, t, ext) for t in tickers]):
            if r.result(): data.append(r.result())
    return pd.DataFrame(data).sort_values('change_pct', ascending=False)

def get_vix_data():
    try:
        info = yf.Ticker("^VIX").info
        p, pc = info.get('regularMarketPrice') or info.get('previousClose'), info.get('previousClose')
        ch = ((p - pc) / pc) * 100 if p and pc and pc > 0 else 0.0
        return {'price': p, 'change_pct': ch}
    except Exception as e:
        print(f"VIX error: {e}")
        return {'price': None, 'change_pct': None}

def get_fear_greed_data():
    try:
        # Try multiple date formats as CNN API can be finicky
        for days_back in range(0, 3):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            r = requests.get(f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date_str}", 
                            headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.cnn.com/markets/fear-and-greed'}, timeout=10)
            if r.status_code == 200:
                score = float(r.json()['fear_and_greed']['score'])
                s = int(round(score))
                rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4]
                return {'score': score, 'rating': rating}
    except Exception as e:
        print(f"F&G error: {e}")
    return {'score': None, 'rating': "N/A"}

def get_aaii_sentiment():
    try:
        r = requests.get("https://www.aaii.com/sentimentsurvey/sent_results", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        m = re.search(r'\w+\s*\d{1,2}.*?([\d\.]+)%.*?([\d\.]+)%', r.text)
        if m:
            b, be = float(m.group(1)), float(m.group(2))
            return {'bullish': b, 'bearish': be, 'spread': b - be}
    except: pass
    return {'bullish': None, 'bearish': None, 'spread': None}

def html(df, vix, fg, aaii, file, ext=False, alerts=None):
    alerts = alerts or {'grouped': [], 'time': ''}
    hours_mode = "Extended" if ext else "Regular"
    update = datetime.now(UTC).astimezone(PST).strftime('%I:%M:%S %p PST on %B %d, %Y')
    
    banner = ""
    if alerts['grouped']:
        banner = '<div class="alert-banner">🚨 <strong>ALERTS</strong> ' + " | ".join(alerts['grouped']) + '</div>'
    
    vix_h = f'VIX: {vix["price"]:.2f} ({vix["change_pct"]:+.2f}%)' if vix['price'] else 'VIX: N/A'
    fg_h = f'F&G: {fg["score"]:.1f} ({fg["rating"]})' if fg['score'] else 'F&G: N/A'
    aaii_h = f'AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%' if aaii['bullish'] else 'AAII: N/A'
    
    # Comprehensive Enhanced HTML with all features
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Enhanced Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{--bg:#f5f5f5;--card:#fff;--text:#333;--border:#ddd;--accent:#0066cc;--pos:#00aa00;--neg:#cc0000}}
[data-theme="dark"]{{--bg:#1a1a1a;--card:#2d2d2d;--text:#e0e0e0;--border:#444;--accent:#3d8bfd;--pos:#4caf50;--neg:#f44336}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);padding:20px;transition:all .3s}}
.container{{max-width:1600px;margin:0 auto}}
.top-bar{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:15px;background:var(--card);padding:15px 20px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
.btn{{padding:10px 18px;background:var(--accent);color:#fff;border:none;border-radius:8px;cursor:pointer;font-weight:600;transition:all .2s}}
.btn:hover{{transform:translateY(-2px);box-shadow:0 4px 8px rgba(0,0,0,.2)}}
.alert-banner{{background:#ff4444;color:#fff;padding:12px 15px;border-radius:8px;margin-bottom:20px}}
.quick-filters{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;background:var(--card);padding:15px;border-radius:12px}}
.chip{{padding:8px 16px;background:var(--bg);border:2px solid var(--border);border-radius:20px;cursor:pointer;font-size:.85em;font-weight:600;transition:all .2s}}
.chip:hover,.chip.active{{border-color:var(--accent);background:var(--accent);color:#fff}}
.views{{display:flex;gap:10px;margin-bottom:20px}}
.view-btn{{padding:10px 20px;background:var(--card);border:2px solid var(--border);border-radius:8px;cursor:pointer;transition:all .2s}}
.view-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
#tableView{{display:block}}
#cardView,#heatView{{display:none}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
.stock-card{{background:var(--card);border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.1);transition:all .3s}}
.stock-card:hover{{transform:translateY(-4px);box-shadow:0 8px 16px rgba(0,0,0,.2)}}
.stock-card h2 a{{color:var(--text);text-decoration:none}}
.stock-card h2 a:hover{{color:var(--accent)}}
.heat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.heat-tile{{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:8px;padding:10px;cursor:pointer;transition:all .3s}}
.heat-tile:hover{{transform:scale(1.05);box-shadow:0 4px 12px rgba(0,0,0,.3)}}
.heat-tile strong{{font-size:1.2em;margin-bottom:5px}}
table{{width:100%;border-collapse:collapse;background:var(--card);box-shadow:0 2px 8px rgba(0,0,0,.1);margin-top:20px}}
th{{background:var(--accent);color:#fff;padding:14px;cursor:pointer;position:sticky;top:0;z-index:10}}
td{{padding:12px;border-bottom:1px solid var(--border)}}
td a{{color:var(--text);text-decoration:none;font-weight:bold}}
td a:hover{{color:var(--accent)}}
.positive{{color:var(--pos);font-weight:bold}}
.negative{{color:var(--neg);font-weight:bold}}
.sparkline{{margin-left:10px;vertical-align:middle}}
.favorite{{cursor:pointer;font-size:1.2em;margin-right:8px}}
.favorite.active{{color:gold}}
.range-bar{{width:100%;height:8px;background:#e0e0e0;border-radius:4px;position:relative;margin:4px 0}}
.range-bar-fill{{height:100%;border-radius:4px;position:absolute;left:0}}
.range-bar-marker{{position:absolute;width:3px;height:12px;background:#000;top:-2px;border-radius:2px}}
.range-labels{{display:flex;justify-content:space-between;font-size:0.75em;color:var(--text-secondary);margin-top:2px}}
.range-container{{margin:8px 0}}
.range-title{{font-size:0.8em;font-weight:600;margin-bottom:4px;color:var(--text-secondary)}}
.toggle-switch{{position:relative;display:inline-block;width:60px;height:30px;margin:0 10px}}
.toggle-switch input{{opacity:0;width:0;height:0}}
.toggle-slider{{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background-color:#ccc;transition:.4s;border-radius:30px}}
.toggle-slider:before{{position:absolute;content:"";height:22px;width:22px;left:4px;bottom:4px;background-color:white;transition:.4s;border-radius:50%}}
input:checked + .toggle-slider{{background-color:var(--accent)}}
input:checked + .toggle-slider:before{{transform:translateX(30px)}}
.hours-toggle{{display:flex;align-items:center;gap:10px;font-weight:600}}
@media(max-width:768px){{
.top-bar,.quick-filters{{flex-direction:column}}
.card-grid{{grid-template-columns:1fr}}
table{{font-size:.85em}}
td,th{{padding:8px}}
}}
@media print{{
.top-bar,.quick-filters,.views,.btn{{display:none}}
}}
</style></head><body>
<div class="container">
{banner}
<div class="top-bar">
<div><h1>📊 Enhanced Dashboard</h1><small>{update}</small></div>
<div style="display:flex;gap:15px;flex-wrap:wrap;align-items:center">
<span>{vix_h}</span><span>{fg_h}</span><span>{aaii_h}</span>
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
<div class="chip" onclick="quickFilter('all')">All</div>
<div class="chip" onclick="quickFilter('oversold')">📉 Oversold (RSI<30)</div>
<div class="chip" onclick="quickFilter('overbought')">📈 Overbought (RSI>70)</div>
<div class="chip" onclick="quickFilter('surge')">🚀 Surge >10%</div>
<div class="chip" onclick="quickFilter('crash')">💥 Crash <-10%</div>
<div class="chip" onclick="quickFilter('meme')">🎮 Meme Stocks</div>
<div class="chip" onclick="quickFilter('volume')">📊 High Volume</div>
<div class="chip" onclick="quickFilter('squeeze')">🔥 Short Squeeze</div>
</div>

<div class="views">
<button class="view-btn active" onclick="setView('table')">📋 Table</button>
<button class="view-btn" onclick="setView('card')">🗂️ Cards</button>
<button class="view-btn" onclick="setView('heat')">🔥 Heatmap</button>
</div>

<div id="tableView">
<table id="stockTable">
<tr><th onclick="sortTable(0)">⭐ TICKER</th><th onclick="sortTable(1)">PRICE</th><th onclick="sortTable(2)">DAY %</th>
<th onclick="sortTable(3)">1M %</th><th onclick="sortTable(4)">6M %</th><th onclick="sortTable(5)">YTD %</th>
<th onclick="sortTable(6)">VOLUME</th><th onclick="sortTable(7)">RANGES</th><th onclick="sortTable(8)">RSI</th><th onclick="sortTable(9)">SENTIMENT</th></tr>
"""
    
    # Build table rows
    for _, r in df.iterrows():
        fav_id = f"fav-{r['ticker']}"
        rsi_cls = "negative" if r['rsi'] and r['rsi'] < 30 else "positive" if r['rsi'] and r['rsi'] > 70 else ""
        sent_cls = "positive" if "Buy" in r['sentiment'] else "negative" if "Sell" in r['sentiment'] else ""
        
        # Build range bars
        day_range = r['day_high'] - r['day_low']
        pos_day = ((r['price'] - r['day_low']) / day_range * 100) if day_range > 0 else 50
        day_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos_day}%, var(--neg) {pos_day}%, var(--neg) 100%)"
        
        range_52w = r['52w_high'] - r['52w_low']
        pos_52w = ((r['price'] - r['52w_low']) / range_52w * 100) if range_52w > 0 else 50
        w52_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos_52w}%, var(--neg) {pos_52w}%, var(--neg) 100%)"
        
        ranges_html = f"""<div class="range-container">
<div class="range-title">Day Range</div>
<div class="range-bar" style="background:{day_color}">
<div class="range-bar-marker" style="left:{pos_day}%"></div>
</div>
<div class="range-labels"><span>${r['day_low']:.2f}</span><span>${r['day_high']:.2f}</span></div>
</div>
<div class="range-container">
<div class="range-title">52W Range</div>
<div class="range-bar" style="background:{w52_color}">
<div class="range-bar-marker" style="left:{pos_52w}%"></div>
</div>
<div class="range-labels"><span>${r['52w_low']:.2f}</span><span>${r['52w_high']:.2f}</span></div>
</div>"""
        
        barchart_url = f"https://www.barchart.com/stocks/quotes/{r['ticker']}"
        
        html += f"""<tr data-ticker="{r['ticker']}" data-change="{r['change_pct']:.2f}" data-rsi="{r['rsi'] or 0}" data-vol="{r['volume_raw']}" data-meme="{r['is_meme_stock']}" data-squeeze="{r['squeeze_level']}">
<td><span class="favorite" id="{fav_id}" onclick="toggleFav('{r['ticker']}','{fav_id}')">☆</span><a href="{barchart_url}" target="_blank">{r['ticker']}</a></td>
<td>${r['price']:.2f} {r['sparkline']}</td>
<td>{fmt_change(r['change_pct'], r['change_abs_day'])}</td>
<td>{fmt_change(r['change_1m'], r['change_abs_1m'])}</td>
<td>{fmt_change(r['change_6m'], r['change_abs_6m'])}</td>
<td>{fmt_change(r['change_ytd'], r['change_abs_ytd'])}</td>
<td>{fmt_vol(r['volume'])}</td>
<td>{ranges_html}</td>
<td><span class="{rsi_cls}">{na(r['rsi'], '{:.1f}')}</span></td>
<td><span class="{sent_cls}">{r['sentiment']}</span></td>
</tr>"""
    
    html += """</table></div>

<div id="cardView">
<div class="card-grid">
"""
    
    # Build card view
    for _, r in df.iterrows():
        bg_color = "rgba(0,170,0,0.1)" if r['change_pct'] > 0 else "rgba(204,0,0,0.1)"
        barchart_url = f"https://www.barchart.com/stocks/quotes/{r['ticker']}"
        html += f"""<div class="stock-card" style="background:{bg_color}">
<h2><a href="{barchart_url}" target="_blank">{r['ticker']}</a> ${r['price']:.2f}</h2>
<div style="font-size:1.5em;margin:10px 0">{fmt_change(r['change_pct'], r['change_abs_day'])}</div>
<div>Volume: {fmt_vol(r['volume'])}</div>
<div>RSI: {na(r['rsi'], '{:.1f}')}</div>
<div>Sentiment: {r['sentiment']}</div>
{r['sparkline']}
</div>"""
    
    html += """</div></div>

<div id="heatView">
<div class="heat-grid">
"""
    
    # Build heatmap
    for _, r in df.iterrows():
        ch = r['change_pct']
        intensity = min(abs(ch) / 15, 1)
        bg = f"rgba(0,170,0,{intensity})" if ch >= 0 else f"rgba(204,0,0,{intensity})"
        barchart_url = f"https://www.barchart.com/stocks/quotes/{r['ticker']}"
        html += f"""<div class="heat-tile" style="background:{bg}" onclick="window.open('{barchart_url}', '_blank')">
<strong>{r['ticker']}</strong>
<div style="font-size:1.2em">{ch:+.1f}%</div>
</div>"""
    
    html += """</div></div>

</div>

<script>
// LocalStorage for user preferences
const STORAGE_KEY = 'dashboard_prefs';

function loadPrefs() {
    const prefs = localStorage.getItem(STORAGE_KEY);
    return prefs ? JSON.parse(prefs) : {theme: 'light', favorites: [], view: 'table'};
}

function savePrefs(prefs) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
}

// Initialize
let prefs = loadPrefs();
document.documentElement.setAttribute('data-theme', prefs.theme);

// Load favorites
prefs.favorites.forEach(ticker => {
    const elem = document.getElementById('fav-' + ticker);
    if (elem) {
        elem.textContent = '★';
        elem.classList.add('active');
    }
});

// Theme toggle
function toggleTheme() {
    prefs.theme = prefs.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', prefs.theme);
    savePrefs(prefs);
}

// Favorite toggle
function toggleFav(ticker, elemId) {
    const elem = document.getElementById(elemId);
    const idx = prefs.favorites.indexOf(ticker);
    if (idx > -1) {
        prefs.favorites.splice(idx, 1);
        elem.textContent = '☆';
        elem.classList.remove('active');
    } else {
        prefs.favorites.push(ticker);
        elem.textContent = '★';
        elem.classList.add('active');
    }
    savePrefs(prefs);
}

// View switching
function setView(view) {
    document.getElementById('tableView').style.display = view === 'table' ? 'block' : 'none';
    document.getElementById('cardView').style.display = view === 'card' ? 'block' : 'none';
    document.getElementById('heatView').style.display = view === 'heat' ? 'block' : 'none';
    
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    prefs.view = view;
    savePrefs(prefs);
}

// Hours toggle
function toggleHours(extended) {
    const currentFile = window.location.pathname.split('/').pop();
    const newFile = extended ? 'extndED_dashboard.html' : 'regED_dashboard.html';
    if (currentFile !== newFile) {
        window.location.href = newFile;
    }
}

// Quick filters
function quickFilter(type) {
    const rows = document.querySelectorAll('#stockTable tr:not(:first-child)');
    
    rows.forEach(row => {
        let show = true;
        const change = parseFloat(row.dataset.change);
        const rsi = parseFloat(row.dataset.rsi);
        const vol = parseFloat(row.dataset.vol);
        const meme = row.dataset.meme === 'True';
        const squeeze = row.dataset.squeeze;
        
        if (type === 'oversold') show = rsi < 30;
        else if (type === 'overbought') show = rsi > 70;
        else if (type === 'surge') show = change > 10;
        else if (type === 'crash') show = change < -10;
        else if (type === 'meme') show = meme;
        else if (type === 'volume') show = vol > 1000000;
        else if (type === 'squeeze') show = squeeze !== 'None';
        else show = true;
        
        row.style.display = show ? '' : 'none';
    });
    
    // Update chip active state
    document.querySelectorAll('.chip').forEach(chip => chip.classList.remove('active'));
    event.target.classList.add('active');
}

// Table sorting
let sortDir = {};
function sortTable(col) {
    const table = document.getElementById('stockTable');
    const rows = Array.from(table.querySelectorAll('tr:not(:first-child)'));
    
    sortDir[col] = !sortDir[col];
    const dir = sortDir[col] ? 1 : -1;
    
    rows.sort((a, b) => {
        let aVal = a.cells[col].textContent.trim();
        let bVal = b.cells[col].textContent.trim();
        
        // Extract numbers from formatted text
        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return (aNum - bNum) * dir;
        }
        return aVal.localeCompare(bVal) * dir;
    });
    
    rows.forEach(row => table.appendChild(row));
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        if (e.key === 'r') {
            e.preventDefault();
            location.reload();
        } else if (e.key === 'd') {
            e.preventDefault();
            toggleTheme();
        }
    }
});

// Auto-refresh countdown (optional)
let countdown = 300; // 5 minutes
setInterval(() => {
    countdown--;
    if (countdown <= 0) {
        location.reload();
    }
}, 1000);

// Restore view on load
if (prefs.view !== 'table') {
    setTimeout(() => {
        const viewBtns = document.querySelectorAll('.view-btn');
        viewBtns.forEach((btn, idx) => {
            if ((prefs.view === 'card' && idx === 1) || (prefs.view === 'heat' && idx === 2)) {
                btn.click();
            }
        });
    }, 100);
}

// Search functionality
function initSearch() {
    const searchBar = document.createElement('input');
    searchBar.type = 'text';
    searchBar.placeholder = '🔍 Search tickers...';
    searchBar.style.cssText = 'padding:10px;border-radius:8px;border:2px solid var(--border);width:250px;margin-right:15px';
    searchBar.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        document.querySelectorAll('#stockTable tr:not(:first-child)').forEach(row => {
            const ticker = row.dataset.ticker.toLowerCase();
            row.style.display = ticker.includes(term) ? '' : 'none';
        });
    });
    document.querySelector('.top-bar > div:last-child').prepend(searchBar);
}
initSearch();

// Add tooltips
document.querySelectorAll('td').forEach(td => {
    td.title = td.textContent.trim();
});

console.log('Enhanced Dashboard loaded. Keyboard shortcuts: Ctrl+R (refresh), Ctrl+D (theme)');
</script>
</body></html>"""
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(html)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', nargs='?', default='data/tickers.csv')
    args = parser.parse_args()

    for ext, file, name in [(False, 'data/regED_dashboard.html', 'Regular'), (True, 'data/extndED_dashboard.html', 'Extended')]:
        try:
            df = dashboard(args.csv_file, ext)
            alerts = check_alerts(df.to_dict('records'))
            html(df, get_vix_data(), get_fear_greed_data(), get_aaii_sentiment(), file, ext, alerts=alerts)
            print(f"✓ {name} Hours Dashboard → {file}")
        except Exception as e:
            print(f"✗ {name} Hours failed: {e}")