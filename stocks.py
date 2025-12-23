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

# Config
ALERTS_FILE = 'data/alerts.json'
UTC = pytz.utc
PST = pytz.timezone('America/Los_Angeles')
MEME_STOCKS = {'GME','AMC','BB','KOSS','EXPR','DJT','HOOD','RDDT','SPCE','RIVN','DNUT','OPEN','KSS','RKLB','GPRO','AEO','BYND','CVNA','PLTR','SMCI'}

def load_alerts():
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except:
        return []

def check_alerts(data):
    custom_alerts = load_alerts()
    now = datetime.now(UTC).astimezone(PST)

    high_52w = []
    low_52w = []
    surge = []
    crash = []
    custom = []

    # Custom user alerts
    for a in custom_alerts:
        s = next((x for x in data if x['ticker'] == a['ticker'].upper()), None)
        if not s: continue
        msg = ""
        if a['condition'] == "price_above" and a.get('value') and s['price'] > a['value']:
            msg = f"price ABOVE ${a['value']:.2f} → ${s['price']:.2f}"
        elif a['condition'] == "price_below" and a.get('value') and s['price'] < a['value']:
            msg = f"price BELOW ${a['value']:.2f} → ${s['price']:.2f}"
        elif a['condition'] == "day_change_above" and a.get('value') and s['change_pct'] > a['value']:
            msg = f"DAY % ABOVE {a['value']}% → {s['change_pct']:+.2f}%"
        elif a['condition'] == "day_change_below" and a.get('value') and s['change_pct'] < a['value']:
            msg = f"DAY % BELOW {a['value']}% → {s['change_pct']:+.2f}%"
        elif a['condition'] == "rsi_oversold" and s['rsi'] is not None and s['rsi'] < 30:
            msg = f"RSI OVERSOLD → {s['rsi']:.1f}"
        elif a['condition'] == "rsi_overbought" and s['rsi'] is not None and s['rsi'] > 70:
            msg = f"RSI OVERBOUGHT → {s['rsi']:.1f}"
        elif a['condition'] == "volume_spike" and s['volume_spike']:
            msg = "VOLUME SPIKE"
        if msg:
            custom.append({'ticker': s['ticker'], 'msg': msg})

    # Built-in alerts
    for s in data:
        ch = s['change_pct']
        if ch > 15:
            surge.append({'ticker': s['ticker'], 'msg': f"SURGED > +15% → {ch:+.2f}%"})
        elif ch < -15:
            crash.append({'ticker': s['ticker'], 'msg': f"CRASHED < -15% → {ch:+.2f}%"})

        if s['52w_high'] > s['52w_low']:
            pos_pct = (s['price'] - s['52w_low']) / (s['52w_high'] - s['52w_low']) * 100
            if pos_pct >= 95:
                high_52w.append({'ticker': s['ticker'], 'msg': f"NEAR 52W HIGH ({pos_pct:.1f}%) @ ${s['price']:.2f} (High: ${s['52w_high']:.2f})"})
            elif pos_pct <= 5:
                low_52w.append({'ticker': s['ticker'], 'msg': f"NEAR 52W LOW ({pos_pct:.1f}%) @ ${s['price']:.2f} (Low: ${s['52w_low']:.2f})"})

    # Build grouped display lines
    grouped = []
    if high_52w:
        tickers = ", ".join(a['ticker'] for a in high_52w)
        details = "<br>".join(f"{a['ticker']}: {a['msg']}" for a in high_52w)
        grouped.append(f"🔥 <strong>Near 52W High:</strong> {tickers}<div class='alert-tooltip'>{details}</div>")
    if low_52w:
        tickers = ", ".join(a['ticker'] for a in low_52w)
        details = "<br>".join(f"{a['ticker']}: {a['msg']}" for a in low_52w)
        grouped.append(f"🔔 <strong>Near 52W Low:</strong> {tickers}<div class='alert-tooltip'>{details}</div>")
    if surge:
        tickers = ", ".join(a['ticker'] for a in surge)
        details = "<br>".join(f"{a['ticker']}: {a['msg']}" for a in surge)
        grouped.append(f"🚀 <strong>Surge >15%:</strong> {tickers}<div class='alert-tooltip'>{details}</div>")
    if crash:
        tickers = ", ".join(a['ticker'] for a in crash)
        details = "<br>".join(f"{a['ticker']}: {a['msg']}" for a in crash)
        grouped.append(f"💥 <strong>Crash <-15%:</strong> {tickers}<div class='alert-tooltip'>{details}</div>")
    if custom:
        details = "<br>".join(f"{a['ticker']}: {a['msg']}" for a in custom)
        grouped.append(f"⚡ <strong>Custom Alerts:</strong> {len(custom)} triggered<div class='alert-tooltip'>{details}</div>")

    return {'grouped': grouped, 'time': now.strftime('%I:%M %p PST')}

def rsi(series):
    if len(series) < 15: return None
    d = series.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l
    return (100 - 100 / (1 + rs)).iloc[-1]

def macd(series):
    if len(series) < 26: return None, None, "N/A"
    e12 = series.ewm(span=12, adjust=False).mean()
    e26 = series.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    label = "Bullish" if line.iloc[-1] > sig.iloc[-1] else "Bearish"
    return line.iloc[-1], sig.iloc[-1], label

def na(val, fmt="{:.2f}"):
    return "N/A" if val is None or pd.isna(val) else fmt.format(val)

def fetch(ticker, ext=False):
    try:
        t = yf.Ticker(ticker)
        info = t.info

        h = t.history(period="2d", interval="1m", prepost=ext)
        if h.empty: h = t.history(period="5d", prepost=ext)
        if h.empty: return None
        price = h['Close'].iloc[-1]

        reg = t.history(period="5d", prepost=False)
        change_pct = 0.0
        change_abs_day = 0.0
        if len(reg) >= 2:
            prev = reg['Close'].iloc[-2]
            if prev and prev > 0:
                change_pct = ((price - prev) / prev) * 100
                change_abs_day = price - prev

        change_1m = change_6m = None
        change_abs_1m = change_abs_6m = None
        for d in [35, 190]:
            hist = t.history(start=(datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d'))
            if len(hist) >= 2:
                start = hist['Close'].iloc[0]
                if start and start > 0:
                    pct = ((price - start) / start) * 100
                    abs_change = price - start
                    if d == 35:
                        change_1m = pct
                        change_abs_1m = abs_change
                    else:
                        change_6m = pct
                        change_abs_6m = abs_change

        ytd_start = datetime(datetime.now().year, 1, 1)
        ytd_hist = t.history(start=ytd_start.strftime('%Y-%m-%d'))
        ytd_pct = ytd_abs = None
        if len(ytd_hist) >= 2:
            ytd_start_price = ytd_hist['Close'].iloc[0]
            if ytd_start_price and ytd_start_price > 0:
                ytd_pct = ((price - ytd_start_price) / ytd_start_price) * 100
                ytd_abs = price - ytd_start_price

        y = t.history(period="1y")
        high52 = y['High'].max() if not y.empty else price
        low52 = y['Low'].min() if not y.empty else price

        vol = t.history(period="1d", interval="1m", prepost=ext)['Volume'].sum()

        hv = None
        h30 = t.history(period="60d")
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
        if short_pct is not None and days_cover is not None:
            if short_pct > 30 and days_cover > 10: squeeze = "Extreme"
            elif short_pct > 20 and days_cover > 7: squeeze = "High"
            elif short_pct > 15 and days_cover > 5: squeeze = "Moderate"

        y1 = t.history(period="1y")
        ma50 = y1['Close'].rolling(50).mean().iloc[-1] if len(y1) >= 50 else None
        ma200 = y1['Close'].rolling(200).mean().iloc[-1] if len(y1) >= 200 else None
        death_cross = ma50 is not None and ma200 is not None and ma50 < ma200

        rsi_val = rsi(t.history(period="60d")['Close'])
        rsi_lbl = "Oversold" if rsi_val and rsi_val < 30 else "Overbought" if rsi_val and rsi_val > 70 else "Neutral"

        macd_val, macd_sig, macd_lbl = macd(t.history(period="100d")['Close'])

        v30 = t.history(period="30d")
        vol_spike = False
        if len(v30) > 1:
            avg = v30['Volume'][:-1].mean()
            if avg and avg > 0: vol_spike = vol > 1.5 * avg

        put_call_vol_ratio = None
        implied_move_pct = None
        implied_high = None
        implied_low = None
        exp_date_used = None

        if t.options:
            nearest_exp = t.options[0]
            exp_date_used = nearest_exp
            try:
                chain = t.option_chain(nearest_exp)
                calls = chain.calls
                puts = chain.puts

                all_strikes = pd.concat([calls['strike'], puts['strike']]).unique()
                if len(all_strikes) > 0:
                    atm_strike = min(all_strikes, key=lambda s: abs(s - price))

                    call_price = calls[calls['strike'] == atm_strike]['lastPrice'].iloc[0] if not calls[calls['strike'] == atm_strike].empty else 0
                    put_price = puts[puts['strike'] == atm_strike]['lastPrice'].iloc[0] if not puts[puts['strike'] == atm_strike].empty else 0

                    straddle_price = call_price + put_price
                    if straddle_price > 0:
                        implied_move_pct = (straddle_price / price) * 100
                        conservative_pct = implied_move_pct * 0.85
                        implied_high = price * (1 + conservative_pct / 100)
                        implied_low = price * (1 - conservative_pct / 100)

                        call_vol = calls['volume'].fillna(0).sum()
                        put_vol = puts['volume'].fillna(0).sum()
                        if call_vol > 0:
                            put_call_vol_ratio = put_vol / call_vol
            except:
                pass

        down_days = h30[h30['Close'] < h30['Open']]
        up_days = h30[h30['Close'] > h30['Open']]
        down_volume_bias = down_days['Volume'].sum() > up_days['Volume'].sum()

        options_direction = "Neutral"
        if put_call_vol_ratio is not None:
            if put_call_vol_ratio > 1.2 and death_cross and down_volume_bias:
                options_direction = "Strong Bearish"
            elif put_call_vol_ratio > 1.0 or death_cross or down_volume_bias:
                options_direction = "Bearish"
            elif put_call_vol_ratio < 0.8 and not death_cross and not down_volume_bias:
                options_direction = "Bullish"

        beta = info.get('beta')
        pe = info.get('trailingPE')
        sentiment = "N/A"
        rec = info.get('recommendationMean')
        if rec is not None:
            sentiment = "Strong Buy" if rec <= 1.5 else "Buy" if rec <= 2.5 else "Hold" if rec <= 3.5 else "Sell" if rec <= 4.5 else "Strong Sell"

        analyst_rating = info.get('recommendationKey', 'none').title().replace('_', ' ')

        upside_potential = None
        target = info.get('targetMeanPrice')
        if target and price > 0: upside_potential = ((target - price) / price) * 100

        time.sleep(1.5)

        ticker_u = ticker.upper()
        return {
            'ticker': ticker_u,
            'links': f"https://finance.yahoo.com/quote/{ticker_u} https://www.barchart.com/stocks/quotes/{ticker_u} https://www.tradingview.com/chart/?symbol={ticker_u} https://finviz.com/quote.ashx?t={ticker_u} https://www.zacks.com/stock/quote/{ticker_u}",
            'price': price,
            'change_pct': change_pct,
            'change_abs_day': change_abs_day,
            'change_1m_pct': change_1m,
            'change_abs_1m': change_abs_1m,
            'change_6m_pct': change_6m,
            'change_abs_6m': change_abs_6m,
            'ytd_pct': ytd_pct,
            'ytd_abs': ytd_abs,
            'volume': vol,
            'volume_raw': vol,
            '52w_high': high52,
            '52w_low': low52,
            'beta': beta,
            'pe': pe,
            'short_percent': short_pct,
            'days_to_cover': days_cover,
            'squeeze_level': squeeze,
            'death_cross': death_cross,
            'rsi': rsi_val,
            'rsi_label': rsi_lbl,
            'macd': macd_val,
            'macd_signal': macd_sig,
            'macd_label': macd_lbl,
            'volume_spike': vol_spike,
            'hv_30_annualized': hv,
            'is_meme_stock': ticker_u in MEME_STOCKS,
            'sentiment': sentiment,
            'analyst_rating': analyst_rating,
            'upside_potential': upside_potential,
            'put_call_vol_ratio': put_call_vol_ratio,
            'down_volume_bias': down_volume_bias,
            'options_direction': options_direction,
            'implied_move_pct': implied_move_pct,
            'implied_high': implied_high,
            'implied_low': implied_low,
            'exp_date_used': exp_date_used
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

def fmt_change(p, abs_change):
    if p is None: return '<span class="neutral" data-sort="0">N/A</span>'
    sign = "▲" if p >= 0 else "▼"
    cls = "positive" if p >= 0 else "negative"
    abs_h = f' ({abs_change:+.2f})' if abs_change is not None else ''
    return f'<span class="{cls}" data-sort="{p:.10f}">{sign} {p:+.2f}%{abs_h}</span>'

def dashboard(csv='data/tickers.csv', ext=False):
    os.makedirs('data', exist_ok=True)
    try:
        tickers = pd.unique(pd.read_csv(csv).iloc[:,0]).tolist()
    except:
        tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','SPY']

    data = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(fetch, t, ext) for t in tickers]
        for r in as_completed(futures):
            res = r.result()
            if res: data.append(res)

    df = pd.DataFrame(data).sort_values('change_pct', ascending=False)
    return df

def get_vix_data():
    try:
        t = yf.Ticker("^VIX")
        info = t.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        prev_close = info.get('previousClose')
        if price and prev_close and prev_close > 0:
            ch_pct = ((price - prev_close) / prev_close) * 100
        else:
            ch_pct = 0.0
        return {'price': price, 'change_pct': ch_pct}
    except Exception as e:
        print(f"VIX fetch error: {e}")
        return {'price': None, 'change_pct': None}

def get_fear_greed_data():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Referer': 'https://www.cnn.com/markets/fear-and-greed'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        fg = data['fear_and_greed']
        score = float(fg['score'])
        s = int(round(score))
        rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
            0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
        ]
        return {'score': score, 'rating': rating}
    except Exception as e:
        print(f"Error fetching Fear & Greed: {e}")
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
    hours = "Extended Hours" if ext else "Regular Hours"
    update = datetime.now(UTC).astimezone(PST).strftime('%I:%M:%S %p PST on %B %d, %Y')

    banner = ""
    if alerts['grouped']:
        banner = f'<div class="alert-banner"><strong>🚨 ALERTS @ {alerts["time"]} 🚨</strong><div class="alert-list">'
        for g in alerts['grouped']:
            banner += f"<div class='alert-item'>{g}</div>"
        banner += "</div></div>"

    # New: Compact meme stock banner
    meme_tickers = sorted(MEME_STOCKS)
    meme_banner = f'<div class="meme-banner">🚀 <strong>Meme Stocks:</strong> {", ".join(meme_tickers)}</div>'

    vix_h = '<span class="neutral">VIX: N/A</span>'
    if vix['price'] is not None:
        cls = "positive" if vix['change_pct'] >= 0 else "negative"
        vix_h = f'<span class="{cls}">VIX: {vix["price"]:.2f} ({vix["change_pct"]:+.2f}%)</span>'

    fg_h = '<span class="neutral">F&G: N/A</span>'
    if fg['score'] is not None:
        cls = "negative" if fg['score'] <= 24 else "high-risk" if fg['score'] <= 44 else "neutral" if fg['score'] <= 55 else "bullish" if fg['score'] <= 74 else "positive"
        fg_h = f'<span class="{cls}">F&G: {fg["score"]:.1f} ({fg["rating"]})</span>'

    aaii_h = '<span class="neutral">AAII: N/A</span>'
    if aaii['bullish'] is not None:
        cls = "positive" if aaii['spread'] > 20 else "bullish" if aaii['spread'] > 0 else "neutral" if aaii['spread'] > -20 else "high-risk" if aaii['spread'] > -40 else "negative"
        aaii_h = f'<span class="{cls}">AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%</span>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Live Dashboard</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
body{{font-family:Arial;margin:20px;background:#f5f5f5}}
.header-container{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;margin-bottom:20px}}
.indicators{{display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
.refresh-btn,.switch-btn{{padding:10px 20px;background:#ff6600;color:white;border:none;border-radius:6px;cursor:pointer;font-weight:bold;text-decoration:none;box-shadow:0 2px 4px rgba(0,0,0,0.2)}}
.refresh-btn{{background:#0066cc}}
.refresh-btn:hover{{background:#0055aa}}
.switch-btn:hover{{background:#e65c00}}
.mode-badge{{padding:10px 20px;border-radius:6px;font-weight:bold;font-size:1em}}
.regular-hours{{background:#0066cc;color:white}}.extended-hours{{background:#ff6600;color:white}}
.alert-banner{{background:#ff4444;color:white;padding:12px 15px;border-radius:8px;margin-bottom:20px;box-shadow:0 4px 8px rgba(0,0,0,0.2);font-size:1.05em}}
.meme-banner{{background:#ffcc00;color:#333;padding:8px 15px;border-radius:6px;margin-bottom:20px;font-size:0.95em;box-shadow:0 2px 6px rgba(0,0,0,0.15)}}
.alert-list{{margin-top:8px}}
.alert-item{{margin:6px 0;position:relative;cursor:help}}
.alert-tooltip{{visibility:hidden;position:absolute;top:100%;left:0;background:#333;color:white;padding:10px;border-radius:6px;font-size:0.9em;width:400px;max-height:300px;overflow-y:auto;z-index:10;opacity:0;transition:opacity 0.3s;box-shadow:0 4px 8px rgba(0,0,0,0.3)}}
.alert-item:hover .alert-tooltip{{visibility:visible;opacity:1}}
table{{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 8px rgba(0,0,0,0.1);margin-top:30px}}
th{{background:#0066cc;color:white;padding:14px;cursor:pointer;position:relative}}
th::after{{content:"⇅";opacity:0.5;margin-left:8px}}
td{{padding:12px 10px;border-bottom:1px solid #ddd;vertical-align:top}}
.positive{{background:#00aa00;color:white;padding:4px 8px;border-radius:3px;font-weight:bold;display:inline-block}}
.negative{{background:#cc0000;color:white;padding:4px 8px;border-radius:3px;font-weight:bold;display:inline-block}}
.bullish{{color:#00aa00;font-weight:bold}}.bearish{{color:#cc0000;font-weight:bold}}
.high-risk{{color:#cc6600;font-weight:bold}}.neutral{{color:#666;font-style:italic}}
.ticker{{font-weight:bold;font-size:1.1em}}.link-group{{display:flex;gap:8px;flex-wrap:wrap;font-size:0.8em}}
.item, .risk-item{{font-size:0.8em;margin:2px 0;position:relative;cursor:help;line-height:1.3}}
.tooltip{{visibility:hidden;position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:#333;color:white;padding:6px 8px;border-radius:4px;white-space:nowrap;font-size:0.75em;z-index:10;opacity:0;transition:opacity 0.3s}}
.item:hover .tooltip, .risk-item:hover .tooltip{{visibility:visible;opacity:1}}
.risk-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px 8px}}
</style></head><body>
{banner}
{meme_banner}
<div class="header-container">
    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
        <h1 style="margin:0">Live Dashboard</h1>
        <span class="mode-badge {'extended-hours' if ext else 'regular-hours'}">{hours}</span>
        <a href="{os.path.basename("data/extnd_dashboard.html" if not ext else "data/reg_dashboard.html")}" class="switch-btn">Switch Hours</a>
    </div>
    <div class="indicators">
        {vix_h}
        {fg_h}
        {aaii_h}
    </div>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
</div>
<div style="margin-bottom:30px"><strong>Last updated:</strong> {update}</div>
<table id="stockTable">
<tr>
    <th>TICKER</th><th>PRICE</th><th>DAY %</th><th>1M %</th><th>6M %</th><th>YTD %</th><th>VOLUME</th><th>52W RANGE</th><th>TECHNICAL</th><th>RISK / SENTIMENT</th>
</tr>"""
    # Table rows unchanged (same as previous version)
    for _, r in df.iterrows():
        pos = ((r['price'] - r['52w_low']) / (r['52w_high'] - r['52w_low'])) * 100 if r['52w_high'] > r['52w_low'] else 50

        tech = []
        if r['rsi'] is not None:
            rc = "oversold" if r['rsi'] < 30 else "overbought" if r['rsi'] > 70 else ""
            tech.append(f'<div class="item"><strong>RSI:</strong> <span class="{rc}">{r["rsi"]:.1f} ({r["rsi_label"]})</span><span class="tooltip">Relative Strength Index (14-day). <30 = Oversold, >70 = Overbought</span></div>')
        if r['macd'] is not None:
            mc = "bullish" if r['macd_label'] == "Bullish" else "bearish"
            tech.append(f'<div class="item"><strong>MACD:</strong> <span class="{mc}">{r["macd"]:+.3f} | {r["macd_signal"]:+.3f} ({r["macd_label"]})</span><span class="tooltip">Moving Average Convergence Divergence. Bullish when MACD > Signal</span></div>')
        tech.append(f'<div class="item"><strong>Vol Spike:</strong> {"Yes" if r["volume_spike"] else "No"}<span class="tooltip">Today\'s volume > 1.5x 30-day average</span></div>')
        tech_h = "<br>".join(tech) or '<span class="neutral">N/A</span>'

        risk = []
        vol_sc = 0.0
        if r['hv_30_annualized'] is not None and pd.notna(r['hv_30_annualized']): vol_sc += min(r['hv_30_annualized']/100*50, 50)
        if r['beta'] is not None and pd.notna(r['beta']): vol_sc += min(max(r['beta']-0.5,0)*33.33, 50)
        vol_sc = int(round(vol_sc))
        v_cls = "negative" if vol_sc >= 81 else "high-risk" if vol_sc >= 61 else "bearish" if vol_sc >= 31 else "neutral"
        v_emoji = "🔥🔥" if vol_sc >= 81 else "🔥" if vol_sc >= 61 else ""
        risk.append(f'<div class="risk-item"><strong>Vol Score:</strong> <span class="{v_cls}">{v_emoji} {vol_sc}/100</span><span class="tooltip">Volatility score based on HV and Beta</span></div>')
        risk.append(f'<div class="risk-item"><strong>Beta:</strong> {na(r["beta"])}<span class="tooltip">Stock volatility vs market</span></div>')
        risk.append(f'<div class="risk-item"><strong>P/E:</strong> {na(r["pe"])}<span class="tooltip">Price to Earnings ratio</span></div>')
        risk.append(f'<div class="risk-item"><strong>Short %:</strong> {na(r["short_percent"], "{:.1f}%")}<span class="tooltip">Percentage of float sold short</span></div>')
        risk.append(f'<div class="risk-item"><strong>Days to Cover:</strong> {na(r["days_to_cover"], "{:.1f}")}<span class="tooltip">Days to cover short interest</span></div>')
        trend_val = '<span class="negative">Death Cross</span>' if r["death_cross"] else 'No'
        risk.append(f'<div class="risk-item"><strong>Trend:</strong> {trend_val}<span class="tooltip">50-day MA below 200-day MA = bearish</span></div>')
        risk.append(f'<div class="risk-item"><strong>P/C Vol Ratio:</strong> {na(r["put_call_vol_ratio"], "{:.2f}")}<span class="tooltip">Put/Call volume ratio (higher = bearish)</span></div>')
        vol_bias_val = '<span class="negative">Down Days Higher</span>' if r["down_volume_bias"] else 'No'
        risk.append(f'<div class="risk-item"><strong>Volume Bias:</strong> {vol_bias_val}<span class="tooltip">Higher volume on down days = bearish</span></div>')
        opt_dir_cls = "bearish" if "Bearish" in r["options_direction"] else "bullish" if "Bullish" in r["options_direction"] else "neutral"
        risk.append(f'<div class="risk-item"><strong>Options Direction:</strong> <span class="{opt_dir_cls}">{r["options_direction"]}</span><span class="tooltip">Options flow sentiment</span></div>')

        if r['implied_move_pct'] is not None:
            move_cls = "high-risk" if r['implied_move_pct'] > 10 else "bearish" if r['implied_move_pct'] > 5 else "neutral"
            range_cls = "high-risk" if (r['implied_high'] - r['implied_low']) / r['price'] * 100 > 10 else "bearish" if (r['implied_high'] - r['implied_low']) / r['price'] * 100 > 5 else "neutral"
            risk.append(f'<div class="risk-item"><strong>Implied Move ({r["exp_date_used"] or "N/A"}):</strong> <span class="{move_cls}">±{r["implied_move_pct"]:.1f}%</span><span class="tooltip">Expected price move from options</span></div>')
            risk.append(f'<div class="risk-item"><strong>Expected Range:</strong> <span class="{range_cls}">${r["implied_low"]:.2f} – ${r["implied_high"]:.2f}</span><span class="tooltip">Likely price range</span></div>')
        else:
            risk.append('<div class="risk-item"><strong>Implied Move:</strong> N/A<span class="tooltip">No options data</span></div>')
            risk.append('<div class="risk-item"><strong>Expected Range:</strong> N/A<span class="tooltip">No options data</span></div>')

        sent_cls = "positive" if "Buy" in r["sentiment"] else "negative" if "Sell" in r["sentiment"] else "neutral"
        risk.append(f'<div class="risk-item"><strong>Sentiment:</strong> <span class="{sent_cls}">{r["sentiment"]}</span><span class="tooltip">Analyst sentiment</span></div>')

        rating_cls = "positive" if "buy" in r["analyst_rating"].lower() else "negative" if "sell" in r["analyst_rating"].lower() else "neutral"
        risk.append(f'<div class="risk-item"><strong>Rating:</strong> <span class="{rating_cls}">{r["analyst_rating"]}</span><span class="tooltip">Analyst rating</span></div>')

        if r['upside_potential'] is not None:
            up_cls = "positive" if r['upside_potential'] > 0 else "negative"
            risk.append(f'<div class="risk-item"><strong>Upside:</strong> <span class="{up_cls}">{r["upside_potential"]:+.1f}%</span><span class="tooltip">Upside to analyst target</span></div>')

        meme_val = '<span class="high-risk">🚀 Yes</span>' if r['is_meme_stock'] else 'No'
        risk.append(f'<div class="risk-item"><strong>Meme:</strong> {meme_val}<span class="tooltip">High retail/meme interest</span></div>')
        risk.append(f'<div class="risk-item"><strong>Short Squeeze:</strong> {r["squeeze_level"]}<span class="tooltip">Short squeeze risk level</span></div>')

        risk_h = f'<div class="risk-grid">{ "".join(risk)}</div>'

        link_urls = r['links'].split()
        link_labels = ["YF", "BC", "TV", "FZ", "Z"]
        links = " ".join([f'<a href="{url}" target="_blank">{lbl}</a>' for url, lbl in zip(link_urls[1:], link_labels[1:])])

        vol_display = fmt_vol(r['volume'])
        vol_sort = r['volume_raw'] if r['volume_raw'] is not None else 0

        html += f"""<tr>
    <td><div class="ticker"><a href="{link_urls[0]}" target="_blank">{r['ticker']}</a></div><div class="link-group">{links}</div></td>
    <td>{r['price']:.2f}</td>
    <td>{fmt_change(r['change_pct'], r['change_abs_day'])}</td>
    <td>{fmt_change(r['change_1m_pct'], r['change_abs_1m'])}</td>
    <td>{fmt_change(r['change_6m_pct'], r['change_abs_6m'])}</td>
    <td>{fmt_change(r['ytd_pct'], r['ytd_abs'])}</td>
    <td data-sort="{vol_sort}">{vol_display}</td>
    <td><div style="width:180px;position:relative;margin:8px 0">
        <div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;position:relative">
            <div style="width:{pos:.1f}%;height:100%;background:#00aa00;position:absolute;left:0;top:0"></div>
            <div style="width:{100-pos:.1f}%;height:100%;background:#cc0000;position:absolute;right:0;top:0"></div>
            <div style="position:absolute;left:{pos:.1f}%;top:-4px;width:3px;height:16px;background:#000;z-index:1"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.8em">
            <span>{r['52w_low']:.2f}</span><span>{r['52w_high']:.2f}</span>
        </div>
    </div></td>
    <td>{tech_h}</td>
    <td>{risk_h}</td>
</tr>"""
    html += """</table>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const headers = document.querySelectorAll('th');
    headers.forEach(th => th.addEventListener('click', () => {
        const table = th.closest('table');
        const rows = Array.from(table.querySelectorAll('tr')).slice(1);
        const index = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.classList.toggle('asc');
        th.classList.toggle('desc', !asc);
        rows.sort((a, b) => {
            let av = a.cells[index].querySelector('[data-sort]') ? parseFloat(a.cells[index].querySelector('[data-sort]').dataset.sort) : a.cells[index].innerText.trim();
            let bv = b.cells[index].querySelector('[data-sort]') ? parseFloat(b.cells[index].querySelector('[data-sort]').dataset.sort) : b.cells[index].innerText.trim();
            av = isNaN(av) ? av : av;
            bv = isNaN(bv) ? bv : bv;
            return (av > bv ? 1 : av < bv ? -1 : 0) * (asc ? 1 : -1);
        });
        rows.forEach(row => table.appendChild(row));
    }));
});
</script>
</body></html>"""

    with open(file, 'w', encoding='utf-8') as f: f.write(html)

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
            print(f"✓ {name} Hours Dashboard → {file}")
        except Exception as e:
            print(f"✗ {name} Hours failed: {e}")

    print("\nDashboards generated successfully!")