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
    alerts = load_alerts()
    triggered = []
    now = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    for a in alerts:
        s = next((x for x in data if x['ticker'] == a['ticker'].upper()), None)
        if not s: continue
        msg = ""
        if a['condition'] == "price_above" and a.get('value') and s['price'] > a['value']:
            msg = f"{s['ticker']} price ABOVE ${a['value']:.2f} → ${s['price']:.2f}"
        elif a['condition'] == "price_below" and a.get('value') and s['price'] < a['value']:
            msg = f"{s['ticker']} price BELOW ${a['value']:.2f} → ${s['price']:.2f}"
        elif a['condition'] == "day_change_above" and a.get('value') and s['change_pct'] > a['value']:
            msg = f"{s['ticker']} DAY % ABOVE {a['value']}% → {s['change_pct']:+.2f}%"
        elif a['condition'] == "day_change_below" and a.get('value') and s['change_pct'] < a['value']:
            msg = f"{s['ticker']} DAY % BELOW {a['value']}% → {s['change_pct']:+.2f}%"
        elif a['condition'] == "rsi_oversold" and s['rsi'] is not None and s['rsi'] < 30:
            msg = f"{s['ticker']} RSI OVERSOLD → {s['rsi']:.1f}"
        elif a['condition'] == "rsi_overbought" and s['rsi'] is not None and s['rsi'] > 70:
            msg = f"{s['ticker']} RSI OVERBOUGHT → {s['rsi']:.1f}"
        elif a['condition'] == "volume_spike" and s['volume_spike']:
            msg = f"{s['ticker']} VOLUME SPIKE"
        if msg: triggered.append({'time': now, 'message': msg})

    for s in data:
        ch = s['change_pct']
        if ch > 15: triggered.append({'time': now, 'message': f"🚨 {s['ticker']} SURGED > +15% → {ch:+.2f}%"})
        elif ch < -15: triggered.append({'time': now, 'message': f"🚨 {s['ticker']} CRASHED < -15% → {ch:+.2f}%"})
    return triggered

def rsi(series):
    if len(series) < 15: return None
    d = series.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    return (100 - 100 / (1 + g/l)).iloc[-1]

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
        if len(reg) >= 2:
            prev = reg['Close'].iloc[-2]
            if prev and prev > 0: change_pct = ((price - prev) / prev) * 100

        change_1m = change_6m = None
        for d in [35, 190]:
            hist = t.history(start=(datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d'))
            if len(hist) >= 2:
                start = hist['Close'].iloc[0]
                if start and start > 0:
                    pct = ((price - start) / start) * 100
                    if d == 35: change_1m = pct
                    else: change_6m = pct

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

        # Options data
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

        analyst_rating = info.get('recommendationKey', 'none').title()

        upside = None
        target = info.get('targetMeanPrice')
        if target and price > 0: upside = ((target - price) / price) * 100

        time.sleep(1.5)

        ticker_u = ticker.upper()
        return {
            'ticker': ticker_u,
            'links': f"https://stockanalysis.com/stocks/{ticker_u.lower()}/ https://www.barchart.com/stocks/quotes/{ticker_u} https://www.tradingview.com/chart/?symbol={ticker_u} https://finviz.com/quote.ashx?t={ticker_u}",
            'price': price,
            'change_pct': change_pct,
            'change_1m_pct': change_1m,
            'change_6m_pct': change_6m,
            'volume': vol,
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
            'upside_potential': upside,
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
        time.sleep(10)
        return None

def fmt_vol(v):
    if v is None: return "N/A"
    if v >= 1e9: return f"{v/1e9:.1f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return str(int(v))

def fmt_change(p):
    if p is None: return '<span class="neutral">N/A</span>'
    sign = "▲" if p >= 0 else "▼"
    cls = "positive" if p >= 0 else "negative"
    return f'<span class="{cls}">{sign} {p:+.2f}%</span>'

def dashboard(csv='data/tickers.csv', ext=False):
    os.makedirs('data', exist_ok=True)
    try:
        tickers = pd.unique(pd.read_csv(csv).iloc[:,0]).tolist()
    except:
        tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','SPY']

    data = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in as_completed([ex.submit(fetch, t, ext) for t in tickers]):
            res = r.result()
            if res: data.append(res)

    df = pd.DataFrame(data).sort_values('change_pct', ascending=False)
    return df

def get_vix_data():
    try:
        v = yf.Ticker("^VIX").history(period="1d")
        if not v.empty:
            p = v['Close'].iloc[-1]
            prev = v.info.get('previousClose', p)
            ch = ((p - prev) / prev) * 100 if prev else 0
            return {'price': p, 'change_pct': ch}
    except: pass
    return {'price': None, 'change_pct': None}

def get_fear_greed_data():
    try:
        r = requests.get("https://edition.cnn.com/markets/fear-and-greed", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        m = re.search(r'Fear & Greed Index\s*\n\s*(\d+)', r.text)
        if m:
            s = int(m.group(1))
            rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4]
            return {'score': s, 'rating': rating}
    except: pass
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

def html(df, vix, fg, aaii, file, ext=False, refresh=60, alerts=None):
    alerts = alerts or []
    hours = "Extended Hours" if ext else "Regular Hours"
    update = datetime.now(UTC).astimezone(PST).strftime('%I:%M:%S %p PST on %B %d, %Y')

    banner = ""
    if alerts:
        banner = '<div class="alert-banner"><strong>🚨 ALERTS 🚨</strong><div class="alert-list">'
        for a in alerts[-5:]:
            banner += f"<div class='alert-item'>{a['time']} — {a['message']}</div>"
        banner += "</div></div>"

    vix_h = '<span class="neutral">N/A</span>'
    if vix['price'] is not None:
        vix_h = f'<span class="{"positive" if vix["change_pct"]>=0 else "negative"}">VIX: {vix["price"]:.2f} ({vix["change_pct"]:+.2f}%)</span>'

    fg_h = '<span class="neutral">N/A</span>'
    if fg['score'] is not None:
        cls = "negative" if fg['score'] <= 24 else "high-risk" if fg['score'] <= 44 else "neutral" if fg['score'] <= 55 else "bullish" if fg['score'] <= 74 else "positive"
        fg_h = f'<span class="{cls}">F&G: {fg["score"]:.0f} ({fg["rating"]})</span>'

    aaii_h = '<span class="neutral">N/A</span>'
    if aaii['bullish'] is not None:
        cls = "positive" if aaii['spread'] > 20 else "bullish" if aaii['spread'] > 0 else "neutral" if aaii['spread'] > -20 else "high-risk" if aaii['spread'] > -40 else "negative"
        aaii_h = f'<span class="{cls}">AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%</span>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Live Dashboard</title><meta http-equiv="refresh" content="{refresh}">
<style>
body{{font-family:Arial;margin:20px;background:#f5f5f5}}
.header-container{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}}
.refresh-btn,.switch-btn{{padding:8px 16px;background:#ff6600;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;text-decoration:none}}
.refresh-btn{{background:#0066cc}}
.mode-badge{{padding:8px 16px;border-radius:4px;font-weight:bold;font-size:0.9em}}
.regular-hours{{background:#0066cc;color:white}}.extended-hours{{background:#ff6600;color:white}}
.alert-banner{{background:#ff4444;color:white;padding:15px;border-radius:8px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;background:white;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
th{{background:#0066cc;color:white;padding:12px;cursor:pointer;position:relative}}
th::after{{content:"⇅";opacity:0.5;margin-left:5px}}
td{{padding:12px 10px;border-bottom:1px solid #ddd;vertical-align:top}}
.positive{{background:#00aa00;color:white;padding:4px 8px;border-radius:3px;font-weight:bold;display:inline-block}}
.negative{{background:#cc0000;color:white;padding:4px 8px;border-radius:3px;font-weight:bold;display:inline-block}}
.bullish{{color:#00aa00;font-weight:bold}}.bearish{{color:#cc0000;font-weight:bold}}
.high-risk{{color:#cc6600;font-weight:bold}}.neutral{{color:#666;font-style:italic}}
.ticker{{font-weight:bold;font-size:1.1em}}.link-group{{display:flex;gap:8px;flex-wrap:wrap;font-size:0.8em}}
.item{{font-size:0.85em;margin:4px 0;position:relative;cursor:help}}
.tooltip{{visibility:hidden;position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:#333;color:white;padding:8px;border-radius:4px;white-space:nowrap;font-size:0.8em;z-index:10;opacity:0;transition:opacity 0.3s}}
.item:hover .tooltip{{visibility:visible;opacity:1}}
</style></head><body>
{banner}
<div class="header-container">
    <div><h1>Live Dashboard</h1>
        <span class="mode-badge {'extended-hours' if ext else 'regular-hours'}">{hours}</span>
        <a href="{os.path.basename("data/extnd_dashboard.html" if not ext else "data/reg_dashboard.html")}" class="switch-btn">Switch Hours</a>
        <span class="vix-display">{vix_h}</span>
        <span class="fear-greed-display">{fg_h}</span>
        <span class="aaii-display">{aaii_h}</span>
    </div>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
</div>
<div><strong>Last updated:</strong> {update}</div>
<table>
<tr>
    <th>TICKER</th><th>PRICE</th><th>DAY %</th><th>1M %</th><th>6M %</th><th>VOLUME</th><th>52W RANGE</th><th>TECHNICAL</th><th>RISK / SENTIMENT</th>
</tr>"""
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
        if r['is_meme_stock']: risk.append('<div class="item"><strong>Meme:</strong> <span class="high-risk">🚀 Yes</span><span class="tooltip">High retail interest / meme stock</span></div>')
        vol_sc = 0.0
        if r['hv_30_annualized'] is not None and pd.notna(r['hv_30_annualized']): vol_sc += min(r['hv_30_annualized']/100*50, 50)
        if r['beta'] is not None and pd.notna(r['beta']): vol_sc += min(max(r['beta']-0.5,0)*33.33, 50)
        vol_sc = int(round(vol_sc))
        v_cls = "negative" if vol_sc >= 81 else "high-risk" if vol_sc >= 61 else "bearish" if vol_sc >= 31 else "neutral"
        v_emoji = "🔥🔥" if vol_sc >= 81 else "🔥" if vol_sc >= 61 else ""
        risk.append(f'<div class="item"><strong>Vol Score:</strong> <span class="{v_cls}">{v_emoji} {vol_sc}/100</span><span class="tooltip">Volatility score based on HV and Beta (higher = riskier)</span></div>')
        risk.append(f'<div class="item"><strong>Beta:</strong> {na(r["beta"])}<span class="tooltip">Stock volatility vs market (1.0 = market average)</span></div>')
        risk.append(f'<div class="item"><strong>P/E:</strong> {na(r["pe"])}<span class="tooltip">Price to Earnings ratio</span></div>')
        risk.append(f'<div class="item"><strong>Short %:</strong> {na(r["short_percent"], "{:.1f}%")}<span class="tooltip">Percentage of float sold short</span></div>')
        risk.append(f'<div class="item"><strong>Days to Cover:</strong> {na(r["days_to_cover"], "{:.1f}")}<span class="tooltip">Days to cover short interest at avg volume</span></div>')
        if r['death_cross']:
            risk.append('<div class="item"><strong>Trend:</strong> <span class="negative">Death Cross</span><span class="tooltip">50-day MA crossed below 200-day MA (bearish signal)</span></div>')
        if r['put_call_vol_ratio'] is not None:
            risk.append(f'<div class="item"><strong>P/C Vol Ratio:</strong> {r["put_call_vol_ratio"]:.2f}<span class="tooltip">Put/Call volume ratio (higher = bearish)</span></div>')
        if r['down_volume_bias']:
            risk.append('<div class="item"><strong>Volume Bias:</strong> <span class="negative">Down Days Higher</span><span class="tooltip">Higher volume on down days (bearish)</span></div>')
        risk.append(f'<div class="item"><strong>Options Direction:</strong> <span class="{"bearish" if "Bearish" in r["options_direction"] else "bullish"}">{r["options_direction"]}</span><span class="tooltip">Options flow sentiment</span></div>')
        if r['implied_move_pct'] is not None:
            risk.append(f'<div class="item"><strong>Implied Move ({r["exp_date_used"] or "N/A"}):</strong> ±{r["implied_move_pct"]*0.85:.1f}%<span class="tooltip">Expected price move from nearest options expiration (conservative)</span></div>')
            risk.append(f'<div class="item"><strong>Expected Range:</strong> ${r["implied_low"]:.2f} – ${r["implied_high"]:.2f}<span class="tooltip">Conservative expected price range</span></div>')
        risk.append(f'<div class="item"><strong>Sentiment:</strong> <span class="positive">{r["sentiment"]}</span><span class="tooltip">Analyst sentiment</span></div>')
        risk.append(f'<div class="item"><strong>Rating:</strong> <span class="positive">{r["analyst_rating"]}</span><span class="tooltip">Analyst rating</span></div>')
        if r['upside_potential'] is not None:
            up_cls = "positive" if r['upside_potential'] > 0 else "negative"
            risk.append(f'<div class="item"><strong>Upside:</strong> <span class="{up_cls}">{r["upside_potential"]:+.1f}%</span><span class="tooltip">Potential upside to analyst target price</span></div>')
        risk_h = "<br>".join(risk) or '<span class="neutral">N/A</span>'

        link_urls = r['links'].split()
        link_labels = ["SA", "BA", "TV", "FZ"]
        links = " ".join([f'<a href="{url}" target="_blank">{lbl}</a>' for url, lbl in zip(link_urls, link_labels)])

        html += f"""<tr>
    <td><div class="ticker"><a href="{link_urls[0]}" target="_blank">{r['ticker']}</a></div><div class="link-group">{links}</div></td>
    <td>{r['price']:.2f}</td>
    <td>{fmt_change(r['change_pct'])}</td>
    <td>{fmt_change(r['change_1m_pct'])}</td>
    <td>{fmt_change(r['change_6m_pct'])}</td>
    <td>{fmt_vol(r['volume'])}</td>
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
            let av = a.children[index].innerText.trim();
            let bv = b.children[index].innerText.trim();
            const num = !isNaN(av) && !isNaN(bv);
            if (num) [av, bv] = [parseFloat(av), parseFloat(bv)];
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
    parser.add_argument('--refresh', type=int, default=60)
    args = parser.parse_args()

    for ext, file, name in [(False, 'data/reg_dashboard.html', 'Regular'), (True, 'data/extnd_dashboard.html', 'Extended')]:
        try:
            df = dashboard(args.csv_file, ext)
            alerts = check_alerts(df.to_dict('records'))
            html(df, get_vix_data(), get_fear_greed_data(), get_aaii_sentiment(), file, ext, args.refresh, alerts)
            print(f"✓ {name} → {file}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")

    print("\nDashboards generated!")