import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json
import requests
import re

# Alert configuration
ALERTS_FILE = 'data/alerts.json'

def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading alerts: {e}")
    return []

def check_alerts(stock_data_list):
    alerts = load_alerts()
    if not alerts:
        return []
    
    triggered = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for alert in alerts:
        ticker = alert['ticker'].upper()
        condition = alert['condition']
        value = alert.get('value')
        
        stock = next((s for s in stock_data_list if s['ticker'] == ticker), None)
        if not stock:
            continue
        
        message = ""
        if condition == "price_above" and value is not None and stock['price'] > value:
            message = f"{ticker} price ABOVE ${value:.2f} → ${stock['price']:.2f}"
        elif condition == "price_below" and value is not None and stock['price'] < value:
            message = f"{ticker} price BELOW ${value:.2f} → ${stock['price']:.2f}"
        elif condition == "day_change_above" and value is not None and stock['change_pct'] > value:
            message = f"{ticker} DAY % ABOVE {value}% → {stock['change_pct']:+.2f}%"
        elif condition == "day_change_below" and value is not None and stock['change_pct'] < value:
            message = f"{ticker} DAY % BELOW {value}% → {stock['change_pct']:+.2f}%"
        elif condition == "rsi_oversold" and stock['rsi'] is not None and stock['rsi'] < 30:
            message = f"{ticker} RSI OVERSOLD → {stock['rsi']:.1f}"
        elif condition == "rsi_overbought" and stock['rsi'] is not None and stock['rsi'] > 70:
            message = f"{ticker} RSI OVERBOUGHT → {stock['rsi']:.1f}"
        elif condition == "volume_spike" and stock['volume_spike']:
            message = f"{ticker} VOLUME SPIKE detected"
        
        if message:
            triggered.append({'time': current_time, 'message': message})
    
    return triggered

def calculate_rsi(series, period=14):
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_rsi_label(rsi):
    if rsi is None:
        return "N/A"
    if rsi < 30:
        return "Oversold"
    elif rsi > 70:
        return "Overbought"
    return "Neutral"

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    if len(close_prices) < slow:
        return None, None
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

def get_macd_label(macd, signal):
    if macd is None or signal is None:
        return "N/A"
    if macd > signal:
        return "Bullish"
    elif macd < signal:
        return "Bearish"
    return "Neutral"

def get_vix_data():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            prev_close = vix.info.get('previousClose', price)
            change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0
            return {'price': price, 'change_pct': change_pct}
    except Exception as e:
        print(f"VIX error: {e}")
    return {'price': None, 'change_pct': None}

def get_fear_greed_data():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        data = r.json()
        fg = data['fear_and_greed']
        return {'score': fg['score'], 'rating': fg['rating'].title()}
    except:
        return {'score': None, 'rating': "N/A"}

def get_aaii_sentiment():
    try:
        url = "https://www.aaii.com/sentimentsurvey/sent_results"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        text = r.text
        bull_match = re.search(r'Bullish\s*:\s*([\d\.]+)%', text)
        bear_match = re.search(r'Bearish\s*:\s*([\d\.]+)%', text)
        if bull_match and bear_match:
            bull = float(bull_match.group(1))
            bear = float(bear_match.group(1))
            return {'bullish': bull, 'bearish': bear}
    except:
        pass
    return {'bullish': None, 'bearish': None}

def get_stock_data(ticker, extended_hours=False):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Price
        if extended_hours:
            hist = stock.history(period="2d", interval="1m", prepost=True)
        else:
            hist = stock.history(period="2d", interval="1m", prepost=False)
        if hist.empty:
            hist = stock.history(period="5d", prepost=extended_hours)
        if hist.empty:
            return None
        current_price = hist['Close'].iloc[-1]

        # Day change
        regular_hist = stock.history(period="5d", prepost=False)
        prev_close = regular_hist['Close'].iloc[-2] if len(regular_hist) >= 2 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100

        # 1M / 6M
        end_date = datetime.now()
        hist_1m = stock.history(start=(end_date - timedelta(days=40)).strftime('%Y-%m-%d'))
        change_1m_pct = ((current_price - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]) * 100 if len(hist_1m) >= 2 else None
        hist_6m = stock.history(start=(end_date - timedelta(days=200)).strftime('%Y-%m-%d'))
        change_6m_pct = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0]) * 100 if len(hist_6m) >= 2 else None

        # 52-week
        year_hist = stock.history(period="1y")
        week_52_high = year_hist['High'].max() if not year_hist.empty else current_price
        week_52_low = year_hist['Low'].min() if not year_hist.empty else current_price

        # Volume
        today_hist = stock.history(period="1d", interval="1m", prepost=extended_hours)
        volume = today_hist['Volume'].sum() if not today_hist.empty else info.get('volume', 0)

        # Indicators
        rsi_hist = stock.history(period="60d")
        rsi = calculate_rsi(rsi_hist['Close']) if len(rsi_hist) >= 15 else None
        rsi_label = get_rsi_label(rsi)

        macd_hist = stock.history(period="100d")
        macd, signal = calculate_macd(macd_hist['Close']) if len(macd_hist) >= 30 else (None, None)
        macd_label = get_macd_label(macd, signal)

        vol_hist = stock.history(period="30d")
        avg_vol = vol_hist['Volume'][:-1].mean() if len(vol_hist) > 1 else 1
        volume_spike = volume > 1.5 * avg_vol

        return {
            'ticker': ticker.upper(),
            'yahoo_link': f"https://finance.yahoo.com/quote/{ticker.upper()}",
            'tradingview_link': f"https://www.tradingview.com/chart/?symbol={ticker.upper()}",
            'finviz_link': f"https://finviz.com/quote.ashx?t={ticker.upper()}",
            'price': round(current_price, 2),
            'change_pct': round(change_pct, 2),
            'change_1m_pct': round(change_1m_pct, 2) if change_1m_pct is not None else None,
            'change_6m_pct': round(change_6m_pct, 2) if change_6m_pct is not None else None,
            'volume': int(volume),
            '52w_high': round(week_52_high, 2),
            '52w_low': round(week_52_low, 2),
            'rsi': round(rsi, 1) if rsi else None,
            'rsi_label': rsi_label,
            'macd_label': macd_label,
            'volume_spike': volume_spike
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def format_volume(v):
    if v >= 1e9: return f"{v/1e9:.1f}B"
    elif v >= 1e6: return f"{v/1e6:.1f}M"
    elif v >= 1e3: return f"{v/1e3:.1f}K"
    return str(int(v))

def format_change(p):
    if p is None: return '<span style="color:#666">—</span>'
    sign = "▲" if p >= 0 else "▼"
    color = "#00aa00" if p >= 0 else "#cc0000"
    return f'<span style="color:{color};font-weight:bold">{sign} {p:+.2f}%</span>'

def create_dashboard(csv_file='data/tickers.csv', extended_hours=False):
    os.makedirs('data', exist_ok=True)
    try:
        tickers = pd.unique(pd.read_csv(csv_file).iloc[:, 0]).astype(str).str.upper().tolist()
    except:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ', 'AVGO', 'ORCL', 'COIN', 'SOXL', 'FNGU', 'TSLL']
    
    data = []
    for t in tickers:
        stock = get_stock_data(t, extended_hours)
        if stock:
            data.append(stock)
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('change_pct', ascending=False)
    return df

def export_to_html(df, vix_data, fg_data, aaii_data, output_file, extended_hours=False, refresh_interval=60, triggered_alerts=None):
    if triggered_alerts is None:
        triggered_alerts = []
    
    hours_type = "Extended Hours" if extended_hours else "Regular Hours"
    last_update = datetime.now().strftime('%I:%M %p on %b %d, %Y')

    other_file = "extnd_dashboard.html" if not extended_hours else "reg_dashboard.html"
    other_mode = "Extended" if not extended_hours else "Regular"

    alert_banner = ''
    if triggered_alerts:
        alert_banner = '<div style="background:#ff4444;color:white;padding:12px;border-radius:8px;margin:10px 0;font-weight:bold">🚨 ALERTS 🚨<br>' + \
                       '<br>'.join([f"{a['time']} — {a['message']}" for a in triggered_alerts[-5:]]) + '</div>'

    vix_html = f"VIX: {vix_data['price']:.2f} ({vix_data['change_pct']:+.2f}%)" if vix_data['price'] else "VIX: N/A"
    fg_html = f"F&G: {fg_data['score']:.0f} ({fg_data['rating']})" if fg_data['score'] else "F&G: N/A"
    aaii_html = f"AAII: Bull {aaii_data['bullish']:.0f}% Bear {aaii_data['bearish']:.0f}%" if aaii_data['bullish'] else "AAII: N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{hours_type} - Stock Dashboard</title>
    <meta http-equiv="refresh" content="{refresh_interval}">
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 12px; background: #f0f0f0; }}
        .container {{ max-width: 1400px; margin: auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ padding: 16px; background: #111; color: white; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }}
        h1 {{ margin: 0; font-size: 1.6em; }}
        .mode {{ padding: 8px 16px; border-radius: 8px; font-weight: bold; background: {'#ff6600' if extended_hours else '#0066cc'}; }}
        .switch-btn {{ padding: 10px 16px; background: #ff6600; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; }}
        .indicators {{ padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 16px; font-weight: bold; }}
        .info {{ padding: 0 16px 12px; color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #0066cc; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f8ff; }}
        .ticker a {{ font-weight: bold; font-size: 1.2em; color: #000; text-decoration: none; }}
        .links a {{ margin-right: 12px; color: #0066cc; font-size: 0.9em; }}
        .bar {{ height: 8px; background: #eee; border-radius: 4px; overflow: hidden; margin: 6px 0; position: relative; }}
        .fill {{ height: 100%; background: #00aa00; }}
        .marker {{ position: absolute; top: 0; width: 3px; height: 100%; background: black; left: 50%; }}
        @media (max-width: 768px) {{ .header {{ flex-direction: column; align-items: flex-start; }} .indicators {{ flex-direction: column; }} }}
    </style>
</head>
<body>
    <div class="container">
        {alert_banner}
        <div class="header">
            <div>
                <h1>Stock Dashboard</h1>
                <span class="mode">{hours_type}</span>
                <a href="{other_file}" class="switch-btn">Switch to {other_mode} Hours</a>
            </div>
            <button onclick="location.reload()" style="padding:10px 16px;background:#0066cc;color:white;border:none;border-radius:8px;cursor:pointer;">Refresh</button>
        </div>
        <div class="indicators">
            <span>{vix_html}</span>
            <span>{fg_html}</span>
            <span>{aaii_html}</span>
        </div>
        <div class="info">Last updated: {last_update} • Auto-refresh every {refresh_interval}s</div>
        <table>
            <thead>
                <tr><th>TICKER</th><th>PRICE</th><th>DAY</th><th>1M</th><th>6M</th><th>VOL</th><th>52W</th><th>TECH</th></tr>
            </thead>
            <tbody>
"""
    for _, row in df.iterrows():
        pos = ((row['price'] - row['52w_low']) / (row['52w_high'] - row['52w_low'])) * 100 if row['52w_high'] > row['52w_low'] else 50
        tech = []
        if row['rsi'] is not None: tech.append(f"RSI {row['rsi']:.1f} ({row['rsi_label']})")
        if row['macd_label'] != "N/A": tech.append(row['macd_label'])
        if row['volume_spike']: tech.append("Vol Spike")
        tech_str = " • ".join(tech) if tech else "—"

        html += f"""
                <tr>
                    <td>
                        <div class="ticker"><a href="{row['yahoo_link']}" target="_blank">{row['ticker']}</a></div>
                        <div class="links">
                            <a href="{row['yahoo_link']}" target="_blank">Yahoo</a>
                            <a href="{row['tradingview_link']}" target="_blank">TV</a>
                            <a href="{row['finviz_link']}" target="_blank">Finviz</a>
                        </div>
                    </td>
                    <td>${row['price']:.2f}</td>
                    <td>{format_change(row['change_pct'])}</td>
                    <td>{format_change(row['change_1m_pct'])}</td>
                    <td>{format_change(row['change_6m_pct'])}</td>
                    <td>{format_volume(row['volume'])}</td>
                    <td>
                        <div class="bar"><div class="fill" style="width:{pos:.0f}%"></div><div class="marker" style="left:{pos:.0f}%"></div></div>
                        <small>${row['52w_low']:.0f} – ${row['52w_high']:.0f}</small>
                    </td>
                    <td>{tech_str}</td>
                </tr>
"""
    if df.empty:
        html += '<tr><td colspan="8" style="text-align:center;padding:50px;color:#999;font-size:1.2em;">No data available — check internet or tickers list</td></tr>'

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', nargs='?', default='data/tickers.csv')
    parser.add_argument('--refresh', type=int, default=60)
    args = parser.parse_args()

    configs = [
        (False, 'data/reg_dashboard.html'),
        (True,  'data/extnd_dashboard.html')
    ]

    print("Generating dashboards...\n")

    for extended, outfile in configs:
        df = create_dashboard(args.csv_file, extended_hours=extended)
        alerts = check_alerts(df.to_dict('records') if not df.empty else [])
        
        export_to_html(
            df=df,
            vix_data=get_vix_data(),
            fg_data=get_fear_greed_data(),
            aaii_data=get_aaii_sentiment(),
            output_file=outfile,
            extended_hours=extended,
            refresh_interval=args.refresh,
            triggered_alerts=alerts
        )
        
        mode = "Extended" if extended else "Regular"
        print(f"✓ {mode} Hours dashboard generated → {outfile}")
        if alerts:
            print(f"   🚨 {len(alerts)} alert(s) triggered")

    print("\nDone! Upload data/reg_dashboard.html and data/extnd_dashboard.html to your GitHub repo.")
    print("Links:")
    print("   https://krmallina.github.io/smi/data/reg_dashboard.html")
    print("   https://krmallina.github.io/smi/data/extnd_dashboard.html")