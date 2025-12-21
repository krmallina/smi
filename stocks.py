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

        if extended_hours:
            hist = stock.history(period="2d", interval="1m", prepost=True)
        else:
            hist = stock.history(period="2d", interval="1m", prepost=False)
        if hist.empty:
            hist = stock.history(period="5d", prepost=extended_hours)
        if hist.empty:
            return None
        current_price = hist['Close'].iloc[-1]

        regular_hist = stock.history(period="5d", prepost=False)
        prev_close = regular_hist['Close'].iloc[-2] if len(regular_hist) >= 2 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100

        end_date = datetime.now()
        hist_1m = stock.history(start=(end_date - timedelta(days=40)).strftime('%Y-%m-%d'))
        change_1m_pct = ((current_price - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]) * 100 if len(hist_1m) >= 2 else None
        hist_6m = stock.history(start=(end_date - timedelta(days=200)).strftime('%Y-%m-%d'))
        change_6m_pct = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0]) * 100 if len(hist_6m) >= 2 else None

        year_hist = stock.history(period="1y", prepost=False)
        week_52_high = year_hist['High'].max() if not year_hist.empty else current_price
        week_52_low = year_hist['Low'].min() if not year_hist.empty else current_price

        today_hist = stock.history(period="1d", interval="1m", prepost=extended_hours)
        volume = today_hist['Volume'].sum() if not today_hist.empty else info.get('volume', 0)

        rsi_hist = stock.history(period="60d")
        rsi = calculate_rsi(rsi_hist['Close']) if len(rsi_hist) >= 15 else None
        rsi_label = get_rsi_label(rsi)

        macd_hist = stock.history(period="100d")
        macd, signal = calculate_macd(macd_hist['Close']) if len(macd_hist) >= 30 else (None, None)
        macd_label = get_macd_label(macd, signal)

        vol_hist = stock.history(period="30d")
        avg_vol = vol_hist['Volume'][:-1].mean() if len(vol_hist) > 1 else 1
        volume_spike = volume > 1.5 * avg_vol

        beta = info.get('beta')
        trailing_pe = info.get('trailingPE')

        short_percent = info.get('shortPercentOfFloat')
        if short_percent is not None:
            short_percent *= 100
        short_interest = info.get('sharesShort')
        avg_volume_10d = info.get('averageDailyVolume10Day') or avg_vol
        days_to_cover = short_interest / avg_volume_10d if short_interest and avg_volume_10d > 0 else None

        ma50 = year_hist['Close'].rolling(50).mean().iloc[-1] if len(year_hist) >= 50 else None
        ma200 = year_hist['Close'].rolling(200).mean().iloc[-1] if len(year_hist) >= 200 else None
        death_cross = ma50 < ma200 if ma50 and ma200 else False

        put_call_vol_ratio = None
        implied_move_pct = None
        exp_date_used = None
        if stock.options:
            nearest_exp = stock.options[0]
            exp_date_used = nearest_exp
            try:
                chain = stock.option_chain(nearest_exp)
                calls = chain.calls
                puts = chain.puts
                atm_strike = min(pd.concat([calls['strike'], puts['strike']]).unique(), key=lambda s: abs(s - current_price))
                call_price = calls[calls['strike'] == atm_strike]['lastPrice'].iloc[0] if not calls[calls['strike'] == atm_strike].empty else 0
                put_price = puts[puts['strike'] == atm_strike]['lastPrice'].iloc[0] if not puts[puts['strike'] == atm_strike].empty else 0
                straddle = call_price + put_price
                if straddle > 0:
                    implied_move_pct = (straddle / current_price) * 100
                call_vol = calls['volume'].sum()
                put_vol = puts['volume'].sum()
                if call_vol > 0:
                    put_call_vol_ratio = put_vol / call_vol
            except:
                pass

        hist_30d = stock.history(period="30d")
        down_volume_bias = hist_30d[hist_30d['Close'] < hist_30d['Open']]['Volume'].sum() > hist_30d[hist_30d['Close'] > hist_30d['Open']]['Volume'].sum()

        options_direction = "Neutral"
        if put_call_vol_ratio is not None:
            if put_call_vol_ratio > 1.2 and death_cross and down_volume_bias:
                options_direction = "Strong Bearish"
            elif put_call_vol_ratio > 1.0 or death_cross or down_volume_bias:
                options_direction = "Bearish"
            elif put_call_vol_ratio < 0.8:
                options_direction = "Bullish"

        sentiment_score = info.get('recommendationMean')
        sentiment = "N/A"
        if sentiment_score:
            if sentiment_score <= 1.5: sentiment = "Strong Buy"
            elif sentiment_score <= 2.5: sentiment = "Buy"
            elif sentiment_score <= 3.5: sentiment = "Hold"
            elif sentiment_score <= 4.5: sentiment = "Sell"
            else: sentiment = "Strong Sell"
        analyst_rating = info.get('recommendationKey', 'none').title()
        upside_potential = info.get('targetMeanPrice')
        if upside_potential:
            upside_potential = ((upside_potential - current_price) / current_price) * 100

        return {
            'ticker': ticker.upper(),
            'price': round(current_price, 2),
            'change_pct': round(change_pct, 2),
            'change_1m_pct': round(change_1m_pct, 2) if change_1m_pct else None,
            'change_6m_pct': round(change_6m_pct, 2) if change_6m_pct else None,
            'volume': int(volume),
            '52w_high': round(week_52_high, 2),
            '52w_low': round(week_52_low, 2),
            'rsi': round(rsi, 1) if rsi else None,
            'rsi_label': rsi_label,
            'macd': round(macd, 3) if macd else None,
            'macd_signal': round(signal, 3) if signal else None,
            'macd_label': macd_label,
            'volume_spike': volume_spike,
            'beta': round(beta, 2) if beta else None,
            'trailing_pe': round(trailing_pe, 1) if trailing_pe else None,
            'short_percent': round(short_percent, 1) if short_percent else None,
            'days_to_cover': round(days_to_cover, 1) if days_to_cover else None,
            'death_cross': death_cross,
            'put_call_vol_ratio': round(put_call_vol_ratio, 2) if put_call_vol_ratio else None,
            'down_volume_bias': down_volume_bias,
            'options_direction': options_direction,
            'implied_move_pct': round(implied_move_pct, 1) if implied_move_pct else None,
            'exp_date_used': exp_date_used,
            'sentiment': sentiment,
            'analyst_rating': analyst_rating,
            'upside_potential': round(upside_potential, 1) if upside_potential else None,
            'yahoo_link': f"https://finance.yahoo.com/quote/{ticker.upper()}",
            'tradingview_link': f"https://www.tradingview.com/chart/?symbol={ticker.upper()}",
            'finviz_link': f"https://finviz.com/quote.ashx?t={ticker.upper()}"
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
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY']
    
    data = [d for t in tickers if (d := get_stock_data(t, extended_hours))]
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
        alert_banner = '<div style="background:#ff4444;color:white;padding:15px;border-radius:8px;margin:10px 0;text-align:center;font-weight:bold">🚨 REAL-TIME ALERTS 🚨</div><div style="background:#333;color:white;padding:10px;border-radius:8px;margin-bottom:15px;">' + \
                       '<br>'.join([f"{a['time']} — {a['message']}" for a in triggered_alerts[-5:]]) + '</div>'

    vix_html = f"VIX {vix_data['price']:.2f} ({vix_data['change_pct']:+.2f}%)" if vix_data['price'] else "VIX N/A"
    fg_html = f"F&G {fg_data['score']:.0f} ({fg_data['rating']})" if fg_data['score'] else "F&G N/A"
    aaii_html = f"AAII Bull {aaii_data['bullish']:.0f}% Bear {aaii_data['bearish']:.0f}%" if aaii_data['bullish'] else "AAII N/A"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{hours_type} Dashboard</title>
    <meta http-equiv="refresh" content="{refresh_interval}">
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 10px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ padding: 16px; background: #000; color: white; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }}
        h1 {{ margin: 0; font-size: 1.8em; }}
        .mode {{ padding: 8px 16px; border-radius: 8px; font-weight: bold; background: {'#ff6600' if extended_hours else '#0066cc'}; }}
        .switch-btn {{ padding: 10px 16px; background: #ff6600; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; }}
        .indicators {{ padding: 12px; background: #eee; display: flex; flex-wrap: wrap; gap: 20px; font-weight: bold; }}
        .info {{ padding: 10px 16px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #0066cc; color: white; padding: 12px; text-align: left; font-size: 0.9em; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; vertical-align: top; }}
        .ticker a {{ font-weight: bold; font-size: 1.2em; color: #000; text-decoration: none; }}
        .links a {{ margin-right: 10px; color: #0066cc; font-size: 0.9em; }}
        .bar-container {{ margin: 8px 0; }}
        .bar {{ height: 20px; background: linear-gradient(to right, #cc0000 0%, #cc0000 var(--pos), #00aa00 var(--pos), #00aa00 100%); border-radius: 10px; position: relative; }}
        .bar-labels {{ display: flex; justify-content: space-between; font-size: 0.8em; margin-top: 4px; }}
        .tech-items, .risk-items {{ font-size: 0.9em; line-height: 1.6; }}
        .badge-red {{ background: #cc0000; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
        .badge-green {{ background: #00aa00; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
        @media (max-width: 768px) {{ 
            th, td {{ padding: 8px; font-size: 0.85em; }}
            .header {{ flex-direction: column; align-items: flex-start; }}
            .indicators {{ flex-direction: column; }}
        }}
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
            <button onclick="location.reload()" style="padding:10px;background:#0066cc;color:white;border:none;border-radius:8px;cursor:pointer;">Refresh</button>
        </div>
        <div class="indicators">
            <div>{vix_html}</div>
            <div>{fg_html}</div>
            <div>{aaii_html}</div>
        </div>
        <div class="info">Last updated: {last_update} • Auto-refresh: {refresh_interval}s</div>
        <table>
            <thead>
                <tr>
                    <th>TICKER</th>
                    <th>PRICE / DAY %</th>
                    <th>1M / 6M %</th>
                    <th>VOLUME</th>
                    <th>52W RANGE</th>
                    <th>TECHNICAL INDICATORS</th>
                    <th>RISK / SENTIMENT</th>
                </tr>
            </thead>
            <tbody>
"""
    for _, row in df.iterrows():
        pos_pct = ((row['price'] - row['52w_low']) / (row['52w_high'] - row['52w_low'])) * 100 if row['52w_high'] > row['52w_low'] else 50

        tech_items = []
        if row['rsi'] is not None:
            tech_items.append(f"RSI(14): {row['rsi']:.1f} ({row['rsi_label']})")
        if row['macd'] is not None:
            macd_class = "badge-green" if row['macd_label'] == "Bullish" else "badge-red"
            tech_items.append(f"MACD: {row['macd']:.3f} / {row['macd_signal']:.3f} (<span class=\"{macd_class}\">{row['macd_label']}</span>)")
        tech_items.append(f"Vol Spike: {'Yes' if row['volume_spike'] else 'No'}")

        risk_items = []
        if row['beta'] is not None:
            risk_items.append(f"Beta: {row['beta']}")
        if row['trailing_pe'] is not None:
            risk_items.append(f"P/E: {row['trailing_pe']}")
        if row['short_percent'] is not None:
            risk_items.append(f"Short %: {row['short_percent']}%")
        if row['days_to_cover'] is not None:
            risk_items.append(f"Days to Cover: {row['days_to_cover']}")
        if row['death_cross']:
            risk_items.append('<span class="badge-red">Trend: Death Cross</span>')
        if row['put_call_vol_ratio'] is not None:
            risk_items.append(f"P/C Vol Ratio: <span class=\"badge-red\">{row['put_call_vol_ratio']}</span>")
        if row['down_volume_bias']:
            risk_items.append('<span class="badge-red">Volume Bias: Down Days Higher</span>')
        dir_class = "badge-red" if "Bearish" in row['options_direction'] else "badge-green"
        risk_items.append(f"Options Direction: <span class=\"{dir_class}\">{row['options_direction']}</span>")
        if row['implied_move_pct'] is not None:
            risk_items.append(f"Implied Move ({row['exp_date_used'] or 'N/A'}): ±{row['implied_move_pct']*0.85:.1f}%")
            low = row['price'] * (1 - row['implied_move_pct']*0.85/100)
            high = row['price'] * (1 + row['implied_move_pct']*0.85/100)
            risk_items.append(f"Expected Range: ${low:.2f} – ${high:.2f}")
        if row['sentiment'] != "N/A":
            sent_class = "badge-green" if "Buy" in row['sentiment'] else "badge-red"
            risk_items.append(f"Sentiment: <span class=\"{sent_class}\">{row['sentiment']}</span>")
        if row['analyst_rating'] != 'None':
            rat_class = "badge-green" if "Buy" in row['analyst_rating'] else "badge-red"
            risk_items.append(f"Rating: <span class=\"{rat_class}\">{row['analyst_rating']}</span>")
        if row['upside_potential'] is not None:
            up_class = "badge-green" if row['upside_potential'] > 0 else "badge-red"
            risk_items.append(f"Upside: <span class=\"{up_class}\">{row['upside_potential']:+.1f}%</span>")

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
                    <td>
                        ${row['price']:.2f}<br>
                        {format_change(row['change_pct'])}
                    </td>
                    <td>
                        {format_change(row['change_1m_pct'])}<br>
                        {format_change(row['change_6m_pct'])}
                    </td>
                    <td>{format_volume(row['volume'])}</td>
                    <td class="bar-container">
                        <div class="bar" style="--pos: {pos_pct:.1f}%;"></div>
                        <div class="bar-labels">
                            <span>${row['52w_low']:.2f}</span>
                            <span>${row['52w_high']:.2f}</span>
                        </div>
                    </td>
                    <td class="tech-items">
                        {"<br>".join(tech_items)}
                    </td>
                    <td class="risk-items">
                        {"<br>".join(risk_items)}
                    </td>
                </tr>
"""
    if df.empty:
        html += '<tr><td colspan="7" style="text-align:center;padding:50px;color:#999;">No data available</td></tr>'

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
        print(f"Generated {'Extended' if extended else 'Regular'} dashboard → {outfile}")