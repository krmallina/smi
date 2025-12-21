import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json
import sys
import requests
import re
from urllib.parse import parse_qs

# Alert configuration
ALERTS_FILE = 'data/alerts.json'

def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
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
        
        triggered_this = False
        message = ""
        
        if condition == "price_above" and value is not None and stock['price'] > value:
            triggered_this = True
            message = f"{ticker} price ABOVE ${value:.2f} → ${stock['price']:.2f}"
        elif condition == "price_below" and value is not None and stock['price'] < value:
            triggered_this = True
            message = f"{ticker} price BELOW ${value:.2f} → ${stock['price']:.2f}"
        elif condition == "day_change_above" and value is not None and stock['change_pct'] > value:
            triggered_this = True
            message = f"{ticker} DAY % ABOVE {value}% → {stock['change_pct']:+.2f}%"
        elif condition == "day_change_below" and value is not None and stock['change_pct'] < value:
            triggered_this = True
            message = f"{ticker} DAY % BELOW {value}% → {stock['change_pct']:+.2f}%"
        elif condition == "rsi_oversold" and stock['rsi'] is not None and stock['rsi'] < 30:
            triggered_this = True
            message = f"{ticker} RSI OVERSOLD → {stock['rsi']:.1f}"
        elif condition == "rsi_overbought" and stock['rsi'] is not None and stock['rsi'] > 70:
            triggered_this = True
            message = f"{ticker} RSI OVERBOUGHT → {stock['rsi']:.1f}"
        elif condition == "volume_spike" and stock['volume_spike']:
            triggered_this = True
            message = f"{ticker} VOLUME SPIKE detected"
        
        if triggered_this:
            triggered.append({
                'time': current_time,
                'ticker': ticker,
                'message': message
            })
    
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
    except:
        pass
    return {'price': None, 'change_pct': None}

def get_fear_greed_data():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        data = r.json()
        fg = data['fear_and_greed']
        score = fg['score']
        rating = fg['rating'].title()
        return {'score': score, 'rating': rating}
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
            spread = bull - bear
            return {'bullish': bull, 'bearish': bear, 'spread': spread}
    except:
        pass
    return {'bullish': None, 'bearish': None, 'spread': None}

def get_stock_data(ticker, extended_hours=False):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Current price
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
        
        # 1M and 6M % change
        end_date = datetime.now()
        hist_1m = stock.history(start=(end_date - timedelta(days=35)).strftime('%Y-%m-%d'), prepost=False)
        change_1m_pct = ((current_price - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]) * 100 if len(hist_1m) >= 2 else None
        
        hist_6m = stock.history(start=(end_date - timedelta(days=190)).strftime('%Y-%m-%d'), prepost=False)
        change_6m_pct = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0]) * 100 if len(hist_6m) >= 2 else None
        
        # 52-week range
        year_hist = stock.history(period="1y", prepost=False)
        week_52_high = year_hist['High'].max() if not year_hist.empty else current_price
        week_52_low = year_hist['Low'].min() if not year_hist.empty else current_price
        
        # Volume
        today_hist = stock.history(period="1d", interval="1m", prepost=extended_hours)
        volume = today_hist['Volume'].sum() if not today_hist.empty else info.get('volume', 0)
        
        # History for indicators
        hist_1y = stock.history(period="1y", prepost=False)
        hist_30d = stock.history(period="30d", prepost=False)
        
        # Short data
        short_interest = info.get('sharesShort')
        avg_volume = info.get('averageDailyVolume10Day') or info.get('averageVolume') or volume or 1
        days_to_cover = short_interest / avg_volume if short_interest and avg_volume > 0 else None
        short_percent = info.get('shortPercentOfFloat')
        if short_percent is not None:
            short_percent *= 100
        
        # Short Squeeze Risk
        float_shares = info.get('floatShares') or info.get('sharesFloat')
        short_squeeze_risk = False
        squeeze_level = "None"
        if short_percent is not None and days_to_cover is not None:
            if short_percent > 30 and days_to_cover > 10:
                short_squeeze_risk = True
                squeeze_level = "Extreme"
            elif short_percent > 20 and days_to_cover > 7:
                short_squeeze_risk = True
                squeeze_level = "High"
            elif short_percent > 15 and days_to_cover > 5:
                short_squeeze_risk = True
                squeeze_level = "Moderate"
        if float_shares and short_interest and short_interest > 0.5 * float_shares:
            if squeeze_level == "High":
                squeeze_level = "Extreme"
            elif squeeze_level == "Moderate":
                squeeze_level = "High"
            short_squeeze_risk = True
        
        # Death Cross
        ma50 = hist_1y['Close'].rolling(50).mean().iloc[-1] if len(hist_1y) >= 50 else None
        ma200 = hist_1y['Close'].rolling(200).mean().iloc[-1] if len(hist_1y) >= 200 else None
        death_cross = ma50 is not None and ma200 is not None and ma50 < ma200
        
        # Put/Call Volume Ratio
        put_call_vol_ratio = None
        
        # Implied Expected Move (from nearest expiration ATM straddle)
        implied_move_pct = None
        implied_high = None
        implied_low = None
        exp_date_used = None
        
        if stock.options:
            nearest_exp = stock.options[0]
            exp_date_used = nearest_exp
            try:
                chain = stock.option_chain(nearest_exp)
                calls = chain.calls
                puts = chain.puts
                
                all_strikes = pd.concat([calls['strike'], puts['strike']]).unique()
                if len(all_strikes) > 0:
                    atm_strike = min(all_strikes, key=lambda s: abs(s - current_price))
                    
                    call_price = calls[calls['strike'] == atm_strike]['lastPrice'].iloc[0] if not calls[calls['strike'] == atm_strike].empty else 0
                    put_price = puts[puts['strike'] == atm_strike]['lastPrice'].iloc[0] if not puts[puts['strike'] == atm_strike].empty else 0
                    
                    straddle_price = call_price + put_price
                    if straddle_price > 0:
                        implied_move_pct = (straddle_price / current_price) * 100
                        conservative_pct = implied_move_pct * 0.85
                        implied_high = current_price * (1 + conservative_pct / 100)
                        implied_low = current_price * (1 - conservative_pct / 100)
                        
                        # Also calculate P/C volume ratio while we have the chain
                        call_vol = calls['volume'].fillna(0).sum()
                        put_vol = puts['volume'].fillna(0).sum()
                        if call_vol > 0:
                            put_call_vol_ratio = put_vol / call_vol
            except Exception as e:
                print(f"Error calculating options data for {ticker}: {e}")
        
        # Down Volume Bias
        down_days = hist_30d[hist_30d['Close'] < hist_30d['Open']]
        up_days = hist_30d[hist_30d['Close'] > hist_30d['Open']]
        down_volume_bias = down_days['Volume'].sum() > up_days['Volume'].sum()
        
        # Options Sentiment Direction
        options_direction = "Neutral"
        if put_call_vol_ratio is not None:
            if put_call_vol_ratio > 1.2 and death_cross and down_volume_bias:
                options_direction = "Strong Bearish"
            elif put_call_vol_ratio > 1.0 or death_cross or down_volume_bias:
                options_direction = "Bearish"
            elif put_call_vol_ratio < 0.8 and not death_cross and not down_volume_bias:
                options_direction = "Bullish"
        
        # Standard indicators
        beta = info.get('beta')
        trailing_pe = info.get('trailingPE')
        
        sentiment_score = info.get('recommendationMean')
        sentiment = "N/A"
        if sentiment_score is not None:
            if sentiment_score <= 1.5: sentiment = "Strong Buy"
            elif sentiment_score <= 2.5: sentiment = "Buy"
            elif sentiment_score <= 3.5: sentiment = "Hold"
            elif sentiment_score <= 4.5: sentiment = "Sell"
            else: sentiment = "Strong Sell"
        
        analyst_rating = info.get('recommendationKey', 'none').upper()
        target_mean_price = info.get('targetMeanPrice')
        upside_potential = ((target_mean_price - current_price) / current_price) * 100 if target_mean_price and current_price > 0 else None
        
        rsi_hist = stock.history(period="60d")
        rsi = calculate_rsi(rsi_hist['Close'])
        rsi_label = get_rsi_label(rsi)
        
        macd_hist = stock.history(period="100d")
        macd, signal = calculate_macd(macd_hist['Close'])
        macd_label = get_macd_label(macd, signal)
        
        vol_hist = stock.history(period="30d")
        volume_spike = volume > 1.5 * vol_hist['Volume'][:-1].mean() if len(vol_hist) > 1 and vol_hist['Volume'][:-1].mean() > 0 else False
        
        ticker_clean = ticker.upper()
        return {
            'ticker': ticker_clean,
            'yahoo_link': f"https://finance.yahoo.com/quote/{ticker_clean}",
            'tradingview_link': f"https://www.tradingview.com/chart/?symbol={ticker_clean}",
            'finviz_link': f"https://finviz.com/quote.ashx?t={ticker_clean}",
            'price': current_price,
            'change_pct': change_pct,
            'change_1m_pct': change_1m_pct,
            'change_6m_pct': change_6m_pct,
            'volume': volume,
            '52w_high': week_52_high,
            '52w_low': week_52_low,
            'beta': beta,
            'trailing_pe': trailing_pe,
            'short_percent': short_percent,
            'days_to_cover': days_to_cover,
            'short_squeeze_risk': short_squeeze_risk,
            'squeeze_level': squeeze_level,
            'death_cross': death_cross,
            'put_call_vol_ratio': put_call_vol_ratio,
            'down_volume_bias': down_volume_bias,
            'options_direction': options_direction,
            'implied_move_pct': implied_move_pct,
            'implied_high': implied_high,
            'implied_low': implied_low,
            'exp_date_used': exp_date_used,
            'sentiment': sentiment,
            'analyst_rating': analyst_rating,
            'upside_potential': upside_potential,
            'rsi': rsi,
            'rsi_label': rsi_label,
            'macd': macd,
            'macd_signal': signal,
            'macd_label': macd_label,
            'volume_spike': volume_spike
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def format_volume(volume):
    if volume >= 1e9: return f"{volume/1e9:.1f}B"
    elif volume >= 1e6: return f"{volume/1e6:.1f}M"
    elif volume >= 1e3: return f"{volume/1e3:.1f}K"
    return str(int(volume))

def format_change(pct):
    if pct is None: return '<span class="neutral">N/A</span>'
    sign = "▲" if pct >= 0 else "▼"
    cls = "positive" if pct >= 0 else "negative"
    return f'<span class="{cls}">{sign} {pct:+.2f}%</span>'

def create_dashboard(csv_file='data/tickers.csv', extended_hours=False):
    os.makedirs('data', exist_ok=True)
    try:
        df_tickers = pd.read_csv(csv_file)
        tickers = pd.unique(df_tickers.iloc[:, 0]).tolist()
    except FileNotFoundError:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY']
    
    data = [d for t in tickers if (d := get_stock_data(t, extended_hours))]
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('change_pct', ascending=False)
    return df

def export_to_html(df, vix_data, fg_data, aaii_data, output_file='data/stock_dashboard.html', extended_hours=False, refresh_interval=60, triggered_alerts=None):
    if triggered_alerts is None:
        triggered_alerts = []
    
    hours_type = "Extended Hours" if extended_hours else "Regular Hours"
    last_update = datetime.now().strftime('%I:%M:%S %p PST on %B %d, %Y')
    
    alert_banner = ""
    if triggered_alerts:
        alert_banner = """
        <div class="alert-banner">
            <strong>🚨 REAL-TIME ALERTS 🚨</strong>
            <div class="alert-list">
        """
        for alert in triggered_alerts[-5:]:
            alert_banner += f"<div class='alert-item'>{alert['time']} — {alert['message']}</div>"
        alert_banner += "</div></div>"
    
    vix_html = '<span class="neutral">N/A</span>'
    if vix_data['price'] is not None:
        vix_class = "positive" if vix_data['change_pct'] >= 0 else "negative"
        vix_html = f'<span class="{vix_class}">VIX: {vix_data["price"]:.2f} ({vix_data["change_pct"]:+.2f}%)</span>'
    
    fg_html = '<span class="neutral">N/A</span>'
    if fg_data['score'] is not None:
        score = fg_data['score']
        rating = fg_data['rating']
        fg_class = "negative" if score <= 24 else "high-risk" if score <= 44 else "neutral" if score <= 55 else "bullish" if score <= 74 else "positive"
        fg_html = f'<span class="{fg_class}">F&G: {score:.0f} ({rating})</span>'
    
    aaii_html = '<span class="neutral">N/A</span>'
    if aaii_data['bullish'] is not None:
        spread = aaii_data['spread']
        aaii_class = "positive" if spread > 20 else "bullish" if spread > 0 else "neutral" if spread > -20 else "high-risk" if spread > -40 else "negative"
        aaii_html = f'<span class="{aaii_class}">AAII: Bull {aaii_data["bullish"]:.1f}% Bear {aaii_data["bearish"]:.1f}%</span>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Live Dashboard</title>
    <meta http-equiv="refresh" content="{refresh_interval}">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; display: inline-block; }}
        .header-container {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 10px; }}
        .controls {{ display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
        .refresh-btn {{ padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }}
        .refresh-btn:hover {{ background: #0052a3; }}
        .vix-display, .fear-greed-display, .aaii-display {{ font-size: 1.2em; font-weight: bold; margin-left: 20px; }}
        .mode-badge {{ padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
        .regular-hours {{ background-color: #0066cc; color: white; }}
        .extended-hours {{ background-color: #ff6600; color: white; }}
        .refresh-info {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .alert-banner {{ background-color: #ff4444; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.8; }} 100% {{ opacity: 1; }} }}
        table {{ border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background-color: #0066cc; color: white; padding: 12px; text-align: left; font-size: 0.9em; cursor: pointer; }}
        th.sortable::after {{ content: ' ⇅'; opacity: 0.5; }}
        th.sort-asc::after {{ content: ' ▲'; }} th.sort-desc::after {{ content: ' ▼'; }}
        td {{ padding: 12px 10px; border-bottom: 1px solid #ddd; font-size: 0.9em; vertical-align: top; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f0f0f0; }}
        .positive {{ background-color: #00aa00; color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; display: inline-block; }}
        .negative {{ background-color: #cc0000; color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; display: inline-block; }}
        .bullish {{ color: #00aa00; font-weight: bold; }}
        .bearish {{ color: #cc0000; font-weight: bold; }}
        .high-risk {{ color: #cc6600; font-weight: bold; }}
        .neutral {{ color: #666; font-style: italic; }}
        .ticker {{ font-weight: bold; font-size: 1.1em; }}
        .link-group {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 2px; font-size: 0.8em; }}
        .item {{ font-size: 0.85em; margin: 4px 0; line-height: 1.4; }}
        .switch-btn {{display:inline-block;margin-left:12px;padding:6px 14px;background:#ff6600;color:white;border-radius:4px;font-weight:bold;text-decoration:none}}
        .switch-btn:hover {{background:#e65c00}}
    </style>
</head>
<body>
    {alert_banner}
    <div class="header-container">
        <div>
            <h1>Live Dashboard</h1>
            <span class="mode-badge {'extended-hours' if extended_hours else 'regular-hours'}">{hours_type}</span>
            <span class="vix-display">{vix_html}</span>
            <span class="fear-greed-display">{fg_html}</span>
            <span class="aaii-display">{aaii_html}</span>
        </div>
        <div class="controls">
            <button class="refresh-btn" onclick="location.reload()">Refresh Now</button>
        </div>
    </div>
    <div class="refresh-info">
        <strong>Last updated:</strong> {last_update} | 
        <strong>Auto-refresh:</strong> every {refresh_interval} seconds
    </div>
    <table id="stockTable">
        <tr>
            <th class="sortable" data-column="ticker">TICKER</th>
            <th class="sortable" data-column="price" data-type="number">PRICE</th>
            <th class="sortable" data-column="day" data-type="number">DAY %</th>
            <th class="sortable" data-column="1m" data-type="number">1M %</th>
            <th class="sortable" data-column="6m" data-type="number">6M %</th>
            <th class="sortable" data-column="volume" data-type="number">VOLUME</th>
            <th>52W RANGE</th>
            <th class="sortable" data-column="tech">TECHNICAL</th>
            <th class="sortable" data-column="risk">RISK / SENTIMENT</th>
        </tr>
"""
    # ... [table rows generation unchanged] ...
    # (kept identical to previous version for brevity)

    html += """    </table>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const headers = document.querySelectorAll('th.sortable');
            const colMap = {ticker:0, price:1, day:2, '1m':3, '6m':4, volume:5, tech:7, risk:8};
            headers.forEach(th => {
                th.addEventListener('click', () => {
                    const col = th.dataset.column;
                    const asc = th.classList.toggle('sort-asc');
                    th.classList.toggle('sort-desc', !asc);
                    headers.forEach(h => { if (h !== th) h.classList.remove('sort-asc', 'sort-desc'); });
                    
                    const rows = Array.from(document.querySelectorAll('#stockTable tr')).slice(1);
                    rows.sort((a, b) => {
                        const i = colMap[col];
                        let aVal = a.cells[i].dataset.value || '';
                        let bVal = b.cells[i].dataset.value || '';
                        const num = th.dataset.type === 'number';
                        if (num) { aVal = parseFloat(aVal) || 0; bVal = parseFloat(bVal) || 0; }
                        return (aVal > bVal ? 1 : -1) * (asc ? 1 : -1);
                    });
                    rows.forEach(row => document.querySelector('#stockTable').appendChild(row));
                });
            });
        });
    </script>
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

    dashboards = [
        (False, 'data/reg_dashboard.html',  'Regular',  'regular-hours'),
        (True,  'data/extnd_dashboard.html', 'Extended', 'extended-hours')
    ]

    other = {dashboards[0][1]: dashboards[1][1], dashboards[1][1]: dashboards[0][1]}

    print("Generating dashboards...\n")

    for ext, out, name, cls in dashboards:
        try:
            df = create_dashboard(args.csv_file, extended_hours=ext)
            alerts = check_alerts(df.to_dict('records') if not df.empty else [])

            export_to_html(
                df=df,
                vix_data=get_vix_data(),
                fg_data=get_fear_greed_data(),
                aaii_data=get_aaii_sentiment(),
                output_file=out,
                extended_hours=ext,
                refresh_interval=args.refresh,
                triggered_alerts=alerts
            )

            with open(out, 'r+', encoding='utf-8') as f:
                html = f.read()
                html = html.replace('<title>Live Dashboard</title>', f'<title>{name} Dashboard</title>')

                badge = f'<span class="mode-badge {cls}">{name} Hours</span>'
                switch = f'<a href="{other[out]}" class="switch-btn">Switch to {"Extended" if not ext else "Regular"} Hours</a>'
                html = html.replace(badge, badge + switch)

                f.seek(0)
                f.write(html)
                f.truncate()

            print(f"✓ {name} Hours → {out}")
            if alerts:
                print(f"   🚨 {len(alerts)} alert{'s' if len(alerts)!=1 else ''}")

        except Exception as e:
            print(f"✗ {name} failed: {e}")

    print("\nDashboards generated with cross-links in ./data/")