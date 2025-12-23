import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import argparse
from scipy.stats import norm
import numpy as np

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0), 1 if S > K else 0, 0, 0, 0
        else:
            return max(K - S, 0), -1 if S < K else 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    theta /= 365
    vega /= 100
    rho /= 100

    return delta, gamma, theta, vega, rho

def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.info.get('regularMarketPreviousClose', 4.5) / 100
        return rate
    except:
        return 0.045

def get_options_data(ticker):
    try:
        t = yf.Ticker(ticker.upper())
        underlying_price = t.info.get('regularMarketPrice') or t.info.get('previousClose')
        if not underlying_price:
            return None, None, None, f"No price for {ticker}"

        options = t.options
        if not options:
            return None, None, None, f"No options for {ticker}"

        r = get_risk_free_rate()
        all_chains = []

        for exp in options:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls.copy()
                puts = chain.puts.copy()

                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                T = max((exp_date - datetime.now()).days / 365.0, 0.001)

                for df, opt_type in [(calls, 'call'), (puts, 'put')]:
                    if df.empty: continue
                    df['type'] = opt_type.title()
                    df['expiration'] = exp
                    greeks = df.apply(
                        lambda row: black_scholes_greeks(
                            S=underlying_price,
                            K=row['strike'],
                            T=T,
                            r=r,
                            sigma=max(row['impliedVolatility'], 0.01),
                            option_type=opt_type
                        ), axis=1
                    )
                    df[['delta', 'gamma', 'theta', 'vega', 'rho']] = pd.DataFrame(greeks.tolist(), index=df.index)
                    all_chains.append(df)
            except:
                continue

        if not all_chains:
            return options, None, None, "No valid chain data"

        full_df = pd.concat(all_chains, ignore_index=True)
        nearest_exp = options[0]
        default_chain = full_df[full_df['expiration'] == nearest_exp]

        return options, default_chain, full_df, None
    except Exception as e:
        return None, None, None, f"Error: {e}"

def generate_html(ticker, expirations, default_chain, full_df, error=None):
    update = datetime.now().strftime('%I:%M:%S %p PST on %B %d, %Y')

    exp_options = ""
    if expirations:
        exp_options = "".join([f'<option value="{exp}">{exp}</option>' for exp in expirations])

    rows = ""
    if default_chain is not None and not default_chain.empty:
        for _, r in default_chain.iterrows():
            opt_type = r['type']
            cls = "call" if opt_type == 'Call' else "put"
            strike = r['strike']
            last = r.get('lastPrice', 0.0)
            bid = r['bid']
            ask = r['ask']
            vol_raw = r.get('volume', 0)
            vol = int(vol_raw) if pd.notna(vol_raw) and vol_raw > 0 else 0
            oi_raw = r.get('openInterest', 0)
            oi = int(oi_raw) if pd.notna(oi_raw) and oi_raw > 0 else 0
            iv = r['impliedVolatility'] * 100
            delta = r['delta']
            gamma = r['gamma']
            theta = r['theta']
            vega = r['vega']
            rho = r['rho']

            rows += f"""
            <tr class="{cls}">
                <td>{opt_type}</td>
                <td data-sort="{strike}">{strike:.2f}</td>
                <td>{last:.2f}</td>
                <td>{bid:.2f}</td>
                <td>{ask:.2f}</td>
                <td data-sort="{vol}">{vol if vol > 0 else '-'}</td>
                <td data-sort="{oi}">{oi if oi > 0 else '-'}</td>
                <td data-sort="{iv}">{iv:.1f}%</td>
                <td data-sort="{delta:.4f}">{delta:+.3f}</td>
                <td data-sort="{gamma:.4f}">{gamma:.3f}</td>
                <td data-sort="{theta:.4f}">{theta:+.3f}</td>
                <td data-sort="{vega:.4f}">{vega:.2f}</td>
                <td data-sort="{rho:.4f}">{rho:+.3f}</td>
            </tr>
            """

    error_msg = f"<div style='color:red;font-weight:bold;text-align:center;margin:20px;'>{error}</div>" if error else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Options - {ticker.upper()}</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
body{{font-family:Arial;margin:20px;background:#f5f5f5}}
h1{{text-align:center}}
.controls{{text-align:center;margin:30px 0}}
select{{padding:10px 15px;font-size:1.1em;border-radius:6px}}
table{{width:100%;border-collapse:collapse;background:white;margin-top:20px;box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
th{{background:#0066cc;color:white;padding:12px;cursor:pointer}}
th::after{{content:" ⇅";opacity:0.7}}
td{{padding:10px;text-align:center;border-bottom:1px solid #eee}}
.call{{background:#f0fff0}}
.put{{background:#fff0f0}}
tr:hover{{background:#f8f8f8}}
</style></head><body>
<h1>Options Dashboard — {ticker.upper()}</h1>
<div class="controls">
    <strong>Expiration:</strong>
    <select>{exp_options}</select>
</div>
{error_msg}
<div style="text-align:center;margin:20px 0"><strong>Last updated:</strong> {update}</div>
<table>
<tr>
    <th>Type</th><th>Strike</th><th>Last</th><th>Bid</th><th>Ask</th><th>Volume</th><th>OI</th><th>IV %</th>
    <th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th><th>Rho</th>
</tr>
{rows}
</table>
<script>
var headers = document.querySelectorAll('th');
headers.forEach(function(header, index) {{
    header.addEventListener('click', function() {{
        var table = header.closest('table');
        var rows = Array.from(table.querySelectorAll('tr')).slice(1);
        var ascending = header.classList.toggle('asc');
        rows.sort(function(rowA, rowB) {{
            var cellA = rowA.cells[index];
            var cellB = rowB.cells[index];
            var a = cellA.dataset.sort || cellA.innerText.trim();
            var b = cellB.dataset.sort || cellB.innerText.trim();
            a = isNaN(a) || a === '-' ? a : parseFloat(a);
            b = isNaN(b) || b === '-' ? b : parseFloat(b);
            if (a === b) return 0;
            return (a > b ? 1 : -1) * (ascending ? 1 : -1);
        }});
        rows.forEach(function(row) {{ table.appendChild(row); }});
    }});
}});
</script>
</body></html>"""

    os.makedirs('data/options', exist_ok=True)
    path = f'data/options/{ticker.upper()}.html'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

if __name__ == "__main__":
    csv_path = 'data/tickers.csv'
    try:
        tickers = pd.unique(pd.read_csv(csv_path).iloc[:,0]).tolist()
        print(f"Loaded {len(tickers)} tickers from {csv_path}")
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        tickers = ['SPY', 'AAPL', 'NVDA', 'TSLA']

    generated = []
    skipped = []

    for ticker in tickers:
        expirations, default_chain, full_df, error = get_options_data(ticker)
        if error:
            skipped.append(f"{ticker.upper()}: {error}")
            continue
        path = generate_html(ticker, expirations, default_chain, full_df, error)
        generated.append((ticker.upper(), path))

    if skipped:
        print("\nSkipped tickers:")
        for s in skipped:
            print(f"  ⚠ {s}")

    index_rows = "".join([f"<tr><td><a href='options/{t}.html'>{t}</a></td></tr>" for t, _ in generated])
    index_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Options Dashboards Index</title>
<style>body{{font-family:Arial;margin:40px;background:#f5f5f5}}
h1{{text-align:center}} table{{width:60%;margin:auto;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
td{{padding:15px;font-size:1.2em;text-align:center;border-bottom:1px solid #ddd}}
a{{text-decoration:none;color:#0066cc}} a:hover{{text-decoration:underline}}
</style></head><body>
<h1>Options Dashboards ({len(generated)} tickers)</h1>
<table>{index_rows}</table>
<p style="text-align:center;margin-top:40px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M %p PST')}</p>
</body></html>"""

    with open('data/options_dashboard_index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

    print(f"\n✓ Generated {len(generated)} options dashboards")
    print("→ Open data/options_dashboard_index.html")