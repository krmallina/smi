#!/usr/bin/env python3
# stockSummary.py
"""
Static HTML dashboard (daily + weekly) sourced from Yahoo Finance for a fixed ticker list.

Install:
  pip install yfinance pandas numpy

Run:
  python stockSummary.py --out data/summary.html --period 1y
  python stockSummary.py --out data/summary.html --period 2y --interval 1d
  python stockSummary.py --out data/summary.html --tickers AAPL,MSFT,SPY,QQQ

Output:
  data/summary.html (static file you can open in a browser)
"""

import argparse
import html
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")


# ----------------------------
# Fixed ticker list (edit as desired)
# ----------------------------
M7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
TOP_ETFS = ["SPY", "QQQ", "VTI", "VOO", "IVV"]

# Mutual funds typically don't have options; use ETF proxies.
FUND_PROXIES = {
    "VTSAX": "VTI",
    "VFIAX": "VOO",
    "VTIAX": "VXUS",
    "FXAIX": "IVV",
    "VBTLX": "BND",
}

DEFAULT_TICKERS = sorted(set(M7 + TOP_ETFS + list(FUND_PROXIES.values())))


# ----------------------------
# Technical Indicators
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ----------------------------
# Formatting helpers
# ----------------------------
def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:+.2f}%"

def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.{digits}f}"

def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"

def css_class_from_value(x: float) -> str:
    if pd.isna(x):
        return "neutral"
    return "positive" if x >= 0 else "negative"

def indicator_signal(rsi_v: float, macd_hist_v: float, close_v: float, sma50_v: float, sma200_v: float) -> str:
    """
    Simple heuristic signal:
      BUY: close > sma50 > sma200 AND MACD_hist > 0 AND RSI in 45–70
      SHORT: close < sma50 < sma200 AND MACD_hist < 0 AND RSI < 55
      SELL: RSI >= 70 (overbought)
      BUY: RSI <= 30 (oversold)
      else HOLD
    """
    if any(pd.isna(v) for v in [rsi_v, macd_hist_v, close_v, sma50_v, sma200_v]):
        return "HOLD"

    bullish = (close_v > sma50_v > sma200_v) and (macd_hist_v > 0) and (45 <= rsi_v <= 70)
    bearish = (close_v < sma50_v < sma200_v) and (macd_hist_v < 0) and (rsi_v < 55)

    if bullish:
        return "BUY"
    if bearish:
        return "SHORT"
    if rsi_v >= 70:
        return "SELL"
    if rsi_v <= 30:
        return "BUY"
    return "HOLD"

def sparkline_svg(values: Iterable[float], width: int = 90, height: int = 28) -> str:
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ""
    vmin, vmax = float(min(vals)), float(max(vals))
    if np.isclose(vmin, vmax):
        vmin -= 1
        vmax += 1

    xs = np.linspace(0, width, num=len(vals))
    ys = [(height - 2) - ((v - vmin) / (vmax - vmin)) * (height - 4) for v in vals]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    color = "#2dd4bf" if vals[-1] >= vals[0] else "#fb7185"

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">
  <polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />
</svg>"""


# ----------------------------
# Yahoo Finance data fetch
# ----------------------------
def fetch_yahoo_ohlcv(tickers, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Returns long-form dataframe with columns:
      date,ticker,open,high,low,close,volume
    """
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if data is None or data.empty:
        raise ValueError("No data returned from Yahoo Finance. Try again later or reduce tickers.")

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if ("Close", t) not in data.columns:
                continue
            frames.append(pd.DataFrame({
                "date": data.index,
                "ticker": t,
                "open": data.get(("Open", t)),
                "high": data.get(("High", t)),
                "low": data.get(("Low", t)),
                "close": data.get(("Close", t)),
                "volume": data.get(("Volume", t)),
            }))
        df = pd.concat(frames, ignore_index=True)
    else:
        # single ticker
        t = tickers[0]
        df = pd.DataFrame({
            "date": data.index,
            "ticker": t,
            "open": data["Open"] if "Open" in data else data["Close"],
            "high": data["High"] if "High" in data else data["Close"],
            "low": data["Low"] if "Low" in data else data["Close"],
            "close": data["Close"],
            "volume": data["Volume"] if "Volume" in data else np.nan,
        })

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ----------------------------
# Indicators + daily/weekly tables
# ----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in df.groupby("ticker", sort=False):
        g = g.copy().sort_values("date")
        g["ret_1d"] = g["close"].pct_change()
        g["sma20"] = sma(g["close"], 20)
        g["sma50"] = sma(g["close"], 50)
        g["sma200"] = sma(g["close"], 200)
        g["rsi14"] = rsi(g["close"], 14)
        _, _, hist = macd(g["close"])
        g["macd_hist"] = hist
        g["atr14"] = atr(g["high"], g["low"], g["close"], 14)
        out.append(g)
    return pd.concat(out, axis=0).reset_index(drop=True)

def build_tables(df: pd.DataFrame):
    latest_date = df["date"].max()

    daily_rows = []
    weekly_rows = []

    for t, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        last = g.iloc[-1]

        sig = indicator_signal(
            last["rsi14"], last["macd_hist"], last["close"], last["sma50"], last["sma200"]
        )

        daily_rows.append({
            "Ticker": t,
            "Close": float(last["close"]),
            "Day %": float(last["ret_1d"]) if pd.notna(last["ret_1d"]) else np.nan,
            "Volume": float(last["volume"]) if pd.notna(last["volume"]) else np.nan,
            "RSI14": float(last["rsi14"]) if pd.notna(last["rsi14"]) else np.nan,
            "MACD_hist": float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else np.nan,
            "SMA50": float(last["sma50"]) if pd.notna(last["sma50"]) else np.nan,
            "SMA200": float(last["sma200"]) if pd.notna(last["sma200"]) else np.nan,
            "ATR14": float(last["atr14"]) if pd.notna(last["atr14"]) else np.nan,
            "Signal": sig,
            "Sparkline": sparkline_svg(g["close"].tail(20).tolist()),
        })

        last5 = g.tail(5)
        if len(last5) >= 2:
            wk_ret = (last5["close"].iloc[-1] / last5["close"].iloc[0]) - 1.0
            weekly_rows.append({
                "Ticker": t,
                "Close": float(last5["close"].iloc[-1]),
                "Week %": float(wk_ret),
                "RSI14": float(last["rsi14"]) if pd.notna(last["rsi14"]) else np.nan,
                "MACD_hist": float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else np.nan,
                "SMA50": float(last["sma50"]) if pd.notna(last["sma50"]) else np.nan,
                "SMA200": float(last["sma200"]) if pd.notna(last["sma200"]) else np.nan,
                "ATR14": float(last["atr14"]) if pd.notna(last["atr14"]) else np.nan,
                "Signal": sig,
                "Sparkline": sparkline_svg(last5["close"].tolist()),
            })

    return pd.DataFrame(daily_rows), pd.DataFrame(weekly_rows), latest_date


# ----------------------------
# HTML rendering
# ----------------------------
def render_html(daily: pd.DataFrame, weekly: pd.DataFrame, asof_date: pd.Timestamp, ticker_list: str) -> str:
    sig_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "SHORT": 3}

    def table_html(df: pd.DataFrame, pct_col: str, table_id: str) -> str:
        if df.empty:
            return "<p class='meta'>No data.</p>"

        df2 = df.copy()
        df2["__sig"] = df2["Signal"].map(sig_order).fillna(9)
        df2 = df2.sort_values(["__sig", pct_col], ascending=[True, False]).drop(columns=["__sig"])

        cols = list(df2.columns)
        headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)

        rows = []
        for _, r in df2.iterrows():
            tds = []
            for c in cols:
                v = r[c]
                if c == pct_col:
                    cls = css_class_from_value(v)
                    tds.append(f"<td class='{cls}'>{fmt_pct(v)}</td>")
                elif c in ("Close", "SMA50", "SMA200", "ATR14"):
                    tds.append(f"<td>{fmt_num(v)}</td>")
                elif c == "Volume":
                    tds.append(f"<td>{fmt_int(v)}</td>")
                elif c in ("RSI14", "MACD_hist"):
                    tds.append(f"<td>{fmt_num(v, 2)}</td>")
                elif c == "Signal":
                    sig = str(v)
                    sig_cls = {"BUY":"sig-buy","SELL":"sig-sell","SHORT":"sig-short","HOLD":"sig-hold"}.get(sig, "sig-hold")
                    tds.append(f"<td class='{sig_cls}'>{html.escape(sig)}</td>")
                elif c == "Sparkline":
                    tds.append(f"<td class='spark'>{v if isinstance(v, str) else ''}</td>")
                else:
                    tds.append(f"<td>{html.escape(str(v))}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")

        return f"""
        <table id="{table_id}">
          <thead><tr>{headers}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        """.strip()

    daily_table = table_html(daily, "Day %", "dailyTable")
    weekly_table = table_html(weekly, "Week %", "weeklyTable")

    asof_str = asof_date.strftime("%Y-%m-%d")
    gen_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Stock Summary Dashboard</title>
<style>
  :root {{
    --bg:#0b1220; --card:#0f1b33; --text:#e8eefc; --muted:#a9b6d3;
    --border:#223153; --pos:#2dd4bf; --neg:#fb7185; --warn:#fbbf24;
    --accent:#60a5fa; --shadow: 0 10px 25px rgba(0,0,0,0.35);
  }}
  body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
         background:var(--bg); color:var(--text); }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
  .top {{ display:flex; gap:12px; align-items:baseline; justify-content:space-between; flex-wrap:wrap; }}
  .title h1 {{ margin:0; font-size: 20px; letter-spacing:0.2px; }}
  .meta {{ color:var(--muted); font-size: 12px; }}
  .tabs {{ display:flex; gap:10px; margin: 12px 0 14px; }}
  .tabbtn {{
    background: transparent; color: var(--text); border:1px solid var(--border);
    padding: 8px 12px; border-radius: 10px; cursor:pointer; font-weight: 600;
  }}
  .tabbtn.active {{ background: var(--accent); border-color: transparent; color:#0b1220; }}
  .panel {{ display:none; }}
  .panel.active {{ display:block; }}
  .card {{
    background:var(--card); border:1px solid var(--border); border-radius: 14px;
    box-shadow: var(--shadow); overflow:hidden;
  }}
  .card-h {{ padding: 12px 14px; border-bottom: 1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; }}
  .card-h .h {{ font-weight:800; }}
  .card-h .sub {{ color:var(--muted); font-size:12px; }}
  .card-b {{ padding: 8px 10px 14px; overflow:auto; }}
  table {{ width:100%; border-collapse:separate; border-spacing:0; min-width: 1100px; }}
  thead th {{
    position: sticky; top: 0; z-index: 1;
    background: #122446; color: var(--text); font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.6px; padding: 10px 10px; border-bottom: 1px solid var(--border);
  }}
  tbody td {{
    padding: 10px 10px; border-bottom: 1px solid rgba(34,49,83,0.55);
    color: var(--text); font-size: 13px; vertical-align: top;
  }}
  tbody tr:hover td {{ background: rgba(96,165,250,0.08); }}
  .positive {{ color: var(--pos); font-weight:700; }}
  .negative {{ color: var(--neg); font-weight:700; }}
  .neutral {{ color: var(--muted); }}
  .sig-buy {{ color: var(--pos); font-weight: 900; }}
  .sig-sell {{ color: var(--warn); font-weight: 900; }}
  .sig-short {{ color: var(--neg); font-weight: 900; }}
  .sig-hold {{ color: var(--muted); font-weight: 900; }}
  .spark svg {{ display:block; }}
  .footer {{ margin-top: 12px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="title">
        <h1>📊 Stock Summary Dashboard</h1>
        <div class="meta">As-of data date: <b>{asof_str}</b> • Generated: {gen_str}</div>
        <div class="meta">Tickers: {html.escape(ticker_list)}</div>
      </div>
      <div class="meta">Static HTML • Daily / Weekly tabs</div>
    </div>

    <div class="tabs">
      <button class="tabbtn active" data-tab="daily">Daily Summary</button>
      <button class="tabbtn" data-tab="weekly">Weekly Summary</button>
    </div>

    <div class="panel active" id="panel-daily">
      <div class="card">
        <div class="card-h">
          <div class="h">Daily Snapshot</div>
          <div class="sub">Latest day % + RSI/MACD + SMA50/200 + ATR + 20D sparkline</div>
        </div>
        <div class="card-b">
          {daily_table}
        </div>
      </div>
    </div>

    <div class="panel" id="panel-weekly">
      <div class="card">
        <div class="card-h">
          <div class="h">Weekly Summary</div>
          <div class="sub">Last 5 trading days % + indicator snapshot + 5D sparkline</div>
        </div>
        <div class="card-b">
          {weekly_table}
        </div>
      </div>
    </div>

    <div class="footer">
      Signals are heuristic (trend + MACD histogram + RSI). Adjust thresholds for your system.
    </div>
  </div>

<script>
  const btns = document.querySelectorAll('.tabbtn');
  function activate(tab) {{
    btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('panel-daily').classList.toggle('active', tab === 'daily');
    document.getElementById('panel-weekly').classList.toggle('active', tab === 'weekly');
  }}
  btns.forEach(b => b.addEventListener('click', () => activate(b.dataset.tab)));
</script>
</body>
</html>
"""


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/summary.html", help="Output HTML file (default: data/summary.html)")
    ap.add_argument("--period", default="1y", help="Yahoo period, e.g. 6mo, 1y, 2y, 5y, max")
    ap.add_argument("--interval", default="1d", help="Yahoo interval, e.g. 1d, 1wk")
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                    help="Comma-separated tickers (optional override)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers specified.")

    df = fetch_yahoo_ohlcv(tickers, period=args.period, interval=args.interval)
    df = add_indicators(df)
    daily, weekly, asof = build_tables(df)

    ticker_list = ", ".join(tickers)
    out_html = render_html(daily, weekly, asof, ticker_list)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")

    print(f"✅ Wrote {out_path} (as-of: {asof.strftime('%Y-%m-%d')})")


if __name__ == "__main__":
    main()