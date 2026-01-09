# summary.py
from __future__ import annotations

import html as _html
from datetime import datetime
from typing import Iterable, Optional, Tuple, List

import pandas as pd
import yfinance as yf

# ---------------- Formatting helpers ----------------
def fmt_num(x, d: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):.{d}f}"
    except Exception:
        return ""

def fmt_int(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{int(float(x)):,}"
    except Exception:
        return ""

def fmt_pct(x, d: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x) * 100:.{d}f}%"
    except Exception:
        return ""

def css_class_from_value(v) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        v = float(v)
        if v > 0:
            return "pos"
        if v < 0:
            return "neg"
        return ""
    except Exception:
        return ""

# ---------------- Indicators ----------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def macd(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(s, fast) - ema(s, slow)
    sgn = ema(m, sig)
    h = m - sgn
    return m, sgn, h

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def indicator_signal(row: pd.Series) -> str:
    """
    Signals based on the timeframe's bars (strict SHORT).
    """
    try:
        close = float(row.get("close", float("nan")))
        sma20 = float(row.get("sma20", float("nan")))
        sma50 = float(row.get("sma50", float("nan")))
        r = row.get("rsi14", pd.NA)
        rsi_v = float(r) if pd.notna(r) else float("nan")
        mh = row.get("macd_hist", pd.NA)
        macd_h = float(mh) if pd.notna(mh) else float("nan")
        retv = row.get("ret", pd.NA)
        ret_v = float(retv) if pd.notna(retv) else float("nan")

        if pd.isna(close) or pd.isna(sma20) or pd.isna(sma50):
            return "HOLD"

        if close > sma20 and sma20 >= sma50 and (pd.isna(rsi_v) or rsi_v < 70):
            return "BUY"

        if (
            close < sma50
            and sma20 < sma50
            and (pd.isna(rsi_v) or rsi_v < 35)
            and (pd.isna(macd_h) or macd_h < 0)
            and (pd.isna(ret_v) or ret_v < 0)
        ):
            return "SHORT"

        if close < sma50:
            return "SELL"
    except Exception:
        pass
    return "HOLD"

# ---------------- Sparklines ----------------
def sparkline_svg(vals: Iterable[float], w: int = 90, h: int = 28) -> str:
    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and pd.isna(x))]
    if len(v) < 2:
        return ""
    mn, mx = min(v), max(v)
    if mn == mx:
        mx = mn + 1e-9
    pts = []
    for i, x in enumerate(v):
        sx = (i / (len(v) - 1)) * (w - 2) + 1
        sy = (1 - (x - mn) / (mx - mn)) * (h - 2) + 1
        pts.append(f"{sx:.2f},{sy:.2f}")
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg"><polyline fill="none" '
        f'stroke="currentColor" stroke-width="1.7" points="{" ".join(pts)}"/></svg>'
    )

def _spark_trend(vals: List[float]) -> str:
    vals = [float(x) for x in vals if x is not None and not (isinstance(x, float) and pd.isna(x))]
    if len(vals) < 2:
        return "flat"
    d = vals[-1] - vals[0]
    if d > 0:
        return "up"
    if d < 0:
        return "down"
    return "flat"

# ---------------- Data pipeline ----------------
def fetch_yahoo_ohlcv(tickers, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame()
    d = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    frames = []
    if isinstance(d.columns, pd.MultiIndex):
        for t in tickers:
            if t in d.columns.levels[0]:
                x = d[t].reset_index()
                x["ticker"] = t
                frames.append(x)
    else:
        x = d.reset_index()
        x["ticker"] = tickers[0]
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df["sma20"] = df.groupby("ticker")["close"].transform(lambda s: sma(s, 20))
    df["sma50"] = df.groupby("ticker")["close"].transform(lambda s: sma(s, 50))
    df["rsi14"] = df.groupby("ticker")["close"].transform(lambda s: rsi(s, 14))

    hist_list = []
    for _, g in df.groupby("ticker", sort=False):
        _, _, h = macd(g["close"])
        hist_list.append(h)
    df["macd_hist"] = pd.concat(hist_list).sort_index()

    df["atr14"] = df.groupby("ticker").apply(lambda g: atr(g["high"], g["low"], g["close"], 14)).reset_index(level=0, drop=True)
    df["day_ret"] = df.groupby("ticker")["close"].pct_change()

    return df

def _make_tf_table(bars: pd.DataFrame, ret_col: str, ret_label: str, spark_points: int) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["ticker","close",ret_col,"sma20","sma50","rsi14","macd_hist","atr14","signal","sparkline","sparktrend"])

    last = bars.groupby("ticker").tail(1).copy()
    last = last.rename(columns={ret_col: "ret"})
    last["signal"] = last.apply(indicator_signal, axis=1)

    spark_vals = bars.groupby("ticker")["close"].apply(lambda s: s.tail(spark_points).tolist())
    last["sparktrend"] = last["ticker"].map(spark_vals.apply(_spark_trend).to_dict())
    last["sparkline"] = last["ticker"].map(spark_vals.apply(sparkline_svg).to_dict())

    out = last[["ticker","close","ret","sma20","sma50","rsi14","macd_hist","atr14","signal","sparkline","sparktrend"]].copy()
    out = out.rename(columns={"ret": ret_label})
    return out


def build_tables(df: pd.DataFrame):
    """
    Single wide table with Daily/Weekly/Monthly side-by-side.
    Display columns kept minimal (Return %, Signal, ATR + ATR%, Spark).
    Adds:
      - Consensus (based on D/W/M signals)
      - Avg ATR% (average of available timeframes)
      - Options Hint (rule-of-thumb based on Consensus + Avg ATR%)
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Daily already has indicators from add_indicators
    daily_tf = _make_tf_table(df, "day_ret", "Day %", spark_points=20)

    # Weekly (Friday close)
    weekly_rows = []
    for t, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()
        w = g.resample("W-FRI").last()
        if w.empty:
            continue
        w["ticker"] = t
        w["sma20"] = sma(w["close"], 20)
        w["sma50"] = sma(w["close"], 50)
        w["rsi14"] = rsi(w["close"], 14)
        _, _, mh = macd(w["close"])
        w["macd_hist"] = mh
        w["atr14"] = atr(w["high"], w["low"], w["close"], 14)
        w["week_ret"] = w["close"].pct_change()
        weekly_rows.append(w.reset_index())
    weekly_df = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    weekly_tf = _make_tf_table(weekly_df, "week_ret", "Week %", spark_points=16) if not weekly_df.empty else pd.DataFrame()

    # Monthly (month-end)
    monthly_rows = []
    for t, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()
        m = g.resample("M").last()
        if m.empty:
            continue
        m["ticker"] = t
        m["sma20"] = sma(m["close"], 20)
        m["sma50"] = sma(m["close"], 50)
        m["rsi14"] = rsi(m["close"], 14)
        _, _, mh = macd(m["close"])
        m["macd_hist"] = mh
        m["atr14"] = atr(m["high"], m["low"], m["close"], 14)
        m["month_ret"] = m["close"].pct_change()
        monthly_rows.append(m.reset_index())
    monthly_df = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    monthly_tf = _make_tf_table(monthly_df, "month_ret", "Month %", spark_points=12) if not monthly_df.empty else pd.DataFrame()

    # Merge into a single wide table keyed by ticker
    def _prep(tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if tf is None or tf.empty:
            return pd.DataFrame(columns=["ticker"])
        x = tf.copy()
        for col in list(x.columns):
            if col == "ticker":
                continue
            x = x.rename(columns={col: f"{prefix}{col}"})
        return x

    d = _prep(daily_tf, "D_")
    w = _prep(weekly_tf, "W_")
    m = _prep(monthly_tf, "M_")

    combined = d.merge(w, how="outer", on="ticker").merge(m, how="outer", on="ticker")

    # Helper to safely compute ATR%
    def _atr_pct(prefix: str) -> pd.Series:
        close = combined.get(f"{prefix}close", pd.Series([pd.NA] * len(combined)))
        atrv = combined.get(f"{prefix}atr14", pd.Series([pd.NA] * len(combined)))
        try:
            return (pd.to_numeric(atrv, errors="coerce") / pd.to_numeric(close, errors="coerce")).astype("float")
        except Exception:
            return pd.Series([pd.NA] * len(combined))

    combined["D_atr_pct"] = _atr_pct("D_")
    combined["W_atr_pct"] = _atr_pct("W_")
    combined["M_atr_pct"] = _atr_pct("M_")

    # Avg ATR% across available timeframes
    
    
    # Consensus signal
    def consensus_row(r: pd.Series) -> str:
        sigs = []
        for k in ("D_signal", "W_signal", "M_signal"):
            v = r.get(k, "")
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            s = str(v).strip().upper()
            if s:
                sigs.append(s)
        if not sigs:
            return ""
        # count
        from collections import Counter
        c = Counter(sigs)
        top, topn = c.most_common(1)[0]
        if topn == len(sigs):
            return top
        if topn >= 2:
            return f"{top} (2/3)"
        return "MIXED"

    combined["Consensus"] = combined.apply(consensus_row, axis=1)

    # Options hint based on Consensus + Avg ATR%
    def options_hint(r: pd.Series) -> str:
        cons = str(r.get("Consensus", "")).upper()
        atrp = r.get("Avg_atr_pct", pd.NA)
        try:
            a = float(atrp) if pd.notna(atrp) else None
        except Exception:
            a = None

        # Vol buckets (ATR as % of price)
        # tweak thresholds here if you want
        if a is None:
            vol = "unknown"
        elif a < 0.02:
            vol = "low"
        elif a < 0.04:
            vol = "mid"
        else:
            vol = "high"

        if "MIXED" in cons or cons == "":
            return "Mixed signals → defined-risk only (small) or wait"

        if cons.startswith("BUY"):
            if vol == "low":
                return "Bullish+low vol → debit call spread / calls"
            if vol == "mid":
                return "Bullish+mid vol → debit call spread / call diagonal"
            return "Bullish+high vol → put credit spread / calendar (defined risk)"

        if cons.startswith("SHORT"):
            if vol == "low":
                return "Bearish+low vol → debit put spread / puts"
            if vol == "mid":
                return "Bearish+mid vol → put diagonal / debit put spread"
            return "Bearish+high vol → call credit spread / put calendar (defined risk)"

        if cons.startswith("SELL"):
            if vol == "low":
                return "Bearish → debit put spread / puts (defined risk)"
            if vol == "mid":
                return "Bearish → call credit spread (defined risk)"
            return "Bearish+high vol → call credit spread / iron condor (wide wings)"

        if cons.startswith("HOLD"):
            if vol == "high":
                return "High vol + no edge → premium-selling only if defined risk"
            return "No clear edge → wait / keep small"

        return "Defined-risk only"

    combined["Options Hint"] = combined.apply(options_hint, axis=1)

    # ---- Format display columns ----
    def fprice(v): return fmt_num(v, 2)
    def fnum(v): return fmt_num(v, 2)
    def fret(v): return fmt_pct(v, 2)
    def fatrp(v): return fmt_pct(v, 2)

    # returns
    for col in ["D_Day %", "W_Week %", "M_Month %"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(fret)

    # price
    if "D_close" in combined.columns:
        combined["D_close"] = combined["D_close"].apply(fprice)

    # ATR values
    for col in ["D_atr14", "W_atr14", "M_atr14"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(fnum)

    # ATR percents
    for col in ["D_atr_pct", "W_atr_pct", "M_atr_pct", "Avg_atr_pct"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(fatrp)

    # Rename columns to user-friendly headings
    rename = {
        "ticker": "Ticker",
        "D_close": "Close",
        "D_Day %": "Day %",
        "W_Week %": "Week %",
        "M_Month %": "Month %",

        "D_signal": "D Signal",
        "W_signal": "W Signal",
        "M_signal": "M Signal",
    
        "D_sparkline": "D Spark",
        "W_sparkline": "W Spark",
        "M_sparkline": "M Spark",

        "D_sparktrend": "D Trend",
        "W_sparktrend": "W Trend",
        "M_sparktrend": "M Trend",
    }
    combined = combined.rename(columns=rename)

    # Ensure columns order (minimal)
    cols = [
        "Ticker","Close","Day %","Week %","Month %",
        "D Signal","W Signal","M Signal","Consensus",
                "D Spark","W Spark","M Spark",
        "Options Hint",
        "D Trend","W Trend","M Trend"
    ]
    cols = [c for c in cols if c in combined.columns]
    combined = combined[cols].copy()

    try:
        asof = df["date"].max().strftime("%Y-%m-%d")
    except Exception:
        asof = None

    # Compatibility with stocks.py: return combined as "daily", empty as "monthly"
    return combined, pd.DataFrame(), asof

# ---------------- HTML ----------------
_CSS = r"""
:root{
  --bg:#f7f8fa; --surface:#fff; --border:#dfe3e8;
  --text:#1f2937; --muted:#6b7280;
  --brand:#c74634;
  --buy:#1f7a1f; --sell:#b42318; --short:#6f2dbd;
  --radius:14px;
  --shadow:0 1px 2px rgba(16,24,40,.06), 0 2px 8px rgba(16,24,40,.06);
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.page{max-width:100%;margin:0;padding:12px 12px 24px;}
.header{display:flex;gap:14px;align-items:flex-start;justify-content:flex-start;flex-wrap:wrap;margin-bottom:14px;}
.hgroup{max-width:100%;}
.hgroup h1{margin:0;font-size:22px;}
.hgroup .meta{margin-top:6px;color:var(--muted);font-size:13px;}
.brand-dot{display:inline-block;width:10px;height:10px;border-radius:999px;background:var(--brand);
  margin-right:8px;vertical-align:middle;}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.controls input{padding:11px 14px;border:1px solid var(--border);border-radius:999px;background:var(--surface);
  font-size:16px;min-width:260px;outline:none;}
.controls input:focus{border-color:rgba(199,70,52,.55);box-shadow:0 0 0 4px rgba(199,70,52,.12);}
.controls .hint{color:var(--muted);font-size:12.5px;}
.card{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:12px;margin-top:12px;}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;}
.card-head h2{margin:0;font-size:16px;}
.table-wrap{overflow:visible;}
table.data-table{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden;}
thead th{position:sticky;top:0;background:#fbfbfc;color:#111827;font-weight:800;font-size:12.5px;
  letter-spacing:.2px;cursor:pointer;user-select:none;border-bottom:1px solid var(--border);}
th,td{padding:10px 12px;border-bottom:1px solid var(--border);white-space:nowrap;font-size:13.5px;}
td.hint{white-space:normal;min-width:260px;max-width:460px;line-height:1.25;}

tbody tr:nth-child(even){background:rgba(17,24,39,.02);}
tbody tr:hover{background:rgba(199,70,52,.06);}
td.num{text-align:right;font-variant-numeric:tabular-nums;}
td.pos{color:var(--buy);font-weight:700;}
td.neg{color:var(--sell);font-weight:700;}
.badge{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-weight:800;font-size:12px;border:1px solid rgba(17,24,39,.08);}
.badge:before{content:"";width:8px;height:8px;border-radius:999px;background:currentColor;opacity:.9;}
.badge.buy{color:var(--buy);background:rgba(31,122,31,.08);}
.badge.sell{color:var(--sell);background:rgba(180,35,24,.08);}
.badge.short{color:var(--short);background:rgba(111,45,189,.08);}
.badge.hold{color:var(--muted);background:rgba(107,114,128,.08);}
td.spark svg{display:block;}
td.spark.trend-up{color:var(--buy);}
td.spark.trend-down{color:var(--sell);}
td.spark.trend-flat{color:var(--muted);}


.footer{margin-top:14px;color:var(--muted);font-size:12.5px;}
@media (max-width: 640px){
  .header{flex-direction:column;align-items:stretch;gap:10px;}
  .controls input{width:100%;min-width:0;}
  .controls .hint{display:none;}
  .table-wrap{overflow-x:auto;}
  table.data-table{min-width:1200px;}
}
"""

_JS = r"""
function sortTable(id, col) {
  const table = document.getElementById(id);
  if (!table) return;
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = table.getAttribute("data-sort") !== "asc";

  rows.sort((a,b)=>{
    const aCell = a.cells[col];
    const bCell = b.cells[col];
    let A = (aCell.getAttribute('data-sort') || aCell.innerText).trim();
    let B = (bCell.getAttribute('data-sort') || bCell.innerText).trim();

    const pA = A.endsWith('%') ? parseFloat(A.slice(0,-1)) : NaN;
    const pB = B.endsWith('%') ? parseFloat(B.slice(0,-1)) : NaN;
    if (!isNaN(pA) && !isNaN(pB)) return asc ? pA - pB : pB - pA;

    const nA = parseFloat(A.replace(/,/g,''));
    const nB = parseFloat(B.replace(/,/g,''));
    if (!isNaN(nA) && !isNaN(nB)) return asc ? nA - nB : nB - nA;

    return asc ? A.localeCompare(B) : B.localeCompare(A);
  });

  rows.forEach(r => tbody.appendChild(r));
  table.setAttribute("data-sort", asc ? "asc" : "desc");
}

function applyFilter() {
  const q = (document.getElementById('tickerFilter')?.value || '').trim().toUpperCase();
  const table = document.getElementById('mainTable');
  if (!table) return;
  Array.from(table.tBodies[0].rows).forEach(r => {
    const t = (r.getAttribute('data-ticker') || '').toUpperCase();
    r.style.display = (!q || t.includes(q)) ? '' : 'none';
  });
}

function decorateMain() {
  const table = document.getElementById('mainTable');
  if (!table) return;
  const headers = Array.from(table.tHead.rows[0].cells).map(th => th.innerText.trim());
  const idxTicker = headers.indexOf("Ticker");

  const signalCols = headers
    .map((h,i)=>({h,i}))
    .filter(x => x.h.endsWith("Signal") || x.h === "Consensus")
    .map(x => x.i);

  const sparkCols = headers
    .map((h,i)=>({h,i}))
    .filter(x => x.h.endsWith("Spark"))
    .map(x => x.i);

  const trendCols = headers
    .map((h,i)=>({h,i}))
    .filter(x => x.h.endsWith("Trend"))
    .map(x => x.i);

  Array.from(table.tBodies[0].rows).forEach(r => {
    if (idxTicker >= 0) r.setAttribute('data-ticker', r.cells[idxTicker].innerText.trim().toUpperCase());

    // badges for each signal cell
    signalCols.forEach(ci => {
      const sig = r.cells[ci].innerText.trim().toUpperCase();
      let cls = "badge hold";
      if (sig === "BUY") cls = "badge buy";
      else if (sig === "SELL") cls = "badge sell";
      else if (sig === "SHORT") cls = "badge short";
      r.cells[ci].innerHTML = `<span class="${cls}">${sig || "—"}</span>`;
    });

    // sparkline coloring using trend cols (aligned by timeframe order)
    // We expect order: D Spark, W Spark, M Spark and D Trend, W Trend, M Trend
    for (let k=0; k<sparkCols.length; k++) {
      const sCol = sparkCols[k];
      const tCol = trendCols[k];
      const trend = (tCol != null && r.cells[tCol]) ? r.cells[tCol].innerText.trim().toLowerCase() : "flat";
      r.cells[sCol].classList.add("spark");
      r.cells[sCol].classList.add("trend-" + (trend || "flat"));
    }
  });

  // Hide trend columns (used only for coloring)
  trendCols.forEach(ci => {
    table.tHead.rows[0].cells[ci].style.display = "none";
    Array.from(table.tBodies[0].rows).forEach(r => r.cells[ci].style.display = "none");
  });
}

document.addEventListener('DOMContentLoaded', () => {
  // default sort: Consensus → Close
  try {
    const table = document.getElementById('mainTable');
    if (table) {
      const headers = Array.from(table.tHead.rows[0].cells).map(th=>th.innerText.trim());
      const cIdx = headers.indexOf('Consensus');
      const closeIdx = headers.indexOf('Close');
      if (cIdx >= 0 && closeIdx >= 0) {
        const priority = { 'BUY':0, 'BUY (2/3)':0, 'SHORT':1, 'SHORT (2/3)':1, 'HOLD':2, 'HOLD (2/3)':2, 'SELL':3, 'SELL (2/3)':3, 'MIXED':9, '':9 };
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);
        rows.sort((a,b)=>{
          const A = (a.cells[cIdx].innerText || '').trim().toUpperCase();
          const B = (b.cells[cIdx].innerText || '').trim().toUpperCase();
          const pA = (priority[A] ?? 9);
          const pB = (priority[B] ?? 9);
          if (pA !== pB) return pA - pB;
          const cA = parseFloat((a.cells[closeIdx].innerText||'').replace(/,/g,'')) || -Infinity;
          const cB = parseFloat((b.cells[closeIdx].innerText||'').replace(/,/g,'')) || -Infinity;
          return cB - cA;
        });
        rows.forEach(r=>tbody.appendChild(r));
      }
    }
  } catch (e) {}

  const f = document.getElementById('tickerFilter');
  if (f) f.addEventListener('input', applyFilter);
  decorateMain();
  applyFilter();
});
"""

def _render_main_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p style='color:#6b7280;margin:0;'>No data available</p>"

    cols = [c for c in list(df.columns) if not c.endswith('__sort')]
    thead = "<thead><tr>" + "".join(
        "<th onclick=\"sortTable('mainTable',%d)\">%s</th>" % (i, _html.escape(c))
        for i, c in enumerate(cols)
    ) + "</tr></thead>"

    def is_num_col(name: str) -> bool:
        n = name.strip().lower()
        return (
            n.endswith("%") or n.endswith("sma20") or n.endswith("sma50")
            or n.endswith("rsi14") or n.endswith("macd") or n.endswith("atr") or n == "close"
        )

    body_rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r.get(c, "")
            if c.endswith("Spark"):
                tds.append("<td>%s</td>" % (v or ""))
                continue
            if c == "Options Hint":
                val = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                tds.append("<td class='hint'>%s</td>" % _html.escape(val))
                continue

            val = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            cls = []
            data_sort = ""
            if c == "Avg ATR%" and ("Avg ATR%__sort" in df.columns):
                try:
                    sv = df.loc[_, "Avg ATR%__sort"]
                    if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                        data_sort = " data-sort='%s'" % _html.escape(f"{float(sv)*100:.6f}")
                except Exception:
                    data_sort = ""
            if is_num_col(c):
                cls.append("num")
            if c.endswith("%"):
                try:
                    num = float(val.replace("%", ""))
                    cls.append(css_class_from_value(num))
                except Exception:
                    pass
            td_cls = (" class='%s'" % " ".join([x for x in cls if x])) if cls else ""
            tds.append("<td%s%s>%s</td>" % (data_sort, td_cls, _html.escape(val)))

        body_rows.append("<tr>%s</tr>" % "".join(tds))

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return "<div class='table-wrap'><table id='mainTable' class='data-table'>%s%s</table></div>" % (thead, tbody)

def render_html(daily_df: pd.DataFrame, _unused, asof: Optional[str], tickers: str) -> str:
    asof = asof or datetime.now().strftime("%Y-%m-%d")
    head = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Stock Summary Dashboard</title>"
        "<style>" + _CSS + "</style>"
        "<script>" + _JS + "</script>"
        "</head>"
    )
    body = (
        "<body><div class='page'>"
        "<div class='header'>"
        "<div class='hgroup'>"
        "<h1><span class='brand-dot'></span>Summary Dashboard</h1>"
        "<div class='meta'><b>As-of:</b> " + _html.escape(asof) + "</div>"
        "</div>"
        "<div class='controls'>"
        "<input id='tickerFilter' type='text' placeholder='Filter tickers (e.g., AAPL)' />"
        "<span class='hint'>Click a column header to sort</span>"
        "</div>"
        "</div>"
        "<div class='card'>"
        "<div class='card-head'><h2>Daily • Weekly • Monthly</h2></div>"
        + _render_main_table(daily_df)
        + "</div>"
        "<div class='footer'>Signals are heuristic. Validate with your own risk/IV filters.</div>"
        "</div></body></html>"
    )
    return head + body

__all__ = [
    "fetch_yahoo_ohlcv",
    "add_indicators",
    "build_tables",
    "render_html",
    "ema",
    "sma",
    "rsi",
    "macd",
    "atr",
    "fmt_pct",
    "fmt_num",
    "fmt_int",
    "css_class_from_value",
    "indicator_signal",
    "sparkline_svg",
]
