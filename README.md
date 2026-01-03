## Quick Links

* [Tracker](https://krmallina.github.io/smi/data/reg_dashboard.html), [News](https://tradingeconomics.com/stream)
* [US 100](https://tradingeconomics.com/us100:ind), [Calendar](https://tradingeconomics.com/calendar), [Sectors](https://www.ssga.com/us/en/intermediary/resources/sector-tracker#currentTab=dayFive)
* [Market Movers](https://www.slickcharts.com/market-movers), [SA: Movers](https://stockanalysis.com/markets/gainers/), [SA: Trending](https://stockanalysis.com/trending), [Heat Map](https://stockanalysis.com/markets/heatmap/?time=1W), [MS: Markets](https://www.morningstar.com/markets)
* [Flows](https://www.trackinsight.com/en), [MS: Flows](https://www.google.com/search?q=https://www.morningstar.com/topics/fund-flows), [SPDR: Flows](https://www.ssga.com/us/en/intermediary/insights/a-feast-of-etf-inflows-and-returns), [ETF.com](https://www.etf.com/sections/daily-etf-flows), [ETFdb](https://etfdb.com/etf-fund-flows/#issuer=blackrock-inc)
* [Fear & Greed Index](https://www.cnn.com/markets/fear-and-greed), [AAII Sentiment](https://www.aaii.com/sentiment-survey)
* [Trading Strategies](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)


![cf48c501-9030-4cec-9dde-cf5cc067dbe1](https://github.com/user-attachments/assets/20251d2d-fdf2-4197-b166-d091e752e3ed)

# Stock Market Intelligence Dashboard

A comprehensive Python-based stock market dashboard with advanced trading signals, real-time data, and interactive visualizations.

## Features

### 📊 Multi-View Dashboard
- **Table View**: Sortable columns with detailed metrics and sparklines
- **Card View**: Rich card-based layout with visual indicators and trend arrows
- **Heatmap View**: Color-coded performance visualization with compact metrics

### 🎯 Trading Signal Framework
Six sophisticated trading strategies with visual indicators (🟢 BUY, 🟠 SELL, 🔴 SHORT):
- **Bollinger Bands (BB)**: Price channel breakout detection
- **RSI**: Oversold/overbought momentum analysis
- **MACD**: Trend following with crossover signals
- **Ichimoku Cloud**: Multi-component trend and support/resistance
- **Combined Strategy**: Requires 2+ strategies to agree
- **BB + Ichimoku (Default)**: OR logic between BB and Ichimoku signals

### 📈 Predicted Trend Indicators
Color-coded trend arrows based on multi-factor technical analysis:
- **↑** Strong uptrend (green) - Multiple bullish indicators aligned
- **↗** Moderate uptrend (green) - Bullish bias detected
- **→** Neutral/sideways (gray) - Mixed or weak signals
- **↘** Moderate downtrend (red) - Bearish bias detected
- **↓** Strong downtrend (red) - Multiple bearish indicators aligned

Trend prediction weighs: MACD, RSI, Bollinger Bands position, active signals, and price momentum

Set strategy via environment variable:
```bash
export TRADING_STRATEGY=bb_ichimoku  # bb, rsi, macd, ichimoku, combined, bb_ichimoku
```

### 🔍 Advanced Filtering
Interactive filter chips for quick analysis:
- 🟢 **Buy Signal** - Stocks with active buy signals
- 🟠 **Sell Signal** - Stocks with active sell signals  
- 🔴 **Short Signal** - Stocks with active short signals
- **Oversold** (RSI < 30) / **Overbought** (RSI > 70)
- **Surge** (>10% gain) / **Crash** (>10% loss)
- **Meme Stocks** / **High Volume** (>50M)
- **BB Squeeze** / **Short Squeeze**
- **Earnings Week** / **Dividend Payers**
- **Category Filters** (M7, Bio, Energy, LX)

### 📈 Market Indicators
Consolidated market overview on a single line:
- **Major Indices**: Dow, S&P 500, Nasdaq with real-time changes
- **VIX**: Volatility index with change tracking
- **CVR3 Signal**: Market-wide Buy/Sell/Short signals
- **Fear & Greed Index**: CNN market sentiment gauge (0-100 scale)
- **AAII Sentiment**: Bull/bear spread from investor survey

### 🚨 Smart Alerts
Automatic alert generation for:
- 52-week highs/lows
- Price surges/crashes (>10%)
- Volume spikes
- Bollinger Band squeezes and breakouts
- Active trading signals (🟢 BUY, 🟠 SELL, 🔴 SHORT)
- Custom user-defined alerts (via `data/alerts.json`)

Alert banner displays at top with color-coded hearts:
- 🔥 52W High alerts
- 📉 52W Low alerts  
- 💚 Buy signals
- 🧡 Sell signals (orange)
- ❤️ Short signals

### ⚡ Performance Optimizations
- **VIX Caching**: 30-minute TTL eliminates redundant API calls
- **Alert Caching**: 30-minute TTL for alert data
- **Fear & Greed Caching**: 30-minute TTL
- **AAII Sentiment Caching**: 30-minute TTL
- **Parallel Fetching**: ThreadPoolExecutor with 5 workers
- **Optimized Calculations**: Vectorized Ichimoku with rolling operations
- **Rate Limiting**: Global limiter prevents API throttling

### 📊 Technical Indicators & Visualizations
- **Sparklines**: Visual price trends for 30-day, 5-day, 1-month periods, and volume
- **Bollinger Bands**: MA20 ± 2σ with squeeze detection
- **RSI**: 14-period momentum oscillator
- **MACD**: 12/26/9 EMA trend following
- **Ichimoku Cloud**: Tenkan/Kijun/Senkou/Chikou analysis
- **Moving Averages**: 50-day and 200-day (death cross detection)
- **Volume Analysis**: Up/down volume bias with sparkline trends
- **Historical Volatility**: 30-day annualized
- **Options Metrics**: Put/Call ratio, implied moves, options direction

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd smi

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run with default ticker file
python3 stocks.py

# Or specify a custom ticker file
python3 stocks.py data/tickers.csv
```

### Running with Custom Trading Strategy

```bash
# Set trading strategy (default: bb_ichimoku)
export TRADING_STRATEGY=bb_ichimoku
python3 stocks.py data/tickers.csv

# Available strategies: bb, rsi, macd, ichimoku, combined, bb_ichimoku
```

### Step-by-Step Instructions

1. **Prepare your ticker list** - Create or edit `data/tickers.csv`:
   ```
   AAPL
   MSFT
   GOOGL
   NVDA
   TSLA
   ```

2. **Optional: Configure custom alerts** - Create `data/alerts.json` (see Custom Alerts section below)

3. **Run the dashboard generator**:
   ```bash
   python3 stocks.py data/tickers.csv
   ```

4. **Wait for completion** - The script will:
   - Fetch data for all tickers in parallel
   - Calculate technical indicators and trading signals
   - Generate both regular and extended hours dashboards
   - Display execution time in minutes

5. **View the dashboards** - Open the generated HTML files:
   - `data/reg_dashboard.html` - Regular trading hours data
   - `data/extnd_dashboard.html` - Extended hours data
   
   Or open directly in browser:
   ```bash
   open data/reg_dashboard.html
   ```

### Output

The script generates:
- `data/reg_dashboard.html` - Regular hours dashboard
- `data/extnd_dashboard.html` - Extended hours dashboard
- Console output with execution time:
  ```
  ✓ Regular Hours Dashboard generated: data/reg_dashboard.html (took 2.34 minutes)
  ✓ Extended Hours Dashboard generated: data/extnd_dashboard.html (took 2.41 minutes)
  Total execution time: 4.75 minutes
  ```

### Ticker File Format
`data/tickers.csv` - One ticker per line (newline or comma-separated):
```
AAPL
MSFT
GOOGL
```

### Custom Alerts
Create `data/alerts.json` for custom alert conditions:
```json
[
  {
    "ticker": "AAPL",
    "condition": "price_above",
    "value": 200
  }
]
```

## Dashboard Features

### Metrics Displayed
- **Price & Changes**: Current price, day/5D/1M/6M/YTD performance
- **Volume**: Trading volume with visual indicators
- **Ranges**: Day and 52-week price ranges with position markers
- **Valuation**: P/E ratio, EPS, Market Cap/AUM, Dividend yield
- **Sentiment**: Analyst ratings, upside potential, options direction
- **Risk Metrics**: Beta, volatility score, short interest, days to cover
- **Earnings**: Next earnings date with week highlighting

### Interactive Features
- **Sortable Columns**: Click any header to sort (supports YTD% and all metrics)
- **Live Search**: Filter by ticker symbol
- **Theme Toggle**: Light/dark mode with persistence
- **View Persistence**: Remembers your preferred view
- **Responsive Design**: Works on desktop and mobile

## Files

- `stocks.py` - Main dashboard generator
- `data/tickers.csv` - Tracked ticker symbols
- `data/alerts.json` - Custom alert definitions
- `data/*_dashboard.html` - Generated dashboard files

## Performance

- **Parallel Processing**: 5 concurrent workers for data fetching
- **Smart Caching**: Eliminates redundant API calls
- **Optimized Calculations**: Vectorized operations for technical indicators
- **Fast Rendering**: Generates both dashboards in ~2-3 minutes for 50 tickers



