## Quick Links

* [Tracker](https://krmallina.github.io/smi/data/reg_dashboard.html), [News](https://tradingeconomics.com/stream)
* [US 100](https://tradingeconomics.com/us100:ind), [Calendar](https://tradingeconomics.com/calendar), [Sectors](https://www.ssga.com/us/en/intermediary/resources/sector-tracker#currentTab=dayFive)
* [Market Movers](https://www.slickcharts.com/market-movers), [SA: Movers](https://stockanalysis.com/markets/gainers/), [SA: Trending](https://stockanalysis.com/trending), [Heat Map](https://stockanalysis.com/markets/heatmap/?time=1W), [MS: Markets](https://www.morningstar.com/markets)
* [Flows](https://www.trackinsight.com/en), [MS: Flows](https://www.google.com/search?q=https://www.morningstar.com/topics/fund-flows), [SPDR: Flows](https://www.ssga.com/us/en/intermediary/insights/a-feast-of-etf-inflows-and-returns), [ETF.com](https://www.etf.com/sections/daily-etf-flows), [ETFdb](https://etfdb.com/etf-fund-flows/#issuer=blackrock-inc)
* [Fear & Greed Index](https://www.cnn.com/markets/fear-and-greed), [AAII Sentiment](https://www.aaii.com/sentiment-survey)
* [Trading Strategies](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)


![cf48c501-9030-4cec-9dde-cf5cc067dbe1](https://github.com/user-attachments/assets/20251d2d-fdf2-4197-b166-d091e752e3ed)

## Features

### 📊 Multi-View Dashboard
- **Table View**: Sortable columns with detailed metrics and sparklines
- **Card View**: Rich card-based layout with visual indicators, trend arrows, and trade setups
  - **Page 1**: Price action, volume, technical indicators, and options metrics
  - **Page 2**: Fundamentals, dividends, earnings, and **ATR-based Trade Setup** recommendations
  - **Page 3**: Visual range charts (Day, 52W, Bollinger Bands, Implied Move)
- **Heatmap View**: Color-coded performance visualization with compact metrics

### 💼 Trade Setup Recommendations
For every ticker with BUY or SHORT signals, the card view displays actionable trade recommendations:

**Trade Setup Box includes:**
- 🟢 **Entry Price**: Current market price
- 🛑 **Stop Loss**: ATR-based stop (2× ATR below/above entry)
  - Dynamically adjusts to stock volatility
  - Shows both price level and risk percentage
- 🎯 **Target Price**: 2:1 risk/reward target (4× ATR)
  - Shows both price level and reward percentage
- 📊 **Risk/Reward Summary**: Visual ratio display (e.g., "5.2% / 10.5% (1:2)")

**Example Display:**
```
🟢 TRADE SETUP (LONG - BUY)
Entry:      $188.81    Current
Stop Loss:  $178.92    -5.2%
Target:     $208.58    +10.5%
─────────────────────────────────
Risk/Reward: 5.2% / 10.5% (1:2)
```

**Features:**
- Automatically shown for all tickers with active BUY/SHORT signals
- Color-coded: Green for LONG/BUY positions, Red for SHORT positions
- Adapts to both light and dark themes
- Uses ATR (Average True Range) for volatility-adjusted levels
- Configurable via `ATR_STOP_MULTIPLIER` environment variable (default: 2.0)

### 🛡️ Risk Management
ATR-based stop loss system with position sizing:
- **ATR Stop Loss**: Dynamic stops based on 14-period Average True Range (always calculated for all tickers)
- **Position Sizing**: Calculates optimal position size to risk 2% per trade (configurable)
- **Risk/Reward Ratios**: Expected R:R calculation using 2:1 targets (4× ATR for targets, 2× ATR for stops)
- **Maximum Position**: 25% account limit to prevent over-concentration
- **Trade Setup Display**: Visual entry/stop/target recommendations on card view page 2
Six sophisticated trading strategies with visual indicators (🟢 BUY, 🟠 SELL, 🔴 SHORT, ⚪ HOLD):

- **Bollinger Bands (BB)**: Price channel breakout detection
  - BUY: Price ≤10% of BB range (oversold)
  - SHORT: Price ≥90% of BB range (overbought)
  - SELL: Price ≥85% AND reversing down (exit longs)
  - HOLD: 10-85% range (normal)
  - Configurable thresholds via `BB_BUY_THRESHOLD`, `BB_SHORT_THRESHOLD`, `BB_SELL_THRESHOLD`

- **RSI**: Oversold/overbought momentum analysis
  - BUY: RSI ≤30 (oversold)
  - SHORT: RSI ≥70 (overbought)
  - SELL: Dropping from overbought OR failing to bounce from oversold
  - HOLD: 30-70 range (normal)
  - Configurable via `RSI_OVERSOLD`, `RSI_OVERBOUGHT`, `RSI_SELL_THRESHOLD`

- **MACD**: Trend following with crossover signals
  - BUY: Bullish crossover (MACD > Signal)
  - SHORT: Bearish crossover (MACD < Signal)
  - SELL: Crossing from bullish to bearish
  - HOLD: No recent crossover
  - Configurable lookback period: `MACD_PERIOD` (50-150 days, default: 150)

- **Ichimoku Cloud**: Multi-component trend and support/resistance
  - BUY: Price crosses above base line in bullish cloud
  - SHORT: Price crosses below base line in bearish cloud
  - SELL: Price crosses below in bearish cloud
  - HOLD: Price above/below base line maintaining bullish/bearish position
  - Optional filters: `ICHIMOKU_VOL_FILTER`, `ICHIMOKU_PRICE_FILTER` (set to 0 to disable)

- **Combined Strategy**: Weighted voting with conflict resolution
  - Aggregates signals from BB, RSI, MACD, Ichimoku with configurable weights
  - Default weights: Ichimoku (1.5), MACD (1.2), BB (1.0), RSI (0.8)
  - BUY/SHORT: Requires weighted score ≥2.0 with no conflicting signals
  - SELL: Only if not conflicting with entry signals
  - HOLD: Only if no strong entry signals exist
  - Configurable via `WEIGHT_*` and `COMBINED_THRESHOLD` variables

- **BB + Ichimoku (Default)**: Multi-mode confirmation strategy
  - **CONFIRM mode** (default): Balanced approach
    - BUY/SHORT: Both BB and Ichimoku must agree
    - SELL: Either can trigger (risk management priority)
    - HOLD: Both must agree
  - **AND mode**: Most conservative - all signals require both strategies
  - **OR mode**: Most aggressive - either strategy can trigger any signal
  - Set via `BB_ICHIMOKU_MODE=CONFIRM|AND|OR`

**Signal Types:**
- 🟢 **BUY** - Strong bullish signal, entry opportunity
- 🔴 **SHORT** - Strong bearish signal, short entry opportunity
- 🟠 **SELL** - Exit signal for existing positions
- ⚪ **HOLD** - Neutral position, no action recommended

### 📈 Predicted Trend Indicators
Color-coded trend arrows based on multi-factor technical analysis:
- **↑** Strong uptrend (green) - Multiple bullish indicators aligned (score ≥4.0)
- **↗** Moderate uptrend (green) - Bullish bias detected (score ≥1.5)
- **→** Neutral/sideways (gray) - Mixed or weak signals (-1.5 to 1.5)
- **↘** Moderate downtrend (red) - Bearish bias detected (score ≤-1.5)
- **↓** Strong downtrend (red) - Multiple bearish indicators aligned (score ≤-4.0)

**Trend Scoring Logic:**
- **MACD** (±2.0): Bullish/Bearish signal (excluded if MACD is active strategy)
- **RSI** (±0.5 to ±1.0): Current momentum direction (>50 = bullish, <50 = bearish)
- **BB Position** (±0.5 to ±1.0): Current trend (>60% = bullish, <40% = bearish)
- **Ichimoku Cloud** (±2.0 to ±2.5): Price above/below cloud with TK cross confirmation
- **Active Signal** (±1.0 to ±2.5): Dynamic weight based on strategy reliability
- **Price Momentum** (±1.0): Daily change threshold (default ±2.0%)

Strategy weights: Ichimoku (2.5), Combined (2.0), BB+Ichimoku (1.8), MACD (1.5), BB (1.2), RSI (1.0)

### 🎯 Trading Signal Framework
Inter-strategy agreement analysis for actionable signals:
- **STRONG (≥75%)**: High agreement across strategies (3+ strategies agreeing)
- **MODERATE (50-75%)**: Partial consensus (2 strategies agreeing)
- **WEAK (<50%)**: Low agreement or conflicting signals
- **Note**: HOLD signals don't receive confidence scores (neutral positions)

Helps filter low-quality signals and focus on high-probability setups. Only BUY, SHORT, and SELL signals are scored.

Set strategy via environment variable:
```bash
export TRADING_STRATEGY=bb_ichimoku  # bb, rsi, macd, ichimoku, combined, bb_ichimoku
```

### 🔍 Advanced Filtering
Interactive filter chips for quick analysis:
- 🟢 **Buy Signal** - Stocks with active buy signals
- 🟠 **Sell Signal** - Stocks with active sell signals  
- 🔴 **Short Signal** - Stocks with active short signals
- ⚪ **Hold Signal** - Stocks with neutral/hold signals
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
- **Bollinger Bands**: MA20 ± 2σ with configurable thresholds and squeeze detection
- **RSI**: 14-period EWM momentum oscillator with extreme level detection
- **MACD**: 12/26/9 EMA trend following with configurable lookback period (50-150 days)
- **Ichimoku Cloud**: Tenkan/Kijun/Senkou/Chikou analysis with optional volume/price filters
- **ATR**: 14-period Average True Range for volatility-based stop losses
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

### Configuration

Customize trading strategies and risk management via environment variables:

#### Trading Strategy Selection
```bash
export TRADING_STRATEGY=bb_ichimoku  # bb, rsi, macd, ichimoku, combined, bb_ichimoku
```

#### Bollinger Bands Configuration
```bash
export BB_BUY_THRESHOLD=10          # Buy below this BB% (default: 10)
export BB_SHORT_THRESHOLD=90        # Short above this BB% (default: 90)
export BB_SELL_THRESHOLD=85         # Sell threshold for reversal detection (default: 85)
```

#### RSI Configuration
```bash
export RSI_OVERSOLD=30              # Oversold threshold (default: 30)
export RSI_OVERBOUGHT=70            # Overbought threshold (default: 70)
export RSI_EXTREME_OVERSOLD=20      # Extreme oversold (default: 20)
export RSI_EXTREME_OVERBOUGHT=80    # Extreme overbought (default: 80)
```

#### MACD Configuration
```bash
export MACD_PERIOD=150              # Historical data period in days (default: 150, range: 50-150)
```

#### Ichimoku Configuration
```bash
export ICHIMOKU_VOL_FILTER=0        # Min volume filter (default: 0 = disabled)
export ICHIMOKU_PRICE_FILTER=0      # Min price filter (default: 0 = disabled)
```

#### Combined Strategy Weights
```bash
export WEIGHT_ICHIMOKU=1.5          # Ichimoku weight (default: 1.5)
export WEIGHT_MACD=1.2              # MACD weight (default: 1.2)
export WEIGHT_BB=1.0                # Bollinger Bands weight (default: 1.0)
export WEIGHT_RSI=0.8               # RSI weight (default: 0.8)
export COMBINED_THRESHOLD=2.0       # Threshold for signals (default: 2.0)
```

#### BB+Ichimoku Mode
```bash
export BB_ICHIMOKU_MODE=CONFIRM     # OR | AND | CONFIRM (default: CONFIRM)
# OR: Either BB or Ichimoku (aggressive)
# AND: Both must agree (conservative)
# CONFIRM: BB primary with Ichimoku confirmation (balanced)
```

#### Trend Prediction
```bash
export TREND_MOMENTUM_THRESHOLD=2.5  # Threshold for trend signal (default: 2.5)
```

#### Risk Management & Trade Setup
```bash
export ATR_STOP_MULTIPLIER=2.0      # ATR multiplier for stop loss (default: 2.0)
export RISK_PER_TRADE=2.0           # % of account to risk per trade (default: 2.0)
```

**Trade Setup Calculation:**
- Stop Loss = Entry Price ± (ATR × ATR_STOP_MULTIPLIER)
- Target = Entry Price ± (ATR × ATR_STOP_MULTIPLIER × 2)  # 2:1 R:R
- Risk % = (|Entry - Stop Loss| / Entry) × 100
- Reward % = (|Target - Entry| / Entry) × 100

**Example with ATR_STOP_MULTIPLIER=2.0:**
- Stock at $100, ATR = $2.50
- BUY Setup:
  - Entry: $100.00
  - Stop Loss: $95.00 (100 - 2.5×2)
  - Target: $110.00 (100 + 2.5×4)
  - Risk/Reward: 5% / 10% (1:2)
export ATR_STOP_MULTIPLIER=2.0      # ATR multiplier for stop loss (default: 2.0)
export RISK_PER_TRADE=2.0           # Risk per trade as % of account (default: 2.0)
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
- **Risk Management**: ATR-14, stop loss levels, risk/reward ratios, position sizing
- **Signal Confidence**: Confidence score (0-100%) and strength (WEAK/MODERATE/STRONG)
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



