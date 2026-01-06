![cf48c501-9030-4cec-9dde-cf5cc067dbe1](https://github.com/user-attachments/assets/20251d2d-fdf2-4197-b166-d091e752e3ed)

## Features
## Machine Learning Integration

### 🚀 ML-Powered Stock Predictions (NEW!)
**Identify the Next NVDA, MU Before They Breakout**: Machine learning model trained on historical winners/losers to predict:
- **Breakout Score (0-100%)**: Stocks with >100% gain potential (like NVDA, MU patterns)
- **Crash Risk (0-100%)**: Stocks at risk of >50% decline
- **Integrated in INDICATORS Column**: ML breakout/crash scores are now fully merged into the INDICATORS column (no separate column).
- Color-coded: 🟢 Green (≥70%), 🟠 Orange (50-69%), Gray (<50%)
- ⚠️ Crash warnings for high-risk stocks
- See [ML_GUIDE.md](ML_GUIDE.md) for complete guide

**Troubleshooting ML N/A Values:**
- If ML values show as N/A, check:
  - The ML model files exist in `data/ml_models/` (see ML Model Training section below)
  - The ticker is supported by the model (U.S. equities only)
  - All required features are available for the ticker
  - The ticker is not an ETF or non-stock symbol
  - See the console output for any error messages

### 📊 Multi-View Dashboard
**Auto-Detecting Market Hours**: The dashboard automatically detects whether it's regular trading hours (9:30 AM - 4:00 PM ET, Mon-Fri) or extended hours, and fetches appropriate data accordingly. Extended hours sessions are indicated with a badge at the top of the dashboard.

- **Table View**: Sortable columns with detailed metrics and sparklines
  - All time period changes (Day, 5D, 1M, 6M, YTD, 1Y) with sparklines
  - Golden Cross / Death Cross indicators in INDICATORS column
  - Click any column header to sort
  - Live ticker search and filtering
- **Card View**: Rich card-based layout with visual indicators, trend arrows, and trade setups
  - **Page 1**: Price action, volume, technical indicators (BB, RSI, MACD, MA), and options metrics
  - **Page 2**: Fundamentals, dividends, earnings, and **ATR-based Trade Setup** recommendations
  - **Page 3**: Visual range charts (Day, 52W, Bollinger Bands, Implied Move)
  - Responsive auto-sizing for all devices (no scrolling needed)
  - Swipeable pages with navigation arrows
- **Heatmap View**: Color-coded performance visualization with compact metrics
  - Responsive auto-sizing tiles for all screen sizes
  - Background color intensity based on price change
  - Links to multiple data sources (Barchart, Yahoo, Finviz, Zacks, StockAnalysis)

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

### 🎯 Trading Strategies
Six sophisticated trading strategies with visual indicators (🟢 BUY, 🟠 SELL, 🔴 SHORT, ⏸️ HOLD):

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
- ⏸️ **HOLD** - Neutral position, no action recommended

### 📈 Predicted Trend Indicators
Color-coded trend arrows based on multi-factor technical analysis:
- **<span style="color:green">↑</span>** Strong uptrend (green) - Multiple bullish indicators aligned (score ≥4.0)
- **<span style="color:green">↗</span>** Moderate uptrend (green) - Bullish bias detected (score ≥1.5)
- **<span style="color:gray">→</span>** Neutral/sideways (gray) - Mixed or weak signals (-1.5 to 1.5)
- **<span style="color:red">↘</span>** Moderate downtrend (red) - Bearish bias detected (score ≤-1.5)
- **<span style="color:red">↓</span>** Strong downtrend (red) - Multiple bearish indicators aligned (score ≤-4.0)

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
- ⏸️ **Hold Signal** - Stocks with neutral/hold signals
- **Oversold** (RSI < 30) / **Overbought** (RSI > 70)
- **Surge** (>10% gain) / **Crash** (>10% loss)
- **Meme Stocks** / **High Volume** (>50M) / **M7 Starred**
- **BB Squeeze** / **Short Squeeze**
- **Earnings Week** / **Dividend Payers**
- **Category Filters**:
  - 🌐 **Tech** - Major technology companies
  - ⚡ **Leveraged** - Leveraged/inverse ETFs (auto-detected via ticker patterns and names)
  - 🏦 **ETFs** - Sector/index ETFs (excludes leveraged products)
  - 🚧 **Emerging Tech** - Nuclear, space, battery, clean energy
  - 🎲 **Speculative** - Meme stocks and high-risk plays
  - 💰 **Dividend** - Dividend-paying stocks

### 📋 Ticker File Management
Flexible ticker list format with section support:
- **Sectioned Format** (`data/tickers.csv`):
  ```
  [MEME]
  GME, AMC, BB, KOSS, ...
  
  [M7]
  AAPL, MSFT, GOOGL, META, NVDA, TSLA, ...
  
  [TICKERS]
  AAPL, MSFT, GOOGL, ...
  ```
**Simple Format** (backward compatible):
  ```
  AAPL, MSFT, GOOGL
  NVDA
  TSLA
  ```
*Sections auto-populate MEME and M7 filter categories. Ticker lists are automatically deduplicated and sectioned, regardless of format.*

### 📈 Market Indicators
Consolidated market overview with all global indexes on a single line:
- **Major Indices**: Dow, S&P 500, Nasdaq with real-time changes
- **VIX**: Volatility index with change tracking
- **Commodities**: Gold, Silver, Copper prices and changes
- **Crypto**: Bitcoin price and change tracking
- **CVR3 Signal**: Market-wide Buy/Sell/Short signals (separate line)
- **Fear & Greed Index**: CNN market sentiment gauge (0-100 scale, separate line)
- **AAII Sentiment**: Bull/bear spread from investor survey (separate line)

### 🎨 UI/UX Features
- **Responsive Design**: Auto-sizing for all devices and screen sizes
  - Card view: tiles adapt from 280px to 350px based on viewport
  - Heatmap view: tiles adapt from 180px to 220px based on viewport
  - No horizontal or vertical scrolling required
- **Theme Toggle**: Light/dark mode with local storage persistence
- **View Persistence**: Remembers your last selected view (table/card/heatmap)
- **Live Search**: Real-time ticker filtering in table view
- **Sortable Columns**: Click any header to sort in table view
- **Color Coding**: Consistent green (positive), red (negative), gray (neutral)
- **External Links**: Quick access to Barchart, Yahoo Finance, Finviz, Zacks, StockAnalysis
- **Swipeable Cards**: Navigate between card pages with arrows or swipe gestures

### 🚨 Smart Alerts
Automatic alert generation for:
- 52-week highs/lows
- Price surges/crashes (>10%)
- Volume spikes
- Bollinger Band squeezes and breakouts
- Active trading signals (🟢 BUY, 🟠 SELL, 🔴 SHORT)
- **ML Breakout alerts** (≥70% breakout score)
- **ML Crash Risk alerts** (≥50% crash risk)
- Custom user-defined alerts (via `data/alerts.json`)

Alert banner displays at top with color-coded hearts:
- 🔥 52W High alerts
- 📉 52W Low alerts  
- 💚 Buy signals
- 🧡 Sell signals (orange)
- ❤️ Short signals
- 🚀 ML Breakout alerts
- ⚠️ ML Crash Risk alerts

### ⚡ Performance Optimizations
- **VIX Caching**: 30-minute TTL eliminates redundant API calls
- **Alert Caching**: 30-minute TTL for alert data
- **Fear & Greed Caching**: 30-minute TTL
- **AAII Sentiment Caching**: 30-minute TTL
- **Parallel Fetching**: ThreadPoolExecutor with 5 workers
- **Optimized Calculations**: Vectorized Ichimoku with rolling operations
- **Rate Limiting**: Global limiter prevents API throttling

### 📊 Technical Indicators & Visualizations
- **Sparklines**: Visual price trends for 30-day, 5-day, 1-month, 6-month, YTD, and 1-year periods, plus volume
- **Bollinger Bands**: MA20 ± 2σ with configurable thresholds and squeeze detection
- **RSI**: 14-period EWM momentum oscillator with extreme level detection
- **MACD**: 12/26/9 EMA trend following with configurable lookback period (50-150 days)
- **Ichimoku Cloud**: Tenkan/Kijun/Senkou/Chikou analysis with optional volume/price filters
- **ATR**: 14-period Average True Range for volatility-based stop losses
- **Moving Averages**: 50-day and 200-day SMAs
  - **Golden Cross**: 🟢 SMA 50 crosses above SMA 200 (bullish)
  - **Death Cross**: 🔴 SMA 50 crosses below SMA 200 (bearish)
  - Displayed prominently in table view INDICATORS column
- **Volume Analysis**: Up/down volume bias with sparkline trends
- **Historical Volatility**: 30-day annualized
- **Options Metrics**: Put/Call ratio, implied moves, options direction

### 📈 Price Change Metrics
- **Day %**: Intraday price change with absolute value
- **5D %**: 5-day price change with sparkline
- **1M %**: 1-month price change with sparkline
- **6M %**: 6-month price change with sparkline
- **YTD %**: Year-to-date price change with sparkline
- **1Y %**: 1-year price change with sparkline (newly added)
- All changes display in table and card views (not in heatmap)
- Color-coded: green for gains, red for losses

### 📊 Metrics Displayed (All Views)
**Price & Performance:**
- Current price with day/5D/1M/6M/YTD/1Y changes
- Sparklines for visual trend representation
- 52-week high/low with position markers
- Day range with visual indicators

**Technical Analysis:**
- Bollinger Bands: position %, width %, squeeze detection
- RSI (14-period) with oversold/overbought levels
- MACD line, signal, and trend label
- Ichimoku Cloud signals *(when sufficient data available)*
- Moving Averages: 50-day and 200-day SMAs *(when sufficient data available)*
- Golden Cross / Death Cross detection ***(conditionally rendered - only when crossover detected)***
- ATR (14-period) for volatility measurement

**Trading Signals:**
- Active signal (BUY/SELL/SHORT/HOLD) with colored indicators
- Predicted trend arrows (<span style="color:green">↑↗</span><span style="color:gray">→</span><span style="color:red">↘↓</span>)
- Signal confidence score ***(conditionally rendered - only for BUY/SHORT/SELL signals, not HOLD)***
- Trade setup recommendations (entry/stop/target) ***(conditionally rendered - only shown for BUY/SHORT signals)***

**Volume & Liquidity:**
- Trading volume with sparkline
- Volume bias (up/down volume)
- Average volume comparison
- Volume spike indicator ***(conditionally rendered - only when volume >150% of average)***

**Valuation & Fundamentals:**
- Market Cap or AUM (for ETFs)
- P/E ratio *(optional - N/A for unprofitable companies or ETFs)*
- EPS *(optional)*
- Dividend yield and payout ratio *(optional - only for dividend-paying stocks)*
- Beta *(optional - volatility vs market)*

**Options & Sentiment:**
- Put/Call ratio *(optional - only for optionable stocks)*
- Implied move percentage *(optional - only for optionable stocks)*
- Options direction (bullish/bearish) *(optional - only for optionable stocks)*
- Analyst ratings and target price upside *(optional - only for analyst-covered stocks)*

**Risk Management:**
- Short interest % and days to cover *(optional - may not be available for all stocks)*
- Historical volatility (30-day annualized)
- Squeeze level (None/Moderate/High/Extreme) ***(conditionally rendered - Moderate/High/Extreme only shown when detected)***
- ATR-based stop loss and targets
- Position sizing recommendations ***(conditionally rendered - only shown with active BUY/SHORT signals)***

**Corporate Events:**
- Next earnings date with week highlighting *(optional - may be N/A)*
- Earnings week badge ***(conditionally rendered - only for earnings within 7 days)***
- Dividend ex-date and payment info *(optional - only for dividend-paying stocks)*

**Alert Indicators** ***(conditionally rendered - only when conditions met):***
- 🔥 52-week high alert
- 📉 52-week low alert
- 💚 Active buy signal
- 🧡 Active sell signal
- ❤️ Active short signal
- Surge/crash indicators (>10% moves)

**Note:** 
- *Fields in italics are "optional"* and may display "N/A" or be omitted when data is unavailable from the data provider.
- ***Fields in bold italic are "conditionally rendered"*** and only appear when specific conditions are met or signals are active.

**Examples of Conditionally Rendered Fields:**

1. **Golden/Death Cross:**
   - ✅ Shown: When SMA 50 crosses above/below SMA 200 (crossover just occurred)
   - ❌ Hidden: When moving averages are stable or no recent crossover

2. **Signal Confidence:**
   - ✅ Shown: "STRONG (85%)" for BUY signal with high agreement
   - ❌ Hidden: Not shown for HOLD signals (neutral positions don't have confidence)

3. **Trade Setup Box:**
   ```
   ✅ Shown for BUY signal:
   🟢 TRADE SETUP (LONG - BUY)
   Entry:      $188.81    Current
   Stop Loss:  $178.92    -5.2%
   Target:     $208.58    +10.5%
   
   ❌ Hidden for HOLD or when no active signal
   ```

4. **Volume Spike Indicator:**
   - ✅ Shown: "🔥 Vol Spike!" when today's volume is 2.5M and average is 1.5M (167%)
   - ❌ Hidden: When volume is 1.3M vs average 1.5M (87%)

5. **Squeeze Level:**
   - ✅ Shown: "Squeeze: High" when short interest >20% and days to cover >7
   - ❌ Hidden: Shows "Squeeze: None" when conditions not met (always displays base value)

6. **Position Sizing:**
   - ✅ Shown: "Pos: 8.5%" in card view when BUY signal is active
   - ❌ Hidden: When signal is HOLD or no active signal

7. **Earnings Week Badge:**
   - ✅ Shown: "📅 Earnings: Jan 15" with highlighted background when within 7 days
   - ❌ Hidden: "Earnings: Feb 28" shown normally when >7 days away

8. **Alert Indicators:**
   - ✅ Shown in banner: "🔥 AAPL at 52W High | 💚 MSFT: BUY Signal"
   - ❌ Hidden: When no alert conditions met (no banner displayed)

### 🔧 Interactive Features
- **Sortable Columns**: Click any header to sort (supports all metrics including YTD% and 1Y%)
- **Live Search**: Filter by ticker symbol in real-time
- **Theme Toggle**: Light/dark mode with persistence across sessions
- **View Persistence**: Remembers your preferred view (table/card/heatmap)
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Filter Chips**: One-click filtering by signals, categories, and conditions
- **Alert Dismissal**: Clickable alert banner with local storage persistence

## Files

- `stocks.py` - Main dashboard generator
- `ml_predictor.py` - ML model training and prediction engine
- `data/tickers.csv` - Tracked ticker symbols
- `data/alerts.json` - Custom alert definitions
- `data/dashboard.html` - Generated dashboard (auto-detects market hours)
- `data/ml_models/` - Trained ML models and scalers
  - `breakout_crash_model.pkl` - Main ML model for breakout/crash predictions
  - `feature_scaler.pkl` - Feature normalization scaler
- `.github/workflows/build.yml` - Automated dashboard updates (4 AM - 5 PM PST, every 30 min)
- `.github/workflows/mlbuild.yml` - Weekly ML model retraining (Mondays at 1 AM PST)

## Performance

- **Parallel Processing**: 5 concurrent workers for data fetching
- **Smart Caching**: Eliminates redundant API calls
- **Optimized Calculations**: Vectorized operations for technical indicators
- **Fast Rendering**: Generates both dashboards in ~2-3 minutes for 50 tickers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd smi

# Install dependencies
pip install -r requirements.txt
```

## ML Model Training & Automation

### Manual ML Training

Train the ML model on your ticker data:

```bash
# Train ML model using tickers from data/tickers.csv
python3 ml_predictor.py
```

This will:
- Fetch historical data for all tickers
- Train a new breakout/crash prediction model
- Save the model to `data/ml_models/breakout_crash_model.pkl`
- Test the model on sample data

### Automated ML Training

The `mlbuild.yml` GitHub Actions workflow automatically retrains the ML model weekly:

- **Schedule**: Every Monday at 1:00 AM PST
- **Trigger**: Manual via GitHub Actions UI
- **Process**: 
  - Fetches latest ticker data
  - Trains new model on current market conditions
  - Commits only the updated model files (no code changes)
  - Maintains flat commit history for model updates

**Benefits:**
- Models stay current with evolving market patterns
- No manual intervention required
- Automatic deployment of improved predictions
- Historical model performance tracking

### ML Model Files

- `data/ml_models/breakout_crash_model.pkl` - Gradient Boosting model trained on historical winners/losers
- `data/ml_models/feature_scaler.pkl` - Feature normalization scaler
- Models are automatically loaded by `stocks.py` for predictions
- If models don't exist, predictions show as "N/A"

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
   - Auto-detect current market hours (regular vs extended)
   - Calculate technical indicators and trading signals
   - Generate dashboard with appropriate data for the session
   - Display execution time in minutes

5. **View the dashboard** - Open the generated HTML file:
   - `data/dashboard.html` - Automatically shows regular or extended hours data based on when it was run
   
   Or open directly in browser:
   ```bash
   open data/dashboard.html
   ```

### Output

The script generates:
- `data/dashboard.html` - Single dashboard file with auto-detected market hours data
- Console output with execution time:
  ```
  ✓ Dashboard generated: data/dashboard.html (took 2.34 minutes)
  
  ⏱️  Total time: 2.34 minutes
  ```

**Market Hours Detection:**
- The dashboard automatically determines if it's regular hours (9:30 AM - 4:00 PM ET, Mon-Fri)
- During regular hours: Fetches regular market data (no badge shown)
- Outside regular hours: Fetches extended hours data (shows "Extended Hours" badge)
- GitHub Actions workflow runs every 30 minutes from 4 AM - 5 PM PST on weekdays

### Ticker File Format
`data/tickers.csv` - One ticker per line (newline or comma-separated):
```
AAPL
MSFT
GOOGL
```

## Configuration

Customize trading strategies and risk management via environment variables:

### Trading Strategy Selection
```bash
export TRADING_STRATEGY=bb_ichimoku  # bb, rsi, macd, ichimoku, combined, bb_ichimoku
```

### Bollinger Bands Configuration
```bash
export BB_BUY_THRESHOLD=10          # Buy below this BB% (default: 10)
export BB_SHORT_THRESHOLD=90        # Short above this BB% (default: 90)
export BB_SELL_THRESHOLD=85         # Sell threshold for reversal detection (default: 85)
```

### RSI Configuration
```bash
export RSI_OVERSOLD=30              # Oversold threshold (default: 30)
export RSI_OVERBOUGHT=70            # Overbought threshold (default: 70)
export RSI_EXTREME_OVERSOLD=20      # Extreme oversold (default: 20)
export RSI_EXTREME_OVERBOUGHT=80    # Extreme overbought (default: 80)
```

### MACD Configuration
```bash
export MACD_PERIOD=150              # Historical data period in days (default: 150, range: 50-150)
```

### Ichimoku Configuration
```bash
export ICHIMOKU_VOL_FILTER=0        # Min volume filter (default: 0 = disabled)
export ICHIMOKU_PRICE_FILTER=0      # Min price filter (default: 0 = disabled)
```

### Combined Strategy Weights
```bash
export WEIGHT_ICHIMOKU=1.5          # Ichimoku weight (default: 1.5)
export WEIGHT_MACD=1.2              # MACD weight (default: 1.2)
export WEIGHT_BB=1.0                # Bollinger Bands weight (default: 1.0)
export WEIGHT_RSI=0.8               # RSI weight (default: 0.8)
export COMBINED_THRESHOLD=2.0       # Threshold for signals (default: 2.0)
```

### BB+Ichimoku Mode
```bash
export BB_ICHIMOKU_MODE=CONFIRM     # OR | AND | CONFIRM (default: CONFIRM)
# OR: Either BB or Ichimoku (aggressive)
# AND: Both must agree (conservative)
# CONFIRM: BB primary with Ichimoku confirmation (balanced)
```

### Trend Prediction
```bash
export TREND_MOMENTUM_THRESHOLD=2.5  # Threshold for trend signal (default: 2.5)
```

### Risk Management & Trade Setup
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

### Custom Alerts
Create `data/alerts.json` to define custom alert conditions for specific tickers. The system supports multiple alert types:

**Available Alert Conditions:**

1. **Price Thresholds:**
   - `price_above` - Alert when price exceeds a specific value
   - `price_below` - Alert when price falls below a specific value

2. **Daily Price Changes:**
   - `day_change_above` - Alert when daily % change exceeds a threshold
   - `day_change_below` - Alert when daily % change falls below a threshold

3. **RSI Levels:**
   - `rsi_oversold` - Alert when RSI drops below 30 (oversold)
   - `rsi_overbought` - Alert when RSI rises above 70 (overbought)

4. **Volume Activity:**
   - `volume_spike` - Alert when volume exceeds 150% of average

5. **Trading Signals:**
   - `buy` - Alert on BUY signals from active strategy
   - `sell` - Alert on SELL signals from active strategy
   - `short` - Alert on SHORT signals from active strategy

**Example `data/alerts.json`:**
```json
[
  {
    "ticker": "AAPL",
    "condition": "price_above",
    "value": 200
  },
  {
    "ticker": "TSLA",
    "condition": "price_below",
    "value": 150
  },
  {
    "ticker": "NVDA",
    "condition": "day_change_above",
    "value": 5
  },
  {
    "ticker": "AMD",
    "condition": "day_change_below",
    "value": -3
  },
  {
    "ticker": "GOOGL",
    "condition": "rsi_oversold"
  },
  {
    "ticker": "MSFT",
    "condition": "rsi_overbought"
  },
  {
    "ticker": "META",
    "condition": "volume_spike"
  },
  {
    "ticker": "AMZN",
    "condition": "buy"
  },
  {
    "ticker": "SPY",
    "condition": "sell"
  },
  {
    "ticker": "QQQ",
    "condition": "short"
  }
]
```

**Notes:**
- Price and percentage conditions require a `value` field
- RSI, volume, and signal conditions do not require a `value`
- Alerts appear in the banner at the top of the dashboard
- Custom alerts are cached for 30 minutes to improve performance
