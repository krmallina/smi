# ML Stock Predictor Guide

## Overview

The ML Stock Predictor identifies breakout candidates (NVDA, MU-like patterns) and crash risks using machine learning trained on historical data.

## How to Run the Model Trainer

You can train the ML model manually or automate it with GitHub Actions:

### Manual Run (Local)

1. Open a terminal in your project directory.
2. Run:
    ```bash
    python3 ml_predictor.py
    ```
3. This will read tickers from `data/tickers.csv`, fetch historical data, train the model, and save the model files to `data/ml_models/`.

### Automated Run (GitHub Actions)

If you want to automate retraining, use the provided workflow:

1. See `.github/workflows/mlbuild.yml` in your repo.
2. This workflow runs weekly (Sunday 5PM PST) and can be triggered manually from the Actions tab.
3. It will train the model and commit the updated model files to the repository automatically.

**Note:** The dashboard (`stocks.py`) will automatically use the latest model files for predictions if they exist.

## Features

- **Breakout Score (0-100%)**: Probability of >100% gain in 6-12 months
- **Crash Risk (0-100%)**: Probability of >50% decline in 6-12 months
- **Prediction**: BREAKOUT | CRASH | NEUTRAL classification
- **Confidence**: Model certainty (0-100%)

## Data Processing Details

### Historical Data Timeline

The ML model processes **2 years (504 trading days)** of historical data per ticker with a specific training methodology:

```
[2 years ago] ← [1 year ago: Features] → [6-12 months: Future performance]
     ↓              ↓                        ↓
   Fetched        Used for                  Used for
   data          prediction               labeling
```

**Step-by-step process:**
1. **Fetch 2 years of data**: `yf.Ticker(ticker).history(period="2y")`
2. **Feature generation**: Use data from exactly 252 trading days ago (1 year back)
3. **Technical analysis**: Calculate RSI, BB, MACD, volume patterns, etc. at that historical point
4. **Future evaluation**: Check price performance over next 6-12 months
5. **Label assignment**: 
   - **BREAKOUT**: +100% or more gain in 6-12 months
   - **CRASH**: -50% or more loss in 6-12 months  
   - **NEUTRAL**: Performance between -50% and +100%

**Why this approach?**
- **252 trading days** = ~1 year (market cycles)
- **504 trading days** = ~2 years (sufficient historical context)
- **6-12 month horizon** = Medium-term breakout/crash detection
- Simulates real-world prediction: "What would we have predicted 1 year ago?"

### Minimum Data Requirements

- **252+ trading days** minimum (1 year of data)
- Stocks with insufficient history are skipped
- Ensures reliable technical indicator calculations

## Quick Start

### 1. Training the Model

The model needs historical data to learn patterns. The training script automatically detects and uses your data:

**Primary Method: Train on Your Tickers (Recommended)**
```bash
# Train ML model using tickers from data/tickers.csv
python3 ml_predictor.py
```

This will:
- Load tickers from `data/tickers.csv` (if it exists)
- Fetch real historical data for each ticker
- Train the model on actual market patterns
- Save model files to `data/ml_models/`

**Fallback: Synthetic Demo Data (if no tickers.csv)**
```bash
# If data/tickers.csv doesn't exist, generates synthetic training data
python3 ml_predictor.py
```

### Manual Training (Advanced Users)

For custom training with specific tickers, create a training script (`train_ml_model.py`):

```python
import yfinance as yf
import pandas as pd
from ml_predictor import train_model
from datetime import datetime, timedelta

# Load tickers from data/tickers.csv (or specify manually)
tickers = ['NVDA', 'MU', 'AMD', 'TSLA', 'GME', 'AAPL', 'MSFT', ...] # Add your tickers
historical_data = []
labels = []

for ticker in tickers:
    # Fetch 2 years of historical data (~504 trading days)
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    
    # Use data from 1 year ago (252 trading days back) for features
    # Check performance over next 6-12 months for labeling
    # Label as BREAKOUT (+100%+), CRASH (-50%-), or NEUTRAL
    
    # ... extract features and determine label ...
    historical_data.append(features_dict)
    labels.append(label)

# Train model
model, scaler = train_model(historical_data, labels)
```

### 2. Running with ML Predictions

Once trained, ML scores appear automatically:

```bash
# Run dashboard (ML predictions included automatically if model exists)
python3 stocks.py data/tickers.csv
```

The dashboard will show:
- **INDICATORS Column**: ML scores integrated with other indicators
  - **🚀 BREAKOUT (≥70%)**: High breakout potential with green highlighting
  - **💥 CRASH (≥50%)**: High crash risk with red highlighting  
  - **NEUTRAL**: Low probability signals
- ⚠️ **ML Alerts**: Breakout and crash alerts in the dashboard banner

## Understanding ML Scores

### Breakout Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| **80-100%** | Strong breakout signal (similar to past winners like NVDA) | Consider for watchlist |
| **60-79%** | Moderate potential | Monitor closely |
| **40-59%** | Mixed signals | Wait for confirmation |
| **<40%** | Low probability | Not a priority |

### Crash Risk Interpretation

| Risk | Meaning | Action |
|------|---------|--------|
| **≥70%** | High crash risk | Avoid or exit positions |
| **50-69%** | Elevated risk | Use tight stops |
| **30-49%** | Normal risk | Monitor |
| **<30%** | Low risk | - |

## Features Used by the Model

The ML model analyzes 18 technical and fundamental features:

### Technical Indicators
- RSI (momentum)
- Bollinger Band position and width
- MACD signal
- ATR (volatility)
- Volume bias and spikes
- Moving average crosses (Golden/Death Cross)

### Price Momentum
- Daily change %
- 5-day change %
- 1-month change %

### Fundamentals
- P/E ratio
- Market capitalization
- Put/Call ratio
- Short interest %

### Trading Signals
- Current active trading signal
- Trend score
- Squeeze indicators

## Real-World Training Example

To train on actual historical winners and losers:

```python
# Example: Identify breakout stocks from 2020-2023
breakout_stocks = {
    'NVDA': {'trained_date': '2020-01-01', 'outcome': 'BREAKOUT', 'gain': 450%},
    'MU': {'trained_date': '2020-06-01', 'outcome': 'BREAKOUT', 'gain': 180%},
    'AMD': {'trained_date': '2019-12-01', 'outcome': 'BREAKOUT', 'gain': 320%},
}

crash_stocks = {
    'COIN': {'trained_date': '2021-11-01', 'outcome': 'CRASH', 'loss': -75%},
    'PTON': {'trained_date': '2021-01-01', 'outcome': 'CRASH', 'loss': -80%},
}

# For each stock:
# 1. Get data from 'trained_date'
# 2. Calculate all technical indicators (RSI, BB, MACD, etc.)
# 3. Label as BREAKOUT/CRASH based on next 6-12 month performance
# 4. Train model
```

## Model Performance Metrics

After training, the model outputs:

```
📊 Model Performance:
              precision    recall  f1-score   support

   BREAKOUT       0.75      0.82      0.78       120
      CRASH       0.71      0.78      0.74       110
    NEUTRAL       0.88      0.85      0.87       170

   accuracy                           0.82       400

🔝 Top 10 Most Important Features:
           feature  importance
        rsi           0.18
        trend_score   0.15
        change_1m     0.12
        ...
```

## Best Practices

### 1. Combine with Human Judgment
- Don't rely solely on ML scores
- Use as one factor in multi-criteria analysis
- Consider macro trends and news

### 2. Regular Retraining
```bash
# Retrain monthly with new data
python3 train_ml_model.py
```

### 3. Backtesting
- Test model on historical data not used in training
- Validate predictions against actual outcomes
- Adjust features/parameters as needed

### 4. Risk Management
- Even 90% breakout score ≠ guaranteed gain
- Always use stop losses
- Position size appropriately

## Troubleshooting

### Model Not Loading
```
⚠️  ML predictions not available (model not trained)
```
**Solution**: Run `python3 ml_predictor.py` or train with real data

### ML Libraries Not Installed
```
❌ Please install ML libraries: pip install scikit-learn xgboost
```
**Solution**: `pip install -r requirements.txt`

### Low Accuracy
**Solutions**:
- Add more training data (>1000 samples recommended)
- Include more diverse market conditions (bull/bear markets)
- Tune hyperparameters (learning_rate, max_depth, etc.)
- Add more features (sector trends, market breadth, etc.)

## Advanced: Custom Features

Add your own features to `ml_predictor.py`:

```python
def extract_features(stock_data):
    features = [
        # ... existing features ...
        
        # Add custom features
        stock_data.get('your_custom_metric') or 0.0,
        calculate_custom_indicator(stock_data),
    ]
    return features

# Update FEATURE_NAMES accordingly
FEATURE_NAMES = [
    # ... existing names ...
    'your_custom_metric',
    'custom_indicator'
]
```

## FAQ

**Q: How often should I retrain?**  
A: Monthly with new data, or after major market regime changes.

**Q: Can it predict the next NVDA?**  
A: It identifies stocks with similar technical/fundamental patterns to past winners. No guarantee of future performance.

**Q: Why use ML vs traditional signals?**  
A: ML can identify complex multi-factor patterns humans might miss. Use together for best results.

**Q: What's the minimum training data needed?**  
A: Ideally 500+ labeled examples across different market conditions.

## Next Steps

1. ✅ Install dependencies
2. ✅ Train model (synthetic or real data)
3. ✅ Run dashboard and review ML scores
4. 📊 Backtest predictions
5. 🔄 Set up monthly retraining schedule
6. 🎯 Integrate into your trading workflow

For questions or issues, check the code comments in `ml_predictor.py` or create an issue in the repository.
