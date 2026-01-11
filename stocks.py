"""
Stock Market Intelligence Dashboard

Trading Strategies:
  The system supports multiple trading signal strategies. Set via TRADING_STRATEGY environment variable:
  
  - 'bb_ichimoku' (default): BB or Ichimoku - Configurable OR/AND/CONFIRM modes
  - 'bb': Bollinger Bands - Buy at lower band, Short at upper band, HOLD in normal range
  - 'rsi': RSI - Buy when oversold (<30), Short when overbought (>70), HOLD in between
  - 'macd': MACD - Buy on bullish crossover, Short on bearish crossover, SELL on reversal
  - 'ichimoku': Ichimoku Cloud - Buy on bullish cloud break, Short on bearish, HOLD in trend
  - 'combined': Weighted voting - Requires weighted score >= threshold (configurable weights)
  
  Signal Types:
  - BUY: Enter long position
  - SHORT: Enter short position
  - SELL: Exit long position (stop/reversal)
  - HOLD: Maintain current position
  - None: Insufficient data or no clear signal
  
  Configurable Thresholds (via environment variables):
  - BB_BUY_THRESHOLD=10 (default: buy when BB% <= 10)
  - BB_SHORT_THRESHOLD=90 (default: short when BB% >= 90)
  - BB_SELL_THRESHOLD=85 (default: exit longs when BB% >= 85 and reversing)
  - RSI_OVERSOLD=30 (default: oversold threshold)
  - RSI_OVERBOUGHT=70 (default: overbought threshold)
  - RSI_SELL_THRESHOLD=65 (default: exit when dropping from overbought)
  - MACD_PERIOD=100 (default: 100-day for stability, use 50 for momentum trading)
  - ICHIMOKU_VOL_FILTER=100000 (default: min volume, set 0 to disable)
  - ICHIMOKU_PRICE_FILTER=20 (default: min price, set 0 to disable)
  - TREND_MOMENTUM_THRESHOLD=2.0 (default: ±2% for trend, use 1.0 for volatile stocks)
  - VIX_EXTREME_FEAR=20 (VIX % above SMA for strong buy signal)
  - VIX_MODERATE_FEAR=10 (VIX % above SMA for buy signal)
  - VIX_COMPLACENCY=-10 (VIX % below SMA for sell signal)
  - VIX_EXTREME_COMPLACENCY=-20 (VIX % below SMA for short signal)
  
  Strategy Weights (for 'combined' strategy):
  - WEIGHT_BB=1.0 (Bollinger Bands weight)
  - WEIGHT_RSI=0.8 (RSI weight - less reliable alone)
  - WEIGHT_MACD=1.2 (MACD weight - strong trend indicator)
  - WEIGHT_ICHIMOKU=1.5 (Ichimoku weight - most comprehensive)
  - COMBINED_THRESHOLD=2.0 (min weighted score to trigger signal)
  
  BB+Ichimoku Strategy Mode (for 'bb_ichimoku' default strategy):
  - BB_ICHIMOKU_MODE=confirm (default: both must agree for BUY/SHORT, either for SELL/HOLD)
  - BB_ICHIMOKU_MODE=or (either BB OR Ichimoku triggers - most sensitive)
  - BB_ICHIMOKU_MODE=and (both BB AND Ichimoku must agree - most conservative)
  
  Example usage:
    export TRADING_STRATEGY=ichimoku
    export BB_BUY_THRESHOLD=5
    export RSI_OVERSOLD=25
    export MACD_PERIOD=50  # Faster MACD for momentum trades
    export ICHIMOKU_VOL_FILTER=0  # Disable volume filter for small caps
    export ICHIMOKU_PRICE_FILTER=5  # Allow stocks >= $5
    export BB_ICHIMOKU_MODE=confirm  # Require confirmation for entries
    python3 stocks.py data/tickers.csv
    
  Weighted voting example (combined strategy):
    # Trust Ichimoku and MACD more than RSI
    export TRADING_STRATEGY=combined
    export WEIGHT_ICHIMOKU=2.0
    export WEIGHT_MACD=1.5
    export WEIGHT_BB=1.0
    export WEIGHT_RSI=0.5
    export COMBINED_THRESHOLD=2.5  # Require strong agreement
    python3 stocks.py data/tickers.csv
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import argparse
import json
import requests
import random
import re
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import threading

# ML Predictor (optional - gracefully degrades if not available)
try:
    from ml_predictor import predict_breakout_crash, batch_predict, ML_AVAILABLE
except ImportError:
    ML_AVAILABLE = False
    def predict_breakout_crash(stock_data):
        return {'breakout_score': 0, 'crash_risk': 0, 'prediction': 'NEUTRAL', 'confidence': 0}

# Summary generator
try:
    from summary import generate_summary_html
except ImportError:
    def generate_summary_html(csv="data/tickers.csv"):
        print("⚠️  Summary generation not available (summary.py not found)")

ALERTS_FILE = "data/alerts.json"
UTC = pytz.utc
PST = pytz.timezone("America/Los_Angeles")
ET = pytz.timezone("America/New_York")

def is_regular_hours():
    """Check if current time is during regular market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    now_et = datetime.now(ET)
    # Check if it's a weekday
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return False
    # Check if time is between 9:30 AM and 4:00 PM ET
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

# Performance and API Configuration Constants
MAX_WORKERS = 5  # Maximum thread pool workers for parallel ticker fetching
MIN_WORKERS = 2  # Minimum thread pool workers
TICKER_BATCH_DIVISOR = 6  # Divisor for calculating worker count from ticker count
RATE_LIMIT_INTERVAL = 0.6  # Minimum seconds between API requests (reduced due to fewer calls)
MAX_RETRIES = 3  # Maximum retry attempts for failed API calls
FETCH_BASE_DELAY = 0.05  # Base delay in seconds for fetch requests (reduced)
FETCH_JITTER_MAX = 0.15  # Maximum random jitter to add to delays
RETRY_DELAY_MULTIPLIER = 0.15  # Multiplier for retry delays
EXPONENTIAL_BACKOFF_BASE = 5  # Base seconds for exponential backoff
BACKOFF_JITTER_MAX = 3  # Maximum random jitter for backoff

# Cache Configuration
TICKER_INFO_CACHE_SIZE = 512  # LRU cache size for ticker info
FG_CACHE_TTL_SECONDS = 1800  # Fear & Greed cache TTL (30 minutes)

# UI Configuration Constants
CARD_HEIGHT_PX = 450  # Fixed height for stock cards in pixels
CARD_PADDING_TOP_PX = 45  # Top padding for stock cards
CARD_PADDING_SIDE_PX = 24  # Side and bottom padding for stock cards
CARD_ARROW_SIZE_PX = 24  # Size of navigation arrow buttons
CARD_MIN_WIDTH_PX = 300  # Minimum width for card grid items
HEAT_TILE_HEIGHT_PX = 250  # Fixed height for heatmap tiles in pixels

# These will be loaded from tickers.csv file
MEME_STOCKS = frozenset()
STAR_STOCKS = frozenset()
M7_STOCKS = frozenset()
EMERGINGTECH_STOCKS = frozenset()
LEVERAGED_STOCKS = frozenset()
ETFS_STOCKS = frozenset()

# Category buckets for filtering chips
CATEGORY_MAP = {
    "major-tech": frozenset({"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "NFLX", "PLTR", "AVGO", "ORCL", "SHOP", "ARM"}),
    "leveraged-etf": LEVERAGED_STOCKS,
    "sector-etf": ETFS_STOCKS,
    "spec-meme": MEME_STOCKS,
    "emerging-tech": EMERGINGTECH_STOCKS,
    "star": STAR_STOCKS,  
    "m7": M7_STOCKS,
}


def get_category(ticker):
    tu = ticker.upper()
    for slug, tickers in CATEGORY_MAP.items():
        if tu in tickers:
            return slug
    return ""


def infer_category_from_info(ticker, info):
    tu = ticker.upper()
    # Start with manual map if present
    mapped = get_category(tu)
    if mapped:
        return mapped

    qtype = (info.get("quoteType") or "").lower()
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    lname = (info.get("longName") or "").lower()
    sname = (info.get("shortName") or "").lower()

    # Detect leveraged/inverse ETFs via quoteType or naming
    if qtype == "etf" or "etf" in lname or "etf" in sname:
        ticker_lower = tu.lower()
        
        # Explicitly exclude common non-leveraged ETFs to prevent false positives
        non_leveraged_tickers = {'spy', 'qqq', 'dia', 'iwm', 'voo', 'vti', 'xlf', 'xle', 
                                'xlk', 'xlv', 'xlu', 'xli', 'xlc', 'xlp', 'xlb', 'xly',
                                'smh', 'xbi', 'iyr', 'gld', 'slv', 'uso'}
        if ticker_lower in non_leveraged_tickers:
            return "sector-etf"
        
        lever_markers = (
            "3x",
            "2x",
            "1.5x",
            "ultra",
            "ultrapro",
            "leveraged",
            "inverse",
            "-1x",
            "-2x",
            "-3x",
            "short",
            "double",
            "triple",
        )
        
        # Check specific leveraged ticker patterns (must be 4+ chars and match specific patterns)
        # TQQQ, SQQQ (4 chars ending with qqq), SPXL, TECL (4 chars ending with xl)
        # SPXS, TECS (4 chars ending with xs)
        is_leveraged_ticker = (
            (len(tu) == 4 and ticker_lower.endswith('qqq'))  # TQQQ, SQQQ (not QQQ which is 3)
            or (len(tu) == 4 and ticker_lower.endswith('xl'))  # SPXL, TECL (not XL_ sector ETFs which are 3)
            or (len(tu) == 4 and ticker_lower.endswith('xs'))  # SPXS, TECS
            or any(m in ticker_lower for m in ("3x", "2x", "ultra", "bull", "bear"))
        )
        # Check name/description for leverage indicators
        is_leveraged_name = (
            any(m in lname for m in lever_markers)
            or any(m in sname for m in lever_markers)
            or any(x in lname.lower() for x in ("bear ", "bull ", " 2x", " 3x", "daily ", "inverse", "leveraged"))
        )
        
        if is_leveraged_ticker or is_leveraged_name:
            return "leveraged-etf"
        return "sector-etf"

    # Meme / speculative
    if tu in MEME_STOCKS:
        return "spec-meme"

    # Major tech/growth heuristics
    if sector == "technology" or any(
        k in industry for k in ("semiconductor", "software", "ai")
    ):
        return "major-tech"

    # Emerging tech / energy heuristics
    if any(
        k in industry for k in ("nuclear", "battery", "clean energy", "solar", "space")
    ) or any(k in lname for k in ("nuclear", "battery", "rocket", "fusion", "space")):
        return "emerging-tech"

    return ""


# ============================================================================
# TRADING SIGNAL STRATEGIES
# ============================================================================

def calculate_ichimoku(high, low, close, conversion_period=9, base_period=26, span_b_period=52):
    """
    Calculate Ichimoku Cloud indicators.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        conversion_period: Period for Conversion Line (default 9)
        base_period: Period for Base Line (default 26)
        span_b_period: Period for Span B (default 52)
    
    Returns:
        Dictionary with Ichimoku components or None if insufficient data
    """
    if len(high) < span_b_period:
        return None
    
    # Optimize: Calculate rolling max/min once per period
    high_rolling = {
        conversion_period: high.rolling(window=conversion_period).max(),
        base_period: high.rolling(window=base_period).max(),
        span_b_period: high.rolling(window=span_b_period).max()
    }
    low_rolling = {
        conversion_period: low.rolling(window=conversion_period).min(),
        base_period: low.rolling(window=base_period).min(),
        span_b_period: low.rolling(window=span_b_period).min()
    }
    
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    conversion_line = (high_rolling[conversion_period] + low_rolling[conversion_period]) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    base_line = (high_rolling[base_period] + low_rolling[base_period]) / 2
    
    # Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    span_a = (conversion_line + base_line) / 2
    
    # Span B (Senkou Span B): (52-period high + 52-period low) / 2
    span_b = (high_rolling[span_b_period] + low_rolling[span_b_period]) / 2
    
    return {
        'conversion_line': conversion_line.iloc[-1] if len(conversion_line) > 0 else None,
        'base_line': base_line.iloc[-1] if len(base_line) > 0 else None,
        'span_a': span_a.iloc[-1] if len(span_a) > 0 else None,
        'span_b': span_b.iloc[-1] if len(span_b) > 0 else None,
        'conversion_line_prev': conversion_line.iloc[-2] if len(conversion_line) > 1 else None,
        'base_line_prev': base_line.iloc[-2] if len(base_line) > 1 else None,
    }


def generate_trading_signals(stock_data):
    """
    Generate buy/sell/short/hold signals based on multiple strategies.
    
    Args:
        stock_data: Dictionary containing stock metrics (bb_position_pct, rsi, macd_label, ichimoku, etc.)
    
    Returns:
        Dictionary with signals from different strategies:
        {
            'bb': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None,
            'rsi': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None,
            'macd': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None,
            'ichimoku': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None,
            'combined': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None,
            'bb_ichimoku': 'BUY' | 'SELL' | 'SHORT' | 'HOLD' | None
        }
    """
    signals = {}
    
    # Strategy 1: Bollinger Bands (with configurable thresholds)
    BB_BUY_THRESHOLD = float(os.getenv('BB_BUY_THRESHOLD', '10'))      # Buy when <= 10%
    BB_SHORT_THRESHOLD = float(os.getenv('BB_SHORT_THRESHOLD', '90'))  # Short when >= 90%
    BB_SELL_THRESHOLD = float(os.getenv('BB_SELL_THRESHOLD', '85'))    # Exit longs when >= 85%
    
    bb_position_pct = stock_data.get('bb_position_pct')
    bb_position_prev = stock_data.get('bb_position_prev')  # Previous position for trend detection
    
    if bb_position_pct is not None:
        if bb_position_pct <= BB_BUY_THRESHOLD:
            # Oversold: Strong buy signal
            signals['bb'] = 'BUY'
        elif bb_position_pct >= BB_SHORT_THRESHOLD:
            # Overbought: Strong short signal
            signals['bb'] = 'SHORT'
        elif bb_position_pct >= BB_SELL_THRESHOLD and bb_position_prev is not None and bb_position_prev > bb_position_pct:
            # Price near upper band AND starting to reverse down: Exit long positions
            signals['bb'] = 'SELL'
        elif BB_BUY_THRESHOLD < bb_position_pct < BB_SELL_THRESHOLD:
            # Normal range: Hold current position
            signals['bb'] = 'HOLD'
        else:
            signals['bb'] = None
    else:
        signals['bb'] = None
    
    # Strategy 2: RSI (with configurable thresholds)
    RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30'))
    RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70'))
    RSI_SELL_THRESHOLD = float(os.getenv('RSI_SELL_THRESHOLD', '65'))  # Exit when dropping from overbought
    
    rsi = stock_data.get('rsi')
    rsi_prev = stock_data.get('rsi_prev')
    
    if rsi is not None:
        if rsi <= RSI_OVERSOLD:
            signals['rsi'] = 'BUY'  # Oversold
        elif rsi >= RSI_OVERBOUGHT:
            signals['rsi'] = 'SHORT'  # Overbought
        elif rsi >= RSI_SELL_THRESHOLD and rsi_prev is not None and rsi_prev > rsi and rsi_prev >= RSI_OVERBOUGHT:
            # Dropping from overbought: Exit longs
            signals['rsi'] = 'SELL'
        elif rsi <= 35 and rsi_prev is not None and rsi < rsi_prev and rsi_prev <= RSI_OVERSOLD:
            # Failing to bounce from oversold (bearish continuation): Exit longs/shorts
            signals['rsi'] = 'SELL'
        elif RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
            signals['rsi'] = 'HOLD'  # Normal range
        else:
            signals['rsi'] = None
    else:
        signals['rsi'] = None
    
    # Strategy 3: MACD
    macd_label = stock_data.get('macd_label')
    macd_prev_label = stock_data.get('macd_prev_label')
    
    if macd_label == 'Bullish':
        signals['macd'] = 'BUY'
    elif macd_label == 'Bearish':
        signals['macd'] = 'SHORT'
    elif macd_prev_label == 'Bullish' and macd_label != 'Bullish':
        # Just crossed bearish: Exit longs
        signals['macd'] = 'SELL'
    elif macd_label is not None:
        signals['macd'] = 'HOLD'
    else:
        signals['macd'] = None
    
    # Strategy 4: Ichimoku Cloud (with configurable filters)
    ichimoku = stock_data.get('ichimoku')
    price = stock_data.get('price')
    price_prev = stock_data.get('price_prev')
    vol_sma_20 = stock_data.get('vol_sma_20')
    price_sma_60 = stock_data.get('price_sma_60')
    
    # Configurable filters (set to 0 to disable)
    ICHIMOKU_VOL_FILTER = float(os.getenv('ICHIMOKU_VOL_FILTER', '100000'))
    ICHIMOKU_PRICE_FILTER = float(os.getenv('ICHIMOKU_PRICE_FILTER', '20'))
    
    if ichimoku and price is not None and all(v is not None for v in ichimoku.values()):
        base_line = ichimoku.get('base_line')
        base_line_prev = ichimoku.get('base_line_prev')
        span_a = ichimoku.get('span_a')
        span_b = ichimoku.get('span_b')
        
        # Apply filters only if thresholds are set (> 0)
        vol_filter = (ICHIMOKU_VOL_FILTER == 0) or (vol_sma_20 is None) or (vol_sma_20 > ICHIMOKU_VOL_FILTER)
        price_filter = (ICHIMOKU_PRICE_FILTER == 0) or (price_sma_60 is None) or (price_sma_60 > ICHIMOKU_PRICE_FILTER)
        
        # Buy Signal: close > span_b AND span_a > span_b AND close crosses above base_line
        if vol_filter and price_filter:
            cloud_bullish = price > span_b and span_a > span_b
            crosses_above = price_prev is not None and base_line_prev is not None and \
                           price_prev <= base_line_prev and price > base_line
            
            # Sell Signal: close < span_a AND span_a < span_b AND base_line crosses above close
            cloud_bearish = price < span_a and span_a < span_b
            crosses_below = price_prev is not None and base_line_prev is not None and \
                           price_prev >= base_line_prev and price < base_line
            
            if cloud_bullish and crosses_above:
                signals['ichimoku'] = 'BUY'
            elif cloud_bearish and crosses_below:
                signals['ichimoku'] = 'SHORT'
            elif price > base_line and cloud_bullish:
                # In bullish position: Hold
                signals['ichimoku'] = 'HOLD'
            elif price < base_line and cloud_bearish:
                # In bearish position (holding SHORT): Hold
                signals['ichimoku'] = 'HOLD'
            elif cloud_bearish and price_prev is not None and price_prev >= base_line:
                # Crossed below in bearish cloud: Exit longs
                signals['ichimoku'] = 'SELL'
            else:
                signals['ichimoku'] = None
        else:
            signals['ichimoku'] = None
    else:
        signals['ichimoku'] = None
    
    # Combined Strategy: Weighted voting with configurable thresholds
    # Strategy weights (configurable via environment variables)
    WEIGHT_BB = float(os.getenv('WEIGHT_BB', '1.0'))
    WEIGHT_RSI = float(os.getenv('WEIGHT_RSI', '0.8'))
    WEIGHT_MACD = float(os.getenv('WEIGHT_MACD', '1.2'))
    WEIGHT_ICHIMOKU = float(os.getenv('WEIGHT_ICHIMOKU', '1.5'))
    
    strategy_weights = {
        'bb': WEIGHT_BB,
        'rsi': WEIGHT_RSI,
        'macd': WEIGHT_MACD,
        'ichimoku': WEIGHT_ICHIMOKU
    }
    
    # Calculate weighted scores
    buy_score = sum(strategy_weights.get(k, 1.0) for k, v in signals.items() if v == 'BUY' and k in strategy_weights)
    short_score = sum(strategy_weights.get(k, 1.0) for k, v in signals.items() if v == 'SHORT' and k in strategy_weights)
    sell_score = sum(strategy_weights.get(k, 1.0) for k, v in signals.items() if v == 'SELL' and k in strategy_weights)
    hold_score = sum(strategy_weights.get(k, 1.0) for k, v in signals.items() if v == 'HOLD' and k in strategy_weights)
    
    # Configurable threshold (default: 2.0 = approximately 2 strategies agreeing)
    COMBINED_THRESHOLD = float(os.getenv('COMBINED_THRESHOLD', '2.0'))
    
    # Determine signal based on highest score above threshold
    max_score = max(buy_score, short_score, sell_score, hold_score)
    
    if max_score >= COMBINED_THRESHOLD:
        if buy_score == max_score and buy_score > short_score:
            signals['combined'] = 'BUY'
        elif short_score == max_score and short_score > buy_score:
            signals['combined'] = 'SHORT'
        elif sell_score == max_score and sell_score > buy_score and sell_score > short_score:
            # SELL only if not conflicting with entry signals
            signals['combined'] = 'SELL'
        elif hold_score == max_score and buy_score < COMBINED_THRESHOLD and short_score < COMBINED_THRESHOLD:
            # HOLD only if no strong entry signals
            signals['combined'] = 'HOLD'
        else:
            # Conflicting signals (e.g., buy_score == short_score, or sell/hold conflicts with entries)
            signals['combined'] = None
    else:
        signals['combined'] = None
    
    # BB or Ichimoku Strategy: Configurable logic mode (default)
    # Modes: 'confirm' (default/balanced), 'or' (most sensitive), 'and' (most conservative)
    BB_ICHIMOKU_MODE = os.getenv('BB_ICHIMOKU_MODE', 'confirm').lower()
    
    bb_sig = signals.get('bb')
    ich_sig = signals.get('ichimoku')
    
    if BB_ICHIMOKU_MODE == 'and':
        # AND mode: Both must agree (lowest false positives)
        if bb_sig == 'BUY' and ich_sig == 'BUY':
            signals['bb_ichimoku'] = 'BUY'
        elif bb_sig == 'SHORT' and ich_sig == 'SHORT':
            signals['bb_ichimoku'] = 'SHORT'
        elif bb_sig == 'SELL' and ich_sig == 'SELL':
            signals['bb_ichimoku'] = 'SELL'
        elif bb_sig == 'HOLD' and ich_sig == 'HOLD':
            signals['bb_ichimoku'] = 'HOLD'
        else:
            signals['bb_ichimoku'] = None
    
    elif BB_ICHIMOKU_MODE == 'confirm':
        # CONFIRM mode: Require both for entries (BUY/SHORT), either for exits (SELL)
        if bb_sig == 'BUY' and ich_sig == 'BUY':
            signals['bb_ichimoku'] = 'BUY'
        elif bb_sig == 'SHORT' and ich_sig == 'SHORT':
            signals['bb_ichimoku'] = 'SHORT'
        elif bb_sig == 'SELL' or ich_sig == 'SELL':
            # Exit quickly if either signals danger (risk management priority)
            signals['bb_ichimoku'] = 'SELL'
        elif bb_sig == 'HOLD' and ich_sig == 'HOLD':
            # Hold only if both suggest maintaining position (not conflicting)
            signals['bb_ichimoku'] = 'HOLD'
        else:
            # One says BUY/SHORT, other says HOLD/None → Insufficient confirmation
            signals['bb_ichimoku'] = None
    
    else:  # Default 'or' mode
        # OR mode: Either BB or Ichimoku triggers (most sensitive, original behavior)
        if bb_sig == 'BUY' or ich_sig == 'BUY':
            signals['bb_ichimoku'] = 'BUY'
        elif bb_sig == 'SHORT' or ich_sig == 'SHORT':
            signals['bb_ichimoku'] = 'SHORT'
        elif bb_sig == 'SELL' or ich_sig == 'SELL':
            signals['bb_ichimoku'] = 'SELL'
        elif bb_sig == 'HOLD' or ich_sig == 'HOLD':
            signals['bb_ichimoku'] = 'HOLD'
        else:
            signals['bb_ichimoku'] = None
    
    return signals


def get_active_strategy():
    """
    Returns the currently active trading strategy.
    Can be configured via environment variable or settings file.
    
    Options: 'bb', 'rsi', 'macd', 'ichimoku', 'combined', 'bb_ichimoku'
    Default: 'bb_ichimoku' (either BB or Ichimoku triggers)
    """
    return os.getenv('TRADING_STRATEGY', 'bb_ichimoku')


def calculate_signal_confidence(all_signals, active_strategy):
    """
    Calculate confidence score and strength label for trading signals.
    
    Args:
        all_signals: Dictionary of all strategy signals
        active_strategy: Currently active strategy name
    
    Returns:
        tuple: (confidence_score, strength_label)
            confidence_score: 0.0-1.0 based on agreement
            strength_label: 'WEAK' | 'MODERATE' | 'STRONG'
    """
    # Count signals by type (excluding None and active strategy itself)
    active_sig = all_signals.get(active_strategy)
    
    # HOLD signals don't need confidence scoring (neutral positions)
    if not active_sig or active_sig == 'HOLD':
        return None, None
    
    # Only actionable signals (BUY/SHORT/SELL) get confidence scores
    if active_sig not in ('BUY', 'SHORT', 'SELL'):
        return 0.0, 'WEAK'
    
    # Check agreement from other strategies
    other_strategies = {k: v for k, v in all_signals.items() 
                       if k != active_strategy and k not in ('bb_ichimoku', 'combined')}
    
    if not other_strategies:
        return 0.6, 'MODERATE'  # Only one strategy available
    
    total_strategies = len(other_strategies)
    agreeing = sum(1 for v in other_strategies.values() if v == active_sig)
    conflicting = sum(1 for v in other_strategies.values() if v in ('BUY', 'SHORT', 'SELL') and v != active_sig)
    
    # Calculate confidence: (agreeing - conflicting) / total
    raw_score = (agreeing - conflicting * 0.5) / total_strategies if total_strategies > 0 else 0.5
    confidence = max(0.0, min(1.0, (raw_score + 1) / 2))  # Normalize to 0-1
    
    # Determine strength label
    if confidence >= 0.75 or agreeing >= 3:
        strength = 'STRONG'
    elif confidence >= 0.5 or agreeing >= 2:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'
    
    return round(confidence, 2), strength


# PERFORMANCE OPTIMIZATION: Cache VIX data globally to avoid fetching for every ticker
_vix_cache = {'data': None, 'time': 0}
VIX_CACHE_TTL = 1800  # 30 minutes

def get_vix_cached():
    """Get VIX data with caching to avoid redundant API calls"""
    global _vix_cache
    now = time.time()
    if _vix_cache['data'] is None or (now - _vix_cache['time']) > VIX_CACHE_TTL:
        try:
            vix_data = safe_history(yf.Ticker("^VIX"), period="60d")
            if len(vix_data) >= 10:
                _vix_cache['data'] = vix_data
                _vix_cache['time'] = now
            else:
                return None
        except Exception:
            return None
    return _vix_cache['data']


# PERFORMANCE OPTIMIZATION: Cache alerts with TTL to avoid constant file reads
_alerts_cache = {"data": None, "time": 0}
CACHE_TTL = 1800  # 30 minutes


def load_alerts():
    """Cached alerts loading with TTL"""
    global _alerts_cache
    now = time.time()
    if _alerts_cache["data"] is None or (now - _alerts_cache["time"]) > CACHE_TTL:
        try:
            with open(ALERTS_FILE) as f:
                _alerts_cache["data"] = tuple(json.load(f))
                _alerts_cache["time"] = now
        except:
            _alerts_cache["data"] = ()
            _alerts_cache["time"] = now
    return _alerts_cache["data"]


def check_alerts(data):
    custom_alerts = load_alerts()
    now = datetime.now(UTC).astimezone(PST)
    (
        high_52w,
        low_52w,
        surge,
        crash,
        volume_spike,
        buy_signals,
        sell_signals,
        short_signals,
        ml_breakout,
        ml_crash,
        custom,
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    stock_dict = {x["ticker"]: x for x in data}

    # --- ROTATION DETECTION LOGIC ---
    rotation_map = {
        "MIDU": "Mid-Cap", "TNA": "Small-Cap", "IWM": "Russell 2000", "DDM": "Dow 2x",
        "^DJI": "Dow 30", "^IXIC": "Nasdaq", "^GSPC": "S&P 500"
    }
    rot_candidates = []
    for sym, label in rotation_map.items():
        s = stock_dict.get(sym)
        if s and s.get("change_3d") is not None and s.get("change_5d") is not None:
            s["rot_score"] = s["change_3d"] - (s["change_5d"] * 0.5)
            s["rot_label"] = label
            rot_candidates.append(s)
    rotation_leader = None
    if rot_candidates:
        rot_candidates.sort(key=lambda x: x["rot_score"], reverse=True)
        if rot_candidates[0]["change_3d"] > 0:
            rotation_leader = rot_candidates[0]
    # ---------------------------------

    for a in custom_alerts:
        s = stock_dict.get(a["ticker"].upper())
        if not s:
            continue
        msg, cond, val = "", a["condition"], a.get("value")
        if cond == "price_above" and val and s["price"] > val:
            msg = f"price ABOVE ${val:.2f} → ${s['price']:.2f}"
        elif cond == "price_below" and val and s["price"] < val:
            msg = f"price BELOW ${val:.2f} → ${s['price']:.2f}"
        elif cond == "day_change_above" and val and s["change_pct"] > val:
            msg = f"DAY % ABOVE {val}% → {s['change_pct']:+.2f}%"
        elif cond == "day_change_below" and val and s["change_pct"] < val:
            msg = f"DAY % BELOW {val}% → {s['change_pct']:+.2f}%"
        elif cond == "rsi_oversold" and s["rsi"] is not None and s["rsi"] < 30:
            msg = f"RSI OVERSOLD → {s['rsi']:.1f}"
        elif cond == "rsi_overbought" and s["rsi"] is not None and s["rsi"] > 70:
            msg = f"RSI OVERBOUGHT → {s['rsi']:.1f}"
        elif cond == "volume_spike" and s["volume_spike"]:
            msg = "VOLUME SPIKE"
        elif cond == "buy":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "BUY":
                buy_signals.append({"ticker": s["ticker"], "msg": f"Custom: BUY signal"})
        elif cond == "sell":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "SELL":
                sell_signals.append({"ticker": s["ticker"], "msg": f"Custom: SELL signal"})
        elif cond == "short":
            active_sig = s.get("active_signal") or s.get("bb_signal")
            if active_sig == "SHORT":
                short_signals.append({"ticker": s["ticker"], "msg": f"Custom: SHORT signal"})
    if msg:
        custom.append({"ticker": s["ticker"], "msg": msg})

    # --- Rotation Leader block will be appended to Signals tab content below ---

    for s in data:
        ch = s["change_pct"]
        if ch > 10:
            surge.append({"ticker": s["ticker"], "msg": f"SURGED > +10% → {ch:+.2f}%"})
        elif ch < -10:
            crash.append({"ticker": s["ticker"], "msg": f"CRASHED < -10% → {ch:+.2f}%"})
        if s["volume_spike"]:
            # Determine volume direction arrow
            arrow = ""
            if "volume_bias" in s:
                try:
                    vb = s["volume_bias"]
                    if vb > 0.05:
                        arrow = "<span style='color:#20813e'>&uarr;</span> "  # green up
                    elif vb < -0.05:
                        arrow = "<span style='color:#c74634'>&darr;</span> "  # red down
                except Exception:
                    pass
            volume_spike.append({"ticker": f"{arrow}{s['ticker']}", "msg": "VOLUME SPIKE"})
        
        # Trading Signal alerts (using active strategy)
        active_sig = s.get("active_signal") or s.get("bb_signal")
        if active_sig == "BUY":
            strategy_name = get_active_strategy().upper()
            buy_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} BUY signal"})
        elif active_sig == "SHORT":
            strategy_name = get_active_strategy().upper()
            short_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} SHORT signal"})
        elif active_sig == "SELL":
            strategy_name = get_active_strategy().upper()
            sell_signals.append({"ticker": s["ticker"], "msg": f"{strategy_name} SELL signal"})
        
        range_52w = s["52w_high"] - s["52w_low"]
        if range_52w > 0:
            pos_pct = (s["price"] - s["52w_low"]) / range_52w * 100
            if pos_pct >= 95:
                high_52w.append(
                    {"ticker": s["ticker"], "msg": f"NEAR 52W HIGH ({pos_pct:.1f}%)"}
                )
            elif pos_pct <= 5:
                low_52w.append(
                    {"ticker": s["ticker"], "msg": f"NEAR 52W LOW ({pos_pct:.1f}%)"}
                )
        
        # ML Breakout and Crash alerts
        ml_score = s.get("ml_breakout_score", 0)
        ml_crash_risk = s.get("ml_crash_risk", 0)
        ml_win_rate = s.get("ml_historical_win_rate", 0)
        ml_expected_return = s.get("ml_expected_return", 0)
        ml_sample_size = s.get("ml_sample_size", 0)
        
        if ml_score >= 70:
            ml_breakout.append({"ticker": s["ticker"], "msg": f"ML BREAKOUT {ml_score:.0f}%"})
        if ml_crash_risk >= 50:
            ml_crash.append({"ticker": s["ticker"], "msg": f"ML CRASH RISK {ml_crash_risk:.0f}%"})
        
        # Historical performance alerts (high confidence predictions with good track record)
        if ml_win_rate >= 70 and ml_sample_size >= 5:
            if ml_score >= 70:
                ml_breakout.append({"ticker": s["ticker"], "msg": f"ML BREAKOUT {ml_win_rate:.0f}% win rate ({ml_sample_size} trades)"})
            if ml_crash_risk >= 50:
                ml_crash.append({"ticker": s["ticker"], "msg": f"ML CRASH {ml_win_rate:.0f}% win rate ({ml_sample_size} trades)"})

    def fmt(items, emoji, label):
        return (
            f"{emoji} <strong>{label}:</strong> {', '.join(a['ticker'] for a in items)}"
        )


    grouped = []
    # --- New: Always render all group links, even if empty, and use a shared details region ---
    accordion_items = []
    accordion_js_contents = []
    idx = 0
    # Add CSS for selected link
    accordion_css = """
    <style>
    .accordion-header { cursor:pointer; padding:2px 8px; border-radius:4px; transition:background 0.2s; }
    .accordion-header.active { background:#e0e7ff; color:#1e40af; font-weight:bold; box-shadow:0 0 2px #1e40af; }
    </style>
    """

    # --- Ordered: Signals | ML Prediction | Surge & Crash | Vol Spike | 52W High/Low | Custom ---
    # Signals
    tickers_buy = ', '.join(a['ticker'] for a in buy_signals) if buy_signals else ''
    tickers_sell = ', '.join(a['ticker'] for a in sell_signals) if sell_signals else ''
    tickers_short = ', '.join(a['ticker'] for a in short_signals) if short_signals else ''
    content = (
        (f"<div><span style='color:#20813e'>🟢 Buy:</span> {tickers_buy}</div>" if tickers_buy else '') +
        (f"<div><span style='color:#d97706'>🟠 Sell:</span> {tickers_sell}</div>" if tickers_sell else '') +
        (f"<div><span style='color:#c74634'>🔴 Short:</span> {tickers_short}</div>" if tickers_short else '')
    )
    # --- Rotation Leader block (append to Signals tab) ---
    rot_content = ""
    if rotation_leader:
        color = "#20813e" if rotation_leader["change_3d"] > 1.5 else "#1e40af"
        rot_content = (
            f"<div style='padding-left: 10px; margin-top:6px;'>"
            f"<strong>🔄 ROTATION LEADER: {rotation_leader['rot_label']} ({rotation_leader['ticker']})</strong><br>"
            f"<span style='font-size:0.9em; color:#444;'>"
            f"Momentum shifting into {rotation_leader['rot_label']}. "
            f"3D Perf: {rotation_leader['change_3d']:+.2f}% vs 5D: {rotation_leader['change_5d']:+.2f}%</span></div>"
        )
    # If no rotation leader, rot_content remains empty
    content += rot_content
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>🚦 Signals</span>")
    accordion_js_contents.append(content)
    idx += 1
    # ML Prediction
    tickers_ml_breakout = ', '.join(a['ticker'] for a in ml_breakout) if ml_breakout else ''
    tickers_ml_crash = ', '.join(a['ticker'] for a in ml_crash) if ml_crash else ''
    content = (
        (f"<div><span style='color:#20813e'>🚀 Breakout:</span> {tickers_ml_breakout}</div>" if tickers_ml_breakout else '') +
        (f"<div><span style='color:#c74634'>⚠️ Crash:</span> {tickers_ml_crash}</div>" if tickers_ml_crash else '')
    )
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>🔮 ML Prediction</span>")
    accordion_js_contents.append(content)
    idx += 1
    # Surge & Crash
    tickers_surge = ', '.join(a['ticker'] for a in surge) if surge else ''
    tickers_crash = ', '.join(a['ticker'] for a in crash) if crash else ''
    content = (
        (f"<div><span style='color:#20813e'>🚀 Surge:</span> {tickers_surge}</div>" if tickers_surge else '') +
        (f"<div><span style='color:#c74634'>💥 Crash:</span> {tickers_crash}</div>" if tickers_crash else '')
    )
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>💣 Surge & Crash</span>")
    accordion_js_contents.append(content)
    idx += 1
    # Vol Spike
    tickers_vol = ', '.join(a['ticker'] for a in volume_spike) if volume_spike else ''
    content = f"<div><span style='color:#c74634'>Vol Spike:</span> {tickers_vol}</div>" if tickers_vol else ''
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>📈 Vol Spike</span>")
    accordion_js_contents.append(content)
    idx += 1
    # 52W High/Low
    tickers_high = ', '.join(a['ticker'] for a in high_52w) if high_52w else ''
    tickers_low = ', '.join(a['ticker'] for a in low_52w) if low_52w else ''
    content = (
        (f"<div><span style='color:#d97706'>🔥 52W High:</span> {tickers_high}</div>" if tickers_high else '') +
        (f"<div><span style='color:#20813e'>📉 52W Low:</span> {tickers_low}</div>" if tickers_low else '')
    )
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>📅 52W High/Low</span>")
    accordion_js_contents.append(content)
    idx += 1
    # Custom (always as a link)
    custom_content = ', '.join(f"{a['ticker']}: {a['msg']}" for a in custom) if custom else ''
    accordion_items.append(f"<span class='accordion-header' onclick='showAccordion({idx})'>⚡ Custom</span>")
    accordion_js_contents.append(f"<div><span style='color:#c74634'>Custom Alerts:</span> {custom_content}</div>" if custom_content else '')
    # Render group links with | separator, no trailing |
    accordion_html = accordion_css + "<div id='alert-accordion' style='display:inline'>" + ' | '.join(accordion_items) + "</div>"
    accordion_html += f"<div id='accordion-details'></div><script>var accordionContents = {json.dumps(accordion_js_contents)};function showAccordion(idx){{document.querySelectorAll('.accordion-header').forEach((el,i)=>el.classList.toggle('active',i===idx));var details=document.getElementById('accordion-details');if(details)details.innerHTML=accordionContents[idx]||'';}}if(accordionContents.length>0)showAccordion(0);</script>"
    grouped.append(accordion_html)
    update_time = datetime.now(UTC).astimezone(PST).strftime("%I:%M:%S %p PST on %B %d, %Y")
    return {"grouped": grouped, "time": update_time}


def rsi(s, return_prev=False):
    """OPTIMIZED: Use EWM instead of rolling mean for faster calculation
    
    Args:
        s: Price series
        return_prev: If True, returns (current_rsi, previous_rsi) tuple
    
    Returns:
        float or tuple: RSI value or (current, previous) if return_prev=True
    """
    if len(s) < 15:
        return (None, None) if return_prev else None
    d = s.diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = g / l
    rsi_series = 100 - 100 / (1 + rs)
    
    if return_prev:
        curr = rsi_series.iloc[-1]
        prev = rsi_series.iloc[-2] if len(rsi_series) > 1 else None
        return curr, prev
    return rsi_series.iloc[-1]


def macd(s, return_prev_label=False):
    """Calculate MACD indicator
    
    Args:
        s: Price series
        return_prev_label: If True, returns previous label as 4th element
    
    Returns:
        tuple: (macd_line, signal_line, label) or (macd_line, signal_line, label, prev_label)
    """
    if len(s) < 26:
        return (None, None, "N/A", None) if return_prev_label else (None, None, "N/A")
    e12, e26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    last_line, last_sig = line.iloc[-1], sig.iloc[-1]
    label = "Bullish" if last_line > last_sig else "Bearish"
    
    if return_prev_label and len(line) > 1:
        prev_line, prev_sig = line.iloc[-2], sig.iloc[-2]
        prev_label = "Bullish" if prev_line > prev_sig else "Bearish"
        return last_line, last_sig, label, prev_label
    elif return_prev_label:
        return last_line, last_sig, label, None
    return last_line, last_sig, label


def na(v, f="{:.2f}"):
    if v is None or pd.isna(v):
        return "N/A"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return "N/A"


def sparkline(prices):
    if len(prices) < 2:
        return ""
    prices = prices[-30:]
    mn, mx = min(prices), max(prices)
    rng = mx - mn if mx != mn else 1
    w, h = 60, 20
    pts = [
        f"{(i/(len(prices)-1))*w:.1f},{h-((p-mn)/rng*h):.1f}"
        for i, p in enumerate(prices)
    ]
    c = "#00aa00" if prices[-1] >= prices[0] else "#cc0000"
    return f'<svg width="{w}" height="{h}"><polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="1.5"/></svg>'


# Safe wrapper for yfinance history calls to avoid hard failures (e.g., 401 Unauthorized)
def safe_history(ticker_obj, **kwargs):
    try:
        return ticker_obj.history(**kwargs)
    except Exception:
        return pd.DataFrame()


def fetch(ticker, ext=False, retry=0):
    try:
        # enforce global rate limit before starting network activity
        RATE_LIMITER.wait()

        t = yf.Ticker(ticker)
        # Small base delay with jitter to spread requests a bit
        # Reduced from 0.25 to 0.1 for faster execution with rate limiter protection
        jitter = random.uniform(0, FETCH_JITTER_MAX)
        time.sleep(FETCH_BASE_DELAY + jitter + (retry * RETRY_DELAY_MULTIPLIER))

        info = get_ticker_info_cached(ticker)
        if info is None:
            info = {}
        
        # OPTIMIZATION: Fetch longer history once and slice it for all needs
        # This reduces API calls from ~8 to ~3 per ticker
        h_all = safe_history(t, period="1y")  # Get 1 year for all calculations
        h_3y = safe_history(t, period="3y")  # Get 3 years for 3-year returns
        h_day = safe_history(t, period="1d", interval="1m", prepost=ext)
        
        if h_day.empty:
            h_day = safe_history(t, period="5d", prepost=ext)
        if h_day.empty:
            return None

        price = h_day["Close"].iloc[-1]
        day_low, day_high = h_day["Low"].min(), h_day["High"].max()
        
        # Slice from h_all instead of separate API calls
        reg = safe_history(t, period="5d", prepost=False)  # Still need this for prepost=False
        h30 = h_all.tail(60) if not h_all.empty else pd.DataFrame()  # Last 60 days
        h6m = h_all.tail(180) if not h_all.empty else pd.DataFrame()  # Last ~6 months
        h3m = h_all.tail(63) if not h_all.empty else pd.DataFrame()  # Last ~3 months (63 trading days)
        h1m = h_all.tail(21) if not h_all.empty else pd.DataFrame()  # Last ~1 month (21 trading days)
        
        change_pct = change_abs_day = 0.0
        if len(reg) >= 2:
            prev = reg["Close"].iloc[-2]
            if prev and prev > 0:
                change_pct = ((price - prev) / prev) * 100
                change_abs_day = price - prev
        # 5-day change: compare to oldest close in the 5-day window when available
        change_5d = None
        change_abs_5d = None
        try:
            if len(reg) >= 2:
                ref5 = reg["Close"].iloc[0]
                if ref5 and ref5 > 0:
                    change_5d = ((price - ref5) / ref5) * 100
                    change_abs_5d = price - ref5
        except Exception:
            change_5d = None
            change_abs_5d = None

        # YTD calculation: Use h_all data already fetched
        ytd = pd.DataFrame()
        if not h_all.empty:
            current_year = datetime.now().year
            # Filter for current year data
            ytd = h_all[h_all.index.year == current_year]
            # If we have current year data but need baseline from previous year
            if len(ytd) >= 1:
                prev_year_data = h_all[h_all.index.year == current_year - 1]
                if not prev_year_data.empty:
                    # Add last day of previous year as first row for comparison
                    ytd = pd.concat([prev_year_data.tail(1), ytd])

        def calc_ch(h):
            if len(h) >= 2 and h["Close"].iloc[0] > 0:
                return (
                    (price - h["Close"].iloc[0]) / h["Close"].iloc[0]
                ) * 100, price - h["Close"].iloc[0]
            return None, None

        ch1m, abs1m = calc_ch(h1m)
        ch3m, abs3m = calc_ch(h3m)
        ch6m, abs6m = calc_ch(h6m)
        chytd, absytd = calc_ch(ytd)
        ch1y, abs1y = calc_ch(h_all)  # 1-year change
        ch3y, abs3y = calc_ch(h_3y)  # 3-year change
        
        # Calculate $10k investment value from 3 years ago
        value_10k = None
        if ch3y is not None:
            value_10k = 10000 * (1 + ch3y / 100)

        high52, low52 = (
            (h_all["High"].max(), h_all["Low"].min()) if not h_all.empty else (price, price)
        )
        vol = h_day["Volume"].sum()

        hv = None
        if len(h30) >= 30:
            r = h30["Close"].pct_change().dropna()
            if len(r) > 1:
                hv = r.std() * (252**0.5) * 100

        # --- Enhanced Yahoo Finance attribute mapping and data quality checks ---
        def safe_get(d, key, default=None):
            v = d.get(key, default)
            if v is None or (isinstance(v, float) and (pd.isna(v) or v != v)):
                return default
            return v

        short_pct = safe_get(info, "shortPercentOfFloat")
        if short_pct is not None:
            short_pct *= 100

        days_cover = None
        shares_short = safe_get(info, "sharesShort")
        avg_vol_10d = safe_get(info, "averageDailyVolume10Day")
        avg_vol = safe_get(info, "averageVolume")
        avg = avg_vol_10d or avg_vol or vol or 1
        if shares_short is not None and avg > 0:
            days_cover = shares_short / avg

        squeeze = "None"
        if short_pct is not None and days_cover is not None:
            if short_pct > 30 and days_cover > 10:
                squeeze = "Extreme"
            elif short_pct > 20 and days_cover > 7:
                squeeze = "High"
            elif short_pct > 15 and days_cover > 5:
                squeeze = "Moderate"

        # --- Map additional Yahoo Finance attributes for richer analytics ---
        pe_ratio = safe_get(info, "trailingPE")
        eps = safe_get(info, "trailingEps")
        target_mean_price = safe_get(info, "targetMeanPrice")
        dividend_rate = safe_get(info, "dividendRate")
        ex_dividend_date = safe_get(info, "exDividendDate")
        beta = safe_get(info, "beta")
        bid = safe_get(info, "bid")
        ask = safe_get(info, "ask")
        day_low = safe_get(info, "dayLow")
        day_high = safe_get(info, "dayHigh")
        fifty_two_week_low = safe_get(info, "fiftyTwoWeekLow")
        fifty_two_week_high = safe_get(info, "fiftyTwoWeekHigh")

        # Option chain attributes (if available)
        implied_vol = None
        open_interest = None
        in_the_money = None
        expiration_dates = None
        try:
            opts = getattr(t, "options", None)
            if opts:
                expiration_dates = opts
                exp_date = opts[0]
                try:
                    chain = t.option_chain(exp_date)
                    # ATM strike logic as before
                    strikes = pd.concat([
                        chain.calls["strike"], chain.puts["strike"]
                    ]).unique()
                    if len(strikes) > 0:
                        atm = min(strikes, key=lambda s: abs(s - price))
                        # Implied volatility, open interest, in the money for ATM call/put
                        call_row = chain.calls[chain.calls["strike"] == atm]
                        put_row = chain.puts[chain.puts["strike"] == atm]
                        if not call_row.empty:
                            implied_vol = safe_get(call_row.iloc[0], "impliedVolatility")
                            open_interest = safe_get(call_row.iloc[0], "openInterest")
                            in_the_money = safe_get(call_row.iloc[0], "inTheMoney")
                        elif not put_row.empty:
                            implied_vol = safe_get(put_row.iloc[0], "impliedVolatility")
                            open_interest = safe_get(put_row.iloc[0], "openInterest")
                            in_the_money = safe_get(put_row.iloc[0], "inTheMoney")
                except Exception:
                    pass
        except Exception:
            pass

        # --- End of enhanced mapping ---

        rsi_val, rsi_prev = rsi(h30["Close"], return_prev=True) if not h30.empty else (None, None)
        
        # OPTIMIZATION: Use h_all for MACD to avoid separate API call
        macd_period = int(os.getenv('MACD_PERIOD', '100'))
        # Use h30 (60 days) or h_all for MACD - most periods need ~100 days
        h_macd = h_all if not h_all.empty else pd.DataFrame()
        if not h_macd.empty:
            macd_val, macd_sig, macd_lbl, macd_prev_lbl = macd(h_macd["Close"], return_prev_label=True)
        else:
            macd_val, macd_sig, macd_lbl, macd_prev_lbl = None, None, "N/A", None

        vol_spike = False
        if len(h30) > 1:
            avg = h30["Volume"][:-1].mean()
            if avg > 0:
                vol_spike = vol > 1.5 * avg

        pc_ratio = impl_move = impl_hi = impl_lo = exp_date = None
        # Options endpoints increasingly return 401; wrap fully and degrade gracefully
        try:
            opts = getattr(t, "options", None)
            if opts:
                exp_date = opts[0]
                try:
                    chain = t.option_chain(exp_date)
                    strikes = pd.concat(
                        [chain.calls["strike"], chain.puts["strike"]]
                    ).unique()
                    if len(strikes) > 0:
                        atm = min(strikes, key=lambda s: abs(s - price))
                        cp = (
                            chain.calls.loc[
                                chain.calls["strike"] == atm, "lastPrice"
                            ].iloc[0]
                            if not chain.calls[chain.calls["strike"] == atm].empty
                            else 0
                        )
                        pp = (
                            chain.puts.loc[
                                chain.puts["strike"] == atm, "lastPrice"
                            ].iloc[0]
                            if not chain.puts[chain.puts["strike"] == atm].empty
                            else 0
                        )
                        straddle = cp + pp
                        if straddle > 0 and price > 0:
                            impl_move = round((straddle / price) * 100, 2)
                            cons = impl_move * 0.85
                            impl_hi = price * (1 + cons / 100)
                            impl_lo = price * (1 - cons / 100)
                            cvol, pvol = (
                                chain.calls["volume"].fillna(0).sum(),
                                chain.puts["volume"].fillna(0).sum(),
                            )
                            if cvol > 0:
                                pc_ratio = pvol / cvol
                except Exception:
                    pass
        except Exception:
            # ignore options entirely on failure (e.g., 401 Unauthorized)
            pass

        down_bias = False
        volume_bias = 0.0
        if len(h30) > 0:
            down_vol = h30[h30["Close"] < h30["Open"]]["Volume"].sum()
            up_vol = h30[h30["Close"] > h30["Open"]]["Volume"].sum()
            total_vol = down_vol + up_vol
            if total_vol > 0:
                volume_bias = (down_vol - up_vol) / total_vol
            down_bias = down_vol > up_vol

        opt_dir = "Neutral"
        if pc_ratio:
            if pc_ratio > 1.2 and down_bias:
                opt_dir = "Strong Bearish"
            elif pc_ratio > 1.0 or down_bias:
                opt_dir = "Bearish"
            elif pc_ratio < 0.8 and not down_bias:
                opt_dir = "Bullish"

        rec_mean = info.get("recommendationMean", 5)
        sentiment = ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")[
            (
                0
                if rec_mean <= 1.5
                else (
                    1
                    if rec_mean <= 2.5
                    else 2 if rec_mean <= 3.5 else 3 if rec_mean <= 4.5 else 4
                )
            )
        ]

        rating = info.get("recommendationKey", "none").title().replace("_", " ")
        target = info.get("targetMeanPrice")
        upside = ((target - price) / price) * 100 if target and price > 0 else None
        spk = sparkline(h30["Close"].tolist() if not h30.empty else [])
        spk_5d = sparkline(reg["Close"].tolist() if not reg.empty else [])
        spk_1m = sparkline(h1m["Close"].tolist() if not h1m.empty else [])
        spk_3m = sparkline(h3m["Close"].tolist() if not h3m.empty else [])
        spk_6m = sparkline(h6m["Close"].tolist() if not h6m.empty else [])
        spk_ytd = sparkline(ytd["Close"].tolist() if not ytd.empty else [])
        spk_1y = sparkline(h_all["Close"].tolist() if not h_all.empty else [])
        spk_3y = sparkline(h_3y["Close"].tolist() if not h_3y.empty else [])
        spk_vol = sparkline(h30["Volume"].tolist() if not h30.empty else [])

        bb_period = 20
        bb_upper = bb_lower = bb_middle = bb_width_pct = bb_position_pct = bb_position_prev = bb_status = (
            None
        )
        if len(h30) >= bb_period:
            close = h30["Close"]
            bb_middle = close.rolling(window=bb_period).mean().iloc[-1]
            std_dev = close.rolling(window=bb_period).std().iloc[-1]
            bb_upper = bb_middle + (std_dev * 2)
            bb_lower = bb_middle - (std_dev * 2)
            if bb_middle > 0:
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
            if (bb_upper - bb_lower) > 0:
                bb_position_pct = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
                bb_position_pct = max(0, min(100, bb_position_pct))
                
                # Calculate previous BB position for trend detection
                if len(h30) > bb_period:
                    price_prev_bb = h30["Close"].iloc[-2]
                    bb_middle_prev = close.rolling(window=bb_period).mean().iloc[-2]
                    std_dev_prev = close.rolling(window=bb_period).std().iloc[-2]
                    bb_upper_prev = bb_middle_prev + (std_dev_prev * 2)
                    bb_lower_prev = bb_middle_prev - (std_dev_prev * 2)
                    if (bb_upper_prev - bb_lower_prev) > 0:
                        bb_position_prev = ((price_prev_bb - bb_lower_prev) / (bb_upper_prev - bb_lower_prev)) * 100
                        bb_position_prev = max(0, min(100, bb_position_prev))
            
            bb_status = (
                "Above Upper"
                if price > bb_upper
                else "Below Lower" if price < bb_lower else "Inside"
            )

        # Calculate Ichimoku Cloud indicators
        ichimoku_data = None
        vol_sma_20 = None
        price_sma_60 = None
        price_prev = None
        sma_50 = None
        sma_200 = None
        death_cross = False
        golden_cross = False
        
        if len(h_all) >= 60:
            ichimoku_data = calculate_ichimoku(h_all["High"], h_all["Low"], h_all["Close"])
            
            # Calculate volume SMA(20)
            if len(h30) >= 20:
                vol_sma_20 = h30["Volume"].rolling(window=20).mean().iloc[-1]
            
            # Calculate price SMA(60)
            price_sma_60 = h_all["Close"].rolling(window=60).mean().iloc[-1]
            
            # Get previous close for crossover detection
            if len(h_all) >= 2:
                price_prev = h_all["Close"].iloc[-2]
        
        # Calculate SMA 50 and SMA 200 for moving average analysis
        if len(h_all) >= 50:
            sma_50 = h_all["Close"].rolling(window=50).mean().iloc[-1]
        
        if len(h_all) >= 200:
            sma_200 = h_all["Close"].rolling(window=200).mean().iloc[-1]
            
            # Death cross: SMA 50 crosses below SMA 200 (bearish)
            # Golden cross: SMA 50 crosses above SMA 200 (bullish)
            if sma_50 is not None and len(h_all) >= 201:
                sma_50_prev = h_all["Close"].rolling(window=50).mean().iloc[-2]
                sma_200_prev = h_all["Close"].rolling(window=200).mean().iloc[-2]
                
                if sma_50_prev >= sma_200_prev and sma_50 < sma_200:
                    death_cross = True
                elif sma_50_prev <= sma_200_prev and sma_50 > sma_200:
                    golden_cross = True

        # Generate trading signals using strategy framework
        signal_data = {
            'bb_position_pct': bb_position_pct,
            'bb_position_prev': bb_position_prev,
            'rsi': rsi_val,
            'rsi_prev': rsi_prev,
            'macd_label': macd_lbl,
            'macd_prev_label': macd_prev_lbl,
            'ichimoku': ichimoku_data,
            'price': price,
            'price_prev': price_prev,
            'vol_sma_20': vol_sma_20,
            'price_sma_60': price_sma_60,
        }
        all_signals = generate_trading_signals(signal_data)
        active_strategy = get_active_strategy()
        primary_signal = all_signals.get(active_strategy)
        
        # Calculate signal confidence and strength
        signal_confidence, signal_strength = calculate_signal_confidence(all_signals, active_strategy)
        
        # Calculate risk management parameters (ATR-based stop loss)
        # ALWAYS calculate ATR for all tickers (not just BUY/SHORT) so trade setups can be shown
        atr_value = None
        stop_loss = None
        risk_reward_ratio = None
        position_size_pct = None
        
        if len(h30) >= 14:
            # Calculate ATR (Average True Range) for risk management
            high = h30['High']
            low = h30['Low']
            close = h30['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_value = tr.rolling(window=14).mean().iloc[-1]
            
            # Calculate stop loss and position sizing for all tickers with ATR data
            if atr_value and atr_value > 0:
                # Configurable ATR multiplier for stop loss
                ATR_STOP_MULTIPLIER = float(os.getenv('ATR_STOP_MULTIPLIER', '2.0'))
                RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '2.0'))  # % of account
                
                # Default to LONG position for stop loss calculation
                stop_loss = price - (atr_value * ATR_STOP_MULTIPLIER)
                # Target: 2x risk (configurable)
                target_price = price + (atr_value * ATR_STOP_MULTIPLIER * 2)
                risk_reward_ratio = 2.0
                
                # Calculate position size (% of account to risk)
                if stop_loss:
                    risk_per_share = abs(price - stop_loss)
                    if risk_per_share > 0:
                        # Position size to risk X% of account
                        position_size_pct = RISK_PER_TRADE / (risk_per_share / price * 100)
                        position_size_pct = min(position_size_pct, 25.0)  # Max 25% position
        
        # Calculate OBV (On Balance Volume)
        obv = None
        if len(h30) >= 2:
            sign = (h30["Close"] > h30["Close"].shift(1)).astype(int) * 2 - 1
            obv = (h30["Volume"] * sign).cumsum().iloc[-1]
        
        # Calculate Stochastic Oscillator
        stoch_k = stoch_d = None
        if len(h30) >= 14:
            low_14 = h30["Low"].rolling(14).min()
            high_14 = h30["High"].rolling(14).max()
            k = ((h30["Close"] - low_14) / (high_14 - low_14)) * 100
            stoch_k = k.iloc[-1]
            if len(k) >= 3:
                stoch_d = k.rolling(3).mean().iloc[-1]
        
        # Calculate additional indicators for ML
        adx = cci = mfi = williams_r = roc = vol_roc = None
        
        if len(h30) >= 14:
            # Williams %R
            high_14 = h30["High"].rolling(14).max()
            low_14 = h30["Low"].rolling(14).min()
            williams_r = ((high_14 - h30["Close"]) / (high_14 - low_14)) * -100
            williams_r = williams_r.iloc[-1]
            
            # ROC (14-period)
            if len(h30) >= 28:  # Need enough data for 14-period ROC
                roc = ((h30["Close"] - h30["Close"].shift(14)) / h30["Close"].shift(14)) * 100
                roc = roc.iloc[-1]
            
            # Volume ROC (14-period)
            if len(h30) >= 28:
                vol_roc = ((h30["Volume"] - h30["Volume"].shift(14)) / h30["Volume"].shift(14)) * 100
                vol_roc = vol_roc.iloc[-1]
        
        if len(h30) >= 20:
            # CCI (20-period)
            tp = (h30["High"] + h30["Low"] + h30["Close"]) / 3
            sma_tp = tp.rolling(20).mean()
            mad = (tp - sma_tp).abs().rolling(20).mean()
            cci = (tp - sma_tp) / (0.015 * mad)
            cci = cci.iloc[-1]
            
            # MFI (14-period Money Flow Index)
            typical_price = (h30["High"] + h30["Low"] + h30["Close"]) / 3
            money_flow = typical_price * h30["Volume"]
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            positive_mf = positive_flow.rolling(14).sum()
            negative_mf = negative_flow.rolling(14).sum()
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
            mfi = mfi.iloc[-1]
        
        # ADX calculation (14-period)
        if len(h30) >= 28:  # Need enough data for ADX
            high = h30["High"]
            low = h30["Low"]
            close = h30["Close"]
            
            # True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = high - high.shift(1)
            dm_minus = low.shift(1) - low
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Smoothed averages
            atr = tr.rolling(14).mean()
            di_plus = 100 * (dm_plus.rolling(14).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(14).mean() / atr)
            
            dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
            adx = dx.rolling(14).mean().iloc[-1]
        
        # Legacy bb_signal for backward compatibility (uses active strategy)
        bb_signal = primary_signal

        # Predict trend based on technical indicators (with configurable thresholds)
        # Strategy-aware weighting: More reliable strategies get higher weight for active signal
        strategy_trend_weights = {
            'ichimoku': 2.5,  # Most comprehensive
            'combined': 2.0,  # Multiple confirmations
            'macd': 1.5,      # Strong trend indicator
            'bb': 1.2,        # Good momentum indicator
            'bb_ichimoku': 1.8,  # Two-factor confirmation
            'rsi': 1.0        # Less reliable alone
        }
        
        # Configurable momentum thresholds (adjust for volatility)
        TREND_MOMENTUM_THRESHOLD = float(os.getenv('TREND_MOMENTUM_THRESHOLD', '2.0'))
        
        trend_score = 0.0
        
        # MACD signal (weight: 2) - Only if not the active strategy to avoid double-counting
        if active_strategy != 'macd':
            if macd_lbl == 'Bullish':
                trend_score += 2.0
            elif macd_lbl == 'Bearish':
                trend_score -= 2.0
        
        # RSI - Current momentum direction (weight: 1.0)
        # Fixed: RSI > 50 indicates upward momentum, RSI < 50 indicates downward momentum
        if rsi_val is not None:
            if rsi_val > 60:  # Strong upward momentum
                trend_score += 1.0
            elif rsi_val > 50:  # Mild upward momentum
                trend_score += 0.5
            elif rsi_val < 40:  # Strong downward momentum
                trend_score -= 1.0
            elif rsi_val < 50:  # Mild downward momentum
                trend_score -= 0.5
        
        # BB Position - Current trend direction (weight: 1.0)
        # Fixed: High BB% indicates uptrend, low BB% indicates downtrend
        if bb_position_pct is not None:
            if bb_position_pct > 80:  # Strong uptrend
                trend_score += 1.0
            elif bb_position_pct > 60:  # Moderate uptrend
                trend_score += 0.5
            elif bb_position_pct < 20:  # Strong downtrend
                trend_score -= 1.0
            elif bb_position_pct < 40:  # Moderate downtrend
                trend_score -= 0.5
        
        # Ichimoku Cloud trend (weight: 2.0 - most reliable trend indicator)
        if ichimoku_data and active_strategy != 'ichimoku':
            conv = ichimoku_data.get('conversion_line')
            base = ichimoku_data.get('base_line')
            span_a = ichimoku_data.get('span_a')
            span_b = ichimoku_data.get('span_b')
            
            if all([price, span_a, span_b, conv, base]):
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)
                
                # Price above cloud = bullish trend
                if price > cloud_top:
                    trend_score += 2.0
                    # TK cross confirmation (Tenkan-Kijun)
                    if conv > base:
                        trend_score += 0.5
                # Price below cloud = bearish trend
                elif price < cloud_bottom:
                    trend_score -= 2.0
                    # TK cross confirmation
                    if conv < base:
                        trend_score -= 0.5
                # Price in cloud = neutral/consolidation (no change to score)
        
        # Active Signal (dynamic weight based on strategy reliability)
        active_weight = strategy_trend_weights.get(active_strategy, 1.5)
        if primary_signal == 'BUY':
            trend_score += active_weight
        elif primary_signal in ('SELL', 'SHORT'):
            trend_score -= active_weight
        
        # Price momentum (configurable threshold)
        if change_pct is not None:
            if change_pct > TREND_MOMENTUM_THRESHOLD:
                trend_score += 1.0
            elif change_pct < -TREND_MOMENTUM_THRESHOLD:
                trend_score -= 1.0
        
        # Determine trend based on score (adjusted thresholds for new scoring)
        if trend_score >= 4:
            predicted_trend = "↑"  # Strong uptrend
            trend_label = "UP"
        elif trend_score >= 1.5:
            predicted_trend = "↗"  # Moderate uptrend
            trend_label = "UP"
        elif trend_score <= -4:
            predicted_trend = "↓"  # Strong downtrend
            trend_label = "DOWN"
        elif trend_score <= -1.5:
            predicted_trend = "↘"  # Moderate downtrend
            trend_label = "DOWN"
        else:
            predicted_trend = "→"  # Neutral/sideways
            trend_label = "NEUTRAL"

        # CVR3 VIX Signal: Generate BUY/SELL/SHORT based on VIX (market, not individual ticker)
        # VIX interpretation: HIGH VIX = fear/volatility = contrarian BUY opportunity
        cvr3_vix_signal = None
        cvr3_vix_pct = None
        cvr3_vix_value = None
        try:
            vix_data = get_vix_cached()
            if vix_data is not None and len(vix_data) >= 10:
                vix_close_prices = vix_data["Close"]
                vix_sma = vix_close_prices.rolling(window=10).mean().iloc[-1]
                if vix_sma > 0:
                    current_vix = vix_close_prices.iloc[-1]
                    prev_vix = (
                        vix_close_prices.iloc[-2]
                        if len(vix_close_prices) >= 2
                        else current_vix
                    )
                    cvr3_vix_value = current_vix
                    cvr3_vix_pct = current_vix - prev_vix
                    pct_diff = ((current_vix - vix_sma) / vix_sma) * 100
                    
                    # Contrarian VIX signals (configurable thresholds)
                    VIX_EXTREME_FEAR = float(os.getenv('VIX_EXTREME_FEAR', '20'))  # Strong buy
                    VIX_MODERATE_FEAR = float(os.getenv('VIX_MODERATE_FEAR', '10'))  # Buy
                    VIX_COMPLACENCY = float(os.getenv('VIX_COMPLACENCY', '-10'))  # Sell
                    VIX_EXTREME_COMPLACENCY = float(os.getenv('VIX_EXTREME_COMPLACENCY', '-20'))  # Short
                    
                    if pct_diff >= VIX_EXTREME_FEAR:
                        # Extreme fear (VIX spike) = Best contrarian BUY opportunity
                        cvr3_vix_signal = "BUY"
                    elif pct_diff >= VIX_MODERATE_FEAR:
                        # Moderate fear = Good contrarian BUY opportunity
                        cvr3_vix_signal = "BUY"
                    elif pct_diff <= VIX_EXTREME_COMPLACENCY:
                        # Extreme complacency (very low VIX) = SHORT opportunity
                        cvr3_vix_signal = "SHORT"
                    elif pct_diff <= VIX_COMPLACENCY:
                        # Complacency (low VIX) = SELL/reduce exposure
                        cvr3_vix_signal = "SELL"
        except Exception:
            cvr3_vix_signal = None
            cvr3_vix_pct = None
            cvr3_vix_value = None

        div_rate = info.get("dividendRate")
        div_yield = info.get("dividendYield")
        ex_dividend_date = info.get("exDividendDate")
        payout_ratio = info.get("payoutRatio")
        dividend_growth = info.get("dividendGrowthRate")
        
        # Format ex-dividend date
        ex_div_date_str = None
        if ex_dividend_date:
            try:
                if isinstance(ex_dividend_date, (int, float)):
                    dt = datetime.fromtimestamp(int(ex_dividend_date), UTC).astimezone(PST)
                    ex_div_date_str = dt.strftime("%b %d, %Y")
            except Exception:
                ex_div_date_str = None
        
        # Determine payout frequency from info
        payout_frequency = None
        if info.get("dividendYield") and div_rate:
            # Common frequencies: Annual, Quarterly, Monthly, Semi-Annual
            # Try to get from info first
            payout_frequency = info.get("dividendsPerShare") or info.get("payoutFrequency")
            if not payout_frequency:
                # Estimate from available data if not provided
                # This is a rough estimate - actual data may vary
                payout_frequency = "Quarterly"  # Most common for US stocks

        # Do not normalize dividend yield — render the raw value as provided
        # by the data source. Keep `div_yield` unchanged.

        # Extract earnings date (display string) and ISO date for JS filtering
        earnings_date = None
        earnings_date_iso = None
        try:

            def _parse_earnings_display_and_iso(val):
                if val is None:
                    return None, None
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                # numeric epoch
                if isinstance(val, (int, float)):
                    dt = datetime.fromtimestamp(int(val), UTC).astimezone(PST)
                    return dt.strftime("%m/%d/%Y"), dt.date().isoformat()
                if isinstance(val, str):
                    s = val.strip()
                    if s.isdigit():
                        dt = datetime.fromtimestamp(int(s), UTC).astimezone(PST)
                        return dt.strftime("%m/%d/%Y"), dt.date().isoformat()
                    try:
                        d = datetime.fromisoformat(s.split("T")[0])
                        d = d.replace(tzinfo=UTC)
                        dt = d.astimezone(PST)
                        return dt.strftime("%m/%d/%Y"), dt.date().isoformat()
                    except Exception:
                        try:
                            d = datetime.strptime(s.split("T")[0], "%Y-%m-%d")
                            d = d.replace(tzinfo=UTC)
                            dt = d.astimezone(PST)
                            return dt.strftime("%m/%d/%Y"), dt.date().isoformat()
                        except Exception:
                            return None, None
                return None, None

            for k in (
                "earningsTimestamp",
                "earningsTimestampStart",
                "earningsDate",
                "nextEarningsDate",
            ):
                ed = info.get(k)
                disp, iso = _parse_earnings_display_and_iso(ed)
                if disp:
                    earnings_date = disp
                    earnings_date_iso = iso
                    break
        except Exception:
            earnings_date = None
            earnings_date_iso = None

        pe = info.get("trailingPE") or info.get("forwardPE")
        eps = (
            info.get("trailingEps")
            or info.get("epsTrailingTwelveMonths")
            or info.get("earningsPerShare")
            or info.get("forwardEps")
        )
        market_cap = info.get("marketCap")
        aum = (
            info.get("totalAssets")
            or info.get("fundTotalAssets")
            or info.get("total_assets")
        )

        tu = ticker.upper()
        category = infer_category_from_info(tu, info) or get_category(tu)
        quote_type = (info.get("quoteType") or "").upper()
       
        # --- Early Rotation Detection: 3d/5d momentum ---
        if len(h_all) >= 6:
            # 3-day change (current vs 3 sessions ago)
            p_3d = h_all['Close'].iloc[-4]
            change_3d = ((price - p_3d) / p_3d) * 100
            # 5-day change (current vs 5 sessions ago)
            p_5d = h_all['Close'].iloc[-6]
            change_5d = ((price - p_5d) / p_5d) * 100
        else:
            change_3d = None
            change_5d = None

        result = {
            "ticker": tu,
            "quote_type": quote_type,
            "price": price,
            "change_pct": change_pct,
            "change_abs_day": change_abs_day,
            "change_1m": ch1m,
            "change_abs_1m": abs1m,
            "change_3m": ch3m,
            "change_abs_3m": abs3m,
            "change_5d": change_5d,
            "change_abs_5d": change_abs_5d,
            "change_6m": ch6m,
            "change_abs_6m": abs6m,
            "change_ytd": chytd,
            "change_abs_ytd": absytd,
            "change_3d": change_3d,
            "change_1y": ch1y,
            "change_abs_1y": abs1y,
            "change_3y": ch3y,
            "change_abs_3y": abs3y,
            "value_10k_3y": value_10k,
            "volume": vol,
            "volume_raw": vol,
            "52w_high": high52,
            "52w_low": low52,
            "day_low": day_low,
            "day_high": day_high,
            "short_percent": short_pct,
            "days_to_cover": days_cover,
            "squeeze_level": squeeze,
            "rsi": rsi_val,
            "macd_label": macd_lbl,
            "volume_spike": vol_spike,
            "is_meme_stock": tu in MEME_STOCKS,
            "sentiment": sentiment,
            "analyst_rating": rating,
            "target_price": target,
            "upside_potential": upside,
            "options_direction": opt_dir,
            "implied_move_pct": impl_move,
            "implied_high": impl_hi,
            "implied_low": impl_lo,
            "down_volume_bias": down_bias,
            "sparkline": spk,
            "sparkline_5d": sparkline(reg["Close"].tolist() if not reg.empty else []),
            "sparkline_1m": spk_1m,
            "sparkline_3m": spk_3m,
            "sparkline_6m": spk_6m,
            "sparkline_ytd": spk_ytd,
            "sparkline_1y": spk_1y,
            "sparkline_3y": spk_3y,
            "sparkline_vol": spk_vol,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "bb_width_pct": bb_width_pct,
            "bb_position_pct": bb_position_pct,
            "bb_status": bb_status,
            "bb_signal": bb_signal,
            "signal_bb": all_signals.get('bb'),
            "signal_rsi": all_signals.get('rsi'),
            "signal_macd": all_signals.get('macd'),
            "signal_ichimoku": all_signals.get('ichimoku'),
            "signal_combined": all_signals.get('combined'),
            "signal_bb_ichimoku": all_signals.get('bb_ichimoku'),
            "active_signal": primary_signal,
            "predicted_trend": predicted_trend,
            "trend_label": trend_label,
            "ichimoku_conversion": ichimoku_data.get('conversion_line') if ichimoku_data else None,
            "ichimoku_base": ichimoku_data.get('base_line') if ichimoku_data else None,
            "ichimoku_span_a": ichimoku_data.get('span_a') if ichimoku_data else None,
            "ichimoku_span_b": ichimoku_data.get('span_b') if ichimoku_data else None,
            "cvr3_vix_signal": cvr3_vix_signal,
            "cvr3_vix_pct": cvr3_vix_pct,
            "cvr3_vix_value": cvr3_vix_value,
            "hv_30_annualized": hv,
            "macd_line": macd_val,
            "macd_signal": macd_sig,
            "macd_label": macd_lbl,
            "pc_ratio": pc_ratio,
            "pe": pe,
            "eps": eps,
            "dividend_rate": div_rate,
            "dividend_yield": div_yield,
            "ex_dividend_date": ex_div_date_str,
            "payout_ratio": payout_ratio,
            "payout_frequency": payout_frequency,
            "dividend_growth": dividend_growth,
            "earnings_date": earnings_date,
            "earnings_date_iso": earnings_date_iso,
            "market_cap": market_cap,
            "aum": aum,
            "category": category,
            # Risk management and confidence metrics
            "atr_14": atr_value,
            "stop_loss_price": stop_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "position_size_pct": position_size_pct,
            "signal_confidence": signal_confidence,
            "signal_strength": signal_strength,
            "obv": obv,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "adx": adx,
            "cci": cci,
            "mfi": mfi,
            "williams_r": williams_r,
            "roc": roc,
            "vol_roc": vol_roc,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "death_cross": death_cross,
            "golden_cross": golden_cross,
            "volume_bias": volume_bias,
            "trend_score": trend_score,
        }
        
        # ML Predictions (if available)
        if ML_AVAILABLE:
            try:
                ml_pred = predict_breakout_crash(result)
                result["ml_breakout_score"] = ml_pred['breakout_score']
                result["ml_crash_risk"] = ml_pred['crash_risk']
                result["ml_prediction"] = ml_pred['prediction']
                result["ml_confidence"] = ml_pred['confidence']
                result["ml_historical_win_rate"] = ml_pred.get('historical_win_rate', 0.0)
                result["ml_expected_return"] = ml_pred.get('expected_return', 0.0)
                result["ml_sample_size"] = ml_pred.get('sample_size', 0)
            except Exception as e:
                result["ml_breakout_score"] = 0
                result["ml_crash_risk"] = 0
                result["ml_prediction"] = "N/A"
                result["ml_confidence"] = 0
                result["ml_historical_win_rate"] = 0.0
                result["ml_expected_return"] = 0.0
                result["ml_sample_size"] = 0
        else:
            result["ml_breakout_score"] = 0
            result["ml_crash_risk"] = 0
            result["ml_prediction"] = "N/A"
            result["ml_confidence"] = 0
            result["ml_historical_win_rate"] = 0.0
            result["ml_expected_return"] = 0.0
            result["ml_sample_size"] = 0
        
        return result
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limit" in error_msg:
            if retry < MAX_RETRIES:
                # exponential backoff with jitter
                wait_time = (2**retry) * 5 + random.uniform(0, 3)
                time.sleep(wait_time)
                return fetch(ticker, ext, retry + 1)
            else:
                return None
        elif "Unauthorized" in error_msg or "401" in error_msg:
            # Yahoo Finance feature gated; skip this ticker gracefully
            return None
        else:
            time.sleep(5)
            return None


def fmt_vol(v):
    if v is None:
        return "N/A"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return str(int(v))


def fmt_mcap(v):
    if v is None or pd.isna(v):
        return "N/A"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v >= 1e12:
        return f"{v/1e12:.2f}T"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    return str(int(v))


def fmt_change(p, a=None):
    if p is None:
        return '<span class="neutral" data-sort="-999999">N/A</span>'
    sign, cls = ("▲", "positive") if p >= 0 else ("▼", "negative")
    abs_str = ""
    if a is not None:
        try:
            abs_str = f" ({float(a):+.2f})"
        except (ValueError, TypeError):
            pass
    return f'<span class="{cls}" data-sort="{p:.10f}">{sign} {p:+.2f}%{abs_str}</span>'


def fmt_3yr10k(pct, val_10k):
    """Format 3-year return with percentage and $10k investment value."""
    if pct is None or val_10k is None:
        return '<span class="neutral" data-sort="-999999">N/A</span>'
    sign, cls = ("▲", "positive") if pct >= 0 else ("▼", "negative")
    return f'<span class="{cls}" data-sort="{pct:.10f}">{sign} {pct:+.2f}%<br><small>${val_10k:,.0f}</small></span>'


@lru_cache(maxsize=32)  # OPTIMIZED: Cache index data
def get_index_data(symbol):
    try:
        # Use fetch() to get full signal/trend info for index/commodity, as in ticker table
        data = fetch(symbol, ext=False)
        if not data:
            # fallback to minimal info if fetch fails
            t = yf.Ticker(symbol)
            info = t.info
            if info is None:
                info = {}
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            ch_pct = info.get("regularMarketChangePercent")
            prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
            if price is None or prev is None:
                hist = safe_history(t, period="5d")
                if len(hist) >= 2:
                    price = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
            ch_abs = None
            if price is not None and prev is not None:
                try:
                    price = float(price)
                    prev = float(prev)
                    ch_abs = price - prev
                except (ValueError, TypeError):
                    ch_abs = None
            if ch_pct is None and price is not None and prev is not None and prev > 0:
                try:
                    ch_pct = ((float(price) - float(prev)) / float(prev)) * 100
                except (ValueError, TypeError):
                    ch_pct = None
            return {"price": price, "change_pct": ch_pct, "change_abs": ch_abs}
        # Return all relevant fields for ticker-style trending arrow logic
        return {
            "price": data.get("price"),
            "change_pct": data.get("change_pct"),
            "change_abs": data.get("change_abs_day"),
            "active_signal": data.get("active_signal"),
            "signal_combined": data.get("signal_combined"),
            "predicted_trend": data.get("predicted_trend"),
            "trend_label": data.get("trend_label"),
            "macd_label": data.get("macd_label"),
            "rsi": data.get("rsi"),
            "bb_signal": data.get("bb_signal"),
            "signal_bb": data.get("signal_bb"),
            "signal_rsi": data.get("signal_rsi"),
            "signal_macd": data.get("signal_macd"),
            "signal_ichimoku": data.get("signal_ichimoku"),
            "signal_bb_ichimoku": data.get("signal_bb_ichimoku"),
            "signal_confidence": data.get("signal_confidence"),
            "signal_strength": data.get("signal_strength"),
        }
    except Exception as e:
        return {"price": None, "change_pct": None, "change_abs": None}


def load_ticker_sections(csv="data/tickers.csv"):
    """Load tickers from CSV file with optional [MEME], [STAR], [M7], [EMERGINGTECH], [LEVERAGED], [ETFS] and [TICKERS] sections.
    
    If sections are not present, treats all tickers as regular tickers.
    """
    global MEME_STOCKS, STAR_STOCKS, M7_STOCKS, EMERGINGTECH_STOCKS, LEVERAGED_STOCKS, ETFS_STOCKS

    meme_list = []
    star_list = []
    m7_list = []
    emergingtech_list = []
    leveraged_list = []
    etfs_list = []
    ticker_list = []
    current_section = None
    has_sections = False
    
    try:
        with open(csv, "r", encoding="utf-8") as f:
            content = f.read()
        # Check if file has section headers
        has_sections = "[MEME]" in content or "[STAR]" in content or "[M7]" in content or "[EMERGINGTECH]" in content or "[LEVERAGED]" in content or "[ETFS]" in content or "[TICKERS]" in content
        if has_sections:
            # Parse with sections
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Check for section headers
                if line == "[MEME]":
                    current_section = "meme"
                    continue
                elif line == "[STAR]":
                    current_section = "star"
                    continue
                elif line == "[M7]":
                    current_section = "m7"
                    continue
                elif line == "[EMERGINGTECH]":
                    current_section = "emergingtech"
                    continue
                elif line == "[LEVERAGED]":
                    current_section = "leveraged"
                    continue
                elif line == "[ETFS]":
                    current_section = "etfs"
                    continue
                elif line == "[TICKERS]":
                    current_section = "tickers"
                    continue
                # Parse tickers from line
                parts = re.split(r"[,]+", line)
                parts = [p.strip().upper() for p in parts if p and p.strip()]
                # Add to appropriate list
                if current_section == "meme":
                    meme_list.extend(parts)
                elif current_section == "m7":
                    m7_list.extend(parts)
                elif current_section == "star":
                    star_list.extend(parts)
                elif current_section == "emergingtech":
                    emergingtech_list.extend(parts)
                elif current_section == "leveraged":
                    leveraged_list.extend(parts)
                elif current_section == "etfs":
                    etfs_list.extend(parts)
                elif current_section == "tickers":
                    ticker_list.extend(parts)
        else:
            # Backward compatible: treat as simple CSV without sections
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            parts = re.split(r"[\n,]+", content.strip())
            parts = [p.strip().upper() for p in parts if p and p.strip()]
            # Skip header row if present
            if parts and parts[0].lower() in ("ticker", "tickers"):
                parts = parts[1:]
            ticker_list = parts
        
        # Update global frozensets (frozensets automatically deduplicate)
        MEME_STOCKS = frozenset(meme_list)
        STAR_STOCKS = frozenset(star_list)
        M7_STOCKS = frozenset(m7_list)
        EMERGINGTECH_STOCKS = frozenset(emergingtech_list)
        LEVERAGED_STOCKS = frozenset(leveraged_list)
        ETFS_STOCKS = frozenset(etfs_list)
        
        # Return unique tickers for all lists
        unique_tickers = pd.Series(ticker_list).unique().tolist()
        unique_meme = list(set(meme_list))
        unique_star = list(set(star_list))
        unique_m7 = list(set(m7_list))
        unique_emergingtech = list(set(emergingtech_list))
        unique_leveraged = list(set(leveraged_list))
        unique_etfs = list(set(etfs_list))
        return unique_tickers, unique_meme, unique_star, unique_m7, unique_emergingtech, unique_leveraged, unique_etfs
    except Exception as e:
        M7_STOCKS = frozenset()
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"], [], [], []


def dashboard(csv="data/tickers.csv", ext=None):
    """Generate dashboard data. If ext=None, auto-detect based on current market hours."""
    os.makedirs("data", exist_ok=True)
    tickers, meme_list, star_list, m7_list, emergingtech_list, leveraged_list, etfs_list = load_ticker_sections(csv)
    # Ensure CATEGORY_MAP uses the latest sets
    CATEGORY_MAP["m7"] = M7_STOCKS
    CATEGORY_MAP["star"] = STAR_STOCKS
    CATEGORY_MAP["emerging-tech"] = EMERGINGTECH_STOCKS
    CATEGORY_MAP["leveraged-etf"] = LEVERAGED_STOCKS
    CATEGORY_MAP["sector-etf"] = ETFS_STOCKS

    # Auto-detect if not explicitly specified
    if ext is None:
        ext = not is_regular_hours()  # Extended hours when NOT in regular hours
    
    data = []
    # Adaptive worker count: Increased from 3 to 5 for better parallelization
    # Rate limiter prevents overwhelming the API
    worker_count = min(MAX_WORKERS, max(MIN_WORKERS, len(tickers) // TICKER_BATCH_DIVISOR + 1))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(fetch, t, ext) for t in tickers]
        for r in as_completed(futures):
            res = r.result()
            if res:
                data.append(res)
    
    df = pd.DataFrame(data)
    if not df.empty and 'change_pct' in df.columns:
        return df.sort_values("change_pct", ascending=False)
    return df


def get_vix_data():
    return get_index_data("^VIX")


# OPTIMIZED: Cache F&G data with 30 minute TTL
_fg_cache = {"data": None, "time": 0}

# Shared requests session for fewer TCP handshakes
SESSION = requests.Session()


# Cache ticker info to avoid repeated yf.Ticker(...).info calls
@lru_cache(maxsize=TICKER_INFO_CACHE_SIZE)
def get_ticker_info_cached(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


# Simple global rate limiter (ensure at least `min_interval` seconds between network starts)
class RateLimiter:
    def __init__(self, min_interval=RATE_LIMIT_INTERVAL):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_time:
                wait_for = self.next_time - now
                time.sleep(wait_for)
                now = time.time()
            self.next_time = now + self.min_interval


RATE_LIMITER = RateLimiter(min_interval=RATE_LIMIT_INTERVAL)


def get_fear_greed_data():
    global _fg_cache
    now = time.time()
    if _fg_cache["data"] is not None and (now - _fg_cache["time"]) < FG_CACHE_TTL_SECONDS:
        return _fg_cache["data"]

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        }
        r = requests.get(url, headers=headers, timeout=5)  # OPTIMIZED: Reduced timeout
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            data = {}
        fg = data.get("fear_and_greed") or data
        score = float(fg["score"])
        s = int(round(score))
        rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
            0 if s <= 24 else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
        ]
        result = {"score": score, "rating": rating, "raw_score": s}
        _fg_cache["data"] = result
        _fg_cache["time"] = now
        return result
    except Exception:
        try:
            r = requests.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                data = {}
            score = float(data["fear_and_greed"]["score"])
            s = int(round(score))
            rating = ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")[
                (
                    0
                    if s <= 24
                    else 1 if s <= 44 else 2 if s <= 55 else 3 if s <= 74 else 4
                )
            ]
            result = {"score": score, "rating": rating, "raw_score": s}
            _fg_cache["data"] = result
            _fg_cache["time"] = now
            return result
        except Exception as e:
            pass
    return {"score": None, "rating": "N/A", "raw_score": None}


# OPTIMIZED: Cache AAII data with 30 minute TTL
_aaii_cache = {"data": None, "time": 0}
AAII_CACHE_TTL = 1800


def get_aaii_sentiment():
    global _aaii_cache
    now = time.time()
    if _aaii_cache["data"] is not None and (now - _aaii_cache["time"]) < AAII_CACHE_TTL:
        return _aaii_cache["data"]

    try:
        r = requests.get(
            "https://www.aaii.com/sentimentsurvey/sent_results",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        m = re.search(r"\w+\s*\d{1,2}.*?([\d\.]+)%.*?([\d\.]+)%", r.text)
        if not m:
            m = re.search(
                r"Bullish.*?([\d\.]+)%.*?Bearish.*?([\d\.]+)%", r.text, re.DOTALL
            )
        if m:
            b, be = float(m.group(1)), float(m.group(2))
            result = {"bullish": b, "bearish": be, "spread": b - be}
            _aaii_cache["data"] = result
            _aaii_cache["time"] = now
            return result
    except Exception as e:
        pass
    return {"bullish": None, "bearish": None, "spread": None}


# NOTE: html() function continues with the full HTML generation...
# Due to character limits, the complete html() function and main block remain unchanged
# from original code. Simply append the original html() function and main block here.


def html(df, vix, fg, aaii, file, ext=False, alerts=None):
    alerts = alerts or {"grouped": [], "time": ""}
    update = datetime.now(UTC).astimezone(PST).strftime("%I:%M:%S %p PST on %B %d, %Y")
    
    # Handle empty dataframe
    if df.empty:
        # Create a minimal HTML file
        f.write(f"<html><body><h1>No Data Available</h1><p>Last updated: {update}</p></body></html>")
        return

    # Always render the banner, even if group links are empty
    banner = (
        '<div class="alert-banner" id="alertBanner">'
        '<span class="alert-content">🚨 <strong>ALERTS</strong> '
        + (alerts["grouped"][0] if alerts["grouped"] else "")
        + '</span>'
        '<button class="alert-dismiss" onclick="dismissAlerts()" title="Dismiss alerts">✕</button>'
        "</div>"
    )

    # Major indices
    dow = get_index_data("^DJI")
    sp = get_index_data("^GSPC")
    spy = get_index_data("SPY")
    nas = get_index_data("^IXIC")
    vix = get_index_data("^VIX")
    
    # Commodities and Crypto
    gold = get_index_data("GC=F")
    silver = get_index_data("SI=F")
    copper = get_index_data("HG=F")
    bitcoin = get_index_data("BTC-USD")

    # Calculate CVR3 Signal
    cvr3_signal = "NEUTRAL"
    try:
        vix_data = safe_history(yf.Ticker("^VIX"), period="60d")
        if len(vix_data) >= 10:
            vix_close = vix_data["Close"]
            vix_sma = vix_close.rolling(window=10).mean().iloc[-1]
            if vix_sma > 0:
                current_vix = vix_close.iloc[-1]
                pct_diff = ((current_vix - vix_sma) / vix_sma) * 100
                if pct_diff >= 20:
                    cvr3_signal = "SHORT"
                elif pct_diff >= 10:
                    cvr3_signal = "BUY"
                elif pct_diff <= -10:
                    cvr3_signal = "SELL"
    except Exception:
        cvr3_signal = "NEUTRAL"

    def signal_and_trend_icons(data):
        # Prefix: signal icon (BUY/SELL/SHORT), Suffix: trend arrow (↑, ↗, →, ↘, ↓)
        signal = data.get("active_signal") or data.get("signal_combined")
        trend = data.get("predicted_trend")
        # Signal icon and color
        sig_icon = ""
        sig_color = "neutral"
        if signal == "BUY":
            sig_icon = "🟢"
            sig_color = "positive"
        elif signal == "SELL":
            sig_icon = "🔴"
            sig_color = "negative"
        elif signal == "SHORT":
            sig_icon = "🟣"
            sig_color = "negative"
        # Trend arrow and color
        trend_arrow = trend if trend in ["↑", "↗", "→", "↘", "↓"] else "→"
        if trend_arrow in ["↑", "↗", "→"]:
            trend_color = "positive"  # Up and sideways are green
        elif trend_arrow in ["↓", "↘"]:
            trend_color = "negative"  # Down is red
        else:
            trend_color = "positive"
        # Compose HTML
        prefix = f'<span class="{sig_color}" style="font-size:1.1em">{sig_icon}</span> ' if sig_icon else ""
        suffix = f' <span class="{trend_color}" style="font-size:1.1em">{trend_arrow}</span>'
        return prefix, suffix, sig_color if sig_icon else trend_color

    def index_str(data, name, invert=False):
        if data is None or data.get("price") is None:
            return f'<span class="neutral">{name}: N/A</span>'
        ch_abs = data.get("change_abs")
        prefix, suffix, color = signal_and_trend_icons(data)
        if invert:
            cls = "negative" if ch_abs is not None and ch_abs >= 0 else "positive"
        else:
            cls = "positive" if ch_abs is not None and ch_abs >= 0 else "negative"
        # Prefix signal icon, name, suffix trend arrow, then price
        return f'{prefix}<span class="{color}">{name}</span>{suffix}: <span class="{cls}">{na(data.get("price"))} ({na(ch_abs, "{:+.2f}")})</span>'

    # CVR3 signal color
    cvr3_color = (
        "positive"
        if cvr3_signal == "BUY"
        else ("negative" if cvr3_signal in ("SELL", "SHORT") else "neutral")
    )
    cvr3_str = f'<span class="{cvr3_color}">CVR3: {cvr3_signal}</span>'
    
    # Commodities line with arrows
    commodities_h = f'{index_str(gold, "Gold")} | {index_str(silver, "Silver")} | {index_str(copper, "Copper")} | {index_str(bitcoin, "Bitcoin")}'

    # Build single-line market summary with all global indexes and arrows, SPY before Nasdaq
    indices_h = f'{index_str(dow, "Dow")} | {index_str(sp, "S&P")} | {index_str(spy, "SPY")} | {index_str(nas, "Nasdaq")} | {index_str(vix, "VIX", invert=True)}'

    fg_h = '<span class="neutral">F&G: N/A</span>'
    if fg and fg.get("score") is not None:
        cls = (
            "negative"
            if fg["score"] <= 24
            else (
                "high-risk"
                if fg["score"] <= 44
                else (
                    "neutral"
                    if fg["score"] <= 55
                    else "bullish" if fg["score"] <= 74 else "positive"
                )
            )
        )
        fg_h = f'<span class="{cls}">F&G: {fg["score"]:.1f} ({fg["rating"]})</span>'

    aaii_h = '<span class="neutral">AAII: N/A</span>'
    if aaii and aaii.get("bullish") is not None:
        spread = aaii["spread"]
        cls = (
            "positive"
            if spread > 20
            else (
                "bullish"
                if spread > 0
                else (
                    "neutral"
                    if spread > -20
                    else "high-risk" if spread > -40 else "negative"
                )
            )
        )
        aaii_h = f'<span class="{cls}">AAII: Bull {aaii["bullish"]:.1f}% Bear {aaii["bearish"]:.1f}%</span>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
@import url('https://fonts.googleapis.com/css2?family=Oracle+Sans:wght@400;500;600;700&display=swap');
:root{{ --bg:#f5f7fa; --card:#ffffff; --text:#312d2a; --border:#d6dbe0; --accent:#0572ce; --accent-hover:#0460b2; --accent-dark:#024a87; --pos:#20813e; --neg:#c74634; --bullish:#20813e; --bearish:#c74634; --surface:#fafbfc; --shadow:0 1px 3px 0 rgba(0,0,0,0.08), 0 1px 2px 0 rgba(0,0,0,0.06); --shadow-hover:0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); --shadow-lg:0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); --radius-sm:4px; --radius-md:6px; --radius-lg:8px }}
[data-theme="dark"]{{ --bg:#1b1f24; --card:#272c33; --text:#e8ecef; --border:#3d4349; --accent:#1f8ffa; --accent-hover:#4da3fb; --accent-dark:#0572ce; --pos:#3eb878; --neg:#e0604f; --bullish:#3eb878; --bearish:#e0604f; --surface:#21262b; --shadow:0 1px 3px 0 rgba(0,0,0,0.2), 0 1px 2px 0 rgba(0,0,0,0.12) }}
*{{box-sizing:border-box}}
body{{font-family:'Oracle Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);padding:10px;margin:0;transition:background 0.2s,color 0.2s;font-size:14px;line-height:1.5}}
.container{{max-width:1600px;margin:auto}}
.top-bar{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:16px;background:var(--card);padding:10px 16px;border-radius:var(--radius-lg);box-shadow:var(--shadow);border:1px solid var(--border);margin-bottom:0}}
.btn{{padding:6px 14px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-md);cursor:pointer;font-weight:600;font-size:13px;transition:all 0.2s;box-shadow:var(--shadow)}}
.btn:hover{{background:var(--accent-hover);box-shadow:var(--shadow-hover);transform:translateY(-1px)}}
.btn:active{{transform:translateY(0);box-shadow:var(--shadow)}}
.alert-banner{{background:var(--card);color:var(--text);padding:6px 60px 6px 14px;border-radius:var(--radius-lg);margin-bottom:6px;text-align:center;font-weight:600;box-shadow:var(--shadow);border:3px solid #c74634;position:relative;display:flex;align-items:center;justify-content:center}}
.alert-content{{flex:1;text-align:center}}
.alert-dismiss{{position:absolute;right:12px;top:50%;transform:translateY(-50%);background:var(--neg);border:2px solid var(--card);color:#fff;font-size:20px;font-weight:bold;cursor:pointer;opacity:0.95;transition:all 0.2s;padding:2px 8px;line-height:1;border-radius:var(--radius-md);box-shadow:var(--shadow)}}
.alert-dismiss:hover{{opacity:1;background:#8b2d23;transform:translateY(-50%) scale(1.15);box-shadow:var(--shadow-hover)}}
.controls-container{{background:var(--card);padding:8px;border-radius:var(--radius-lg);margin-bottom:6px;box-shadow:var(--shadow);border:3px solid #a8b2bd}}
.hours-toggle-container{{display:flex;gap:15px;flex-wrap:wrap;align-items:center;justify-content:space-between;padding-bottom:6px;border-bottom:1px solid var(--border);margin-bottom:6px}}
.quick-links-section{{background:var(--card);border:2px solid var(--border);border-radius:var(--radius-lg);margin-bottom:6px;box-shadow:var(--shadow)}}
.quick-links-section summary{{padding:8px 12px;cursor:pointer;font-weight:600;font-size:14px;user-select:none;display:flex;align-items:center;gap:8px;color:var(--text);transition:background 0.2s}}
.quick-links-section summary:hover{{background:var(--surface)}}
.quick-links-section summary::marker{{font-size:12px}}
.quick-links-content{{padding:0 12px 10px 12px;line-height:1.5}}
.quick-links-content a{{color:var(--accent);text-decoration:none;font-weight:500;transition:all 0.2s}}
.quick-links-content a:hover{{color:var(--accent-hover);text-decoration:underline}}
.quick-links-content ul{{margin:0;padding-left:20px}}
.quick-links-content li{{margin:4px 0}}
.quick-filters{{display:flex;flex-wrap:wrap;gap:6px;padding-bottom:6px;border-bottom:1px solid var(--border);margin-bottom:6px}}
.chip{{padding:4px 10px;background:var(--surface);border:1.5px solid var(--border);border-radius:20px;cursor:pointer;font-weight:600;font-size:12px;transition:all 0.2s;color:var(--text)}}
.chip.active{{background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:var(--shadow)}}
.chip:hover{{background:var(--accent-hover);border-color:var(--accent-hover);color:#fff;transform:translateY(-1px)}}
.views{{display:flex;gap:8px}}
.view-btn{{padding:6px 14px;background:var(--card);border:1.5px solid var(--border);border-radius:var(--radius-md);cursor:pointer;transition:all 0.2s;font-weight:600;font-size:13px;color:var(--text)}}
.view-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent);box-shadow:var(--shadow)}}
.view-btn:hover{{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}}
.view-btn.active:hover{{color:#fff}}
#tableView{{display:block}}
#cardView,#heatView{{display:none}}
#cardView{{width:100%;max-width:100vw;overflow:hidden;box-sizing:border-box}}
#heatView{{width:100%;max-width:100vw;overflow:hidden;box-sizing:border-box}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;padding:6px;max-width:100%;box-sizing:border-box}}
.stock-card{{background:var(--card);border-radius:var(--radius-lg);padding:28px 16px 16px 16px;box-shadow:var(--shadow);transition:all 0.2s;border:1px solid var(--border);position:relative;min-height:320px;height:auto;overflow:hidden;display:flex;flex-direction:column;box-sizing:border-box}}
.stock-card:hover{{transform:translateY(-2px);box-shadow:var(--shadow-lg);border-color:var(--accent)}}
.card-content-scroll{{display:flex;overflow-x:scroll;overflow-y:hidden;scrollbar-width:none;-ms-overflow-style:none;scroll-snap-type:x mandatory;scroll-behavior:smooth;flex:1;width:100%;min-height:0}}
.card-content-scroll::-webkit-scrollbar{{display:none}}
.card-page{{flex:0 0 100%;width:100%;min-width:100%;max-width:100%;scroll-snap-align:start;scroll-snap-stop:always;box-sizing:border-box;overflow-y:auto;padding-right:5px;min-height:0;line-height:1.4;font-size:0.95em}}
.card-page > div{{margin:4px 0}}
.card-page h2{{margin:0 0 8px 0;font-size:1.3em;line-height:1.2}}
.card-page::-webkit-scrollbar{{width:6px}}
.card-page::-webkit-scrollbar-track{{background:var(--surface);border-radius:3px}}
.card-page::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
.card-page::-webkit-scrollbar-thumb:hover{{background:var(--accent)}}
.card-scroll-btn{{position:absolute;top:8px;background:var(--accent);color:#fff;border:none;border-radius:50%;width:16px;height:16px;font-size:10px;cursor:pointer;z-index:10;box-shadow:var(--shadow);transition:all 0.2s;display:flex;align-items:center;justify-content:center;opacity:0.85;padding:0;line-height:1}}
@media (min-width: 768px) {{
  .card-grid{{grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
  .stock-card{{padding:30px 18px 18px 18px;min-height:340px}}
  .card-scroll-btn{{width:18px;height:18px;font-size:10px}}
}}
@media (min-width: 1200px) {{
  .card-grid{{grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}}
  .stock-card{{padding:32px 20px 20px 20px;min-height:360px}}
  .card-scroll-btn{{width:20px;height:20px;font-size:11px}}
}}
.card-scroll-btn:hover{{opacity:1;transform:scale(1.15)}}
.card-scroll-btn:disabled{{opacity:0.3;cursor:not-allowed;pointer-events:none}}
.card-scroll-left{{left:8px}}
.card-scroll-right{{left:36px}}
.heat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;padding:8px;max-width:100%;box-sizing:border-box}}
.heat-tile{{min-height:140px;height:auto;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:var(--radius-md);padding:8px;cursor:pointer;transition:all 0.2s;border:1px solid transparent;overflow:hidden;box-sizing:border-box}}
@media (min-width: 768px) {{
  .heat-grid{{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
  .heat-tile{{min-height:150px}}
}}
@media (min-width: 1200px) {{
  .heat-grid{{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}}
  .heat-tile{{min-height:160px}}
}}
.heat-tile:hover{{transform:scale(1.03);box-shadow:var(--shadow-lg);border-color:var(--accent)}}
table{{width:100%;border-collapse:separate;border-spacing:0;background:var(--card);box-shadow:var(--shadow);border-radius:var(--radius-lg);overflow:hidden;border:1px solid var(--border)}}
th{{background:linear-gradient(180deg,var(--accent),var(--accent-dark));color:#fff;padding:16px;cursor:pointer;position:sticky;top:0;z-index:10;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid var(--accent-dark)}}
th:hover{{background:linear-gradient(180deg,var(--accent-hover),var(--accent))}}
td{{padding:14px 16px;border-bottom:1px solid var(--border);vertical-align:top;font-size:13px}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--surface)}}
.positive{{color:var(--pos);font-weight:600}}
.negative{{color:var(--neg);font-weight:600}}
.neutral{{color:#697883}}
.bullish{{color:var(--bullish);font-weight:600}}
.bearish{{color:var(--bearish);font-weight:600}}
.extreme-fear{{color:var(--neg);font-weight:700}}
.fear{{color:#d97706}}
.greed{{color:#65a30d}}
.extreme-greed{{color:var(--pos);font-weight:700}}
.strong-bull{{color:#166534;font-weight:700}}
.bull{{color:var(--pos)}}
.strong-bear{{color:var(--neg);font-weight:700}}
.bear{{color:var(--neg);font-weight:600}}
.vol-hot{{color:var(--neg);font-weight:700}}
.range-bar{{width:100%;height:10px;background:var(--surface);border-radius:var(--radius-sm);position:relative;margin:6px 0;border:1px solid var(--border)}}
.range-bar-marker{{position:absolute;width:3px;height:14px;background:var(--accent);top:-2px;border-radius:2px}}
.range-labels{{display:flex;justify-content:space-between;font-size:0.75em;margin-top:4px;color:var(--text);opacity:0.8}}
.range-container{{margin:10px 0}}
.range-title{{font-size:0.75em;font-weight:600;margin-bottom:4px;color:var(--text);opacity:0.9}}
input#tickerFilter{{padding:6px 10px;border:1.5px solid var(--border);border-radius:var(--radius-md);background:var(--card);color:var(--text);font-size:13px;font-family:'Oracle Sans',-apple-system,sans-serif;transition:all 0.2s;outline:none;margin-left:auto;min-width:120px;width:100%;max-width:260px;box-sizing:border-box}}
/* Responsive table view: horizontal scroll on small screens */
.table-wrap{{width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;}}
table.data-table{{min-width:900px;}}
@media (max-width: 900px) {{
    .table-wrap{{padding-bottom:8px;}}
    table.data-table{{min-width:700px;}}
}}
@media (max-width: 600px) {{
    .table-wrap{{padding-bottom:8px;}}
    table.data-table{{min-width:520px;}}
    input#tickerFilter {{
        min-width: 0;
        width: 100%;
        margin-left: 0;
        max-width: 100%;
    }}
}}
input#tickerFilter:hover{{border-color:var(--accent)}}
input#tickerFilter:focus{{border-color:var(--accent);box-shadow:0 0 0 3px rgba(5,114,206,0.1)}}
</style></head><body>
<div class="container">
{banner}
<div class="top-bar" style="margin-bottom:0">
<div style="display:flex;align-items:baseline;gap:12px"><h1 style="margin:0">📊 Dashboard</h1><small>{update}</small></div>
<details class="quick-links-section" style="margin-left:auto">
<summary>🔗 Quick Links</summary>
<div class="quick-links-content">
<ul>
<li><a href="https://tradingeconomics.com/us100:ind" target="_blank">US 100</a> | <a href="https://tradingeconomics.com/calendar" target="_blank">Calendar</a> | <a href="https://www.ssga.com/us/en/intermediary/resources/sector-tracker#currentTab=dayFive" target="_blank">Sectors</a> | <a href="https://tradingeconomics.com/stream" target="_blank">News</a> | <a href="https://finviz.com/news.ashx" target="_blank">Finviz</a> | <a href="https://www.optionsprofitcalculator.com/calculator/call-spread.html" target="_blank">Options</a></li>
<li><a href="https://www.slickcharts.com/market-movers" target="_blank">Market Movers</a> | <a href="https://stockanalysis.com/markets/gainers/" target="_blank">SA: Movers</a> | <a href="https://stockanalysis.com/trending" target="_blank">SA: Trending</a> | <a href="https://stockanalysis.com/markets/heatmap/?time=1W" target="_blank">Heat Map</a> | <a href="https://www.morningstar.com/markets" target="_blank">MS: Markets</a></li>
<li><a href="https://www.trackinsight.com/en" target="_blank">Flows</a> | <a href="https://www.google.com/search?q=https://www.morningstar.com/topics/fund-flows" target="_blank">MS: Flows</a> | <a href="https://www.ssga.com/us/en/intermediary/insights/a-feast-of-etf-inflows-and-returns" target="_blank">SPDR: Flows</a> | <a href="https://www.etf.com/sections/daily-etf-flows" target="_blank">ETF.com</a> | <a href="https://etfdb.com/etf-fund-flows/#issuer=blackrock-inc" target="_blank">ETFdb</a> | <a href="https://krmallina.github.io/smi/data/summary.html" target="_blank">Summary</a></li>
<li><a href="https://www.cnn.com/markets/fear-and-greed" target="_blank">Fear & Greed Index</a> | <a href="https://www.aaii.com/sentiment-survey" target="_blank">AAII Sentiment</a> | <a href="https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models" target="_blank">Trading Strategies</a> | <a href="https://krmallina.github.io/smi/" target="_blank">Readme</a></li>
</ul>
</div>
</details>
</div>
<div style="padding:4px 15px;background:var(--card);border-bottom:1px solid var(--border);font-size:0.9em;text-align:center;margin-top:0;margin-bottom:6px">
<span>{commodities_h}</span> | <span>{cvr3_str}</span> | <span>{fg_h}</span> | <span>{aaii_h}</span>
</div>
<div style="padding:4px 15px;background:var(--card);border-bottom:1px solid var(--border);font-size:0.9em;text-align:center;margin-top:0;margin-bottom:6px">
<span id="marketIndices">{indices_h}</span>
</div>

<div class="controls-container">
<div class="hours-toggle-container" style="display:flex;gap:15px;flex-wrap:wrap;align-items:center;justify-content:space-between">
<div style="display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px;color:var(--text)">
{'<span style="background:var(--accent);color:#fff;padding:4px 12px;border-radius:var(--radius-md);box-shadow:var(--shadow)">Extended Hours</span>' if ext else ''}
</div>
<div style="display:flex;gap:10px">
<button class="btn" onclick="toggleTheme()">🌓</button>
<button class="btn" onclick="refreshPage()">🔄</button>
</div>
</div>

<div class="quick-filters">
<div class="chip active" data-filter="all">All</div>
<div class="chip" data-filter="volume">📊 High Vol</div>
<div class="chip" data-filter="m7">💎 M7+</div>
<div class="chip" data-filter="star">⭐ Starred</div>
<div class="chip" data-filter="earnings-week">📅 Earnings</div>
<div class="chip" data-filter="cat-major-tech">🌐 Growth</div>
<div class="chip" data-filter="cat-leveraged-etf">⚡ Leveraged</div>
<div class="chip" data-filter="cat-sector-etf">🏦 ETFs</div>
<div class="chip" data-filter="signal-buy">🟢 Buy</div>
<div class="chip" data-filter="signal-sell">🟠 Sell</div>
<div class="chip" data-filter="signal-short">🔴 Short</div>
<div class="chip" data-filter="signal-hold">⏸️ Hold</div>
<div class="chip" data-filter="oversold">📉 Oversold</div>
<div class="chip" data-filter="overbought">📈 Overbought</div>
<div class="chip" data-filter="surge">🚀 Surge</div>
<div class="chip" data-filter="crash">💥 Crash</div>
<div class="chip" data-filter="cat-emerging-tech">🚧 Emerging</div>
<div class="chip" data-filter="dividend">💰 Dividend</div>
<div class="chip" data-filter="cat-spec-meme">🎲 Speculative</div>
<div class="chip" data-filter="squeeze">🔥 Squeeze</div>
<div class="chip" data-filter="bb-squeeze">📏 BB Squeeze</div>
</div>

<div class="views" style="margin-top:6px">
<button class="view-btn" onclick="setView(this,'heat')">🔥 Heatmap</button>
<button class="view-btn" onclick="setView(this,'card')">🗂️ Cards</button>
<button class="view-btn" onclick="setView(this,'table')">📋 Table</button>
<input id="tickerFilter" placeholder="Filter tickers..." oninput="applyFilter()">
</div>
</div>

<div id="tableView">
<div class="table-wrap">
<table id="stockTable">
<tr>
<th data-sort="ticker">TICKER</th>
<th data-sort="price">PRICE</th>
<th data-sort="volume_raw">VOLUME</th>
<th data-sort="change_pct">DAY %</th>
<th data-sort="change_5d">5D %</th>
<th data-sort="change_1m">1M %</th>
<th data-sort="change_3m">3M %</th>
<th data-sort="change_6m">6M %</th>
<th data-sort="change_ytd">YTD %</th>
<th data-sort="change_1y">1Y %</th>
<th data-sort="change_3y">3YR10K</th>
<th style="width:160px;min-width:140px;max-width:200px;">RANGES</th>
<th style="width:160px;min-width:140px;max-width:200px;">INDICATORS</th>
</tr>
"""
    for _, r in df.iterrows():
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.75em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.75em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.75em">🟠</span> '
        elif active_sig == "HOLD":
            signal_icon = '<span style="font-size:0.755em">⏸️</span> '
        
        # Get color-coded trend arrow
        trend_arrow = ""
        trend_label = r.get('trend_label', 'NEUTRAL')
        predicted_trend = r.get('predicted_trend', '→')
        if trend_label == 'UP':
            trend_arrow = f'<span style="color:var(--pos);font-weight:bold"> {predicted_trend}</span>'
        elif trend_label == 'DOWN':
            trend_arrow = f'<span style="color:var(--neg);font-weight:bold"> {predicted_trend}</span>'
        else:
            trend_arrow = f'<span style="color:var(--neutral);font-weight:bold"> {predicted_trend}</span>'
        
        bb_icon = signal_icon  # Use active signal icon
        hv = r["hv_30_annualized"]
        hv_cls = "negative" if hv and hv > 50 else "neutral"
        hv_str = na(hv, "{:.1f}%")

        macd_cls = (
            "bullish"
            if r["macd_label"] == "Bullish"
            else "bearish" if r["macd_label"] == "Bearish" else "neutral"
        )
        opt_dir_cls = (
            "bullish"
            if "Bullish" in r["options_direction"]
            else "bearish" if "Bearish" in r["options_direction"] else "neutral"
        )
        bias_cls = "bearish" if r["down_volume_bias"] else "bullish"

        sent_cls = (
            "bullish"
            if "Buy" in r["sentiment"]
            else "bearish" if "Sell" in r["sentiment"] else "neutral"
        )
        # include analyst rating in sentiment display and adjust class if needed
        analyst_rating = r.get("analyst_rating") or ""
        # if analyst rating suggests Buy/Sell, reflect that in the class (fallback to sentiment)
        if analyst_rating:
            if "Buy" in analyst_rating:
                sent_cls = "bullish"
            elif "Sell" in analyst_rating:
                sent_cls = "bearish"
        upside_cls = (
            "bullish"
            if r["upside_potential"] and r["upside_potential"] > 0
            else (
                "bearish"
                if r["upside_potential"] and r["upside_potential"] < 0
                else "neutral"
            )
        )

        bb_bar = ""
        # Only render Bollinger Bands bar when BB values are present and numeric
        if (
            pd.notna(r.get("bb_position_pct"))
            and pd.notna(r.get("bb_lower"))
            and pd.notna(r.get("bb_middle"))
            and pd.notna(r.get("bb_upper"))
        ):
            try:
                pos = float(r["bb_position_pct"])
                pos = max(0.0, min(100.0, pos))
                bb_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos}%, var(--neg) {pos}%, var(--neg) 100%)"
                bb_bar = f'<div class="range-container"><div class="range-title">Bollinger Bands</div><div class="range-bar" style="background:{bb_color}"><div class="range-bar-marker" style="left:{pos}%"></div></div><div class="range-labels"><span>${na(r["bb_lower"])}</span><span>${na(r["bb_middle"])}</span><span>${na(r["bb_upper"])}</span></div><div style="font-size:0.75em;text-align:center">Width: {na(r["bb_width_pct"],"{:.1f}")}% – {r["bb_status"]}</div></div>'
            except Exception:
                bb_bar = ""

        impl_bar = ""
        # Only render implied-move chart when values are present and numeric (avoid NaN%)
        if (
            pd.notna(r.get("implied_move_pct"))
            and pd.notna(r.get("implied_low"))
            and pd.notna(r.get("implied_high"))
        ):
            try:
                im_pct = float(r["implied_move_pct"])
                if im_pct > 0:
                    left_pct = 50 - im_pct / 2
                    right_pct = 50 + im_pct / 2
                    i_color = f"linear-gradient(to right, var(--neg) 0%, var(--neg) {left_pct}%, var(--pos) {right_pct}%, var(--pos) 100%)"
                    impl_bar = f'<div class="range-container"><div class="range-title">Implied Move ±{im_pct:.1f}%</div><div class="range-bar" style="background:{i_color}"><div class="range-bar-marker" style="left:50%"></div></div><div class="range-labels"><span>${na(r["implied_low"])}</span><span>${na(r["implied_high"])}</span></div></div>'
            except Exception:
                impl_bar = ""

        # Only render range charts when values are present and valid (avoid NaN% rendering)
        day_block = ""
        if (
            pd.notna(r.get("day_low"))
            and pd.notna(r.get("day_high"))
            and r["day_high"] is not None
            and r["day_low"] is not None
            and (r["day_high"] - r["day_low"]) > 0
        ):
            day_pos = (r["price"] - r["day_low"]) / (r["day_high"] - r["day_low"]) * 100
            day_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {day_pos}%, var(--neg) {day_pos}%, var(--neg) 100%)"
            day_block = f"""
    <div class="range-container"><div class="range-title">Day</div><div class="range-bar" style="background:{day_color}"><div class="range-bar-marker" style="left:{day_pos}%"></div></div><div class="range-labels"><span>${r['day_low']:.2f}</span><span>${r['day_high']:.2f}</span></div></div>
    """

        y52_block = ""
        if (
            pd.notna(r.get("52w_low"))
            and pd.notna(r.get("52w_high"))
            and r["52w_high"] is not None
            and r["52w_low"] is not None
            and (r["52w_high"] - r["52w_low"]) > 0
        ):
            y52_pos = (r["price"] - r["52w_low"]) / (r["52w_high"] - r["52w_low"]) * 100
            y52_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {y52_pos}%, var(--neg) {y52_pos}%, var(--neg) 100%)"
            y52_block = f"""
    <div class="range-container"><div class="range-title">52W</div><div class="range-bar" style="background:{y52_color}"><div class="range-bar-marker" style="left:{y52_pos}%"></div></div><div class="range-labels"><span>${r['52w_low']:.2f}</span><span>${r['52w_high']:.2f}</span></div></div>
    """

        ranges_html = f"{day_block}{y52_block}{bb_bar}{impl_bar}"

        # CVR3 Signal color coding
        cvr3_color = "var(--neutral)"
        if r.get("cvr3_vix_signal") == "BUY":
            cvr3_color = "var(--pos)"
        elif r.get("cvr3_vix_signal") == "SELL":
            cvr3_color = "var(--neg)"
        elif r.get("cvr3_vix_signal") == "SHORT":
            cvr3_color = "#ff8800"

        # VIX change color coding
        vix_change_color = "var(--neutral)"
        if r.get("cvr3_vix_pct") is not None:
            if r.get("cvr3_vix_pct") > 0:
                vix_change_color = "var(--neg)"
            elif r.get("cvr3_vix_pct") < 0:
                vix_change_color = "var(--pos)"

        cvr3_html = ""
        # CVR3/VIX removed from table view - retained at page level only
        
        # Risk management and confidence metrics
        confidence_val = r.get('signal_confidence')
        strength_val = r.get('signal_strength')
        atr_val = r.get('atr_14')
        stop_loss_val = r.get('stop_loss_price')
        rr_ratio = r.get('risk_reward_ratio')
        pos_size = r.get('position_size_pct')
        
        # Confidence color coding
        if confidence_val is not None and strength_val is not None and confidence_val == confidence_val:
            if confidence_val >= 0.6:
                conf_cls = "positive"
            elif confidence_val >= 0.3:
                conf_cls = "neutral"
            else:
                conf_cls = "negative"
            conf_str = f'<span class="{conf_cls}">{strength_val} ({confidence_val:.0%})</span>'
        else:
            conf_str = '<span class="neutral">HOLD</span>'
        
        # Risk management display (only when values available)
        risk_parts = []
        if atr_val is not None:
            risk_parts.append(f'ATR: ${atr_val:.2f}')
        if stop_loss_val is not None:
            risk_parts.append(f'SL: ${stop_loss_val:.2f}')
        if rr_ratio is not None:
            risk_parts.append(f'R:R: {rr_ratio:.1f}:1')
        if pos_size is not None:
            risk_parts.append(f'Pos: {pos_size:.1f}%')
        
        risk_str = ' | '.join(risk_parts) if risk_parts else ''
        
        # Build indicators HTML with death/golden cross signals
        ma_cross_html = ""
        if r.get('golden_cross'):
            ma_cross_html = '<span class="bullish">🟢 Golden Cross</span><br>'
        elif r.get('death_cross'):
            ma_cross_html = '<span class="bearish">🔴 Death Cross</span><br>'
        
        indicators_html = f"""{ma_cross_html}<span class="{macd_cls}">MACD: {r['macd_label']}</span><br>
Short: {na(r['short_percent'],"{:.1f}%")} ({na(r['days_to_cover'],"{:.1f}d")})<br>
<span class="{hv_cls}">Volatility: {hv_str}</span><br>
<span class="{opt_dir_cls}">Opt Dir: {r['options_direction']}</span><br>
<span class="{bias_cls}">Bias: {'Down' if r['down_volume_bias'] else 'Up'}</span>"""
        
        # Add ML breakout score and crash risk
        ml_score = r.get('ml_breakout_score', 0)
        ml_crash = r.get('ml_crash_risk', 0)
        ml_win_rate = r.get('ml_historical_win_rate', 0.0)
        ml_expected_return = r.get('ml_expected_return', 0.0)
        ml_sample_size = r.get('ml_sample_size', 0)
        
        if ml_score > 0:
            if ml_score >= 70:
                ml_html = f'<span style="color:var(--pos);font-weight:bold;font-size:1.1em">🚀 {ml_score}%</span>'
            elif ml_score >= 50:
                ml_html = f'<span style="color:#ff8800;font-weight:600">🚀 {ml_score}%</span>'
            else:
                ml_html = f'<span class="ml-low" style="font-size:0.9em">🚀 {ml_score}%</span>'
        else:
            ml_html = '<span class="ml-na">🚀 N/A</span>'
        if ml_crash >= 50:
            ml_html += f'<br><span style="color:var(--neg);font-size:0.85em">⚠️ {ml_crash}%</span>'
        
        # Add historical performance if available
        if ml_win_rate > 0 and ml_sample_size > 0:
            perf_color = "var(--pos)" if ml_win_rate >= 70 else "var(--neg)" if ml_win_rate < 50 else "#ff8800"
            ml_html += f'<br><span style="color:{perf_color};font-size:0.8em">📊 {ml_win_rate:.0f}% win rate ({ml_sample_size} trades)</span>'
            if abs(ml_expected_return) > 0.1:  # Only show if meaningful
                ret_color = "var(--pos)" if ml_expected_return > 0 else "var(--neg)"
                ml_html += f'<br><span style="color:{ret_color};font-size:0.8em">💰 Expected: {ml_expected_return:+.1f}%</span>'
        
        # Add options strategy suggestions for table view based on technical indicators
        trend_score = r.get('trend_score', 0)
        death_cross = r.get('death_cross', False)
        change_pct = r.get('change_pct', 0)
        rsi = r.get('rsi', 50)
        
        if trend_score >= 70 and change_pct > 2:
            ml_html += f'<br><span style="color:var(--pos);font-size:0.75em">📈 <strong>Calls</strong> - strong uptrend</span>'
        elif trend_score <= -70 or death_cross:
            ml_html += f'<br><span style="color:var(--neg);font-size:0.75em">📉 <strong>Puts</strong> - bearish signals</span>'
        elif rsi >= 70 and change_pct > 1:
            ml_html += f'<br><span style="color:#ff8800;font-size:0.75em">📈 <strong>Calls</strong> - momentum</span>'
        elif rsi <= 30 and change_pct < -1:
            ml_html += f'<br><span style="color:#ff8800;font-size:0.75em">📉 <strong>Puts</strong> - oversold</span>'
        
        # Add earnings move suggestions for table view
        earnings_date_iso = r.get('earnings_date_iso')
        implied_move_pct = r.get('implied_move_pct')
        
        if earnings_date_iso and implied_move_pct:
            try:
                earnings_dt = datetime.fromisoformat(earnings_date_iso).replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                days_until_earnings = (earnings_dt - now).days
                
                if 0 <= days_until_earnings <= 30:  # Within a month
                    if implied_move_pct >= 8:  # High volatility expected
                        if trend_score >= 50:  # Bullish technicals
                            ml_html += f'<br><span style="color:#ff6b35;font-size:0.75em">💥 <strong>Straddle</strong> - high vol earnings ({days_until_earnings}d)</span>'
                        elif trend_score <= -50:  # Bearish technicals
                            ml_html += f'<br><span style="color:#ff6b35;font-size:0.75em">💥 <strong>Strangle</strong> - vol play ({days_until_earnings}d)</span>'
                        else:  # Neutral technicals
                            ml_html += f'<br><span style="color:#ff6b35;font-size:0.75em">💥 <strong>Straddle</strong> - earnings vol ({days_until_earnings}d)</span>'
                    elif implied_move_pct >= 5:  # Moderate volatility
                        if rsi >= 70 and change_pct > 1:
                            ml_html += f'<br><span style="color:#ff8800;font-size:0.75em">📈 <strong>Calls</strong> - earnings momentum ({days_until_earnings}d)</span>'
                        elif rsi <= 30 and change_pct < -1:
                            ml_html += f'<br><span style="color:#ff8800;font-size:0.75em">📉 <strong>Puts</strong> - earnings weakness ({days_until_earnings}d)</span>'
                        elif trend_score >= 30:
                            ml_html += f'<br><span style="color:#4CAF50;font-size:0.75em">📅 <strong>Pre-earnings</strong> - watch calls ({days_until_earnings}d)</span>'
                        elif trend_score <= -30:
                            ml_html += f'<br><span style="color:#f44336;font-size:0.75em">📅 <strong>Pre-earnings</strong> - watch puts ({days_until_earnings}d)</span>'
                elif days_until_earnings < 0 and abs(days_until_earnings) <= 1:  # Earnings today or yesterday
                    if implied_move_pct >= 6:
                        ml_html += f'<br><span style="color:#9c27b0;font-size:0.75em">⚡ <strong>Post-earnings</strong> - gap opportunity</span>'
                        
            except Exception:
                pass  # Skip earnings suggestions if date parsing fails
        
        indicators_html += f"<br>{ml_html}"
        
        # Add confidence and risk management if available
        if conf_str != 'N/A':
            indicators_html += f"<br>Confidence: {conf_str}"
        if risk_str:
            indicators_html += f"<br><span style='font-size:0.85em'>{risk_str}</span>"

        # include dividend dataset (percent) for filtering
        div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # sentiment text (exclude analyst rating)
        sent_text = r["sentiment"]
        # Get active signal for filtering
        active_sig = r.get("active_signal") or r.get("bb_signal") or ""
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        html += f"""<tr class="stock-row" data-ticker="{r['ticker']}" data-change="{r['change_pct']}" data-change-5d="{r.get('change_5d') or ''}" data-earnings="{r.get('earnings_date_iso') or ''}" data-rsi="{r['rsi'] or 50}" data-vol="{r['volume_raw']}" data-meme="{r['is_meme_stock']}" data-squeeze="{r['squeeze_level']}" data-bb-width="{bb_width_val}" data-dividend="{div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig}">
    <td><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}{trend_arrow}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.9em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.9em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.9em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.9em">S</a>)</td>
<td data-sort="{r['price']:.2f}">${r['price']:.2f} {r['sparkline']}</td>
<td data-sort="{r['volume_raw']}">{fmt_vol(r['volume'])} {r.get('sparkline_vol', '')}</td>
<td>{fmt_change(r['change_pct'], r['change_abs_day'])}</td>
<td>{fmt_change(r.get('change_5d'), r.get('change_abs_5d'))} {r.get('sparkline_5d', '')}</td>
<td>{fmt_change(r['change_1m'], r['change_abs_1m'])} {r.get('sparkline_1m', '')}</td>
<td>{fmt_change(r.get('change_3m'), r.get('change_abs_3m'))} {r.get('sparkline_3m', '')}</td>
<td>{fmt_change(r['change_6m'], r['change_abs_6m'])} {r.get('sparkline_6m', '')}</td>
<td>{fmt_change(r['change_ytd'], r['change_abs_ytd'])} {r.get('sparkline_ytd', '')}</td>
<td>{fmt_change(r.get('change_1y'), r.get('change_abs_1y'))} {r.get('sparkline_1y', '')}</td>
<td>{fmt_3yr10k(r.get('change_3y'), r.get('value_10k_3y'))} {r.get('sparkline_3y', '')}</td>
<td style="width:90px;min-width:70px;max-width:100px;padding-right:1px;">{ranges_html}</td>
<td style="width:90px;min-width:70px;max-width:100px;padding-left:1px;">{indicators_html}<hr style="margin:2px 0;opacity:0.3"> <span class="{sent_cls}">{sent_text}</span><br><span class="{upside_cls}">Upside: {na(r['upside_potential'],"{:+.1f}%")}</span></td>
</tr>"""

    html += "</table></div></div><div id='cardView'><div class='card-grid'>"
    for _, r in df.iterrows():
        bg = "rgba(0,170,0,0.1)" if r["change_pct"] > 0 else "rgba(204,0,0,0.1)"
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.5em">🟠</span> '
        bb_icon = signal_icon  # Alias for backward compatibility
        
        # Get color-coded trend arrow
        trend_arrow = ""
        trend_label = r.get('trend_label', 'NEUTRAL')
        predicted_trend = r.get('predicted_trend', '→')
        if trend_label == 'UP':
            trend_arrow = f'<span style="color:var(--pos);font-weight:bold"> {predicted_trend}</span>'
        elif trend_label == 'DOWN':
            trend_arrow = f'<span style="color:var(--neg);font-weight:bold"> {predicted_trend}</span>'
        else:
            trend_arrow = f'<span style="color:var(--neutral);font-weight:bold"> {predicted_trend}</span>'
        
        hv = r["hv_30_annualized"]
        hv_cls = "negative" if hv and hv > 50 else "neutral"
        hv_str = na(hv, "{:.1f}%")
        # Color coding for card attributes
        macd_num_cls = "neutral"
        try:
            ml = r.get("macd_line")
            macd_num_cls = (
                "positive"
                if ml is not None and float(ml) >= 0
                else "negative" if ml is not None else "neutral"
            )
        except Exception:
            macd_num_cls = "neutral"

        pc_val = r.get("pc_ratio")
        if pc_val is None:
            pc_cls = "neutral"
        else:
            try:
                pv = float(pc_val)
                pc_cls = (
                    "negative" if pv > 1.2 else "positive" if pv < 0.8 else "neutral"
                )
            except Exception:
                pc_cls = "neutral"

        pe_val = r.get("pe")
        if pe_val is None:
            pe_cls = "neutral"
        else:
            try:
                pv = float(pe_val)
                pe_cls = "negative" if pv > 30 else "positive" if pv < 15 else "neutral"
            except Exception:
                pe_cls = "neutral"

        div_val = r.get("dividend_yield")
        div_cls = "positive" if div_val is not None and div_val > 0 else "neutral"
        # Display dividend yield with a '%' suffix but keep the raw dataset unmodified
        div_yield_display = na(div_val, "{}") + "%" if div_val is not None else "N/A"

        mcap_val = r.get("market_cap")
        aum_val = r.get("aum")
        # determine display value and label (Market Cap preferred, fallback to AUM)
        display_val = None
        display_label = "Market Cap"
        if mcap_val is not None and pd.notna(mcap_val):
            display_val = mcap_val
            display_label = "Market Cap"
        elif aum_val is not None and pd.notna(aum_val):
            display_val = aum_val
            display_label = "AUM"
        try:
            base_val = float(display_val) if display_val is not None else None
        except Exception:
            base_val = None
        try:
            mcap_cls = (
                "strong-bull"
                if base_val is not None and base_val >= 200e9
                else "bull" if base_val is not None and base_val >= 10e9 else "neutral"
            )
        except Exception:
            mcap_cls = "neutral"

        # 52W display for card view
        card_y52_low = r.get("52w_low")
        card_y52_high = r.get("52w_high")
        if pd.notna(card_y52_low) and pd.notna(card_y52_high):
            card_y52_display = f"${card_y52_low:.2f} - ${card_y52_high:.2f}"
        else:
            card_y52_display = "N/A"

        # include dividend dataset for card
        card_div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # Option direction and short percent/days color coding for card
        opt_dir_val = r.get("options_direction") or "Neutral"
        if "Strong Bear" in opt_dir_val or "Strong Bearish" in opt_dir_val:
            opt_dir_cls = "strong-bear"
        elif "Bear" in opt_dir_val:
            opt_dir_cls = "bear"
        elif "Strong Bull" in opt_dir_val or "Strong Bullish" in opt_dir_val:
            opt_dir_cls = "strong-bull"
        elif "Bull" in opt_dir_val:
            opt_dir_cls = "bull"
        else:
            opt_dir_cls = "neutral"

        short_pct = r.get("short_percent")
        days_cover = r.get("days_to_cover")
        try:
            sp = float(short_pct) if short_pct is not None else None
        except Exception:
            sp = None
        if sp is None:
            short_cls = "neutral"
        elif sp >= 20:
            short_cls = "strong-bear"
        elif sp >= 10:
            short_cls = "bear"
        elif sp >= 5:
            short_cls = "vol-hot"
        else:
            short_cls = "neutral"
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        active_sig = r.get("active_signal") or r.get("bb_signal") or ""
        # Risk management and confidence for card view
        card_confidence_val = r.get('signal_confidence')
        card_strength_val = r.get('signal_strength')
        card_atr_val = r.get('atr_14')
        card_stop_loss_val = r.get('stop_loss_price')
        card_rr_ratio = r.get('risk_reward_ratio')
        card_pos_size = r.get('position_size_pct')
        
        # Build confidence display
        card_conf_html = ''
        if card_confidence_val is not None and card_strength_val is not None and card_confidence_val == card_confidence_val:
            if card_confidence_val >= 0.6:
                card_conf_cls = "strong-bull"
            elif card_confidence_val >= 0.3:
                card_conf_cls = "neutral"
            else:
                card_conf_cls = "strong-bear"
            card_conf_html = f'<div>Signal Confidence: <span class="{card_conf_cls}"><strong>{card_strength_val} ({card_confidence_val:.0%})</strong></span></div>'
        else:
            # Show HOLD for neutral positions or when confidence is unavailable
            card_conf_html = '<div>Signal Confidence: <span class="neutral"><strong>HOLD</strong></span></div>'
        
        # Build risk management display (only when values available)
        card_risk_parts = []
        if card_atr_val is not None:
            card_risk_parts.append(f'ATR(14): ${card_atr_val:.2f}')
        if card_stop_loss_val is not None:
            card_risk_parts.append(f'Stop Loss: ${card_stop_loss_val:.2f}')
        if card_rr_ratio is not None:
            card_risk_parts.append(f'R:R: {card_rr_ratio:.1f}:1')
        if card_pos_size is not None:
            card_risk_parts.append(f'Position: {card_pos_size:.1f}%')
        
        card_risk_html = ''
        if card_risk_parts:
            card_risk_html = f'<div style="font-size:0.9em;margin-top:5px;padding-top:5px;border-top:1px solid rgba(255,255,255,0.1)">'
            card_risk_html += ' | '.join(card_risk_parts)
            card_risk_html += '</div>'
        
        # Build Trade Setup display (Entry, Stop Loss, Target with visual indicators)
        card_trade_setup_html = ''
        active_sig = r.get("active_signal") or r.get("bb_signal")
        
        # Show trade setup recommendations for all tickers with ATR data
        if r.get('price') is not None and card_atr_val is not None and card_atr_val > 0:
            entry_price = r['price']
            ATR_MULTIPLIER = 2.0
            
            # Determine trade direction based on signal, default to LONG if no signal
            if active_sig == 'SHORT':
                trade_type = 'SHORT'
                stop_loss_price = entry_price + (card_atr_val * ATR_MULTIPLIER)
                target_price = entry_price - (card_atr_val * ATR_MULTIPLIER * 2.0)
                setup_color = 'var(--neg)'
                setup_icon = '🔴'
            else:
                # Show LONG setup for BUY signals or as default recommendation
                trade_type = 'LONG' if active_sig != 'BUY' else 'BUY'
                stop_loss_price = entry_price - (card_atr_val * ATR_MULTIPLIER)
                target_price = entry_price + (card_atr_val * ATR_MULTIPLIER * 2.0)
                setup_color = 'var(--pos)'
                setup_icon = '🟢'
            
            # Calculate percentages
            risk_per_share = abs(entry_price - stop_loss_price)
            risk_pct = (risk_per_share / entry_price) * 100
            
            if trade_type in ('BUY', 'LONG'):
                reward_pct = ((target_price - entry_price) / entry_price) * 100
            else:  # SHORT
                reward_pct = ((entry_price - target_price) / entry_price) * 100
            
            # Add signal indicator
            signal_label = f" - {active_sig}" if active_sig in ('BUY', 'SHORT', 'SELL', 'HOLD') else ""
            
            card_trade_setup_html = f'''
<div style="margin-top:10px;padding:12px;border:2px solid {setup_color};border-radius:8px;background:var(--card);box-shadow:0 2px 8px rgba(0,0,0,0.2)">
<div style="font-weight:bold;font-size:1.1em;margin-bottom:10px;color:{setup_color}">{setup_icon} TRADE SETUP ({trade_type}{signal_label})</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.95em;color:var(--text)">
<div style="text-align:left"><strong>Entry:</strong> ${entry_price:.2f}</div>
<div style="text-align:right;color:var(--text);opacity:1">Current</div>
<div style="text-align:left"><strong>Stop Loss:</strong> ${stop_loss_price:.2f}</div>
<div style="text-align:right;color:#ff4444;font-weight:bold">-{risk_pct:.1f}%</div>
<div style="text-align:left"><strong>Target:</strong> ${target_price:.2f}</div>
<div style="text-align:right;color:#00cc00;font-weight:bold">+{reward_pct:.1f}%</div>
</div>
<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);text-align:center;font-size:0.9em;color:var(--text)">
<strong>Risk/Reward:</strong> <span style="color:{setup_color};font-weight:bold">{risk_pct:.1f}% / {reward_pct:.1f}% (1:2)</span>
</div>
</div>'''
        
        # Upside color coding
        card_upside_cls = (
            "bullish"
            if r.get("upside_potential") and r.get("upside_potential") > 0
            else (
                "bearish"
                if r.get("upside_potential") and r.get("upside_potential") < 0
                else "neutral"
            )
        )
        
        # Build moving averages display with cross detection
        card_ma_html = ''
        if r.get('sma_50') or r.get('sma_200'):
            ma_parts = []
            if r.get('sma_50'):
                ma_parts.append(f'50d: ${r.get("sma_50"):.2f}')
            if r.get('sma_200'):
                ma_parts.append(f'200d: ${r.get("sma_200"):.2f}')
            
            cross_indicator = ''
            if r.get('death_cross'):
                cross_indicator = ' <strong class="bearish">⚠ DEATH CROSS</strong>'
            elif r.get('golden_cross'):
                cross_indicator = ' <strong class="bullish">✓ GOLDEN CROSS</strong>'
            
            card_ma_html = f'<div>MA: {" | ".join(ma_parts)}{cross_indicator}</div>'
        
        # Build ranges_html for card view (recalculate per card, not reuse from table view)
        card_bb_bar = ""
        if (
            pd.notna(r.get("bb_position_pct"))
            and pd.notna(r.get("bb_lower"))
            and pd.notna(r.get("bb_middle"))
            and pd.notna(r.get("bb_upper"))
        ):
            try:
                pos = float(r["bb_position_pct"])
                pos = max(0.0, min(100.0, pos))
                bb_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {pos}%, var(--neg) {pos}%, var(--neg) 100%)"
                card_bb_bar = f'<div class="range-container"><div class="range-title">Bollinger Bands</div><div class="range-bar" style="background:{bb_color}"><div class="range-bar-marker" style="left:{pos}%"></div></div><div class="range-labels"><span>${na(r["bb_lower"])}</span><span>${na(r["bb_middle"])}</span><span>${na(r["bb_upper"])}</span></div><div style="font-size:0.75em;text-align:center">Width: {na(r["bb_width_pct"],"{:.1f}")}% – {r["bb_status"]}</div></div>'
            except Exception:
                card_bb_bar = ""

        card_impl_bar = ""
        if (
            pd.notna(r.get("implied_move_pct"))
            and pd.notna(r.get("implied_low"))
            and pd.notna(r.get("implied_high"))
        ):
            try:
                im_pct = float(r["implied_move_pct"])
                if im_pct > 0:
                    left_pct = 50 - im_pct / 2
                    right_pct = 50 + im_pct / 2
                    i_color = f"linear-gradient(to right, var(--neg) 0%, var(--neg) {left_pct}%, var(--pos) {right_pct}%, var(--pos) 100%)"
                    card_impl_bar = f'<div class="range-container"><div class="range-title">Implied Move ±{im_pct:.1f}%</div><div class="range-bar" style="background:{i_color}"><div class="range-bar-marker" style="left:50%"></div></div><div class="range-labels"><span>${na(r["implied_low"])}</span><span>${na(r["implied_high"])}</span></div></div>'
            except Exception:
                card_impl_bar = ""

        card_day_block = ""
        if (
            pd.notna(r.get("day_low"))
            and pd.notna(r.get("day_high"))
            and r["day_high"] is not None
            and r["day_low"] is not None
            and (r["day_high"] - r["day_low"]) > 0
        ):
            day_pos = (r["price"] - r["day_low"]) / (r["day_high"] - r["day_low"]) * 100
            day_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {day_pos}%, var(--neg) {day_pos}%, var(--neg) 100%)"
            card_day_block = f"""
    <div class="range-container"><div class="range-title">Day</div><div class="range-bar" style="background:{day_color}"><div class="range-bar-marker" style="left:{day_pos}%"></div></div><div class="range-labels"><span>${r['day_low']:.2f}</span><span>${r['day_high']:.2f}</span></div></div>
    """

        card_y52_block = ""
        if (
            pd.notna(r.get("52w_low"))
            and pd.notna(r.get("52w_high"))
            and r["52w_high"] is not None
            and r["52w_low"] is not None
            and (r["52w_high"] - r["52w_low"]) > 0
        ):
            y52_pos = (r["price"] - r["52w_low"]) / (r["52w_high"] - r["52w_low"]) * 100
            y52_color = f"linear-gradient(to right, var(--pos) 0%, var(--pos) {y52_pos}%, var(--neg) {y52_pos}%, var(--neg) 100%)"
            card_y52_block = f"""
    <div class="range-container"><div class="range-title">52W</div><div class="range-bar" style="background:{y52_color}"><div class="range-bar-marker" style="left:{y52_pos}%"></div></div><div class="range-labels"><span>${r['52w_low']:.2f}</span><span>${r['52w_high']:.2f}</span></div></div>
    """

        card_ranges_html = f"{card_day_block}{card_y52_block}{card_bb_bar}{card_impl_bar}"
        
        html += f"""<div class="stock-card stock-row" style="background:{bg}" 
            data-ticker="{r['ticker']}" 
            data-change="{r['change_pct']}" 
            data-change-5d="{r.get('change_5d') or ''}" 
            data-earnings="{r.get('earnings_date_iso') or ''}"
            data-rsi="{r['rsi'] or 50}" 
            data-vol="{r['volume_raw']}" 
            data-meme="{r['is_meme_stock']}" 
            data-squeeze="{r['squeeze_level']}" 
            data-bb-width="{bb_width_val}" data-dividend="{card_div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig}">
    <button class="card-scroll-btn card-scroll-left" onclick="scrollCardContent(this, -1)">&lt;</button>
    <button class="card-scroll-btn card-scroll-right" onclick="scrollCardContent(this, 1)">&gt;</button>
    <div class="card-content-scroll">
        <div class="card-page">
    <h2><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}{trend_arrow}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.8em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.8em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.8em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.8em">S</a>) ${r['price']:.2f}</h2>
<div style="font-size:1.5em">{fmt_change(r['change_pct'], r['change_abs_day'])}</div>
{r['sparkline']}
<div>5D: {fmt_change(r.get('change_5d'), r.get('change_abs_5d'))} {r.get('sparkline_5d', '')}</div>
<div>1M: {fmt_change(r['change_1m'], r['change_abs_1m'])} {r.get('sparkline_1m', '')}</div>
<div>6M: {fmt_change(r['change_6m'], r['change_abs_6m'])} {r.get('sparkline_6m', '')}</div>
<div>YTD: {fmt_change(r['change_ytd'], r['change_abs_ytd'])} {r.get('sparkline_ytd', '')}</div>
<div>1Y: {fmt_change(r.get('change_1y'), r.get('change_abs_1y'))} {r.get('sparkline_1y', '')}</div>
<div>Volume: {fmt_vol(r['volume'])} {r.get('sparkline_vol', '')}</div>
<div><strong>52W: {card_y52_display}</strong></div>
<div>{display_label}: <span class="{mcap_cls}"><strong>{fmt_mcap(display_val)}</strong></span></div>
<div><span class="{hv_cls}">Volatility: {hv_str}</span></div>
<div>BB: {r['bb_status']} ({na(r['bb_width_pct'], '{:.1f}%')})</div>
<div>MACD: <span class="{macd_num_cls}">{na(r.get('macd_line'), '{:+.3f}')}</span> | <span class="{macd_num_cls}">{na(r.get('macd_signal'), '{:+.3f}')}</span> (<span class="{ 'bullish' if r.get('macd_label')=='Bullish' else 'bearish' if r.get('macd_label')=='Bearish' else 'neutral' }">{r.get('macd_label','N/A')}</span>)</div>
{card_ma_html}
<div>P/C Vol Ratio: <span class="{pc_cls}">{na(r.get('pc_ratio'), '{:.2f}')}</span></div>
<div><strong>Opt Dir: <span class="{opt_dir_cls}">{opt_dir_val}</span> &nbsp; Short: <span class="{short_cls}">{na(short_pct, '{:.1f}%')}</span> ({na(days_cover, '{:.1f}d')})</strong></div>
        </div>
        <div class="card-page">
<div>P/E: <span class="{pe_cls}">{na(r.get('pe'), '{:.2f}')}</span></div>
<div>EPS: <span class="{pe_cls}">{na(r.get('eps'), '{:.2f}')}</span></div>
<div>1y Target Est: <span class="{card_upside_cls}">{na(r.get('target_price'), '${:.2f}')}</span></div>
<div>Upside %: <span class="{card_upside_cls}">{na(r.get('upside_potential'), '{:+.1f}%')}</span></div>
<div>Div: <span class="{div_cls}">{na(r.get('dividend_rate'), '${:.2f}')}</span> (<span class="{div_cls}">{div_yield_display}</span>)</div>
<div>Annual Dividend: <span class="{div_cls}">{na(r.get('dividend_rate'), '${:.2f}')}</span></div>
<div>Ex-Dividend Date: {r.get('ex_dividend_date') or 'N/A'}</div>
<div>Payout Frequency: {r.get('payout_frequency') or 'N/A'}</div>
<div>Payout Ratio: <span class="{div_cls}">{na(r.get('payout_ratio'), '{:.2f}%')}</span></div>
<div>Dividend Growth: <span class="{div_cls}">{na(r.get('dividend_growth'), '{:.2f}%')}</span></div>
<div>Earnings: <strong>{r.get('earnings_date') or 'N/A'}</strong></div>
{card_conf_html}
{card_risk_html}
{card_trade_setup_html}
        </div>
        <div class="card-page">
<div><strong>ML Predictions</strong></div>"""
        
        # Add ML predictions to card view
        card_ml_score = r.get('ml_breakout_score', 0)
        card_ml_crash = r.get('ml_crash_risk', 0)
        card_ml_win_rate = r.get('ml_historical_win_rate', 0.0)
        card_ml_expected_return = r.get('ml_expected_return', 0.0)
        card_ml_sample_size = r.get('ml_sample_size', 0)
        
        if card_ml_score > 0:
            if card_ml_score >= 70:
                html += f'<div><span style="color:var(--pos);font-weight:bold">🚀 BREAKOUT: {card_ml_score}%</span></div>'
            elif card_ml_score >= 50:
                html += f'<div><span style="color:#ff8800;font-weight:600">🚀 BREAKOUT: {card_ml_score}%</span></div>'
            else:
                html += f'<div><span style="font-size:0.9em">🚀 BREAKOUT: {card_ml_score}%</span></div>'
        if card_ml_crash >= 50:
            html += f'<div><span style="color:var(--neg)">⚠️ CRASH RISK: {card_ml_crash}%</span></div>'
        
        # Add historical performance if available
        if card_ml_win_rate > 0 and card_ml_sample_size > 0:
            perf_color = "var(--pos)" if card_ml_win_rate >= 70 else "var(--neg)" if card_ml_win_rate < 50 else "#ff8800"
            html += f'<div style="font-size:0.9em;margin-top:8px"><span style="color:{perf_color}">📊 {card_ml_win_rate:.0f}% win rate ({card_ml_sample_size} trades)</span></div>'
            if abs(card_ml_expected_return) > 0.1:  # Only show if meaningful
                ret_color = "var(--pos)" if card_ml_expected_return > 0 else "var(--neg)"
                html += f'<div style="font-size:0.9em"><span style="color:{ret_color}">💰 Expected: {card_ml_expected_return:+.1f}%</span></div>'
        
        # Add options strategy suggestions based on technical indicators
        trend_score = r.get('trend_score', 0)
        death_cross = r.get('death_cross', False)
        golden_cross = r.get('golden_cross', False)
        change_pct = r.get('change_pct', 0)
        rsi = r.get('rsi', 50)
        
        if trend_score >= 70 and change_pct > 2:
            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:var(--pos)">📈 <strong>Options:</strong> Consider buying calls - strong uptrend</span></div>'
        elif trend_score <= -70 or death_cross:
            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:var(--neg)">📉 <strong>Options:</strong> Consider buying puts - bearish signals</span></div>'
        elif rsi >= 70 and change_pct > 1:
            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff8800">📈 <strong>Options:</strong> Consider buying calls - overbought but momentum</span></div>'
        elif rsi <= 30 and change_pct < -1:
            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff8800">📉 <strong>Options:</strong> Consider buying puts - oversold with weakness</span></div>'
        
        # Add earnings move suggestions for card view
        earnings_date_iso = r.get('earnings_date_iso')
        implied_move_pct = r.get('implied_move_pct')
        
        if earnings_date_iso and implied_move_pct:
            try:
                earnings_dt = datetime.fromisoformat(earnings_date_iso).replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                days_until_earnings = (earnings_dt - now).days
                
                if 0 <= days_until_earnings <= 30:  # Within a month
                    if implied_move_pct >= 8:  # High volatility expected
                        if trend_score >= 50:  # Bullish technicals
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff6b35">💥 <strong>Earnings:</strong> Consider straddle - high volatility expected ({days_until_earnings}d)</span></div>'
                        elif trend_score <= -50:  # Bearish technicals
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff6b35">💥 <strong>Earnings:</strong> Consider strangle - volatility play ({days_until_earnings}d)</span></div>'
                        else:  # Neutral technicals
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff6b35">💥 <strong>Earnings:</strong> Consider straddle - earnings volatility ({days_until_earnings}d)</span></div>'
                    elif implied_move_pct >= 5:  # Moderate volatility
                        if rsi >= 70 and change_pct > 1:
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff8800">📈 <strong>Earnings:</strong> Consider calls - momentum into earnings ({days_until_earnings}d)</span></div>'
                        elif rsi <= 30 and change_pct < -1:
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#ff8800">📉 <strong>Earnings:</strong> Consider puts - weakness into earnings ({days_until_earnings}d)</span></div>'
                        elif trend_score >= 30:
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#4CAF50">📅 <strong>Earnings:</strong> Watch for call opportunities pre-earnings ({days_until_earnings}d)</span></div>'
                        elif trend_score <= -30:
                            html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#f44336">📅 <strong>Earnings:</strong> Watch for put opportunities pre-earnings ({days_until_earnings}d)</span></div>'
                elif days_until_earnings < 0 and abs(days_until_earnings) <= 1:  # Earnings today or yesterday
                    if implied_move_pct >= 6:
                        html += f'<div style="font-size:0.9em;margin-top:4px"><span style="color:#9c27b0">⚡ <strong>Earnings:</strong> Post-earnings gap trading opportunity</span></div>'
                        
            except Exception:
                pass  # Skip earnings suggestions if date parsing fails
        
        html += f"""
{card_ranges_html}
        </div>
    </div>
</div>"""

    html += "</div></div><div id='heatView'><div class='heat-grid'>"
    for _, r in df.iterrows():
        intensity = min(abs(r["change_pct"]) / 15, 1)
        bg = (
            f"rgba(0,170,0,{intensity})"
            if r["change_pct"] >= 0
            else f"rgba(204,0,0,{intensity})"
        )
        bb_width_val = r["bb_width_pct"] if r["bb_width_pct"] is not None else 100
        price_display = (
            f"${r['price']:.2f}"
            if (r.get("price") is not None and pd.notna(r.get("price")))
            else "N/A"
        )

        # Prefer Market Cap, fallback to AUM when market cap missing
        mcap_val = r.get("market_cap")
        aum_val = r.get("aum")
        display_val = None
        display_label = "Market Cap"
        if mcap_val is not None and pd.notna(mcap_val):
            display_val = mcap_val
            display_label = "Market Cap"
        elif aum_val is not None and pd.notna(aum_val):
            display_val = aum_val
            display_label = "AUM"
        try:
            base_val = float(display_val) if display_val is not None else None
        except Exception:
            base_val = None
        try:
            mcap_cls = (
                "strong-bull"
                if base_val is not None and base_val >= 200e9
                else "bull" if base_val is not None and base_val >= 10e9 else "neutral"
            )
        except Exception:
            mcap_cls = "neutral"

        # 52W range for heatmap view
        heat_y52_low = r.get("52w_low")
        heat_y52_high = r.get("52w_high")
        if pd.notna(heat_y52_low) and pd.notna(heat_y52_high):
            heat_y52_display = f"${heat_y52_low:.2f} - ${heat_y52_high:.2f}"
        else:
            heat_y52_display = "N/A"

        # include dividend dataset for heat tiles
        heat_div_ds = (
            r.get("dividend_yield")
            if r.get("dividend_yield") is not None
            else (r.get("dividend_rate") if r.get("dividend_rate") is not None else "")
        )
        # Get signal icon from active strategy
        signal_icon = ""
        active_sig = r.get("active_signal") or r.get("bb_signal")
        if active_sig == "BUY":
            signal_icon = '<span style="font-size:0.5em">🟢</span> '
        elif active_sig == "SHORT":
            signal_icon = '<span style="font-size:0.5em">🔴</span> '
        elif active_sig == "SELL":
            signal_icon = '<span style="font-size:0.5em">🟠</span> '
        bb_icon = signal_icon  # Alias for backward compatibility
        
        # Get color-coded trend arrow
        trend_arrow = ""
        trend_label = r.get('trend_label', 'NEUTRAL')
        predicted_trend = r.get('predicted_trend', '→')
        if trend_label == 'UP':
            trend_arrow = f'<span style="color:var(--pos);font-weight:bold"> {predicted_trend}</span>'
        elif trend_label == 'DOWN':
            trend_arrow = f'<span style="color:var(--neg);font-weight:bold"> {predicted_trend}</span>'
        else:
            trend_arrow = f'<span style="color:var(--neutral);font-weight:bold"> {predicted_trend}</span>'
        
        # Build Zacks URL based on quote type
        ticker_l = r['ticker'].lower()
        qt = r.get('quote_type', '').upper()
        if qt == 'ETF':
            zacks_url = f"https://www.zacks.com/funds/etf/{r['ticker']}/profile?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/etf/{ticker_l}/"
        elif qt == 'MUTUALFUND':
            zacks_url = f"https://www.zacks.com/funds/mutual-fund/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/quote/mutf/{r['ticker']}/"
        else:
            zacks_url = f"https://www.zacks.com/stock/quote/{r['ticker']}?q={ticker_l}"
            stock_analysis_url = f"https://stockanalysis.com/stocks/{ticker_l}/"
        active_sig_heat = r.get("active_signal") or r.get("bb_signal") or ""
        html += f"""<div class="heat-tile stock-row" style="background:{bg}" 
        data-ticker="{r['ticker']}" 
        data-change="{r['change_pct']}" 
        data-change-5d="{r.get('change_5d') or ''}"
        data-earnings="{r.get('earnings_date_iso') or ''}"
        data-rsi="{r['rsi'] or 50}" 
        data-vol="{r['volume_raw']}" 
        data-meme="{r['is_meme_stock']}" 
        data-squeeze="{r['squeeze_level']}" 
        data-bb-width="{bb_width_val}" data-dividend="{heat_div_ds}" data-category="{r.get('category') or ''}" data-signal="{active_sig_heat}">
    <strong><a href="https://www.barchart.com/stocks/quotes/{r['ticker']}" target="_blank">{bb_icon}{r['ticker']}{trend_arrow}</a> (<a href="https://finance.yahoo.com/quote/{r['ticker']}" target="_blank" style="font-size:0.85em">Y</a>, <a href="https://finviz.com/quote.ashx?t={r['ticker']}" target="_blank" style="font-size:0.85em">F</a>, <a href="{zacks_url}" target="_blank" style="font-size:0.85em">Z</a>, <a href="{stock_analysis_url}" target="_blank" style="font-size:0.85em">S</a>) {price_display}</strong>
    <div style="margin-top:6px">{fmt_change(r['change_pct'], r.get('change_abs_day'))}</div>
    <div style="font-size:0.85em">5D: {fmt_change(r.get('change_5d'), r.get('change_abs_5d'))} {r.get('sparkline_5d', '')}</div>
    <div style="font-size:0.85em">1M: {fmt_change(r['change_1m'], r['change_abs_1m'])} {r.get('sparkline_1m', '')}</div>
    <div style="font-size:0.85em">Vol: {fmt_vol(r['volume'])} {r.get('sparkline_vol', '')}</div>
    <div style="font-size:0.9em;margin-top:6px"><strong>52W: {heat_y52_display}</strong></div>
    <div style="font-size:0.9em">{display_label}: <span class="{mcap_cls}"><strong>{fmt_mcap(display_val)}</strong></span></div>
    </div>"""

    html += "</div></div></div>"

    # Inject M7_TICKERS and STAR_TICKERS arrays into JavaScript
    m7_tickers_js = json.dumps(list(M7_STOCKS))
    star_tickers_js = json.dumps(list(STAR_STOCKS))
    html += f"""
<script>
const M7_TICKERS = {m7_tickers_js};
const STAR_TICKERS = {star_tickers_js};
"""
    html += """const prefsKey = 'dash_prefs';
let prefs = JSON.parse(localStorage.getItem(prefsKey) || '{"theme":"light","view":"heat"}');
document.documentElement.setAttribute('data-theme', prefs.theme);

function setView(btn, view) {
    document.getElementById('tableView').style.display = view==='table' ? 'block' : 'none';
    document.getElementById('cardView').style.display = view==='card' ? 'block' : 'none';
    document.getElementById('heatView').style.display = view==='heat' ? 'block' : 'none';
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    prefs.view = view;
    localStorage.setItem(prefsKey, JSON.stringify(prefs));
    applyFilter();
    
    // Update card arrows when switching to card view
    if (view === 'card') {
        setTimeout(() => {
            document.querySelectorAll('.stock-card').forEach(card => {
                updateCardArrows(card);
            });
        }, 50);
    }
}

let currentFilter = 'all';
function applyFilter() {
    const rows = document.querySelectorAll('.stock-row');
    const tickerVal = (document.getElementById('tickerFilter') && document.getElementById('tickerFilter').value) ? document.getElementById('tickerFilter').value.trim().toLowerCase() : '';
    rows.forEach(r => {
        let show = true;
        const ch = parseFloat(r.dataset.change || 0);
        const ch5 = parseFloat(r.dataset.change5d || 0);
        const rsi = parseFloat(r.dataset.rsi || 50);
        const vol = parseFloat(r.dataset.vol || 0);
        const meme = r.dataset.meme === 'True';
        const cat = (r.dataset.category || '').toLowerCase();
        const sq = r.dataset.squeeze || 'None';
        const bbw = parseFloat(r.dataset.bbWidth || 100);
        const sig = (r.dataset.signal || '').toUpperCase();
        const ticker = (r.dataset.ticker || '').toUpperCase();
        if (currentFilter === 'oversold') show = rsi < 30;
        else if (currentFilter === 'overbought') show = rsi > 70;
        else if (currentFilter === 'surge') show = (ch > 10) || (ch5 > 10);
        else if (currentFilter === 'crash') show = (ch < -10) || (ch5 < -10);
        else if (currentFilter === 'meme') show = meme;
        else if (currentFilter === 'volume') show = vol > 5e7;
        else if (currentFilter === 'star') show = STAR_TICKERS.includes(ticker);
        else if (currentFilter === 'm7') show = M7_TICKERS.includes(ticker);
        else if (currentFilter === 'squeeze') show = sq !== 'None';
        else if (currentFilter === 'earnings-week') show = (function(){
            const ed = r.dataset.earnings;
            if (!ed) return false;
            try {
                const edDate = new Date(ed + 'T00:00:00');
                edDate.setHours(0,0,0,0);
                const now = new Date();
                const day = now.getDay();
                const start = new Date(now);
                start.setHours(0,0,0,0);
                start.setDate(now.getDate() - day); // week starts Sunday
                const end = new Date(start);
                end.setDate(start.getDate() + 6);
                return edDate >= start && edDate <= end;
            } catch (e) {
                return false;
            }
        })();
        else if (currentFilter === 'bb-squeeze') show = bbw < 6;
        else if (currentFilter === 'dividend') show = parseFloat(r.dataset.dividend || 0) > 0;
        else if (currentFilter === 'signal-buy') show = sig === 'BUY';
        else if (currentFilter === 'signal-sell') show = sig === 'SELL';
        else if (currentFilter === 'signal-short') show = sig === 'SHORT';
        else if (currentFilter === 'signal-hold') show = sig === 'HOLD';
        else if (currentFilter.startsWith('cat-')) show = cat === currentFilter.replace('cat-','');
        if (tickerVal) {
            const tk = (r.dataset.ticker || '').toString().toLowerCase();
            show = show && tk.includes(tickerVal);
        }
        r.style.display = show ? '' : 'none';
    });
}

document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', function() {
    document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
    this.classList.add('active');
    currentFilter = this.dataset.filter;
    applyFilter();
}));

document.querySelectorAll('th[data-sort]').forEach(function(th) {
    th.onclick = function() {
        const col = th.cellIndex;
        const table = document.getElementById('stockTable');
        const rows = Array.from(table.querySelectorAll('tr:nth-child(n+2)'));
        const dir = th.dataset.dir = (th.dataset.dir === 'asc' ? 'desc' : 'asc');
        const multiplier = dir === 'asc' ? 1 : -1;
        
        // Pre-extract sort values for better performance with large datasets
        const rowsWithValues = rows.map(function(row) {
            const cell = row.cells[col];
            const sortValue = cell.dataset.sort || (cell.querySelector('[data-sort]')?.dataset.sort) || cell.textContent.trim();
            return { row: row, value: parseFloat(sortValue) || 0 };
        });
        
        // Sort numerically (all data-sort values are numeric)
        rowsWithValues.sort((a, b) => (a.value - b.value) * multiplier);
        
        // Use DocumentFragment for efficient DOM updates
        const fragment = document.createDocumentFragment();
        rowsWithValues.forEach(item => fragment.appendChild(item.row));
        table.appendChild(fragment);
    };
});

applyFilter();
// Always set the correct view button on load
const viewBtnMap = {
    'heat': 0,
    'card': 1,
    'table': 2
};
const btns = document.querySelectorAll('.view-btn');
const idx = viewBtnMap[prefs.view] ?? 0;
if (btns[idx]) setView(btns[idx], prefs.view);

function toggleTheme() {
    prefs.theme = prefs.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', prefs.theme);
    localStorage.setItem(prefsKey, JSON.stringify(prefs));
}

function scrollCardContent(btn, direction) {
    const card = btn.closest('.stock-card');
    const content = card.querySelector('.card-content-scroll');
    if (content) {
        const currentScroll = content.scrollLeft;
        const cardWidth = content.clientWidth;
        const targetScroll = direction > 0 
            ? Math.ceil(currentScroll / cardWidth) * cardWidth + cardWidth
            : Math.floor(currentScroll / cardWidth) * cardWidth - cardWidth;
        
        content.scrollTo({
            left: Math.max(0, targetScroll),
            behavior: 'smooth'
        });
        
        // Update arrow states after scroll completes
        setTimeout(() => updateCardArrows(card), 300);
    }
}

function updateCardArrows(card) {
    const content = card.querySelector('.card-content-scroll');
    const leftBtn = card.querySelector('.card-scroll-left');
    const rightBtn = card.querySelector('.card-scroll-right');
    
    if (content && leftBtn && rightBtn) {
        const scrollLeft = content.scrollLeft;
        const maxScroll = content.scrollWidth - content.clientWidth;
        
        // Disable left arrow on first page
        leftBtn.disabled = scrollLeft <= 1;
        
        // Disable right arrow on last page
        rightBtn.disabled = scrollLeft >= maxScroll - 1;
    }
}

// Initialize arrow states on page load and add scroll listeners
window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.stock-card').forEach(card => {
        updateCardArrows(card);
        
        const content = card.querySelector('.card-content-scroll');
        if (content) {
            content.addEventListener('scroll', () => updateCardArrows(card));
        }
    });
});

function refreshPage() {
    // Clear dismissed alerts state so they show on refresh
    localStorage.removeItem('alertsDismissed');
    location.reload();
}

function dismissAlerts() {
    const banner = document.getElementById('alertBanner');
    if (banner) {
        banner.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        banner.style.opacity = '0';
        banner.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            banner.style.display = 'none';
        }, 300);
        // Store dismissed state with timestamp
        localStorage.setItem('alertsDismissed', Date.now());
    }
}

// Check if alerts were recently dismissed (within last 5 minutes)
window.addEventListener('DOMContentLoaded', () => {
    const dismissedTime = localStorage.getItem('alertsDismissed');
    const banner = document.getElementById('alertBanner');
    if (banner && dismissedTime) {
        const fiveMinutes = 5 * 60 * 1000;
        if (Date.now() - parseInt(dismissedTime) < fiveMinutes) {
            banner.style.display = 'none';
        }
    }
    
    // Fetch live market indices for GitHub Pages
    fetchMarketIndices();
});

async function fetchMarketIndices() {
    const symbols = [
        {ticker: '^DJI', name: 'Dow'},
        {ticker: '^GSPC', name: 'S&P'},
        {ticker: '^IXIC', name: 'Nasdaq'},
        {ticker: '^VIX', name: 'VIX'}
    ];
    
    try {
        const results = await Promise.all(symbols.map(async ({ticker, name}) => {
            try {
                const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=5d`, {
                    headers: {'User-Agent': 'Mozilla/5.0'}
                });
                if (!response.ok) return null;
                const data = await response.json();
                const quote = data.chart.result[0];
                const meta = quote.meta;
                const price = meta.regularMarketPrice || meta.previousClose;
                const prevClose = meta.chartPreviousClose || meta.previousClose;
                const change = price - prevClose;
                return {name, price, change};
            } catch {
                return null;
            }
        }));
        
        let html = results.map(r => {
            if (!r) return `<span class="neutral">${r?.name || '?'}: N/A</span>`;
            const cls = r.change >= 0 ? 'positive' : 'negative';
            return `<span class="${cls}">${r.name}: ${r.price.toFixed(2)} (${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)})</span>`;
        }).join(' | ');
        
        const indicesEl = document.getElementById('marketIndices');
        if (indicesEl && results.some(r => r !== null)) {
            indicesEl.innerHTML = html;
        }
    } catch (error) {
        console.log('Market indices fetch failed (using server values):', error);
    }
}
</script>
</body></html>"""

    with open(file, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    import time as time_module
    start_time = time_module.time()
    
    os.makedirs("data", exist_ok=True)
    # Pre-load alerts cache at startup
    load_alerts()
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default="data/tickers.csv")
    args = parser.parse_args()

    try:
        dashboard_start = time_module.time()
        # Auto-detect market hours (None = auto-detect)
        df = dashboard(args.csv_file, ext=None)
        ext = not is_regular_hours()  # Determine which session we're in
        file = "data/dashboard.html"
        
        # Clean up old dashboard files if they exist
        for old_file in ["data/reg_dashboard.html", "data/extnd_dashboard.html"]:
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except Exception:
                    pass
        
        alerts = check_alerts(df.to_dict("records"))
        html(
            df,
            get_vix_data(),
            get_fear_greed_data(),
            get_aaii_sentiment(),
            file,
            ext,
            alerts=alerts,
        )
        dashboard_elapsed = time_module.time() - dashboard_start
        print(f"✓ Dashboard generated: {file} (took {dashboard_elapsed / 60:.2f} minutes)")
        
        # Generate summary dashboard
        try:
            summary_start = time_module.time()
            generate_summary_html(args.csv_file)
            summary_elapsed = time_module.time() - summary_start
        except Exception as e:
            print(f"⚠️  Summary generation failed: {e}")
    except Exception as e:
        print(f"✗ Dashboard generation failed: {e}")
    
    total_elapsed = time_module.time() - start_time
    print(f"\n⏱️  Total time: {total_elapsed / 60:.2f} minutes")