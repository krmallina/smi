# --- FIX FUTURE IMPORT POSITION ---
from __future__ import annotations

# --- NOISE REDUCTION: SMOOTHED INDICATORS ---
def get_smoothed_rsi(prices: pd.Series, window: int = 14, smooth_window: int = 3) -> pd.Series:
    """Reduces noise by smoothing the RSI with a secondary moving average."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rolling(window=smooth_window).mean()


import logging
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ML and yfinance imports - install with: pip install scikit-learn xgboost yfinance
ML_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
except Exception as e:
    print(f"[ML_IMPORT_ERROR] {e}")
    import traceback
    traceback.print_exc()
    ML_AVAILABLE = False

try:
    import yfinance as yf  # type: ignore
    YF_AVAILABLE = True
except Exception:
    yf = None  # type: ignore
    YF_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/ml_models")
MODEL_FILE = MODEL_DIR / "breakout_crash_model.pkl"
SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"
PERFORMANCE_FILE = MODEL_DIR / "prediction_performance.pkl"

# Data caching for efficiency
DATA_CACHE_DIR = Path("data/stock_cache")
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class PredictionPerformanceTracker:
    """
Tracks historical predictions and their outcomes to calculate performance metrics.
Stores predictions with timestamps and tracks their actual performance over time.
Provides win rates, expected returns, and confidence intervals.
    """
    
    def __init__(self):
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load historical performance data from disk."""
        if PERFORMANCE_FILE.exists():
            try:
                with open(PERFORMANCE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance data: {e}")
        
        # Initialize empty performance data
        return {
            "predictions": [],  # List of prediction records
            "last_updated": None,
            "metrics": {
                "breakout_win_rate": 0.0,
                "crash_win_rate": 0.0,
                "avg_breakout_return": 0.0,
                "avg_crash_return": 0.0,
                "total_predictions": 0,
                "breakout_predictions": 0,
                "crash_predictions": 0,
            }
        }
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            with open(PERFORMANCE_FILE, "wb") as f:
                pickle.dump(self.performance_data, f)
        except Exception as e:
            logger.warning(f"Failed to save performance data: {e}")
    
    def record_prediction(self, ticker: str, prediction: str, confidence: int, 
                         breakout_score: int, crash_risk: int, current_price: float):
        """
        Record a new prediction for tracking.
        
        Args:
            ticker: Stock ticker
            prediction: BREAKOUT, CRASH, or NEUTRAL
            confidence: Model confidence 0-100
            breakout_score: Breakout probability 0-100
            crash_risk: Crash risk probability 0-100
            current_price: Current stock price
        """
        prediction_record = {
            "ticker": ticker,
            "prediction": prediction,
            "confidence": confidence,
            "breakout_score": breakout_score,
            "crash_risk": crash_risk,
            "entry_price": current_price,
            "timestamp": pd.Timestamp.now(),
            "outcome_price": None,  # Will be filled when outcome is known
            "outcome_date": None,
            "actual_return": None,
            "outcome_determined": False,
        }
        
        self.performance_data["predictions"].append(prediction_record)
        self._cleanup_old_predictions()
        self._update_metrics()
        self._save_performance_data()
    
    def _cleanup_old_predictions(self):
        """
        Clean up old predictions to maintain rolling window:
        - Keep only last 5 trading days (approx 7 calendar days)
        - Limit to maximum 5 entries per ticker
        """
        predictions = self.performance_data["predictions"]
        if not predictions:
            return
            
        now = pd.Timestamp.now()
        cutoff_date = now - pd.Timedelta(days=7)  # 7 calendar days ≈ 5 trading days
        
        # Filter to recent predictions only
        recent_predictions = [
            pred for pred in predictions 
            if pred["timestamp"] >= cutoff_date
        ]
        
        # Group by ticker and limit to 5 most recent per ticker
        ticker_predictions = {}
        for pred in recent_predictions:
            ticker = pred["ticker"]
            if ticker not in ticker_predictions:
                ticker_predictions[ticker] = []
            ticker_predictions[ticker].append(pred)
        
        # Keep only 5 most recent for each ticker
        cleaned_predictions = []
        for ticker_preds in ticker_predictions.values():
            # Sort by timestamp (most recent first) and take top 5
            sorted_preds = sorted(ticker_preds, key=lambda x: x["timestamp"], reverse=True)
            cleaned_predictions.extend(sorted_preds[:5])
        
        # Update the predictions list
        self.performance_data["predictions"] = cleaned_predictions
    
    def update_outcomes(self):
        """
        Update prediction outcomes by checking current prices.
        This should be called periodically to track performance.
        """
        updated = False
        
        for prediction in self.performance_data["predictions"]:
            if prediction["outcome_determined"]:
                continue
                
            # Check if enough time has passed (6 months minimum)
            days_since_prediction = (pd.Timestamp.now() - prediction["timestamp"]).days
            
            if days_since_prediction < 180:  # 6 months
                continue
                
            try:
                # Get current price
                import yfinance as yf
                ticker_data = yf.Ticker(prediction["ticker"]).history(period="1d")
                
                if not ticker_data.empty and "Close" in ticker_data.columns and len(ticker_data) > 0:
                    current_price = ticker_data["Close"].iloc[-1]
                    actual_return = (current_price - prediction["entry_price"]) / prediction["entry_price"] * 100
                    
                    prediction["outcome_price"] = current_price
                    prediction["outcome_date"] = pd.Timestamp.now()
                    prediction["actual_return"] = actual_return
                    prediction["outcome_determined"] = True
                    updated = True
                    
            except Exception as e:
                logger.debug(f"Could not update outcome for {prediction['ticker']}: {e}")
        
        if updated:
            self._update_metrics()
            self._save_performance_data()
    
    def _update_metrics(self):
        """Calculate and update performance metrics."""
        predictions = self.performance_data["predictions"]
        
        if not predictions:
            return
            
        # Separate by prediction type
        breakout_preds = [p for p in predictions if p["prediction"] == "BREAKOUT" and p["outcome_determined"]]
        crash_preds = [p for p in predictions if p["prediction"] == "CRASH" and p["outcome_determined"]]
        
        # Calculate win rates and returns
        breakout_wins = sum(1 for p in breakout_preds if p["actual_return"] > 50)  # >50% gain = win
        crash_wins = sum(1 for p in crash_preds if p["actual_return"] < -30)     # >30% loss = win
        
        breakout_win_rate = (breakout_wins / len(breakout_preds) * 100) if breakout_preds else 0.0
        crash_win_rate = (crash_wins / len(crash_preds) * 100) if crash_preds else 0.0
        
        avg_breakout_return = np.mean([p["actual_return"] for p in breakout_preds]) if breakout_preds else 0.0
        avg_crash_return = np.mean([p["actual_return"] for p in crash_preds]) if crash_preds else 0.0
        
        self.performance_data["metrics"] = {
            "breakout_win_rate": breakout_win_rate,
            "crash_win_rate": crash_win_rate,
            "avg_breakout_return": avg_breakout_return,
            "avg_crash_return": avg_crash_return,
            "total_predictions": len(predictions),
            "breakout_predictions": len(breakout_preds),
            "crash_predictions": len(crash_preds),
        }
        
        self.performance_data["last_updated"] = pd.Timestamp.now()
    
    def get_performance_summary(self, prediction_type: str = None) -> Dict[str, Any]:
        """
        Get performance summary for predictions.
        
        Args:
            prediction_type: "BREAKOUT", "CRASH", or None for overall
            
        Returns:
            Dictionary with performance metrics
        """
        self.update_outcomes()  # Ensure outcomes are up to date
        
        if prediction_type == "BREAKOUT":
            return {
                "win_rate": self.performance_data["metrics"]["breakout_win_rate"],
                "expected_return": self.performance_data["metrics"]["avg_breakout_return"],
                "sample_size": self.performance_data["metrics"]["breakout_predictions"],
                "description": f"Breakout predictions have been correct {self.performance_data['metrics']['breakout_win_rate']:.1f}% of the time, with average returns of {self.performance_data['metrics']['avg_breakout_return']:.1f}% (based on {self.performance_data['metrics']['breakout_predictions']} predictions)"
            }
        elif prediction_type == "CRASH":
            return {
                "win_rate": self.performance_data["metrics"]["crash_win_rate"],
                "expected_return": self.performance_data["metrics"]["avg_crash_return"],
                "sample_size": self.performance_data["metrics"]["crash_predictions"],
                "description": f"Crash predictions have been correct {self.performance_data['metrics']['crash_win_rate']:.1f}% of the time, with average returns of {self.performance_data['metrics']['avg_crash_return']:.1f}% (based on {self.performance_data['metrics']['crash_predictions']} predictions)"
            }
        else:
            return {
                "total_predictions": self.performance_data["metrics"]["total_predictions"],
                "breakout_performance": self.get_performance_summary("BREAKOUT"),
                "crash_performance": self.get_performance_summary("CRASH"),
            }


# Global performance tracker instance
performance_tracker = PredictionPerformanceTracker()

# Model hyperparameters
MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "subsample": 0.8,
    "random_state": 42,
}

# Default feature values
DEFAULT_FEATURES = {
    "rsi": 50.0,
    "bb_position_pct": 50.0,
    "bb_width_pct": 10.0,
    "atr_14": 0.0,
    "obv": 0.0,
    "stoch_k": 50.0,
    "stoch_d": 50.0,
    "adx": 25.0,
    "cci": 0.0,
    "mfi": 50.0,
    "williams_r": -50.0,
    "roc": 0.0,
    "vol_roc": 0.0,
    "volume_bias": 0.0,
    "change_pct": 0.0,
    "change_5d": 0.0,
    "change_1m": 0.0,
    "pe_ratio": 25.0,
    "pe_rel_to_5y": 1.0,
    "pe_volatility_1y": 0.0,
    "market_cap": 1e9,
    "eps": 5.0,
    "put_call_ratio": 1.0,
    "short_interest": 0.0,
    "trend_score": 0.0,
}


def load_or_fetch_stock_data(ticker: str, force_refresh: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Smart data loading: Use cached data when available, only fetch new data.
    Returns up to 2 years of historical data (yfinance history DataFrame).

    If yfinance is not available, returns an empty DataFrame.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    cache_file = DATA_CACHE_DIR / f"{ticker}.pkl"

    if cache_file.exists() and not force_refresh:
        try:
            cached_data = pd.read_pickle(cache_file)
            last_date = cached_data.index.max()
            
            # Ensure consistent timezone handling
            current_time = pd.Timestamp.now(tz='UTC')  # Explicitly UTC
            if last_date.tz is None:
                # Make last_date tz-aware to match current_time
                last_date = last_date.tz_localize('UTC')
            elif last_date.tz != current_time.tz:
                # Convert both to UTC
                last_date = last_date.tz_convert('UTC')
                
            days_since_update = (current_time - last_date).days

            if days_since_update < 7:
                if verbose:
                    logger.info(f"📂 Using cached data for {ticker} (last updated: {last_date.date()})")
                return cached_data

            # Fetch only new data since last cache
            if not YF_AVAILABLE:
                if verbose:
                    logger.warning(f"⚠️  yfinance not available; returning cached data for {ticker}")
                return cached_data

            if verbose:
                logger.info(f"🔄 Updating {ticker} data (last: {last_date.date()})...")
            try:
                new_data = yf.Ticker(ticker).history(start=last_date + pd.Timedelta(days=1))
                if new_data is not None and hasattr(new_data, 'empty') and not new_data.empty:
                    # Ensure timezone consistency for concatenation
                    if cached_data.index.tz != new_data.index.tz:
                        if cached_data.index.tz is None:
                            cached_data.index = cached_data.index.tz_localize('UTC')
                        if new_data.index.tz is None:
                            new_data.index = new_data.index.tz_localize('UTC')
                        elif new_data.index.tz != cached_data.index.tz:
                            new_data.index = new_data.index.tz_convert(cached_data.index.tz)
                            
                    combined_data = pd.concat([cached_data, new_data])
                    cutoff_date = current_time - pd.DateOffset(years=2)
                    combined_data = combined_data[combined_data.index >= cutoff_date]
                    combined_data.to_pickle(cache_file)
                    if verbose:
                        logger.info(f"💾 Updated cache for {ticker} ({len(new_data)} new days)")
                    return combined_data
                else:
                    if verbose:
                        logger.info(f"📂 No new data for {ticker}, using cache")
                    return cached_data
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️  Failed to update {ticker}: {e}, using cache")
                return cached_data

        except Exception as e:
            if verbose:
                logger.warning(f"⚠️  Failed to load cache for {ticker}: {e}, fetching fresh data")

    # Fetch fresh data
    if not YF_AVAILABLE:
        if verbose:
            logger.error("❌ yfinance not installed; cannot fetch fresh data.")
        return pd.DataFrame()

    if verbose:
        logger.info(f"📥 Fetching fresh data for {ticker}...")

    try:
        stock_data = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
        # Ensure we got a valid DataFrame
        if stock_data is None:
            raise ValueError("yfinance returned None")
        if hasattr(stock_data, 'empty') and not stock_data.empty:
            stock_data.to_pickle(cache_file)
            if verbose:
                logger.info(f"💾 Cached data for {ticker} ({len(stock_data)} days)")
        return stock_data if stock_data is not None else pd.DataFrame()
    except Exception as e:
        if verbose:
            logger.error(f"❌ Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()


def extract_features(stock_data: Dict[str, Any]) -> List[float]:
    """
    Extract ML features from a stock_data dictionary.

    Returns a list of numerical features in the consistent order of FEATURE_NAMES.
    Uses DEFAULT_FEATURES for missing values and handles special encoding cases.
    """
    features: List[float] = []

    # Extract features in exact FEATURE_NAMES order
    for feature_name in FEATURE_NAMES:
        if feature_name in DEFAULT_FEATURES:
            # Standard numerical features
            features.append(float(stock_data.get(feature_name, DEFAULT_FEATURES[feature_name])))
        else:
            # Special encoded features
            if feature_name == "macd_signal":
                macd_label = stock_data.get("macd_label", None)
                features.append(1.0 if macd_label == "Bullish" else (-1.0 if macd_label == "Bearish" else 0.0))
            elif feature_name == "volume_spike":
                features.append(1.0 if stock_data.get("volume_spike", False) else 0.0)
            elif feature_name == "ma_cross":
                features.append(1.0 if stock_data.get("golden_cross", False) else (-1.0 if stock_data.get("death_cross", False) else 0.0))
            elif feature_name == "squeeze_score":
                squeeze_level = stock_data.get("squeeze_level", "None")
                squeeze_score = {"None": 0, "Moderate": 1, "High": 2, "Extreme": 3}.get(squeeze_level, 0)
                features.append(float(squeeze_score))
            elif feature_name == "trading_signal":
                active_signal = stock_data.get("active_signal", "HOLD")
                signal_map = {"BUY": 1.0, "SHORT": -1.0, "SELL": -0.5, "HOLD": 0.0}
                features.append(float(signal_map.get(active_signal, 0.0)))
            else:
                # Fallback for any missing features
                features.append(0.0)

    # Sanity check: feature length must match FEATURE_NAMES
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"extract_features produced {len(features)} features but FEATURE_NAMES has {len(FEATURE_NAMES)} entries")

    return features


FEATURE_NAMES = [
    "rsi",
    "bb_position_pct",
    "bb_width_pct",
    "macd_signal",
    "atr_14",
    "obv",
    "stoch_k",
    "stoch_d",
    "adx",
    "cci",
    "mfi",
    "williams_r",
    "roc",
    "vol_roc",
    "volume_bias",
    "volume_spike",
    "change_pct",
    "change_5d",
    "change_1m",
    "ma_cross",
    "pe_ratio",
    "pe_rel_to_5y",
    "pe_volatility_1y",
    "market_cap",
    "eps",
    "put_call_ratio",
    "short_interest",
    "squeeze_score",
    "trading_signal",
    "trend_score",
]


def train_model(historical_data: List[Dict[str, Any]], labels: List[str], verbose: bool = False) -> Tuple[Any, Any]:
    """
    Train ML model on historical stock data.

    Args:
        historical_data: List of stock data dictionaries
        labels: List of labels ('BREAKOUT', 'CRASH', 'NEUTRAL')
        verbose: When True, print detailed training metrics

    Returns:
        Trained model and scaler
    """
    if not ML_AVAILABLE:
        raise ImportError("ML libraries not installed. Install scikit-learn and related packages.")

    # Extract features
    X = []
    y = []

    for stock, label in zip(historical_data, labels):
        features = extract_features(stock)
        X.append(features)
        y.append(label)

    X = np.array(X, dtype=float)
    y = np.array(y)

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data with stratify fallback if class counts are too small
    counts = Counter(y)
    do_stratify = len(counts) > 1 and all(c >= 2 for c in counts.values())
    stratify_arg = y if do_stratify else None

    if verbose:
        logger.info(f"Training samples: {len(X_scaled)}, class distribution: {dict(counts)}")
        logger.info(f"Using stratify={do_stratify}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(**MODEL_CONFIG, verbose=0)

    logger.info(f"Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Validation results
    try:
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        logger.info(f"✅ Model trained with {accuracy:.1%} accuracy on test set")
    except Exception:
        y_pred = []
        logger.warning("Could not compute predictions on test set for reporting")

    if verbose and len(y_pred) > 0:
        logger.info("\n📊 Model Performance:")
        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_test, y_pred)))

    # Feature importance
    try:
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        if verbose:
            logger.info("\n🔝 Top 10 Most Important Features:")
            logger.info("\n" + feature_imp.head(10).to_string(index=False))
        else:
            top_features = feature_imp.head(3)["feature"].tolist()
            logger.info(f"🔝 Top features: {', '.join(top_features)}")
    except Exception:
        logger.debug("Model does not expose feature_importances_")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"✅ Model saved to {MODEL_FILE}")
    except Exception as e:
        logger.error(f"⚠️  Failed to save model: {e}")

    return model, scaler


def load_model() -> Tuple[Optional[Any], Optional[Any]]:
    """Load trained model and scaler from disk. Returns (model, scaler) or (None, None)."""
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        logger.debug("Model or scaler files not found on disk.")
        return None, None

    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        logger.warning(f"⚠️  Error loading model: {e}")
        return None, None


def predict_breakout_crash(
    stock_data: Dict[str, Any], model: Optional[Any] = None, scaler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Predict breakout/crash probability for a single stock.

    Args:
        stock_data: Dictionary with stock metrics
        model: Trained model (loads from disk if None)
        scaler: Feature scaler (loads from disk if None)

    Returns:
        Dictionary with:
            - breakout_score: 0-100 (probability of breakout)
            - crash_risk: 0-100 (probability of crash)
            - prediction: label string ("BREAKOUT", "CRASH", or "NEUTRAL")
            - confidence: 0-100 (model confidence)
            - historical_win_rate: 0-100 (historical accuracy for this prediction type)
            - expected_return: float (average historical return for this prediction type)
            - sample_size: int (number of historical predictions for this type)
    """
    if not ML_AVAILABLE:
        logger.warning("ML libraries not available - returning default neutral prediction")
        return {"breakout_score": 0, "crash_risk": 0, "prediction": "NEUTRAL", "confidence": 0}

    # Load model if not provided
    if model is None or scaler is None:
        model, scaler = load_model()
        if model is None or scaler is None:
            logger.warning("Model or scaler not available on disk - returning default neutral prediction")
            return {"breakout_score": 0, "crash_risk": 0, "prediction": "NEUTRAL", "confidence": 0}

    # Extract features
    features = extract_features(stock_data)
    
    # Check if scaler matches feature count, retrain if not
    if scaler.n_features_in_ != len(features):
        logger.info(f"Feature count mismatch: scaler expects {scaler.n_features_in_}, got {len(features)}. Retraining model...")
        try:
            # Generate mock training data
            mock_data, mock_labels = generate_synthetic_training_data(n_samples=1000)
            model, scaler = train_model(mock_data, mock_labels, verbose=False)
            # Save the new model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(model, f)
            with open(SCALER_FILE, "wb") as f:
                pickle.dump(scaler, f)
            logger.info("Model retrained and saved successfully")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            return {"breakout_score": 0, "crash_risk": 0, "prediction": "NEUTRAL", "confidence": 0}

    X = np.array([features], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure scaler can transform
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        logger.error(f"Failed to scale features: {e}")
        return {"breakout_score": 0, "crash_risk": 0, "prediction": "NEUTRAL", "confidence": 0}

    # Get prediction and probabilities (safe handling)
    try:
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(X_scaled)[0]
            classes = list(model.classes_)
            proba_map = {cls: float(proba_arr[i]) for i, cls in enumerate(classes)}
            breakout_score = int(proba_map.get("BREAKOUT", 0.0) * 100)
            crash_risk = int(proba_map.get("CRASH", 0.0) * 100)
            confidence = int(max(proba_arr) * 100)
        else:
            # Fallback if model does not implement predict_proba
            pred = model.predict(X_scaled)[0]
            breakout_score = 100 if pred == "BREAKOUT" else 0
            crash_risk = 100 if pred == "CRASH" else 0
            confidence = 100
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"breakout_score": 0, "crash_risk": 0, "prediction": "NEUTRAL", "confidence": 0}

    try:
        prediction = model.predict(X_scaled)[0]
    except Exception:
        prediction = "NEUTRAL"

    # Record prediction in performance tracker
    ticker = stock_data.get("ticker", "UNKNOWN")
    current_price = stock_data.get("close", stock_data.get("current_price", 0.0))
    
    try:
        performance_tracker.record_prediction(
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            breakout_score=breakout_score,
            crash_risk=crash_risk,
            current_price=current_price
        )
    except Exception as e:
        logger.debug(f"Failed to record prediction for {ticker}: {e}")

    # Get historical performance metrics
    try:
        if prediction == "BREAKOUT":
            perf_summary = performance_tracker.get_performance_summary("BREAKOUT")
            historical_win_rate = perf_summary["win_rate"]
            expected_return = perf_summary["expected_return"]
            sample_size = perf_summary["sample_size"]
        elif prediction == "CRASH":
            perf_summary = performance_tracker.get_performance_summary("CRASH")
            historical_win_rate = perf_summary["win_rate"]
            expected_return = perf_summary["expected_return"]
            sample_size = perf_summary["sample_size"]
        else:
            historical_win_rate = 0.0
            expected_return = 0.0
            sample_size = 0
    except Exception as e:
        logger.debug(f"Failed to get performance summary: {e}")
        historical_win_rate = 0.0
        expected_return = 0.0
        sample_size = 0

    return {
        "breakout_score": breakout_score,
        "crash_risk": crash_risk,
        "prediction": prediction,
        "confidence": confidence,
        "historical_win_rate": historical_win_rate,
        "expected_return": expected_return,
        "sample_size": sample_size,
    }


def batch_predict(stock_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict for multiple stocks efficiently.

    Args:
        stock_list: List of stock data dictionaries

    Returns:
        List of prediction dictionaries
    """
    model, scaler = load_model()
    if model is None or scaler is None:
        logger.warning("Model not found; running individual predictions with no model (defaults).")
        return [predict_breakout_crash(s, None, None) for s in stock_list]

    results: List[Dict[str, Any]] = []
    for stock in stock_list:
        pred = predict_breakout_crash(stock, model, scaler)
        results.append(pred)

    return results


def get_tickers_from_file(file_path: str) -> List[str]:
    """
    Parse tickers from a file with simple sections and comma separated tickers.
    Returns a sorted list of unique tickers.
    """
    tickers = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                continue
            if line:
                tickers.update([t.strip() for t in line.split(",") if t.strip()])
    return sorted(list(tickers))


# Example training data generator (for demonstration)
def generate_realistic_training_data(n_samples: int = 1000) -> Tuple[list[dict], list[str]]:
    """
    Replaces purely random synthetic data with correlated 'market-like' data.
    Ensures that 'BREAKOUT' labels actually align with indicator patterns.
    """
    np.random.seed(42)
    data = []
    labels = []
    for _ in range(n_samples):
        market_trend = np.random.normal(0, 1)
        stock = {
            "rsi": 50 + (market_trend * 15) + np.random.normal(0, 5),
            "volume_spike": True if market_trend > 1.5 else False,
            "bb_width_pct": 5 + np.random.exponential(5),
            "change_1m": (market_trend * 10) + np.random.normal(0, 2),
            "adx": 20 + (abs(market_trend) * 10),
            "put_call_ratio": 1.0 - (market_trend * 0.2),
        }
        if stock["rsi"] > 65 and stock["volume_spike"] and stock["change_1m"] > 15:
            label = "BREAKOUT"
        elif stock["rsi"] < 35 and stock["change_1m"] < -15:
            label = "CRASH"
        else:
            label = "NEUTRAL"
        data.append(stock)
        labels.append(label)
    return data, labels
# --- NOISE REDUCTION: IMPROVED FEATURE EXTRACTION ---
def extract_features_denoised(stock_data: dict) -> list[float]:
    """
    Filters features to prioritize low-noise indicators like BB Squeeze 
    and Volume Bias over lagging MA crosses.
    """
    features = {
        "rsi_smooth": stock_data.get("rsi", 50.0),
        "vol_bias": stock_data.get("volume_bias", 0.0),
        "squeeze_score": {"None": 0, "Moderate": 1, "High": 2, "Extreme": 3}.get(
            stock_data.get("squeeze_level", "None"), 0
        ),
        "adx_dynamic": stock_data.get("adx", 20.0),
    }
    return list(features.values())


def run_basic_tests() -> bool:
    """
    Run basic unit tests for core functionality.
    Returns True if all tests pass, False otherwise.
    """
    logger.info("🧪 Running basic tests...")

    # Test feature extraction
    test_stock = {
        "rsi": 65.0,
        "bb_position_pct": 75.0,
        "bb_width_pct": 15.0,
        "macd_label": "Bullish",
        "atr_14": 2.5,
        "volume_spike": True,
        "golden_cross": True,
        "pe_ratio": 20.0,
        "squeeze_level": "High",
        "active_signal": "BUY",
        "trend_score": 2.0,
    }

    try:
        features = extract_features(test_stock)
        assert len(features) == len(FEATURE_NAMES), f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}"
        assert features[0] == 65.0, "RSI feature extraction failed"
        assert features[3] == 1.0, "MACD bullish encoding failed"  # macd_signal is at index 3
        logger.info("✅ Feature extraction tests passed")
    except Exception as e:
        logger.error(f"❌ Feature extraction test failed: {e}")
        return False

    # Test prediction with defaults (should return neutral)
    try:
        result = predict_breakout_crash({})
        assert result["prediction"] == "NEUTRAL", "Default prediction failed"
        assert result["breakout_score"] == 0, "Default breakout score failed"
        logger.info("✅ Default prediction tests passed")
    except Exception as e:
        logger.error(f"❌ Default prediction test failed: {e}")
        return False

    logger.info("✅ All basic tests passed!")
    return True


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ML Stock Predictor - Train breakout/crash detection models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console output")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🤖 ML Stock Predictor - Training Demo (verbose)")

    # Run basic tests first
    if not run_basic_tests():
        logger.error("❌ Basic tests failed. Exiting.")
        sys.exit(1)

    if not ML_AVAILABLE:
        logger.error("❌ Please install ML libraries: pip install scikit-learn")
        sys.exit(1)

    if not YF_AVAILABLE:
        logger.warning("⚠️ yfinance is not installed. Fetching real historical data will not be possible.")

    # --- Use real data from tickers.csv for training if available ---
    TICKER_FILE = Path("data/tickers.csv")
    if not TICKER_FILE.exists():
        logger.warning(f"❌ Ticker file not found at {TICKER_FILE}")
        logger.info("Falling back to synthetic data generation.")
        logger.info("📊 Generating synthetic training data...")
        training_data, labels = generate_synthetic_training_data(n_samples=2000)
    else:
        logger.info(f"✅ Found ticker file at {TICKER_FILE}")
        if args.verbose:
            logger.info("⚙️  Processing real historical data for training. This may take a while...")

        tickers = get_tickers_from_file(str(TICKER_FILE))
        logger.info(f"⚙️  Processing {len(tickers)} tickers...")

        training_data = []
        labels = []

        for i, ticker in enumerate(tickers):
            # Show progress every 50 tickers (or all if verbose)
            if args.verbose or (i + 1) % 50 == 0 or i == 0:
                progress_msg = f"Processing {ticker} ({i+1}/{len(tickers)})" if args.verbose else f"Processing tickers... ({i+1}/{len(tickers)})"
                logger.info(progress_msg)

            try:
                stock_hist = load_or_fetch_stock_data(ticker, verbose=args.verbose)
                
                # Validate data quality (relaxed requirements for training)
                if stock_hist.empty:
                    if args.verbose:
                        logger.warning(f"  Skipping {ticker}: No data available")
                    continue
                    
                min_required_days = 100  # Reduced from 252 to allow more tickers
                if len(stock_hist) < min_required_days:
                    if args.verbose:
                        logger.warning(f"  Skipping {ticker}: Insufficient data ({len(stock_hist)} days < {min_required_days} required)")
                    continue
                
                # Check for excessive missing values (relaxed from 10% to 20%)
                if stock_hist["Close"].isna().sum() > len(stock_hist) * 0.2:
                    if args.verbose:
                        logger.warning(f"  Skipping {ticker}: Too many missing values")
                    continue

                # Use a single-row snapshot from 1 year ago (or earliest available data)
                # Check if we have at least 252 days, otherwise use the earliest available data
                if len(stock_hist) >= 252:
                    one_year_ago_data = stock_hist.iloc[-252]
                else:
                    # For stocks with less than 252 days, use the first available data point
                    one_year_ago_data = stock_hist.iloc[0]

                # Calculate real technical indicators from historical data
                # Use the last available data for feature calculation (minimum 100 days)
                recent_data = stock_hist.tail(min(252, len(stock_hist)))  # Use up to 1 year of data
                
                # Import indicator functions from stocks.py
                try:
                    from stocks import rsi, macd
                    
                    # Calculate RSI
                    rsi_val = rsi(recent_data["Close"]) if len(recent_data) >= 14 else 50.0
                    
                    # Calculate MACD
                    macd_val, macd_sig, macd_label = macd(recent_data["Close"]) if len(recent_data) >= 26 else (0, 0, None)
                    
                    # Calculate Bollinger Bands
                    bb_position_pct = 50.0  # Default
                    bb_width_pct = 10.0     # Default
                    if len(recent_data) >= 20:
                        sma = recent_data["Close"].rolling(20).mean()
                        std = recent_data["Close"].rolling(20).std()
                        upper_bb = sma + (2 * std)
                        lower_bb = sma - (2 * std)
                        
                        # Check if we have valid BB calculations
                        if len(upper_bb.dropna()) > 0 and len(lower_bb.dropna()) > 0 and len(sma.dropna()) > 0:
                            current_price = recent_data["Close"].iloc[-1]
                            bb_position_pct = ((current_price - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])) * 100
                            bb_width_pct = ((upper_bb.iloc[-1] - lower_bb.iloc[-1]) / sma.iloc[-1]) * 100
                    
                    # Calculate Stochastic Oscillator
                    stoch_k = stoch_d = 50.0
                    if len(recent_data) >= 14:
                        low_14 = recent_data["Low"].rolling(14).min()
                        high_14 = recent_data["High"].rolling(14).max()
                        k = ((recent_data["Close"] - low_14) / (high_14 - low_14)) * 100
                        
                        # Check if we have valid stochastic calculations
                        if len(k.dropna()) > 0:
                            stoch_k = k.iloc[-1]
                            if len(k.dropna()) >= 3:
                                stoch_d = k.rolling(3).mean().iloc[-1]
                    
                    # Calculate ATR (Average True Range)
                    atr_val = 0.0
                    if len(recent_data) >= 14:
                        high = recent_data["High"]
                        low = recent_data["Low"]
                        close = recent_data["Close"]
                        tr1 = high - low
                        tr2 = (high - close.shift(1)).abs()
                        tr3 = (low - close.shift(1)).abs()
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        
                        # Check if we have valid ATR calculations
                        atr_series = tr.rolling(14).mean()
                        if len(atr_series.dropna()) > 0:
                            atr_val = atr_series.iloc[-1]
                    
                    # Calculate OBV (On-Balance Volume)
                    obv_val = 0.0
                    if len(recent_data) >= 2:
                        obv = pd.Series(index=recent_data.index, dtype=float)
                        obv.iloc[0] = recent_data["Volume"].iloc[0]
                        for i in range(1, len(recent_data)):
                            if recent_data["Close"].iloc[i] > recent_data["Close"].iloc[i-1]:
                                obv.iloc[i] = obv.iloc[i-1] + recent_data["Volume"].iloc[i]
                            elif recent_data["Close"].iloc[i] < recent_data["Close"].iloc[i-1]:
                                obv.iloc[i] = obv.iloc[i-1] - recent_data["Volume"].iloc[i]
                            else:
                                obv.iloc[i] = obv.iloc[i-1]
                        
                        # Check if OBV calculation was successful
                        if len(obv.dropna()) > 0:
                            obv_val = obv.iloc[-1]
                    
                except ImportError:
                    # Fallback to defaults if stocks.py not available
                    rsi_val = np.random.uniform(30, 70)
                    macd_label = None
                    bb_position_pct = 50.0
                    bb_width_pct = 10.0
                    stoch_k = 50.0
                    stoch_d = 50.0
                    atr_val = 0.0
                    obv_val = 0.0
                
                # Calculate price changes with proper bounds checking
                current_price = recent_data["Close"].iloc[-1] if len(recent_data) > 0 else 0
                change_pct = ((current_price - recent_data["Close"].iloc[-2]) / recent_data["Close"].iloc[-2]) * 100 if len(recent_data) >= 2 else 0
                change_5d = ((current_price - recent_data["Close"].iloc[-6]) / recent_data["Close"].iloc[-6]) * 100 if len(recent_data) >= 6 else 0
                
                # Calculate moving average crosses
                golden_cross = death_cross = False
                if len(recent_data) >= 50:
                    sma_50 = recent_data["Close"].rolling(50).mean()
                    sma_200 = recent_data["Close"].rolling(200).mean()
                    if len(sma_50) >= 2 and len(sma_200) >= 2:
                        prev_50 = sma_50.iloc[-2]
                        prev_200 = sma_200.iloc[-2]
                        curr_50 = sma_50.iloc[-1]
                        curr_200 = sma_200.iloc[-1]
                        if prev_50 <= prev_200 and curr_50 > curr_200:
                            golden_cross = True
                        elif prev_50 >= prev_200 and curr_50 < curr_200:
                            death_cross = True
                
                # Volume analysis
                volume_bias = 0.0
                volume_spike = False
                if len(recent_data) >= 20:
                    avg_volume_series = recent_data["Volume"].rolling(20).mean()
                    if len(avg_volume_series.dropna()) > 0:
                        avg_volume = avg_volume_series.iloc[-1]
                        current_volume = recent_data["Volume"].iloc[-1]
                        if avg_volume == 0 or np.isnan(avg_volume):
                            volume_bias = 0
                        else:
                            volume_bias = ((current_volume - avg_volume) / avg_volume) * 100
                        volume_spike = current_volume > avg_volume * 2  # 2x average volume
                
                # Calculate P/E relative to 5-year average and P/E volatility (1-year std)
                pe_ratio = one_year_ago_data.get("P/E Ratio", 25) if isinstance(one_year_ago_data, pd.Series) else 25
                pe_5y_avg = None
                pe_volatility_1y = None
                if isinstance(stock_hist, pd.DataFrame) and "P/E Ratio" in stock_hist.columns:
                    pe_series = stock_hist["P/E Ratio"].dropna()
                    if len(pe_series) >= 252:
                        pe_5y_avg = pe_series[-252*5:].mean() if len(pe_series) >= 252*5 else pe_series.mean()
                        pe_volatility_1y = pe_series[-252:].std()
                pe_rel_to_5y = (pe_ratio / pe_5y_avg) if pe_5y_avg and pe_5y_avg > 0 else 1.0
                pe_volatility_1y = pe_volatility_1y if pe_volatility_1y is not None else 0.0

                features = {
                    "rsi": rsi_val,
                    "bb_position_pct": bb_position_pct,
                    "bb_width_pct": bb_width_pct,
                    "macd_label": macd_label,
                    "atr": atr_val,
                    "obv": obv_val,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "adx": 25.0,  # Placeholder - would need full ADX calculation
                    "cci": 0.0,   # Placeholder
                    "mfi": 50.0,  # Placeholder
                    "williams_r": -50.0,  # Placeholder
                    "roc": 0.0,   # Placeholder
                    "volume_roc": 0.0,  # Placeholder
                    "volume_bias": volume_bias,
                    "volume_spike": volume_spike,
                    "change_pct": change_pct,
                    "change_5d": change_5d,
                    "change_1m": (current_price - recent_data["Close"].iloc[0]) / recent_data["Close"].iloc[0] * 100 if len(recent_data) >= 21 else 0,
                    "golden_cross": golden_cross,
                    "death_cross": death_cross,
                    "pe_ratio": pe_ratio,
                    "pe_rel_to_5y": pe_rel_to_5y,
                    "pe_volatility_1y": pe_volatility_1y,
                    "eps": one_year_ago_data.get("EPS", 5.0) if isinstance(one_year_ago_data, pd.Series) else 5.0,
                    "market_cap": one_year_ago_data.get("Market Cap", 1e9) if isinstance(one_year_ago_data, pd.Series) else 1e9,
                    "put_call_ratio": 1.0,  # Placeholder
                    "short_interest": 0.0,  # Placeholder
                    "squeeze_level": "None",  # Placeholder
                    "active_signal": "HOLD",  # Placeholder
                    "trend_score": 0.0,  # Placeholder
                }

                # Improved labeling logic based on multiple criteria
                try:
                    # Get price data points with bounds checking
                    initial_price = one_year_ago_data["Close"]
                    future_price_3m = stock_hist.iloc[-189]["Close"] if len(stock_hist) > 189 else initial_price  # 3 months
                    future_price_6m = stock_hist.iloc[-126]["Close"] if len(stock_hist) > 126 else initial_price  # 6 months  
                    future_price_12m = stock_hist.iloc[-1]["Close"] if len(stock_hist) > 0 else initial_price  # 12 months
                    
                    # Calculate returns
                    return_3m = (future_price_3m - initial_price) / initial_price
                    return_6m = (future_price_6m - initial_price) / initial_price
                    return_12m = (future_price_12m - initial_price) / initial_price
                    
                    # Calculate volatility (standard deviation of returns)
                    if len(recent_data) >= 20:
                        daily_returns = recent_data["Close"].pct_change().dropna()
                        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                    else:
                        volatility = 0.3  # Default assumption
                    
                    # Determine breakout/crash labels based on multiple factors
                    label = "NEUTRAL"
                    
                    # BREAKOUT conditions:
                    # 1. Significant price appreciation (50%+ in 6-12 months)
                    # 2. Consistent upward momentum
                    # 3. Not excessively volatile
                    breakout_score = 0
                    if return_6m > 0.5: breakout_score += 2  # 50% in 6 months
                    if return_12m > 1.0: breakout_score += 3  # 100% in 12 months
                    if return_3m > 0.15: breakout_score += 1  # 15% in 3 months
                    if return_6m > return_3m * 0.8: breakout_score += 1  # Consistent momentum
                    if volatility < 0.6: breakout_score += 1  # Not too volatile
                    
                    # CRASH conditions:
                    # 1. Significant price decline (30%- in 6-12 months)
                    # 2. Consistent downward momentum
                    crash_score = 0
                    if return_6m < -0.3: crash_score += 2  # 30% drop in 6 months
                    if return_12m < -0.5: crash_score += 3  # 50% drop in 12 months
                    if return_3m < -0.1: crash_score += 1  # 10% drop in 3 months
                    if return_6m < return_3m * 1.2: crash_score += 1  # Consistent decline
                    if volatility > 0.8: crash_score += 1  # High volatility (crashes are volatile)
                    
                    # Final classification with confidence thresholds
                    if breakout_score >= 4:
                        label = "BREAKOUT"
                    elif crash_score >= 4:
                        label = "CRASH"
                    # NEUTRAL if neither condition strongly met
                    
                except Exception as e:
                    if args.verbose:
                        logger.warning(f"Could not calculate labels for {ticker}: {e}")
                    continue


                # Denoised CRASH filter: Only label as CRASH if technicals and high relative/volatile P/E agree
                if label == "CRASH":
                    # Use as a filter: require high relative P/E or high P/E volatility
                    if pe_rel_to_5y < 1.2 and pe_volatility_1y < 10:  # thresholds can be tuned
                        label = "NEUTRAL"

                training_data.append(features)
                labels.append(label)

            except Exception as e:
                if args.verbose:
                    logger.exception(f"  Could not process {ticker}: {e}")

        if not training_data:
            logger.error("❌ No training data could be generated from the tickers. Exiting.")
            sys.exit(1)

    # Train model
    logger.info("\n🏋️  Training model...")
    model, scaler = train_model(training_data, labels, verbose=args.verbose)

    # Test prediction
    logger.info("\n🔮 Testing prediction on sample stock...")
    test_stock = {
        "rsi": 25,
        "bb_position_pct": 5,
        "bb_width_pct": 3,
        "macd_label": "Bullish",
        "atr_14": 2.5,
        "volume_bias": 0.6,
        "volume_spike": True,
        "change_pct": 3.2,
        "change_5d": 8.5,
        "change_1m": 15.0,
        "golden_cross": True,
        "death_cross": False,
        "pe_ratio": 25,
        "market_cap": 50e9,
        "put_call_ratio": 0.8,
        "short_interest": 15,
        "squeeze_level": "High",
        "active_signal": "BUY",
        "trend_score": 3.5,
    }

    result = predict_breakout_crash(test_stock, model, scaler)
    if args.verbose:
        logger.info(f"\n📈 Prediction Results:")
        logger.info(f"   Breakout Score: {result['breakout_score']}%")
        logger.info(f"   Crash Risk: {result['crash_risk']}%")
        logger.info(f"   Prediction: {result['prediction']}")
        logger.info(f"   Confidence: {result['confidence']}%")
        logger.info("\n✅ Model ready for production use!")
    else:
        logger.info(f"✅ Model ready! Sample prediction: {result['prediction']} ({result['breakout_score']}% breakout, {result['crash_risk']}% crash risk)")