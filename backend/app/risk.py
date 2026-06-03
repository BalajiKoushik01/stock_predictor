import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from app.models import TFTAttentionRegressor

def run_conformal_forecasting(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    horizon_steps: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits Temporal Attention model and computes calibrated conformal prediction intervals
    using a rolling residuals split-conformal approach.
    Uncertainty bands are calibrated directly for the forecast horizon.
    """
    base_model = TFTAttentionRegressor(epochs=30, batch_size=16, lr=0.005)

    try:
        n = len(train_features)
        seq_len = base_model.seq_len

        # Need enough data: train on first 80%, calibrate on last 20%
        if n < seq_len * 3:
            raise ValueError(f"Not enough training rows ({n}) for conformal split (need {seq_len * 3}).")

        split = max(seq_len + 10, int(n * 0.80))
        fit_features  = train_features[:split]
        calib_features = train_features[split:]
        calib_targets  = train_targets[split:]

        # Fit on training portion
        base_model.fit(fit_features, train_targets[:split])

        # Get calibration residuals
        calib_preds = base_model.predict(calib_features)
        # Ensure same length as calib_targets
        min_len = min(len(calib_preds), len(calib_targets))
        residuals = np.abs(calib_targets[-min_len:] - calib_preds[-min_len:])

        # Empirical quantile conformal scores
        q90 = float(np.quantile(residuals, 0.90)) if len(residuals) > 0 else float(np.std(calib_targets) * 1.645)
        q95 = float(np.quantile(residuals, 0.95)) if len(residuals) > 0 else float(np.std(calib_targets) * 1.960)

        # Predict on test set
        y_pred = base_model.predict(test_features)
        # Trim / pad to correct length
        n_test = len(test_features)
        if len(y_pred) > n_test:
            y_pred = y_pred[-n_test:]
        elif len(y_pred) < n_test:
            y_pred = np.pad(y_pred, (n_test - len(y_pred), 0), mode='edge')

        # Calibrated conformal bands directly using target-scale quantile scores
        lower_90 = y_pred - q90
        upper_90 = y_pred + q90
        lower_95 = y_pred - q95
        upper_95 = y_pred + q95

        return y_pred, np.column_stack([lower_90, upper_90]), np.column_stack([lower_95, upper_95])

    except Exception as e:
        print(f"[WARNING] Conformal regression failed: {e}. Falling back to parametric residuals calibration.")

        try:
            base_model.fit(train_features, train_targets)
            y_pred = base_model.predict(test_features)
        except Exception as e2:
            print(f"[WARNING] Base model fit also failed: {e2}. Using last known price.")
            last_price = float(train_targets[-1]) if len(train_targets) > 0 else 100.0
            y_pred = np.full(len(test_features), last_price)

        # Parametric bounds from training residuals
        try:
            train_preds = base_model.predict(train_features)
            min_len = min(len(train_preds), len(train_targets))
            residuals = train_targets[-min_len:] - train_preds[-min_len:]
            std_error = float(np.std(residuals)) if len(residuals) > 0 else float(np.std(train_targets) * 0.05)
        except Exception:
            std_error = float(np.std(train_targets) * 0.05) if len(train_targets) > 0 else 1.0

        lower_90 = y_pred - 1.645 * std_error
        upper_90 = y_pred + 1.645 * std_error
        lower_95 = y_pred - 1.960 * std_error
        upper_95 = y_pred + 1.960 * std_error

        return y_pred, np.column_stack([lower_90, upper_90]), np.column_stack([lower_95, upper_95])


class WalkForwardBacktester:
    """
    Simulates out-of-sample execution of Project Apex's forecasting strategies.
    Computes institutional performance metrics including Sharpe, Drawdown, and accuracy.
    """
    def __init__(self, risk_free_rate: float = 0.06):
        # Default 6% Indian RBI risk free rate
        self.daily_rf = risk_free_rate / 252.0

    def evaluate_strategy(
        self, 
        aligned_df: pd.DataFrame, 
        horizon_steps: int = 10,
        min_train_days: int = 120
    ) -> Dict[str, Any]:
        """
        Runs walk-forward rolling out-of-sample backtesting on the aligned DataFrame.
        Uses Ensemble forecasting, HMM regime filters, and Conformal stop-losses.
        """
        N = len(aligned_df)
        if N < min_train_days + horizon_steps:
            raise ValueError(f"Insufficient history ({N} rows) to backtest with training size {min_train_days}.")

        close_prices = aligned_df['close_raw'].values
        volatility = aligned_df['rolling_volatility'].values
        sentiment = aligned_df['sentiment_score'].values
        pcr = aligned_df['pcr_oi'].values
        rsi = aligned_df['rsi'].values if 'rsi' in aligned_df.columns else np.full(N, 50.0)
        macd = aligned_df['macd'].values if 'macd' in aligned_df.columns else np.zeros(N)
        log_returns = aligned_df['log_returns'].values
        regimes = aligned_df['regime'].values if 'regime' in aligned_df.columns else np.zeros(N)

        # Build basic 2D feature matrix X:
        X = np.column_stack([
            close_prices,
            volatility,
            sentiment,
            pcr,
            rsi,
            macd,
            log_returns
        ])
        
        # Target: Forward price levels at the horizon step
        y = np.roll(close_prices, -horizon_steps)
        # Clean target bounds
        y[-horizon_steps:] = close_prices[-1]

        # Containers for strategy outcomes
        signals = np.zeros(N)
        forecasts = np.zeros(N)
        
        # Track accuracy
        directional_matches = 0
        prediction_count = 0

        # We step through the backtest window out-of-sample
        step_size = 10
        print(f"Executing Walk-Forward Backtester across {N - min_train_days} sessions...")
        
        from app.models import EnsembleForecaster

        for i in range(min_train_days, N - horizon_steps, step_size):
            # Dynamic rolling split
            train_features = X[:i]
            train_targets = y[:i]
            
            test_features = X[i : i + step_size]
            test_prices = close_prices[i : i + step_size]
            actual_targets = y[i : i + step_size]
            test_regimes = regimes[i : i + step_size]

            # Fit Ensemble Forecaster
            ensemble = EnsembleForecaster(seq_len=15)
            try:
                ensemble.fit(train_features, train_targets)
                
                # Predict on the test segment passing preceding lookback context
                lookback_test_features = X[i - ensemble.seq_len : i + step_size]
                preds_dict = ensemble.predict(lookback_test_features)
                preds = preds_dict["ensemble"]
                
                test_preds = preds[-step_size:]
                
                # Compute training residuals for conformal stop-loss
                train_preds = ensemble.predict(train_features)["ensemble"]
                min_len = min(len(train_preds), len(train_targets))
                residuals = np.abs(train_targets[-min_len:] - train_preds[-min_len:])
                q95 = float(np.quantile(residuals, 0.95)) if len(residuals) > 0 else float(np.std(train_targets) * 1.96)

                for t_idx in range(len(test_preds)):
                    curr_idx = i + t_idx
                    pred_price = test_preds[t_idx]
                    regime = test_regimes[t_idx]
                    
                    forecasts[curr_idx] = pred_price
                    
                    # Compute directional accuracy
                    pred_up = pred_price > close_prices[curr_idx]
                    actual_up = actual_targets[t_idx] > close_prices[curr_idx]
                    
                    if pred_up == actual_up:
                        directional_matches += 1
                    prediction_count += 1

                    # Trading Signal Rules:
                    # 1. Bullish signal: forecast price is at least 1% higher than current price
                    is_bullish_forecast = pred_price > close_prices[curr_idx] * 1.01
                    
                    # 2. HMM Regime Filter: Bull regime (0) is safe, Bear regime (1) is high-risk
                    is_bull_regime = (regime == 0)
                    
                    # 3. Conformal Stop-Loss: If the current price drops below the 95% conformal lower bound, stay out/exit
                    lower_bound_95 = pred_price - q95
                    is_stop_loss_triggered = close_prices[curr_idx] < lower_bound_95
                    
                    if is_bullish_forecast and is_bull_regime and not is_stop_loss_triggered:
                        signals[curr_idx] = 1.0
                    else:
                        signals[curr_idx] = 0.0
            except Exception as e:
                # Catch training failures gracefully
                continue

        # Strategy Performance Calculation
        # Shift signals by 1 to represent execution lag
        execution_signals = np.roll(signals, 1)
        execution_signals[0] = 0.0
        
        # Calculate strategy log returns
        strategy_returns = execution_signals * log_returns
        
        # Compute Sharpe Ratio
        active_returns = strategy_returns[min_train_days:]
        excess_returns = active_returns - self.daily_rf
        
        avg_excess = np.mean(excess_returns)
        std_returns = np.std(active_returns)
        
        sharpe = (avg_excess / std_returns * np.sqrt(252)) if std_returns > 0 else 0.0

        # Compute Maximum Drawdown (MDD)
        wealth_index = np.exp(np.cumsum(active_returns))
        peaks = np.maximum.accumulate(wealth_index)
        drawdowns = (wealth_index - peaks) / peaks
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Mean Absolute Percentage Error (MAPE)
        valid_indices = (forecasts > 0)
        mape = float(np.mean(np.abs(forecasts[valid_indices] - close_prices[valid_indices]) / close_prices[valid_indices])) if np.sum(valid_indices) > 0 else 0.0
        
        directional_accuracy = float(directional_matches / prediction_count) if prediction_count > 0 else 0.0

        return {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy),
            "total_strategy_return": float(np.exp(np.sum(active_returns)) - 1.0)
        }

# Singleton instance
apex_backtester = WalkForwardBacktester()
