import os
import sys
import unittest
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils import is_nse_holiday, get_nse_trading_days
from app.models import (
    TFTAttentionRegressor, 
    RobustRidgeRegressor, 
    RobustGBRegressor, 
    HoltWintersRegressor,
    EnsembleForecaster
)
from app.risk import run_conformal_forecasting, WalkForwardBacktester

class TestQuantitativeEngineRefinement(unittest.TestCase):
    """
    Validation test suite for the Project Apex quantitative engine changes.
    """
    
    def setUp(self):
        # Generate dummy dataset covering late 2026 to mid 2027
        # to test holiday handling and future years.
        np.random.seed(42)
        self.start_date = datetime(2026, 10, 1)
        self.end_date = datetime(2027, 5, 1)
        
        # Build consecutive calendar days
        delta = self.end_date - self.start_date
        all_dates = [self.start_date + timedelta(days=i) for i in range(delta.days)]
        
        # Filter for trading days only
        trading_dates = [d for d in all_dates if not is_nse_holiday(d)]
        self.trading_dates = trading_dates
        self.n_days = len(trading_dates)
        
        # Simulate price series (Random Walk with Drift)
        prices = [100.0]
        for _ in range(self.n_days - 1):
            ret = np.random.normal(0.0005, 0.015)
            prices.append(prices[-1] * np.exp(ret))
            
        self.prices = np.array(prices)
        self.volatility = np.array([np.std(self.prices[max(0, i-10):i+1]) for i in range(self.n_days)])
        self.sentiment = np.random.uniform(-1, 1, self.n_days)
        self.pcr = np.random.uniform(0.5, 1.5, self.n_days)
        self.rsi = np.random.uniform(20, 80, self.n_days)
        self.macd = np.random.normal(0, 5, self.n_days)
        self.log_returns = np.log(self.prices / pd.Series(self.prices).shift(1)).fillna(0.0).values
        
        # Feature Matrix X (close, vol, sentiment, pcr, rsi, macd, returns)
        self.X = np.column_stack([
            self.prices,
            self.volatility,
            self.sentiment,
            self.pcr,
            self.rsi,
            self.macd,
            self.log_returns
        ])
        
        # Target: 5-step ahead prediction
        self.h = 5
        self.y = np.roll(self.prices, -self.h)
        self.y[-self.h:] = self.prices[-1]
        
    def test_market_holidays_2026_2027_dynamic(self):
        """
        1. Test holiday detection across known holidays and future fallbacks.
        """
        # 2026 holiday (Ganesh Chaturthi Sept 14, 2026)
        self.assertTrue(is_nse_holiday(date(2026, 9, 14)))
        # 2026 Weekend
        self.assertTrue(is_nse_holiday(date(2026, 9, 19))) # Saturday
        
        # 2027 holiday (Ramzan Id March 10, 2027)
        self.assertTrue(is_nse_holiday(date(2027, 3, 10)))
        # 2027 Weekend
        self.assertTrue(is_nse_holiday(date(2027, 3, 13))) # Saturday
        
        # 2028 Dynamic Fallback (Republic Day Jan 26, 2028)
        self.assertTrue(is_nse_holiday(date(2028, 1, 26)))
        # 2028 Dynamic Fallback (Independence Day Aug 15, 2028)
        self.assertTrue(is_nse_holiday(date(2028, 8, 15)))
        # 2028 normal trading day (Wednesday Feb 2, 2028)
        self.assertFalse(is_nse_holiday(date(2028, 2, 2)))
        
        # Trading days range builder
        trading_days = get_nse_trading_days("2027-01-01", "2027-01-10")
        # Jan 1 (Fri), Jan 4 (Mon) to Jan 8 (Fri). Jan 2, 3, 9, 10 are weekends.
        # Check if length is exactly 6
        self.assertEqual(len(trading_days), 6)
        
    def test_model_architectures_and_scaling(self):
        """
        2. Test all individual architectures to verify target scaling, sequence builders, and prediction shapes.
        """
        seq_len = 15
        
        # Split train/test
        split = int(self.n_days * 0.8)
        X_train, y_train = self.X[:split], self.y[:split]
        X_test = self.X[split:]
        
        # A. TFT Attention Regressor
        tft = TFTAttentionRegressor(seq_len=seq_len, epochs=2)
        tft.fit(X_train, y_train)
        pred_tft = tft.predict(X_test)
        self.assertEqual(len(pred_tft), len(X_test))
        # Ensure no NaNs or Inf values
        self.assertFalse(np.isnan(pred_tft).any())
        self.assertFalse(np.isinf(pred_tft).any())
        
        # B. Robust Ridge Regressor
        ridge = RobustRidgeRegressor(seq_len=seq_len)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        self.assertEqual(len(pred_ridge), len(X_test))
        self.assertFalse(np.isnan(pred_ridge).any())
        
        # C. Robust Gradient Boosting Regressor
        gbr = RobustGBRegressor(seq_len=seq_len, n_estimators=10, max_depth=3)
        gbr.fit(X_train, y_train)
        pred_gbr = gbr.predict(X_test)
        self.assertEqual(len(pred_gbr), len(X_test))
        self.assertFalse(np.isnan(pred_gbr).any())
        
        # D. Holt-Winters Regressor
        hw = HoltWintersRegressor(seasonal_periods=5)
        hw.fit(X_train, y_train)
        pred_hw = hw.predict(X_test)
        self.assertEqual(len(pred_hw), len(X_test))
        self.assertFalse(np.isnan(pred_hw).any())

    def test_ensemble_forecaster(self):
        """
        3. Test EnsembleForecaster for dynamic inverse-MAPE weighting.
        """
        ensemble = EnsembleForecaster(seq_len=15)
        
        # Fit on dataset
        ensemble.fit(self.X, self.y)
        
        # Verify dynamic weights sum to approximately 1.0
        total_weight = sum(ensemble.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        self.assertIn("gbr", ensemble.weights)
        self.assertIn("hw", ensemble.weights)
        
        # Predict
        preds_dict = ensemble.predict(self.X[-20:])
        self.assertEqual(len(preds_dict["ensemble"]), 20)
        self.assertEqual(len(preds_dict["tft"]), 20)
        self.assertEqual(len(preds_dict["gbr"]), 20)
        self.assertEqual(len(preds_dict["hw"]), 20)
        self.assertFalse(np.isnan(preds_dict["ensemble"]).any())

    def test_conformal_uncertainty_bounds(self):
        """
        4. Test conformal calibration bounds and horizon alignment.
        """
        split = int(self.n_days * 0.8)
        train_features = self.X[:split]
        train_targets = self.y[:split]
        test_features = self.X[split:]
        
        y_pred, bounds_90, bounds_95 = run_conformal_forecasting(
            train_features, train_targets, test_features, horizon_steps=self.h
        )
        
        self.assertEqual(len(y_pred), len(test_features))
        self.assertEqual(bounds_90.shape, (len(test_features), 2))
        self.assertEqual(bounds_95.shape, (len(test_features), 2))
        
        # Ensure confidence interval lower bounds are less than or equal to upper bounds
        self.assertTrue((bounds_90[:, 0] <= bounds_90[:, 1]).all())
        self.assertTrue((bounds_95[:, 0] <= bounds_95[:, 1]).all())
        
    def test_walk_forward_backtester(self):
        """
        5. Test the out-of-sample backtester with HMM regime filters and stop-loss logic.
        """
        # Create a DataFrame containing all required backtest columns
        df = pd.DataFrame({
            'close_raw': self.prices,
            'rolling_volatility': self.volatility,
            'sentiment_score': self.sentiment,
            'pcr_oi': self.pcr,
            'rsi': self.rsi,
            'macd': self.macd,
            'log_returns': self.log_returns
        })
        
        # Inject mock regimes
        df['regime'] = np.random.choice([0, 1], size=self.n_days, p=[0.7, 0.3])
        
        backtester = WalkForwardBacktester()
        metrics = backtester.evaluate_strategy(df, horizon_steps=self.h, min_train_days=50)
        
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("directional_accuracy", metrics)
        self.assertIn("total_strategy_return", metrics)
        
        # Metrics should be real floats
        self.assertTrue(isinstance(metrics["sharpe_ratio"], float))
        self.assertTrue(isinstance(metrics["max_drawdown"], float))
        self.assertTrue(isinstance(metrics["mape"], float))
        
if __name__ == '__main__':
    unittest.main()
