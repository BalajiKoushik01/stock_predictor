import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hmmlearn.hmm import GaussianHMM
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Union, List, Tuple

# Optimize CPU: Limit PyTorch CPU threads to prevent 100% host CPU lockup
try:
    torch.set_num_threads(2)
    print("[CPU OPTIMIZED] PyTorch CPU thread allocation limited to 2 to secure system stability.")
except Exception as e:
    print(f"Could not throttle PyTorch CPU threads: {e}")

class RegimeDetector:
    """
    Continuous Gaussian Hidden Markov Model (HMM) for detecting latent market regimes
    (State 0: Low Volatility/Bull, State 1: High Volatility/Bear) based on rolling returns and volatility.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=self.n_components, 
            covariance_type="diag", 
            n_iter=100, 
            random_state=42
        )
        self.is_fitted = False

    def fit(self, log_returns: pd.Series, volatility: pd.Series):
        """
        Fits the HMM on the rolling returns and volatility series.
        """
        # Create an observation matrix (N x 2)
        obs = np.column_stack([
            log_returns.fillna(0.0).values,
            volatility.fillna(0.0).values
        ])
        
        # Avoid zero variance or extreme scaling issues
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            self.model.fit(obs)
            self.is_fitted = True
            
            # Map states so State 0 is always the lower volatility regime
            # (which represents standard institutional Low Volatility / Bull trends)
            state_covs = [np.mean(self.model.covars_[i]) for i in range(self.n_components)]
            if state_covs[0] > state_covs[1]:
                # Swap components mapping
                self._swap_states()
            print("HMM Regime Detector calibrated successfully.")
        except Exception as e:
            print(f"HMM calibration failed: {e}. Falling back to default volatility thresholds.")
            self.is_fitted = False

    def predict(self, log_returns: pd.Series, volatility: pd.Series) -> np.ndarray:
        """
        Predicts discrete regimes (0 or 1) for the given series.
        """
        if not self.is_fitted:
            # Fallback threshold classification if HMM fails to fit (e.g. too little data)
            vol_mean = volatility.mean()
            return np.where(volatility.values > vol_mean, 1, 0)
            
        obs = np.column_stack([
            log_returns.fillna(0.0).values,
            volatility.fillna(0.0).values
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            return self.model.predict(obs)
        except Exception:
            vol_mean = volatility.mean()
            return np.where(volatility.values > vol_mean, 1, 0)

    def _swap_states(self):
        """
        Swaps state mappings to guarantee State 0 has the lowest variance/volatility.
        """
        self.model.means_ = self.model.means_[::-1]
        self.model.covars_ = self.model.covars_[::-1]
        self.model.transmat_ = np.fliplr(self.model.transmat_[::-1])


# PyTorch Network Architecture implementing Self-Attention (TFT Core Elements)
class PyTorchTemporalAttentionNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input feature projection layer
        self.feature_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-Head Attention layer mapping temporal dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        # Gated Temporal Linear Layer (analogous to TFT's Gated Residual Network)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), # SiLU is the continuous GELU/Gated activation
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection predicting single-step or multi-step price
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 1. Project features
        proj_x = self.feature_proj(x) # (batch, seq, hidden)

        # 2. Self-Attention over chronological sequence
        attn_out, _ = self.attention(proj_x, proj_x, proj_x)
        
        # Residual connection
        x_res = proj_x + attn_out

        # 3. Gated Residual Network
        gated_out = self.gate_layer(x_res)
        x_norm = self.layer_norm(x_res + gated_out)

        # 4. Global pooling across sequence steps (use last element for forecast)
        last_step = x_norm[:, -1, :] # (batch, hidden)
        
        out = self.output_proj(last_step) # (batch, 1)
        return out.squeeze(-1)


class TFTAttentionRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compliant wrapper for our PyTorch Temporal Attention forecasting model.
    Enables direct integration with MAPIE conformal calibration.
    """
    def __init__(
        self, 
        input_dim: int = 5, 
        seq_len: int = 15, 
        hidden_dim: int = 32, 
        num_heads: int = 2,
        lr: float = 0.005,
        epochs: int = 40,
        batch_size: int = 16
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.scaler_X = None
        self.scaler_y = None

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a standard 2D feature matrix into 3D sequential blocks of shape (samples, seq_len, features).
        """
        num_samples = len(X)
        if num_samples <= self.seq_len:
            raise ValueError(f"Data length ({num_samples}) is shorter than sequence steps ({self.seq_len}).")
            
        X_seq = []
        y_seq = []
        
        for i in range(num_samples - self.seq_len):
            X_seq.append(X[i : i + self.seq_len])
            if y is not None:
                # y[i + seq_len - 1] corresponds to target value at the forecast horizon
                # since the sequence ending is at index i + seq_len - 1, and y is already rolled by h
                y_seq.append(y[i + self.seq_len - 1])
                
        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'TFTAttentionRegressor':
        from sklearn.preprocessing import StandardScaler
        # Standardize inputs
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Build seq sets
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_scaled)
        self.input_dim = X.shape[1]

        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Instantiate Network
        self.net = PyTorchTemporalAttentionNet(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        # Training loop
        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.net is None:
            raise ValueError("Model must be fitted before running predictions.")
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_original = len(X)
        X_scaled = self.scaler_X.transform(X)

        self.net.eval()
        
        # Pad inputs if length is exactly seq_len to allow single-step predictions
        is_single = False
        if len(X_scaled) <= self.seq_len:
            # Pad with repeated first row so sequence builder can extract at least 1 window
            pad_rows = self.seq_len + 1 - len(X_scaled)
            X_scaled = np.vstack([np.tile(X_scaled[0:1], (pad_rows, 1)), X_scaled])
            is_single = True

        X_seq, _ = self._prepare_sequences(X_scaled)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds_scaled = self.net(X_tensor).cpu().numpy().flatten()

        # Inverse scale targets
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # Always return an array of exactly n_original elements
        out = np.zeros(n_original)
        if is_single:
            out[:] = preds[-1]
        else:
            n_preds = len(preds)
            if n_preds >= n_original:
                out = preds[-n_original:]
            else:
                out[-n_preds:] = preds
                out[:-n_preds] = X[:n_original - n_preds, 0]
        
        return out


from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class RobustRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Robust Ridge Regression estimator for time-series forecasting.
    Includes automated lagging to construct high-res tabular features.
    """
    def __init__(self, seq_len: int = 15, alpha: float = 1.0):
        self.seq_len = seq_len
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.scaler_X = None
        self.scaler_y = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustRidgeRegressor':
        from sklearn.preprocessing import StandardScaler
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        n = len(X)
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        if n <= self.seq_len:
            pad_rows = self.seq_len + 1 - n
            X_scaled = np.vstack([np.tile(X_scaled[0:1], (pad_rows, 1)), X_scaled])
            y_scaled = np.concatenate([np.tile(y_scaled[0:1], pad_rows), y_scaled])
            n = len(X_scaled)
            
        # construct lag feature matrix
        X_lags = []
        y_targets = []
        for i in range(n - self.seq_len):
            X_lags.append(X_scaled[i : i + self.seq_len].flatten())
            y_targets.append(y_scaled[i + self.seq_len - 1])
            
        self.model.fit(np.array(X_lags), np.array(y_targets))
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n = len(X)
        X_scaled = self.scaler_X.transform(X)
        
        if len(X_scaled) <= self.seq_len:
            # Pad to fit seq_len
            pad_rows = self.seq_len + 1 - len(X_scaled)
            X_scaled = np.vstack([np.tile(X_scaled[0:1], (pad_rows, 1)), X_scaled])
            
        X_lags = []
        for i in range(len(X_scaled) - self.seq_len):
            X_lags.append(X_scaled[i : i + self.seq_len].flatten())
            
        preds_scaled = self.model.predict(np.array(X_lags))
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        out = np.zeros(n)
        n_preds = len(preds)
        if n_preds >= n:
            out = preds[-n:]
        else:
            out[-n_preds:] = preds
            out[:-n_preds] = X[:n - n_preds, 0]
        return out


class RobustGBRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Regressor for time-series forecasting.
    Uses lagged features and technical indicators for high accuracy.
    """
    def __init__(self, seq_len: int = 15, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.seq_len = seq_len
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        self.scaler_X = None
        self.scaler_y = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustGBRegressor':
        from sklearn.preprocessing import StandardScaler
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        n = len(X)
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        if n <= self.seq_len:
            pad_rows = self.seq_len + 1 - n
            X_scaled = np.vstack([np.tile(X_scaled[0:1], (pad_rows, 1)), X_scaled])
            y_scaled = np.concatenate([np.tile(y_scaled[0:1], pad_rows), y_scaled])
            n = len(X_scaled)
            
        # construct lag feature matrix
        X_lags = []
        y_targets = []
        for i in range(n - self.seq_len):
            X_lags.append(X_scaled[i : i + self.seq_len].flatten())
            y_targets.append(y_scaled[i + self.seq_len - 1])
            
        self.model.fit(np.array(X_lags), np.array(y_targets))
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n = len(X)
        X_scaled = self.scaler_X.transform(X)
        
        if len(X_scaled) <= self.seq_len:
            # Pad to fit seq_len
            pad_rows = self.seq_len + 1 - len(X_scaled)
            X_scaled = np.vstack([np.tile(X_scaled[0:1], (pad_rows, 1)), X_scaled])
            
        X_lags = []
        for i in range(len(X_scaled) - self.seq_len):
            X_lags.append(X_scaled[i : i + self.seq_len].flatten())
            
        preds_scaled = self.model.predict(np.array(X_lags))
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        out = np.zeros(n)
        n_preds = len(preds)
        if n_preds >= n:
            out = preds[-n:]
        else:
            out[-n_preds:] = preds
            out[:-n_preds] = X[:n - n_preds, 0]
        return out


class HoltWintersRegressor(BaseEstimator, RegressorMixin):
    """
    Holt-Winters Exponential Smoothing statistical TSA model.
    Learns seasonal and trend cycles to project prices.
    """
    def __init__(self, seasonal_periods: int = 5):
        self.seasonal_periods = seasonal_periods
        self.last_price = 100.0
        self.fitted_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HoltWintersRegressor':
        if isinstance(y, pd.Series):
            y_series = y.ffill().fillna(method='bfill').values
        else:
            y_series = np.array(y)
            
        self.last_price = float(y_series[-1]) if len(y_series) > 0 else 100.0
        
        if len(y_series) < 15:
            self.fitted_model = None
            return self
            
        try:
            model = ExponentialSmoothing(
                y_series,
                trend='add',
                seasonal='add',
                seasonal_periods=self.seasonal_periods,
                initialization_method='estimated'
            )
            self.fitted_model = model.fit(optimized=True)
        except Exception:
            try:
                model = ExponentialSmoothing(y_series, trend='add', seasonal=None)
                self.fitted_model = model.fit(optimized=True)
            except Exception:
                self.fitted_model = None
                
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self.fitted_model is None:
            return np.full(n, self.last_price)
            
        try:
            forecasts = self.fitted_model.forecast(n)
            return np.array(forecasts)
        except Exception:
            return np.full(n, self.last_price)


class EnsembleForecaster:
    """
    State-of-the-art Multi-Model Ensemble engine wrapping TFT Attention,
    Robust Ridge, Gradient Boosting, and Holt-Winters Exponential Smoothing.
    Fits models, evaluates out-of-fold MAPE, and dynamically yields predictions.
    Customizes ensembling weights and conformal envelopes using fundamental metrics.
    """
    def __init__(self, seq_len: int = 15):
        self.seq_len = seq_len
        self.tft = TFTAttentionRegressor(seq_len=seq_len, epochs=25)
        self.ridge = RobustRidgeRegressor(seq_len=seq_len)
        self.gbr = RobustGBRegressor(seq_len=seq_len)
        self.hw = HoltWintersRegressor()
        self.weights = {"tft": 0.4, "ridge": 0.2, "gbr": 0.2, "hw": 0.2}
        self.conformal_multiplier = 1.0
        self.regime_label = "⚖️ STANDARD COMPOSITE"

    def calibrate_weights_with_fundamentals(self, fundamentals: dict) -> Tuple[dict, float, str]:
        """
        Calibrates base ensemble weights and conformal scale multiplier based on fundamentals:
        market_cap, pe_ratio, roce, roe, debt_to_equity, dividend_yield, book_value, sales_growth.
        """
        if not fundamentals:
            return {"tft": 0.30, "ridge": 0.25, "gbr": 0.25, "hw": 0.20}, 1.0, "⚖️ STANDARD COMPOSITE"
            
        pe = fundamentals.get("pe_ratio", 0.0)
        roe = fundamentals.get("roe", 0.0)
        de = fundamentals.get("debt_to_equity", 0.0)
        
        # 1. 💎 HIGH-GROWTH QUALITY
        # Criteria: High ROE (>= 15%) and moderate/high PE (>= 15) indicating high quality growth
        if roe >= 15.0 and pe >= 15.0:
            weights = {"tft": 0.50, "ridge": 0.15, "gbr": 0.25, "hw": 0.10}
            return weights, 0.95, "💎 HIGH-GROWTH QUALITY"
            
        # 2. ⚠️ HIGH LEVERAGE / RISK
        # Criteria: High Debt to Equity (>= 1.5) or negative ROE indicating financial leverage stress
        if de >= 1.5 or roe < 0.0:
            # GBR tree structures capture sudden shocks, Ridge/HW are structural/defensive
            weights = {"tft": 0.20, "ridge": 0.20, "gbr": 0.40, "hw": 0.20}
            return weights, 1.25, "⚠️ HIGH LEVERAGE / RISK"
            
        # 3. 📈 VALUE / CYCLICAL
        # Criteria: Low PE (< 15) and low Debt to Equity (< 1.0) indicating solid value stocks
        if pe > 0.0 and pe < 15.0 and de < 1.0:
            # Linear & statistical trends dominate mean-reverting defensive value stocks
            weights = {"tft": 0.15, "ridge": 0.35, "gbr": 0.20, "hw": 0.30}
            return weights, 1.0, "📈 VALUE / CYCLICAL"
            
        # 4. ⚖️ STANDARD COMPOSITE (Default)
        weights = {"tft": 0.30, "ridge": 0.25, "gbr": 0.25, "hw": 0.20}
        return weights, 1.0, "⚖️ STANDARD COMPOSITE"
        
    def fit(self, X: np.ndarray, y: np.ndarray, fundamentals: dict = None):
        n = len(X)
        split = max(self.seq_len * 2, int(n * 0.80))
        
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        
        # Fit models on training split
        self.tft.fit(X_train, y_train)
        self.ridge.fit(X_train, y_train)
        self.gbr.fit(X_train, y_train)
        self.hw.fit(X_train, y_train)
        
        # Calibrate base prior weights using Screener fundamentals
        prior_weights, self.conformal_multiplier, self.regime_label = self.calibrate_weights_with_fundamentals(fundamentals)
        
        # Predict on validation set to get data-driven validation weights
        if len(X_val) > self.seq_len + 2:
            preds_tft = self.tft.predict(X_val)
            preds_ridge = self.ridge.predict(X_val)
            preds_gbr = self.gbr.predict(X_val)
            preds_hw = self.hw.predict(X_val)
            
            # Compute MAPEs
            mape_tft = float(np.mean(np.abs(y_val - preds_tft) / np.abs(y_val + 1e-9)))
            mape_ridge = float(np.mean(np.abs(y_val - preds_ridge) / np.abs(y_val + 1e-9)))
            mape_gbr = float(np.mean(np.abs(y_val - preds_gbr) / np.abs(y_val + 1e-9)))
            mape_hw = float(np.mean(np.abs(y_val - preds_hw) / np.abs(y_val + 1e-9)))
            
            # Avoid extremes or zeros
            mape_tft = max(1e-5, mape_tft)
            mape_ridge = max(1e-5, mape_ridge)
            mape_gbr = max(1e-5, mape_gbr)
            mape_hw = max(1e-5, mape_hw)
            
            # Calculate dynamic weights proportional to 1/MAPE
            inv_tft = 1.0 / mape_tft
            inv_ridge = 1.0 / mape_ridge
            inv_gbr = 1.0 / mape_gbr
            inv_hw = 1.0 / mape_hw
            
            total_inv = inv_tft + inv_ridge + inv_gbr + inv_hw
            val_weights = {
                "tft": inv_tft / total_inv,
                "ridge": inv_ridge / total_inv,
                "gbr": inv_gbr / total_inv,
                "hw": inv_hw / total_inv
            }
            
            # Bayesian update: 60% fundamental prior, 40% validation performance
            for m in self.weights:
                self.weights[m] = round(0.60 * prior_weights[m] + 0.40 * val_weights[m], 3)
                
            # Normalize to sum to exactly 1.0
            sum_w = sum(self.weights.values())
            for m in self.weights:
                self.weights[m] = round(self.weights[m] / sum_w, 3)
            # Re-verify and absorb small rounding residual in tft
            self.weights["tft"] = round(1.0 - sum(w for k, w in self.weights.items() if k != "tft"), 3)
            
            print(f"[ENSEMBLE CALIBRATED] Fundamentals-driven Weights: {self.weights} under regime {self.regime_label}")
        else:
            self.weights = prior_weights
            
        # Fit on whole dataset to prepare for final forecast
        self.tft.fit(X, y)
        self.ridge.fit(X, y)
        self.gbr.fit(X, y)
        self.hw.fit(X, y)
        
    def predict(self, X: np.ndarray) -> dict:
        """
        Generates individual and weighted ensemble predictions.
        """
        p_tft = self.tft.predict(X)
        p_ridge = self.ridge.predict(X)
        p_gbr = self.gbr.predict(X)
        p_hw = self.hw.predict(X)
        
        p_ensemble = (
            p_tft * self.weights["tft"] +
            p_ridge * self.weights["ridge"] +
            p_gbr * self.weights["gbr"] +
            p_hw * self.weights["hw"]
        )
        
        return {
            "ensemble": p_ensemble,
            "tft": p_tft,
            "ridge": p_ridge,
            "gbr": p_gbr,
            "hw": p_hw,
            "weights": self.weights,
            "conformal_multiplier": self.conformal_multiplier,
            "regime_label": self.regime_label
        }

