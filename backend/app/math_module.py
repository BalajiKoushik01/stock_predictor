import numpy as np
import pandas as pd
from scipy.signal import lfilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

# Try to import PyEMD, fallback to statsmodels HP Filter or moving averages if EMD signal is unavailable
try:
    from PyEMD import EMD
    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    print("Warning: PyEMD (EMD-signal) not installed. Using statsmodels Hodrick-Prescott Filter as fallback for EMD.")

def get_ffd_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """
    Computes binomial weights for Fixed-Width Window Fractional Differentiation (FFD)
    following Marcos López de Prado's 'Advances in Financial Machine Learning'.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w)

def fractional_differencing_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Applies Fixed-Width Window Fractional Differentiation to a time series
    using a high-speed vectorized convolution (scipy.signal.lfilter).
    
    Args:
        series: Pandas Series containing the price or log price data.
        d: The fractional differencing order (0 <= d <= 1).
        thres: Threshold for the weight truncation.
        
    Returns:
        Fractionally differenced series, with first (width - 1) terms set to NaN.
    """
    if d == 0.0:
        return series.copy()
        
    w = get_ffd_weights(d, thres)
    width = len(w)
    
    # Fill any internal NaNs before differentiation
    series_filled = series.ffill().fillna(0.0).values
    
    # Vectorized FFD convolution via digital filter
    # lfilter computes y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1]...
    # Here, a = [1.0] and b = w
    differenced_values = lfilter(w, [1.0], series_filled)
    
    # Mask edge-effect terms at the beginning of the series
    differenced_series = pd.Series(differenced_values, index=series.index)
    if width > 1:
        differenced_series.iloc[:width - 1] = np.nan
        
    return differenced_series

def find_optimal_d(series: pd.Series, step: float = 0.05, thres: float = 1e-4) -> float:
    """
    Grid-searches the minimum fractional differencing order d in (0, 1) 
    that achieves stationarity (Augmented Dickey-Fuller p-value < 0.05).
    """
    series_clean = series.dropna()
    if len(series_clean) < 15:
        return 0.5 # Return a default if series is too short
        
    # Check if raw series is already stationary
    try:
        raw_p = adfuller(series_clean, maxlag=1, regression='c', autolag=None)[1]
        if raw_p < 0.05:
            return 0.0
    except Exception:
        pass
        
    # Search from 0.05 to 1.0
    for d in np.arange(0.05, 1.01, step):
        d_rounded = round(d, 4)
        diff_series = fractional_differencing_ffd(series, d_rounded, thres).dropna()
        
        if len(diff_series) < 10:
            continue
            
        try:
            # We use maxlag=1 for standard fast estimation during optimization
            adf_stat, p_val, _, _, _, _ = adfuller(diff_series, maxlag=1, regression='c', autolag=None)
            if p_val < 0.05:
                return d_rounded
        except Exception:
            continue
            
    return 1.0 # Default to integer differencing if stationarity not reached

def decompose_emd_smoothing(series: pd.Series, num_noise_imfs: int = 1) -> pd.Series:
    """
    Applies Empirical Mode Decomposition (EMD) to a signal and reconstructions 
    the series by summing only the lower-frequency IMFs and monotonic residual
    to isolate true market cycles and remove high-frequency stochastic noise.
    """
    series_clean = series.ffill().bfill().fillna(0.0)
    x = series_clean.values
    t = np.arange(len(x))
    
    if PYEMD_AVAILABLE:
        try:
            emd = EMD()
            imfs = emd.emd(x, t)
            num_imfs = imfs.shape[0]
            
            # If the number of extracted IMFs is less than or equal to noise IMF count,
            # return the series as is
            if num_imfs <= num_noise_imfs:
                return series.copy()
                
            # Reconstruct the signal by leaving out the first 'num_noise_imfs' (which represent highest frequency noise)
            smoothed_signal = np.sum(imfs[num_noise_imfs:], axis=0)
            return pd.Series(smoothed_signal, index=series.index)
        except Exception as e:
            print(f"PyEMD failed: {e}. Falling back to HP-Filter.")
            
    # Fallback: Hodrick-Prescott Filter to extract trend
    from statsmodels.tsa.filters.hp_filter import hpfilter
    try:
        # HP filter returns cycle, trend. Trend is our smoothed series.
        cycle, trend = hpfilter(series_clean, lamb=1600)
        return pd.Series(trend, index=series.index)
    except Exception:
        # Final fallback: simple exponential moving average
        return series.ewm(span=10).mean()

def run_johansen_cointegration(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """
    Runs Johansen Cointegration test on a set of I(1) asset price series.
    
    Args:
        df: DataFrame containing price series of assets.
        det_order: Deterministic terms order (-1: none, 0: constant, 1: linear trend).
        k_ar_diff: Number of lagged differences in the VAR model.
        
    Returns:
        Johansen cointegration test results.
    """
    if df.shape[1] < 2:
        raise ValueError("Cointegration requires at least two asset series.")
    
    df_clean = df.dropna()
    result = coint_johansen(df_clean, det_order, k_ar_diff)
    
    # Extract trace and max eigenvalue statistics and critical values (90%, 95%, 99%)
    return {
        "eigenvalues": result.eig.tolist(),
        "trace_stat": result.lr1.tolist(),
        "trace_crit": result.cvt.tolist(), # Matrix of critical values
        "max_eig_stat": result.lr2.tolist(),
        "max_eig_crit": result.cvm.tolist()
    }

def fit_macro_var(df: pd.DataFrame, lags: int = 5) -> dict:
    """
    Fits a Vector Autoregression (VAR) model to map dynamic lagged cross-dependencies
    between the target asset and macroeconomic/sector variables (e.g., Nifty IT Index).
    
    Returns:
        A dictionary containing the model coefficients and lag order.
    """
    df_clean = df.dropna()
    if len(df_clean) < (lags * 2):
        raise ValueError("Insufficient data points to fit the specified VAR lags.")
        
    model = VAR(df_clean)
    results = model.fit(maxlags=lags, ic='aic')
    
    return {
        "aic": float(results.aic),
        "bic": float(results.bic),
        "order": int(results.k_ar),
        "pvalues": results.pvalues.to_dict()
    }

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates standard Relative Strength Index (RSI) technical indicator.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0.0, 1e-9) # prevent divide by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0) # default to neutral 50

def calculate_macd(series: pd.Series) -> pd.Series:
    """
    Calculates MACD (12-period EMA - 26-period EMA).
    """
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def calculate_index_metrics(stock_prices: pd.Series, index_prices: pd.Series) -> dict:
    """
    Computes Beta, Jensen's Alpha, and Correlation relative to the benchmark Index.
    """
    stock_returns = stock_prices.pct_change().fillna(0.0)
    index_returns = index_prices.pct_change().fillna(0.0)
    
    cov = np.cov(stock_returns, index_returns)
    if cov.shape == (2, 2) and cov[1, 1] > 0:
        beta = float(cov[0, 1] / cov[1, 1])
    else:
        beta = 1.0 # default fallback to market beta
        
    # Jensen's Alpha assuming a 6% annualized Risk Free Rate (approx 0.0238% daily)
    rf_daily = 0.06 / 252.0
    stock_cum = (1 + stock_returns).prod() - 1
    index_cum = (1 + index_returns).prod() - 1
    
    # annualized alpha
    alpha = float((stock_returns.mean() - rf_daily) - beta * (index_returns.mean() - rf_daily)) * 252
    
    # Pearson Correlation Coefficient
    corr = float(stock_returns.corr(index_returns))
    if np.isnan(corr):
        corr = 0.0
        
    return {
        "beta": round(beta, 3),
        "alpha_annualized": round(alpha, 4),
        "correlation": round(corr, 3)
    }

