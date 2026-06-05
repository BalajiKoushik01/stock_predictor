import sys
# Mask curl_cffi from being imported by yfinance to prevent TLS certificate errors
sys.modules['curl_cffi'] = None
sys.modules['curl_cffi.requests'] = None

import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable standard library SSL validation
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Disable requests validation globally
original_requests_request = requests.Session.request
def patched_requests_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_requests_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_requests_request


from app.database import db_manager
from app.data_ingestion import apex_ingestor
from app.math_module import (
    fractional_differencing_ffd,
    find_optimal_d,
    decompose_emd_smoothing,
    run_johansen_cointegration,
    fit_macro_var,
    calculate_rsi,
    calculate_macd,
    calculate_index_metrics
)
from app.models import RegimeDetector, EnsembleForecaster
from app.risk import run_conformal_forecasting, apex_backtester
from app.utils import get_nse_trading_days

# Load local NSE stock database for instant autocomplete
_NSE_DB_PATH = os.path.join(os.path.dirname(__file__), 'nse_stocks.json')
try:
    with open(_NSE_DB_PATH, 'r', encoding='utf-8') as _f:
        NSE_STOCKS: list = json.load(_f)
except Exception:
    NSE_STOCKS = []

app = FastAPI(
    title="Project Apex API",
    description="Institutional-grade Quantitative Forecasting & Mathematical Feature Engineering Engine",
    version="1.0.0"
)

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    from datetime import timezone
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_connected": db_manager.db_path is not None
    }

@app.get("/api/stocks/search")
def local_stock_search(q: str = Query(..., description="Stock name or symbol prefix")):
    """
    Instant local search from bundled NSE stock database. Returns sub-millisecond results.
    Falls back to Yahoo Finance for unknown stocks.
    """
    query = q.strip().upper()
    if not query:
        return []
    
    # Priority 1: exact prefix match on symbol
    results = [s for s in NSE_STOCKS if s['symbol'].startswith(query)]
    
    # Priority 2: name contains query (case-insensitive)
    name_matches = [s for s in NSE_STOCKS 
                    if query.lower() in s['name'].lower() and s not in results]
    results.extend(name_matches)
    
    # Limit to 10 suggestions
    results = results[:10]
    
    # If local DB has no matches, fall back to Yahoo Finance
    if not results:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            r = requests.get(
                f'https://query2.finance.yahoo.com/v1/finance/search?q={q}&lang=en-IN&region=IN',
                headers=headers, timeout=4
            )
            if r.ok:
                quotes = r.json().get('quotes', [])
                seen = set()
                for qt in quotes:
                    sym = qt.get('symbol', '')
                    exch = qt.get('exchange', '')
                    if exch in ('NSI', 'BSE') or sym.endswith('.NS') or sym.endswith('.BO'):
                        clean = sym.split('.')[0].upper()
                        if clean not in seen:
                            seen.add(clean)
                            results.append({
                                'symbol': clean,
                                'name': qt.get('longname', qt.get('shortname', clean)),
                                'exchange': 'NSE' if exch == 'NSI' or sym.endswith('.NS') else 'BSE'
                            })
        except Exception:
            pass
    
    return results[:10]

@app.get("/api/ticker/search")
def search_suggest_ticker(q: str = Query(..., description="Fuzzy query of company name or ticker")):
    """
    Intelligently maps search queries (like 'dixon technologies') to exact listed tickers in NSE/BSE
    utilizing Yahoo Finance search query indexing with browser simulation headers.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return []
            
        quotes = r.json().get("quotes", [])
        results = []
        seen_symbols = set()
        for quote in quotes:
            symbol = quote.get("symbol", "")
            exchange = quote.get("exchange", "")
            longname = quote.get("longname", quote.get("shortname", ""))
            
            # Standardize exchange and prioritize NSE and BSE listings
            if exchange in ("NSI", "BSE") or symbol.endswith(".NS") or symbol.endswith(".BO") or "NSE" in exchange:
                clean_symbol = symbol.split(".")[0].upper()
                if clean_symbol not in seen_symbols:
                    seen_symbols.add(clean_symbol)
                    results.append({
                        "symbol": clean_symbol,
                        "name": longname,
                        "exchange": "NSE" if (exchange == "NSI" or symbol.endswith(".NS")) else "BSE"
                    })
        return results
    except Exception as e:
        print(f"Fuzzy search failed: {e}")
        return []

@app.post("/api/pipeline/upload")
async def upload_custom_dataset(
    file: UploadFile = File(...),
    ticker_name: str = Query("CUSTOM_UPLOAD", description="Arbitrary name for the custom asset")
):
    """
    Parses and standardizes custom uploaded CSV/XLSX price records, fits feature pipelines,
    HMM classification, and conformal predictions immediately.
    """
    try:
        filename = file.filename.lower()
        contents = await file.read()
        
        # Parse into Pandas DataFrame
        if filename.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            import io
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or XLSX.")
            
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        # Standardize column headers
        # Find timestamp / date column
        date_col = None
        for col in df.columns:
            if col.lower() in ("timestamp", "date", "datetime", "time"):
                date_col = col
                break
                
        if not date_col:
            # Assume first column is index/timestamp
            date_col = df.columns[0]
            
        # Find close price column
        close_col = None
        for col in df.columns:
            if col.lower() in ("close", "close_raw", "last", "settle"):
                close_col = col
                break
        if not close_col:
            # Look for any column containing 'close' or 'price'
            for col in df.columns:
                if "close" in col.lower() or "price" in col.lower() or "val" in col.lower():
                    close_col = col
                    break
        if not close_col:
            raise HTTPException(status_code=400, detail="Could not identify Close price column. Ensure a column header named 'Close' exists.")
            
        # Standardize naming
        df = df.rename(columns={date_col: 'timestamp', close_col: 'close'})
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing optional pricing columns
        for col in ('open', 'high', 'low'):
            if col not in df.columns:
                df[col] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0.0
            
        # Handle advanced features
        if 'pcr_oi' not in df.columns:
            df['pcr_oi'] = 1.0 # default neutral Put-Call Ratio
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0 # default neutral sentiment score
            
        # Ensure chronological sorting
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)
        df['rolling_volatility'] = df['log_returns'].rolling(window=20).std().fillna(0.0)
        
        # Apply EMD and FFD
        close_series = df['close']
        optimal_d = find_optimal_d(np.log(close_series))
        
        df['close_ffd'] = fractional_differencing_ffd(close_series, d=optimal_d)
        
        if len(close_series) > 1000:
            latest_close = close_series.iloc[-1000:]
            latest_emd = decompose_emd_smoothing(latest_close)
            close_emd = pd.Series(index=close_series.index, dtype=float)
            close_emd.iloc[-1000:] = latest_emd
            close_emd.iloc[:-1000] = close_series.iloc[:-1000].rolling(window=20, min_periods=1).mean()
        else:
            close_emd = decompose_emd_smoothing(close_series)
        df['close_emd_smoothed'] = close_emd
        
        # HMM regimes
        hmm = RegimeDetector()
        hmm.fit(df['log_returns'], df['rolling_volatility'])
        df['regime'] = hmm.predict(df['log_returns'], df['rolling_volatility'])
        
        # Save to DuckDB under custom ticker name
        custom_ticker = f"UPLOAD_{ticker_name.upper().replace(' ', '_')}"
        
        # Clean NaN/inf values to prevent JSON serialization crash
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Calculate technical indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'] = calculate_macd(df['close'])

        features_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'ticker': custom_ticker,
            'close_raw': df['close'],
            'close_ffd': df['close_ffd'].bfill().fillna(0.0),
            'close_emd_smoothed': df['close_emd_smoothed'].fillna(0.0),
            'pcr_oi': df['pcr_oi'].fillna(1.0),
            'sentiment_score': df['sentiment_score'].fillna(0.0),
            'rolling_volatility': df['rolling_volatility'].fillna(0.0),
            'rsi': df['rsi'].fillna(50.0),
            'macd': df['macd'].fillna(0.0)
        })
        
        db_manager.execute(f"DELETE FROM processed_features WHERE ticker = '{custom_ticker}'")
        db_manager.save_dataframe("processed_features", features_df, if_exists="append")
        
        return {
            "status": "success",
            "ticker": custom_ticker,
            "optimal_d": optimal_d,
            "data_count": len(df),
            "preview": df.tail(100).to_dict(orient="records")
        }
    except Exception as e:
        print(f"Custom file uploader failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/run")
def trigger_pipeline(
    ticker: str = Query(..., description="NSE Stock Ticker (e.g. RELIANCE, TCS)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD). Use '1900-01-01' for earliest IPO data."),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    try:
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            start_date = '1900-01-01'

        print(f"Triggering Apex Pipeline for {ticker} from {start_date} to {end_date}")

        # Solve fuzzy ticker match automatically on execution if it has a non-NSE format
        if not ticker.isupper() or len(ticker) > 10:
            matches = search_suggest_ticker(ticker)
            if matches:
                # Resolve to NSE symbol (ends with .NS) or top matched symbol
                best_match = next((m["symbol"] for m in matches if m["exchange"] == "NSE"), matches[0]["symbol"])
                print(f"[FUZZY RESOLVED] Query '{ticker}' resolved to exact listed ticker: {best_match}")
                ticker = best_match.replace(".NS", "")

        aligned_df = apex_ingestor.run_pipeline(ticker, start_date, end_date)
        if aligned_df.empty:
            raise HTTPException(status_code=404, detail="No historical data retrieved.")

        # Apply Mathematical Preprocessing
        close_series = aligned_df['close']
        log_close = np.log(close_series)

        optimal_d = find_optimal_d(log_close)
        close_ffd = fractional_differencing_ffd(close_series, d=optimal_d)
        aligned_df['close_ffd'] = close_ffd

        # High-Performance Lookback for EMD cycles (prevents CPU lockups on long IPO series)
        if len(close_series) > 1000:
            latest_close = close_series.iloc[-1000:]
            latest_emd = decompose_emd_smoothing(latest_close)
            close_emd = pd.Series(index=close_series.index, dtype=float)
            close_emd.iloc[-1000:] = latest_emd
            close_emd.iloc[:-1000] = close_series.iloc[:-1000].rolling(window=20, min_periods=1).mean()
        else:
            close_emd = decompose_emd_smoothing(close_series)
        aligned_df['close_emd_smoothed'] = close_emd

        aligned_df['rolling_volatility'] = aligned_df['log_returns'].rolling(window=20).std()
        aligned_df['rolling_volatility'] = aligned_df['rolling_volatility'].fillna(0.0)

        # HMM Regime Detection
        print("Calibrating Hidden Markov Model regimes...")
        hmm = RegimeDetector()
        hmm.fit(aligned_df['log_returns'], aligned_df['rolling_volatility'])
        regimes = hmm.predict(aligned_df['log_returns'], aligned_df['rolling_volatility'])
        aligned_df['regime'] = regimes

        # Technical Indicators calculation
        aligned_df['rsi'] = calculate_rsi(aligned_df['close'])
        aligned_df['macd'] = calculate_macd(aligned_df['close'])

        # Compute Index Benchmarking statistics relative to Nifty 50 (^NSEI)
        beta, alpha, correlation = 1.0, 0.0, 1.0
        try:
            # Query the concurrently fetched Nifty 50 close prices
            bench_q = "SELECT timestamp, close FROM benchmark_data WHERE ticker = '^NSEI' ORDER BY timestamp ASC"
            bench_df = db_manager.load_dataframe(bench_q)
            if not bench_df.empty:
                bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp'])
                # Align both dataframes on timestamp
                merged = pd.merge(
                    aligned_df[['timestamp', 'close']],
                    bench_df,
                    on='timestamp',
                    suffixes=('_stock', '_index')
                ).dropna()
                
                if len(merged) > 30:
                    metrics = calculate_index_metrics(merged['close_stock'], merged['close_index'])
                    beta = metrics['beta']
                    alpha = metrics['alpha_annualized']
                    correlation = metrics['correlation']
                    print(f"[BENCHMARK CALCULATED] Beta: {beta}, Alpha: {alpha}, Corr: {correlation}")
        except Exception as e:
            print(f"Failed to calculate benchmark stats: {e}")

        # Clean NaN/inf values to prevent JSON serialization crash
        aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        features_df = pd.DataFrame({
            'timestamp': aligned_df['timestamp'],
            'ticker': ticker,
            'close_raw': aligned_df['close'],
            'close_ffd': aligned_df['close_ffd'].bfill().fillna(0.0),
            'close_emd_smoothed': aligned_df['close_emd_smoothed'].fillna(0.0),
            'pcr_oi': aligned_df['pcr_oi'].fillna(1.0),
            'sentiment_score': aligned_df['sentiment_score'].fillna(0.0),
            'rolling_volatility': aligned_df['rolling_volatility'].fillna(0.0),
            'rsi': aligned_df['rsi'].fillna(50.0),
            'macd': aligned_df['macd'].fillna(0.0)
        })

        # Remove previous custom files with same name to prevent accumulation
        db_manager.execute(f"DELETE FROM processed_features WHERE ticker = '{ticker}'")
        db_manager.save_dataframe("processed_features", features_df, if_exists="append")
        
        # Fetch and save fundamentals from Screener.in / yfinance fallback
        fundamentals = {"market_cap": 0.0, "pe_ratio": 0.0, "roce": 0.0, "roe": 0.0, "debt_to_equity": 0.0, "dividend_yield": 0.0, "book_value": 0.0, "sales_growth": 0.0, "source": "None"}
        try:
            fundamentals = apex_ingestor.scrape_screener_fundamentals(ticker)
            fund_df = pd.DataFrame([{
                'ticker': ticker,
                'market_cap': fundamentals['market_cap'],
                'pe_ratio': fundamentals['pe_ratio'],
                'roce': fundamentals['roce'],
                'roe': fundamentals['roe'],
                'debt_to_equity': fundamentals['debt_to_equity'],
                'dividend_yield': fundamentals['dividend_yield'],
                'book_value': fundamentals['book_value'],
                'sales_growth': fundamentals['sales_growth'],
                'source': fundamentals['source'],
                'updated_at': datetime.now()
            }])
            db_manager.save_dataframe("fundamental_metrics", fund_df, if_exists="append")
        except Exception as e:
            print(f"Failed to fetch and save fundamentals: {e}")

        # Ensure timestamp is clean string representation
        aligned_df['timestamp'] = pd.to_datetime(aligned_df['timestamp']).dt.strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "ticker": ticker,
            "optimal_d": optimal_d,
            "data_count": len(aligned_df),
            "benchmark": {
                "beta": beta,
                "alpha_annualized": alpha,
                "correlation": correlation
            },
            "fundamentals": fundamentals,
            "preview": aligned_df.tail(100).to_dict(orient="records")
        }
    except Exception as e:
        print(f"Pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/preview")
def get_data_preview(
    ticker: str = Query(..., description="NSE Ticker"),
    limit: int = Query(500, description="Max historical points to retrieve")
):
    try:
        query = f"""
            SELECT * FROM processed_features 
            WHERE ticker = '{ticker}' 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        df = db_manager.load_dataframe(query)
        if df.empty:
            return {"status": "empty", "ticker": ticker, "data": []}
            
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df.iloc[::-1]
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
        
        # Calculate regimes dynamically
        log_returns = np.log(df['close_raw'] / df['close_raw'].shift(1)).fillna(0.0)
        hmm = RegimeDetector()
        hmm.fit(log_returns, df['rolling_volatility'])
        regimes = hmm.predict(log_returns, df['rolling_volatility'])
        df['regime'] = regimes.tolist()
        
        return {
            "status": "success",
            "ticker": ticker,
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/predict")
def predict_forecast_envelope(
    ticker: str = Query(..., description="NSE Ticker"),
    horizon_steps: int = Query(10, description="Forecast horizon steps ahead")
):
    """
    Executes the multi-model weighted ensemble forecasting engine, calibrates error residual models
    via conformal regression, and outputs the strict 90% and 95% forecast envelopes.
    Customizes ensembling and conformal widths based on Screener/Yahoo Finance fundamental metrics.
    """
    try:
        # 1. Load aligned preprocessed features
        query = f"SELECT * FROM processed_features WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
        df = db_manager.load_dataframe(query)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No historic feature sets found for {ticker}. Please run pipeline first.")
            
        N = len(df)
        if N < 50:
            raise HTTPException(status_code=400, detail="Insufficient data to train quantitative forecasting models.")

        # Load fundamentals from DuckDB or dynamically scrape if missing
        fundamentals = {"market_cap": 0.0, "pe_ratio": 0.0, "roce": 0.0, "roe": 0.0, "debt_to_equity": 0.0, "dividend_yield": 0.0, "book_value": 0.0, "sales_growth": 0.0, "source": "None"}
        try:
            fund_q = f"SELECT * FROM fundamental_metrics WHERE ticker = '{ticker}'"
            fund_df = db_manager.load_dataframe(fund_q)
            if not fund_df.empty:
                fundamentals = fund_df.iloc[0].to_dict()
                if 'updated_at' in fundamentals:
                    fundamentals['updated_at'] = str(fundamentals['updated_at'])
            else:
                fundamentals = apex_ingestor.scrape_screener_fundamentals(ticker)
                save_df = pd.DataFrame([{
                    'ticker': ticker,
                    'market_cap': fundamentals['market_cap'],
                    'pe_ratio': fundamentals['pe_ratio'],
                    'roce': fundamentals['roce'],
                    'roe': fundamentals['roe'],
                    'debt_to_equity': fundamentals['debt_to_equity'],
                    'dividend_yield': fundamentals['dividend_yield'],
                    'book_value': fundamentals['book_value'],
                    'sales_growth': fundamentals['sales_growth'],
                    'source': fundamentals['source'],
                    'updated_at': datetime.now()
                }])
                db_manager.save_dataframe("fundamental_metrics", save_df, if_exists="append")
        except Exception as e:
            print(f"Fundamentals resolution failed in predict: {e}")

        # Calculate returns
        log_returns = np.log(df['close_raw'] / df['close_raw'].shift(1)).fillna(0.0).values

        # 2. Extract inputs X (including technical indicators)
        X = np.column_stack([
            df['close_raw'].values,
            df['rolling_volatility'].values,
            df['sentiment_score'].values,
            df['pcr_oi'].values,
            df['rsi'].values,
            df['macd'].values,
            log_returns
        ])
        
        seq_len = 15
        test_features = X[-seq_len:]
        
        # Train HMM to detect current market regime
        hmm = RegimeDetector()
        vol_pd = pd.Series(df['rolling_volatility'].values)
        ret_pd = pd.Series(log_returns)
        hmm.fit(ret_pd, vol_pd)
        regimes = hmm.predict(ret_pd, vol_pd)
        current_regime = int(regimes[-1])

        # Generate future trading dates (excluding weekends & holidays)
        last_date = pd.to_datetime(df['timestamp'].iloc[-1])
        future_dates = get_nse_trading_days(
            start_date=last_date + timedelta(days=1), 
            end_date=last_date + timedelta(days=horizon_steps * 2.5)
        )[:horizon_steps]

        # ─── OPTIMIZATION: Train TFT once upfront, reuse for all horizon steps ───
        # Previously: 10 steps × 2 fits (split+full) + 10 conformal fits = 30 TFT trains (~20+ min on CPU)
        # Now: 1 TFT train upfront, reused via tft_epochs=0 across all steps (~1-3 min total)
        forecast_records = []
        ensemble_weights = {"tft": 0.40, "ridge": 0.20, "gbr": 0.20, "hw": 0.20}
        fundamental_regime = "⚖️ STANDARD COMPOSITE"
        conformal_multiplier = 1.0
        print(f"Generating dynamic ensemble forecast path across {horizon_steps} sessions...")

        # Step 1: Warm up TFT on full dataset once (5 epochs) and a shared conformal TFT
        print("[OPTIMIZE] Pre-training shared TFT model on full feature set...")
        from app.models import TFTAttentionRegressor
        shared_tft = TFTAttentionRegressor(seq_len=seq_len, epochs=5, batch_size=16, lr=0.005)
        shared_tft.fit(X, df['close_raw'].values)
        # Shared conformal TFT: fitted once on calibration split with 0 re-training epochs afterward
        shared_conformal_tft = TFTAttentionRegressor(seq_len=seq_len, epochs=5, batch_size=16, lr=0.005)
        conformal_split = max(seq_len + 10, int(len(X) * 0.80))
        shared_conformal_tft.fit(X[:conformal_split], df['close_raw'].values[:conformal_split])
        # Mark as pre-trained so all subsequent calls skip re-fitting
        shared_tft.epochs = 0
        shared_conformal_tft.epochs = 0
        print("[OPTIMIZE] Shared TFT models pre-trained. Running horizon sweep with frozen TFT weights...")

        for h in range(1, horizon_steps + 1):
            y_h = np.roll(df['close_raw'].values, -h)
            y_h[-h:] = df['close_raw'].values[-1]
            
            train_features_h = X[:-h]
            train_targets_h = y_h[:-h]
            
            # Fit ensemble with tft_epochs=0 to skip TFT re-train; Ridge/GBR/HW still fit fresh (fast, <1s each)
            ensemble = EnsembleForecaster(seq_len=seq_len, tft_epochs=0)
            # Inject pre-trained TFT weights directly
            ensemble.tft = shared_tft
            ensemble.fit(train_features_h, train_targets_h, fundamentals=fundamentals)
            
            # Save weights from the final step for return metadata
            if h == horizon_steps:
                ensemble_weights = ensemble.weights
                fundamental_regime = ensemble.regime_label
                conformal_multiplier = ensemble.conformal_multiplier
                
            pred_dict = ensemble.predict(test_features)
            
            # Use Conformal bounds — pass the pre-trained conformal TFT to avoid re-training
            _, bounds_90_h, bounds_95_h = run_conformal_forecasting(
                train_features_h, train_targets_h, test_features, horizon_steps=h,
                base_model=shared_conformal_tft
            )
            
            pred_val = float(pred_dict["ensemble"][-1])
            tft_val = float(pred_dict["tft"][-1])
            ridge_val = float(pred_dict["ridge"][-1])
            gbr_val = float(pred_dict["gbr"][-1])
            hw_val = float(pred_dict["hw"][-1])
            
            l90, u90 = float(bounds_90_h[-1, 0]), float(bounds_90_h[-1, 1])
            l95, u95 = float(bounds_95_h[-1, 0]), float(bounds_95_h[-1, 1])
            
            # Apply fundamental-driven scaling to conformal bounds
            if conformal_multiplier != 1.0:
                half_width_90 = (u90 - l90) * conformal_multiplier / 2.0
                l90 = pred_val - half_width_90
                u90 = pred_val + half_width_90
                
                half_width_95 = (u95 - l95) * conformal_multiplier / 2.0
                l95 = pred_val - half_width_95
                u95 = pred_val + half_width_95
            
            # Enforce non-negative bounds
            l90 = max(0.0, l90)
            l95 = max(0.0, l95)
            
            future_date = future_dates[h - 1] if (h - 1) < len(future_dates) else (last_date + timedelta(days=h)).date()
            
            forecast_records.append({
                "timestamp": future_date.strftime('%Y-%m-%d'),
                "forecast_close": pred_val,
                "tft_close": tft_val,
                "ridge_close": ridge_val,
                "gbr_close": gbr_val,
                "hw_close": hw_val,
                "lower_90": l90,
                "upper_90": u90,
                "lower_95": l95,
                "upper_95": u95
            })

        return {
            "status": "success",
            "ticker": ticker,
            "current_regime": current_regime,
            "regime_label": "High Volatility / Bearish" if current_regime == 1 else "Low Volatility / Bullish",
            "fundamental_regime": fundamental_regime,
            "fundamentals": fundamentals,
            "horizon_steps": horizon_steps,
            "ensemble_weights": ensemble_weights,
            "forecasts": forecast_records
        }
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/backtest")
def trigger_backtesting(
    ticker: str = Query(..., description="NSE Ticker"),
    horizon_steps: int = Query(10, description="Forecast horizon steps")
):
    """
    Rigorously evaluates strategy out-of-sample performance over historic rolling indices.
    """
    try:
        # Load aligned datasets
        query = f"SELECT * FROM processed_features WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
        df = db_manager.load_dataframe(query)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No dataset found for backtesting ticker {ticker}.")
            
        N = len(df)
        if N < 150:
            raise HTTPException(status_code=400, detail=f"Insufficient history ({N} rows) to execute out-of-sample backtests. Need at least 150 sessions.")

        # Re-inject log returns
        df['log_returns'] = np.log(df['close_raw'] / df['close_raw'].shift(1)).fillna(0.0)

        # Train HMM to detect regimes
        hmm = RegimeDetector()
        vol_pd = pd.Series(df['rolling_volatility'].values)
        ret_pd = pd.Series(df['log_returns'].values)
        hmm.fit(ret_pd, vol_pd)
        regimes = hmm.predict(ret_pd, vol_pd)
        df['regime'] = regimes

        # Run Walk-Forward Optimizer
        metrics = apex_backtester.evaluate_strategy(df, horizon_steps=horizon_steps, min_train_days=120)
        
        return {
            "status": "success",
            "ticker": ticker,
            "horizon_steps": horizon_steps,
            "metrics": metrics
        }
    except Exception as e:
        print(f"Backtesting execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/math/cointegration")
def calculate_cointegration(
    tickers: List[str] = Query(..., description="List of NSE stock tickers to test (at least two)")
):
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Cointegration testing requires a list of at least two tickers.")

    try:
        prices_dict = {}
        missing = []
        for t in tickers:
            query = f"SELECT timestamp, close_raw FROM processed_features WHERE ticker = '{t}' ORDER BY timestamp ASC"
            df = db_manager.load_dataframe(query)
            if df.empty:
                missing.append(t)
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            prices_dict[t] = df['close_raw']

        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"No data for: {', '.join(missing)}. Run pipeline for each ticker first."
            )

        # Build aligned price matrix using outer join + forward-fill to handle calendar gaps
        prices_df = pd.DataFrame(prices_dict)
        prices_df = prices_df.ffill().bfill().dropna()

        if len(prices_df) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(prices_df)} overlapping sessions found after alignment. Need at least 30. Ensure both tickers cover the same period."
            )

        results = run_johansen_cointegration(prices_df)
        
        return {
            "status": "success",
            "tickers": tickers,
            "overlapping_sessions": len(prices_df),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/math/var")
def calculate_var(
    target_ticker: str = Query(..., description="Target equity ticker (e.g. INFOSYS)"),
    macro_tickers: List[str] = Query(..., description="Macroeconomic/sector indexes (e.g. NIFTYIT)"),
    lags: int = Query(5, description="Lag order for VAR model")
):
    try:
        all_tickers = [target_ticker] + macro_tickers
        features = {}
        missing = []
        for t in all_tickers:
            query = f"SELECT timestamp, close_raw FROM processed_features WHERE ticker = '{t}' ORDER BY timestamp ASC"
            df = db_manager.load_dataframe(query)
            if df.empty:
                missing.append(t)
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['log_returns'] = np.log(df['close_raw'] / df['close_raw'].shift(1)).fillna(0.0)
            df = df.set_index('timestamp')
            features[t] = df['log_returns']

        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"No data for: {', '.join(missing)}. Run pipeline for each ticker first."
            )

        # Align with ffill to handle minor calendar gaps between exchanges
        features_df = pd.DataFrame(features).ffill().bfill().dropna()

        if len(features_df) < (lags * 3):
            raise HTTPException(status_code=400, detail=f"Insufficient historic data ({len(features_df)} rows) for VAR with {lags} lags.")
            
        results = fit_macro_var(features_df, lags=lags)
        return {
            "status": "success",
            "target": target_ticker,
            "macro_inputs": macro_tickers,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
