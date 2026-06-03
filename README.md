# Project Apex: Institutional Quantitative Forecasting & Risk Engine

Project Apex is a production-grade quantitative forecasting and walk-forward backtesting platform. It implements advanced Time Series Analysis (TSA) and machine learning models to forecast asset prices, calculate conformal uncertainty envelopes, and simulate dynamic risk-adjusted trading strategies.

---

## 🧠 Core Mathematical Foundations

### 1. Fractional Differencing (FFD)
Traditional integer differencing (e.g., $d=1$) removes non-stationarity but completely wipes out the memory of historic price series. Project Apex utilizes **Fractional Differencing** to find the minimum differencing threshold ($d \approx 0.35$) that achieves stationarity (via ADF test validation) while preserving maximum historical information.

### 2. Empirical Mode Decomposition (EMD)
To isolate true market trends from high-frequency market noise, the engine decomposes raw price series into **Intrinsic Mode Functions (IMFs)**, generating noise-filtered trend envelopes for robust model input.

### 3. Volatility Regime Detection (Gaussian HMM)
A **Gaussian Hidden Markov Model (HMM)** classifies market states into distinct volatility regimes:
- **Regime 0 (Low Volatility Bullish)**: Characterized by steady upward drift and high model weight towards trend-following predictors.
- **Regime 1 (High Volatility Bearish)**: High variance, where risk controls (stop-loss boundaries) are tightened and model weights adjust dynamically.

### 4. Multi-Paradigm Ensemble Forecasting
The prediction engine combines four diverse model architectures:
- **Temporal Fusion Transformer (TFT)**: Self-attention networks mapping complex temporal interactions.
- **Robust Ridge Regression**: L2-regularized linear baseline for stable structural trends.
- **Gradient Boosting Regressor (GBR)**: Non-linear tree-based ensembles capturing non-linear feature maps.
- **Holt-Winters Exponential Smoothing**: Classic statistical forecasting to capture seasonal and trend drift.

### 5. Conformal Prediction (MAPIE)
Rather than raw point forecasts, Project Apex computes **distribution-free out-of-sample conformal intervals** (using MAPIE logic) at the 90% and 95% confidence bounds. This guarantees mathematically bounded uncertainty margins based on empirical residuals.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python), DuckDB (high-density OLAP storage), Scikit-Learn, Statsmodels, PyTorch (TFT attention)
- **Frontend**: Next.js (React, TypeScript), lightweight-charts (canvas-based high-density financial charting), Lucide Icons
- **Deployment**: Docker, Docker Compose

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- *Or local installations of Python 3.10+ and Node.js 18+*

### Method 1: Running with Docker Compose (Recommended)
From the root workspace, initialize the unified container stack:
```bash
docker-compose up --build
```
The application will launch on:
- Frontend Dashboard: [http://localhost:3000](http://localhost:3000)
- Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Method 2: Manual Local Startup

#### 1. Backend Server Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app/main.py
```

#### 2. Frontend Dashboard Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Walk-Forward Backtester & Risk Analysis

The platform incorporates an out-of-sample **Walk-Forward Backtester** that simulates historical performance:
- **Directional Accuracy (Hit Ratio)**: Percentage of correct sign predictions.
- **Sharpe Ratio**: Annualized risk-adjusted excess returns over risk-free benchmarks.
- **Maximum Drawdown**: Largest peak-to-trough equity curve decline.
- **Nifty 50 Index Benchmarking**: Automated calculation of systematic Beta ($\beta$), annualized alpha ($\alpha$), and co-movement correlation vector ($\rho$).
