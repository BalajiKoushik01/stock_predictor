# Project Apex — time-series forecasting prototype

Project Apex is an experimental financial time-series application. Its Python/FastAPI backend contains data ingestion, indicators, forecasting and risk-analysis modules; a Next.js interface displays analytical output.

**Status:** research prototype, not investment advice or a production trading system.

## Repository map

| Path | Purpose |
| --- | --- |
| `backend/app/main.py` | FastAPI service |
| `backend/app/data_ingestion.py` | Data acquisition |
| `backend/app/math_module.py` | Statistical and technical-indicator utilities |
| `backend/app/models.py` | Forecasting components |
| `backend/app/risk.py` | Risk and retrospective analysis |
| `frontend/` | Next.js web client |
| `docker-compose.yml` | Local multi-container setup |

## Local setup

```bash
git clone https://github.com/BalajiKoushik01/stock_predictor.git
cd stock_predictor
docker compose up --build
```

The compose file exposes ports 8000 and 3000 and mounts source directories for development. Check installed Docker/Compose versions and dependency availability. It is a local-development configuration, not a hardened production deployment.

You can also install `backend/requirements.txt` in a Python virtual environment and run the web client from `frontend/` with `npm install && npm run dev`. Review the backend startup entry point before running it outside Docker.

## Security warning — action required

**Do not deploy the current backend to a public production environment.** At the time of review, `backend/app/main.py` disables TLS certificate verification and monkey-patches the requests client to set `verify=False`. This weakens protection against interception of financial data and other network traffic. Remove those overrides, use verified CA certificates and test HTTPS data sources before any deployment. The current FastAPI CORS configuration also needs restrictive origins and credential handling.

## Forecasts and backtests

Models can be wrong, especially during regime changes, data outages or look-ahead leakage. Backtested figures do not establish future returns. Validate time-based splits, transaction costs, uncertainty assumptions and the raw data before sharing any performance results.

No universal prediction accuracy, mathematically guaranteed interval or validated financial performance is claimed here.
