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

## Security and deployment status

The blanket TLS verification bypass and permissive CORS settings identified in the earlier version were addressed in the November 2026 code remediation. The current FastAPI application uses the `ALLOWED_ORIGINS` environment variable and local development defaults. The database layer also includes a non-destructive migration for missing feature columns and tests for selected query-safety behavior.

**This remains an experimental research application, not a publicly hardened investment platform.** It requires authentication, upload controls, network review, financial-data provider licensing and external model validation before a production release.

### Verification

```bash
# Backend unit tests (install backend test requirements first)
cd backend
python -m unittest discover -s tests -v
cd ..

# Frontend
cd frontend
npm ci
npm run build
```

See [engineering status](docs/ENGINEERING_STATUS.md) for additional release gates.

## Forecasts and backtests

Models can be wrong, especially during regime changes, data outages or look-ahead leakage. Backtested figures do not establish future returns. Validate time-based splits, transaction costs, uncertainty assumptions and the raw data before sharing any performance results.

No universal prediction accuracy, mathematically guaranteed interval or validated financial performance is claimed here.
