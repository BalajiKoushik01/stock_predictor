# Project Apex: engineering and model validation

## Code-level improvements present

- `backend/app/main.py` uses configured `ALLOWED_ORIGINS` without wildcard browser origins.
- `backend/app/database.py` adds missing processed-feature columns without dropping stored tables.
- The repository CI checks Python syntax, database contract regression tests and the Next.js frontend build.
- An earlier global TLS verification bypass was removed. Do not reintroduce it to work around vendor connection errors.

## Requirements before exposing as a service

1. Review external stock and market-data licensing, timestamps, errors and HTTP request behavior.
2. Add authenticated user access, request throttling, file size/CSV row limits and isolation of user data.
3. Test slow or failing upstream providers, concurrent computations, invalid ticks and unexpected market closures.
4. Train and test using forward-in-time splits; measure leakage, baseline alternatives and sensitivity to financial regime changes.
5. Include realistic trading costs, slippage, data quality changes and uncertainty calibration in backtests.
6. Record data version, model hash, training settings and reproducible evaluation commands.
7. Set up deployment-specific backups, logs, secret storage and rollback.

No result here establishes predictable profits or live-market reliability. An automated build is not evidence of valid forecasts.
