# Project State

**Last Updated:** 2024-05-24

## Completed
- Initialised project skeleton matching the exact folder structure specified in `MASTER_CONTEXT.md`.
- Configured `pyproject.toml` using `uv` with all required dependencies (`pandas`, `pandas-ta`, `ccxt`, `yfinance`, `duckdb`, `scikit-learn`, `xgboost`, `lightgbm`, `vectorbt`, `mlflow`, `optuna`, `pytest`, `ruff`).
- Updated `.gitignore` to exclude `.env`, `data/`, `mlruns/`, `__pycache__/`, and other standard artefacts.
- Created a reusable pytest helper (`tests/test_no_lookahead.py`) to enforce strict temporal boundaries and prevent lookahead bias in feature calculations.
- **Implemented `src/data_engine/crypto_fetcher.py`**: Full OHLCV ingestion pipeline with ccxt pagination, clock-skew alignment, duplicate removal, gap classification (outage/illiquid/data_error), and outlier flagging (magnitude & OHLC integrity). Includes atomic parquet storage to prevent corruption.
- **Implemented `src/data_engine/macro_fetcher.py`**: Macro-economic data ingestion via yfinance (with deterministic scale corrections for yield indices) and CoinGecko for BTC dominance. Handles retries, timezone alignment, and atomic parquet merging.
- **Implemented comprehensive unit tests** (`tests/test_data_engine.py`) covering gap classification logic, outlier detection, OHLC integrity checks, and strict no-lookahead temporal boundary enforcement.
- **Implemented live smoke tests** (`tests/test_binance_real_data.py`) to validate the full pipeline against real Binance network data, checking history depth, index monotonicity, and cleaning/classification sanity checks.

## Current Issues
- Feature engineering, predictive core, backtesting scaffold, risk/sizing, and monitoring modules remain as empty placeholders awaiting implementation.
- Live smoke tests require manual execution (`pytest -v -s`) and are not integrated into CI due to network dependency and rate limits.
- No automated scheduling or cron-job integration is yet in place for the daily macro/CoinGecko data appends.

## Next Steps
1. **Feature Engineering**: Develop no-lookahead feature calculators using `pandas-ta`, validated against the temporal boundary helper.
2. **Predictive Core**: Build the k-NN analog engine, gradient boosting models (XGBoost/LightGBM), and linear ensemble meta-model.
3. **Backtesting Scaffold**: Set up walk-forward validation using `vectorbt`, ensuring train/validation/test folds never overlap.
4. **Macro/MTF/Volatility Logic**: Implement regime classification, multi-timeframe alignment scoring, and volatility state detection per the exact tables in `MASTER_CONTEXT.md`.
5. **Risk & Sizing**: Code quarter-Kelly sizing with hard execution-layer caps (2% single-trade, 6% correlated, 10% aggregate).
6. **Monitoring & Kill-Switches**: Develop live monitoring for calibration drift, feature drift, and unconditional halt triggers requiring manual sign-off.
7. **Testing & Validation**: Expand `pytest` coverage to include regime classification accuracy, risk gate enforcement, and backtesting logic validation.
