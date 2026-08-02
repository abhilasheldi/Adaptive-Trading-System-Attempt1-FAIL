# Project State

## Completed
- Initialised project skeleton matching the exact folder structure specified in `MASTER_CONTEXT.md`.
- Configured `pyproject.toml` using `uv` with all required dependencies (`pandas`, `pandas-ta`, `ccxt`, `yfinance`, `duckdb`, `scikit-learn`, `xgboost`, `lightgbm`, `vectorbt`, `mlflow`, `optuna`, `pytest`, `ruff`).
- Updated `.gitignore` to exclude `.env`, `data/`, `mlruns/`, `__pycache__/`, and other standard artefacts.
- Created a reusable pytest helper (`tests/test_no_lookahead.py`) to enforce strict temporal boundaries and prevent lookahead bias in feature calculations.

## Current Issues
- All source modules (`src/data_engine`, `src/features`, `src/predictive_core`, etc.) are currently empty placeholders awaiting implementation.
- No data ingestion or normalisation pipelines are active yet.
- Testing suite is minimal, limited to the temporal boundary helper.

## Next Steps
1. **Data Engine**: Implement ingestion logic for crypto OHLCV (`ccxt`) and macro data (`yfinance`, `FRED`), storing raw/normalised data in `duckdb` with strict temporal partitioning.
2. **Feature Engineering**: Develop no-lookahead feature calculators using `pandas-ta`, validated against the new pytest helper.
3. **Predictive Core**: Build the k-NN analog engine, gradient boosting models (XGBoost/LightGBM), and linear ensemble meta-model.
4. **Backtesting Scaffold**: Set up walk-forward validation using `vectorbt`, ensuring train/validation/test folds never overlap.
5. **Macro/MTF/Volatility Logic**: Implement regime classification, multi-timeframe alignment scoring, and volatility state detection per the exact tables in `MASTER_CONTEXT.md`.
6. **Risk & Sizing**: Code quarter-Kelly sizing with hard execution-layer caps (2% single-trade, 6% correlated, 10% aggregate).
7. **Monitoring & Kill-Switches**: Develop live monitoring for calibration drift, feature drift, and unconditional halt triggers requiring manual sign-off.
8. **Testing & Validation**: Expand `pytest` coverage to include temporal-boundary checks, risk gate enforcement, and regime classification accuracy.
