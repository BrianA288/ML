# Stock-return research extension

This is an experimental 21-trading-day **stock return** model, separate from the original volatility project. It uses the existing `data/processed/stock_level_features_h21.parquet` and produces archived, held-out forecasts for analysis.

Run locally:

```bash
python -m pip install pandas numpy pyarrow scikit-learn pytest
python -m pytest -q tests/test_stock_return_research.py
python experiments/stock_return_research.py
```

Alternatively, on GitHub choose **Actions → Historical stock-return research → Run workflow**. Download the CSV artifact once finished. The action is **manual only**. It neither commits generated CSV files nor updates live market data.

## Experimental procedure

- **Target:** compounded individual-stock return from sessions `t+1` through `t+21`; require all 21 subsequent market sessions.
- **Features:** numeric pre-existing stock features, excluding stock returns and all target columns. Feature-timing assumptions are inherited from the original notebooks and must be independently audited before operational use.
- **Model selection:** evaluate Ridge and histogram gradient boosting on 2023 and 2024; rank by mean validation MAE.
- **Held-out test:** dated 2025 onward, expanding-window refit every 21 forecast origins, and a 21-session label-maturity gap.
- **Metrics:** MAE, RMSE, directional accuracy; compare against zero-return and historical-mean baselines before considering trading signals.
- **Important:** Validation includes overlapping forward labels; additional leakage, turnover, backtest and threshold tests are required. Predictions only cover **matured historical** observations, not the latest unseen 21 days.

The current Streamlit dashboard continues to show archived volatility research. Do not interpret these return predictions as validated Buy/Hold/Sell signals.
