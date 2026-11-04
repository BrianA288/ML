# Stock-return research extension

This is an experimental 21-trading-day **stock return** model, separate from the original volatility project. It uses the existing `data/processed/stock_level_features_h21.parquet` and produces archived, held-out forecasts for analysis.

Run locally:

```bash
python -m pip install pandas numpy pyarrow scikit-learn pytest
python -m pytest -q tests/test_stock_return_research.py
python experiments/stock_return_research.py
```

Alternatively, on GitHub choose **Actions → Historical stock-return research → Run workflow**. Download the CSV artifact once finished. The workflow runs manually on `main` and automatically on commits to the dedicated repair branch. It neither commits generated CSV files nor updates live market data.

## Experimental procedure

- **Target:** compounded individual-stock return from **adjusted closes in the full raw OHLCV panel** at `t` and `t+21`; require prices at every intervening market session. Leaving the dynamic top-50 universe does not truncate an otherwise valid stock target.
- **Features:** numeric pre-existing stock features, excluding stock returns and all target columns. Feature-timing assumptions are inherited from the original notebooks and must be independently audited before operational use.
- **Model selection:** evaluate Ridge and histogram gradient boosting on 2023 and 2024; rank by mean validation MAE.
- **Held-out test:** dated 2025 onward, expanding-window refit every 21 forecast origins, and a 21-session label-maturity gap.
- **Metrics:** MAE, RMSE, directional accuracy; zero-return baseline implemented. Add a rolling historical-mean and momentum benchmark before considering trading signals.
- **Important:** Validation includes overlapping forward labels; additional leakage, turnover, backtest and threshold tests are required. Predictions only cover **matured historical** observations, not the latest unseen 21 days.

The current Streamlit dashboard continues to show archived volatility research. Do not interpret these return predictions as validated Buy/Hold/Sell signals.

## Verified held-out run (4 November 2026)

[GitHub Actions run 41823910311](https://github.com/BrianA288/ML/actions/runs/41823910311) **passed** after correcting pandas 3 compatibility. Its four synthetic tests passed, and the full historical evaluation completed.

- Selected model: `HGB_leaf50`; 20,700 historical stock/date predictions.
- Model MAE: **0.11941**; zero-return MAE: **0.11635**.
- Model RMSE: **0.16220**; zero-return RMSE: **0.15736**.
- Directional accuracy: **50.86%**.

**Conclusion:** The model underperforms the zero-return benchmark on both MAE and RMSE, so it is **not approved for Buy/Hold/Sell signals**. These percentages are calculated across overlapping historical 21-session outcomes, not an independently deployable live trading system. The dataset acquisition and individual feature release timing still require additional audit.
