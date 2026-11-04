# Stock-return v2: momentum and benchmark research

This experiment is **separate from the original assignment**. It is for **historical research only** and produces no live prices, Buy/Hold/Sell signals or trading instructions.

## Procedure

- Compute 21-market-session adjusted-price returns from the full historical stock-price panel, not only top-50 membership dates.
- Generate strictly past-or-present 5/21/63-session momentum, relative 21-session momentum and trailing 21-session volatility.
- Compare existing numeric features, expanded momentum features, and momentum-only inputs, with Ridge and histogram gradient boosting.
- Evaluate zero-return, ticker-specific historical-mean and naive 21-session momentum forecast baselines.
- Select using **2023 and 2024 chronological validation only** (minimum mean absolute error averaged by validation year).
- Apply the selected candidate in a **2025+ held-out evaluation**, using mature-label training cutoffs and refits every 21 origins for learned models.
- Save validation results, held-out forecasts, and a comparison summary to `results/tables/09_return_v2_*.csv` (workflow artifact). These files are not committed into the original project results.
- Point-in-time feature availability beyond the computed momentum inputs remains unaudited; this is not suitable for live recommendations.

## Execute

```bash
python -m pip install numpy pandas pyarrow scikit-learn pytest
python -m pytest -q tests/test_stock_return_research.py tests/test_stock_return_v2.py
python -m experiments.stock_return_v2
```

Run GitHub Actions workflow **Stock return v2 research comparison** manually to download the CSV artifact.

## Verified GitHub results — 4 November 2026

[Successful full-data run](https://github.com/BrianA288/ML/actions/runs/41824890620).

Selected validation candidate: **Zero** (predict 0% 21-day return). Held-out results over **20,700 overlapping stock/date forecasts**:

| Measure | Validation-selected zero baseline | Naive unscaled 21-session momentum |
| --- | ---: | ---: |
| MAE | 0.116346 | 0.209928 |
| RMSE | 0.157363 | 1.876542 |
| Directional accuracy | 42.95% | 50.90% |

Direction accuracy for the zero baseline represents the share of nonpositive realised returns, **not** a good directional classifier. The extreme error from unscaled momentum is an additional reason to audit corporate-action adjustments, outliers and price quality. These forecasts overlap substantially and the test metrics do not imply profitable trading.

**Decision:** No tested ML candidate beat zero return on validation MAE. No trading recommendations should be generated from these models. Future research should prioritise economically meaningful cross-sectional ranking, directional classification, robust price-data outlier audits and trading-cost-aware paper backtests, with 2025+ reserved for evaluation.
