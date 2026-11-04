# ASX Materials volatility dashboard

This is a historical research dashboard, not a live recommendation engine.

Copy `dashboard/` into the root of the existing `BrianA288/ML` repository, which contains the required `results/tables/06_FINAL_common_sample_predictions.csv` and `06_FINAL_common_sample_metrics.csv` files.

Run locally:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

To deploy: commit the dashboard folder to GitHub, then use Streamlit Community Cloud (https://share.streamlit.io/), select the repository and set the main file path to `dashboard/app.py`.

A daily live recommendation system will require a separate return model, a scheduled data/forecast job and point-in-time backtesting. No buy/sell signals are inferred from volatility alone.
