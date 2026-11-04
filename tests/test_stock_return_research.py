from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stock_return_research import prepare, mature_training


def sample():
    dates = pd.bdate_range("2020-01-01", periods=80)
    rows, prices = [], []
    for i, date in enumerate(dates):
        for ticker, daily in [("AAA.AX", .01), ("BBB.AX", -.005)]:
            prices.append({"Date": date, "Yahoo_Ticker": ticker,
                           "Adj_Close": 100 * (1 + daily) ** i})
            # AAA leaves the top 50, while its full prices remain available.
            if ticker == "AAA.AX" and 5 <= i <= 30:
                continue
            rows.append({"Date": date, "Yahoo_Ticker": ticker,
                         "Stock_Return": daily, "Known_Feature": float(i),
                         "Target_Stock_RV_21": 9999})
    return pd.DataFrame(rows), pd.DataFrame(prices)


def test_forward_returns_after_leaving_top50():
    features, prices = sample()
    df, cols = prepare(features, prices)
    a = df.loc[df.Yahoo_Ticker.eq("AAA.AX")].set_index("Date")
    assert abs(a.Target_21D_Return.iloc[0] - (1.01 ** 21 - 1)) < 1e-10
    assert a.Target_21D_Return.tail(21).isna().all()
    assert "Target_Stock_RV_21" not in cols and "Target_21D_Return" not in cols
    assert "Known_Feature" in cols


def test_missing_price_invalidates_horizon():
    features, prices = sample()
    date = sorted(prices.Date.unique())[13]
    prices.loc[prices.Date.eq(date) & prices.Yahoo_Ticker.eq("AAA.AX"), "Adj_Close"] = np.nan
    df, _ = prepare(features, prices)
    a = df[df.Yahoo_Ticker.eq("AAA.AX")].set_index("Date")
    assert pd.isna(a.Target_21D_Return.iloc[0])


def test_label_maturity_gap():
    features, prices = sample()
    d, _ = prepare(features, prices)
    dates = pd.DatetimeIndex(sorted(d.Date.unique()))
    tr, cutoff = mature_training(d, dates, dates[60])
    assert cutoff == dates[38]
    assert tr.Date.max() <= cutoff


def test_price_source_must_be_unique():
    features, prices = sample()
    duplicate = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    try:
        prepare(features, duplicate)
    except ValueError as exc:
        assert "Duplicate full-price" in str(exc)
    else:
        raise AssertionError("Duplicate price rows accepted")
