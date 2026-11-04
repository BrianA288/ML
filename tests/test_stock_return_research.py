from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stock_return_research import prepare, mature_training


def sample():
    days = pd.bdate_range("2020-01-01", periods=80)
    return pd.DataFrame([
        {"Date": date, "Yahoo_Ticker": ticker, "Stock_Return": ret,
         "Known_Feature": idx, "Target_Stock_RV_21": 9999}
        for idx, date in enumerate(days)
        for ticker, ret in [("AAA.AX", .01), ("BBB.AX", -.005)]
    ])


def test_forward_returns_and_no_leakage():
    df, features = prepare(sample())
    stock = df[df.Yahoo_Ticker.eq("AAA.AX")]
    assert abs(stock.Target_21D_Return.iloc[0] - (1.01**21 - 1)) < 1e-10
    assert stock.Target_21D_Return.tail(21).isna().all()
    assert "Target_Stock_RV_21" not in features
    assert "Target_21D_Return" not in features
    assert "Known_Feature" in features


def test_training_labels_mature_before_forecast():
    d, _ = prepare(sample())
    dates = pd.DatetimeIndex(sorted(d.Date.unique()))
    tr, cutoff = mature_training(d, dates, dates[60])
    assert cutoff == dates[38]
    assert tr.Date.max() <= cutoff


def test_missing_future_session_is_not_compressed():
    d = sample()
    missing = d.Date.unique()[13]
    d.loc[d.Date.eq(missing) & d.Yahoo_Ticker.eq("AAA.AX"), "Stock_Return"] = float("nan")
    result, _ = prepare(d)
    assert pd.isna(result.loc[result.Yahoo_Ticker.eq("AAA.AX"), "Target_21D_Return"].iloc[0])
