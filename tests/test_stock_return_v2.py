from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stock_return_v2 import augment, prepare_v2, predict_candidate
from experiments.stock_return_research import mature_training


def samples():
    dates = pd.bdate_range("2020-01-01", periods=110)
    price_rows, feature_rows = [], []
    for i, date in enumerate(dates):
        for ticker, step in [("AAA.AX", 0.01), ("BBB.AX", -0.005)]:
            price_rows.append({"Date": date, "Yahoo_Ticker": ticker,
                               "Adj_Close": 100 * (1 + step) ** i})
            if ticker == "AAA.AX" and 8 <= i <= 25:
                continue
            feature_rows.append({"Date": date, "Yahoo_Ticker": ticker,
                                 "Stock_Return": step, "Known_Factor": float(i),
                                 "Target_Stock_RV_21": 10000,
                                 "Weight": 0.5})
    return pd.DataFrame(feature_rows), pd.DataFrame(price_rows)


def test_momentum_uses_only_past_prices():
    f, p = samples()
    z = augment(f, p).set_index(["Date", "Yahoo_Ticker"])
    date = sorted(f.Date.unique())[70]
    assert np.isclose(z.loc[(date, "AAA.AX"), "Momentum_21D"], 1.01 ** 21 - 1)
    assert np.isclose(z.loc[(date, "BBB.AX"), "Momentum_21D"], .995 ** 21 - 1)
    # Manipulating only later prices must never affect an earlier signal.
    p.loc[p.Date.gt(date), "Adj_Close"] *= 200
    changed = augment(f, p).set_index(["Date", "Yahoo_Ticker"])
    assert np.isclose(changed.loc[(date, "AAA.AX"), "Momentum_21D"],
                      z.loc[(date, "AAA.AX"), "Momentum_21D"])


def test_full_price_target_and_exclusions():
    f, p = samples()
    d, feature_sets = prepare_v2(f, p)
    first = d.loc[d.Yahoo_Ticker.eq("AAA.AX")].iloc[0]
    assert np.isclose(first.Target_21D_Return, 1.01 ** 21 - 1)
    for cols in feature_sets.values():
        assert "Target_Stock_RV_21" not in cols
        assert "Target_21D_Return" not in cols
        assert "Stock_Return" not in cols
        assert "Weight" not in cols


def test_maturity_and_basic_baselines():
    f, p = samples()
    d, feature_sets = prepare_v2(f, p)
    sessions = pd.DatetimeIndex(sorted(d.Date.unique()))
    tr, cutoff = mature_training(d, sessions, sessions[80])
    assert cutoff == sessions[58]
    te = d.loc[d.Date.eq(sessions[80])]
    assert len(predict_candidate("Zero", tr, te, feature_sets)) == len(te)
    assert np.all(np.isfinite(predict_candidate("Historical_Mean", tr, te, feature_sets)))
    assert np.all(np.isfinite(predict_candidate("Momentum_21D", tr, te, feature_sets)))
