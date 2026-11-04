"""21-day stock-return research: archival out-of-sample predictions, not trade signals."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H = 21
BASE = Path(__file__).resolve().parents[1]
EXCLUDE = {"Date", "Yahoo_Ticker", "Stock_Return", "Target_Stock_RV_5",
           "Target_Stock_RV_21", "Target_Stock_RV_63"}


def forward_returns(data, horizon=H):
    """Compute compounded return over strictly the next h market sessions."""
    r = data.pivot(index="Date", columns="Yahoo_Ticker", values="Stock_Return").sort_index()
    lr = np.log1p(r.where(r > -1))
    future = [lr.shift(-j) for j in range(1, horizon + 1)]
    counts = sum(v.notna().astype(int) for v in future)
    target = np.expm1(sum(v.fillna(0) for v in future)).where(counts.eq(horizon))
    target = target.stack(dropna=False).rename("Target_21D_Return").reset_index()
    return data.merge(target, on=["Date", "Yahoo_Ticker"], validate="one_to_one")


def prepare(source):
    d = source.copy()
    required = {"Date", "Yahoo_Ticker", "Stock_Return"}
    if not required.issubset(d.columns):
        raise ValueError(f"Missing columns: {required - set(d.columns)}")
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values(["Date", "Yahoo_Ticker"]).reset_index(drop=True)
    if d.duplicated(["Date", "Yahoo_Ticker"]).any():
        raise ValueError("Duplicate ticker/date observations")
    d = forward_returns(d)
    features = [c for c in d.select_dtypes(include="number").columns
                if c not in EXCLUDE and not c.startswith("Target_")]
    if not features:
        raise ValueError("No predictor features available")
    return d, features


def models():
    return {
        "Ridge_100": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), ("model", Ridge(alpha=100))]),
        "HGB_leaf50": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=120, learning_rate=.05, min_samples_leaf=50,
                random_state=42))]),
    }


def mature_training(d, sessions, origin):
    pos = sessions.searchsorted(np.datetime64(origin))
    if pos <= H:
        raise ValueError("Not enough mature history")
    # Training label must finish strictly before the forecast-origin date.
    cutoff = sessions[pos - H - 1]
    return d.loc[d.Date.le(cutoff) & d.Target_21D_Return.notna()], cutoff


def scores(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return {"N": len(a), "MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "Directional_Accuracy": float(np.mean((a > 0) == (p > 0)))}


def run(source, output, refit_every=21):
    d, features = prepare(source)
    sessions = pd.DatetimeIndex(sorted(d.Date.unique()))
    validation = []
    for year in [2023, 2024]:
        tr, cutoff = mature_training(d, sessions, pd.Timestamp(f"{year}-01-01"))
        va = d.loc[d.Date.dt.year.eq(year) & d.Target_21D_Return.notna()]
        if tr.empty or va.empty:
            raise ValueError(f"Missing training or validation data for {year}")
        for name, model in models().items():
            model.fit(tr[features], tr.Target_21D_Return)
            p = model.predict(va[features])
            validation.append({"Year": year, "Model": name, "Training_Cutoff": cutoff,
                               **scores(va.Target_21D_Return, p)})
    val = pd.DataFrame(validation)
    selected = val.groupby("Model").MAE.mean().idxmin()
    # Historical held-out evaluation; this deliberately excludes unmatured dates.
    test = d.loc[d.Date.ge("2025-01-01") & d.Target_21D_Return.notna()]
    dates = sorted(test.Date.unique())
    if not dates:
        raise ValueError("No matured held-out test dates")
    rows, model = [], None
    for j, date in enumerate(dates):
        if model is None or j % refit_every == 0:
            tr, cutoff = mature_training(d, sessions, date)
            model = models()[selected]
            model.fit(tr[features], tr.Target_21D_Return)
        batch = test.loc[test.Date.eq(date), ["Date", "Yahoo_Ticker",
                                               "Stock_Return", "Target_21D_Return"]].copy()
        batch["Predicted_21D_Return"] = model.predict(
            d.loc[test.Date.eq(date) & d.Date.ge("2025-01-01"), features]
        ) if False else np.nan
        # Match prediction rows to the original feature rows by date and ticker.
        x = d.loc[d.Date.eq(date) & d.Target_21D_Return.notna()]
        batch["Predicted_21D_Return"] = model.predict(x[features])
        batch["Model"] = selected
        batch["Training_Cutoff"] = cutoff
        rows.append(batch)
    preds = pd.concat(rows, ignore_index=True)
    summary = {"Selected_Model": selected, "Horizon": H,
               "Validation_Years": "2023,2024", "Test_Start": "2025-01-01",
               "Refit_Every": refit_every,
               **scores(preds.Target_21D_Return, preds.Predicted_21D_Return)}
    output.mkdir(parents=True, exist_ok=True)
    val.to_csv(output / "08_return_validation.csv", index=False)
    preds.to_csv(output / "08_return_historical_predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "08_return_test_summary.csv", index=False)
    print(summary)
    return val, preds, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BASE / "data/processed/stock_level_features_h21.parquet")
    parser.add_argument("--output", type=Path, default=BASE / "results/tables")
    args = parser.parse_args()
    run(pd.read_parquet(args.input), args.output)
