"""Experimental 21-session stock-return forecasting and baselines, never live signals.

Evaluation is on matured historical data. Existing project/assignment results unchanged.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .stock_return_research import forward_returns, mature_training, scores, H, BASE

TECH = ["Momentum_5D", "Momentum_21D", "Momentum_63D",
        "Relative_Momentum_21D", "Trailing_Volatility_21D"]
BAD = ("target_", "future_", "forward_", "label_", "prediction_", "predicted_")


def augment(source, price_history):
    """Derive lagged-window prices only from observations at/before each origin."""
    features = source.copy()
    features["Date"] = pd.to_datetime(features["Date"])
    prices = price_history[["Date", "Yahoo_Ticker", "Adj_Close"]].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    if prices.duplicated(["Date", "Yahoo_Ticker"]).any():
        raise ValueError("Duplicate adjusted prices")
    calendar = pd.DatetimeIndex(sorted(features.Date.unique()))
    close = prices.pivot(index="Date", columns="Yahoo_Ticker", values="Adj_Close").reindex(calendar)
    close.index.name = "Date"
    close = close.where(np.isfinite(close) & close.gt(0))
    daily = close.pct_change(fill_method=None)
    momentum = {}
    for n in (5, 21, 63):
        valid = close.notna().rolling(n + 1, min_periods=n + 1).sum().eq(n + 1)
        momentum[f"Momentum_{n}D"] = (close.div(close.shift(n)).sub(1)).where(valid)
    vol = (daily.pow(2).rolling(21, min_periods=21).mean() * 252).pow(.5)
    momentum["Trailing_Volatility_21D"] = vol
    # The relative signal is calculated within the contemporaneously observed
    # feature universe, with no future membership or future returns.
    wanted = features[["Date", "Yahoo_Ticker"]].drop_duplicates()
    for name, matrix in momentum.items():
        long = matrix.stack(future_stack=True).rename(name).reset_index()
        wanted = wanted.merge(long, on=["Date", "Yahoo_Ticker"], how="left", validate="one_to_one")
    wanted["Relative_Momentum_21D"] = (
        wanted["Momentum_21D"]
        - wanted.groupby("Date")["Momentum_21D"].transform("median")
    )
    return features.merge(wanted, on=["Date", "Yahoo_Ticker"], how="left", validate="one_to_one")


def prepare_v2(source, prices):
    d = augment(source, prices)
    d = forward_returns(d, prices)
    d = d.sort_values(["Date", "Yahoo_Ticker"]).reset_index(drop=True)
    numeric = d.select_dtypes(include="number").columns
    # Defence in depth: never train on any label or raw future columns.
    exclude = {"Stock_Return", "Weight", "MCap_Rank", "Denominator",
               "Contribution", "Lagged_MCap", "Market_Cap_Reconstructed"}
    prior = [c for c in numeric if c not in TECH and c not in exclude
             and not any(c.lower().startswith(b) for b in BAD)]
    expanded = prior + [c for c in TECH if c in d]
    if not prior or len(expanded) == len(prior):
        raise ValueError("Missing feature columns or technical signals")
    return d, {"Original": prior, "OriginalPlusMomentum": expanded,
               "MomentumOnly": TECH}


def make_model(name):
    if name == "Ridge_100":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("m", Ridge(alpha=100))])
    if name == "HGB_leaf50":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", HistGradientBoostingRegressor(
                             max_iter=120, learning_rate=.05,
                             min_samples_leaf=50, random_state=42))])
    raise ValueError(name)


def predict_candidate(name, tr, te, feature_sets):
    if name == "Zero":
        return np.zeros(len(te))
    if name == "Momentum_21D":
        return te["Momentum_21D"].fillna(0).to_numpy(float)
    if name == "Historical_Mean":
        fallback = tr.Target_21D_Return.mean()
        means = tr.groupby("Yahoo_Ticker").Target_21D_Return.mean()
        return te.Yahoo_Ticker.map(means).fillna(fallback).to_numpy(float)
    model_name, feature_name = name.split("__", 1)
    model = make_model(model_name)
    columns = feature_sets[feature_name]
    model.fit(tr[columns], tr.Target_21D_Return)
    return model.predict(te[columns])


def candidates():
    return ["Zero", "Historical_Mean", "Momentum_21D"] + [
        f"{name}__{feat}"
        for name in ["Ridge_100", "HGB_leaf50"]
        for feat in ["Original", "OriginalPlusMomentum", "MomentumOnly"]
    ]


def run(source, prices, output, refit_every=21):
    d, feature_sets = prepare_v2(source, prices)
    sessions = pd.DatetimeIndex(sorted(d.Date.unique()))
    val_rows = []
    for year in [2023, 2024]:
        tr, cutoff = mature_training(d, sessions, pd.Timestamp(f"{year}-01-01"))
        va = d.loc[d.Date.dt.year.eq(year) & d.Target_21D_Return.notna()]
        if tr.empty or va.empty:
            raise ValueError(f"Missing year {year} train/validation observations")
        for name in candidates():
            prediction = predict_candidate(name, tr, va, feature_sets)
            val_rows.append({"Year": year, "Candidate": name, "Cutoff": cutoff,
                             **scores(va.Target_21D_Return, prediction)})
    validation = pd.DataFrame(val_rows)
    best = validation.groupby("Candidate").MAE.mean().sort_values().index[0]
    test = d.loc[d.Date.ge("2025-01-01") & d.Target_21D_Return.notna()].copy()
    dates = sorted(test.Date.unique())
    if not dates:
        raise ValueError("No historical held-out labels")
    frames = []
    model = None
    selected_features = None
    # Simple baselines and trained ML predictions, refit on the same 21-origin schedule.
    for j, date in enumerate(dates):
        if j % refit_every == 0 or model is None:
            tr, cutoff = mature_training(d, sessions, date)
            if "__" in best:
                mn, fn = best.split("__", 1)
                selected_features = feature_sets[fn]
                model = make_model(mn)
                model.fit(tr[selected_features], tr.Target_21D_Return)
            else:
                model = best
                if best == "Historical_Mean":
                    means = tr.groupby("Yahoo_Ticker").Target_21D_Return.mean()
                    mean_fallback = tr.Target_21D_Return.mean()
        te = test.loc[test.Date.eq(date)].copy()
        if best == "Zero":
            pred = np.zeros(len(te))
        elif best == "Momentum_21D":
            pred = te.Momentum_21D.fillna(0).to_numpy(float)
        elif best == "Historical_Mean":
            pred = te.Yahoo_Ticker.map(means).fillna(mean_fallback).to_numpy(float)
        else:
            pred = model.predict(te[selected_features])
        frame = te[["Date", "Yahoo_Ticker", "Target_21D_Return"]].copy()
        frame["Predicted_21D_Return"] = pred
        frame["Zero_Prediction"] = 0.0
        frame["Momentum_Prediction"] = te.Momentum_21D.fillna(0).to_numpy(float)
        frame["Training_Cutoff"] = cutoff
        frame["Candidate"] = best
        frames.append(frame)
    preds = pd.concat(frames, ignore_index=True)
    y = preds.Target_21D_Return
    result = {"Selected_Candidate": best, "N": len(preds),
              **scores(y, preds.Predicted_21D_Return)}
    for name, arr in [("Zero", preds.Zero_Prediction),
                      ("Momentum", preds.Momentum_Prediction)]:
        for k, v in scores(y, arr).items():
            if k != "N":
                result[f"{name}_{k}"] = v
    result["Beats_Zero_MAE"] = bool(result["MAE"] < result["Zero_MAE"])
    result["Beats_Zero_RMSE"] = bool(result["RMSE"] < result["Zero_RMSE"])
    output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output / "09_return_v2_validation.csv", index=False)
    preds.to_csv(output / "09_return_v2_test_predictions.csv", index=False)
    pd.DataFrame([result]).to_csv(output / "09_return_v2_summary.csv", index=False)
    print(result, flush=True)
    return validation, preds, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BASE / "data/processed/stock_level_features_h21.parquet")
    parser.add_argument("--prices", type=Path, default=BASE / "data/raw/materials_ohlcv.parquet")
    parser.add_argument("--output", type=Path, default=BASE / "results/tables")
    args = parser.parse_args()
    run(pd.read_parquet(args.input), pd.read_parquet(args.prices), args.output)
