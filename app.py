from pathlib import Path
import pandas as pd
import streamlit as st

# app.py is at the repository root, alongside results/.
ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
PRED = TABLES / "06_FINAL_common_sample_predictions.csv"
METRICS = TABLES / "06_FINAL_common_sample_metrics.csv"

NAMES = {
    "Stock_Level_ML_Aggregated": "Pooled stock ML",
    "Selected_ML_WalkForward": "Direct ML",
    "GARCH_11": "GARCH",
    "HAR_RV": "HAR",
    "EWMA_094": "EWMA",
    "Persistence_21D": "Persistence",
    "ARIMA_101": "ARIMA",
    "Historical_Matched": "Historical matched",
    "Naive_h_Target_Lag": "Lagged target",
}

st.set_page_config(page_title="ASX Materials Volatility Lab", page_icon="📈", layout="wide")
st.title("ASX Materials | Volatility Lab")
st.caption("Historical out-of-sample research · Not a live trading system")
st.info(
    "Archived 2025+ research forecasts only. Buy/Hold/Sell signals are not "
    "available until a separate return model is validated."
)

@st.cache_data
def load():
    return pd.read_csv(PRED, parse_dates=["Date"]), pd.read_csv(METRICS)

try:
    p, m = load()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
    st.error(f"Missing or invalid results CSV: {e}")
    st.stop()

horizons = sorted(p.Horizon.dropna().astype(int).unique().tolist())
h = st.sidebar.selectbox(
    "Forecast horizon (trading days)",
    horizons,
    index=horizons.index(21) if 21 in horizons else 0,
)
models = [c for c in NAMES if c in p.columns]
selected = st.sidebar.multiselect(
    "Compare forecasts",
    models,
    default=[
        c for c in [
            "Stock_Level_ML_Aggregated",
            "Selected_ML_WalkForward",
            "GARCH_11",
        ] if c in models
    ],
    format_func=lambda c: NAMES[c],
)
d = p.loc[p.Horizon.eq(h)].sort_values("Date").copy()
mm = m.loc[m.Horizon.eq(h)].sort_values("QLIKE").copy()
if d.empty:
    st.warning("No observations for selected horizon")
    st.stop()

st.caption(
    f"Forecast origins: {d.Date.min():%d %b %Y} to {d.Date.max():%d %b %Y}. "
    "Actual forward volatility is known only after the horizon completes."
)
a, b, c = st.columns(3)
a.metric("Historical forecasts", f"{len(d):,}")
b.metric(
    "Best model by QLIKE",
    NAMES.get(str(mm.iloc[0].Model), str(mm.iloc[0].Model)) if len(mm) else "—",
)
c.metric("Best QLIKE", f"{mm.iloc[0].QLIKE:.4f}" if len(mm) else "—")

st.subheader("Predicted vs actual forward realised volatility")
chart = (
    d.set_index("Date")[["Actual"] + selected]
    .apply(pd.to_numeric, errors="coerce")
    .mul(100)
    .rename(columns={"Actual": "Actual forward RV", **NAMES})
)
st.line_chart(chart, height=380)
st.caption(
    "Annualised volatility, %. These forecasts are historical evaluations, "
    "not current forecasts."
)

st.subheader("Model comparison")
if len(mm):
    table = mm[["Model", "N", "QLIKE", "MAE", "RMSE"]].copy()
    table["Model"] = table.Model.map(lambda x: NAMES.get(x, x))
    st.dataframe(table, hide_index=True, use_container_width=True)
st.caption(
    "Lower QLIKE, MAE and RMSE indicate smaller errors. "
    "MAE/RMSE are expressed in annualised volatility units."
)
st.download_button(
    "Download historical predictions CSV",
    d.to_csv(index=False).encode(),
    file_name=f"materials_h{h}_historical_predictions.csv",
    mime="text/csv",
)
st.divider()
st.subheader("Stock recommendations | Planned")
st.warning("No Buy/Hold/Sell recommendations are calculated or displayed.")
st.write(
    "Next phase: train and validate a separate 21-trading-day stock return "
    "model, combine out-of-sample return and volatility forecasts, test "
    "transaction costs and signal thresholds, then schedule daily inference."
)
