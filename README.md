# Integrated Australian Materials Volatility Forecasting

This repository contains the **actual notebooks, datasets, diagnostics, figures and result tables** for the materials sector volatility forecasting project. The project compares conventional models with direct sector and pooled stock-level machine learning at 5, 21 and 63 trading day horizons. The primary result uses the 21 trading day horizon and QLIKE.

## Standalone local setup

Use Python 3.11 or newer with a local Jupyter environment. Download the repository ZIP and extract it, or clone it:

```bash
git clone https://github.com/BrianA288/ML.git
cd ML
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: ` .\.venv\Scripts\Activate.ps1 `
- macOS/Linux: `source .venv/bin/activate`

Then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python prepare_colab.py
python -m jupyterlab
```

Open the numbered notebooks in `notebooks/` and run their cells in order. The setup helper restores the 31.8 MB interim equity panel from its two bundled parts at `data/interim/materials_equity_panel.parquet` and verifies its SHA-256 checksum. Its existing filename is retained to preserve the file path.

The notebooks resolve project paths from either the repository root or the `notebooks/` directory. Keep the bundled folder structure intact. The project runs locally; no hosted notebook service or cloud runtime is required.

## Data access and saved results

The repository includes the raw data, processed features and saved results. Notebook 0 retrieves equity data from Yahoo Finance, and notebook 2 retrieves external market and economic data from Yahoo Finance, ABS and RBA; those acquisition steps require internet access. Updating those inputs can change subsequent forecasts and results.

To work from the saved processed datasets, run notebooks 3–7. To inspect the final results and regenerate report figures, start with notebook 7. Install dependencies before starting; model fitting and report generation then use the bundled local files.

## Repository layout

- `notebooks/` — executable research notebooks, numbered in workflow order.
- `source_data/` — ASX ticker and acquired macro inputs.
- `data/raw/` — historical shares and OHLCV prices.
- `data/interim/` — equity panel.
- `data/processed/` — sector and stock feature datasets.
- `results/diagnostics/` — data, timing and model audits.
- `results/tables/` — forecasts and evaluation metrics.
- `results/figures/` — report charts.

## Notebook order

0. `00_Data_Acquisition.ipynb`
1. `01_Sector_Construction_and_EDA.ipynb`
2. `02_Macro_Data_and_Feature_Engineering.ipynb`
3. `03_Traditional_Volatility_Models.ipynb`
4. `04_Direct_Sector_Machine_Learning.ipynb`
5. `05_Pooled_Stock_ML_and_Aggregation.ipynb`
6. `06_Final_Common_Walk_Forward_Evaluation.ipynb`
7. `07_Final_Results_and_Report_Figures.ipynb`

To inspect the final results without rerunning acquisition and training, start with notebook 7 and the existing `results/tables/06_FINAL_common_sample_metrics.csv` and `results/tables/07_primary_21d_report_table.csv`. Running all notebooks can take substantial time and live market data may have changed since these saved results were generated. The original study uses a top-50 materials universe and shared evaluation dates. The stock-level ML branch refits every 21 origins, while direct sector ML refits daily in final walk-forward evaluation.
