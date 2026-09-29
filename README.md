# Integrated Australian Materials Volatility Forecasting

This repository contains the **actual notebooks, datasets, diagnostics, figures and result tables** for the materials sector volatility forecasting project. The project compares conventional models with direct sector and pooled stock-level machine learning at 5, 21 and 63 trading day horizons. The primary result uses the 21 trading day horizon and QLIKE.

## Open in Google Colab

In a new Colab notebook, run:

```python
!git clone https://github.com/BrianA288/ML.git /content/ML
%cd /content/ML
!pip -q install -r requirements.txt
```

Then open a notebook from the `notebooks/` folder in Colab. Before running its other cells in a new runtime, set its working directory:

```python
%cd /content/ML
```

The notebooks use paths relative to the repository root. Colab runtimes are temporary, so clone and change directory again after a runtime reset. You can also open an individual notebook via GitHub's **Open in Colab** option, but still run the setup above in that runtime first.

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
