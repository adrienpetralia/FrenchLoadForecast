# French Load Curve Analysis

Short-term French electricity load forecasting.

## Outline 📝

This repository contains the **source code**,  **notebooks**, and **scripts** to:

- Explore half-hourly French load data (RTE) and daily temperature data.
- Train different forecasters (ARIMAX, Linear Regression, GAM, CatBoost).
- Build an ensembler prediction pipeline using OPERA expert aggregation.
- Reproduce results and visualize baselines performance.

---

### Getting Started 🚀

To install the dependencies, you can use the following commands.

```bash
pip install uv
git clone https://github.com/adrienpetralia/FrenchLoadForecast
cd FrenchLoadForecast
uv sync
```

---

### Data 📦

Place data under ```./data/``` (2018 to 2024):

```
data/
├── load_rte/   # RTE eCO2mix (half-hourly, xls)
└── temp/       # Temperature (daily tmin/tmax/tmean, CSV)
```

- **RTE load**: half-hourly power (MW). Place the downloaded xls files and preprocess them using:
```
uv run -m scripts.process_rte_file
```

- Temperature: rename the corresponding temperature columns as 'tmean', 'tmin','tmax' and place the CSV in ```data/temp/```.

---

### Notebooks 📒

1. Data Analysis ```notebooks/data_analysis.ipynb```
    - Data exploration: visualizes seasonality, weekly/daily patterns, temperature–load relationships, etc.

2. Interactive Training ```notebooks/interactive_training.ipynb```
    - Train the different baselines (ARIMAX / Linear / GAM / CatBoost).
    - Live plots of predictions, residuals, and metrics.

---

### Code Structure 📁

```
.
├── data                           # data folder
│   ├── load_rte                   # load data files from RTE
│   └── temp                       # french temperature folder
├── notebooks
│   ├── data_analysis.ipynb        # data presentation and analysis
│   └── interactive_training.ipynb # interactive training & results visualization
├── outputs                        # experiments outputs (predictions, metrics)
├── scripts
│   ├── process_rte_file.py        # process raw RTE files -> standardized format
│   └── run_experiment_pipeline.py # end-to-end training/eval/ensembling
├── src
│   ├── ensembler                  # ensembling (incl. OPERA-based stacking)
│   ├── forecasters                # ARIMAX, Linear, GAM, CatBoost
│   └── utils                      # IO, feature engineering, helpers
└── pyproject.toml                 # project setup
```

---

### Launch an Experiment ⚙️

To run the experiments, use the command below (via uv and using module):
```
uv run -m scripts.run_experiment_pipeline \
  --load-dir data/load_rte \
  --temp-dir data/temp \
  --train-from 2018 --train-to 2023 \
  --test-from 2023 --test-to 2024 \
  --date-start 2024-01-01 --date-end 2024-12-31 \
  --models arimax linreg gam catboost \
  --ensemble MLpol \
  --outdir outputs/ --prefix rte_run
```
