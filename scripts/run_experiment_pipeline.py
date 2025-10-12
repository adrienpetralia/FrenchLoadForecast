#!/usr/bin/env python3
"""
Run daily-load forecasting pipeline.

Example:
  uv run -m scripts.run_experiment_pipeline \
    --load-dir data/load_rte \
    --temp-dir data/temp \
    --train-from 2018 --train-to 2023 \
    --test-from 2023 --test-to 2024 \
    --date-start 2024-01-01 --date-end 2024-12-31 \
    --models arimax linreg gam catboost \
    --ensemble BOA \
    --outdir outputs --prefix rte_2018_2024
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# --- Project imports (expect script to be run from repo root) ---
from src.utils.helpers import load_rte_data, load_temperature
from src.forecasters.gam import GAMConfig, GAMForecaster
from src.forecasters.arimax import ARIMAConfig, ARIMAForecaster
from src.forecasters.linear import LRConfig, LinearRegForecaster
from src.forecasters.catboost import CatBoostConfig, CatBoostForecaster
from src.ensembler.opera_ensembler import OperaStackConfig, OperaStackEnsembler


# ----------------------------- Utilities --------------------------------------
def overall_metrics(preds_df: pd.DataFrame, models: List[str] | None = None) -> pd.DataFrame:
    """
    Compute overall MAE, RMSE, MAPE, WAPE by model.
    Expects columns: the model names + 'true'.
    """
    if models is None:
        models = [c for c in preds_df.columns if c != "true"]
    df = preds_df.copy()
    # ensure numeric
    for c in models + ["true"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["true"])
    out = []
    true = df["true"].values
    true_sum = np.nansum(true)
    for m in models:
        pred = df[m].values
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.nanmean(abs_err))
        rmse = float(np.sqrt(np.nanmean(err**2)))
        # safe MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_vec = np.where(true != 0, abs_err / true, np.nan)
        mape = float(np.nanmean(mape_vec))
        wape = float(np.nansum(abs_err) / true_sum) if true_sum != 0 else np.nan
        out.append({
            "model": m, "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape,
            "N": int(np.isfinite(pred).sum())
        })
    return pd.DataFrame(out).set_index("model").sort_values("RMSE")


def build_predictors(models: List[str]) -> List[tuple[str, object]]:
    """Instantiate forecasters based on requested model names."""
    name_map: Dict[str, str] = {
        "arimax": "arimax",
        "linreg": "linreg",
        "gam": "gam",
        "catboost": "catboost",
    }
    predictors: List[tuple[str, object]] = []
    for m in models:
        key = m.lower()
        if key == "arimax":
            cfg = ARIMAConfig()
            predictors.append((name_map[key], ARIMAForecaster(cfg)))
        elif key == "linreg":
            cfg = LRConfig()
            predictors.append((name_map[key], LinearRegForecaster(cfg)))
        elif key == "gam":
            cfg = GAMConfig()
            predictors.append((name_map[key], GAMForecaster(cfg)))
        elif key == "catboost":
            cfg = CatBoostConfig()
            predictors.append((name_map[key], CatBoostForecaster(cfg)))
        else:
            raise ValueError(f"Unknown model '{m}'. Supported: arimax, linreg, gam, catboost")
    return predictors


# ----------------------------- Main runner ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run load forecasting pipeline (train/test/ensemble/metrics).")
    parser.add_argument("--load-dir", type=str, default="data/load_rte/", help="Path to RTE load data directory.")
    parser.add_argument("--temp-dir", type=str, default="data/temp/", help="Path to temperature data directory.")
    parser.add_argument("--train-from", type=int, default=2018, help="Training start year (inclusive).")
    parser.add_argument("--train-to", type=int, default=2023, help="Training end year (inclusive).")
    parser.add_argument("--test-from", type=int, default=2023, help="Test start year (inclusive).")
    parser.add_argument("--test-to", type=int, default=2024, help="Test end year (inclusive).")
    parser.add_argument("--date-start", type=str, default="2024-01-01", help="Start date for online prediction window (YYYY-MM-DD).")
    parser.add_argument("--date-end", type=str, default="2024-12-31", help="End date for online prediction window (YYYY-MM-DD).")
    parser.add_argument("--models", nargs="+", default=["arimax", "linreg", "gam", "catboost"],
                        help="Which base models to run. Choices: arimax linreg gam catboost")
    parser.add_argument("--ensemble", type=str, default="BOA", help="OperaStack model name (e.g., BOA, MLPol, etc.).")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--prefix", type=str, default="rte_run", help="Output filename prefix.")
    parser.add_argument("--save-parquet", action="store_true", help="Also save preds as Parquet.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pipeline")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Data ----------------
    log.info("Loading data...")
    train_df = load_rte_data(args.load_dir, from_year=args.train_from, to_year=args.train_to)
    temperature = load_temperature(args.temp_dir)
    test_df = load_rte_data(args.load_dir, from_year=args.test_from, to_year=args.test_to)

    # ---------------- Models ----------------
    predictors = build_predictors(args.models)

    # Fit all base models
    for name, model in predictors:
        log.info(f"Fitting {name}...")
        model.fit(train_df, temperature)

    # ---------------- Ensemble & Evaluation ----------------
    date_start_pred = args.date_start
    date_end_pred = args.date_end
    log.info(f"Evaluating online window: {date_start_pred} → {date_end_pred} with Opera '{args.ensemble}'")

    ens_cfg = OperaStackConfig(model=args.ensemble)
    ens = OperaStackEnsembler(ens_cfg)

    preds_df, _ = ens.evaluate_2024(
        predictors,
        test_df,
        temperature,
        date_start_online_pred=date_start_pred,
        date_end_online_pred=date_end_pred,
    )

    # Compute metrics for the base models + ensemble if present
    model_cols = [n for n, _ in predictors]
    if "ensemble" in preds_df.columns:
        model_cols = model_cols + ["ensemble"]

    metrics_df = overall_metrics(preds_df, models=model_cols)

    # ---------------- Save artifacts ----------------
    prefix = args.prefix
    preds_csv = outdir / f"{prefix}_preds.csv"
    metrics_csv = outdir / f"{prefix}_metrics.csv"
    preds_df.to_csv(preds_csv, index=True)
    metrics_df.to_csv(metrics_csv, index=True)
    if args.save_parquet:
        preds_parq = outdir / f"{prefix}_preds.parquet"
        preds_df.to_parquet(preds_parq, index=True)

    # Also persist basic run config for traceability
    run_info = {
        "models": args.models,
        "ensemble": args.ensemble,
        "train_years": [args.train_from, args.train_to],
        "test_years": [args.test_from, args.test_to],
        "date_window": [args.date_start, args.date_end],
    }
    pd.Series(run_info, dtype="object").to_json(outdir / f"{prefix}_runinfo.json", indent=2)

    log.info(f"Saved predictions → {preds_csv}")
    log.info(f"Saved metrics     → {metrics_csv}")
    log.info("Done.")


if __name__ == "__main__":
    main()
