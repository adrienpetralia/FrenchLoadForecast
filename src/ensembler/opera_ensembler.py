# opera_ensembler.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from src.ensembler.opera import Mixture 


@dataclass
class OperaStackConfig:
    model: str = "BOA"            # "BOA" or "ML-Poly"
    coefficients: str = "uniform" # init weights: "uniform"
    loss_type: str = "mse"        # squared loss
    eta: Optional[float] = None   # learning rate
    tz: str = "Europe/Paris"


class OperaStackEnsembler:
    """
    Online stacking over daily origins D:
      - For each D in 2024, build experts' predictions for Y_{D+1}.
      - Feed them to OPERA to get the ensemble prediction.
      - Update weights sequentially with the realized Y_{D+1}.

    Forecasters must expose:
        predict_one_day(date_D, load_30min, temp_forecast) -> float
    """

    def __init__(self, config: Optional[OperaStackConfig] = None):
        self.cfg = config or OperaStackConfig()
        self.expert_names_: Optional[List[str]] = None
        self.weights_history_: Optional[pd.DataFrame] = None  # optional, if available from OPERA
        self.preds_: Optional[pd.DataFrame] = None            # daily preds incl. ensemble
        self.metrics_: Optional[Dict[str, float]] = None

    # ---------- Public API ----------

    def evaluate_2024(
        self,
        predictors: List[Tuple[str, object]], # e.g., [("arimax", arx), ("linreg", lf), ...]
        load_30min: pd.DataFrame,
        temperature: pd.DataFrame,
        date_start_online_pred: str,
        date_end_online_pred: str
    ) -> Tuple[Mixture, pd.DataFrame, Dict[str, float]]:
        """
        Returns:
            preds_df: index=D (midnight 2024), columns=[each expert, 'ensemble', 'true']
            metrics : dict (MAE, RMSE, WAPE, MAPE) for the ensemble
        """
        tz = self.cfg.tz
        # --- Build daily truth for 2024 ---
        half = load_30min.copy()
        if half.index.tz is None: half.index = half.index.tz_localize(tz)
        else:                     half.index = half.index.tz_convert(tz)
        half = half.sort_index()
        daily_mwh = (half["load_mw_30min"] * 0.5).groupby(half.index.floor("D")).sum()

        daily_mwh = daily_mwh.loc[
            (daily_mwh.index >= pd.Timestamp(date_start_online_pred, tz=tz) - pd.Timedelta(days=1)) &
            (daily_mwh.index <= pd.Timestamp(date_end_online_pred, tz=tz))
        ]

        # Valid origins D: need D-1 & D available; target = D+1 must be in 2024; temps must have D & D+1.
        all_days = daily_mwh.index.sort_values()
        valid_D = []
        for D in all_days:
            Dm1, Dp1 = D - pd.Timedelta(days=1), D + pd.Timedelta(days=1)
            if (Dm1 in all_days) and (D in all_days) and (Dp1 in daily_mwh.index):
                if (D in temperature.index) and (Dp1 in temperature.index):
                    # keep targets strictly inside 2024
                    if Dp1.year == 2024:
                        valid_D.append(D)
        valid_D = pd.DatetimeIndex(valid_D).tz_convert(tz)

        # --- Build experts' prediction matrix (rows=origins D, cols=expert names) ---
        experts_names = [name for name, _ in predictors]
        preds_mat = pd.DataFrame(index=valid_D, columns=experts_names, dtype=float)

        for name, model in predictors:
            vals = []
            for D in valid_D:
                yhat = float(model.predict_one_day(D, load_30min, temperature))
                vals.append(yhat)
            preds_mat[name] = vals

        # True Y_{D+1}
        y_true = pd.Series(
            [float(daily_mwh.loc[D + pd.Timedelta(days=1)]) for D in valid_D],
            index=valid_D, name="true"
        )


        mix, ens_pred, weights_hist = self._run_opera(preds_mat, y_true)

        # --- Assemble output table ---
        out = preds_mat.copy()
        out["ensemble"] = ens_pred
        out["true"] = y_true

        # Metrics for ensemble
        abs_err = (out["ensemble"] - out["true"]).abs()
        mae  = float(abs_err.mean())
        rmse = float(np.sqrt(((out["ensemble"] - out["true"]) ** 2).mean()))
        wape = float(abs_err.sum() / out["true"].sum())
        mape = float(np.nanmean(abs_err / out["true"].replace(0, np.nan)))

        self.expert_names_ = experts_names
        self.weights_history_ = weights_hist
        self.preds_ = out
        self.metrics_ = {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape}

        return mix, out, self.metrics_

    # ---------- Internals ----------

    def _run_opera(self, experts_df: pd.DataFrame, y_true: pd.Series):
        """
        Stream one day at a time with your local OPERA API:
        Mixture(y, experts, model=..., coefficients=..., loss_type=..., parameters=...)
        """

        # Empty history with correct expert columns so names are registered
        empty_experts = pd.DataFrame(columns=list(experts_df.columns), dtype=float)
        empty_y = pd.Series([], dtype=float)  # IMPORTANT: pandas Series

        params = None
        if getattr(self.cfg, "eta", None) is not None:
            params = {"eta": float(self.cfg.eta)}

        mix = Mixture(
            y=empty_y,
            experts=empty_experts,
            model=self.cfg.model,                # e.g. "BOA" or "ML-Poly"
            coefficients=self.cfg.coefficients,  # e.g. "uniform"
            loss_type=self.cfg.loss_type,        # e.g. "mse"
            loss_gradient=True,
            parameters=params
        )

        ens_preds = []
        weights_hist = []
        n_exp = experts_df.shape[1]
        cols = list(experts_df.columns)

        for D in experts_df.index:
            row = experts_df.loc[[D]]  # 1-row DataFrame with named columns

            # 1) predict using current mixture
            yhat = float(mix.predict(row)[0])
            ens_preds.append(yhat)

            # 2) update mixture with realized outcome (must pass both X and y as pandas objects)
            yD = pd.Series([float(y_true.loc[D])], index=[D])
            mix.update(row, yD)

            # 3) (optional) log weights only if we can extract a vector of right length
            w = self._extract_weights(mix, n_exp)
            if w is not None:
                weights_hist.append(pd.Series(w.copy(), index=cols, name=D))
            # else: silently skip — some builds don’t expose per-expert weights

        weights_df = pd.DataFrame(weights_hist) if weights_hist else None
        return mix, pd.Series(ens_preds, index=experts_df.index, name="ensemble"), weights_df
    
    def _extract_weights(self, m, n_experts: int):
        """
        Try several attribute names; return a length-n_experts 1D numpy array or None.
        Different OPERA versions expose different names/shapes.
        """
        candidates = [
            getattr(m, "coefficients", None),
            getattr(m, "weights", None),
            getattr(m, "w", None),
            getattr(m, "coefficients_", None),
            getattr(m, "last_coefficients", None),
        ]
        for c in candidates:
            if c is None:
                continue
            arr = np.asarray(c).ravel()
            if arr.size == n_experts and np.all(np.isfinite(arr)):
                return arr
        return None
