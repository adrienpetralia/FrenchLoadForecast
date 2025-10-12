from __future__ import annotations
import numpy as np
import pandas as pd
import holidays
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LRConfig:
    tz: str = "Europe/Paris"
    # keep the same temp columns convention as GAM
    temp_cols: tuple = ("tmin", "tmax", "tmean")
    # include a *linear* interaction between seasonal position and tmean forecast
    use_temp_season_interaction: bool = True
    # whether to z-score inputs (excluding the intercept) using train stats
    standardize_X: bool = False


class LinearRegForecaster:
    """
    Same features as GAMForecaster, but fitted with simple OLS (closed form).

    Forecasts next-day (D+1) *total* electricity consumption using:
      - calendar features of D,
      - *partial-day* sum of consumption on D (00:00 -> 12:00),
      - recent daily totals (Y_{D-1}, Y_{D-7}),
      - forecast (during training: observed) *D+1* temperature features.
    """

    def __init__(self, config: Optional[LRConfig] = None):
        self.cfg = config or LRConfig()
        self.coef_: Optional[np.ndarray] = None   # includes intercept as coef_[0]
        self.feature_columns_: Optional[list[str]] = None
        self.train_dates_: Optional[pd.DatetimeIndex] = None
        self._train_stats_: Dict[str, Any] = {}
        self._x_mean_: Optional[pd.Series] = None
        self._x_std_: Optional[pd.Series] = None

    # ---------- Public API ----------

    def fit(
        self,
        load_30min: pd.DataFrame,
        temp_daily: pd.DataFrame,
    ) -> None:
        """
        Fit OLS on the historical period. Intercept is included automatically.
        """
        X_df, y = self._make_features(load_30min, temp_daily, training=True)

        # optional standardization
        if self.cfg.standardize_X:
            self._x_mean_ = X_df.mean()
            self._x_std_ = X_df.std(ddof=0).replace(0.0, 1.0)
            Xn = (X_df - self._x_mean_) / self._x_std_
        else:
            Xn = X_df

        # Add intercept column
        X_design = np.c_[np.ones(len(Xn)), Xn.to_numpy()]

        # Closed-form OLS using lstsq (numerically stable)
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.coef_ = coef
        self.feature_columns_ = list(X_df.columns)
        self.train_dates_ = X_df.index
        self._train_stats_["n_days"] = len(y)
        self._train_stats_["train_rmse"] = float(np.sqrt(np.mean((X_design @ coef - y) ** 2)))

        logger.info("Fitted LinearRegForecaster: %d samples, RMSE=%.3f",
                    len(y), self._train_stats_["train_rmse"])

    def predict_one_day(
        self,
        date_D: pd.Timestamp,
        load_30min: pd.DataFrame,
        temp_forecast: pd.DataFrame,
    ) -> float:
        """
        Predict the total consumption for D+1, made at noon on D.
        """
        assert self.coef_ is not None, "Call fit() first."

        # Build full features (no target); then select the row for day D
        X_full, _ = self._make_features(load_30min, temp_forecast, training=False)

        D_midnight = self._as_midnight(date_D)
        if D_midnight not in X_full.index:
            raise ValueError(
                f"No features built for day {D_midnight.date()} — ensure load_30min "
                f"includes that day (at least 00:00→12:00) and enough history for lags."
            )

        x_row = X_full.loc[[D_midnight]].copy()

        # Ensure column order matches training
        if self.feature_columns_ is not None:
            x_row = x_row[self.feature_columns_]

        # Standardize like training if needed
        if self.cfg.standardize_X:
            assert self._x_mean_ is not None and self._x_std_ is not None
            x_row = (x_row - self._x_mean_) / self._x_std_

        X_design = np.c_[np.ones(len(x_row)), x_row.to_numpy()]
        yhat = float(X_design @ self.coef_)
        return yhat

    # ---------- Private helpers ----------

    def _make_features(
        self,
        load_30min: pd.DataFrame,
        temp_daily: Optional[pd.DataFrame],
        training: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        tz = self.cfg.tz

        # --- Build daily totals & partial sum (00:00 → 12:00) ---
        half = load_30min.copy()
        if half.index.tz is None:
            half.index = half.index.tz_localize(tz)
        else:
            half.index = half.index.tz_convert(tz)
        half = half.sort_index()
        half["mwh"] = half["load_mw_30min"] * 0.5

        features = half["mwh"].groupby(half.index.floor("D")).sum().to_frame("total_daily_cons")
        mask_noon = half.index.hour < 12
        partial = (
            half.loc[mask_noon, "mwh"]
            .groupby(half.index.floor("D")[mask_noon]).sum()
            .rename("Y_D_partial")
        )
        features = features.join(partial, how="left").fillna(0.0)

        # --- Temperatures (daily at midnight) ---
        assert temp_daily is not None, "temp_daily must be provided."
        tmp = temp_daily.copy()
        if tmp.index.tz is None:
            tmp.index = tmp.index.tz_localize(tz)
        else:
            tmp.index = tmp.index.tz_convert(tz)
        tmp = tmp.sort_index()
        tmp["temp95"]    = self._ema(tmp["tmean"], 0.95)
        tmp["temp99"]    = self._ema(tmp["tmean"], 0.99)
        tmp["tempmin99"] = self._ema(tmp["tmin"],  0.99)
        tmp["tempmax99"] = self._ema(tmp["tmax"],  0.99)

        # Align as "forecast for D+1 available at D"
        for c in ["tmin","tmax","tmean","temp95","temp99","tempmin99","tempmax99"]:
            features[f"{c}_fc"] = tmp[c].shift(-1)

        # --- Calendar features ---
        idx = features.index
        if idx.tz is None:
            idx = idx.tz_localize(tz)
            features.index = idx
        features["dow"] = idx.dayofweek
        features["toy"] = (idx.day_of_year - 1) / 366.0  # [0,1], leap-safe
        features["is_weekend"] = features["dow"].isin([5, 6]).astype(int)
        fr = holidays.France()
        features["is_holiday"] = [int(d.date() in fr) for d in idx]

        # --- Load lags ---
        features["Y_Dm1"] = features["total_daily_cons"].shift(1)
        features["Y_Dm2"] = features["total_daily_cons"].shift(2)
        features["Y_Dm3"] = features["total_daily_cons"].shift(3)
        features["Y_Dm7"] = features["total_daily_cons"].shift(7)

        # Optional linear interaction (season × temp)
        if self.cfg.use_temp_season_interaction:
            features["tmean_fc_x_toy"] = features["tmean_fc"] * features["toy"]

        # --- Training target (D+1 total) ---
        if training:
            features["Y"] = features["total_daily_cons"].shift(-1)

        # --- Final design matrix ---
        base_cols = [
            "dow", "is_weekend", "is_holiday", "toy",
            "tmin_fc", "tmax_fc", "tmean_fc", "temp95_fc", "temp99_fc", "tempmin99_fc", "tempmax99_fc",
            "Y_D_partial", "Y_Dm1", "Y_Dm2", "Y_Dm3", "Y_Dm7",
        ]
        if self.cfg.use_temp_season_interaction:
            base_cols.append("tmean_fc_x_toy")

        if training:
            features = features[["Y"] + base_cols].dropna(subset=["Y"] + base_cols)
            X_df = features[base_cols]
            y = features["Y"].to_numpy()
            return X_df, y
        else:
            X_df = features[base_cols].dropna(subset=base_cols, how="any")
            # keep only rows with complete features for safe prediction
            y_dummy = np.array([])
            return X_df, y_dummy

    def _ema(self, x: pd.Series, alpha: float) -> pd.Series:
        return x.ewm(alpha=alpha, adjust=False).mean()

    def _as_midnight(self, ts: pd.Timestamp) -> pd.Timestamp:
        tz = self.cfg.tz
        if ts.tz is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        return ts.normalize().tz_convert(tz)

    # ---------- Convenience inspection ----------

    def coef_as_series(self) -> pd.Series:
        """
        Return coefficients (including intercept) as a labeled Series.
        """
        assert self.coef_ is not None and self.feature_columns_ is not None
        idx = ["intercept"] + self.feature_columns_
        return pd.Series(self.coef_, index=idx)
