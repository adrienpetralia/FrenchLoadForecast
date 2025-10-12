# arima_forecaster.py
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
import holidays
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ValueWarning


warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    module=r"statsmodels\.tsa\.base\.tsa_model"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"statsmodels\.tsa\.base\.tsa_model"
)

@dataclass
class ARIMAConfig:
    tz: str = "Europe/Paris"
    # ARIMA orders on the shifted target y_t = Y_{t+1}
    order: Tuple[int, int, int] = (2, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)  # weekly seasonality (daily data)
    trend: Optional[str] = "c"  # "n", "c", "t", "ct"
    freq: str = "D"
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True


class ARIMAForecaster:
    """
    Next-day (D+1) total load forecast at noon D with SARIMAX (ARIMAX):

      endog:  y_t = Y_{t+1}  (total MWh of *next* day)
      exog:   features known at day t (calendar, D+1 temp fc, partial-noon on t, lags Y_{t-k})

    Rolling backtest does 2-step-ahead forecasts per origin:
      - Train through t = D-2 (last y observed is Y_{D-1})
      - Forecast steps=2 with exog at t = D-1 and t = D
      - Take the 2nd step → Y_{D+1}
    """

    def __init__(self, config: Optional[ARIMAConfig] = None):
        self.cfg = config or ARIMAConfig()
        self.model_ = None
        self.result_ = None
        self.feature_columns_: Optional[List[str]] = None
        self.train_index_: Optional[pd.DatetimeIndex] = None

    # ---------- Public API ----------

    def fit(
        self,
        load_30min: pd.DataFrame,   # ['load_mw_30min'] tz-aware 30-min index
        temp_daily: pd.DataFrame,   # ['tmin','tmax','tmean'] daily midnight index
    ) -> None:
        """Fit on the full history provided (for one-off inference or as initial fit)."""
        X_df, y = self._make_features(load_30min, temp_daily, training=True)
        self.train_index_ = X_df.index
        self.feature_columns_ = list(X_df.columns)

        self.model_ = SARIMAX(
            endog=y,
            exog=X_df,
            order=self.cfg.order,
            seasonal_order=self.cfg.seasonal_order,
            trend=self.cfg.trend,
            enforce_stationarity=self.cfg.enforce_stationarity,
            enforce_invertibility=self.cfg.enforce_invertibility,
        )
        # two-stage optimizer often helps
        res0 = self.model_.fit(method="powell", maxiter=200, disp=False)
        self.result_ = self.model_.fit(start_params=res0.params, method="lbfgs", maxiter=2000, disp=False)

    def predict_one_day(
        self,
        date_D: pd.Timestamp,
        load_30min: pd.DataFrame,
        temp_forecast: pd.DataFrame,
    ) -> float:
        """
        Predict the total consumption for D+1, made at noon on D.
        `temp_forecast` is a *daily* DataFrame (midnight index) with at least
        ['tmin','tmax','tmean'] (others are derived); it must contain rows for D and D+1.
        """
        assert self.result_ is not None, "Call fit() first."

        tz = self.cfg.tz
        # Build exog features using the provided forecast (same path as GAM)
        X_full, _ = self._make_features(load_30min, temp_forecast, training=False)

        # Normalize D to local midnight
        D = self._as_midnight(date_D)

        # Validate we have the two needed rows: t = D-1 and t = D
        needed = [D - pd.Timedelta(days=1), D]
        for t in needed:
            if t not in X_full.index:
                raise ValueError(
                    f"Missing features for {t.date()} — ensure load_30min covers that day "
                    f"(at least 00:00→12:00) and temp_forecast includes rows at {D.date()} and {(D + pd.Timedelta(days=1)).date()}."
                )

        # Keep only the two-step exog block, in training column order
        exog_future = X_full.loc[needed].copy()
        if self.feature_columns_ is not None:
            exog_future = exog_future[self.feature_columns_]

        # Forecast 2 steps ahead; take the 2nd step = Y_{D+1}
        fc = self.result_.get_forecast(steps=2, exog=exog_future)
        return float(fc.predicted_mean.iloc[-1])

    # ---------- Internals ----------

    def _make_features(
        self,
        load_30min: pd.DataFrame,
        temp_daily: pd.DataFrame,
        training: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build daily X_t (exog at day t) and, if training, y_t = Y_{t+1}.
        Feature set mirrors your GAM (no time-of-day).
        """
        tz = self.cfg.tz

        half = load_30min.copy()
        if half.index.tz is None:
            half.index = half.index.tz_localize(tz)
        else:
            half.index = half.index.tz_convert(tz)
        half = half.sort_index()
        half["mwh"] = half["load_mw_30min"] * 0.5

        # Daily total and partial 00:00<=t<12:00
        X = half["mwh"].groupby(half.index.floor("D")).sum().to_frame("total_daily_cons")
        mask_noon = half.index.hour < 12
        partial = (half.loc[mask_noon, "mwh"]
                     .groupby(half.index.floor("D")[mask_noon]).sum()
                     .rename("Y_D_partial"))
        X = X.join(partial, how="left").fillna(0.0)

        # Temperature features
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
        # Forecast for t+1 at time t
        for c in ["tmin","tmax","tmean","temp95","temp99","tempmin99","tempmax99"]:
            X[f"{c}_fc"] = tmp[c].shift(-1)

        # Calendar
        idx = X.index
        if idx.tz is None:
            idx = idx.tz_localize(tz)
            X.index = idx
        X["dow"] = idx.dayofweek
        X["toy"] = (idx.day_of_year - 1) / 366.0
        X["is_weekend"] = X["dow"].isin([5, 6]).astype(int)
        fr = holidays.France()
        X["is_holiday"] = [int(d.date() in fr) for d in idx]

        # Load lags (known at day t)
        X["Y_Dm1"] = X["total_daily_cons"].shift(1)
        X["Y_Dm2"] = X["total_daily_cons"].shift(2)
        X["Y_Dm3"] = X["total_daily_cons"].shift(3)
        X["Y_Dm7"] = X["total_daily_cons"].shift(7)

        # Final exog columns (order fixed)
        cols = ["dow","is_weekend","is_holiday","toy",
                "tmin_fc","tmax_fc","tmean_fc","temp95_fc","temp99_fc","tempmin99_fc","tempmax99_fc",
                "Y_D_partial","Y_Dm1","Y_Dm2","Y_Dm3","Y_Dm7"]

        # Target y_t = Y_{t+1}
        if training:
            y_series = X["total_daily_cons"].shift(-1).rename("Y_target")

            df = pd.concat([X[cols], y_series], axis=1).dropna(subset=["Y_target"] + cols)

            X_df = df[cols].astype(float)
            y_series = df["Y_target"].astype(float)

            freq = pd.infer_freq(X_df.index)
            if freq is None:
                pass
            else:
                X_df.index = pd.DatetimeIndex(X_df.index, freq=freq)
                y_series.index = pd.DatetimeIndex(y_series.index, freq=freq)

            return X_df, y_series
        else:
            freq = pd.infer_freq(X.index)
            if freq is None:
                pass
            else:
                X.index = pd.DatetimeIndex(X.index, freq=freq)
                
            return X[cols], pd.Series()

    def _daily_total(self, load_30min: pd.DataFrame) -> pd.Series:
        """Return daily total MWh indexed by midnight (local tz)."""
        tz = self.cfg.tz
        half = load_30min.copy()
        if half.index.tz is None:
            half.index = half.index.tz_localize(tz)
        else:
            half.index = half.index.tz_convert(tz)
        half = half.sort_index()
        half["mwh"] = half["load_mw_30min"] * 0.5
        return half["mwh"].groupby(half.index.floor("D")).sum()

    # ---------- small utils ----------

    def _ema(self, x: pd.Series, alpha: float) -> pd.Series:
        return x.ewm(alpha=alpha, adjust=False).mean()

    def _as_midnight(self, ts: pd.Timestamp) -> pd.Timestamp:
        tz = self.cfg.tz
        if ts.tz is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        return ts.normalize().tz_convert(tz)
