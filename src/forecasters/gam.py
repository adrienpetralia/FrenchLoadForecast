from __future__ import annotations
import numpy as np
import pandas as pd
import holidays
from functools import reduce
from operator import add
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Tuple
from pygam import LinearGAM, s, f, te

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GAMConfig:
    tz: str = "Europe/Paris"
    # basis sizes
    k_doy: int = 20
    k_temp: int = 20
    k_partial: int = 10
    k_level: int = 10
    # whether to include tensor interaction temp x season
    use_temp_season_interaction: bool = True
    # which columns from temp daily to use
    temp_cols: tuple = ("tmin", "tmax", "tmean")


class GAMForecaster:
    """
    Forecasts next-day (D+1) *total* electricity consumption using:
      - calendar features of D,
      - *partial-day* sum of consumption on D (00:00 -> 12:00),
      - recent daily totals (Y_{D-1}, Y_{D-7}),
      - forecast (during training: observed) *D+1* temperature features.

    Training target is the *daily total of D+1*.

    Inputs
    ------
    load_30min_df:
        tz-aware (Europe/Paris) DataFrame with a single column 'load_mw_30min'
        indexed by timestamp at 30-minute frequency.
    temp_daily_df:
        tz-aware (Europe/Paris) DataFrame with daily temperature columns
        (e.g., 'tmin','tmax','tmean'). Indexed by midnight timestamps.

    Notes
    -----
    - We align features so each row corresponds to *day D*, predicting Y_{D+1}.
    - During training, we use observed temps of D+1 as a proxy for “perfect forecast”.
      At prediction, you must pass *forecasted* D+1 temperatures.
    - Holidays: hook provided via an optional DataFrame; otherwise only DoW/weekend.
    """

    def __init__(self, config: Optional[GAMConfig] = None):
        self.cfg = config or GAMConfig()
        self.gam: Optional[LinearGAM] = None
        self.feature_columns_: Optional[list[str]] = None
        self.train_dates_: Optional[pd.DatetimeIndex] = None
        self._train_stats_: Dict[str, Any] = {}

    # ---------- Public API ----------

    def fit(
        self,
        load_30min: pd.DataFrame,
        temp_daily: pd.DataFrame,
    ) -> None:
        """
        Fit the GAM on a historical period.
        """
        X_df, y = self._make_features(load_30min, temp_daily, training=True)

        self.train_dates_ = X_df.index

        # Build GAM structure (column indices match X_df columns order)
        # cols = ["dow","is_weekend","is_holiday","toy",
        #         "tmin_fc","tmax_fc","tmean_fc","temp95_fc","temp99_fc","tempmin99_fc","tempmax99_fc",
        #         "Y_D_partial","Y_Dm1","Y_Dm2","Y_Dm3","Y_Dm7"]
        terms = []
        terms.append(f(0))  # dow
        terms.append(f(1))  # is_weekendtoy
        terms.append(f(2))  # is_holiday
        terms.append(s(3, n_splines=self.cfg.k_doy, basis="cp"))  #  cyclic

        # temperatures
        terms.append(s(4, n_splines=self.cfg.k_temp))   # tmin_fc
        terms.append(s(5, n_splines=self.cfg.k_temp))   # tmax_fc
        terms.append(s(6, n_splines=self.cfg.k_temp))   # tmean_fc
        terms.append(s(7, n_splines=self.cfg.k_temp))   # temp95_fc
        terms.append(s(8, n_splines=self.cfg.k_temp))   # temp99_fc
        terms.append(s(9, n_splines=self.cfg.k_temp))   # tempmin99_fc
        terms.append(s(10, n_splines=self.cfg.k_temp))  # tempmax99_fc

        # partial and recent levels
        terms.append(s(11, n_splines=self.cfg.k_partial))  # Y_D_partial
        terms.append(s(12, n_splines=self.cfg.k_level))    # Y_Dm1
        terms.append(s(13, n_splines=self.cfg.k_level))    # Y_Dm2
        terms.append(s(14, n_splines=self.cfg.k_level))    # Y_Dm3
        terms.append(s(15, n_splines=self.cfg.k_level))    # Y_Dm7

        # optional temp x season interaction: te(tmean_fc, toy)
        if self.cfg.use_temp_season_interaction:
            terms.append(te(6, 3))

        model_terms = reduce(add, terms)

        gam = LinearGAM(model_terms)
        gam.gridsearch(X_df.to_numpy(), y)  # tunes lambdas

        self.gam = gam
        self.feature_columns_ = list(X_df.columns)
        self._train_stats_["n_days"] = len(y)

    def predict_one_day(
            self,
            date_D: pd.Timestamp,
            load_30min: pd.DataFrame,
            temp_forecast: pd.DataFrame,
        ) -> float:
            """
            Predict the total consumption for D+1, made at noon on D.
            """
            assert self.gam is not None, "Call fit() first."

            tz = self.cfg.tz

            # Build historical daily aggregates & features (training=False so no target)
            # We'll then take the row for day D and inject D+1 temps.
            X_full, _ = self._make_features(load_30min, temp_forecast, training=False)

            # Identify day D (midnight in local tz)
            D_midnight = self._as_midnight(date_D)

            if D_midnight not in X_full.index:
                raise ValueError(f"No features built for day {D_midnight.date()} — "
                                f"ensure load_30min includes that day (at least 00:00→12:00) "
                                f"and enough history for lags.")

            # DataFrame with at least tmin,tmax,tmean indexed by midnight dates (tz-aware or naive)
            tmp = temp_forecast.copy()
            if tmp.index.tz is None:
                tmp.index = tmp.index.tz_localize(tz)
            else:
                tmp = tmp.tz_convert(tz)
            tmp = tmp.sort_index()

            Dplus1 = (D_midnight + pd.Timedelta(days=1))
            if Dplus1 not in tmp.index:
                raise ValueError(f"temp_forecast must contain a row at {Dplus1.date()} (midnight).")

            # Take the row for D and overwrite *_fc columns
            x_row = X_full.loc[[D_midnight]].copy()

            # Ensure column order matches training
            if self.feature_columns_ is not None:
                x_row = x_row[self.feature_columns_]

            yhat = float(self.gam.predict(x_row.to_numpy())[0])
            return yhat

    # ---------- Private helpers ---------

    def _make_features(
            self,
            load_30min: pd.DataFrame,
            temp_daily: Optional[pd.DataFrame],
            training: bool = True,
        ) -> Tuple[pd.DataFrame, np.ndarray]:

        tz = self.cfg.tz

        # daily totals & partial sum (00:00 → 12:00)
        half = load_30min.copy()
        if half.index.tz is None:
            half.index = half.index.tz_localize(tz)
        else:
            half.index = half.index.tz_convert(tz)
        half = half.sort_index()
        half["mwh"] = half["load_mw_30min"] * 0.5

        # Calculate total cons and daily partial (before noon)
        features = half["mwh"].groupby(half.index.floor("D")).sum().to_frame("total_daily_cons")
        mask_noon = half.index.hour < 12  # 00:00 <= t < 12:00 local time
        partial = (half.loc[mask_noon, "mwh"]
                .groupby(half.index.floor("D")[mask_noon]).sum()
                .rename("Y_D_partial"))
        features = features.join(partial, how="left").fillna(0.0)

        # temps already daily at midnight index (tmin,tmax,tmean)
        assert temp_daily is not None, "temp_daily must be provided for training."
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

        # align as "forecast for D+1 available at D"
        for c in ["tmin","tmax","tmean","temp95","temp99","tempmin99","tempmax99"]:
            features[f"{c}_fc"] = tmp[c].shift(-1)


        # calendar
        idx = features.index
        # ensure tz-aware for accessors
        if idx.tz is None:
            idx = idx.tz_localize(tz)
            features.index = idx
        features["dow"] = idx.dayofweek
        # map DOY to [0,1]; use 366 to be leap-safe, cyclic basis handles wrap
        features["toy"] = (idx.day_of_year - 1) / 366.0
        features["is_weekend"] = features["dow"].isin([5,6]).astype(int)
        fr = holidays.France()
        features["is_holiday"] = [int(d.date() in fr) for d in idx]

        # load lags (avoid Y_D at noon)
        features["Y_Dm1"] = features["total_daily_cons"].shift(1)
        features["Y_Dm2"] = features["total_daily_cons"].shift(2)
        features["Y_Dm3"] = features["total_daily_cons"].shift(3)
        features["Y_Dm7"] = features["total_daily_cons"].shift(7)

        # target for training (D+1 total)
        if training:
            features["Y"] = features["total_daily_cons"].shift(-1)

        # final training features table
        cols = ["dow","is_weekend","is_holiday","toy",
                "tmin_fc","tmax_fc","tmean_fc","temp95_fc","temp99_fc","tempmin99_fc","tempmax99_fc",
                "Y_D_partial","Y_Dm1", "Y_Dm2", "Y_Dm3", "Y_Dm7"]

        if training:
            features = features[['Y'] + cols].dropna(subset=["Y"] + cols)
            X_df = features[cols]
            y = features["Y"].to_numpy()
            return X_df, y
        else:
            # For prediction we keep all rows (NaNs allowed in temp fc for now)
            X_df = features[cols]
            y_dummy = np.array([])  # not used

            return X_df, y_dummy
    
    def _ema(self, x, alpha):
        return x.ewm(alpha=alpha, adjust=False).mean()

    def _as_midnight(self, ts: pd.Timestamp) -> pd.Timestamp:
        tz = self.cfg.tz
        if ts.tz is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        return ts.normalize().tz_convert(tz)
