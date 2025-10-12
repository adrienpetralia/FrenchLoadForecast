from __future__ import annotations
import numpy as np
import pandas as pd
import holidays
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

from catboost import CatBoostRegressor, Pool

logger = logging.getLogger(__name__)

@dataclass
class CatBoostConfig:
    tz: str = "Europe/Paris"
    temp_cols: tuple = ("tmin", "tmax", "tmean")
    # add linear interactions with seasonal basis (recommended True)
    use_temp_season_interaction: bool = True

    # CatBoost core params (tweak as you like)
    depth: int = 8
    learning_rate: float = 0.06
    n_estimators: int = 2000
    l2_leaf_reg: float = 3.0
    random_seed: int = 42
    loss_function: str = "RMSE"

    # Early stopping / overfitting detection
    use_early_stopping: bool = True
    od_type: str = "Iter"        # "Iter" or "IncToDec"
    od_wait: int = 100           # patience
    eval_fraction: float = 0.1   # simple time-split for eval if early stopping is on

    verbose: bool = False        # CatBoost training verbosity


class CatBoostForecaster:
    """
    Same target definition as your GAM class:
    Predict Y_{D+1} using features available at D (noon cut for partial load).

    Calendar is encoded with *cyclical* sin/cos bases:
      - dow_sin, dow_cos  (period = 7)
      - doy_sin, doy_cos  (period = 366, leap-safe)
    Optionally adds linear interactions: tmean_fc × doy_sin / doy_cos.
    """

    def __init__(self, config: Optional[CatBoostConfig] = None):
        self.cfg = config or CatBoostConfig()
        self.model: Optional[CatBoostRegressor] = None
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
        Fit CatBoost on historical data.
        """
        X_df, y = self._make_features(load_30min, temp_daily, training=True)

        # Time-aware simple split for evaluation (keeps chronology)
        eval_set = None
        if self.cfg.use_early_stopping and len(X_df) > 10:
            n_eval = max(1, int(len(X_df) * self.cfg.eval_fraction))
            X_train, y_train = X_df.iloc[:-n_eval], y[:-n_eval]
            X_eval, y_eval = X_df.iloc[-n_eval:], y[-n_eval:]
            eval_set = Pool(X_eval, y_eval)
        else:
            X_train, y_train = X_df, y

        params = dict(
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            n_estimators=self.cfg.n_estimators,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            random_seed=self.cfg.random_seed,
            loss_function=self.cfg.loss_function,
        )

        if self.cfg.use_early_stopping:
            params.update(dict(od_type=self.cfg.od_type, od_wait=self.cfg.od_wait))

        self.model = CatBoostRegressor(**params)
        self.model.fit(
            Pool(X_train, y_train),
            eval_set=eval_set,
            verbose=self.cfg.verbose,
            use_best_model=self.cfg.use_early_stopping and (eval_set is not None),
        )

        self.feature_columns_ = list(X_df.columns)
        self.train_dates_ = X_df.index
        self._train_stats_["n_days"] = len(y)

        # store simple train RMSE
        train_pred = self.model.predict(X_train)
        self._train_stats_["train_rmse"] = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
        logger.info("Fitted CatBoostForecaster: %d samples, train RMSE=%.3f",
                    len(y), self._train_stats_["train_rmse"])

    def predict_one_day(
        self,
        date_D: pd.Timestamp,
        load_30min: pd.DataFrame,
        temp_forecast: pd.DataFrame,
    ) -> float:
        """
        Predict total consumption for D+1, using data up to noon on D and D+1 temp forecast.
        """
        assert self.model is not None, "Call fit() first."

        X_full, _ = self._make_features(load_30min, temp_forecast, training=False)
        D_midnight = self._as_midnight(date_D)

        if D_midnight not in X_full.index:
            raise ValueError(
                f"No features built for day {D_midnight.date()} — ensure load_30min "
                f"includes that day (00:00→12:00) and enough history for lags."
            )

        x_row = X_full.loc[[D_midnight]].copy()
        if self.feature_columns_ is not None:
            # keep same column order as training
            x_row = x_row[self.feature_columns_]

        yhat = float(self.model.predict(x_row)[0])
        return yhat

    def get_feature_importance(self) -> pd.Series:
        """
        CatBoost 'FeatureImportance' (loss change). Returns a Series indexed by feature name.
        """
        assert self.model is not None and self.feature_columns_ is not None
        imp = self.model.get_feature_importance(type="FeatureImportance")
        return pd.Series(imp, index=self.feature_columns_).sort_values(ascending=False)

    # ---------- Private helpers ----------

    def _make_features(
        self,
        load_30min: pd.DataFrame,
        temp_daily: pd.DataFrame,
        training: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        tz = self.cfg.tz

        # --- Daily totals & partial (00:00 → 12:00) ---
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

        # D+1 forecast available at D
        for c in ["tmin", "tmax", "tmean", "temp95", "temp99", "tempmin99", "tempmax99"]:
            features[f"{c}_fc"] = tmp[c].shift(-1)

        # --- Calendar (cyclical encoding) ---
        idx = features.index
        if idx.tz is None:
            idx = idx.tz_localize(tz)
            features.index = idx

        # Day-of-week in [0..6]
        dow = idx.dayofweek
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Day-of-year in [0..365] (leap-safe, period=366)
        doy0 = (idx.day_of_year - 1)  # 0-based
        features["doy_sin"] = np.sin(2 * np.pi * doy0 / 366.0)
        features["doy_cos"] = np.cos(2 * np.pi * doy0 / 366.0)

        # Weekend + French holiday flags (still useful)
        features["is_weekend"] = pd.Series(dow).isin([5, 6]).astype(int).to_numpy()
        fr = holidays.France()
        features["is_holiday"] = [int(d.date() in fr) for d in idx]

        # --- Load lags ---
        features["Y_Dm1"] = features["total_daily_cons"].shift(1)
        features["Y_Dm2"] = features["total_daily_cons"].shift(2)
        features["Y_Dm3"] = features["total_daily_cons"].shift(3)
        features["Y_Dm7"] = features["total_daily_cons"].shift(7)

        # Optional linear interactions between season and tmean_fc
        if self.cfg.use_temp_season_interaction:
            features["tmean_fc_x_doy_sin"] = features["tmean_fc"] * features["doy_sin"]
            features["tmean_fc_x_doy_cos"] = features["tmean_fc"] * features["doy_cos"]

        # --- Training target (D+1 total) ---
        if training:
            features["Y"] = features["total_daily_cons"].shift(-1)

        # Final feature set (all numeric)
        cols = [
            # cyclical calendar
            "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_holiday",
            # temps (D+1 forecast)
            "tmin_fc", "tmax_fc", "tmean_fc", "temp95_fc", "temp99_fc", "tempmin99_fc", "tempmax99_fc",
            # partial and lags
            "Y_D_partial", "Y_Dm1", "Y_Dm2", "Y_Dm3", "Y_Dm7",
        ]
        if self.cfg.use_temp_season_interaction:
            cols += ["tmean_fc_x_doy_sin", "tmean_fc_x_doy_cos"]

        if training:
            # Require complete features and target
            features = features[["Y"] + cols].dropna(subset=["Y"] + cols, how="any")
            X_df = features[cols].astype(float)
            y = features["Y"].to_numpy()
            return X_df, y
        else:
            # For prediction time, keep only rows with complete features
            X_df = features[cols].dropna(how="any").astype(float)
            y_dummy = np.array([])
            return X_df, y_dummy

    # --- Utilities ---

    def _ema(self, x: pd.Series, alpha: float) -> pd.Series:
        return x.ewm(alpha=alpha, adjust=False).mean()

    def _as_midnight(self, ts: pd.Timestamp) -> pd.Timestamp:
        tz = self.cfg.tz
        if ts.tz is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        return ts.normalize().tz_convert(tz)
