import re
import pandas as pd
import numpy as np
from unidecode import unidecode

# Columns Translation
FRA_TO_ENG = {
    "Perimetre": "zone",
    "Nature": "data_status",
    "Date": "date",
    "Heures": "hour",
    "Consommation": "load_mw",
    "Prevision J-1": "day_minus_1_forecast_mw",
    "Prevision J": "day_forecast_mw",
    "Fioul": "oil_total_mw",
    "Charbon": "coal_mw",
    "Gaz": "gas_total_mw",
    "Nucleaire": "nuclear_mw",
    "Eolien": "wind_total_mw",
    "Solaire": "solar_mw",
    "Hydraulique": "hydro_total_mw",
    "Pompage": "pumped_storage_pumping_mw",
    "Bioenergies": "bioenergy_total_mw",
    "Ech. physiques": "physical_exchanges_mw",
    "Taux de Co2": "co2_intensity_g_per_kwh",
    "Ech. comm. Angleterre": "commercial_exchange_gb_mw",
    "Ech. comm. Espagne": "commercial_exchange_es_mw",
    "Ech. comm. Italie": "commercial_exchange_it_mw",
    "Ech. comm. Suisse": "commercial_exchange_ch_mw",
    "Ech. comm. Allemagne-Belgique": "commercial_exchange_de_be_mw",

    "Fioul - TAC": "oil_ocgt_mw",
    "Fioul - Cogen.": "oil_cogen_mw",
    "Fioul - Cogen.": "oil_cogen_mw",
    "Fioul - Cogen": "oil_cogen_mw",
    "Fioul - Autres": "oil_other_mw",

    "Gaz - TAC": "gas_ocgt_mw",
    "Gaz - Cogen.": "gas_cogen_mw",
    "Gaz - CCG": "gas_ccgt_mw",
    "Gaz - Autres": "gas_other_mw",

    "Hydraulique - Fil de l’eau + eclusee": "hydro_run_of_river_and_poundage_mw",
    "Hydraulique - Lacs": "hydro_reservoirs_mw",
    "Hydraulique - STEP turbinage": "pumped_storage_turbining_mw",

    "Bioenergies - Dechets": "bioenergy_waste_mw",
    "Bioenergies - Biomasse": "bioenergy_biomass_mw",
    "Bioenergies - Biogaz": "bioenergy_biogas_mw",

    "Stockage batterie": "battery_charge_mw",
    "Destockage batterie": "battery_discharge_mw",

    "Eolien terrestre": "onshore_wind_mw",
    "Eolien offshore": "offshore_wind_mw",
}


def _to_snake(g: str) -> str:
    g = re.sub(r"[^\w]+", "_", g)
    g = re.sub(r"_+", "_", g).strip("_")
    return g.lower()

def _normalize(col: str) -> str:
    col = col.replace("l?eau", "l’eau")
    col = unidecode(col)
    col = col.strip()
    return col

def translate_rte_columns(cols: list) -> list:
    new_cols = []

    for c in cols:
        c_norm = _normalize(c)
        # try exact map first
        if c_norm in FRA_TO_ENG:
            new_cols.append(FRA_TO_ENG[c_norm])
            continue
        # try a few near-duplicates (e.g., stray spaces, dots)
        c_norm_simple = c_norm.replace(".", "").replace("’", "'")
        if c_norm_simple in FRA_TO_ENG:
            new_cols.append(FRA_TO_ENG[c_norm_simple])
            continue

        new_cols.append(_to_snake(c_norm))

    return new_cols


def read_rte_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="cp1252",
        decimal=",",
        low_memory=False,
    ).iloc[:-1]

    df.columns = [c.strip() for c in df.columns]

    # Re shift columns
    tmp_columns = translate_rte_columns(list(df.columns))

    df = df.reset_index().iloc[:, :-1]
    df.columns = tmp_columns

    return df


def load_rte_data(path: str, 
                from_year: int =2018, 
                to_year: int=2023
                ) -> pd.DataFrame:

    all_data = pd.DataFrame()
    for year in range(from_year, to_year+1):
        if year==2020:
            continue

        df = pd.read_csv(path + f"{year}.csv")

        df = df[['date', 'hour', 'load_mw']]

        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['hour'], errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_localize(
            'Europe/Paris', nonexistent='shift_forward', ambiguous='NaT'
        )
        ts = df.set_index('timestamp').sort_index()[['load_mw']]

        df = (
            ts
            .resample('30min', label='left', closed='left') 
            .mean(numeric_only=True)
            .rename(columns={'load_mw': 'load_mw_30min'})
        )

        all_data = pd.concat((all_data, df), axis=0) if all_data.size else df

    return all_data


def load_temperature(path: str,
                   from_date: str = "2018-01-01",
                   to_date: str = "2024-12-31"
                   ) -> pd.DataFrame:
    temp = pd.read_csv(path + 'daily_average.csv', sep=';')
    temp = temp.groupby('date').mean(numeric_only=True).reset_index()
    temp['date'] = pd.to_datetime(temp['date'])
    temp['date'] = temp['date'].dt.tz_localize(
        'Europe/Paris', nonexistent='shift_forward', ambiguous='NaT'
    )
    mask = (temp["date"] >= from_date) & (temp["date"] <= to_date)
    temp = temp.loc[mask]
    temp.set_index('date', inplace=True)
    
    return temp


def keep_only_load_30min(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    df = df.copy()
    df = df[['date', 'hour', 'load_mw']]
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['hour'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.tz_localize(
        'Europe/Paris', nonexistent='shift_forward', ambiguous='NaT'
    )
    ts = df.set_index('timestamp').sort_index()[['load_mw']]

    load_30min = (
        ts
        .resample('30min', label='left', closed='left') 
        .mean(numeric_only=True)
        .rename(columns={'load_mw': 'load_mw_30min'})
    )
    return load_30min

def halfhour_to_daily_mwh(load_30min: pd.DataFrame, col="load_mw") -> pd.Series:
    """Convert 15/30-min MW series to daily MWh."""
    s = _ensure_index(load_30min)[col].astype(float)
    # detect time step in minutes (use mode)
    step_min = int(pd.Series(s.index).diff().dropna().dt.total_seconds().mode()[0] // 60)
    dt_h = 0.25 if step_min == 15 else 0.5
    y = (s * dt_h).resample("D").sum()
    y.name = "load_mwh"
    return y

def _ensure_index(df, ts_col="timestamp"):
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if ts_col in out.columns:
            out = out.set_index(ts_col)
        else:
            raise ValueError("Provide a DatetimeIndex or a 'timestamp' column.")
    if out.index.tz is None:
        out.index = out.index.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    else:
        out.index = out.index.tz_convert("Europe/Paris")
    return out.sort_index()

def overall_metrics(preds_df: pd.DataFrame, models=None) -> pd.DataFrame:
    """
    Compute overall MAE, RMSE, MAPE (and WAPE) by model.
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
        out.append({"model": m, "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape})

    return pd.DataFrame(out).set_index("model").sort_values("RMSE")
