import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from dateutil.easter import easter as _easter_date

from src.utils.helpers import (
    halfhour_to_daily_mwh,
    _ensure_index
)

# ---------------------------
# Utilities
# ---------------------------

def _season_mask(idx, season="winter"):
    m = idx.month
    if season == "winter":  # DJF
        return (m == 12) | (m == 1) | (m == 2)
    if season == "summer":  # JJA
        return (m == 6) | (m == 7) | (m == 8)
    raise ValueError("season must be 'winter' or 'summer'")

# Full-year plots (30-min and daily)
def plot_full_year_30min(load_30min, col="load_mw", title="France load (30-min) — full year"):
    s = _ensure_index(load_30min)[col].astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("MW")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return ax

def _clip_to_data(start, end, idx):
    lo, hi = idx.min(), idx.max()
    if end < lo or start > hi:
        return None  # no overlap
    return max(start, lo), min(end, hi)

def _yearly_events(year):
    """Return a list of (label, start_dt, end_dt) in Europe/Paris time for one year."""
    # Fixed-date holidays
    events = [
        ("New Year",        (year, 1, 1),   (year, 1, 1)),
        ("Labour Day",      (year, 5, 1),   (year, 5, 1)),
        ("Victory in Europe",(year, 5, 8),  (year, 5, 8)),
        ("Bastille Day",    (year, 7, 14),  (year, 7, 14)),
        # Assumption – highlight the surrounding vacation week
        ("Assumption week", (year, 8, 10),  (year, 8, 20)),
        #("All Saints",      (year, 11, 1),  (year, 11, 1)),
        ("Armistice",       (year, 11, 11), (year, 11, 11)),
        ("Christmas",       (year, 12, 24), (year, 12, 26)),
    ]
    # Moveable feasts (only if dateutil.easter is available)
    eas = _easter_date(year)  # Easter Sunday (date)
    # Convert to tuples (Y,M,D)
    def _d(d): return (d.year, d.month, d.day)
    events += [
        ("Easter Monday", _d(eas + timedelta(days=1)),   _d(eas + timedelta(days=1))),
        ("Ascension Thu", _d(eas + timedelta(days=39)),  _d(eas + timedelta(days=39))),
        #("Whit Monday",   _d(eas + timedelta(days=50)),  _d(eas + timedelta(days=50))),
    ]
    return events


def plot_full_year_daily(daily_mwh: pd.Series, 
                         title: str="France daily energy — full year",
                         add_events: bool = False):
    y = daily_mwh.copy()
    y = y.resample("D").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("MWh / day")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    if add_events:
        # Build and draw event spans
        years = np.unique(y.index.tz_convert("Europe/Paris").year)
        added_labels = set()
        colors = {
            "Christmas":         "#f94144",
            "Assumption week":   "#f9844a",
            "Bastille Day":      "#f9c74f",
            "Labour Day":        "#90be6d",
            "Victory in Europe": "#43aa8b",
            "All Saints":        "#577590",
            "Armistice":         "#277da1",
            "New Year":          "#e76f51",
            "Easter Monday":     "#8ecae6",
            "Ascension Thu":     "#219ebc",
            "Whit Monday":       "#023047",
        }

        idx = y.index
        for year in years:
            for label, (ys, ms, ds), (ye, me, de) in _yearly_events(int(year)):
                # build tz-aware bounds for the full-day span
                start = pd.Timestamp(datetime(ys, ms, ds), tz="Europe/Paris")
                end   = pd.Timestamp(datetime(ye, me, de) + timedelta(days=1), tz="Europe/Paris")  # inclusive day -> next 00:00
                clipped = _clip_to_data(start, end, idx)
                if clipped is None:
                    continue
                c0, c1 = clipped
                colr = colors.get(label, "#bbbbbb")
                # only add label once to keep legend tidy
                leg = label if label not in added_labels else None
                ax.axvspan(c0, c1, color=colr, alpha=0.15, label=leg)
                added_labels.add(label)

        ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)

    fig.tight_layout()


# Average daily profiles (intraday 30-min curves)
def _avg_intraday_30min(load_30min, col="load_mw", mask=None):
    s = _ensure_index(load_30min)[col].astype(float)
    if mask is not None:
        s = s.loc[mask]
    # ensure strict 30-min grid (resample if needed)
    s = s.resample("30min").mean()
    # group by time-of-day to get mean profile (48 points)
    prof = s.groupby([s.index.time]).mean()
    return prof


def plot_avg_intraday_profiles(load_30min, col="load_mw", title_prefix="Average intraday load (30-min)"):
    idx = _ensure_index(load_30min).index
    prof_all    = _avg_intraday_30min(load_30min, col=col)
    prof_winter = _avg_intraday_30min(load_30min, col=col, mask=_season_mask(idx, "winter"))
    prof_summer = _avg_intraday_30min(load_30min, col=col, mask=_season_mask(idx, "summer"))

    fig, ax = plt.subplots(figsize=(7,4))
    for name, prof in [("All year", prof_all), ("Winter (DJF)", prof_winter), ("Summer (JJA)", prof_summer)]:
        # x-axis as HH:MM labels every ~2h
        x = pd.to_datetime(
            [f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}" for t in prof.index],
            format="%H:%M:%S"
        )

        ax.plot(x, prof.values, linewidth=2, label=name)
        ax.set_title(f"{title_prefix}")
        ax.set_ylabel("MW")
        ax.set_xlabel("Time of day")
        ax.grid(True, alpha=0.3)
        # nice ticks: every 2 hours
        ticks = pd.date_range(x.min().normalize(), x.min().normalize() + pd.Timedelta(hours=24), freq="120min")[:-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([t.strftime("%H:%M") for t in ticks], rotation=0)
        ax.legend()
        fig.tight_layout()

def plot_avg_intraweek_profiles(load_30min: pd.DataFrame,
                                col: str = "load_mw",
                                ts_col: str = "timestamp",
                                title_prefix: str = "Average intraweek load profile (30 min)"):
    """
    Expects a 30-min series (or finer) with a timestamp column or index.
    Builds a 7x48 heatmap: rows = Mon..Sun, cols = 00:00..23:30.
    """
    # --- index on timestamp ---
    df = load_30min.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        # ensure Europe/Paris for correct DOW
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
        df = df.set_index(ts_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    df = df.sort_index()

    # --- resample to exact 30-min grid & keep the column ---
    s = df[col].astype(float).resample("30min").mean()

    # build tidy frame
    frame = s.to_frame("mw")
    frame["dow"] = frame.index.dayofweek                   # 0=Mon
    frame["tod"] = (frame.index.hour * 60 + frame.index.minute)  # minutes since midnight

    # full grids to avoid missing columns/rows
    dows = range(7)
    tods = np.arange(0, 24*60, 30)  # 0,30,...,1410

    pivot = (frame
             .pivot_table(index="dow", columns="tod", values="mw", aggfunc="mean")
             .reindex(index=dows, columns=tods))

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="upper")  # im is the mappable
    fig.colorbar(im, ax=ax, label="MW")  # pass im here ✅

    # y ticks (Mon..Sun)
    ax.set_yticks(range(7), labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

    # x ticks every 2 hours
    step = 4  # 4*30min = 2h
    xticks = np.arange(0, pivot.shape[1], step)
    xtick_labels = [(pd.Timestamp(0, unit="m") + pd.Timedelta(minutes=int(tods[i]))).strftime("%H:%M")
                    for i in xticks]
    ax.set_xticks(xticks, labels=xtick_labels, rotation=0)

    ax.set_xlabel("Time of day")
    ax.set_title(title_prefix)
    fig.tight_layout()

# Average weekly profiles (by day-of-week)
def _avg_daily_by_dow(daily_mwh: pd.Series):
    df = daily_mwh.to_frame("load_mwh").copy()
    df["dow"] = df.index.dayofweek  # Mon=0
    g = df.groupby("dow")["load_mwh"].mean()
    # order Monday..Sunday
    g = g.reindex(range(7))
    g.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return g

def _daily_mwh_from_30min(df, col="load_mw", ts_col="timestamp"):
    """Fallback if you don't want to rely on halfhour_to_daily_mwh()."""
    x = df.copy()
    if ts_col in x.columns:
        x[ts_col] = pd.to_datetime(x[ts_col], errors="coerce")
        if x[ts_col].dt.tz is None:
            x[ts_col] = x[ts_col].dt.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
        x = x.set_index(ts_col)
    if x.index.tz is None:
        x.index = x.index.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    x = x.sort_index()

    step_min = int(pd.Series(x.index).diff().dropna().dt.total_seconds().mode()[0] // 60)
    dt_h = 0.25 if step_min == 15 else 0.5
    return (x[col].astype(float) * dt_h).resample("D").sum().rename("mwh")

def plot_avg_weekly_profiles(load_30min: pd.DataFrame,
                             col: str = "load_mw",
                             title_prefix: str = "Average daily energy by weekday",
                             use_internal_daily: bool = True):
    """
    Grouped bars of average daily MWh by weekday for:
      - All year
      - Winter (DJF)
      - Summer (JJA)
    """
    # 1) Daily energy
    if use_internal_daily:
        y = _daily_mwh_from_30min(load_30min, col=col)
    else:
        y = halfhour_to_daily_mwh(load_30min, col)  # your function
        if isinstance(y, pd.DataFrame):
            # try common column names
            for c in ["mwh", "load_mwh", col]:
                if c in y.columns:
                    y = y[c]
                    break
        y = y.astype(float)

    # Ensure tz-aware Paris index for correct DOW
    if y.index.tz is None:
        y.index = y.index.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    y = y.sort_index()

    # 2) Helper: weekday mean with all 7 days present
    def _avg_by_dow(series):
        if series.empty:
            return pd.Series([np.nan]*7, index=range(7), dtype=float)
        g = series.groupby(series.index.dayofweek).mean()
        return g.reindex(range(7), fill_value=np.nan).astype(float)

    # Season masks
    months = y.index.month
    winter_mask = months.isin([12,1,2])
    summer_mask = months.isin([6,7,8])

    g_all    = _avg_by_dow(y)
    g_winter = _avg_by_dow(y.loc[winter_mask])
    g_summer = _avg_by_dow(y.loc[summer_mask])

    groups = [("All year", g_all.values),
              ("Winter (DJF)", g_winter.values),
              ("Summer (JJA)", g_summer.values)]

    # Filter out groups that are entirely NaN
    groups = [(name, vals) for name, vals in groups if not np.all(np.isnan(vals))]
    if not groups:
        raise ValueError("No non-empty groups to plot (check your season masks and data coverage).")

    # 3) Geometry for grouped bars
    labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    n_days = 7
    x = np.arange(n_days)
    n_groups = len(groups)
    width = 0.8 / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1)/2) * width

    # 4) Plot
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (name, vals) in enumerate(groups):
        # Replace NaN with 0 height, but use alpha<1 so “missing” days are visually subtle
        heights = np.nan_to_num(vals, nan=0.0)
        ax.bar(x + offsets[i], heights, width=width, label=name, alpha=0.9)

    ax.set_xticks(x, labels)
    ax.set_ylabel("MWh / day")
    ax.set_xlabel("Day of week")
    ax.set_title(title_prefix)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_temperature(temp: pd.DataFrame,
                     date_start: str = "2024-01-01",
                     date_end: str = "2024-12-31"):
    # Ensure a DatetimeIndex
    if not isinstance(temp.index, pd.DatetimeIndex):
        temp = temp.copy()
        temp.index = pd.to_datetime(temp.index, errors="coerce")

    # Slice (works with tz-aware too)
    mask = (temp.index >= date_start) & (temp.index <= date_end) 
    viz = temp.loc[mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in viz.columns:
        ax.plot(viz.index, viz[col], label=col, linewidth=1.2)

    ax.set_title("Temperature")
    ax.set_xlabel("Date")
    ax.set_ylabel("°C")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Series")
    fig.tight_layout


def plot_temp_vs_consumption(df: pd.DataFrame,
                             temp_col: str = "tmean",
                             cons_col: str | None = None,
                             pivot_c: float = 15.0,
                             bins: int = 24):
    """
    Scatter of daily consumption (MWh) vs temperature (°C),
    with a binned-average curve and a vertical pivot line.
    """
    # pick the consumption column automatically if not given
    if cons_col is None:
        for c in ["total_daily_cons", "load_mwh", "consumption_mwh"]:
            if c in df.columns:
                cons_col = c
                break
        if cons_col is None:
            raise ValueError("Could not find a consumption column. Pass cons_col explicitly.")

    d = df[[temp_col, cons_col]].dropna().copy()
    x = d[temp_col].astype(float).to_numpy()
    y = d[cons_col].astype(float).to_numpy()

    # build binned mean curve
    edges = np.linspace(np.floor(x.min()), np.ceil(x.max()), bins + 1)
    inds = np.digitize(x, edges)
    xm, ym = [], []
    for i in range(1, len(edges)):
        mask = inds == i
        if np.any(mask):
            xm.append(x[mask].mean())
            ym.append(y[mask].mean())

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=14, alpha=0.45)
    ax.plot(xm, ym, linewidth=2)
    ax.axvline(pivot_c, linestyle="--")

    ax.set_xlabel("Mean temperature (°C)")
    ax.set_ylabel("Daily consumption (MWh)")
    ax.set_title("Daily electricity consumption vs temperature")

    # nice grid & tight layout
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return ax

def add_piecewise_fit(ax, x, y, pivot_c=15.0):
    x, y = np.asarray(x), np.asarray(y)
    left  = x <= pivot_c
    right = x >= pivot_c
    if left.sum() >= 2:
        b1, a1 = np.polyfit(x[left],  y[left],  1)  # y ≈ a1 + b1*x
        xx = np.linspace(x[left].min(), pivot_c, 50)
        ax.plot(xx, a1 + b1*xx, linewidth=2)
    if right.sum() >= 2:
        b2, a2 = np.polyfit(x[right], y[right], 1)
        xx = np.linspace(pivot_c, x[right].max(), 50)
        ax.plot(xx, a2 + b2*xx, linewidth=2)


# -------------------
# Global styling
# -------------------
def _style_mpl():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E0E0E0",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E9ECEF",
        "grid.linestyle": "-",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.8,
        "font.size": 11,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 120,
    })

# pleasant, distinct colors for each series
PALETTE = {
    "true":      "#27A043",  # slate
    "ensemble":  "#0957BC",  # blue
    "gam":       "#C3C059",  # teal
    "catboost":  "#D3941DD0",  # amber
    "linreg":    "#9586C1",  # violet
    "arimax":    "#AE3D65AD",  # pink
}

LINE_KW = dict(linewidth=1.8, alpha=0.95)  # default line style

def _ensure_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _format_thousands(x, pos):
    # 1,234,567 style
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)

def _setup_date_axis(ax, df_index):
    # Monthly ticks + nicer formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    ax.margins(x=0.01)

# -------------------
# Plots
# -------------------
def plot_forecast_all_model(df):
    """True vs predictions over time (single chart)."""
    _style_mpl()
    cols = ["arimax", "linreg","gam","catboost","ensemble","true"]
    df = _ensure_numeric(df, cols).dropna(subset=["true"]).copy()

    fig, ax = plt.subplots(figsize=(11.5, 4.5))

    # plot truth a bit thicker
    ax.plot(df.index, df["true"].values, label="true",
            color=PALETTE["true"], linewidth=2.4, alpha=1.0)

    # plot models
    for c in ["arimax", "linreg","gam","catboost","ensemble"]:
        ax.plot(df.index, df[c].values, label=c,
                linestyle='--',
                color=PALETTE.get(c, None), **LINE_KW)
        
    _setup_date_axis(ax, df.index)
    ax.set_title("Daily load prediction — individual forecasters vs ensemble")
    ax.set_xlabel("Date")
    ax.set_ylabel("MWh (daily total)")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))

    # top legend, multi-column
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                    ncols=6, columnspacing=1.1, handlelength=2.2)
    for line in leg.get_lines():
        line.set_linewidth(2.2)

    fig.tight_layout()
    plt.show()


def plot_scatter_true_vs_preds(df):
    """Scatter True vs Predictions (single chart with multiple series)."""
    _style_mpl()
    cols = ["arimax","linreg","gam","catboost","ensemble","true"]
    df = _ensure_numeric(df, cols).dropna(subset=["true"]).copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    y = df["true"].values

    for c in ["arimax","linreg","gam","catboost","ensemble"]:
        ax.scatter(y, df[c].values, s=16, label=c, alpha=0.7, edgecolor="none",
                   color=PALETTE.get(c, None))

    # 45-degree reference
    all_vals = np.concatenate([y, df[["arimax","linreg","gam","catboost","ensemble"]].values.ravel()])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    ax.plot([vmin, vmax], [vmin, vmax], color="#94A3B8", linewidth=1.2)

    ax.set_title("True vs Predicted (daily)")
    ax.set_xlabel("True (MWh)")
    ax.set_ylabel("Predicted (MWh)")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_forecast_error(df):
    """Absolute error over time per method (single chart)."""
    _style_mpl()
    cols = ["arimax","linreg","gam","catboost","ensemble","true"]
    df = _ensure_numeric(df, cols).dropna(subset=["true"]).copy()

    err = df[["arimax","linreg","gam","catboost","ensemble"]].sub(df["true"], axis=0).abs()

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    for c in err.columns:
        ax.plot(err.index, err[c].values, label=c, color=PALETTE.get(c, None), **LINE_KW)

    _setup_date_axis(ax, err.index)
    ax.set_title("Absolute error over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("|Error| (MWh)")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    plt.show()


def plot_monthly_mae_error(df):
    """Monthly MAE bars for each method (single chart)."""
    _style_mpl()
    cols = ["arimax","linreg","gam","catboost","ensemble","true"]
    df = _ensure_numeric(df, cols).dropna(subset=["true"]).copy()

    df["month"] = df.index.to_period("M").astype(str)
    maes = {}
    for c in ["arimax","linreg","gam","catboost","ensemble"]:
        maes[c] = (df[c] - df["true"]).abs().groupby(df["month"]).mean()
    maes_df = pd.DataFrame(maes).sort_index()

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    width = 0.16
    x = np.arange(len(maes_df.index))
    for i, c in enumerate(maes_df.columns):
        ax.bar(x + i*width, maes_df[c].values, width=width,
               label=c, color=PALETTE.get(c, None), edgecolor="white", linewidth=0.6)

    ax.set_xticks(x + (len(maes_df.columns)-1)*width/2)
    ax.set_xticklabels(maes_df.index, rotation=45, ha="right")
    ax.set_title("Monthly MAE by model")
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE (MWh)")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    plt.show()


def _style_mpl_metrics():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E0E0E0",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E9ECEF",
        "grid.linestyle": "-",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.8,
        "font.size": 11,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 120,
    })

def _fmt_thousands(x, pos):
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)

def _bar_labels(ax, rects, is_pct=False):
    """
    Add value labels above bars.
    """
    for r in rects:
        h = r.get_height()
        if np.isnan(h):
            continue
        txt = f"{h:,.2f}%" if is_pct else f"{int(round(h)):n}".replace(",", " ")
        ax.annotate(
            txt,
            xy=(r.get_x() + r.get_width()/2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color="#475569"
        )


def plot_metrics_bars(metrics: pd.DataFrame):
    """
    Create barplots for the metrics DataFrame.
    - One chart for MAE and RMSE (MWh)
    - One chart for MAPE and WAPE (shown as %)
    Expects: index = models, columns include ["MAE","RMSE","MAPE","WAPE"].
    """
    _style_mpl_metrics()

    required = ["MAE", "RMSE", "MAPE", "WAPE"]
    for c in required:
        if c not in metrics.columns:
            raise ValueError(f"Column '{c}' not found in metrics DataFrame.")
    m = metrics.copy()[required].sort_values("RMSE")

    # Colors per model (bars), legend will use neutral proxies
    colors = [PALETTE.get(idx, None) for idx in m.index]

    x = np.arange(len(m.index))
    width = 0.36

    # -------- Figure 1: MAE & RMSE (MWh) --------
    fig1, ax1 = plt.subplots(figsize=(9.5, 4.6))
    r1 = ax1.bar(x - width/2, m["MAE"].values,  width=width,
                 color=colors, edgecolor="white", linewidth=0.6, alpha=0.95)
    r2 = ax1.bar(x + width/2, m["RMSE"].values, width=width,
                 color=colors, edgecolor="white", linewidth=0.6, alpha=0.75)

    ax1.set_xticks(x)
    ax1.set_xticklabels(m.index, rotation=0)
    ax1.set_ylabel("Error (MWh)")
    ax1.set_title("Overall MAE & RMSE by model (2024)")
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    # --- Neutral legend proxies (fixed look, independent of bar colors) ---
    mae_proxy  = Patch(facecolor="0.3", edgecolor="white", alpha=0.95, label="MAE")
    rmse_proxy = Patch(facecolor="0.3", edgecolor="white", alpha=0.75, label="RMSE")
    ax1.legend(handles=[mae_proxy, rmse_proxy], loc="upper left", title="Metric")

    _bar_labels(ax1, r1, is_pct=False)
    _bar_labels(ax1, r2, is_pct=False)
    fig1.tight_layout()
    plt.show()

    # -------- Figure 2: MAPE & WAPE (%) --------
    m_pct = m[["MAPE", "WAPE"]] * 100.0
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.6))
    r3 = ax2.bar(x - width/2, m_pct["MAPE"].values,  width=width,
                 color=colors, edgecolor="white", linewidth=0.6, alpha=0.95)
    r4 = ax2.bar(x + width/2, m_pct["WAPE"].values, width=width,
                 color=colors, edgecolor="white", linewidth=0.6, alpha=0.75)

    ax2.set_xticks(x)
    ax2.set_xticklabels(m.index, rotation=0)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Overall MAPE & WAPE by model (2024)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}%"))

    # Neutral proxies for percent metrics
    mape_proxy = Patch(facecolor="0.3", edgecolor="white", alpha=0.95, label="MAPE")
    wape_proxy = Patch(facecolor="0.3", edgecolor="white", alpha=0.75, label="WAPE")
    ax2.legend(handles=[mape_proxy, wape_proxy], loc="upper left", title="Metric")

    _bar_labels(ax2, r3, is_pct=True)
    _bar_labels(ax2, r4, is_pct=True)
    fig2.tight_layout()
    plt.show()
