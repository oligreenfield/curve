"""
UK Index-Linked Gilt Real Yield Curve — McCulloch-style cubic spline fit,
duration-weighted, Excel-driven, with 60-day residual z-score, range/box
visualization, a dual-dropdown widget (metric: Z-Score/Residual, bond:
pick one) for the rolling time series, and a historical curve-fit
diagnostic viewer (date dropdown showing fitted curve vs. market yields
and residuals, for debugging non-mean-reverting residual patterns).

Required sheets (all in ONE workbook):
  Bonds            : Name, BaseIndex, RealPrice, [LagMonths]
  Cashflows        : Name, PaymentDate, RealCashflow, [RefMonth]
                      -- full life-of-bond schedule (issue to maturity),
                         NOT just cashflows future as of today. Both the
                         live pricing path and the historical snapshot
                         path filter to "future as of their own date".
  RPICurve         : Date, ProjectedRPI
  SeasonalFactors  : Month, Factor       <-- ADDITIVE, in percentage points
  HistoricalPrices : Date (rows) x Bond names (columns) -- real clean price
  HistoricalRPI    : Date (rows) x Tenor labels (columns, e.g. '2y','6m')
                      -- projected RPI LEVEL at (date + tenor)
"""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple, Dict
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==========================================================================
# 1. Natural cubic spline basis (McCulloch-equivalent construction)
# ==========================================================================

def _d(x, tj, tK):
    return (np.maximum(x - tj, 0.0) ** 3 - np.maximum(x - tK, 0.0) ** 3) / (tK - tj)

def _d_prime(x, tj, tK):
    return (3 * np.maximum(x - tj, 0.0) ** 2 - 3 * np.maximum(x - tK, 0.0) ** 2) / (tK - tj)

def natural_spline_basis(x, knots):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    tK = knots[-1]
    N = np.zeros((len(x), K))
    N[:, 0] = 1.0
    N[:, 1] = x
    for j in range(K - 2):
        tj = knots[j]
        N[:, j + 2] = _d(x, tj, tK) - _d(x, knots[-2], tK)
    return N

def natural_spline_basis_deriv(x, knots):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    tK = knots[-1]
    Np = np.zeros((len(x), K))
    Np[:, 0] = 0.0
    Np[:, 1] = 1.0
    for j in range(K - 2):
        tj = knots[j]
        Np[:, j + 2] = _d_prime(x, tj, tK) - _d_prime(x, knots[-2], tK)
    return Np


# ==========================================================================
# 2. Bond container
# ==========================================================================

@dataclass
class Bond:
    name: str
    times: np.ndarray
    real_cashflows: np.ndarray
    real_price: float
    base_index: float = 100.0
    ref_months: Optional[np.ndarray] = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.real_cashflows = np.asarray(self.real_cashflows, dtype=float)


def _flat_yield_duration(times: np.ndarray, cashflows: np.ndarray, price: float) -> float:
    def pv(y):
        return np.sum(cashflows * np.exp(-y * times)) - price
    try:
        y0 = brentq(pv, -0.20, 0.50)
    except ValueError:
        y0 = np.log(np.sum(cashflows) / price) / max(times[-1], 1e-6)
    disc = cashflows * np.exp(-y0 * times)
    return float(np.sum(times * disc) / np.sum(disc))


# ==========================================================================
# 3. Excel loaders — current-day curve inputs
# ==========================================================================

def load_bonds_from_excel(path: str, valuation_date, bond_sheet="Bonds",
                           cashflow_sheet="Cashflows") -> List[Bond]:
    bonds_df = pd.read_excel(path, sheet_name=bond_sheet)
    cf_df = pd.read_excel(path, sheet_name=cashflow_sheet)

    valuation_date = pd.Timestamp(valuation_date)
    cf_df["PaymentDate"] = pd.to_datetime(cf_df["PaymentDate"])

    # Cashflows now holds each bond's FULL life-of-bond schedule, so we
    # must filter to flows still outstanding as of valuation_date -- the
    # same "future as of this date" filter build_dated_bond_snapshots
    # already applies per historical date. Without this, matured coupons
    # produce negative Time values and break the brentq bracket in
    # real_ytm / _flat_yield_duration.
    cf_df = cf_df[cf_df["PaymentDate"] > valuation_date]

    cf_df["Time"] = (cf_df["PaymentDate"] - valuation_date).dt.days / 365.25

    if "LagMonths" not in bonds_df.columns:
        bonds_df["LagMonths"] = 3
    lag_lookup = bonds_df.set_index("Name")["LagMonths"]

    if "RefMonth" not in cf_df.columns:
        def _ref_month(row):
            lag = int(lag_lookup.get(row["Name"], 3))
            ref_date = row["PaymentDate"] - pd.DateOffset(months=lag)
            return ref_date.month
        cf_df["RefMonth"] = cf_df.apply(_ref_month, axis=1)

    bonds = []
    for _, brow in bonds_df.iterrows():
        sub = cf_df[cf_df["Name"] == brow["Name"]].sort_values("Time")
        if sub.empty:
            raise ValueError(f"No future cashflows found for bond '{brow['Name']}' as of {valuation_date.date()} — check Name spelling matches between sheets, or that the bond hasn't matured.")
        bonds.append(Bond(
            name=str(brow["Name"]),
            times=sub["Time"].to_numpy(),
            real_cashflows=sub["RealCashflow"].to_numpy(),
            real_price=float(brow["RealPrice"]),
            base_index=float(brow["BaseIndex"]),
            ref_months=sub["RefMonth"].to_numpy(),
        ))
    return bonds


def load_rpi_curve_from_excel(path: str, valuation_date, sheet="RPICurve") -> Callable:
    df = pd.read_excel(path, sheet_name=sheet)
    valuation_date = pd.Timestamp(valuation_date)
    df["Date"] = pd.to_datetime(df["Date"])
    times = ((df["Date"] - valuation_date).dt.days / 365.25).to_numpy()
    levels = df["ProjectedRPI"].to_numpy(dtype=float)
    order = np.argsort(times)
    times_sorted, levels_sorted = times[order], levels[order]

    def rpi_forward_curve(t):
        return np.interp(np.asarray(t, dtype=float), times_sorted, levels_sorted)
    return rpi_forward_curve


def load_seasonal_factors_from_excel(path: str, sheet="SeasonalFactors") -> Callable:
    df = pd.read_excel(path, sheet_name=sheet)
    multiplicative = np.exp(df.set_index("Month")["Factor"].to_numpy(dtype=float) / 100.0)
    gmean = np.exp(np.mean(np.log(multiplicative)))
    multiplicative = multiplicative / gmean
    vec = dict(zip(df["Month"].astype(int), multiplicative))

    def seasonal_factor(m):
        return vec[int(m)]
    return seasonal_factor


# ==========================================================================
# 4. Excel loaders — historical (wide-format) sheets for backfill
# ==========================================================================

def parse_tenor(tenor_str) -> float:
    s = str(tenor_str).strip().lower()
    m = re.match(r"^([0-9.]+)\s*([ymwd])?$", s)
    if not m:
        raise ValueError(f"Cannot parse tenor string: '{tenor_str}'")
    value, unit = float(m.group(1)), m.group(2)
    if unit is None or unit == "y":
        return value
    if unit == "m":
        return value / 12.0
    if unit == "w":
        return value * 7 / 365.25
    if unit == "d":
        return value / 365.25
    raise ValueError(f"Unrecognised tenor unit in: '{tenor_str}'")


def load_historical_prices_from_excel(path: str, sheet="HistoricalPrices") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()


def load_historical_rpi_from_excel(path: str, sheet="HistoricalRPI") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
    df = df.rename(columns={c: parse_tenor(c) for c in df.columns})
    return df[sorted(df.columns)]


def make_rpi_forward_curve_for_row(rpi_row: pd.Series) -> Callable:
    tenors = rpi_row.index.to_numpy(dtype=float)
    levels = rpi_row.to_numpy(dtype=float)
    order = np.argsort(tenors)
    tenors_sorted, levels_sorted = tenors[order], levels[order]

    def rpi_forward_curve(t):
        return np.interp(np.asarray(t, dtype=float), tenors_sorted, levels_sorted)
    return rpi_forward_curve


def build_dated_bond_snapshots(
    excel_path: str,
    bond_sheet: str = "Bonds",
    cashflow_sheet: str = "Cashflows",
    historical_price_sheet: str = "HistoricalPrices",
    historical_rpi_sheet: str = "HistoricalRPI",
) -> List[Tuple[pd.Timestamp, List[Bond], Callable]]:
    bonds_static = pd.read_excel(excel_path, sheet_name=bond_sheet)
    cf_df = pd.read_excel(excel_path, sheet_name=cashflow_sheet)
    cf_df["PaymentDate"] = pd.to_datetime(cf_df["PaymentDate"])

    if "LagMonths" not in bonds_static.columns:
        bonds_static["LagMonths"] = 3
    lag_lookup = bonds_static.set_index("Name")["LagMonths"]
    base_index_lookup = bonds_static.set_index("Name")["BaseIndex"]

    if "RefMonth" not in cf_df.columns:
        def _ref_month(row):
            lag = int(lag_lookup.get(row["Name"], 3))
            return (row["PaymentDate"] - pd.DateOffset(months=lag)).month
        cf_df["RefMonth"] = cf_df.apply(_ref_month, axis=1)

    price_panel = load_historical_prices_from_excel(excel_path, historical_price_sheet)
    rpi_panel = load_historical_rpi_from_excel(excel_path, historical_rpi_sheet)

    common_dates = price_panel.index.intersection(rpi_panel.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between HistoricalPrices and HistoricalRPI sheets.")

    snapshots = []
    for hist_date in common_dates:
        bonds_this_date = []
        for bond_name in price_panel.columns:
            price = price_panel.loc[hist_date, bond_name]
            if pd.isna(price):
                continue

            sub = cf_df[(cf_df["Name"] == bond_name) & (cf_df["PaymentDate"] > hist_date)].sort_values("PaymentDate")
            if sub.empty:
                continue

            times = ((sub["PaymentDate"] - hist_date).dt.days / 365.25).to_numpy()
            bonds_this_date.append(Bond(
                name=bond_name,
                times=times,
                real_cashflows=sub["RealCashflow"].to_numpy(),
                real_price=float(price),
                base_index=float(base_index_lookup.get(bond_name, 100.0)),
                ref_months=sub["RefMonth"].to_numpy(),
            ))

        if len(bonds_this_date) < 4:
            continue

        rpi_curve_fn = make_rpi_forward_curve_for_row(rpi_panel.loc[hist_date])
        snapshots.append((hist_date, bonds_this_date, rpi_curve_fn))

    return snapshots


# ==========================================================================
# 5. Knot selection
# ==========================================================================

def pick_knots(maturities, n_knots: Optional[int] = None):
    maturities = np.sort(np.asarray(maturities, dtype=float))
    if n_knots is None:
        n_knots = max(4, int(round(np.sqrt(len(maturities)))))
    knots = np.quantile(maturities, np.linspace(0, 1, n_knots))
    knots[0] = 0.0
    knots[-1] = maturities.max()
    return np.unique(knots)


# ==========================================================================
# 6. Curve fitting — DURATION-WEIGHTED OLS
# ==========================================================================

def fit_mcculloch_curve(bonds: List[Bond], knots: np.ndarray, price_attr: str = "real_price"):
    K = len(knots)
    X, y = [], []
    for b in bonds:
        N = natural_spline_basis(b.times, knots)
        cf_sum = np.sum(b.real_cashflows)
        price = getattr(b, price_attr)
        duration = _flat_yield_duration(b.times, b.real_cashflows, price)

        row = np.zeros(K - 1)
        row[0] = np.sum(b.real_cashflows * b.times)
        for j in range(2, K):
            row[j - 1] = np.sum(b.real_cashflows * N[:, j])

        X.append(row / duration)
        y.append((price - cf_sum) / duration)

    X = np.array(X)
    y = np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta_full = np.concatenate([[1.0], beta])

    def discount(m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        return natural_spline_basis(m, knots) @ beta_full

    def zero_yield(m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        return -np.log(discount(m)) / np.maximum(m, 1e-8)

    def instantaneous_forward(m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        d = discount(m)
        dprime = natural_spline_basis_deriv(m, knots) @ beta_full
        return -dprime / d

    return discount, zero_yield, instantaneous_forward, beta_full


def price_bond(discount_fn, bond: Bond) -> float:
    return float(np.sum(bond.real_cashflows * discount_fn(bond.times)))


def real_ytm(times, cashflows, price, freq: int = 2) -> float:
    def f(y):
        return np.sum(cashflows / (1 + y / freq) ** (times * freq)) - price
    return brentq(f, -0.15, 0.30)


# ==========================================================================
# 7. Seasonal adjustment
# ==========================================================================

def seasonal_adjust_bond(bond: Bond, rpi_forward_curve: Callable, seasonal_factor: Callable) -> dict:
    idx_plain = rpi_forward_curve(bond.times) / bond.base_index
    months = bond.ref_months if bond.ref_months is not None else \
        np.array([1 + int(round(t * 12)) % 12 for t in bond.times])
    seas = np.array([seasonal_factor(int(m)) for m in months])
    idx_seasonal = idx_plain * seas

    nominal_cf_plain = bond.real_cashflows * idx_plain
    nominal_cf_seasonal = bond.real_cashflows * idx_seasonal

    y_real = real_ytm(bond.times, bond.real_cashflows, bond.real_price)
    disc = 1.0 / (1 + y_real / 2) ** (bond.times * 2)

    nominal_price_plain = float(np.sum(nominal_cf_plain * disc))
    nominal_price_seasonal = float(np.sum(nominal_cf_seasonal * disc))
    seasonal_residual = nominal_price_seasonal - nominal_price_plain

    seas_adj_real_price = bond.real_price + seasonal_residual
    seas_adj_real_yield = real_ytm(bond.times, bond.real_cashflows, seas_adj_real_price)

    return dict(
        name=bond.name, real_yield=y_real, real_price=bond.real_price,
        nominal_price_plain=nominal_price_plain,
        nominal_price_seasonal=nominal_price_seasonal,
        seas_adj_real_price=seas_adj_real_price,
        seas_adj_real_yield=seas_adj_real_yield,
    )


# ==========================================================================
# 8. Spread history store (with defensive schema handling)
# ==========================================================================

def load_spread_history(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Name", "Spread_bp"])


def append_spread_history(path: str, valuation_date, spread_by_bond: Dict[str, float]) -> pd.DataFrame:
    df = load_spread_history(path)
    df = df.reindex(columns=["Date", "Name", "Spread_bp"])
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Name"] = df["Name"].astype(str)
        df["Spread_bp"] = df["Spread_bp"].astype(float)

    valuation_date = pd.Timestamp(valuation_date)
    df = df[df["Date"] != valuation_date]

    names = list(spread_by_bond.keys())
    values = [float(np.ravel(v)[0]) for v in spread_by_bond.values()]
    if len(names) != len(values):
        raise ValueError(f"spread_by_bond keys/values length mismatch: {len(names)} names vs {len(values)} values")

    new_rows = pd.DataFrame({
        "Date": [valuation_date] * len(names),
        "Name": names,
        "Spread_bp": values,
    })

    df = pd.concat([df, new_rows], ignore_index=True, sort=False)
    df = df.sort_values(["Name", "Date"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def compute_today_spreads(bonds_adj: List[Bond], discount_fn) -> Dict[str, float]:
    out = {}
    for b in bonds_adj:
        mkt_y = real_ytm(b.times, b.real_cashflows, b.real_price)
        fit_y = real_ytm(b.times, b.real_cashflows, price_bond(discount_fn, b))
        out[b.name] = (mkt_y - fit_y) * 1e4
    return out


def backfill_spread_history(
    history_path: str,
    dated_bond_snapshots: List[Tuple[pd.Timestamp, List[Bond], Callable]],
    seasonal_factor: Callable,
    n_knots: Optional[int] = None,
):
    for date, bonds, rpi_curve in dated_bond_snapshots:
        knots = pick_knots([b.times[-1] for b in bonds], n_knots=n_knots)
        seas_rows = [seasonal_adjust_bond(b, rpi_curve, seasonal_factor) for b in bonds]
        bonds_adj = [
            Bond(b.name, b.times, b.real_cashflows, r["seas_adj_real_price"], b.base_index, b.ref_months)
            for b, r in zip(bonds, seas_rows)
        ]
        disc, _, _, _ = fit_mcculloch_curve(bonds_adj, knots, price_attr="real_price")
        adj_yield = {r["name"]: r["seas_adj_real_yield"] for r in seas_rows}
        spreads = {}
        for b in bonds_adj:
            fit_y = real_ytm(b.times, b.real_cashflows, price_bond(disc, b))
            spreads[b.name] = (adj_yield[b.name] - fit_y) * 1e4
        append_spread_history(history_path, date, spreads)
        print(f"  backfilled {date.date()}: {len(bonds)} bonds")


# ==========================================================================
# 9. Z-score: single-date cross-section AND rolling time series
# ==========================================================================

def compute_zscores(history_df: pd.DataFrame, as_of_date, lookback_days: int = 60) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of_date)
    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    window = history_df[(history_df["Date"] > window_start) & (history_df["Date"] <= as_of_date)]

    rows = []
    for name, g in window.groupby("Name"):
        g = g.sort_values("Date")
        if len(g) < 5:
            continue
        today_val = g[g["Date"] == as_of_date]["Spread_bp"]
        if today_val.empty:
            continue
        today_val = float(today_val.iloc[-1])
        mean = g["Spread_bp"].mean()
        std = g["Spread_bp"].std(ddof=1)
        z = (today_val - mean) / std if std > 1e-9 else 0.0
        rows.append(dict(
            Name=name, Today_bp=today_val, Mean_bp=mean, Std_bp=std, Z=z,
            Min_bp=g["Spread_bp"].min(), Max_bp=g["Spread_bp"].max(),
            N_obs=len(g), history=g["Spread_bp"].to_numpy(),
        ))
    return pd.DataFrame(rows)


def compute_rolling_zscore_series(history_df: pd.DataFrame, lookback_days: int = 60,
                                   min_obs: int = 5) -> pd.DataFrame:
    """Time series (per date, per bond) of BOTH the raw residual (Spread_bp)
    and its rolling z-score (Z) -- one computation serves both dropdown views."""
    results = []
    for name, g in history_df.groupby("Name"):
        g = g.sort_values("Date").set_index("Date")
        roll_mean = g["Spread_bp"].rolling(f"{lookback_days}D", min_periods=min_obs).mean()
        roll_std = g["Spread_bp"].rolling(f"{lookback_days}D", min_periods=min_obs).std(ddof=1)
        z = (g["Spread_bp"] - roll_mean) / roll_std
        out = pd.DataFrame({
            "Date": g.index, "Name": name,
            "Spread_bp": g["Spread_bp"].to_numpy(), "Z": z.to_numpy(),
        })
        results.append(out)
    return pd.concat(results, ignore_index=True)


# ==========================================================================
# 10. Visuals
# ==========================================================================

def build_zscore_figure(zscore_df: pd.DataFrame, lookback_days: int = 60) -> go.Figure:
    zscore_df = zscore_df.sort_values("Today_bp")

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45],
        subplot_titles=(f"{lookback_days}-day residual range vs. today's value",
                         f"{lookback_days}-day residual z-score"),
        vertical_spacing=0.12,
    )

    for _, row in zscore_df.iterrows():
        fig.add_trace(go.Box(
            y=row["history"], name=row["Name"], boxpoints=False,
            marker_color="rgba(100,150,220,0.5)", line=dict(color="rgba(100,150,220,0.8)"),
            showlegend=False,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=zscore_df["Name"], y=zscore_df["Today_bp"], mode="markers",
        marker=dict(color="crimson", size=10, symbol="diamond"),
        name="Today", showlegend=True,
    ), row=1, col=1)
    fig.update_yaxes(title_text="Spread (bp)", row=1, col=1)

    colors = ["crimson" if abs(z) >= 2 else "orange" if abs(z) >= 1 else "steelblue" for z in zscore_df["Z"]]
    fig.add_trace(go.Bar(
        x=zscore_df["Name"], y=zscore_df["Z"], marker_color=colors, name="Z-score", showlegend=False,
    ), row=2, col=1)
    for level, dash in [(2, "dash"), (-2, "dash"), (1, "dot"), (-1, "dot")]:
        fig.add_hline(y=level, line_dash=dash, line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)

    fig.update_layout(height=800, title=f"Bond richness/cheapness — {lookback_days}-day lookback")
    return fig


def build_metric_bond_widget_html(history_df: pd.DataFrame, lookback_days: int = 60,
                                   min_obs: int = 5, div_id: str = "metric_bond_widget") -> str:
    """
    Two independent dropdowns: (1) metric = Z-Score or Residual, (2) bond.
    Native Plotly updatemenus don't coordinate with each other, so this
    uses a small custom JS listener on the 'plotly_buttonclicked' event
    (a documented Plotly.js event) to combine both dropdowns' state and
    recompute which single trace should be visible. Returns a raw HTML
    string (figure + script) ready to be embedded in the page.
    """
    z_series = compute_rolling_zscore_series(history_df, lookback_days=lookback_days, min_obs=min_obs)
    names = sorted(z_series["Name"].unique())
    n_bonds = len(names)
    metrics = ["Z-Score", "Residual"]
    n_metrics = len(metrics)

    fig = go.Figure()
    for metric_idx, metric in enumerate(metrics):
        col = "Z" if metric == "Z-Score" else "Spread_bp"
        for bond_idx, name in enumerate(names):
            sub = z_series[z_series["Name"] == name].sort_values("Date")
            fig.add_trace(go.Scatter(
                x=sub["Date"], y=sub[col], mode="lines+markers", name=name,
                visible=(metric_idx == 0 and bond_idx == 0),
                showlegend=False,
            ))

    metric_buttons = [dict(label=m, method="skip", args=[]) for m in metrics]
    bond_buttons = [dict(label=nm, method="skip", args=[]) for nm in names]

    fig.update_layout(
        updatemenus=[
            dict(buttons=metric_buttons, active=0, x=0.0, xanchor="left", y=1.15, yanchor="top",
                 direction="down", showactive=True),
            dict(buttons=bond_buttons, active=0, x=0.35, xanchor="left", y=1.15, yanchor="top",
                 direction="down", showactive=True),
        ],
        title=f"{names[0]} — Z-Score ({lookback_days}-day rolling)",
        xaxis_title="Date", height=520, margin=dict(t=120),
    )

    zscore_shapes = []
    for level, dash, color in [(2, "dash", "crimson"), (-2, "dash", "crimson"),
                                (1, "dot", "orange"), (-1, "dot", "orange"), (0, "solid", "gray")]:
        zscore_shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y",
                                   y0=level, y1=level, line=dict(color=color, dash=dash, width=1), opacity=0.6))
    residual_shapes = [dict(type="line", xref="paper", x0=0, x1=1, yref="y",
                             y0=0, y1=0, line=dict(color="gray", dash="solid", width=1), opacity=0.6)]
    fig.update_layout(shapes=zscore_shapes)

    inner_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

    names_json = str(names).replace("'", '"')
    metrics_json = str(metrics).replace("'", '"')
    zscore_shapes_json = str(zscore_shapes).replace("'", '"')
    residual_shapes_json = str(residual_shapes).replace("'", '"')

    script = f"""
<script>
(function() {{
    var gd = document.getElementById("{div_id}");
    var N_BONDS = {n_bonds};
    var N_METRICS = {n_metrics};
    var NAMES = {names_json};
    var METRICS = {metrics_json};
    var ZSCORE_SHAPES = {zscore_shapes_json};
    var RESIDUAL_SHAPES = {residual_shapes_json};
    var currentMetric = 0;
    var currentBond = 0;

    function traceIndex(metricIdx, bondIdx) {{
        return metricIdx * N_BONDS + bondIdx;
    }}

    function applySelection() {{
        var total = N_METRICS * N_BONDS;
        var visible = new Array(total).fill(false);
        visible[traceIndex(currentMetric, currentBond)] = true;
        var shapes = (currentMetric === 0) ? ZSCORE_SHAPES : RESIDUAL_SHAPES;
        var titleText = NAMES[currentBond] + " \\u2014 " + METRICS[currentMetric] +
                         (currentMetric === 0 ? " (rolling)" : " (bp, actual vs. fitted yield)");
        Plotly.restyle(gd, {{visible: visible}});
        Plotly.relayout(gd, {{shapes: shapes, title: titleText}});
    }}

    gd.on("plotly_buttonclicked", function(data) {{
        var menuIndex = gd.layout.updatemenus.indexOf(data.menu);
        if (menuIndex === 0) {{
            currentMetric = data.active;
        }} else if (menuIndex === 1) {{
            currentBond = data.active;
        }}
        applySelection();
    }});
}})();
</script>
"""
    return inner_html + script


def build_historical_curve_diagnostic(
    dated_bond_snapshots: List[Tuple[pd.Timestamp, List[Bond], Callable]],
    seasonal_factor: Callable,
    n_knots: Optional[int] = None,
    out_path: str = "historical_curve_diagnostic.html",
) -> str:
    """
    Refits the seasonally-adjusted McCulloch spline on every historical
    snapshot date (same logic as backfill_spread_history) and produces a
    single HTML page with a date dropdown showing:

      Top panel:    market yields (scatter) vs. the fitted curve (line),
                    with knot locations marked
      Bottom panel: per-bond residual (bp) for that date

    Use this to check whether a non-mean-reverting residual pattern is
    curve-wide on a given date (knot placement / seasonal factor issue),
    concentrated in one bond especially near maturity (duration-weighting
    issue), or a genuine slow-moving market effect.
    """
    all_data = []

    for date, bonds, rpi_curve in dated_bond_snapshots:
        knots = pick_knots([b.times[-1] for b in bonds], n_knots=n_knots)

        seas_rows = [seasonal_adjust_bond(b, rpi_curve, seasonal_factor) for b in bonds]
        bonds_adj = [
            Bond(b.name, b.times, b.real_cashflows, r["seas_adj_real_price"], b.base_index, b.ref_months)
            for b, r in zip(bonds, seas_rows)
        ]

        disc, zy, fwd, beta = fit_mcculloch_curve(bonds_adj, knots, price_attr="real_price")

        adj_yield = {r["name"]: r["seas_adj_real_yield"] for r in seas_rows}
        maturities = np.array([b.times[-1] for b in bonds_adj])
        mkt_yields = np.array([adj_yield[b.name] for b in bonds_adj])
        fit_yields = np.array([
            real_ytm(b.times, b.real_cashflows, price_bond(disc, b)) for b in bonds_adj
        ])
        residuals_bp = (mkt_yields - fit_yields) * 1e4

        grid = np.linspace(max(knots[1] * 0.1, 0.05), maturities.max(), 150)
        curve_y = zy(grid) * 100

        all_data.append(dict(
            date=date,
            names=[b.name for b in bonds_adj],
            maturities=maturities,
            mkt_yields=mkt_yields * 100,
            grid=grid,
            curve_y=curve_y,
            residuals_bp=residuals_bp,
            knots=knots,
            n_bonds=len(bonds_adj),
        ))

    if not all_data:
        raise ValueError("No snapshots to plot -- check dated_bond_snapshots is non-empty.")

    n_dates = len(all_data)
    traces_per_date = 4  # market scatter, fitted line, knot markers, residual bar

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        subplot_titles=("Market yield vs. fitted curve", "Residual to fitted curve (bp)"),
        vertical_spacing=0.12,
    )

    default_idx = n_dates - 1  # most recent date shown first

    for i, d in enumerate(all_data):
        vis = (i == default_idx)

        fig.add_trace(go.Scatter(
            x=d["maturities"], y=d["mkt_yields"], mode="markers+text",
            text=d["names"], textposition="top center", textfont=dict(size=9),
            name="Market yield", marker=dict(size=9, color="crimson"),
            visible=vis, showlegend=vis,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=d["grid"], y=d["curve_y"], mode="lines",
            name="Fitted spline", line=dict(color="steelblue", width=2),
            visible=vis, showlegend=vis,
        ), row=1, col=1)

        knot_y = np.interp(d["knots"], d["grid"], d["curve_y"])
        fig.add_trace(go.Scatter(
            x=d["knots"], y=knot_y, mode="markers",
            name="Knots", marker=dict(size=7, color="gray", symbol="line-ns-open"),
            visible=vis, showlegend=vis,
        ), row=1, col=1)

        colors = ["crimson" if abs(r) >= 10 else "orange" if abs(r) >= 5 else "steelblue"
                  for r in d["residuals_bp"]]
        fig.add_trace(go.Bar(
            x=d["names"], y=d["residuals_bp"], marker_color=colors,
            name="Residual (bp)", showlegend=False, visible=vis,
        ), row=2, col=1)

    buttons = []
    for i, d in enumerate(all_data):
        vis = [False] * (n_dates * traces_per_date)
        base = i * traces_per_date
        vis[base:base + traces_per_date] = [True] * traces_per_date
        buttons.append(dict(
            label=str(d["date"].date()),
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Curve fit — {d['date'].date()}  ({d['n_bonds']} bonds, {len(d['knots'])} knots)"},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, active=default_idx,
            x=0.0, xanchor="left", y=1.15, yanchor="top", direction="down",
        )],
        height=850,
        title=f"Curve fit — {all_data[default_idx]['date'].date()}  "
              f"({all_data[default_idx]['n_bonds']} bonds, {len(all_data[default_idx]['knots'])} knots)",
    )
    fig.update_yaxes(title_text="Real yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bp)", row=2, col=1)
    fig.update_xaxes(title_text="Maturity (years)", row=1, col=1)

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Diagnostic written to {out_path} ({n_dates} dates)")
    return out_path


# ==========================================================================
# 11. Full pipeline: fit both curves, compute spreads, build HTML report
# ==========================================================================

def build_report(bonds: List[Bond], rpi_forward_curve: Callable, seasonal_factor: Callable,
                  n_knots: Optional[int] = None, out_path: str = "real_yield_curve_report.html"):

    maturities = np.array([b.times[-1] for b in bonds])
    knots = pick_knots(maturities, n_knots=n_knots)

    disc_u, zy_u, fwd_u, _ = fit_mcculloch_curve(bonds, knots, price_attr="real_price")
    mkt_yield = np.array([real_ytm(b.times, b.real_cashflows, b.real_price) for b in bonds])
    fit_yield_u = np.array([real_ytm(b.times, b.real_cashflows, price_bond(disc_u, b)) for b in bonds])
    spread_u_bp = (mkt_yield - fit_yield_u) * 1e4

    seas_rows = [seasonal_adjust_bond(b, rpi_forward_curve, seasonal_factor) for b in bonds]
    bonds_adj = [
        Bond(b.name, b.times, b.real_cashflows, r["seas_adj_real_price"], b.base_index, b.ref_months)
        for b, r in zip(bonds, seas_rows)
    ]
    disc_a, zy_a, fwd_a, _ = fit_mcculloch_curve(bonds_adj, knots, price_attr="real_price")
    adj_mkt_yield = np.array([r["seas_adj_real_yield"] for r in seas_rows])
    fit_yield_a = np.array([real_ytm(b.times, b.real_cashflows, price_bond(disc_a, b)) for b in bonds_adj])
    spread_a_bp = (adj_mkt_yield - fit_yield_a) * 1e4

    grid = np.linspace(max(knots[1] * 0.1, 0.1), maturities.max(), 200)

    fig1 = make_subplots(rows=3, cols=1, row_heights=[0.42, 0.23, 0.35],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Unadjusted real yield curve", "Spread to fitted curve (bp)", None),
        vertical_spacing=0.06)
    fig1.add_trace(go.Scatter(x=maturities, y=mkt_yield * 100, mode="markers",
                               name="Market yield", marker=dict(size=8)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=grid, y=zy_u(grid) * 100, mode="lines", name="Fitted spline"), row=1, col=1)
    fig1.add_trace(go.Bar(x=[b.name for b in bonds], y=spread_u_bp, name="Spread (bp)"), row=2, col=1)
    fig1.add_trace(go.Table(
        header=dict(values=["Name", "Real Yield (%)", "Real Price", "Fitted Yield (%)", "Spread (bp)"],
                    fill_color="#1f2937", font=dict(color="white")),
        cells=dict(values=[
            [b.name for b in bonds], [f"{y*100:.3f}" for y in mkt_yield],
            [f"{b.real_price:.3f}" for b in bonds], [f"{y*100:.3f}" for y in fit_yield_u],
            [f"{s:+.1f}" for s in spread_u_bp]])), row=3, col=1)
    fig1.update_layout(height=950, title="UK Index-Linked Gilt Real Yield Curve — Unadjusted", showlegend=True)
    fig1.update_yaxes(title_text="Real yield (%)", row=1, col=1)
    fig1.update_yaxes(title_text="bp", row=2, col=1)

    fig2 = make_subplots(rows=3, cols=1, row_heights=[0.42, 0.23, 0.35],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Seasonally adjusted real yield curve", "Spread to adjusted fitted curve (bp)", None),
        vertical_spacing=0.06)
    fig2.add_trace(go.Scatter(x=maturities, y=adj_mkt_yield * 100, mode="markers",
                               name="Seas.-adj. market yield", marker=dict(size=8)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=grid, y=zy_a(grid) * 100, mode="lines", name="Fitted spline (adj.)"), row=1, col=1)
    fig2.add_trace(go.Bar(x=[b.name for b in bonds], y=spread_a_bp, name="Spread (bp)"), row=2, col=1)
    fig2.add_trace(go.Table(
        header=dict(values=["Name", "Real Yield (%)", "Real Price", "CPI Proj. Nom. Price",
                             "CPI Proj. Nom. w/ Seasonal", "Seas.-Adj. Real Price", "Seas.-Adj. Real Yield (%)"],
                    fill_color="#1f2937", font=dict(color="white")),
        cells=dict(values=[
            [r["name"] for r in seas_rows], [f"{r['real_yield']*100:.3f}" for r in seas_rows],
            [f"{r['real_price']:.3f}" for r in seas_rows], [f"{r['nominal_price_plain']:.3f}" for r in seas_rows],
            [f"{r['nominal_price_seasonal']:.3f}" for r in seas_rows],
            [f"{r['seas_adj_real_price']:.3f}" for r in seas_rows],
            [f"{r['seas_adj_real_yield']*100:.3f}" for r in seas_rows]])), row=3, col=1)
    fig2.update_layout(height=950, title="UK Index-Linked Gilt Real Yield Curve — Seasonally Adjusted", showlegend=True)
    fig2.update_yaxes(title_text="Real yield (%)", row=1, col=1)
    fig2.update_yaxes(title_text="bp", row=2, col=1)

    html_parts = ["<html><head><title>UK Linker Real Yield Curve</title></head><body>",
                  fig1.to_html(full_html=False, include_plotlyjs="cdn"), "<hr>",
                  fig2.to_html(full_html=False, include_plotlyjs=False)]

    return dict(html_parts=html_parts, knots=knots, disc_unadj=disc_u, zero_yield_unadj=zy_u,
                disc_adj=disc_a, zero_yield_adj=zy_a, bonds_adj=bonds_adj, seas_rows=seas_rows,
                spread_unadj_bp=spread_u_bp, spread_adj_bp=spread_a_bp, out_path=out_path)


# ==========================================================================
# 12. Entry point — point this at your workbook
# ==========================================================================

if __name__ == "__main__":
    EXCEL_PATH = "uk_linker_data.xlsx"      # <-- your workbook
    VALUATION_DATE = "2026-08-26"           # <-- your pricing date
    HISTORY_PATH = "spread_history.csv"     # <-- persisted residual history
    LOOKBACK_DAYS = 60

    bonds = load_bonds_from_excel(EXCEL_PATH, VALUATION_DATE)
    rpi_forward_curve = load_rpi_curve_from_excel(EXCEL_PATH, VALUATION_DATE)
    seasonal_factor = load_seasonal_factors_from_excel(EXCEL_PATH)

    results = build_report(bonds, rpi_forward_curve, seasonal_factor)

    # --- one-off backfill (comment out once your history CSV is populated) ---
    snapshots = build_dated_bond_snapshots(EXCEL_PATH)
    print(f"Backfilling {len(snapshots)} historical dates...")
    backfill_spread_history(HISTORY_PATH, snapshots, seasonal_factor)

    # --- historical curve-fit diagnostic (debug non-mean-reverting residuals) ---
    build_historical_curve_diagnostic(snapshots, seasonal_factor, out_path="curve_diagnostic.html")

    # --- record today, compute z-scores, build all report sections ---
    today_spreads = compute_today_spreads(results["bonds_adj"], results["disc_adj"])
    append_spread_history(HISTORY_PATH, VALUATION_DATE, today_spreads)

    history_df = load_spread_history(HISTORY_PATH)
    z_df = compute_zscores(history_df, VALUATION_DATE, lookback_days=LOOKBACK_DAYS)
    zscore_fig = build_zscore_figure(z_df, lookback_days=LOOKBACK_DAYS)
    metric_bond_widget_html = build_metric_bond_widget_html(history_df, lookback_days=LOOKBACK_DAYS)

    html_parts = (
        results["html_parts"]
        + ["<hr>", zscore_fig.to_html(full_html=False, include_plotlyjs=False)]
        + ["<hr>", metric_bond_widget_html]
        + ["</body></html>"]
    )
    with open(results["out_path"], "w") as f:
        f.write("\n".join(html_parts))
    print(f"Report written to {results['out_path']}")
