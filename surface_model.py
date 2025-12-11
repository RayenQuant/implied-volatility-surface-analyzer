from typing import List
from data_loader import get_iv_chain
import pandas as pd
import numpy as np


from iv_engine import bs_delta, bs_gamma, bs_vega
from data_loader import get_iv_chain
def collect_iv_surface_data(
    ticker: str,
    expiries: List[str],
    q: float = 0.0,
    moneyness_min: float = 0.8,
    moneyness_max: float = 1.2,
    min_points: int = 15,
) -> pd.DataFrame:
    """
    Collect IV & Greeks across several expiries for building surfaces.
    Uses 'surface_eligible' flag from compute_iv_for_chain.
    """
    dfs = []
    for expiry in expiries:
        calls_iv, puts_iv, S, r = get_iv_chain(ticker, expiry, q=q)
        temp = pd.concat([calls_iv, puts_iv], ignore_index=True)

        # Only those that passed quality checks
        temp = temp[temp.get("surface_eligible", False)]

        # Central moneyness band
        temp = temp[temp["moneyness"].between(moneyness_min, moneyness_max)]

        if len(temp) >= min_points:
            temp["expiry"] = expiry
            temp = add_greeks(temp, q=q)  # <--- compute Greeks here
            dfs.append(temp)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


import numpy as np
from scipy.interpolate import griddata

def build_metric_surface_grid(
    surface_df: pd.DataFrame,
    metric: str = "iv",
    use_moneyness: bool = True,
    grid_size: int = 50,
):
    """
    Build a regular grid for plotting a smooth surface of a given metric
    (e.g., 'iv', 'delta', 'gamma', 'vega_bs').
    """
    if metric not in surface_df.columns:
        return None, None, None

    valid = surface_df.dropna(subset=[metric])
    if len(valid) < 20:
        return None, None, None

    if use_moneyness:
        x = valid["moneyness"].values
    else:
        x = valid["strike"].values
    y = valid["T"].values
    z = valid[metric].values

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    Z_grid = griddata(
        points=(x, y),
        values=z,
        xi=(X_grid, Y_grid),
        method="linear",   # keep linear for stability
    )

    return X_grid, Y_grid, Z_grid

def add_greeks(df: pd.DataFrame, q: float = 0.0) -> pd.DataFrame:
    """
    Compute delta, gamma, vega for each row where IV is available.
    Adds 'delta', 'gamma', 'vega' columns (NaN if IV missing).
    """
    df = df.copy()

    def _row_greeks(row):
        if pd.isna(row["iv"]) or row["T"] <= 0:
            return pd.Series({"delta": np.nan, "gamma": np.nan, "vega_bs": np.nan})

        S = row["S"]
        K = row["strike"]
        T = row["T"]
        r = row.get("r", 0.0)
        sigma = row["iv"]
        opt_type = row["option_type"]

        delta = bs_delta(S, K, T, r, sigma, option_type=opt_type, q=q)
        gamma = bs_gamma(S, K, T, r, sigma, q=q)
        vega_val = bs_vega(S, K, T, r, sigma, q=q)

        return pd.Series({"delta": delta, "gamma": gamma, "vega_bs": vega_val})

    greeks = df.apply(_row_greeks, axis=1)
    df[["delta", "gamma", "vega_bs"]] = greeks
    return df
