from typing import List
from data_loader import get_iv_chain
import pandas as pd
import numpy as np


from iv_engine import bs_delta, bs_gamma, bs_vega
from data_loader import get_iv_chain


def remove_iv_outliers(df: pd.DataFrame, col: str = "iv", factor: float = 2.0) -> pd.DataFrame:
    """
    Remove outliers from a metric column using IQR method.
    Keeps points within [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    if df.empty or col not in df.columns:
        return df
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    
    return df[(df[col] >= lower) & (df[col] <= upper)]


def collect_iv_surface_data(
    ticker: str,
    expiries: List[str],
    q: float = 0.0,
    moneyness_min: float = 0.85,
    moneyness_max: float = 1.15,
    min_points: int = 5,  # reduced from 15 to allow more expiries
    calls_only_for_greeks: bool = True,  # Use only calls for cleaner Greek surfaces
) -> pd.DataFrame:
    """
    Collect IV & Greeks across several expiries for building surfaces.
    Uses 'surface_eligible' flag from compute_iv_for_chain.
    Now also removes outliers and relaxes constraints.
    
    For Greeks surfaces, we use calls only to avoid the discontinuity between
    call delta (0 to 1) and put delta (-1 to 0).
    """
    dfs = []
    for expiry in expiries:
        calls_iv, puts_iv, S, r = get_iv_chain(ticker, expiry, q=q)
        
        # For IV surface: use both calls and puts
        # For Greeks: use calls only (cleaner surfaces)
        temp = pd.concat([calls_iv, puts_iv], ignore_index=True)

        # Only those that passed quality checks
        temp = temp[temp.get("surface_eligible", False)]

        # Central moneyness band
        temp = temp[temp["moneyness"].between(moneyness_min, moneyness_max)]
        
        # Remove IV outliers per expiry
        temp = remove_iv_outliers(temp, col="iv", factor=1.5)

        if len(temp) >= min_points:
            temp["expiry"] = expiry
            temp = add_greeks(temp, q=q)
            dfs.append(temp)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    
    # Final global outlier pass
    combined = remove_iv_outliers(combined, col="iv", factor=2.0)
    
    return combined


import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def build_metric_surface_grid(
    surface_df: pd.DataFrame,
    metric: str = "iv",
    use_moneyness: bool = True,
    grid_size: int = 50,
    smooth_sigma: float = 1.0,
):
    """
    Build a regular grid for plotting a smooth surface of a given metric
    (e.g., 'iv', 'delta', 'gamma', 'vega_bs').
    
    For Greeks (delta, gamma, vega), uses calls only to avoid discontinuities.
    For IV, uses both calls and puts.
    """
    if metric not in surface_df.columns:
        return None, None, None

    valid = surface_df.dropna(subset=[metric]).copy()
    
    # For Greeks, use calls only to get proper continuous surfaces
    if metric in ["delta", "gamma", "vega_bs"]:
        if "option_type" in valid.columns:
            valid = valid[valid["option_type"] == "call"]
    
    if len(valid) < 10:
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

    # Try cubic interpolation first (smoother), fallback to linear
    Z_grid = griddata(
        points=(x, y),
        values=z,
        xi=(X_grid, Y_grid),
        method="cubic",
    )
    
    # Fill any NaN from cubic with linear interpolation
    nan_mask = np.isnan(Z_grid)
    if np.any(nan_mask):
        Z_linear = griddata(
            points=(x, y),
            values=z,
            xi=(X_grid, Y_grid),
            method="linear",
        )
        Z_grid = np.where(nan_mask, Z_linear, Z_grid)
    
    # Clamp values to valid ranges BEFORE smoothing
    if Z_grid is not None:
        if metric == "delta":
            Z_grid = np.clip(Z_grid, 0.0, 1.0)
        elif metric == "gamma":
            Z_grid = np.clip(Z_grid, 0.0, None)
        elif metric == "vega_bs":
            Z_grid = np.clip(Z_grid, 0.0, None)
        elif metric == "iv":
            Z_grid = np.clip(Z_grid, 0.001, 5.0)
    
    # Apply Gaussian smoothing
    if Z_grid is not None and smooth_sigma > 0:
        nan_mask = np.isnan(Z_grid)
        if not np.all(nan_mask):
            # Fill NaN with nearest valid value for smoothing
            from scipy.ndimage import distance_transform_edt
            
            if np.any(nan_mask):
                # Find nearest non-NaN value for each NaN cell
                indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
                Z_filled = Z_grid[tuple(indices)]
            else:
                Z_filled = Z_grid
            
            # Apply smoothing
            Z_smoothed = gaussian_filter(Z_filled, sigma=smooth_sigma)
            
            # Keep NaN only at the very edges where we have no data at all
            # (where nearest neighbor filling would be too far)
            Z_grid = Z_smoothed
    
    # Final clamp after smoothing
    if Z_grid is not None:
        if metric == "delta":
            Z_grid = np.clip(Z_grid, 0.0, 1.0)
        elif metric == "gamma":
            Z_grid = np.clip(Z_grid, 0.0, None)
        elif metric == "vega_bs":
            Z_grid = np.clip(Z_grid, 0.0, None)
        elif metric == "iv":
            Z_grid = np.clip(Z_grid, 0.001, 5.0)

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
