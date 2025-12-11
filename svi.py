# svi.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.optimize import least_squares


@dataclass
class SVIParameters:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray, params: SVIParameters) -> np.ndarray:
    """
    SVI total variance w(k) as a function of log-moneyness k.
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def fit_svi_smile(df: pd.DataFrame) -> Optional[Tuple[SVIParameters, pd.DataFrame]]:
    """
    Fit SVI to a single-expiry smile.

    df must have:
      - 'iv' (implied vol)
      - 'moneyness' or 'strike'
      - 'T' (time to expiry in years)

    Returns:
      (SVIParameters, df_with_k_w)
      or None if fitting fails.
    """
    df = df.dropna(subset=["iv"]).copy()
    if df.empty:
        return None

    # Use moneyness if present, else fallback to strike (normalized)
    if "moneyness" in df.columns:
        moneyness = df["moneyness"].values
    else:
        # Fallback: normalize strike by its median
        moneyness = df["strike"].values / np.median(df["strike"].values)

    # Approximate log-moneyness k = log(K/F). Here F ~ S, and K/S ~ moneyness.
    k = np.log(moneyness)

    # Use average T for this expiry (they're all the same or almost)
    T = float(df["T"].mean())
    if T <= 0:
        return None

    # Market total implied variance: w = (sigma^2) * T
    iv = df["iv"].values
    w_market = (iv ** 2) * T

    # Initial guess for parameters: a, b, rho, m, sigma
    a0 = np.maximum(w_market.min(), 1e-4)
    b0 = 0.5 * (w_market.max() - w_market.min())
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.2
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    # Bounds: ensure b > 0, sigma > 0, |rho| <= 1
    # (a can be slightly negative but we limit it)
    lower = np.array([-1.0, 1e-6, -0.999, -2.0, 1e-4])
    upper = np.array([+5.0, 10.0, 0.999, 2.0, 5.0])

    def residuals(x):
        params = SVIParameters(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
        w_model = svi_total_variance(k, params)
        return w_model - w_market

    try:
        res = least_squares(residuals, x0, bounds=(lower, upper), verbose=0)
        if not res.success:
            return None

        a, b, rho, m, sigma = res.x
        params = SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)

        # Add k and w_market to df for diagnostics
        df["k"] = k
        df["w_market"] = w_market

        return params, df
    except Exception:
        return None


def svi_iv_on_grid(params: SVIParameters, k_grid: np.ndarray, T: float) -> np.ndarray:
    """
    Given SVI parameters and a grid of log-moneyness k, return implied vols.
    """
    w = svi_total_variance(k_grid, params)
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / T)

def check_svi_butterfly_arbitrage(
    params: SVIParameters,
    k_min: float,
    k_max: float,
    n_points: int = 200,
    tol: float = -1e-4,
):
    """
    Check SVI smile for butterfly arbitrage via convexity of total variance.

    We compute total variance w(k) on a k-grid and approximate the
    second derivative w''(k). For an arbitrage-free smile we should
    have w''(k) >= 0 (up to numerical tolerance).

    Returns:
        has_arbitrage (bool), min_second_derivative (float)
    """
    k_grid = np.linspace(k_min, k_max, n_points)
    w = svi_total_variance(k_grid, params)

    dk = k_grid[1] - k_grid[0]
    # second derivative via central finite differences
    w2 = (w[:-2] - 2.0 * w[1:-1] + w[2:]) / (dk * dk)
    min_second = float(np.min(w2))

    has_arb = min_second < tol
    return has_arb, min_second

from typing import List, Dict, Tuple

def check_svi_calendar_arbitrage(
    svi_fits: List[Dict],
    k_min: float = -0.5,
    k_max: float = 0.5,
    n_points: int = 200,
    tol: float = -1e-4,
) -> List[Tuple[str, str, float]]:
    """
    Check SVI term structure for simple calendar arbitrage.

    Args:
        svi_fits: list of dicts like
            {"label": "2026-01-17", "T": 0.25, "params": SVIParameters(...)}
        k_min, k_max: k-range to test
        n_points: number of k points
        tol: tolerance; if w(T2,k) - w(T1,k) < tol for some k, we flag it

    Returns:
        violations: list of (label1, label2, min_diff) for each pair with arbitrage
                    where label1 has shorter maturity than label2.
    """
    if len(svi_fits) < 2:
        return []

    # sort by maturity
    svi_sorted = sorted(svi_fits, key=lambda d: d["T"])
    k_grid = np.linspace(k_min, k_max, n_points)

    violations = []

    for (fit1, fit2) in zip(svi_sorted[:-1], svi_sorted[1:]):
        T1, p1, lab1 = fit1["T"], fit1["params"], fit1["label"]
        T2, p2, lab2 = fit2["T"], fit2["params"], fit2["label"]

        w1 = svi_total_variance(k_grid, p1)
        w2 = svi_total_variance(k_grid, p2)

        diff = w2 - w1
        min_diff = float(np.min(diff))

        if min_diff < tol:
            violations.append((lab1, lab2, min_diff))

    return violations
