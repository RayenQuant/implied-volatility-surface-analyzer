import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Black-Scholes price for European call/put with continuous dividend yield q.
    S: spot
    K: strike
    T: time to maturity (in years)
    r: risk-free rate (annualized)
    sigma: volatility (annualized)
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        # At expiry, price is intrinsic
        intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
        return intrinsic

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """
    Vega of the option (derivative of price w.r.t sigma).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def implied_vol_newton(market_price, S, K, T, r, option_type='call', q=0.0,
                       initial_guess=0.2, tol=1e-6, max_iter=100,
                       sigma_min=1e-4, sigma_max=5.0):
    """
    Compute implied volatility using Newton-Raphson.
    Returns np.nan if no convergence or if market_price is inconsistent.
    """
    # Handle trivial / invalid cases
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # Basic no-arbitrage intrinsic bounds
    intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    if market_price < intrinsic:
        return np.nan  # below intrinsic, impossible

    sigma = float(initial_guess)

    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        vega = bs_vega(S, K, T, r, sigma, q)
        if vega <= 1e-8:  # almost flat, Newton will blow up
            break

        # Newton update
        sigma = sigma - diff / vega
        # Clamp to reasonable range
        sigma = min(max(sigma, sigma_min), sigma_max)

    # If we get here, no convergence
    return np.nan

def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes delta for European call/put with continuous dividend yield q.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        # put delta
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes gamma (same for calls and puts).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def implied_vol_with_status(market_price, S, K, T, r, option_type='call', q=0.0,
                            initial_guess=0.2, tol=1e-6, max_iter=100,
                            sigma_min=1e-4, sigma_max=5.0):
    """
    Returns (sigma, status)
    status ∈ {"ok", "no_price", "bad_inputs", "below_intrinsic", "low_vega", "no_convergence"}
    """
    if market_price <= 0:
        return np.nan, "no_price"
    if T <= 0 or S <= 0 or K <= 0:
        return np.nan, "bad_inputs"

    intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    if market_price < intrinsic:
        return np.nan, "below_intrinsic"

    sigma = float(initial_guess)

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma, "ok"

        vega = bs_vega(S, K, T, r, sigma, q)
        if vega <= 1e-8:
            return np.nan, "low_vega"

        sigma -= diff / vega
        sigma = min(max(sigma, sigma_min), sigma_max)

    return np.nan, "no_convergence"

