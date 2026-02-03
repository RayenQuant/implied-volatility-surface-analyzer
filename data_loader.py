# data_loader.py
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# --- Smile filtering thresholds ---
SMILE_MIN_MID = 0.10           # ignore very cheap options (< 10 cents)
SMILE_MONEYNESS_MIN = 0.7      # central region only
SMILE_MONEYNESS_MAX = 1.3
SMILE_IV_FACTOR_LOW = 0.4      # keep iv in [0.4 * median, 1.6 * median]
SMILE_IV_FACTOR_HIGH = 1.6

def get_underlying_price(ticker: str) -> float:
    y_ticker = yf.Ticker(ticker)
    data = y_ticker.history(period="1d")
    if data.empty:
        raise ValueError("No price data found for ticker.")
    return float(data["Close"].iloc[-1])


def get_expiration_dates(ticker: str):
    y_ticker = yf.Ticker(ticker)
    return list(y_ticker.options)  # list of strings like '2025-01-17'

def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")
        # ^TNX is in percentage points (e.g. 4.50), convert to decimal
        return float(tnx["Close"].iloc[-1]) / 100.0
    except Exception:
        # fallback constant (e.g. 2%)
        return 0.02

def get_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_ticker = yf.Ticker(ticker)
    chain = y_ticker.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    return calls, puts

from iv_engine import implied_vol_newton  # already there

def get_iv_chain(ticker: str, expiry: str, q: float = 0.0):
    S = get_underlying_price(ticker)
    r = get_risk_free_rate()
    calls_raw, puts_raw = get_option_chain(ticker, expiry)

    calls = prepare_options_df(calls_raw, S, expiry, "call")
    puts = prepare_options_df(puts_raw, S, expiry, "put")

    calls_iv = compute_iv_for_chain(calls, r, q)
    puts_iv = compute_iv_for_chain(puts, r, q)

    # Keep r in each row (useful for Greeks)
    calls_iv["r"] = r
    puts_iv["r"] = r

    return calls_iv, puts_iv, S, r


def prepare_options_df(df: pd.DataFrame, S: float, expiry: str, option_type: str) -> pd.DataFrame:
    """
    Build a clean options DataFrame and compute:
      - mid: price used for IV (bid/ask if available, else lastPrice)
      - T: time to expiry in years
      - moneyness: K/S
    We only drop rows where we truly have no usable price.
    """
    df = df.copy()

    # Cast to float & fill NaNs
    for col in ["bid", "ask", "lastPrice"]:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0.0)
        else:
            df[col] = 0.0

    bid = df["bid"].values
    ask = df["ask"].values
    last = df["lastPrice"].values

    # Priority: (bid,ask) mid if both > 0; then bid; then ask; then lastPrice
    mid = np.where(
        (bid > 0) & (ask > 0),
        0.5 * (bid + ask),
        np.where(
            bid > 0,
            bid,
            np.where(
                ask > 0,
                ask,
                np.where(last > 0, last, np.nan)  # fallback to lastPrice
            ),
        ),
    )
    df["mid"] = mid

    # Drop only rows where we still have no price
    df = df.dropna(subset=["mid"])
    df = df[df["mid"] > 0]

    # Time to expiry in years
    exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    today = dt.date.today()
    T_days = (exp_date - today).days
    df["T"] = np.maximum(T_days / 365.0, 0.0)

    df["option_type"] = option_type
    df["S"] = S
    df["moneyness"] = df["strike"] / S

    return df

from iv_engine import implied_vol_newton
from iv_engine import implied_vol_with_status

# Global quality thresholds (tweak if needed)
MIN_MID = 0.05       # ignore options priced below 5 cents
MIN_OI = 0           # no open interest requirement (many valid options have OI=0)
IV_MIN = 0.01        # min allowed IV  = 1%
IV_MAX = 3.0         # max allowed IV  = 300%



def compute_iv_for_chain(df: pd.DataFrame, r: float, q: float = 0.0) -> pd.DataFrame:
    df = df.copy()

    def _compute_row_iv(row):
        sigma, status = implied_vol_with_status(
            market_price=row["mid"],
            S=row["S"],
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type=row["option_type"],
            q=q,
        )
        return pd.Series({"iv": sigma, "iv_status": status})

    iv_res = df.apply(_compute_row_iv, axis=1)
    df["iv"] = iv_res["iv"]
    df["iv_status"] = iv_res["iv_status"]
    df = df.replace([np.inf, -np.inf], np.nan)

    # quality filters for the surface
    cond_price = df["mid"] >= MIN_MID
    cond_oi = df.get("openInterest", 0).fillna(0) >= MIN_OI
    cond_iv_range = df["iv"].between(IV_MIN, IV_MAX).fillna(False)
    df["surface_eligible"] = cond_price & cond_oi & cond_iv_range & (df["iv_status"] == "ok")

    return df

def filter_smile_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean options data for smile plotting:
    - drop rows with NaN iv / moneyness / mid
    - remove very cheap options (mid < SMILE_MIN_MID)
    - restrict to a central moneyness band
    - remove IV outliers relative to the median
    - keep only rows where IV status is 'ok' (if present)
    """
    df = df.dropna(subset=["iv", "moneyness", "mid"]).copy()
    if df.empty:
        return df

    # 1) price filter
    df = df[df["mid"] >= SMILE_MIN_MID]
    if df.empty:
        return df

    # 2) moneyness band
    df = df[df["moneyness"].between(SMILE_MONEYNESS_MIN, SMILE_MONEYNESS_MAX)]
    if df.empty:
        return df

    # 3) IV outlier filter
    median_iv = df["iv"].median()
    low = median_iv * SMILE_IV_FACTOR_LOW
    high = median_iv * SMILE_IV_FACTOR_HIGH
    df = df[df["iv"].between(low, high)]
    if df.empty:
        return df

    # 4) solver status filter (if available)
    if "iv_status" in df.columns:
        df = df[df["iv_status"] == "ok"]

    return df

