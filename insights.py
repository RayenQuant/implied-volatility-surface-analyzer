# insights.py
import numpy as np
import pandas as pd

def _safe_pct(x):
    if np.isnan(x):
        return "n/a"
    return f"{x:.1%}"


def analyze_smile(df: pd.DataFrame, ticker: str, expiry: str) -> str:
    """
    Generate a short commentary for one expiry's smile.
    Expects df to have columns: ['iv', 'moneyness', 'option_type', 'T'].
    Uses only rows with non-null iv.
    """
    txt = []

    df = df.dropna(subset=["iv", "moneyness"]).copy()
    if df.empty:
        return f"For {ticker} {expiry}, no reliable implied volatilities are available."

    # ATM region (closest to moneyness 1)
    df["atm_dist"] = (df["moneyness"] - 1.0).abs()
    atm_row = df.sort_values("atm_dist").iloc[0]
    atm_iv = atm_row["iv"]
    txt.append(
        f"For {ticker} {expiry}, the at-the-money implied volatility is around "
        f"{_safe_pct(atm_iv)}."
    )

    # Overall level
    iv_mean = df["iv"].mean()
    iv_min = df["iv"].min()
    iv_max = df["iv"].max()
    txt.append(
        f"Implied volatilities range roughly between {_safe_pct(iv_min)} and "
        f"{_safe_pct(iv_max)}, with an average level near {_safe_pct(iv_mean)}."
    )

    # Skew: slope of iv vs moneyness
    skew_df = df[["moneyness", "iv"]].sort_values("moneyness")
    if len(skew_df) >= 3:
        slope = (skew_df["iv"].iloc[-1] - skew_df["iv"].iloc[0]) / (
            skew_df["moneyness"].iloc[-1] - skew_df["moneyness"].iloc[0]
        )

        if slope < -0.1:
            skew_desc = "a pronounced downside skew (puts richer than calls)"
        elif slope < -0.02:
            skew_desc = "a moderate downside skew"
        elif slope > 0.1:
            skew_desc = "an upside skew (call wing trading at higher vols)"
        elif slope > 0.02:
            skew_desc = "a mild upside skew"
        else:
            skew_desc = "a fairly flat smile"

        txt.append(
            f"The smile exhibits {skew_desc}, based on the slope of implied volatility "
            "across strikes."
        )

    # Curvature: difference between wings and ATM
    low = df.nsmallest(max(3, len(df) // 5), "moneyness")["iv"].mean()
    high = df.nlargest(max(3, len(df) // 5), "moneyness")["iv"].mean()
    wings = np.nanmean([low, high])
    curvature = wings - atm_iv
    if curvature > 0.05:
        txt.append(
            "Both wings trade at significantly higher vol than the ATM region, "
            "indicating a pronounced smile and demand for convex payoff structures."
        )
    elif curvature > 0.01:
        txt.append(
            "There is some convexity in the smile, with wings slightly more expensive "
            "than the ATM region."
        )
    else:
        txt.append(
            "The wings are not much richer than ATM, so smile convexity is limited."
        )

    return " ".join(txt)


def analyze_term_structure(surface_df: pd.DataFrame, ticker: str) -> str:
    """
    Simple commentary on term structure using surface_df.
    Expects columns: ['T', 'iv'] (and possibly multiple expiries).
    """
    df = surface_df.dropna(subset=["iv", "T"]).copy()
    if df.empty:
        return ""

    # bucket by maturity (short/medium/long)
    short = df[df["T"] <= 0.1]["iv"]
    medium = df[(df["T"] > 0.1) & (df["T"] <= 0.5)]["iv"]
    long = df[df["T"] > 0.5]["iv"]

    parts = []
    if not short.empty:
        parts.append(f"short-dated tenors ~{_safe_pct(short.mean())}")
    if not medium.empty:
        parts.append(f"medium maturities ~{_safe_pct(medium.mean())}")
    if not long.empty:
        parts.append(f"long-dated tenors ~{_safe_pct(long.mean())}")

    if parts:
        summary = ", ".join(parts)
        text = f"Across the term structure for {ticker}, average implied vols are {summary}."
    else:
        text = ""

    # Compare short vs long
    if not short.empty and not long.empty:
        diff = short.mean() - long.mean()
        if diff > 0.05:
            text += " Short-term vols are markedly higher than long-dated vols, consistent with event risk priced in the front end."
        elif diff > 0.01:
            text += " Short-end vols are slightly elevated relative to long-dated tenors."
        elif diff < -0.05:
            text += " Long-dated vols are higher than the front end, suggesting structurally elevated uncertainty over the long run."
        elif diff < -0.01:
            text += " The back end of the curve is somewhat richer in vol than the very front."
        else:
            text += " The term structure is relatively flat in volatility terms."

    return text
