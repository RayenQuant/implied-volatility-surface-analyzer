import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from svi import fit_svi_smile, svi_iv_on_grid
from scipy.signal import savgol_filter

def plot_iv_surface(X, Y, Z, use_moneyness=True):
    x_label = "Moneyness (K/S)" if use_moneyness else "Strike"
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            showscale=True
        )
    ])

    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility"
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig



def _smooth_series(y: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """
    Apply a light Savitzky–Golay smoothing filter.
    If not enough points, return original array.
    """
    n = len(y)
    if n < 5:
        return y

    # window must be odd and <= n
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return y

    return savgol_filter(y, window_length=w, polyorder=polyorder)


def plot_vol_smile(df_expiry,
                   use_moneyness=True,
                   title_suffix="",
                   fit_svi=False,
                   smooth=True):
    """
    Plot IV vs moneyness/strike with line+markers for calls & puts,
    optionally smoothed, and optionally overlay an SVI fitted curve.
    """
    df_plot = df_expiry.dropna(subset=["iv"]).copy()
    if df_plot.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid implied volatilities.")
        return fig

    # Choose x-axis
    if use_moneyness:
        x_col = "moneyness"
        x_label = "Moneyness (K/S)"
    else:
        x_col = "strike"
        x_label = "Strike"

    fig = go.Figure()

    # Lines for calls & puts
    for opt_type, name in [("call", "Calls"), ("put", "Puts")]:
        sub = df_plot[df_plot["option_type"] == opt_type].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)

        y = sub["iv"].values
        if smooth:
            y = _smooth_series(y)

        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=y,
                mode="lines+markers",
                name=name,
            )
        )

    # Optional SVI overlay
    if fit_svi:
        fit_result = fit_svi_smile(df_plot)
        if fit_result is not None:
            params, df_with_k = fit_result

            x_vals = df_plot[x_col].values
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            x_grid = np.linspace(x_min, x_max, 200)

            # Convert to log-moneyness k
            if use_moneyness:
                k_grid = np.log(x_grid)
            else:
                S = float(df_plot["S"].iloc[0])
                moneyness_grid = x_grid / S
                k_grid = np.log(moneyness_grid)

            T = float(df_plot["T"].mean())
            iv_svi = svi_iv_on_grid(params, k_grid, T)

            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=iv_svi,
                    mode="lines",
                    name="SVI fit",
                    line=dict(width=2, dash="dot"),
                )
            )

    fig.update_layout(
        title=f"Volatility Smile {title_suffix}",
        xaxis_title=x_label,
        yaxis_title="Implied Volatility",
    )

    return fig


def plot_surface(X, Y, Z, x_label, y_label, z_label, title):
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig
