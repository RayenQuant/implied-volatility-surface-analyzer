import streamlit as st
import pandas as pd
from svi import fit_svi_smile, check_svi_butterfly_arbitrage, check_svi_calendar_arbitrage


from data_loader import get_expiration_dates, get_iv_chain, filter_smile_data
from surface_model import collect_iv_surface_data, build_metric_surface_grid
from plots import plot_iv_surface, plot_vol_smile, plot_surface
from insights import analyze_smile, analyze_term_structure

st.set_page_config(page_title="Implied Volatility Surface Analyzer", layout="wide")

st.title("Implied Volatility Surface Analyzer")

# Sidebar: inputs
ticker = st.sidebar.text_input("Ticker (US Equity)", value="SPY")
st.sidebar.write("Example: SPY, AAPL, TSLA")

if ticker:
    expiries = get_expiration_dates(ticker)
    selected_expiry = st.sidebar.selectbox("Select expiration", expiries)

    # For full surface, allow user to choose how many expiries
    num_expiries = st.sidebar.slider("Number of expiries for surface", 1, min(5, len(expiries)), 3)
    surface_expiries = expiries[:num_expiries]

    tabs = st.tabs(["3D Surface", "2D Smile", "Raw Data & Metrics"])

    # --- 2D Smile & Raw Data use selected_expiry ---
    with tabs[1]:
        st.subheader(f"Volatility Smile – {ticker} – {selected_expiry}")

        calls_iv, puts_iv, S, r = get_iv_chain(ticker, selected_expiry)
        df_expiry = pd.concat([calls_iv, puts_iv], ignore_index=True)

        # Raw IVs (for backup / debug)
        df_raw = df_expiry.dropna(subset=["iv"]).copy()

        # Filtered data for plotting & SVI
        df_for_smile = filter_smile_data(df_raw)

        if df_for_smile.empty:
            st.warning("No valid smile points after filtering – showing raw IVs instead.")
            data_for_plot = df_raw
            smooth_flag = False
        else:
            data_for_plot = df_for_smile
            smooth_flag = True

        if data_for_plot.empty:
            st.warning("No valid implied vols for this expiry.")
        else:
            fit_svi_flag = st.checkbox("Overlay SVI fit on smile", value=True)

            smile_fig = plot_vol_smile(
                data_for_plot,
                use_moneyness=True,
                title_suffix=f"({selected_expiry})",
                fit_svi=fit_svi_flag,
                smooth=smooth_flag,
            )
            st.plotly_chart(smile_fig, use_container_width=True)

            # --- SVI model + parameters + arbitrage check ---
            if fit_svi_flag:
                fit_result = fit_svi_smile(data_for_plot)
                if fit_result is not None:
                    params, df_with_k = fit_result

                    st.markdown("### SVI model")

                    # SVI formulas (now rendered correctly)
                    st.markdown("**Total implied variance (SVI)**")
                    st.latex(
                        r"w(k) = a + b\left(\rho (k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)"
                    )

                    st.markdown(
                        "where $k = \\log(K/F)$ is log-moneyness and "
                        "$w(k) = \\sigma_{\\text{imp}}(k)^2 \\, T$."
                    )

                    st.markdown("**Implied volatility from SVI**")
                    st.latex(r"\sigma_{\text{imp}}(k) = \sqrt{\frac{w(k)}{T}}")

                    # Parameters
                    st.markdown("#### Fitted SVI parameters")
                    st.write(
                        {
                            "a": params.a,
                            "b": params.b,
                            "rho": params.rho,
                            "m": params.m,
                            "sigma": params.sigma,
                        }
                    )

                    # --- Butterfly arbitrage check ---
                    k_vals = df_with_k["k"].values
                    if len(k_vals) >= 3:
                        k_min, k_max = float(k_vals.min()), float(k_vals.max())
                        has_arb, min_second = check_svi_butterfly_arbitrage(
                            params, k_min, k_max
                        )
                        if has_arb:
                            st.error(
                                f"⚠ This SVI smile exhibits butterfly arbitrage "
                                f"(min w''(k) ≈ {min_second:.3e})."
                            )
                        else:
                            st.success(
                                f"✓ No butterfly arbitrage detected on the SVI smile "
                                f"(min w''(k) ≈ {min_second:.3e})."
                            )
                    else:
                        st.info("Not enough k-points to run a reliable butterfly check.")
                else:
                    st.info("SVI fit did not converge for this smile.")
    




    with tabs[2]:
        st.subheader("Options Chain with Implied Volatility")
        st.dataframe(
            df_expiry[["option_type", "strike", "T", "mid", "iv", "moneyness", "surface_eligible","iv_status"]]
        )

        # Metrics based on eligible rows
        valid_iv = df_expiry[df_expiry["surface_eligible"]].dropna(subset=["iv"])
        if not valid_iv.empty:
            st.write(f"Average IV (eligible points): {valid_iv['iv'].mean():.2%}")
            skew = valid_iv[["moneyness", "iv"]].sort_values("moneyness")
            if len(skew) > 2:
                slope = (skew["iv"].iloc[-1] - skew["iv"].iloc[0]) / (
                    skew["moneyness"].iloc[-1] - skew["moneyness"].iloc[0]
                )
                st.write(f"Approx. skew steepness: {slope:.2f} vol per unit moneyness")
        else:
            st.write("Average IV: n/a (no valid points).")

        # --- NEW: Auto commentary ---
        st.markdown("### Volatility Commentary")
        smile_comment = analyze_smile(valid_iv, ticker, selected_expiry)
        st.write(smile_comment)


    # --- 3D Surface using multiple expiries ---


# inside the "3D Surface" tab:
    with tabs[0]:
        st.subheader(f"IV and GREEKS Surfaces – {ticker}")

        # choose metric
        metric_name = st.selectbox(
            "Metric for surface",
            ["Implied Volatility", "Delta", "Gamma", "Vega"],
            index=0,
        )

        metric_map = {
            "Implied Volatility": ("iv", "Implied Volatility"),
            "Delta": ("delta", "Delta"),
            "Gamma": ("gamma", "Gamma"),
            "Vega": ("vega_bs", "Vega"),
        }
        metric_col, z_label = metric_map[metric_name]

        surface_df = collect_iv_surface_data(ticker, surface_expiries)

        if surface_df.empty:
            st.warning("Not enough data to build the surface (no valid points).")
        else:
            X, Y, Z = build_metric_surface_grid(surface_df, metric=metric_col)
            if X is None:
                st.warning(f"Not enough valid {metric_col} data for a surface.")
            else:
                fig_surface = plot_surface(
                    X,
                    Y,
                    Z,
                    x_label="Moneyness (K/S)",
                    y_label="Time to Expiry (years)",
                    z_label=z_label,
                    title=f"{metric_name} Surface",
                )
                st.plotly_chart(fig_surface, use_container_width=True)
            svi_fits = []
            for expiry in surface_expiries:
                # build a per-expiry df for smile
                calls_iv, puts_iv, S, r = get_iv_chain(ticker, expiry)
                df_e = pd.concat([calls_iv, puts_iv], ignore_index=True)
                df_e = df_e.dropna(subset=["iv"]).copy()
                if df_e.empty:
                    continue

                fit_result = fit_svi_smile(df_e)
                if fit_result is None:
                    continue

                params, df_with_k = fit_result
                T_e = float(df_with_k["T"].mean())
                svi_fits.append({"label": expiry, "T": T_e, "params": params})

            if len(svi_fits) >= 2:
                cal_viol = check_svi_calendar_arbitrage(svi_fits)
                st.markdown("### Calendar Arbitrage Check (SVI)")

                if not cal_viol:
                    st.success("✓ No calendar arbitrage detected between the fitted SVI smiles.")
                else:
                    for lab1, lab2, min_diff in cal_viol:
                        st.error(
                            f"⚠ Calendar arbitrage: SVI total variance at some strikes "
                            f"is lower for {lab2} than for shorter maturity {lab1} "
                            f"(min w(T2,k) - w(T1,k) ≈ {min_diff:.3e})."
                        )
            else:
                st.info("Not enough SVI fits available to run a calendar arbitrage check.")        
