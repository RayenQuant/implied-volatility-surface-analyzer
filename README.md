# Implied Volatility Surface Analyzer

Interactive web application for visualizing and analyzing implied volatility smiles and surfaces for US equities and indices.

**Live demo:** _add the Streamlit URL here after deployment_

---

## Features

- Fetches live option chains from **Yahoo Finance** via `yfinance`
- Computes **implied volatility** using a custom **Newton–Raphson** Black–Scholes solver
- Cleans quotes (mid-price selection, liquidity filters, IV outlier removal)
- Builds **2D volatility smiles** (calls & puts) by moneyness (K/S)
- Fits a **parametric SVI model** to the smile and overlays the SVI curve
- Displays **3D surfaces** for:
  - Implied volatility
  - Delta, Gamma and Vega (Greeks)
- Performs **arbitrage checks**:
  - Butterfly arbitrage (convexity in strike)
  - Simple calendar arbitrage (total variance increasing in maturity)
- Generates automatic **commentary** on skew and term structure

---

## Tech Stack

- **Python**
- **Streamlit** for the web UI
- **NumPy, SciPy, Pandas** for numerical methods and data handling
- **Plotly** for interactive 2D & 3D visualizations
- **yfinance** for live options data

---

## Running Locally

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/implied-volatility-surface-analyzer.git
cd implied-volatility-surface-analyzer
