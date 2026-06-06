import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
import gc # IMPORTING GARBAGE COLLECTOR

# NO MORE sys.path HACKS! We import natively.
from src.recommendations import categorize_funds, build_diversified_portfolio, calculate_suitability_score
from src.models import run_strategy_backtest

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MF Quant Engine", layout="wide")
st.title("Mutual Fund Quant & Forecasting")

# 1. Cache Data Loading
@st.cache_data(ttl=86400)
def load_data():
    try:
        df = pd.read_parquet("clean_nav_data.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        st.error("Data file missing. Run the local pipeline and shrink_data.py first.")
        st.stop()

# 2. Cache Correlation Matrix
@st.cache_data(ttl=86400)
def get_correlation_matrix(data: pd.DataFrame, top_funds: list) -> pd.DataFrame:
    returns_pivot = data[data['scheme_name'].isin(top_funds)].pivot_table(
        index='date', columns='scheme_name', values='nav'
    ).pct_change().dropna()
    return returns_pivot.corr()

# 3. Load Pre-Computed Math (ZERO Cloud Compute required!)
@st.cache_data(ttl=86400)
def load_master_stats():
    try:
        return pd.read_parquet("master_stats.parquet")
    except FileNotFoundError:
        st.error("master_stats.parquet missing. Run shrink_data.py locally.")
        st.stop()

# 4. Load Pre-Computed ML Forecasts
@st.cache_data(ttl=86400)
def load_forecasts():
    try:
        return pd.read_parquet("precomputed_forecasts.parquet")
    except FileNotFoundError:
        st.error("precomputed_forecasts.parquet missing. Run generate_forecasts.py locally.")
        st.stop()

df = load_data()
funds_list = sorted(df["scheme_name"].unique())

tab1, tab2, tab3 = st.tabs(["Recommendations", "Prophet Forecast", "Strategy Backtest"])

# ==========================================
# TAB 1: RECOMMENDATIONS & PORTFOLIO
# ==========================================
with tab1:
    st.subheader("Fund Recommendations & Smart Portfolio")

    master_stats_df = load_master_stats()
    col1, col2 = st.columns(2)
    with col1:
        horizon_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=10, value=5)
    with col2:
        risk_profile = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=2)

    display_cols = [
        "scheme_name", "Suitability Score", "Expected Annual Return (%)",
        "Annualized Volatility (%)", "Sharpe Ratio"
    ]

    recs = master_stats_df.copy()
    recs["Suitability Score"] = recs.apply(
        lambda row: calculate_suitability_score(row, risk_profile, horizon_years) * 100, axis=1
    )
    recs = recs.sort_values("Suitability Score", ascending=False)

    core_funds, specialized_funds = categorize_funds(recs)

    st.markdown("### Top 15 Core Diversified Funds")
    st.dataframe(core_funds[display_cols].head(15), use_container_width=True)

    if not specialized_funds.empty:
        st.markdown("### Top 5 Specialized & Thematic Funds")
        st.dataframe(specialized_funds[display_cols].head(5), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛡️ Build Your Smart Diversified Portfolio")

    top_fund_names = core_funds.head(30)['scheme_name'].tolist()
    selected_anchor = st.selectbox("Choose your Anchor Fund:", top_fund_names)

    if selected_anchor:
        with st.spinner("Calculating Correlation Matrix..."):
            correlation_matrix = get_correlation_matrix(df, top_fund_names)

            anchor_row = core_funds[core_funds['scheme_name'] == selected_anchor]
            other_candidates = core_funds[core_funds['scheme_name'] != selected_anchor]
            custom_candidates = pd.concat([anchor_row, other_candidates])

            final_portfolio = build_diversified_portfolio(
                core_candidates=custom_candidates,
                correlation_matrix=correlation_matrix,
                portfolio_size=3,
                threshold=0.85
            )

            st.success(f"Generated low-correlation portfolio anchored around {selected_anchor}!")
            st.dataframe(final_portfolio[display_cols], use_container_width=True)

# ==========================================
# TAB 2: PROPHET FORECAST
# ==========================================
with tab2:
    st.subheader("1-Year Machine Learning NAV Forecast")

    # Load the pre-computed predictions
    forecast_df = load_forecasts()
    available_funds = sorted(forecast_df["scheme_name"].unique())

    selected_fund_forecast = st.selectbox(
        "Select a Top 50 Fund to view its forecast:",
        available_funds,
        key="forecast_fund"
    )

    # We don't even need a button anymore. It's instant!
    # Get historical data for the chart
    history = df[df["scheme_name"] == selected_fund_forecast][["date", "nav"]].copy()
    history = history.rename(columns={"date": "Date", "nav": "Historical NAV"}).set_index("Date")

    # Get predicted data for the chart
    future = forecast_df[forecast_df["scheme_name"] == selected_fund_forecast].copy()
    future = future.rename(columns={"ds": "Date", "yhat": "Predicted NAV"}).set_index("Date")

    # Merge them together so Streamlit can draw a beautiful chart
    combined_chart_data = history.join(future, how="outer")

    st.line_chart(combined_chart_data)
# ==========================================
# TAB 3: STRATEGY BACKTEST
# ==========================================
with tab3:
    st.subheader("Strategy Backtest (With Expense & Slippage)")
    selected_fund_bt = st.selectbox("Select Fund", funds_list, key="bt_fund")

    if st.button("Run Backtest"):
        with st.spinner("Running institutional backtest..."):
            fund_df = df[df["scheme_name"] == selected_fund_bt].copy()
            fund_df = fund_df.sort_values("date").drop_duplicates("date", keep="last")

            results = run_strategy_backtest(fund_df)

            if results is None:
                st.error("Model failed to converge. Insufficient volatility or data points.")
            else:
                strat_return = results["Strategy_Ret"] * 100
                bench_return = results["Benchmark_Ret"] * 100

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy Return", f"{strat_return:.2f}%")
                col2.metric("Buy & Hold Return", f"{bench_return:.2f}%")
                col3.metric("MAPE (Error)", f"{results['MAPE']:.2f}%")
                col4.metric("Sharpe Ratio", f"{results['Sharpe_Ratio']:.2f}")

                st.line_chart(results["TestData"].set_index("date")[["bench_cum", "strat_cum"]])

            # Clean up memory after backtest finishes
            gc.collect()