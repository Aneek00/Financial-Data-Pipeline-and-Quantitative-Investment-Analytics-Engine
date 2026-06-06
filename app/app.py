import streamlit as st
import pandas as pd
import warnings

# We only import the lightweight UI functions now.
from src.recommendations import categorize_funds, build_diversified_portfolio, calculate_suitability_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MF Quant Engine", layout="wide")

# ==========================================
# SIDEBAR: The Architecture Flex
# ==========================================
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown("""
    **Compute Layer (Offline / Air-Gapped):**
    * Multi-threaded AMFI API extraction.
    * Facebook Prophet ML models & Dual-Factor Backtesting.
    * Agglomerative Hierarchical Clustering.

    **Presentation Layer (Cloud):**
    * Decoupled, lightweight Streamlit UI.
    * Sub-second latency via pre-computed, Brotli-compressed Parquet matrices.
    """)
    st.markdown("---")
    st.markdown("*Built for institutional-grade quantitative research.*")

st.title("Mutual Fund Quant & Forecasting Engine")

# ==========================================
# CACHE & DATA LOADING (Strictly Read-Only)
# ==========================================
@st.cache_data(ttl=86400)
def load_data():
    df = pd.read_parquet("clean_nav_data.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=86400)
def load_master_stats():
    return pd.read_parquet("master_stats.parquet")

@st.cache_data(ttl=86400)
def load_correlation_matrix():
    return pd.read_parquet("correlation_matrix.parquet")

@st.cache_data(ttl=86400)
def load_forecasts():
    return pd.read_parquet("precomputed_forecasts.parquet")

@st.cache_data(ttl=86400)
def load_backtests():
    metrics = pd.read_parquet("backtest_metrics.parquet")
    charts = pd.read_parquet("backtest_charts.parquet")
    return metrics, charts

try:
    df = load_data()
    master_stats_df = load_master_stats()
except FileNotFoundError:
    st.error("Pre-computed data missing. Run the local backend scripts first.")
    st.stop()

funds_list = sorted(df["scheme_name"].unique())

tab1, tab2, tab3 = st.tabs(["Smart Portfolio", "Prophet ML Forecast", "Strategy Backtest"])

# ==========================================
# TAB 1: RECOMMENDATIONS & PORTFOLIO
# ==========================================
with tab1:
    st.subheader("Fund Recommendations & Smart Portfolio")

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
        # Load the pre-computed matrix instead of freezing the server
        try:
            correlation_matrix = load_correlation_matrix()

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
        except FileNotFoundError:
            st.warning("Correlation matrix missing. Please run shrink_data.py locally to generate it.")

# ==========================================
# TAB 2: PROPHET FORECAST (Clean Chart Fix)
# ==========================================
with tab2:
    st.subheader("1-Year Machine Learning NAV Forecast")

    try:
        forecast_df = load_forecasts()
        available_funds = sorted(forecast_df["scheme_name"].unique())

        selected_fund_forecast = st.selectbox(
            "Select a Top 50 Fund to view its forecast:",
            available_funds,
            key="forecast_fund"
        )

        # 1. Clean History
        hist = df[df["scheme_name"] == selected_fund_forecast].copy()
        hist["Date"] = pd.to_datetime(hist["date"]).dt.normalize()
        hist = hist.drop_duplicates(subset=["Date"], keep="last")
        hist_series = hist.set_index("Date")["nav"]

        # 2. Clean Forecast
        fut = forecast_df[forecast_df["scheme_name"] == selected_fund_forecast].copy()
        fut["Date"] = pd.to_datetime(fut["ds"]).dt.normalize()
        fut = fut.drop_duplicates(subset=["Date"], keep="last")
        fut_series = fut.set_index("Date")["yhat"]

        # 3. Clean Merge for Beautiful Multi-line Chart
        combined_chart_data = pd.DataFrame({
            "Historical NAV": hist_series,
            "Predicted NAV": fut_series
        })

        st.line_chart(combined_chart_data)
    except FileNotFoundError:
        st.warning("Forecasts missing. Run generate_forecasts.py locally to generate them.")

# ==========================================
# TAB 3: STRATEGY BACKTEST (Instant Load)
# ==========================================
with tab3:
    st.subheader("Uncompromised Strategy Backtest (Prophet ML + EMA)")

    try:
        bt_metrics, bt_charts = load_backtests()
        available_bt_funds = sorted(bt_metrics["scheme_name"].unique())

        selected_fund_bt = st.selectbox("Select a Top 50 Fund:", available_bt_funds, key="bt_fund")

        # Get metrics
        fund_metrics = bt_metrics[bt_metrics["scheme_name"] == selected_fund_bt].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{fund_metrics['Strategy_Ret'] * 100:.2f}%")
        col2.metric("Buy & Hold Return", f"{fund_metrics['Benchmark_Ret'] * 100:.2f}%")
        col3.metric("MAPE (Error)", f"{fund_metrics['MAPE']:.2f}%")
        col4.metric("Sharpe Ratio", f"{fund_metrics['Sharpe_Ratio']:.2f}")

        # Clean Chart Data
        chart_data = bt_charts[bt_charts["scheme_name"] == selected_fund_bt].copy()
        chart_data["Date"] = pd.to_datetime(chart_data["date"]).dt.normalize()
        chart_data = chart_data.drop_duplicates(subset=["Date"], keep="last").set_index("Date")

        chart_data = chart_data[["bench_cum", "strat_cum"]].rename(
            columns={"bench_cum": "Buy & Hold", "strat_cum": "Prophet Strategy"}
        )

        st.line_chart(chart_data)
    except FileNotFoundError:
        st.warning("Backtest metrics missing. Run generate_backtests.py locally to generate them.")