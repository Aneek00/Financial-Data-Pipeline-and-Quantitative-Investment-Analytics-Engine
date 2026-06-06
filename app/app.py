import streamlit as st
import pandas as pd
import warnings
import plotly.graph_objects as go
import sys
import os
import gc

# 🚨 THE PATH RESCUE 🚨
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.recommendations import categorize_funds, build_diversified_portfolio, calculate_suitability_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MF Quant Engine", layout="wide")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown("""
    **Compute Layer (Offline / Air-Gapped):**
    * Multi-threaded AMFI API extraction.
    * Facebook Prophet ML models & Dual-Factor Backtests.
    * Agglomerative Hierarchical Clustering.

    **Presentation Layer (Cloud):**
    * Decoupled, lightweight Streamlit UI.
    * Fragmented UI rendering for sub-second latency.
    """)
    st.markdown("---")
    st.markdown("*Built for institutional-grade quantitative research.*")

st.title("Mutual Fund Quant & Forecasting Engine")

# ==========================================
# CACHE & DATA LOADING
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
def get_correlation_matrix(df_clean: pd.DataFrame, top_funds: list) -> pd.DataFrame:
    try:
        return pd.read_parquet("correlation_matrix.parquet")
    except FileNotFoundError:
        returns_pivot = df_clean[df_clean['scheme_name'].isin(top_funds)].pivot_table(
            index='date', columns='scheme_name', values='nav'
        ).pct_change().dropna()
        return returns_pivot.corr()

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
# FRAGMENT 1: RECOMMENDATIONS
# ==========================================
@st.fragment
def render_tab1():
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
    st.dataframe(core_funds[display_cols].head(15), width="stretch") # FIX FOR WARNINGS

    if not specialized_funds.empty:
        st.markdown("### Top 5 Specialized & Thematic Funds")
        st.dataframe(specialized_funds[display_cols].head(5), width="stretch") # FIX FOR WARNINGS

    st.markdown("---")
    st.markdown("### 🛡️ Build Your Smart Diversified Portfolio")

    top_fund_names = core_funds.head(30)['scheme_name'].tolist()
    selected_anchor = st.selectbox("Choose your Anchor Fund:", top_fund_names)

    if selected_anchor:
        with st.spinner("Analyzing correlation matrix..."):
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
            st.dataframe(final_portfolio[display_cols], width="stretch") # FIX FOR WARNINGS

            gc.collect()

# ==========================================
# FRAGMENT 2: FORECASTS
# ==========================================
@st.fragment
def render_tab2():
    st.subheader("1-Year Machine Learning NAV Forecast")

    try:
        forecast_df = load_forecasts()
        available_funds = sorted(forecast_df["scheme_name"].unique())

        selected_fund_forecast = st.selectbox(
            "Select a Top 50 Fund to view its forecast:",
            available_funds,
            key="forecast_fund"
        )

        hist = df[df["scheme_name"] == selected_fund_forecast].copy()
        hist["Date"] = pd.to_datetime(hist["date"]).dt.normalize()
        hist = hist.drop_duplicates(subset=["Date"], keep="last")
        hist_series = hist.set_index("Date")["nav"]

        fut = forecast_df[forecast_df["scheme_name"] == selected_fund_forecast].copy()
        fut["Date"] = pd.to_datetime(fut["ds"]).dt.normalize()
        fut = fut.drop_duplicates(subset=["Date"], keep="last")
        fut_series = fut.set_index("Date")["yhat"]

        # --- THE CLASSIC PROPHET AESTHETIC ---
        fig = go.Figure()

        # Actuals: Black Dots
        fig.add_trace(go.Scatter(
            x=hist_series.index, y=hist_series.values,
            mode='markers', name='Actual NAV',
            marker=dict(color='black', size=4)
        ))

        # Forecast: Solid Blue Line
        fig.add_trace(go.Scatter(
            x=fut_series.index, y=fut_series.values,
            mode='lines', name='Predicted Trend',
            line=dict(color='#0072B2', width=2)
        ))

        fig.update_layout(
            title="NAV Projection",
            xaxis_title="Date",
            yaxis_title="Net Asset Value",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',         # Force white background
            paper_bgcolor='white',        # Force white outer background
            font=dict(color='black'),     # Force black text
            xaxis=dict(showgrid=True, gridcolor='#E5E5E5', linecolor='black'),
            yaxis=dict(showgrid=True, gridcolor='#E5E5E5', linecolor='black')
        )
        st.plotly_chart(fig, width="stretch")

        # Kill Plotly object from memory immediately after drawing
        del fig, hist, fut
        gc.collect()

    except FileNotFoundError:
        st.warning("Forecasts missing. Run generate_forecasts.py locally to generate them.")

# ==========================================
# FRAGMENT 3: BACKTESTS
# ==========================================
@st.fragment
def render_tab3():
    st.subheader("Uncompromised Strategy Backtest (Prophet ML + EMA)")

    try:
        bt_metrics, bt_charts = load_backtests()
        available_bt_funds = sorted(bt_metrics["scheme_name"].unique())

        selected_fund_bt = st.selectbox("Select a Top 50 Fund:", available_bt_funds, key="bt_fund")

        fund_metrics = bt_metrics[bt_metrics["scheme_name"] == selected_fund_bt].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{fund_metrics['Strategy_Ret'] * 100:.2f}%")
        col2.metric("Buy & Hold Return", f"{fund_metrics['Benchmark_Ret'] * 100:.2f}%")
        col3.metric("MAPE (Error)", f"{fund_metrics['MAPE']:.2f}%")
        col4.metric("Sharpe Ratio", f"{fund_metrics['Sharpe_Ratio']:.2f}")

        chart_data = bt_charts[bt_charts["scheme_name"] == selected_fund_bt].copy()
        chart_data["Date"] = pd.to_datetime(chart_data["date"]).dt.normalize()
        chart_data = chart_data.drop_duplicates(subset=["Date"], keep="last").set_index("Date")

        chart_data = chart_data[["bench_cum", "strat_cum"]].rename(
            columns={"bench_cum": "Buy & Hold", "strat_cum": "Prophet Strategy"}
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Buy & Hold"], mode='lines', name='Buy & Hold Benchmark', line=dict(color='#a3a8b8')))
        fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Prophet Strategy"], mode='lines', name='Prophet AI Strategy', line=dict(color='#00ff88', width=2)))

        fig2.update_layout(
            title="Cumulative Strategy Returns", xaxis_title="Date", yaxis_title="Return (%)",
            yaxis_tickformat='.1%', hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig2, width="stretch") # FIX FOR WARNINGS

        # Kill Plotly object from memory immediately after drawing
        del fig2, chart_data
        gc.collect()

    except FileNotFoundError:
        st.warning("Backtest metrics missing. Run generate_backtests.py locally to generate them.")

# ==========================================
# RENDER THE TABS
# ==========================================
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()