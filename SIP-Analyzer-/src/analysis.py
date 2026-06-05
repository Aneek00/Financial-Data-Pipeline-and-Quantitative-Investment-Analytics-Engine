import pandas as pd
import numpy as np

def calculate_cagr(start_value: pd.Series, end_value: pd.Series, years: pd.Series) -> pd.Series:
    """
    Vectorized CAGR calculation for high performance.
    Handles NaN values and prevents division by zero or negative timelines.
    """
    years = pd.to_numeric(years, errors='coerce')
    start_value = pd.to_numeric(start_value, errors='coerce')
    end_value = pd.to_numeric(end_value, errors='coerce')

    result = pd.Series(np.nan, index=start_value.index)
    valid_mask = (start_value > 0) & (end_value > 0) & (years > 0.25)

    s = start_value[valid_mask]
    e = end_value[valid_mask]
    y = years[valid_mask]

    result.loc[valid_mask] = ((e / s) ** (1 / y) - 1) * 100
    return result

def generate_statistical_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Expected Return, Annualized Volatility, and Sharpe Ratio
    for all funds in the dataset.
    """
    if 'daily_return' not in df.columns:
        df['daily_return'] = df.groupby('scheme_code')['nav'].pct_change()

    daily_returns_grouped = df.groupby('scheme_code')['daily_return']
    stats_df = pd.DataFrame()
    stats_df['mean'] = daily_returns_grouped.mean()
    stats_df['std'] = daily_returns_grouped.std()

    stats_df['Expected Annual Return (%)'] = stats_df['mean'] * 252 * 100
    stats_df['Annualized Volatility (%)'] = stats_df['std'] * np.sqrt(252) * 100

    # Assuming 4% annual risk-free rate
    risk_free_rate_daily = (1.04 ** (1/252)) - 1
    stats_df['Sharpe Ratio'] = (stats_df['mean'] - risk_free_rate_daily) / stats_df['std'] * np.sqrt(252)

    return stats_df.reset_index()

def calculate_technical_indicators(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical indicators (SMAs, Bollinger Bands, RSI)
    for a specific fund's deep-dive analysis.
    """
    df = fund_df.copy()
    df['50_day_sma'] = df['nav'].rolling(window=50).mean()
    df['200_day_sma'] = df['nav'].rolling(window=200).mean()
    df['20_day_sma'] = df['nav'].rolling(window=20).mean()
    df['20_day_std'] = df['nav'].rolling(window=20).std()

    df['bollinger_upper'] = df['20_day_sma'] + (df['20_day_std'] * 2)
    df['bollinger_lower'] = df['20_day_sma'] - (df['20_day_std'] * 2)

    delta = df['nav'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df

def precompute_fund_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes institutional-grade statistics: Geometric Return (CAGR),
    accurate fund age, and protected Sharpe Ratios.
    """
    latest_date = data["date"].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)

    # 1. Age Calculation (Exact days, not row count)
    age_df = data.groupby("scheme_name").agg(
        min_date=("date", "min"),
        max_date=("date", "max")
    )
    age_df["Fund Age (Yrs)"] = (age_df["max_date"] - age_df["min_date"]).dt.days / 365.25
    # Prevent 0-year age for brand new funds
    age_df["Fund Age (Yrs)"] = np.where(age_df["Fund Age (Yrs)"] > 0, age_df["Fund Age (Yrs)"], 0.01)

    recent_data = data[data["date"] >= one_year_ago].copy()
    recent_data["return"] = recent_data.groupby("scheme_name")["nav"].pct_change()

    metrics = recent_data.groupby("scheme_name").agg(current_nav=("nav", "last"),start_nav=("nav", "first"),daily_mean=("return", "mean"),daily_std=("return", "std"),).reset_index()

    # 2. Expected Return: Geometric (CAGR) instead of Arithmetic
    metrics["Expected Annual Return (%)"] = (
        (metrics["current_nav"] / metrics["start_nav"]) ** 1 - 1
    ) * 100

    metrics["Annualized Volatility (%)"] = metrics["daily_std"] * np.sqrt(252) * 100

    # 3. Sharpe Ratio with Zero-Division Protection
    risk_free = (1.04 ** (1 / 252)) - 1
    metrics["Sharpe Ratio"] = np.where(
        metrics["daily_std"] > 0,
        (metrics["daily_mean"] - risk_free) / metrics["daily_std"] * np.sqrt(252),
        np.nan
    )

    metrics = metrics.merge(age_df[["Fund Age (Yrs)"]].reset_index(), on="scheme_name")
    return metrics.dropna()