import pandas as pd
import numpy as np
from datetime import timedelta
from juliacall import Main as jl

# Initialize and compile the Julia module once when the script loads
jl.seval('include("src/compute.jl")')

def get_max_drawdown(returns_series: pd.Series) -> float:
    """Calculates the maximum drawdown of a returns series."""
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def run_strategy_backtest(fund_df: pd.DataFrame) -> dict:
    """
    Trend-following + Momentum strategy accelerated via an embedded Julia math engine.
    """
    df = fund_df.copy()

    # Handle split date partitioning
    split_date = df['date'].max() - timedelta(days=365)
    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date].copy()

    if len(train_df) < 60 or len(test_df) < 10:
        return None # Graceful exit for short-timeseries funds

    # Generate the momentum array safely inside Python before handing off
    test_df['momentum_5d'] = test_df['nav'].shift(5).fillna(test_df['nav'])

    # Extract NumPy data vectors for direct zero-copy memory access inside Julia
    nav_array = test_df['nav'].to_numpy(dtype=np.float64)
    mom_array = test_df['momentum_5d'].to_numpy(dtype=np.float64)

    slippage_cost = 0.001       # 0.1% cost per trade
    expense_drag = 0.0075 / 252 # 0.75% annual expense ratio

    # Hand data over to the compiled Julia module
    ema_fast, ema_slow, signals, daily_ret, strat_ret = jl.FastCompute.execute_backtest_loop(
        nav_array, mom_array, slippage_cost, expense_drag
    )

    # Convert Julia arrays back into your existing Pandas DataFrame structure seamlessly
    test_df['ema_fast'] = list(ema_fast)
    test_df['ema_slow'] = list(ema_slow)
    test_df['signal'] = list(signals)
    test_df['daily_ret'] = list(daily_ret)
    test_df['strat_ret'] = list(strat_ret)

    # Compute final cumulative curves natively
    test_df['bench_cum'] = (1 + test_df['daily_ret']).cumprod() - 1
    test_df['strat_cum'] = (1 + test_df['strat_ret']).cumprod() - 1

    # Track structural trend deviation proxy metric
    trend_deviation = np.mean(np.abs((test_df['nav'] - test_df['ema_slow']) / test_df['nav'])) * 100

    std_ret = test_df['strat_ret'].std()
    sharpe = (test_df['strat_ret'].mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0

    return {
        "MAPE": trend_deviation,
        "Strategy_Ret": test_df['strat_cum'].iloc[-1],
        "Benchmark_Ret": test_df['bench_cum'].iloc[-1],
        "Max_Drawdown": get_max_drawdown(test_df['strat_ret']),
        "Sharpe_Ratio": sharpe,
        "TestData": test_df
    }