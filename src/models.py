# src/models.py
import numpy as np
from juliacall import Main as jl

# Initialize and compile the Julia module once when the script loads
jl.seval('include("src/compute.jl")')

def get_max_drawdown(strat_ret: np.ndarray) -> float:
    """Calculates the maximum drawdown using pure NumPy array operations."""
    cum_returns = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(cum_returns)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (cum_returns - peak) / peak
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

def run_strategy_backtest(nav_array: np.ndarray) -> dict:
    """
    Trend-following strategy execution core.
    Accepts and returns ONLY stateless primitive data containers.
    """
    # Guard check for short timeseries arrays using hardcoded execution bounds
    if len(nav_array) < 60:
        return None

    # Generate the momentum array safely inside Python via pure NumPy shifting
    momentum_array = np.empty_like(nav_array)
    momentum_array[:5] = nav_array[:5]
    momentum_array[5:] = nav_array[:-5]

    slippage_cost = 0.001       # 0.1% cost per trade
    expense_drag = 0.0075 / 252 # 0.75% annual expense ratio

    # Hand raw data vectors over to the compiled Julia module
    ema_fast, ema_slow, signals, daily_ret, strat_ret = jl.FastCompute.execute_backtest_loop(
        nav_array, momentum_array, slippage_cost, expense_drag
    )

    # Cast returned Julia objects back to native NumPy arrays for processing
    strat_ret_np = np.array(strat_ret)
    daily_ret_np = np.array(daily_ret)

    # Compute final cumulative curves natively using vector mathematics
    bench_cum = np.cumprod(1 + daily_ret_np) - 1
    strat_cum = np.cumprod(1 + strat_ret_np) - 1

    # Track structural trend deviation proxy metric over raw arrays
    trend_deviation = float(np.mean(np.abs((nav_array - np.array(ema_slow)) / nav_array)) * 100)

    std_ret = float(np.std(strat_ret_np))
    sharpe = (float(np.mean(strat_ret_np)) / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0

    return {
        "MAPE": trend_deviation,
        "Strategy_Ret": float(strat_cum[-1]),
        "Benchmark_Ret": float(bench_cum[-1]),
        "Max_Drawdown": get_max_drawdown(strat_ret_np),
        "Sharpe_Ratio": sharpe,
        "Arrays": {
            "ema_fast": np.array(ema_fast),
            "ema_slow": np.array(ema_slow),
            "signal": np.array(signals),
            "daily_ret": daily_ret_np,
            "strat_ret": strat_ret_np
        }
    }