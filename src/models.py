import pandas as pd
import numpy as np
from datetime import timedelta

def get_max_drawdown(returns_series: pd.Series) -> float:
    """Calculates the maximum drawdown of a returns series."""
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def run_strategy_backtest(fund_df: pd.DataFrame) -> dict:
    """
    Trend-following + Momentum strategy with realistic transaction costs.
    (Cloud-Safe Version: Replaced Prophet ML with Pandas 5-Day Momentum)
    """
    df = fund_df.copy()
    df['ema_fast'] = df['nav'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['nav'].ewm(span=50, adjust=False).mean()
    df['pct_change'] = df['nav'].pct_change()
    df = df.bfill()

    split_date = df['date'].max() - timedelta(days=365)
    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date].copy()

    if len(train_df) < 60:
        return None # Graceful exit instead of crashing the app

    # 1. Cloud-Safe Momentum Fallback Protection
    # Calculate what the price was 5 days ago
    test_df['momentum_5d'] = test_df['nav'].shift(5).fillna(test_df['nav'])

    test_df['signal'] = 0
    # BUY SIGNAL: Fast EMA > Slow EMA
    # AND Current Price is at least 98% of the price 5 days ago (avoids buying into a sudden free-fall)
    test_df.loc[(test_df['ema_fast'] > test_df['ema_slow']) &
                (test_df['nav'] > test_df['momentum_5d'] * 0.98), 'signal'] = 1

    # 2. Transaction Costs & Realistic Returns
    test_df['trade'] = test_df['signal'].diff().abs().fillna(0)
    slippage_cost = 0.001  # 0.1% cost per trade
    expense_drag = 0.0075 / 252  # 0.75% annual expense ratio

    test_df['daily_ret'] = test_df['nav'].pct_change().fillna(0)
    test_df['bench_cum'] = (1 + test_df['daily_ret']).cumprod() - 1

    test_df['strat_ret'] = (test_df['daily_ret'] * test_df['signal'].shift(1).fillna(0)) - (test_df['trade'] * slippage_cost) - expense_drag
    test_df['strat_cum'] = (1 + test_df['strat_ret']).cumprod() - 1

    # 3. Institutional Metrics
    # Since Prophet is gone, we replace Prediction Error with "Trend Deviation"
    # (How far the price strays from its 50-day EMA) so app.py doesn't break.
    trend_deviation = np.mean(np.abs((test_df['nav'] - test_df['ema_slow']) / test_df['nav'])) * 100

    std_ret = test_df['strat_ret'].std()
    sharpe = (test_df['strat_ret'].mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0

    return {
        "MAPE": trend_deviation, # Sent to app.py as a proxy for error
        "Strategy_Ret": test_df['strat_cum'].iloc[-1],
        "Benchmark_Ret": test_df['bench_cum'].iloc[-1],
        "Max_Drawdown": get_max_drawdown(test_df['strat_ret']),
        "Sharpe_Ratio": sharpe,
        "TestData": test_df
    }