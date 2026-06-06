import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import gc

# Silence Prophet logs in production
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

def generate_ensemble_forecast(fund_df: pd.DataFrame, periods: int = 365) -> pd.DataFrame:
    """
    Runs Prophet and Holt-Winters concurrently and blends them into an ensemble forecast.
    """
    # 1. Prophet Model
    prophet_df = fund_df[['date', 'nav']].rename(columns={'date': 'ds', 'nav': 'y'})
    model_prophet = Prophet(daily_seasonality=True).fit(prophet_df)
    future_prophet = model_prophet.make_future_dataframe(periods=periods)
    forecast_prophet = model_prophet.predict(future_prophet)

    # 2. Holt-Winters Model
    hw_df = fund_df.set_index('date')
    # Prevent seasonal period from exceeding data length
    seasonal_periods = min(365, max(2, len(hw_df) // 2))
    model_hw = ExponentialSmoothing(
        hw_df['nav'], trend='add', seasonal='add', seasonal_periods=seasonal_periods
    ).fit()
    forecast_hw = model_hw.forecast(periods)

# 3. Ensemble Blending
    prophet_future = forecast_prophet[-periods:]['yhat'].values
    ensemble_values = (prophet_future + forecast_hw.values) / 2
    forecast_dates = pd.date_range(start=fund_df['date'].iloc[-1], periods=periods + 1)[1:]

    final_df = pd.DataFrame({
        'date': forecast_dates,
        'prophet_forecast': prophet_future,
        'hw_forecast': forecast_hw.values,
        'ensemble_forecast': ensemble_values
    })

    # Clean the memory BEFORE you hit the exit door!
    gc.collect()

    return final_df
def get_max_drawdown(returns_series: pd.Series) -> float:
    """Calculates the maximum drawdown of a returns series."""
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def run_strategy_backtest(fund_df: pd.DataFrame) -> dict:
    """
    Trend-following + Momentum strategy with realistic transaction costs
    and institutional error metrics.
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
        raise ValueError("Insufficient data to train the backtest model.")

    # 1. Prophet Fallback Protection & tuned seasonality
    model_df = train_df[['date', 'nav']].rename(columns={'date': 'ds', 'nav': 'y'})
    try:
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        m.fit(model_df)
        forecast = m.predict(m.make_future_dataframe(periods=len(test_df)))
    except Exception as e:
        logging.error(f"Prophet model failed to fit: {e}")
        return None

    test_df = test_df.merge(forecast[['ds', 'yhat']], left_on='date', right_on='ds')

    test_df['signal'] = 0
    test_df.loc[(test_df['ema_fast'] > test_df['ema_slow']) &
                (test_df['yhat'] > test_df['nav'] * 0.98), 'signal'] = 1

    # 2. Transaction Costs & Realistic Returns
    test_df['trade'] = test_df['signal'].diff().abs().fillna(0)
    slippage_cost = 0.001  # 0.1% cost per trade
    expense_drag = 0.0075 / 252  # 0.75% annual expense ratio

    test_df['daily_ret'] = test_df['nav'].pct_change().fillna(0)
    test_df['bench_cum'] = (1 + test_df['daily_ret']).cumprod() - 1

    test_df['strat_ret'] = (test_df['daily_ret'] * test_df['signal'].shift(1).fillna(0)) - (test_df['trade'] * slippage_cost) - expense_drag
    test_df['strat_cum'] = (1 + test_df['strat_ret']).cumprod() - 1

    # 3. Institutional Metrics (RMSE and MAPE)
    rmse = np.sqrt(((test_df['nav'] - test_df['yhat']) ** 2).mean())
    mape = np.mean(np.abs((test_df['nav'] - test_df['yhat']) / test_df['nav'])) * 100
    std_ret = test_df['strat_ret'].std()
    sharpe = (test_df['strat_ret'].mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0

    return {
        "RMSE": rmse,
        "MAPE": mape,
        "Strategy_Ret": test_df['strat_cum'].iloc[-1],
        "Benchmark_Ret": test_df['bench_cum'].iloc[-1],
        "Max_Drawdown": get_max_drawdown(test_df['strat_ret']),
        "Sharpe_Ratio": sharpe,
        "TestData": test_df
    }