
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta
import logging
import warnings

# Setup
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

class ProfessionalMFEngine:
    def __init__(self, df):
        self.df = df

    def add_indicators(self, df):
        """Standard production feature engineering"""
        df = df.copy()
        # 1. EMA (Exponential Moving Averages) - faster response to trends
        df['ema_fast'] = df['nav'].ewm(span=20, adjust=False).mean()
        df['ema_slow'] = df['nav'].ewm(span=50, adjust=False).mean()

        # 2. ATR (Average True Range) for volatility-based sizing
        df['pct_change'] = df['nav'].pct_change()
        df['volatility'] = df['pct_change'].rolling(window=20).std()

        return df.bfill()

    def run_backtest(self, scheme_name):
        print(f"--- Running Production Backtest for: {scheme_name} ---")
        fund_df = self.df[self.df['scheme_name'] == scheme_name].copy()
        fund_df = self.add_indicators(fund_df)

        split_date = fund_df['date'].max() - timedelta(days=365)
        train_df = fund_df[fund_df['date'] <= split_date]
        test_df = fund_df[fund_df['date'] > split_date].copy()

        # Forecasting
        model_df = train_df[['date', 'nav']].rename(columns={'date': 'ds', 'nav': 'y'})
        m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(model_df)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        test_df = test_df.merge(forecast[['ds', 'yhat']], left_on='date', right_on='ds')

        # --- REWRITTEN SIGNAL LOGIC: TREND-FOLLOWING + MOMENTUM ---
        # Logic: Buy if (Fast EMA > Slow EMA) AND (Prophet is not predicting a reversal)
        test_df['signal'] = 0
        test_df.loc[(test_df['ema_fast'] > test_df['ema_slow']) &
                    (test_df['yhat'] > test_df['nav'] * 0.98), 'signal'] = 1

        # Returns calculation
        test_df['daily_ret'] = test_df['nav'].pct_change().fillna(0)
        test_df['bench_cum'] = (1 + test_df['daily_ret']).cumprod() - 1
        test_df['strat_ret'] = test_df['daily_ret'] * test_df['signal'].shift(1).fillna(0)
        test_df['strat_cum'] = (1 + test_df['strat_ret']).cumprod() - 1

        # --- RISK METRICS ---
        def get_max_drawdown(returns_series):
            cum_returns = (1 + returns_series).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns - peak) / peak
            return drawdown.min()

        accuracy = (np.sign(test_df['nav'].diff()) == np.sign(test_df['yhat'].diff())).mean()
        sharpe = (test_df['strat_ret'].mean() / test_df['strat_ret'].std()) * np.sqrt(252) if test_df['strat_ret'].std() != 0 else 0

        return {
            "Accuracy": accuracy,
            "Strategy_Ret": test_df['strat_cum'].iloc[-1],
            "Benchmark_Ret": test_df['bench_cum'].iloc[-1],
            "Max_Drawdown": get_max_drawdown(test_df['strat_ret']),
            "Sharpe_Ratio": sharpe
        }

# --- Execution ---
if __name__ == "__main__":
    # Generate realistic "Bull Market" data
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', periods=1200)
    # 0.08% daily drift = strong bull market
    nav_values = 100 * (1 + np.random.normal(0.0008, 0.01, 1200)).cumprod()
    data = pd.DataFrame({'date': dates, 'nav': nav_values, 'scheme_name': 'HDFC Flexi Cap'})

    engine = ProfessionalMFEngine(data)
    results = engine.run_backtest('HDFC Flexi Cap')

    print("\n" + "="*45)
    print("      QUANTITATIVE PERFORMANCE REPORT")
    print("="*45)
    print(f"Directional Accuracy:     {results['Accuracy']*100:.1f}%")
    print(f"Total Strategy Return:    {results['Strategy_Ret']*100:.1f}%")
    print(f"Benchmark Return:         {results['Benchmark_Ret']*100:.1f}%")
    print("-" * 45)
    print(f"Max Strategy Drawdown:    {results['Max_Drawdown']*100:.1f}%")
    print(f"Annualized Sharpe Ratio:  {results['Sharpe_Ratio']:.2f}")
    print("="*45)
