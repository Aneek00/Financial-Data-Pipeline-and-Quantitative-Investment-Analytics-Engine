import pandas as pd
import numpy as np
from src.models import run_strategy_backtest
from juliacall import Main as jl

# 1. Load your Julia file into the Python environment
jl.seval('include("src/compute.jl")')

def run_julia_backtest(df):
    # 2. Extract the NAV column as a raw NumPy array (Float64)
    # Julia reads NumPy arrays natively with zero-copy overhead!
    nav_array = df['nav'].to_numpy(dtype=np.float64)

    # 3. Call the Julia function
    # Note: We access it via jl.FastCompute (the module name we created in Julia)
    signals = jl.FastCompute.calculate_ema_signals(nav_array, 20, 50)

    # 4. Slap the fast results back into your Pandas DataFrame
    df['Strategy_Signal'] = signals

    return df

def main():
    print("--- Running Standalone Production Backtest ---")

    # 1. Generate realistic "Bull Market" dummy data (just like your original code)
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', periods=1200)
    # 0.08% daily drift = strong bull market
    nav_values = 100 * (1 + np.random.normal(0.0008, 0.01, 1200)).cumprod()

    dummy_fund_df = pd.DataFrame({
        'date': dates,
        'nav': nav_values,
        'scheme_name': 'HDFC Flexi Cap'
    })

    # 2. Run the engine (imported from our clean src/ directory)
    results = run_strategy_backtest(dummy_fund_df)

    # 3. Print the Report
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

if __name__ == "__main__":
    main()
