# scripts/run_backtest.py
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from src.config import db_config
from src.models import run_strategy_backtest, run_grid_search

def main():
    print("--- Running Walk-Forward Optimization Pipeline ---")

    engine = create_engine(db_config.url)
    query = "SELECT date, nav FROM nav_data WHERE scheme_code = '100033' ORDER BY date ASC"

    with engine.connect() as conn:
        raw_df = pd.read_sql(query, conn)

    if raw_df.empty or len(raw_df) < 200:
        print("[X] Insufficient historical data found.")
        return

    # Stripping data layer down to pure NumPy array
    full_nav_array = raw_df['nav'].to_numpy(dtype=np.float64)
    total_len = len(full_nav_array)

    # Chronological Boundary Split: 80% In-Sample (Train), 20% Out-of-Sample (Test)
    split_idx = int(total_len * 0.80)
    train_array = full_nav_array[:split_idx]
    test_array = full_nav_array[split_idx:]

    print(f"[✓] Data partitioned: {len(train_array)} days In-Sample, {len(test_array)} days Out-of-Sample.")

    # 1. Run optimization engine strictly on training history
    best_fast, best_slow, best_mult = run_grid_search(train_array)

    # 2. Run clean out-of-sample backtest with the optimal parameters found
    print("[*] Evaluating selected parameters on unseen Out-of-Sample data...")
    oos_results = run_strategy_backtest(test_array, fast_span=best_fast, slow_span=best_slow, momentum_mult=best_mult)

    if oos_results is None:
        print("[X] Walk-Forward validation failed.")
        return

    print("\n" + "="*45)
    print("    OUT-OF-SAMPLE VALIDATION REPORT (STEP 3)")
    print("="*45)
    print(f"Test Horizon Length:      {len(test_array)} trading days")
    print(f"Parameters Employed:      Fast={best_fast} | Slow={best_slow} | Mult={best_mult}")
    print("-" * 45)
    print(f"OOS Strategy Return:      {oos_results['Strategy_Ret']*100:.1f}%")
    print(f"OOS Benchmark Return:     {oos_results['Benchmark_Ret']*100:.1f}%")
    print(f"OOS Max Strategy DD:      {oos_results['Max_Drawdown']*100:.1f}%")
    print(f"OOS Sharpe Ratio:         {oos_results['Sharpe_Ratio']:.2f}")
    print("="*45)

if __name__ == "__main__":
    main()