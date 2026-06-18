# scripts/run_backtest.py
import sys
import os
import pandas as pd
import numpy as np

# Force Python to recognize the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from src.config import db_config
from src.models import run_strategy_backtest

def main():
    print("--- Running Production DB-Linked DOP Backtest ---")

    # 1. Connect to your actual database using your config properties
    engine = create_engine(db_config.url)

    # 2. Extract real historical data for an active scheme code
    # We choose a single scheme code, sorting strictly by date chronologically
    query = "SELECT date, nav FROM nav_data WHERE scheme_code = '100033' ORDER BY date ASC"

    try:
        with engine.connect() as conn:
            raw_df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"[X] Database read failed: {e}")
        print("[*] Falling back to a generalized table scan to grab the first valid scheme...")
        fallback_query = "SELECT date, nav FROM nav_data WHERE scheme_code = (SELECT scheme_code FROM nav_data LIMIT 1) ORDER BY date ASC"
        with engine.connect() as conn:
            raw_df = pd.read_sql(fallback_query, conn)

    if raw_df.empty or len(raw_df) < 60:
        print(f"[X] Error: Extracted series contains insufficient data history ({len(raw_df)} rows found).")
        return

    print(f"[✓] Extracted {len(raw_df)} historical records from the database.")

    # 3. STRIP PANDAS AWAY IMMEDIATELY AT THE BOUNDARY
    # Extract only the primitive float array for math processing
    nav_production_array = raw_df['nav'].to_numpy(dtype=np.float64)

    # 4. Execute the stateless engine
    results = run_strategy_backtest(nav_production_array)

    if results is None:
        print("[X] Execution failed inside backend layers.")
        return

    # 5. Print the verified report
    print("\n" + "="*45)
    print("      QUANTITATIVE PERFORMANCE REPORT (REAL DB DATA)")
    print("="*45)
    print(f"Total History Checked:    {len(nav_production_array)} trading days")
    print(f"Trend Deviation (MAPE):   {results['MAPE']:.2f}%")
    print(f"Total Strategy Return:    {results['Strategy_Ret']*100:.1f}%")
    print(f"Benchmark Return:         {results['Benchmark_Ret']*100:.1f}%")
    print("-" * 45)
    print(f"Max Strategy Drawdown:    {results['Max_Drawdown']*100:.1f}%")
    print(f"Annualized Sharpe Ratio:  {results['Sharpe_Ratio']:.2f}")
    print("="*45)

if __name__ == "__main__":
    main()