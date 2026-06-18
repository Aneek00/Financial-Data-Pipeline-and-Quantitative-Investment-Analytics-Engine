# scripts/simulate_risk.py
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from src.config import db_config

def run_monte_carlo_var(nav_array: np.ndarray, simulations: int = 10000, days_ahead: int = 252):
    """
    Projects 10,000 stochastic price paths using Geometric Brownian Motion (GBM).
    Returns the 5% Value at Risk (VaR) and Expected Shortfall (CVaR).
    """
    print(f"[*] Initializing stochastic risk engine ({simulations} paths, {days_ahead} days)...")

    # 1. Calculate historical drift (mean return) and volatility (standard deviation)
    # Using the most recent 3 years (approx 750 days) for relevant recent volatility
    recent_nav = nav_array[-750:] if len(nav_array) > 750 else nav_array
    daily_returns = np.log(recent_nav[1:] / recent_nav[:-1])

    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)

    # 2. Vectorized Generation of Random Shocks (Standard Normal Distribution)
    # Shape: [days_ahead, simulations]
    np.random.seed(42) # Seeded for reproducible dashboard reporting
    random_shocks = np.random.normal(loc=0.0, scale=1.0, size=(days_ahead, simulations))

    # 3. Geometric Brownian Motion Math: Drift + Stochastic Shock
    daily_drift = mu - (0.5 * sigma**2)
    stochastic_returns = np.exp(daily_drift + sigma * random_shocks)

    # 4. Project future cumulative paths
    price_paths = np.zeros_like(stochastic_returns)
    price_paths[0] = nav_array[-1] # Start all universes at today's exact price

    for t in range(1, days_ahead):
        price_paths[t] = price_paths[t-1] * stochastic_returns[t]

    # 5. Calculate final portfolio returns across all 10,000 parallel universes
    final_returns = (price_paths[-1] - price_paths[0]) / price_paths[0]

    # 6. Extract Institutional Risk Percentiles
    var_95 = np.percentile(final_returns, 5)  # The threshold where the worst 5% begins
    cvar_95 = np.mean(final_returns[final_returns <= var_95]) # Average loss of that worst 5%
    median_return = np.percentile(final_returns, 50)

    return var_95, cvar_95, median_return

def main():
    print("--- Running Institutional Monte Carlo Risk Simulation ---")

    engine = create_engine(db_config.url)
    # Testing on the same Scheme Code
    query = "SELECT date, nav FROM nav_data WHERE scheme_code = '100033' ORDER BY date ASC"

    try:
        with engine.connect() as conn:
            raw_df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"[X] Database connection failed: {e}")
        return

    if raw_df.empty:
        print("[X] Data missing for simulation.")
        return

    # Pure DOP boundary: Extract raw array
    nav_array = raw_df['nav'].to_numpy(dtype=np.float64)

    # Execute Stochastic Simulation
    var_95, cvar_95, median_return = run_monte_carlo_var(nav_array)

    print("\n" + "="*50)
    print("    FORWARD-LOOKING RISK PROJECTION (252 DAYS)")
    print("="*50)
    print(f"Current Asset Price:       ₹{nav_array[-1]:.2f}")
    print(f"Projected Median Return:   {median_return*100:+.2f}%")
    print("-" * 50)
    print(f"95% Value at Risk (VaR):   {var_95*100:+.2f}%")
    print(f"Expected Shortfall (CVaR): {cvar_95*100:+.2f}%")
    print("="*50)
    print(f"Translation: There is a 95% mathematical probability that this")
    print(f"asset will NOT drop by more than {abs(var_95)*100:.1f}% over the next year.")
    print(f"If a black swan event occurs (the worst 5%), the expected average")
    print(f"loss is {abs(cvar_95)*100:.1f}%.")

if __name__ == "__main__":
    main()