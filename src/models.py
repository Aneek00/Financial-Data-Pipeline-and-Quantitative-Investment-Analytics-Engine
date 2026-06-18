# src/models.py
import numpy as np
from sklearn.mixture import GaussianMixture
from src.features import calculate_asymmetric_features
from juliacall import Main as jl

jl.seval('include("src/compute.jl")')

def get_max_drawdown(strat_ret: np.ndarray) -> float:
    cum_returns = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(cum_returns)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (cum_returns - peak) / peak
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

def run_strategy_backtest(nav_array: np.ndarray, fast_span=20, slow_span=50, momentum_mult=0.98) -> dict:
    """Advanced backtest running asymmetric GMM feature extraction and probabilistic allocation."""
    if len(nav_array) < max(60, slow_span):
        return None

    # A. Extract Asymmetric Directional Volatility
    log_rets, upside_vols, downside_vols = calculate_asymmetric_features(nav_array, window=20)
    feature_matrix = np.column_stack((log_rets, upside_vols, downside_vols))

    # B. Fit Unsupervised GMM Model
    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
    gmm.fit(feature_matrix)

    # C. Extract soft probabilities instead of rigid labels
    probs = gmm.predict_proba(feature_matrix)

    # D. Mathematically isolate the true "Downside Risk" cluster index
    # The cluster with the higher average downside volatility is flagged as the crash regime
    mean_downside_0 = np.mean(downside_vols[gmm.predict(feature_matrix) == 0])
    mean_downside_1 = np.mean(downside_vols[gmm.predict(feature_matrix) == 1])
    risk_cluster_idx = 1 if mean_downside_1 > mean_downside_0 else 0

    # Extract the exact continuous probability curve of being in that crash state
    crash_probabilities = probs[:, risk_cluster_idx].astype(np.float64)

    # E. Construct standard momentum tracking array
    momentum_array = np.empty_like(nav_array)
    momentum_array[:5] = nav_array[:5]
    momentum_array[5:] = nav_array[:-5]

    slippage_cost = 0.001
    expense_drag = 0.0075 / 252

    # F. Pass continuous crash probabilities down to Julia for fractional allocation
    ema_fast, ema_slow, allocation_curves, daily_ret, strat_ret = jl.FastCompute.execute_backtest_loop(
        nav_array, momentum_array, crash_probabilities, slippage_cost, expense_drag,
        int(fast_span), int(slow_span), float(momentum_mult)
    )

    strat_ret_np = np.array(strat_ret)
    daily_ret_np = np.array(daily_ret)
    bench_cum = np.cumprod(1 + daily_ret_np) - 1
    strat_cum = np.cumprod(1 + strat_ret_np) - 1

    std_ret = float(np.std(strat_ret_np))
    sharpe = (float(np.mean(strat_ret_np)) / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0

    return {
        "Strategy_Ret": float(strat_cum[-1]),
        "Benchmark_Ret": float(bench_cum[-1]),
        "Max_Drawdown": get_max_drawdown(strat_ret_np),
        "Sharpe_Ratio": sharpe
    }

def run_grid_search(nav_train_array: np.ndarray) -> tuple:
    """Sweeps expanded parameter combinations to find the highest Sharpe Ratio."""
    print("[*] Launching expanded high-performance Julia matrix sweep...")

    # Unleashing the grid boundaries
    fast_options = [10, 12, 15, 20]
    slow_options = [26, 40, 50, 60]
    momentum_multipliers = [0.95, 0.97, 0.98, 0.99, 1.00]

    best_sharpe = -float('inf')
    best_params = (20, 50, 0.98)


    for fast in fast_options:
        for slow in slow_options:
            if fast >= slow:
                continue
            for mult in momentum_multipliers:
                res = run_strategy_backtest(nav_train_array, fast_span=fast, slow_span=slow, momentum_mult=mult)
                if res and res["Sharpe_Ratio"] > best_sharpe:
                    best_sharpe = res["Sharpe_Ratio"]
                    best_params = (fast, slow, mult)
    return best_params