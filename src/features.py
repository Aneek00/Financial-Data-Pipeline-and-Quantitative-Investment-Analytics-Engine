# src/features.py
import numpy as np

def calculate_asymmetric_features(nav_array: np.ndarray, window: int = 20) -> tuple:
    """
    Computes rolling log returns, upside semi-deviation, and downside semi-deviation.
    Provides asymmetric directional variance to the GMM engine.
    """
    n = len(nav_array)
    log_returns = np.zeros(n, dtype=np.float64)
    upside_vol = np.zeros(n, dtype=np.float64)
    downside_vol = np.zeros(n, dtype=np.float64)

    if n < window:
        return log_returns, upside_vol, downside_vol

    # Log returns: ln(p_t / p_{t-1})
    log_returns[1:] = np.log(nav_array[1:] / nav_array[:-1])

    for i in range(window, n):
        window_returns = log_returns[i - window + 1 : i + 1]

        # Isolate positive and negative deviations
        pos_rets = window_returns[window_returns > 0]
        neg_rets = window_returns[window_returns < 0]

        # Calculate annualized semi-deviations
        upside_vol[i] = np.std(pos_rets) * np.sqrt(252) if len(pos_rets) > 1 else 0.0
        downside_vol[i] = np.std(neg_rets) * np.sqrt(252) if len(neg_rets) > 1 else 0.0

    # Backfill early uninitialized rows
    first_valid_up = upside_vol[window]
    first_valid_down = downside_vol[window]
    upside_vol[:window] = first_valid_up
    downside_vol[:window] = first_valid_down

    return log_returns, upside_vol, downside_vol