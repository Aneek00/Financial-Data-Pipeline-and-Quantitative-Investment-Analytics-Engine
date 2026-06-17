# src/compute.jl
module FastCompute

"""
Calculates Exponential Moving Average manually at bare-metal compilation speeds.
"""
# Change 1: Relaxed input type to AbstractVector
function calculate_ema(prices::AbstractVector{Float64}, span::Int)::Vector{Float64}
    n = length(prices)
    ema = zeros(Float64, n)
    if n == 0
        return ema
    end

    ema[1] = prices[1]
    multiplier = 2.0 / (span + 1)

    for i in 2:n
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    end
    return ema
end

"""
Executes the entire strategy simulation, signals, and transaction costs in a single ultra-fast loop.
"""
# Change 2: Relaxed input types to AbstractVector here as well
function execute_backtest_loop(nav::AbstractVector{Float64}, momentum_5d::AbstractVector{Float64}, slippage_cost::Float64, expense_drag::Float64)
    n = length(nav)

    # Calculate underlying EMAs via our optimized function
    ema_fast = calculate_ema(nav, 20)
    ema_slow = calculate_ema(nav, 50)

    # Pre-allocate arrays for maximum memory efficiency
    signals = zeros(Int64, n)
    trades = zeros(Float64, n)
    daily_ret = zeros(Float64, n)
    strat_ret = zeros(Float64, n)

    for i in 2:n
        daily_ret[i] = (nav[i] - nav[i-1]) / nav[i-1]

        # Signal evaluation logic
        if (ema_fast[i] > ema_slow[i]) && (nav[i] > momentum_5d[i] * 0.98)
            signals[i] = 1
        else
            signals[i] = 0
        end

        # Calculate trade flips based on current vs previous day signal
        trades[i] = abs(signals[i] - signals[i-1])

        # Return calculation utilizes previous day's signal exposure
        strat_ret[i] = (daily_ret[i] * signals[i-1]) - (trades[i] * slippage_cost) - expense_drag
    end

    return ema_fast, ema_slow, signals, daily_ret, strat_ret
end

end # module