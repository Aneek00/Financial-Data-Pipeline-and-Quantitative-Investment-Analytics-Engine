# src/compute.jl
module FastCompute

function calculate_ema(prices::AbstractVector{Float64}, span::Int)::Vector{Float64}
    n = length(prices)
    ema = zeros(Float64, n)
    if n == 0 return ema end

    ema[1] = prices[1]
    multiplier = 2.0 / (span + 1)

    @inbounds for i in 2:n
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    end
    return ema
end

"""
Executes backtest logic with a Structural Trend Floor to prevent bull-market cash drag.
"""
function execute_backtest_loop(
    nav::AbstractVector{Float64},
    momentum_5d::AbstractVector{Float64},
    crash_prob::AbstractVector{Float64},
    slippage_cost::Float64,
    expense_drag::Float64,
    fast_span::Int,
    slow_span::Int,
    momentum_mult::Float64
)
    n = length(nav)

    ema_fast = calculate_ema(nav, fast_span)
    ema_slow = calculate_ema(nav, slow_span)

    signals = zeros(Float64, n)
    trades = zeros(Float64, n)
    daily_ret = zeros(Float64, n)
    strat_ret = zeros(Float64, n)

    @inbounds for i in 2:n
        daily_ret[i] = (nav[i] - nav[i-1]) / nav[i-1]

        # Determine if base trend rules are met
        if (ema_fast[i] > ema_slow[i]) && (nav[i] > momentum_5d[i] * momentum_mult)
            base_trend_signal = 1.0
        else
            base_trend_signal = 0.0
        end

        # IMPLEMENTING THE AGGRESSIVE STRUCTURAL TREND FLOOR
        if ema_fast[i] > ema_slow[i]
            # Structural Uptrend: Cap the GMM cash penalty so exposure never drops below 75%
            signals[i] = base_trend_signal * max(0.75, 1.0 - crash_prob[i])
        else
            # Structural Downtrend/Neutral: Allow GMM to scale exposure all the way to 0%
            signals[i] = base_trend_signal * (1.0 - crash_prob[i])
        end

        trades[i] = abs(signals[i] - signals[i-1])
        strat_ret[i] = (daily_ret[i] * signals[i-1]) - (trades[i] * slippage_cost) - expense_drag
    end

    return ema_fast, ema_slow, signals, daily_ret, strat_ret
end

end # module