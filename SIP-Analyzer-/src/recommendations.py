import pandas as pd

# Extracted constants for thematic/specialized funds
SPECIALIZED_KEYWORDS = [
    'DEFENCE', 'PHARMA', 'HEALTHCARE', 'TECHNOLOGY', 'INFRASTRUCTURE',
    'BANKING', 'FINANCIAL SERVICES', 'PSU', 'COMMODITIES', 'CONSUMPTION',
    'ENERGY', 'AUTO', 'CHILDREN', 'BENEFIT', 'RETIREMENT', 'SAVER'
]
SPECIALIZED_PATTERN = '|'.join(SPECIALIZED_KEYWORDS)

def calculate_suitability_score(row: pd.Series, risk_profile: str, investment_horizon: int) -> float:
    """
    Scores a fund strictly based on the user's risk tolerance and timeline.
    """
    exp_return_norm = max(0, min(1, row['Expected Annual Return (%)'] / 50))
    sharpe_norm = max(0, min(1, row['Sharpe Ratio'] / 3))
    volatility_norm = max(0, min(1, 1 - (row['Annualized Volatility (%)'] / 50)))

    if risk_profile == 'Low':
        score = (sharpe_norm * 0.6) + (volatility_norm * 0.3) + (exp_return_norm * 0.1)
    elif risk_profile == 'Medium':
        score = (exp_return_norm * 0.4) + (sharpe_norm * 0.4) + (volatility_norm * 0.2)
    elif risk_profile == 'High':
        score = (exp_return_norm * 0.7) + (sharpe_norm * 0.2) + (volatility_norm * 0.1)
    else:
        score = (exp_return_norm * 0.4) + (sharpe_norm * 0.4) + (volatility_norm * 0.2)

    # Bonus for age maturity
    if row['Fund Age (Yrs)'] > investment_horizon:
        score *= 1.1

    return score

def categorize_funds(df: pd.DataFrame) -> tuple:
    """
    Splits the ranked funds into Diversified (Core) and Thematic (Specialized).
    """
    is_specialized = df['scheme_name'].str.contains(SPECIALIZED_PATTERN, case=False, na=False)
    core = df[~is_specialized].sort_values('Suitability Score', ascending=False)
    specialized = df[is_specialized].sort_values('Suitability Score', ascending=False)
    return core, specialized

def build_diversified_portfolio(core_candidates: pd.DataFrame,
                                correlation_matrix: pd.DataFrame,
                                portfolio_size: int = 3,
                                threshold: float = 0.85) -> pd.DataFrame:
    """
    Iteratively constructs a portfolio ensuring no two funds are highly correlated.
    """
    valid_candidates = core_candidates[core_candidates['scheme_name'].isin(correlation_matrix.columns)]

    portfolio = []
    if not valid_candidates.empty:
        portfolio.append(valid_candidates.iloc[0]) # Start with the absolute best fund

    for _, candidate in valid_candidates.iloc[1:].iterrows():
        if len(portfolio) >= portfolio_size:
            break

        is_different_enough = True
        # Check candidate against all currently selected funds
        for selected in portfolio:
            corr_val = correlation_matrix.loc[selected['scheme_name'], candidate['scheme_name']]
            if corr_val > threshold:
                is_different_enough = False
                break

        if is_different_enough:
            portfolio.append(candidate)

    return pd.DataFrame(portfolio)