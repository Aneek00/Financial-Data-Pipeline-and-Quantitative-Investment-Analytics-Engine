import pandas as pd
from tqdm import tqdm
import warnings
from src.models import run_strategy_backtest

warnings.filterwarnings("ignore")

def main():
    print("Loading clean data...")
    df = pd.read_parquet("clean_nav_data.parquet")

    print("Filtering out Dividend/IDCW funds to protect calculation integrity...")
    valid_df = df[~df["scheme_name"].str.contains("IDCW|Dividend|Div", case=False, na=False)]

    top_funds = valid_df["scheme_name"].unique().tolist()
    print(f"Total valid funds to process: {len(top_funds)}")

    all_metrics = []
    all_charts = []

    print(f"Running Julia-Accelerated Backtests on Top {len(top_funds)} Funds...")

    for fund in tqdm(top_funds, desc="Backtesting"):
        fund_df = valid_df[valid_df["scheme_name"] == fund].copy()
        results = run_strategy_backtest(fund_df)

        if results is not None:
            all_metrics.append({
                "scheme_name": fund,
                "MAPE": results["MAPE"],
                "Strategy_Ret": results["Strategy_Ret"],
                "Benchmark_Ret": results["Benchmark_Ret"],
                "Max_Drawdown": results["Max_Drawdown"],
                "Sharpe_Ratio": results["Sharpe_Ratio"]
            })

            chart_df = results["TestData"][['date', 'bench_cum', 'strat_cum']].copy()
            chart_df['scheme_name'] = fund
            all_charts.append(chart_df)

    # Save metrics with advanced Brotli compression
    pd.DataFrame(all_metrics).to_parquet("backtest_metrics.parquet", compression="brotli")
    pd.concat(all_charts, ignore_index=True).to_parquet("backtest_charts.parquet", compression="brotli")
    print("✅ Successfully saved Julia-accelerated backtest results!")

if __name__ == "__main__":
    main()