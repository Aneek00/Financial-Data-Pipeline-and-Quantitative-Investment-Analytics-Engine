# scripts/generate_forecasts.py
import pandas as pd
from prophet import Prophet
import warnings
import logging
from tqdm import tqdm  # This gives us a beautiful progress bar!

# Suppress Prophet's spammy logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

def main():
    print("Loading clean data...")
    df = pd.read_parquet("clean_nav_data.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # Grab the Top 50 funds with the most historical data
    print("Filtering out Dividend/IDCW funds to protect Prophet math...")
    valid_df = df[~df["scheme_name"].str.contains("IDCW|Dividend|Div", case=False, na=False)]

    # Take ALL remaining valid public funds (Small Cap, Large Cap, Direct, Growth, etc.)
    top_funds = valid_df["scheme_name"].unique().tolist()
    print(f"Total valid funds to process: {len(top_funds)}")

    all_forecasts = []

    print(f"Running Facebook Prophet Machine Learning on Top {len(top_funds)} Funds...")

    # tqdm creates a loading bar in your terminal so you know it hasn't frozen
    for fund in tqdm(top_funds, desc="Forecasting"):
        fund_df = df[df["scheme_name"] == fund][["date", "nav"]].copy()
        fund_df = fund_df.sort_values("date").drop_duplicates("date", keep="last")

        if len(fund_df) >= 365:
            # 1. Train the Model
            prophet_df = fund_df.rename(columns={"date": "ds", "nav": "y"})
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            m.fit(prophet_df)

            # 2. Predict 1 Year into the future
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            # 3. Keep ONLY the date and the predicted NAV (yhat) to save space
            forecast_minimal = forecast[['ds', 'yhat']].copy()
            forecast_minimal['scheme_name'] = fund

            all_forecasts.append(forecast_minimal)

    # Combine all 50 forecasts into one master table
    final_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Save it highly compressed
    final_forecasts.to_parquet("precomputed_forecasts.parquet", compression="brotli")
    print("✅ Successfully saved precomputed_forecasts.parquet!")

if __name__ == "__main__":
    main()