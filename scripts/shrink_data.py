# scripts/shrink_data.py
import pandas as pd
from src.config import EXCLUDED_FUNDS_PATTERN

def main():
    print("Loading heavy raw data...")
    # Assuming this comes from your pipeline's output
    df = pd.read_parquet("final_nav_data.parquet")

    # 1. Filter out the bad funds using our CENTRALIZED config regex!
    unique_funds = pd.Series(df["scheme_name"].unique())
    valid_fund_names = unique_funds[
        ~unique_funds.str.contains(EXCLUDED_FUNDS_PATTERN, regex=True, na=False)
    ]

    # 2. Filter for mature funds only
    fund_counts = df["scheme_name"].value_counts()
    mature_funds = fund_counts[fund_counts > 400].index

    # 3. Apply filters and compress
    final_valid_funds = set(valid_fund_names).intersection(set(mature_funds))
    df_clean = df[df["scheme_name"].isin(final_valid_funds)].copy()

    # Downcast to float32 to save memory in Streamlit
    df_clean["nav"] = df_clean["nav"].astype("float32")

    print(f"Original Rows: {len(df)} | Clean Rows: {len(df_clean)}")

    print("Saving compressed parquet file for Streamlit...")
    df_clean.to_parquet("clean_nav_data.parquet", compression="brotli")
    print("Finished. Ready for app.py!")

if __name__ == "__main__":
    main()