import pandas as pd
import logging
from io import StringIO
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from db_loader import DBLoader

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_resilient_session():
    """
    Creates a requests session with automatic retry logic for network reliability.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,                # Total number of retries
        backoff_factor=1,       # Wait 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504], # Retry on these server errors
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

def fetch_latest_daily_data() -> pd.DataFrame:
    """
    Fetches the latest daily NAV data for all funds from AMFI using a resilient session.
    """
    logging.info("Fetching latest daily data from AMFI...")
    amfi_url = "https://www.amfiindia.com/spages/NAVAll.txt"
    session = create_resilient_session()

    try:
        response = session.get(amfi_url, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # The rest of the parsing logic remains the same
        all_lines = response.text.strip().splitlines()
        header_index = next((i for i, line in enumerate(all_lines) if "Scheme Name" in line), -1)
        if header_index == -1:
            logging.error("Could not find the header row in the AMFI data.")
            return pd.DataFrame()

        data_lines = [line for line in all_lines[header_index + 1:] if line.count(';') > 4]
        if not data_lines:
            logging.warning("No valid data lines found after the header.")
            return pd.DataFrame()

        data_string = "\n".join(data_lines)
        df = pd.read_csv(
            StringIO(data_string), sep=';', header=None, usecols=[0, 3, 4, 5],
            names=['scheme_code', 'scheme_name', 'nav', 'date']
        )

        # Clean the data
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        df.dropna(subset=['nav', 'date', 'scheme_code'], inplace=True)
        df = df[df['nav'] >= 0].copy() # Filter out invalid negative NAVs
        df['scheme_code'] = df['scheme_code'].astype(str).str.strip()

        logging.info(f"Successfully parsed {len(df)} records from AMFI.")
        return df

    except requests.RequestException as e:
        logging.error(f"Failed to fetch data from AMFI after retries: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An error occurred during data parsing: {e}")
        return pd.DataFrame()

def main():
    """Main function to run the daily update."""
    daily_df = fetch_latest_daily_data()
    if daily_df.empty:
        logging.warning("No daily data fetched. Exiting.")
        return

    logging.info(f"Fetched {len(daily_df)} new records. Loading to database...")
    loader = DBLoader()
    # Use 'append' mode - the loader will handle duplicates
    success = loader.load_to_db(daily_df, 'nav_data', if_exists='append')

    if success:
        logging.info("✅ Daily update completed successfully.")
    else:
        logging.error("❌ Daily update failed.")

if __name__ == "__main__":
    main()