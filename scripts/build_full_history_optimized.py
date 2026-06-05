import requests
import pandas as pd
import logging
from tqdm import tqdm
from io import StringIO
import time
import os
import pickle
from db_loader import DBLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_WORKERS = 20
CHECKPOINT_INTERVAL_SECONDS = 300
PROGRESS_FILE = "_progress.pkl"
ALL_FUNDS_FILE = "_all_funds.pkl"
BACKUP_FILE = "final_nav_data.parquet"

# (All the fetch functions remain the same as before)
def fetch_all_scheme_codes() -> pd.DataFrame:
    if os.path.exists(ALL_FUNDS_FILE):
        logging.info("Loading master fund list from local cache...")
        return pd.read_pickle(ALL_FUNDS_FILE)
    logging.info("Fetching master list of all fund schemes from API...")
    try:
        response = requests.get("https://api.mfapi.in/mf", timeout=30)
        response.raise_for_status()
        funds_df = pd.DataFrame(response.json())
        funds_df.to_pickle(ALL_FUNDS_FILE)
        logging.info(f"Found and cached {len(funds_df)} total schemes.")
        return funds_df
    except Exception as e:
        logging.error(f"Could not fetch the master fund list: {e}")
        return pd.DataFrame()

def fetch_one_fund_history(session: requests.Session, fund_info: dict) -> pd.DataFrame:
    scheme_code = fund_info['schemeCode']
    scheme_name = fund_info['schemeName']
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = session.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            records = data.get("data")
            if data.get("status") == "SUCCESS" and records:
                df = pd.DataFrame(records)
                df['scheme_code'] = scheme_code
                df['scheme_name'] = scheme_name
                return df[['date', 'scheme_code', 'scheme_name', 'nav']]
    except requests.RequestException:
        pass
    return pd.DataFrame()


def main():
    """Main pipeline with all optimizations."""
    # Check if we have a backup file to use instead of re-downloading
    if os.path.exists(BACKUP_FILE):
        logging.warning(f"--- Found existing backup file '{BACKUP_FILE}'. Loading from it instead of re-downloading. ---")
        final_df = pd.read_parquet(BACKUP_FILE)
    else:
        # (The entire download logic is here, but will be skipped if backup exists)
        all_funds = fetch_all_scheme_codes()
        if all_funds.empty: return
        all_data_frames = []
        processed_codes = set()

        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'rb') as f: all_data_frames = pickle.load(f)
            for df in all_data_frames: processed_codes.update(df['scheme_code'].unique())
            logging.info(f"Resuming with {len(processed_codes)} funds already downloaded.")

        remaining_funds = all_funds[~all_funds['schemeCode'].isin(processed_codes)]

        if not remaining_funds.empty:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
            session.mount('https://', adapter)
            fund_list = remaining_funds.to_dict('records')
            last_checkpoint_time = time.time()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_fund = {executor.submit(fetch_one_fund_history, session, fund): fund for fund in fund_list}
                for future in tqdm(as_completed(future_to_fund), total=len(fund_list), desc="Fetching Full History"):
                    result_df = future.result()
                    if not result_df.empty: all_data_frames.append(result_df)
                    if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL_SECONDS:
                        with open(PROGRESS_FILE, 'wb') as f: pickle.dump(all_data_frames, f)
                        tqdm.write("Checkpoint saved.")
                        last_checkpoint_time = time.time()

        with open(PROGRESS_FILE, 'wb') as f: pickle.dump(all_data_frames, f)
        if not all_data_frames:
            logging.error("CRITICAL: No data could be downloaded. Halting.")
            return

        logging.info("Combining all data sources and cleaning...")
        final_df = pd.concat(all_data_frames, ignore_index=True)
        final_df['nav'] = pd.to_numeric(final_df['nav'], errors='coerce')
        final_df['date'] = pd.to_datetime(final_df['date'], dayfirst=True, errors='coerce')
        final_df.dropna(subset=['nav', 'date', 'scheme_code', 'scheme_name'], inplace=True)
        final_df['scheme_code'] = final_df['scheme_code'].astype(str).str.strip()
        final_df.sort_values('date', ascending=True, inplace=True)
        final_df.drop_duplicates(subset=['date', 'scheme_code'], keep='last', inplace=True)

        logging.info(f"--- Backup --- Saving final combined data to '{BACKUP_FILE}'...")
        final_df.to_parquet(BACKUP_FILE, index=False)

    # --- FINAL PROCESSING ---
    logging.info("Final data processing and validation...")

    # --- ADD THIS LINE TO FIX THE ERROR ---
    final_df = final_df[final_df['nav'] >= 0].copy()

    logging.info(f"Preparing to load {len(final_df)} valid records into the database.")
    loader = DBLoader()
    success = loader.load_to_db(final_df, 'nav_data', if_exists='replace')

    if success:
        logging.info("✅ Full historical database has been built successfully!")
        if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
        if os.path.exists(ALL_FUNDS_FILE): os.remove(ALL_FUNDS_FILE)
    else:
        logging.error("❌ Database loading failed.")

if __name__ == "__main__":
    main()