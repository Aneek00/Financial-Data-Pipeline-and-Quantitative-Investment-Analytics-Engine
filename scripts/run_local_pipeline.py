"""
This script runs the local data ingestion pipeline to populate the database.
"""
import sys
import os

# --- THE FIX: Force Python to recognize the root folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from src.data_extractor import MFDataExtractor
from src.db_loader import DBLoader

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """Runs the data extraction and loading pipeline."""
    logging.info("--- Starting Local Data Pipeline Run ---")
    try:
        # 1. Initialize components
        extractor = MFDataExtractor()
        loader = DBLoader()

        # 2. Extraction from the official AMFI source
        logging.info("Extracting data from AMFI source...")
        nav_df = extractor.get_all_nav_data()

        if nav_df.empty:
            raise ValueError("No data was extracted from AMFI. Halting pipeline.")

        # 3. Loading the data into the database using 'append'
        logging.info(f"Loading {len(nav_df)} records to the database...")
        # This will now correctly append data and build a history
        success = loader.load_to_db(nav_df, 'nav_data', if_exists='append')

        if not success:
            raise ValueError("Data loading failed during database operation.")

        logging.info("--- Data Pipeline Completed Successfully ---")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()
