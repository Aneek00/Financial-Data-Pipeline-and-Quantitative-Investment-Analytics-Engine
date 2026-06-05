import pandas as pd
from sqlalchemy import create_engine, text
from config import db_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def view_data():
    """Connects to the database and displays the most recent data."""
    logging.info(f"Connecting to database '{db_config.database}'...")

    try:
        engine = create_engine(db_config.url)
        with engine.connect() as conn:
            logging.info("✅ Connection successful!")

            query = text("SELECT * FROM nav_data ORDER BY date DESC LIMIT 10")
            df = pd.read_sql(query, conn)

            if df.empty:
                logging.warning("Database connected, but the 'nav_data' table is empty.")
            else:
                print("\n--- 10 Most Recent Records in Your Database ---")
                print(df.to_string())

    except Exception as e:
        logging.error(f"❌ Failed to connect or query the database: {e}")

if __name__ == "__main__":
    view_data()
