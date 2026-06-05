import pandas as pd
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO

logger = logging.getLogger(__name__)

class MFDataExtractor:
    """
    Handles all data extraction from external APIs (AMFI, mfapi.in).
    """
    def __init__(self):
        self.session = self._create_resilient_session()

    def _create_resilient_session(self):
        """Creates a requests session with automatic retry logic for network reliability."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,                # Total number of retries
            backoff_factor=1,       # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on server errors
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def get_all_nav_data(self) -> pd.DataFrame:
        """
        Fetches the latest daily NAV data for all funds from AMFI.
        Used by the daily update and local pipeline scripts.
        """
        logger.info("Fetching latest daily data from AMFI...")
        amfi_url = "https://www.amfiindia.com/spages/NAVAll.txt"

        try:
            response = self.session.get(amfi_url, timeout=30)
            response.raise_for_status()

            # Parse the text file
            all_lines = response.text.strip().splitlines()
            header_index = next((i for i, line in enumerate(all_lines) if "Scheme Name" in line), -1)

            if header_index == -1:
                logger.error("Could not find the header row in the AMFI data.")
                return pd.DataFrame()

            data_lines = [line for line in all_lines[header_index + 1:] if line.count(';') > 4]
            if not data_lines:
                logger.warning("No valid data lines found after the header.")
                return pd.DataFrame()

            data_string = "\n".join(data_lines)
            df = pd.read_csv(
                StringIO(data_string), sep=';', header=None, usecols=[0, 3, 4, 5],
                names=['scheme_code', 'scheme_name', 'nav', 'date']
            )

            # Clean and format the data
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
            df.dropna(subset=['nav', 'date', 'scheme_code'], inplace=True)
            df = df[df['nav'] >= 0].copy() # Filter out invalid negative NAVs
            df['scheme_code'] = df['scheme_code'].astype(str).str.strip()

            logger.info(f"Successfully parsed {len(df)} records from AMFI.")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from AMFI after retries: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An error occurred during data parsing: {e}")
            return pd.DataFrame()