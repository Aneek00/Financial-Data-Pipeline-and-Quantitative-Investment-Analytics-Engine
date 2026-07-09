import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def fetch_daily_amfi():
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    lines = response.text.splitlines()
    data_lines = [line for line in lines if line.count(';') >= 4 and "Scheme Name" not in line]

    if not data_lines:
        logging.error("No data extracted.")
        return

    df = pd.read_csv(
        StringIO("\n".join(data_lines)), sep=';', header=None, usecols=[0, 3, 4, 5],
        names=['scheme_code', 'scheme_name', 'nav', 'date']
    )

    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
    df.dropna(subset=['nav', 'date', 'scheme_code'], inplace=True)
    df = df[df['nav'] >= 0].copy()

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_filename = f"amfi_daily_{today_str}.parquet"

    df.to_parquet(output_filename, engine='pyarrow', compression='snappy')
    logging.info(f"Successfully generated {output_filename} with {len(df)} records.")

if __name__ == "__main__":
    fetch_daily_amfi()