import sys
import os

# --- THE FIX: Force Python to recognize the root folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from src.config import db_config
import logging
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.errors import SchemaError

logger = logging.getLogger(__name__)

class DBLoader:
    def __init__(self):
        self.engine = self._create_engine()

    def _create_engine(self):
        return create_engine(
            db_config.url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600
        )

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validates the DataFrame against a predefined schema."""
        schema = DataFrameSchema({
            "date": Column(pd.Timestamp),
            # FIX: Allow NAV to be zero or greater
            "nav": Column(float, checks=[Check.greater_than_or_equal_to(0)]),
            "scheme_code": Column(str),
            "scheme_name": Column(str)
        })
        try:
            schema.validate(df, lazy=True)
            return True
        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            return False

    def load_to_db(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        if not self.validate_data(df):
            return False

        try:
            with self.engine.begin() as conn:
                # Only attempt to delete overlapping dates if we are appending
                if if_exists == 'append' and not df.empty:
                    unique_dates = df['date'].dt.strftime('%Y-%m-%d').unique().tolist()

                    if unique_dates:
                        # Institutional standard: Parameterized SQL injection protection
                        binds = {f"d_{i}": d for i, d in enumerate(unique_dates)}
                        in_clause = ", ".join([f":{k}" for k in binds.keys()])

                        delete_sql = text(f"DELETE FROM {table_name} WHERE date IN ({in_clause})")
                        result = conn.execute(delete_sql, binds)
                        logger.info(f"Removed {result.rowcount} existing records for dates being loaded.")

                # Proceed with inserting the new data
                df.to_sql(
                    name=table_name, con=conn, if_exists=if_exists,
                    index=False, method='multi', chunksize=1000
                )
                logger.info(f"Successfully loaded {len(df)} records to table '{table_name}' with mode '{if_exists}'.")
                return True

        except SQLAlchemyError as e:
            logger.error(f"DB Error during load to '{table_name}': {e}", exc_info=True)
            return False