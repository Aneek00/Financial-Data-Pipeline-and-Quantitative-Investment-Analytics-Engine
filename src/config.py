# src/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

# 1. Centralized Constants
EXCLUDED_FUND_KEYWORDS = [
    "FMP", "FIXED MATURITY", "FIXED TERM", "SERIES", "INTERVAL FUND",
    "CAPITAL PROTECTION", "CLOSED ENDED", "CLOSE ENDED", "CLOSE-ENDED",
    "CAP PROTECTION", "LIMITED OFFER", "NFO", "MATURITY", "TARGET MATURITY",
    "SEGREGATED PORTFOLIO", "LOCK-IN", "LIMITED PERIOD",
    "OVERNIGHT", "LIQUID", "ULTRA SHORT", "ULTRA-SHORT", "MONEY MARKET",
    "GILT", "ARBITRAGE", "SHORT DURATION", "LOW DURATION", "CORPORATE BOND",
    "CREDIT RISK", "DYNAMIC BOND", "BANKING AND PSU", "ETF"
]

# Create a pre-compiled regex pattern for efficient filtering across the app
import re
EXCLUDED_FUNDS_PATTERN = re.compile("|".join(EXCLUDED_FUND_KEYWORDS), re.IGNORECASE)

# 2. Environment Settings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class DatabaseSettings(BaseSettings):
    # --- THE FIX: This is how Pydantic v2 reads .env files ---
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    """Database connection settings."""
    user: str = Field(..., alias='DB_USER')
    password: str = Field(..., alias='DB_PASSWORD')
    host: str = Field('localhost', alias='DB_HOST')
    port: int = Field(3306, alias='DB_PORT')
    database: str = Field(..., alias='DB_NAME')
    pool_size: int = Field(10, alias='DB_POOL_SIZE')
    max_overflow: int = Field(5, alias='DB_MAX_OVERFLOW')

    @property
    def url(self) -> str:
        """Constructs the database connection URL."""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# Initialize the config object
db_config = DatabaseSettings()