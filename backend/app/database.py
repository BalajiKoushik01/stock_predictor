import os
import duckdb
import pandas as pd
from typing import Optional

DATABASE_PATH = os.getenv("DUCKDB_DATABASE_PATH", "data/apex.db")

class DatabaseManager:
    """
    Manages connections and transactions with the local DuckDB database.
    """
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.initialize_tables()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Returns a connection to the DuckDB database.
        """
        return duckdb.connect(self.db_path)

    def initialize_tables(self):
        """
        Creates target tables in DuckDB if they do not exist.
        """
        conn = self.get_connection()
        try:
            # 1. Primary Equity OHLCV Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    timestamp TIMESTAMP,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    log_returns DOUBLE,
                    PRIMARY KEY (timestamp, ticker)
                )
            """)

            # 2. Options Microstructure Table (PCR & OI)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_microstructure (
                    timestamp TIMESTAMP,
                    ticker VARCHAR,
                    pcr_oi DOUBLE,
                    pcr_volume DOUBLE,
                    total_oi DOUBLE,
                    ce_oi DOUBLE,
                    pe_oi DOUBLE,
                    PRIMARY KEY (timestamp, ticker)
                )
            """)

            # 3. Financial Sentiment Scores Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    timestamp TIMESTAMP,
                    ticker VARCHAR,
                    sentiment_score DOUBLE,
                    article_count INTEGER,
                    PRIMARY KEY (timestamp, ticker)
                )
            """)

            # If the table exists but is missing the new columns, recreate it
            try:
                cols = conn.execute("PRAGMA table_info(processed_features)").df()
                if not cols.empty and len(cols) < 15:
                    print("Updating processed_features table schema to include institutional indicators...")
                    conn.execute("DROP TABLE processed_features")
            except Exception as pe:
                print(f"Error checking processed_features table: {pe}")

            # 4. Final Preprocessed Feature Table for ML ingestion
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_features (
                    timestamp TIMESTAMP,
                    ticker VARCHAR,
                    close_raw DOUBLE,
                    close_ffd DOUBLE,
                    close_emd_smoothed DOUBLE,
                    pcr_oi DOUBLE,
                    sentiment_score DOUBLE,
                    rolling_volatility DOUBLE,
                    rsi DOUBLE,
                    macd DOUBLE,
                    atr DOUBLE,
                    bb_pct DOUBLE,
                    obv DOUBLE,
                    rolling_skew DOUBLE,
                    rolling_kurt DOUBLE,
                    PRIMARY KEY (timestamp, ticker)
                )
            """)

            # 5. Benchmark Data Table (e.g. Nifty 50 / ^NSEI close prices)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_data (
                    timestamp TIMESTAMP,
                    ticker VARCHAR,
                    close DOUBLE,
                    PRIMARY KEY (timestamp, ticker)
                )
            """)

            # 6. Fundamental Metrics Table (Screener.in & yfinance)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_metrics (
                    ticker VARCHAR PRIMARY KEY,
                    market_cap DOUBLE,
                    pe_ratio DOUBLE,
                    roce DOUBLE,
                    roe DOUBLE,
                    debt_to_equity DOUBLE,
                    dividend_yield DOUBLE,
                    book_value DOUBLE,
                    sales_growth DOUBLE,
                    source VARCHAR,
                    updated_at TIMESTAMP
                )
            """)
            print("DuckDB tables initialized successfully.")
        except Exception as e:
            print(f"Error initializing DuckDB tables: {e}")
            raise e
        finally:
            conn.close()

    def save_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = "append"):
        """
        Inserts a Pandas DataFrame into the specified DuckDB table.
        Args:
            table_name: Name of target table.
            df: Pandas DataFrame.
            if_exists: 'append' to insert, or 'replace' to delete existing records first.
        """
        if df.empty:
            return

        conn = self.get_connection()
        try:
            if if_exists == "replace":
                conn.execute(f"DELETE FROM {table_name}")
            
            # DuckDB natively reads local pandas DataFrames directly from python scope
            # by executing SQL on the DataFrame variable name 'df'
            conn.execute(f"INSERT OR REPLACE INTO {table_name} SELECT * FROM df")
            print(f"Successfully saved {len(df)} rows to {table_name}.")
        except Exception as e:
            print(f"Error saving DataFrame to {table_name}: {e}")
            raise e
        finally:
            conn.close()

    def load_dataframe(self, query: str) -> pd.DataFrame:
        """
        Loads query results from DuckDB into a Pandas DataFrame.
        """
        conn = self.get_connection()
        try:
            df = conn.execute(query).df()
            return df
        except Exception as e:
            print(f"Error querying DuckDB: {e}")
            raise e
        finally:
            conn.close()

    def execute(self, query: str):
        """
        Executes a SQL query directly on the DuckDB database connection.
        """
        conn = self.get_connection()
        try:
            conn.execute(query)
        except Exception as e:
            print(f"Database execution error: {e}")
            raise e
        finally:
            conn.close()

# Singleton instance
db_manager = DatabaseManager()
