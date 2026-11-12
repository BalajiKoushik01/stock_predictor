"""Regression coverage for non-destructive DuckDB feature migrations."""
import os
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from app.database import DatabaseManager


class DatabaseContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "test.duckdb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_feature_table_is_migrated_without_dropping_rows(self):
        with duckdb.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE processed_features ("
                "timestamp TIMESTAMP, ticker VARCHAR, close_raw DOUBLE)"
            )
            conn.execute(
                "INSERT INTO processed_features VALUES "
                "('2026-01-01', 'TEST', 120.0)"
            )
        mgr = DatabaseManager(self.path)
        with mgr.get_connection() as conn:
            rows = conn.execute(
                "SELECT ticker, close_raw FROM processed_features"
            ).fetchall()
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(processed_features)"
                ).fetchall()
            }
        self.assertEqual(rows, [("TEST", 120.0)])
        self.assertTrue({"atr", "bb_pct", "rolling_skew", "macd"} <= cols)

    def test_parameterized_delete_does_not_affect_other_tickers(self):
        mgr = DatabaseManager(self.path)
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "ticker": ["SAFE", "SECOND"],
                "close_raw": [100.0, 150.0],
            }
        )
        mgr.save_dataframe("processed_features", data)
        mgr.execute(
            "DELETE FROM processed_features WHERE ticker = ?",
            ["SAFE' OR 1=1 --"],
        )
        with mgr.get_connection() as conn:
            count = conn.execute(
                "SELECT count(*) FROM processed_features"
            ).fetchone()[0]
        self.assertEqual(count, 2)

    def test_rejects_arbitrary_table_identifiers(self):
        mgr = DatabaseManager(self.path)
        with self.assertRaises(ValueError):
            mgr.save_dataframe(
                "processed_features;DROP TABLE processed_features",
                pd.DataFrame({"ticker": ["A"]}),
            )


if __name__ == "__main__":
    unittest.main()
