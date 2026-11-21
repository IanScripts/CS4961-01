
import sqlite3
import tempfile
import unittest

from contextlib import closing
from datetime import date
from pathlib import Path

from src.ai_prediction.read_occurrences import iter_occurrences


class TestReadOccurrences(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name) / "test.gpkg"

        with closing(sqlite3.connect(self.data_path)) as connection:
            connection.execute("""
                CREATE TABLE Fires (
                    FOD_ID INTEGER,
                    DISCOVERY_DATE TEXT,
                    LATITUDE REAL,
                    LONGITUDE REAL,
                    STATE TEXT,
                    COUNTY TEXT
                )
            """)

            connection.executemany(
                "INSERT INTO Fires VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "2018-07-01T00:00:00+00:00", 34.0, -118.0, "CA", "Los Angeles"),
                    (2, "2018-07-03T00:00:00+00:00", 34.1, -118.1, "CA", "Los Angeles"),
                    (3, "2018-07-04T00:00:00+00:00", 34.2, -118.2, "CA", "Los Angeles")
                ]
            )

            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_date_range(self):
        events = list(iter_occurrences(
            self.data_path,
            date(2018, 7, 1),
            date(2018, 7, 3)
        ))

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["fod_id"], 1)
        self.assertEqual(
            events[1]["discovery_date"],
            date(2018, 7, 3)
        )

    def test_duplicate_ids(self):
        with closing(sqlite3.connect(self.data_path)) as connection:
            connection.execute(
                "INSERT INTO Fires VALUES (?, ?, ?, ?, ?, ?)",
                (1, "2018-07-02", 34.0, -118.0, "CA", "Los Angeles")
            )
            connection.commit()

        with self.assertRaises(ValueError):
            list(iter_occurrences(
                self.data_path,
                date(2018, 7, 1),
                date(2018, 7, 3)
            ))


if __name__ == "__main__":
    unittest.main()
