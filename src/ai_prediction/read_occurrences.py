
import sqlite3

from contextlib import closing
from datetime import date, timedelta
from pathlib import Path


def iter_occurrences(data_path, start_date, end_date):

    # Check input dates
    if start_date > end_date:
        raise ValueError("Start date must be before end date")

    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    # Open the USDA database in read-only mode
    database_uri = data_path.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(database_uri, uri=True)) as connection:

        cursor = connection.execute(
            """
            SELECT FOD_ID, DISCOVERY_DATE,
                   LATITUDE, LONGITUDE,
                   STATE, COUNTY
            FROM Fires
            WHERE DISCOVERY_DATE >= ?
              AND DISCOVERY_DATE < ?
            ORDER BY DISCOVERY_DATE, FOD_ID
            """,
            (
                start_date.isoformat(),
                (end_date + timedelta(days=1)).isoformat()
            )
        )

        seen_ids = set()

        for fod_id, discovery, lat, lon, state, county in cursor:

            # Don't silently count duplicate events
            if fod_id is None or fod_id in seen_ids:
                raise ValueError(f"Missing or duplicate FOD_ID: {fod_id}")

            seen_ids.add(fod_id)

            # Preserve the published discovery calendar date
            fire_date = date.fromisoformat(discovery[:10])

            yield {
                "fod_id": fod_id,
                "discovery_date": fire_date,
                "latitude": lat,
                "longitude": lon,
                "state": state,
                "county": county
            }
