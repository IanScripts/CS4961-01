import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from retry_iem_2025 import fetch_with_backoff, retry_after_seconds, remaining_stations


def http_error(code, retry_after=None):
    response = requests.Response()
    response.status_code = code
    if retry_after is not None:
        response.headers['Retry-After'] = retry_after
    return requests.HTTPError(f'HTTP {code}', response=response)


class TestIEMRetry(unittest.TestCase):
    def test_retry_after_seconds(self):
        self.assertEqual(retry_after_seconds('75'), 75)
        self.assertEqual(retry_after_seconds('not valid'), None)
        self.assertEqual(retry_after_seconds('-5'), 0)

    def test_retry_after_http_date(self):
        now = datetime(2026, 11, 19, tzinfo=timezone.utc)
        self.assertEqual(retry_after_seconds('Thu, 19 Nov 2026 00:01:15 GMT', now=now), 75)

    def test_429_recovers_and_honors_retry_after(self):
        fetch = Mock(side_effect=[http_error(429, '65'), pd.DataFrame({'test':[1]})])
        pause = Mock()
        data = fetch_with_backoff(fetch, station='AAT', year=2025,
                                  max_attempts=3, base_backoff=30, sleeper=pause)
        self.assertEqual(len(data), 1)
        pause.assert_called_once_with(65)
        self.assertEqual(fetch.call_count, 2)

    def test_persistent_429_stops_safely(self):
        fetch = Mock(side_effect=[http_error(429), http_error(429)])
        pause = Mock()
        with self.assertRaises(SystemExit):
            fetch_with_backoff(fetch, station='BYS', year=2025,
                               max_attempts=2, base_backoff=30, sleeper=pause)
        pause.assert_called_once_with(30)

    def test_404_not_retried(self):
        fetch = Mock(side_effect=http_error(404))
        pause = Mock()
        with self.assertRaises(requests.HTTPError):
            fetch_with_backoff(fetch, station='AAT', year=2025, sleeper=pause)
        pause.assert_not_called()
        self.assertEqual(fetch.call_count, 1)

    def test_connection_error_recovers(self):
        fetch = Mock(side_effect=[requests.ConnectionError('offline'), 'ok'])
        pause = Mock()
        self.assertEqual(fetch_with_backoff(fetch, station='AAT', year=2025,
                                            sleeper=pause), 'ok')
        pause.assert_called_once_with(30)

    def test_manifest_skips_only_complete_year(self):
        catalog = pd.DataFrame({'station_id':['AAT','BYS','BUR']})
        manifest = pd.DataFrame({'station_id':['AAT','BYS','BUR'],
                                 'year':[2025,2025,2026],
                                 'requested_start':['2025-01-01','2025-03-01','2026-01-01'],
                                 'requested_end':['2025-12-31','2025-12-31','2026-12-31']})
        pending = remaining_stations(catalog,manifest,2025)
        self.assertEqual(pending.station_id.tolist(), ['BYS','BUR'])

    def test_no_manifest_all_pending(self):
        catalog = pd.DataFrame({'station_id':['AAT','BUR']})
        self.assertEqual(len(remaining_stations(catalog,pd.DataFrame(),2025)),2)


if __name__ == '__main__':
    unittest.main()
