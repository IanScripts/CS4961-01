import tempfile
import unittest
from pathlib import Path
from src.dashboard.iem_daily_views import statewide_explorer_html, _validated_daily

SAMPLE = '''station_id,local_date,day_timezone,source,data_type,expected_local_hours,hour_slots,hours_with_reports,report_coverage_pct,temperature_hours,temperature_mean_f,temperature_max_f,humidity_hours,humidity_mean_pct,humidity_min_pct,wind_hours,wind_mean_mph,wind_max_mph,precipitation_report_hours
9L2,2025-01-01,America/Los_Angeles,IEM ASOS archive,observed,24,24,24,100.0,24,41.77,62.2,24,40.28,20.03,24,2.45,10.357,24
9L2,2025-01-02,America/Los_Angeles,IEM ASOS archive,observed,24,24,12,50.0,0,,,6,33.8,17.89,12,0.77,5.75,5
ABC,2025-01-01,America/Los_Angeles,IEM ASOS archive,observed,24,24,21,87.5,20,50,66,21,54,25,19,4,10,0
'''

class StatewideExplorerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.daily = root / 'daily.csv'
        self.quality = root / 'quality.csv'
        self.daily.write_text(SAMPLE, encoding='utf-8')
        self.quality.write_text('station_id,quality_tier\n9L2,Strong\nABC,Good\n', encoding='utf-8')

    def test_offline_chart_station_select(self):
        html=statewide_explorer_html(self.daily,self.quality)
        self.assertIn('2025 California Statewide Weather Explorer',html)
        self.assertIn('v4-station',html)
        self.assertIn('"quality":"Strong"',html)
        self.assertIn('"quality":"Good"',html)
        self.assertIn('temperature_mean_f',html)
        self.assertIn('No estimated values, fire-danger indexes, or wildfire probabilities.',html)

    def test_missing_values_preserved(self):
        html=statewide_explorer_html(self.daily,self.quality)
        self.assertIn('"temperature_mean_f":[41.77,null]',html)

    def test_invalid_observation_provenance_rejected(self):
        self.daily.write_text(SAMPLE.replace(',observed,',',forecast,'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'not labeled observed'):
            _validated_daily(self.daily)

    def test_incomplete_hour_grid_rejected(self):
        self.daily.write_text(SAMPLE.replace('24,24,24,100.0','24,23,24,100.0'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'incomplete local-day'):
            _validated_daily(self.daily)

    def test_absent_daily_has_nonfatal_placeholder(self):
        html=statewide_explorer_html(self.daily.parent/'absent.csv',self.quality)
        self.assertIn('Daily IEM summaries not provided',html)

if __name__=='__main__':
    unittest.main()
