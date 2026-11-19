import unittest
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fire_data.fire_sources import valid_bounds, fetch_firms, fetch_calfire_history

class FireSourcesTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(valid_bounds([-125, 32, -115, 42]), (-125, 32, -115, 42))
        for invalid in ([0,0,10,10], [-121,35,-122,34], [1,2,3]):
            with self.assertRaises(ValueError): valid_bounds(invalid)

    def test_firms_requires_key(self):
        with patch.dict('os.environ', {'NASA_FIRMS_MAP_KEY':''}):
            result=fetch_firms()
        self.assertEqual(result['status'],'needs_key')

    @patch('src.fire_data.fire_sources.get_text')
    def test_firms_csv_parser(self, mock):
        mock.return_value='latitude,longitude,acq_date,frp,confidence\n34.2,-118.1,2026-11-13,1.2,n\n'
        result=fetch_firms(api_key='test')
        self.assertEqual(result['count'],1)
        self.assertEqual(result['features'][0]['geometry']['coordinates'],[-118.1,34.2])

    @patch('src.fire_data.fire_sources.get_text')
    def test_calfire_geojson(self,mock):
        mock.return_value='{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-120,36],[-119,36],[-119,37],[-120,36]]]},"properties":{"FIRE_NAME":"Test","YEAR_":2024}}]}'
        result=fetch_calfire_history()
        self.assertEqual(result['count'],1)
        self.assertIn('historical',result['source'])

if __name__=='__main__': unittest.main()
