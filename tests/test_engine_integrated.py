import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.wildfire_engine.experimental_score import WeatherDay, REGIONS, demo_weather, score, describe_region

class EngineTests(unittest.TestCase):
    def test_bounds(self):
        low=score(WeatherDay('test',0,100,0,100))
        high=score(WeatherDay('test',50,0,100,0))
        self.assertEqual(low['score'], 0)
        self.assertEqual(high['score'], 100)
    def test_higher_wind_increases_score(self):
        a=score(WeatherDay('test',30,22,10,0))['score']
        b=score(WeatherDay('test',30,22,60,0))['score']
        self.assertGreater(b,a)
    def test_rain_reduces_score(self):
        a=score(WeatherDay('test',30,20,30,0))['score']
        b=score(WeatherDay('test',30,20,30,12))['score']
        self.assertGreater(a,b)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            score(WeatherDay('test',30,101,25,0))
    def test_demo_has_seven_days(self):
        for key in REGIONS:
            region=describe_region(key,demo_weather(key),'demo')
            self.assertEqual(len(region['forecast']),7)
            self.assertTrue(all(0<=f['score']<=100 for f in region['forecast']))
    def test_deterministic(self):
        w=WeatherDay('2026-01-01',35,15,65,0)
        self.assertEqual(score(w),score(w))
if __name__=='__main__': unittest.main()
