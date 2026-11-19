"""Offline tests for optional IEM observed-station views in EmberScope V3."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.dashboard.iem_views import read_iem_hourly, summary_html, station_charts


def sample() -> pd.DataFrame:
    data=[]
    for station, name in [('AAT','Alturas'),('BUR','Burbank')]:
        for h in range(3):
            missing=(station=='AAT' and h==1)
            data.append({
                'station_id':station,'station_name':name,
                'time_utc':f'2026-10-01T0{h}:00:00+00:00',
                'source':'IEM ASOS archive','data_type':'observed',
                'report_count':0 if missing else 1,'missing_hour':missing,
                'temperature_f':None if missing else 65+h,
                'relative_humidity_pct':None if missing else 50,
                'wind_speed_mph':None if missing else 4.5,
                'precipitation_report_1h_in':None if missing else 0.0,
            })
    return pd.DataFrame(data)


class TestIEMDashboard(unittest.TestCase):
    def load(self, data=None):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'iem.csv'
            (sample() if data is None else data).to_csv(p,index=False)
            return read_iem_hourly(p)

    def test_missing_optional_file_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,note=read_iem_hourly(Path(tmp)/'does-not-exist.csv')
        self.assertTrue(d.empty)
        self.assertIn('No standardized',note)

    def test_station_hourly_loaded_with_missing_gap(self):
        d,note=self.load()
        self.assertEqual(len(d),6)
        self.assertEqual(int(d.missing_hour.sum()),1)
        self.assertIn('5/6 station-hours',note)
        a=d[d.station_id=='AAT'].reset_index(drop=True)
        self.assertTrue(pd.isna(a.loc[1,'temperature_f']))
        self.assertEqual(a.loc[0,'data_type'],'observed')
        self.assertEqual(str(a.loc[0,'time_california'].tz), 'America/Los_Angeles')

    def test_summary_counts_are_field_specific(self):
        d,_=self.load()
        html=summary_html(d)
        self.assertIn('2/3',html)
        self.assertIn('3/3',html)
        self.assertIn('Alturas',html)
        self.assertIn('Burbank',html)

    def test_plots_show_only_first_station_and_no_filled_gaps(self):
        d,_=self.load()
        a,b,names=station_charts(d)
        self.assertEqual(names,['AAT','BUR'])
        self.assertEqual(len(a.data),4)
        self.assertEqual(len(b.data),4)
        self.assertEqual([bool(t.visible) for t in a.data],[True,True,False,False])
        self.assertFalse(a.data[0].connectgaps)
        self.assertTrue(pd.isna(list(a.data[0].y)[1]))

    def test_duplicate_station_hour_rejected(self):
        d=sample()
        with self.assertRaisesRegex(ValueError,'duplicate station-hours'):
            self.load(pd.concat([d,d.iloc[[0]]],ignore_index=True))

    def test_source_mismatch_rejected(self):
        d=sample(); d.loc[0,'data_type']='forecast'
        with self.assertRaisesRegex(ValueError,'not labeled observed'):
            self.load(d)

    def test_missing_flag_and_report_count_mismatch_rejected(self):
        d=sample(); d.loc[0,'missing_hour']=True
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.load(d)

    def test_station_name_escaped_in_summary(self):
        d=sample(); d.loc[d.station_id=='AAT','station_name']='<script>alert(1)</script>'
        out,_=self.load(d)
        html=summary_html(out)
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)


if __name__=='__main__':
    unittest.main()
