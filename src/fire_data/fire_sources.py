"""Independent fire observation adapters. Do not confuse detections with incidents."""
import csv
import io
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FIRMS_BASE = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv'
CALFIRE_LAYER = os.environ.get('CALFIRE_LAYER_URL', 'https://services5.arcgis.com/xSQIR8wivxIXbeLK/arcgis/rest/services/California_Fire_Perimeters__1950_2024/FeatureServer/0')
CAL_BOUNDS = (-125, 32, -113.5, 42.1)


def get_text(url, timeout=16):
    request = Request(url, headers={'User-Agent': 'EmberScope/0.2 (educational research; source attributed)', 'Accept': 'application/json,text/csv,*/*'})
    with urlopen(request, timeout=timeout) as response:
        return response.read(8_000_000).decode('utf-8-sig')


def valid_bounds(bounds):
    if len(bounds) != 4: raise ValueError('Expected west,south,east,north')
    w,s,e,n = map(float,bounds)
    if not (-126 <= w < e <= -112 and 31 <= s < n <= 43):
        raise ValueError('Bounding box must be in or near California')
    return (w,s,e,n)


def fetch_firms(bounds=CAL_BOUNDS, days=1, api_key=None):
    api_key = (api_key if api_key is not None else os.getenv('NASA_FIRMS_MAP_KEY','')).strip()
    if not api_key: return {'status':'needs_key','message':'Set NASA_FIRMS_MAP_KEY to enable satellite detections.','features':[],'source':'NASA FIRMS · VIIRS NOAA-21 NRT'}
    if not (1 <= int(days) <= 5): raise ValueError('days must be 1–5')
    bounds = valid_bounds(bounds)
    bbox = ','.join(f'{x:.4f}' for x in bounds)
    url = f'{FIRMS_BASE}/{api_key}/VIIRS_NOAA21_NRT/{bbox}/{int(days)}'
    raw = get_text(url)
    if raw.lstrip().startswith('<') or raw.lstrip().startswith('{'):
        raise RuntimeError('FIRMS returned an unexpected response; check MAP_KEY and quota')
    records = list(csv.DictReader(io.StringIO(raw)))
    features=[]
    for i,r in enumerate(records[:1500]):
        try:
            lat=float(r['latitude']); lon=float(r['longitude'])
            if not (bounds[0]<=lon<=bounds[2] and bounds[1]<=lat<=bounds[3]): continue
            properties={k:str(r.get(k,'') or '')[:120] for k in ('acq_date','acq_time','confidence','frp','satellite','daynight','bright_ti4')}
            features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[lon,lat]},'properties':properties,'id':i})
        except (ValueError,KeyError): continue
    return {'status':'ok','features':features,'source':'NASA FIRMS · VIIRS NOAA-21 NRT','count':len(features),'note':'Satellite thermal anomalies are not confirmed wildfire incidents.'}


def fetch_calfire_history(bounds=CAL_BOUNDS):
    """CAL FIRE FRAP historical perimeter data, NOT active wildfire perimeter data."""
    w,s,e,n = valid_bounds(bounds)
    params = {
      'where':'1=1','geometry':f'{w},{s},{e},{n}','geometryType':'esriGeometryEnvelope',
      'inSR':'4326','spatialRel':'esriSpatialRelIntersects','outFields':'*',
      'outSR':'4326','returnGeometry':'true','resultRecordCount':'150','f':'geojson'
    }
    raw=get_text(CALFIRE_LAYER.rstrip('/')+'/query?'+urlencode(params))
    obj=json.loads(raw)
    if obj.get('error'): raise RuntimeError('CAL FIRE GIS service: '+str(obj['error'].get('message','error')))
    if obj.get('type') != 'FeatureCollection':
        raise RuntimeError('CAL FIRE GIS service did not return GeoJSON; configure CALFIRE_LAYER_URL for a GeoJSON-compatible layer')
    features=[]
    for i,f in enumerate(obj.get('features',[])[:150]):
        if f.get('geometry',{}).get('type') not in ('Polygon','MultiPolygon'): continue
        props = f.get('properties') or {}
        fields = ('FIRE_NAME','YEAR_','YEAR','GIS_ACRES','ACRES','FIRE_NUM','ALARM_DATE','FIRE_ID')
        clean = {str(k)[:80]:str(v)[:120] for k,v in props.items() if str(k).upper() in fields and v is not None}
        features.append({'type':'Feature','id':i,'geometry':f['geometry'],'properties':clean})
    return {'status':'ok','features':features,'source':'CAL FIRE FRAP · historical perimeters (1950–2024 service)','count':len(features),'note':'Historical shapes only; not active fires. May be truncated to 150 features in the displayed area.'}
