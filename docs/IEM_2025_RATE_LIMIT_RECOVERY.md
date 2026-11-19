# EmberScope — Resume statewide 2025 IEM collection after HTTP 429

`python scripts/retry_iem_2025.py` resumes the **original** IEM downloader. It does not replace the established download/parser/manifest code. The existing archive output and manifest stay in `live/california_sources/iem/` and do not belong in Git.

## Why this runner exists

During the first full-year California collection, 126 of 161 cataloged stations returned observations, 25 requests received HTTP 429, and 10 returned empty results. The original script pauses only 0.35 seconds between requests. IEM's legacy ASOS CGI downloads are resource-intensive; this wrapper uses 5 seconds between station requests and exponential backoff on 429, 5xx, and connection/timeouts, respecting a server `Retry-After` header. Persistent 429 stops the run rather than hammering the remaining stations.

## Use

Back up the observation CSV and manifest before restarting. From the repository root:

```
python -m unittest discover -s tests -p "test_iem_retry.py" -v
python scripts/retry_iem_2025.py
```

Default options: `--year 2025 --pause-seconds 5 --max-attempts 4 --base-backoff 30`.

The runner loads the saved IEM catalog and manifest, selects only stations lacking a complete manifest entry for 2025, then calls the original `download_observations`. The underlying data collector still handles duplicate `station_id`/`valid` rows on normal completion. If halted on persistent 429, later runs can resume by manifest; already completed station-years are not requested again.

An empty result is **not** considered successfully downloaded and remains retryable. Some stations may genuinely lack data. The existence of a 2025 manifest entry indicates a completed request, **not** continuous hourly station coverage; assess quality separately. The station catalog is not a guarantee of active reporting.

## Future pipeline

After the 2025 retries, run a per-station coverage audit, then normalize only the 2025 records to a statewide hourly CSV. Keep 2025/2026 dates and historical/modelled data sources distinct. Do not commit downloaded CSVs. For V4 Open-Meteo comparisons, request modeled data at each station's actual catalog coordinates and align by station plus UTC timestamp; do not assume a nearby city represents the station.
