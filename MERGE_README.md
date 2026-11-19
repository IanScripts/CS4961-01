# Additions to CS4961-01 (phase 1, no overwrites)

This is an **additive** package to extract into the root of your **existing cloned** `CS4961-01-git` repository. It intentionally does **not** contain your five existing API scripts, redundant EmberScope `weather.py`, standalone `server.py`/`static/`, NFDRS4 upstream code, or any credentials.

Included:

- `src/wildfire_engine/experimental_score.py`: teammate's EmberScope four-feature experimental scoring implementation, copied from their ZIP; retain attribution
- `src/fire_data/fire_sources.py`: teammate's CAL FIRE historical geography / optional NASA FIRMS data-source adapters
- `scripts/score_open_meteo_csv.py`: new adapter that uses your **existing Open-Meteo CSV**, not a duplicate API integration
- `tests/`: teammate's unit tests adapted to the new import paths, plus new adapter tests
- `docs/PHASE1_INTEGRATION.md`: ownership, scope, priority and runbook

Before adding, have the teammate confirm that these are their original files and they permit their inclusion and attribution. The scripts are in an offline-tested integration candidate; live API requests are not verified.

PowerShell:

```powershell
cd "C:\Users\ianbp\Downloads\CS4961-01-git"
# Extract this ZIP into the current directory, preserving folders
python -m unittest discover -s tests -p 'test*.py' -v
# Requires existing dependencies (including pandas):
python src/weather_apis/open_meteo_api.py
python scripts/score_open_meteo_csv.py
```

Review with `git status --short`, then commit to a feature branch after passing tests. The new files don't overwrite previously committed files.
