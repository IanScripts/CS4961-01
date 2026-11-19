
from collections import defaultdict
from datetime import timedelta


def aggregate_event_counts(events, regions, eligible_cells):
    # Count unique fire discoveries by grid cell and date
    fire_counts = defaultdict(int)
    seen_ids = set()

    for fire in events:
        fod_id = fire["fod_id"]

        if fod_id in seen_ids:
            raise ValueError(f"Duplicate fire ID: {fod_id}")

        seen_ids.add(fod_id)

        lat = fire["latitude"]
        lon = fire["longitude"]

        if lat is None or lon is None:
            continue

        from src.ai_prediction.label_regions import assign_occurrence

        assignment = assign_occurrence(lat, lon, regions)

        if assignment is None:
            continue

        cell = (assignment["ix"], assignment["iy"])

        if assignment["county_fips"] == "06037":
            split = "la_evaluation"
        elif cell in eligible_cells:
            split = "non_la"
        else:
            continue

        key = (
            split,
            assignment["grid_cell"],
            fire["discovery_date"]
        )

        fire_counts[key] += 1

    return fire_counts


def iter_label_rows(cells_by_split, start_date, end_date, fire_counts):
    # Generate every cell-day, including days with zero fires
    current_date = start_date

    while current_date <= end_date:
        for split, cells in cells_by_split.items():
            for ix, iy in sorted(cells):
                grid_cell = f"{ix}_{iy}"

                count = fire_counts.get(
                    (split, grid_cell, current_date),
                    0
                )

                yield {
                    "split": split,
                    "grid_cell": grid_cell,
                    "date": current_date.isoformat(),
                    "fire_count": count,
                    "fire_label": int(count > 0)
                }

        current_date += timedelta(days=1)
