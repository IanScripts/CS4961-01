
import math

from collections import Counter
from datetime import date

from src.ai_prediction.read_occurrences import iter_occurrences
from src.ai_prediction.label_regions import (
    load_county_regions,
    enumerate_intersecting_cells,
    exclude_la_buffer,
    assign_occurrence
)


FIRE_DATA = "wildfire-datasets/FPA_FOD_20260615.gpkg"
BOUNDARIES = "wildfire-datasets/boundaries/counties_2024.zip"

START_DATE = date(2018, 7, 1)
END_DATE = date(2018, 7, 3)


def main():
    # Load the nine Southern California counties
    regions = load_county_regions(BOUNDARIES)

    la_geometry = regions.loc[
        regions["GEOID"] == "06037", "geometry"
    ].iloc[0]

    # Identify candidate cells outside LA County
    training_cells = set()

    for _, county in regions.iterrows():
        if county["GEOID"] != "06037":
            training_cells.update(
                enumerate_intersecting_cells(county.geometry)
            )

    # Remove cells within the LA exclusion buffer
    eligible_cells = set(
        exclude_la_buffer(training_cells, la_geometry)
    )

    counts = Counter()
    county_counts = Counter()

    # Read real historical wildfire records
    for fire in iter_occurrences(
        FIRE_DATA, START_DATE, END_DATE
    ):
        counts["total_records"] += 1

        lat = fire["latitude"]
        lon = fire["longitude"]

        if (
            lat is None or lon is None
            or not math.isfinite(lat)
            or not math.isfinite(lon)
        ):
            counts["missing_coordinates"] += 1
            continue

        # Determine the county and geographic grid cell
        assignment = assign_occurrence(lat, lon, regions)

        if assignment is None:
            counts["outside_or_ambiguous"] += 1
            continue

        county_counts[assignment["county_name"]] += 1

        cell = (assignment["ix"], assignment["iy"])

        if assignment["county_fips"] == "06037":
            counts["la_events"] += 1

        elif cell in eligible_cells:
            counts["eligible_events"] += 1

        else:
            counts["buffer_excluded_events"] += 1

    print("\nSouthern California Wildfire Investigation")
    print("-" * 45)
    print("Dates:", START_DATE, "to", END_DATE)
    print("Nationwide records:", counts["total_records"])
    print("Missing coordinates:", counts["missing_coordinates"])
    print("Outside or ambiguous:", counts["outside_or_ambiguous"])
    print("LA evaluation events:", counts["la_events"])
    print("Eligible non-LA events:", counts["eligible_events"])
    print("Buffer-excluded events:", counts["buffer_excluded_events"])

    print("\nReported Events by County")
    print("-" * 45)

    for county, count in sorted(county_counts.items()):
        print(f"{county}: {count}")


if __name__ == "__main__":
    main()
