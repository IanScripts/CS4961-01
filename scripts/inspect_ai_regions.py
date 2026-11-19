
from collections import defaultdict

from src.ai_prediction.label_regions import (
    load_county_regions,
    enumerate_intersecting_cells,
    exclude_la_buffer
)

BOUNDARIES = "wildfire-datasets/boundaries/counties_2024.zip"

regions = load_county_regions(BOUNDARIES)

all_cells = set()
la_cells = set()
training_cells = set()

print("\nSouthern California 10 km Grid Coverage")
print("-" * 45)

for _, county in regions.iterrows():
    cells = set(enumerate_intersecting_cells(county.geometry))

    print(f"{county['NAME']}: {len(cells)} cells")

    all_cells.update(cells)

    if county["GEOID"] == "06037":
        la_cells.update(cells)
    else:
        training_cells.update(cells)

overlapping = la_cells & training_cells

print("\nRegional Summary")
print("-" * 45)
print("Total unique cells:", len(all_cells))
print("Los Angeles cells:", len(la_cells))
print("Non-LA cells:", len(training_cells))
print("Cells shared with LA:", len(overlapping))
print("Example shared cells:", sorted(overlapping)[:10])


# Get the actual Los Angeles County polygon
la_geometry = regions.loc[
    regions["GEOID"] == "06037", "geometry"
].iloc[0]

# Remove non-LA cells too close to Los Angeles
eligible_cells = exclude_la_buffer(
    training_cells,
    la_geometry,
    buffer_meters=10000
)

excluded_cells = training_cells - set(eligible_cells)

print("\nTraining Region After LA Buffer")
print("-" * 45)
print("Original non-LA cells:", len(training_cells))
print("Cells excluded by LA buffer:", len(excluded_cells))
print("Eligible training cells:", len(eligible_cells))
print("LA evaluation cells:", len(la_cells))

# Confirm no eligible training cell intersects LA
assert not (set(eligible_cells) & la_cells)
print("Geographic separation check: PASSED")

