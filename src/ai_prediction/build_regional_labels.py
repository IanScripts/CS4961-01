
import json
from pathlib import Path
import hashlib

from src.ai_prediction.label_regions import (
    load_county_regions,
    enumerate_intersecting_cells,
    exclude_la_buffer
)

import csv
from datetime import date

from src.ai_prediction.read_occurrences import iter_occurrences
from src.ai_prediction.regional_labels import (
    aggregate_event_counts,
    iter_label_rows
)



CONFIG_PATH = Path("config/ai/regions-v1.json")


def load_config(config_path):
    # Read our regional dataset settings
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if config["geography"]["grid_crs"] != "EPSG:5070":
        raise ValueError("Unsupported grid CRS")

    if config["geography"]["grid_size_meters"] != 10000:
        raise ValueError("Grid size does not match grid.py")

    if config["pilot"]["target_cells"] < 1:
        raise ValueError("Pilot must contain at least one cell")

    return config


def build_cell_population(config):
    # Load the county boundaries
    regions = load_county_regions(
        config["sources"]["counties"]
    )

    la_fips = config["geography"]["evaluation_county"]
    training_fips = set(
        config["geography"]["training_counties"]
    )

    la_geometry = regions.loc[
        regions["GEOID"] == la_fips, "geometry"
    ].iloc[0]

    la_cells = set(enumerate_intersecting_cells(
        la_geometry
    ))

    training_cells = set()

    for _, county in regions.iterrows():
        if county["GEOID"] in training_fips:
            training_cells.update(
                enumerate_intersecting_cells(county.geometry)
            )

    eligible_cells = set(exclude_la_buffer(
        training_cells,
        la_geometry,
        config["geography"]["buffer_meters"]
    ))

    # Ensure no training cell overlaps LA
    if eligible_cells & la_cells:
        raise ValueError("Training cells overlap LA")

    return regions, eligible_cells, la_cells


def select_pilot_cells(eligible_cells, target_count):
    # Stable, evenly spaced selection from sorted cell IDs
    candidates = sorted(eligible_cells)

    if len(candidates) < target_count:
        raise ValueError("Not enough eligible cells")

    if target_count == 1:
        return [candidates[len(candidates) // 2]]

    indices = [
        i * (len(candidates) - 1) // (target_count - 1)
        for i in range(target_count)
    ]

    return [candidates[i] for i in indices]


def generate_pilot_dataset(config, regions, eligible_cells, pilot_cells, output_dir=None):
    # Create the output directory
    output_dir = Path(
        output_dir if output_dir is not None
        else config["outputs"]["directory"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # Export selected geographic grid cells
    # ---------------------------------------
    grid_path = output_dir / config["outputs"]["grid_cells"]

    with grid_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["grid_cell", "ix", "iy", "split"]
        )
        writer.writeheader()

        for ix, iy in pilot_cells:
            writer.writerow({
                "grid_cell": f"{ix}_{iy}",
                "ix": ix,
                "iy": iy,
                "split": "non_la"
            })

    # ---------------------------------------
    # Generate historical wildfire labels
    # ---------------------------------------
    label_path = output_dir / config["outputs"]["fire_labels"]

    total_rows = 0
    positive_days = 0

    with label_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "window",
                "split",
                "grid_cell",
                "date",
                "fire_count",
                "fire_label"
            ]
        )
        writer.writeheader()

        for window in config["pilot"]["windows"]:
            start = date.fromisoformat(window["start_date"])
            end = date.fromisoformat(window["end_date"])

            if start > end:
                raise ValueError("Invalid date window")

            print(f"\nProcessing {window['name']}: {start} to {end}")

            # Read real USDA wildfire reports
            events = iter_occurrences(
                config["sources"]["wildfires"],
                start,
                end
            )

            # Assign wildfire events to eligible cells
            fire_counts = aggregate_event_counts(
                events,
                regions,
                eligible_cells
            )

            # Generate daily labels for our 12 pilot cells
            rows = iter_label_rows(
                {"non_la": pilot_cells},
                start,
                end,
                fire_counts
            )

            window_rows = 0
            window_positive = 0

            for row in rows:
                writer.writerow({
                    "window": window["name"],
                    **row
                })

                total_rows += 1
                window_rows += 1

                if row["fire_label"] == 1:
                    positive_days += 1
                    window_positive += 1

            print("Cell-day rows:", window_rows)
            print("Positive cell-days:", window_positive)

    print("\nDataset Generation Complete")
    print("-" * 45)
    print("Grid cells:", len(pilot_cells))
    print("Total cell-days:", total_rows)
    print("Positive cell-days:", positive_days)
    print("Negative cell-days:", total_rows - positive_days)
    print("Grid CSV:", grid_path)
    print("Labels CSV:", label_path)


    write_manifest(
        config,
        pilot_cells,
        total_rows,
        positive_days,
        output_dir=output_dir
    )



def file_sha256(path):
    # Calculate a checksum without loading the entire file into memory
    checksum = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)

    return checksum.hexdigest()


def write_manifest(config, pilot_cells, total_rows, positive_days, output_dir=None):
    output_dir = Path(
            output_dir if output_dir is not None
            else config["outputs"]["directory"]
        )
    grid_path = output_dir / config["outputs"]["grid_cells"]
    label_path = output_dir / config["outputs"]["fire_labels"]
    manifest_path = output_dir / config["outputs"]["manifest"]

    manifest = {
        "dataset_name": config["dataset_name"],
        "config_version": config["config_version"],
        "grid": config["geography"],
        "pilot_windows": config["pilot"]["windows"],
        "pilot_cells": [
            f"{ix}_{iy}" for ix, iy in pilot_cells
        ],
        "sources": {
            "wildfires": {
                "path": config["sources"]["wildfires"],
                "sha256": file_sha256(config["sources"]["wildfires"])
            },
            "counties": {
                "path": config["sources"]["counties"],
                "sha256": file_sha256(config["sources"]["counties"])
            }
        },
        "outputs": {
            "grid_cells": {
                "path": str(grid_path),
                "sha256": file_sha256(grid_path)
            },
            "fire_labels": {
                "path": str(label_path),
                "sha256": file_sha256(label_path),
                "rows": total_rows,
                "positive_cell_days": positive_days,
                "negative_cell_days": total_rows - positive_days
            }
        }
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    print("Manifest:", manifest_path)


def main():
    config = load_config(CONFIG_PATH)

    regions, eligible_cells, la_cells = (
        build_cell_population(config)
    )

    pilot_cells = select_pilot_cells(
        eligible_cells,
        config["pilot"]["target_cells"]
    )

    print("\nRegional Wildfire Dataset Builder")
    print("-" * 45)
    print("Dataset:", config["dataset_name"])
    print("Eligible non-LA cells:", len(eligible_cells))
    print("LA evaluation cells:", len(la_cells))
    print("Selected pilot cells:", len(pilot_cells))

    print("\nPilot Grid Cells")
    for ix, iy in pilot_cells:
        print(f"{ix}_{iy}")

    
    generate_pilot_dataset(
        config,
        regions,
        eligible_cells,
        pilot_cells
    )

    



if __name__ == "__main__":
    main()
