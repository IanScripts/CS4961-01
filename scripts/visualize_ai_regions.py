
from pathlib import Path

import plotly.graph_objects as go
from shapely.geometry import box

from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.label_regions import (
    load_county_regions,
    enumerate_intersecting_cells,
    exclude_la_buffer
)
from src.ai_prediction.build_regional_labels import (
    load_config,
    select_pilot_cells
)


CONFIG = Path("config/ai/regions-v1.json")
OUTPUT = Path("data/ai/regional_v1/ai_regions_map.html")

COLORS = {
    "training": "#20A66A",
    "excluded": "#F59E0B",
    "evaluation": "#EF4444",
    "pilot": "#2563EB"
}


def cell_polygon(ix, iy):
    return box(
        ix * CELL_SIZE,
        iy * CELL_SIZE,
        (ix + 1) * CELL_SIZE,
        (iy + 1) * CELL_SIZE
    )


def add_grid_layer(fig, cells, name, color):
    # Collect cell boundaries into one Plotly trace
    # instead of creating a separate trace per cell.
    xs = []
    ys = []
    centers_x = []
    centers_y = []
    hover_text = []

    for ix, iy in sorted(cells):
        polygon = cell_polygon(ix, iy)

        x, y = polygon.exterior.xy

        xs.extend(list(x) + [None])
        ys.extend(list(y) + [None])

        centers_x.append((ix + 0.5) * CELL_SIZE)
        centers_y.append((iy + 0.5) * CELL_SIZE)

        hover_text.append(
            f"Grid cell: {ix}_{iy}<br>"
            f"Category: {name}"
        )

    # Visible grid boundaries
    fig.add_trace(go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        line=dict(color=color, width=1.2),
        hoverinfo="skip",
        legendgroup=name
    ))

    # Invisible markers make individual cells hoverable
    fig.add_trace(go.Scattergl(
        x=centers_x,
        y=centers_y,
        mode="markers",
        name=name + " details",
        marker=dict(size=9, color=color, opacity=0.01),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        legendgroup=name
    ))


def add_county_boundaries(fig, regions):
    # County geometry is already projected to EPSG:5070
    # so these coordinates use meters.
    for _, county in regions.iterrows():
        geometry = county.geometry

        polygons = (
            list(geometry.geoms)
            if geometry.geom_type == "MultiPolygon"
            else [geometry]
        )

        for polygon in polygons:
            x, y = polygon.exterior.xy

            fig.add_trace(go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(color="#64748B", width=2),
                name="County boundaries",
                showlegend=False,
                hovertemplate=(
                    county["NAME"] +
                    " County<extra></extra>"
                )
            ))


def main():
    config = load_config(CONFIG)

    regions = load_county_regions(
        config["sources"]["counties"]
    )

    la_fips = config["geography"]["evaluation_county"]

    la_geometry = regions.loc[
        regions["GEOID"] == la_fips, "geometry"
    ].iloc[0]

    la_cells = set(
        enumerate_intersecting_cells(la_geometry)
    )

    training_candidates = set()

    for _, county in regions.iterrows():
        if county["GEOID"] != la_fips:
            training_candidates.update(
                enumerate_intersecting_cells(county.geometry)
            )

    eligible_cells = set(exclude_la_buffer(
        training_candidates,
        la_geometry,
        config["geography"]["buffer_meters"]
    ))

    # All 90 excluded candidate cells
    excluded_cells = training_candidates - eligible_cells

    # 47 of these excluded cells also intersect LA.
    # Draw categories without overlapping colored squares.
    buffer_only_cells = excluded_cells - la_cells

    pilot_cells = select_pilot_cells(
        eligible_cells,
        config["pilot"]["target_cells"]
    )

    fig = go.Figure()

    add_grid_layer(
        fig, eligible_cells,
        "Eligible training", COLORS["training"]
    )

    add_grid_layer(
        fig, buffer_only_cells,
        "Buffer excluded", COLORS["excluded"]
    )

    add_grid_layer(
        fig, la_cells,
        "LA evaluation", COLORS["evaluation"]
    )

    add_county_boundaries(fig, regions)

    # Highlight the 12 pilot cells
    pilot_x = [
        (ix + 0.5) * CELL_SIZE
        for ix, iy in pilot_cells
    ]

    pilot_y = [
        (iy + 0.5) * CELL_SIZE
        for ix, iy in pilot_cells
    ]

    pilot_names = [
        f"{ix}_{iy}" for ix, iy in pilot_cells
    ]

    fig.add_trace(go.Scatter(
        x=pilot_x,
        y=pilot_y,
        mode="markers",
        name="12 pilot cells",
        marker=dict(
            color=COLORS["pilot"],
            size=11,
            symbol="circle",
            line=dict(color="white", width=1.5)
        ),
        text=pilot_names,
        hovertemplate=(
            "Pilot cell: %{text}<extra></extra>"
        )
    ))

    # Use equal scaling so the 10 km squares
    # appear as actual squares on the map.
    fig.update_layout(
        title=(
            "Southern California AI Geographic Grid"
            "<br><sup>"
            f"{len(eligible_cells):,} eligible training cells | "
            f"{len(la_cells)} LA cells | "
            f"{len(excluded_cells)} buffer exclusions | "
            f"{len(pilot_cells)} pilot cells"
            "</sup>"
        ),
        template="plotly_white",
        height=800,
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=-0.12
        ),
        xaxis=dict(
            title="EPSG:5070 Easting (meters)",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="EPSG:5070 Northing (meters)",
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=40, r=30, t=90, b=80)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        OUTPUT,
        include_plotlyjs=True,
        full_html=True
    )

    print("\nAI Geographic Visualization")
    print("-" * 45)
    print("Eligible training cells:", len(eligible_cells))
    print("LA evaluation cells:", len(la_cells))
    print("Excluded candidate cells:", len(excluded_cells))
    print("Buffer-only cells:", len(buffer_only_cells))
    print("Pilot cells:", len(pilot_cells))
    print("Saved map:", OUTPUT)


if __name__ == "__main__":
    main()
