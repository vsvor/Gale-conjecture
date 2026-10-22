#!/usr/bin/env python3
"""Render ten camera views of the saved 101-point Borsuk configuration.

Input:
    SELECTED_3d.zip, a solution directory, or its points.csv file.

Output:
    Ten PNGs, ten SVGs, a camera/source manifest, and an archive.

Examples:
    python make_101_views.py SELECTED_3d.zip --out figures_101
    python make_101_views.py 3d_088891 --out figures_101
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Eight oblique views around the set, followed by upper and lower views.
# Entries are (name, elevation in degrees, azimuth in degrees).
CAMERAS = [
    ("01_oblique",        25.0,   0.0),
    ("02_oblique",        25.0,  45.0),
    ("03_oblique",        25.0,  90.0),
    ("04_oblique",        25.0, 135.0),
    ("05_oblique",        25.0, 180.0),
    ("06_oblique",        25.0, 225.0),
    ("07_oblique",        25.0, 270.0),
    ("08_oblique",        25.0, 315.0),
    ("09_from_above",     75.0,  35.0),
    ("10_from_below",    -50.0,  35.0),
]


def load_source(path: Path):
    """Read canonical coordinates and their partition without extracting a ZIP."""
    path = path.expanduser()

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist()
                if not name.startswith("__MACOSX/")
                and (
                    name == "3d_088891/points.csv"
                    or name.endswith("/3d_088891/points.csv")
                )
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Expected exactly one 3d_088891/points.csv in the ZIP; "
                    f"found {len(candidates)}."
                )

            coordinate_name = candidates[0]
            prefix = coordinate_name.rsplit("/", 1)[0] + "/"
            coordinate_bytes = archive.read(coordinate_name)
            partition_bytes = archive.read(prefix + "partition.json")
            source_description = f"{path.resolve()}::{coordinate_name}"

    else:
        if path.is_dir():
            if not (path / "points.csv").is_file():
                path = path / "best"
            path = path / "points.csv"

        coordinate_bytes = path.read_bytes()
        partition_bytes = path.with_name("partition.json").read_bytes()
        source_description = str(path.resolve())

    points = np.loadtxt(
        io.StringIO(coordinate_bytes.decode("utf-8-sig")),
        delimiter=",",
        ndmin=2,
    )
    labels = np.asarray(json.loads(partition_bytes))

    if points.shape != (101, 3) or not np.isfinite(points).all():
        raise ValueError("Expected exactly 101 finite points in R^3.")
    if (
        labels.shape != (101,)
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
        or len(np.unique(labels)) != 4
    ):
        raise ValueError("Expected a saved partition into four nonempty parts.")

    return (
        points,
        labels.astype(int),
        source_description,
        hashlib.sha256(coordinate_bytes).hexdigest(),
    )


def render_views(points, labels, output: Path, dpi: int):
    # Recompute numerical geometry, without modifying the original coordinates.
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=2
    )
    diameter = float(distances.max())
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("The configuration must have positive diameter.")

    same_part = labels[:, None] == labels[None, :]
    partition_ratio = float(distances[same_part].max() / diameter)

    # A translation and common scale only: no projection onto fitted curves.
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    xyz = (points - center) / diameter
    half_width = 0.42 * float(np.ptp(xyz, axis=0).max())

    camera_records = []
    part_labels = np.unique(labels)
    part_counts = [int(np.count_nonzero(labels == c)) for c in part_labels]

    for number, (name, elevation, azimuth) in enumerate(CAMERAS, start=1):
        fig = plt.figure(figsize=(12, 9), dpi=dpi)

        # The plotting rectangle ends at x=0.75. All right-column text
        # and the legend begin beyond x=0.78.
        ax = fig.add_axes([0.0, 0.0, 0.95, 0.95], projection="3d")
        handles = []

        for label, count in zip(part_labels, part_counts):
            subset = xyz[labels == label]
            handle = ax.scatter(
                subset[:, 0], subset[:, 1], subset[:, 2],
                s=44,
                depthshade=False,
                label=f"Part {label + 1}: {count} points",
            )
            handles.append(handle)

        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_zlim(-half_width, half_width)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.view_init(elev=elevation, azim=azimuth)

        # All captions belong to the figure, not the 3D axes.
   #     fig.text(
   #         0.045, 0.965,
   #         "101-point configuration",
   #         fontsize=21, weight="bold", va="top",
   #     )
   #     fig.text(
   #         0.045, 0.917,
   #         rf"$R_4 \approx {partition_ratio:.12f}$"
   #         "   |   Diameter normalized to 1",
   #         fontsize=12, va="top",
   #     )
        fig.text(
            0.40, 0.965,
            f"VIEW {number:02d} / 10",
            fontsize=13, weight="bold", va="top",
        )

   #     legend = fig.legend(
   #         handles=handles,
   #         labels=[h.get_label() for h in handles],
   #         loc="upper left",
   #         bbox_to_anchor=(0.785, 0.77),
   #         bbox_transform=fig.transFigure,
   #         borderaxespad=0,
   #         frameon=False,
   #         fontsize=11,
   #         title="Saved partition",
   #         title_fontsize=12,
   #         labelspacing=1.1,
   #     )

        fig.text(
            0.33, 0.93,
            "Orthographic view    "
            f"Elevation: {elevation:g}°   "
            f"Azimuth: {azimuth:g}°",
            fontsize=11,
            va="top",
            linespacing=1.55,
        )
    #    fig.text(
    #        0.045, 0.045,
    #        "All 101 observations retained. "
    #        "No curve fitting or geometric deformation.",
    #        fontsize=10,
    #        va="bottom",
    #    )

        # Check that annotations and the legend do not intrude into the
        # plotting rectangle. No per-vertex labels are drawn.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        plot_box = ax.get_window_extent(renderer)
  #      for artist in [*fig.texts]:
  #          if artist.get_window_extent(renderer).overlaps(plot_box):
  #              plt.close(fig)
  #              raise RuntimeError("An annotation overlaps the plotting area.")

        # Do not use tight cropping: every file keeps an identical canvas.
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=dpi)

        plt.close(fig)
        camera_records.append({
            "file": f"{name}.png",
            "elevation_degrees": elevation,
            "azimuth_degrees": azimuth,
        })
        print(f"Saved view {number:02d}: {name}")

    return {
        "points": 101,
        "dimension": 3,
        "original_diameter": diameter,
        "numerically_recomputed_partition_ratio": partition_ratio,
        "part_counts": part_counts,
        "projection": "orthographic",
        "axes_visible": False,
        "vertex_labels": False,
        "png_dimensions": [12 * dpi, 9 * dpi],
        "cameras": camera_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures_101"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    output = args.out.expanduser()
    archive_path = output.with_name(output.name + ".zip")

    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    if output.exists() or archive_path.exists():
        parser.error("Use a new output name; existing results are not overwritten.")

    try:
        points, labels, description, checksum = load_source(args.source)
        output.mkdir(parents=True)
        report = render_views(points, labels, output, args.dpi)
        report.update({
            "source": description,
            "source_csv_sha256": checksum,
        })
        (output / "views.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

#        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
#            for file in sorted(output.iterdir()):
#                archive.write(file, arcname=f"{output.name}/{file.name}")

        print(f"\nFigures: {output.resolve()}")
        print(f"Archive: {archive_path.resolve()}")

    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
