#!/usr/bin/env python3
"""View a saved borsuk_search solution; no SAT solver is needed.

Examples (run from the extracted borsuk_search directory):
    python view_solution.py best_known/3d_81p_088871
    python view_solution.py results3 --labels --hull
    python view_solution.py results3/best --edges
    python view_solution.py points.csv --save solution.png --no-show
    python view_solution.py best --gif turntable.gif --hull --frames 96

Requires numpy and matplotlib; --hull additionally requires scipy.
GIF output requires Pillow, typically installed together with matplotlib.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_solution(path: Path) -> tuple[Path, np.ndarray, np.ndarray | None, dict]:
    """Accept a result root, a solution directory, or a coordinate file."""
    path = path.expanduser()
    if path.is_dir():
        candidates = [path / 'points.csv', path / 'points.npy',
                      path / 'best' / 'points.csv', path / 'best' / 'points.npy']
        path = next((p for p in candidates if p.is_file()), path / 'points.csv')
    if not path.is_file():
        raise ValueError(f'No coordinate file found: {path}')

    suffix = path.suffix.lower()
    if suffix == '.npy':
        x = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            key = 'points' if 'points' in data else 'x'
            if key not in data:
                raise ValueError('The NPZ file must contain "points" or "x".')
            x = data[key]
    elif suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        x = data.get('points', data.get('x')) if isinstance(data, dict) else data
    else:
        x = np.loadtxt(path, delimiter=',' if suffix == '.csv' else None, ndmin=2)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('Coordinates must be a finite m-by-n array with m >= 2.')

    colors, metadata = None, {}
    if path.name in ('points.csv', 'points.npy'):
        partition = path.with_name('partition.json')
        if partition.is_file():
            data = json.loads(partition.read_text(encoding='utf-8'))
            if data is not None:
                raw = np.asarray(data)
                if raw.shape != (len(x),) or raw.dtype.kind not in 'iu' or (raw < 0).any():
                    raise ValueError('partition.json must contain one nonnegative integer per point.')
                colors = raw.astype(int)
        certificate = path.with_name('certificate.json')
        if certificate.is_file():
            metadata = json.loads(certificate.read_text(encoding='utf-8'))
            if not isinstance(metadata, dict):
                raise ValueError('certificate.json must contain an object.')
            if metadata.get('m', len(x)) != len(x) or metadata.get('dimension', x.shape[1]) != x.shape[1]:
                raise ValueError('certificate.json dimensions do not match the coordinates.')
    return path, x, colors, metadata


def configure_axes(ax, xyz: np.ndarray, axis_names: tuple[str, str, str], show_axes: bool) -> None:
    """Use equal coordinate scales and optionally hide the entire axis frame."""
    midpoint = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = 0.55 * float(np.ptp(xyz, axis=0).max())
    for center, setter in zip(midpoint, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(center - radius, center + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type('ortho')
    if show_axes:
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_zlabel(axis_names[2])
    else:
        ax.set_axis_off()
        # Keep box aspect and projection, but suppress all panes/ticks/spines.
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', type=Path, default=Path('.'),
                        help='Result root, solution directory, or coordinate file (default: .).')
    parser.add_argument('--labels', action='store_true', help='Label vertices using zero-based indices.')
    parser.add_argument('--edges', action='store_true', help='Draw the saved witness graph.')
    parser.add_argument('--hull', action='store_true', help='Draw a triangulated convex-hull wireframe.')
    parser.add_argument('--no-partition', action='store_true', help='Use one point group instead of partition colors.')
    parser.add_argument('--save', type=Path, help='Save a still figure, e.g. solution.png.')
    parser.add_argument('--gif', type=Path, help='Save a rotating animation as a GIF file.')
    parser.add_argument('--frames', type=int, default=72, help='Number of frames in the GIF animation.')
    parser.add_argument('--fps', type=int, default=18, help='Frames per second for GIF output.')
    parser.add_argument('--elev', type=float, default=22.0, help='Camera elevation in degrees.')
    parser.add_argument('--azim-start', type=float, default=35.0, help='Starting azimuth in degrees.')
    parser.add_argument('--azim-stop', type=float, default=395.0,
                        help='Final azimuth in degrees; use start+360 for one full turn.')
    parser.add_argument('--title-size', type=float, default=11.0)
    parser.add_argument('--point-size', type=float, default=55.0)
    parser.add_argument('--axes', action='store_true', help='Show axes, ticks, and labels (hidden by default).')
    parser.add_argument('--no-show', action='store_true', help='Do not open a window; useful with --save or --gif.')
    args = parser.parse_args(argv)
    if args.no_show and args.save is None and args.gif is None:
        parser.error('--no-show requires --save or --gif.')
    if args.frames < 2:
        parser.error('--frames must be at least 2.')
    if args.fps < 1:
        parser.error('--fps must be positive.')

    try:
        path, x, colors, metadata = load_solution(args.path)
        m, dim = x.shape
        distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        diameter = float(distances.max())
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('The configuration must have a finite, positive diameter.')
        print(f'Loaded: {path.resolve()}\nPoints: {m}; dimension: {dim}; diameter: {diameter:.12g}')

        title = f'{m} points in R^{dim}'
        geometry = metadata.get('geometry', {})
        lower, upper = geometry.get('lower_decimal'), geometry.get('upper_decimal')
        if lower is not None and upper is not None:
            k = metadata.get('k', '?')
            interval = f'Saved R_{k} bounds: [{lower}, {upper}]'
            print(interval + ' (not re-certified by this viewer)')
            title += '\n' + interval
        if colors is not None:
            within = colors[:, None] == colors[None, :]
            ratio = float(distances[within].max() / diameter)
            print(f'Saved partition ratio (recomputed in R^{dim}): {ratio:.12g}')

        if dim > 3:
            centered = x - x.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            xyz = centered @ vt[:3].T
            xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
            axis_names = ('PC1', 'PC2', 'PC3')
            title += '\n3D PCA projection: distances may be shortened'
            print('The plot is a PCA projection, not a distance-preserving embedding.')
        else:
            xyz = np.pad(x, ((0, 0), (0, 3 - dim)))
            axis_names = ('x1', 'x2', 'x3')

        segments: list[tuple[np.ndarray, str]] = []
        if args.edges:
            edge_file = path.with_name('witness_edges.csv')
            if not edge_file.is_file() or not edge_file.read_text().strip():
                raise ValueError(f'No nonempty witness edge file found: {edge_file}')
            raw = np.loadtxt(edge_file, delimiter=',', ndmin=2)
            if (raw.shape[1] != 2 or not np.isfinite(raw).all() or
                    (raw < 0).any() or (raw >= m).any() or (raw != np.floor(raw)).any()):
                raise ValueError('witness_edges.csv must contain valid zero-based vertex pairs.')
            edges = raw.astype(int)
            if (edges[:, 0] == edges[:, 1]).any():
                raise ValueError('Witness edges cannot be self-loops.')
            segments.append((xyz[edges], 'witness'))
            witness_ratio = float(distances[edges[:, 0], edges[:, 1]].min() / diameter)
            print(f'Witness: {len(edges)} edges; minimum edge / diameter: {witness_ratio:.12g}')
            print('Witness non-colorability is not checked. Partition colors need not properly color this graph.')
        if args.hull:
            try:
                from scipy.spatial import ConvexHull, QhullError
            except ImportError as exc:
                raise ValueError('--hull requires scipy: python -m pip install scipy') from exc
            try:
                hull = ConvexHull(xyz)
            except QhullError as exc:
                raise ValueError('A 3D hull requires noncoplanar points; omit --hull for this configuration.') from exc
            edges = sorted({tuple(sorted(pair)) for face in hull.simplices
                            for pair in combinations(face, 2)})
            segments.append((xyz[np.asarray(edges, dtype=int)], 'hull'))

        import matplotlib
        if args.no_show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        for lines, kind in segments:
            ax.add_collection3d(Line3DCollection(
                lines, linewidths=0.7 if kind == 'witness' else 1.0,
                alpha=0.18 if kind == 'witness' else 0.35,
                linestyles='-' if kind == 'witness' else '--'))
        if colors is None or args.no_partition:
            ax.scatter(*xyz.T, s=args.point_size, depthshade=False)
        else:
            for label in np.unique(colors):
                selected = xyz[colors == label]
                ax.scatter(*selected.T, s=args.point_size, depthshade=False,
                           label=f'Part {label + 1} ({len(selected)})')
            ax.legend(loc='upper left')
        if args.labels:
            for i, point in enumerate(xyz):
                ax.text(*point, f' {i}', fontsize=8)

        configure_axes(ax, xyz, axis_names, show_axes=args.axes)
        ax.view_init(elev=args.elev, azim=args.azim_start)
        ax.set_title(title, fontsize=args.title_size, pad=18)
        fig.tight_layout()

        if args.save:
            destination = args.save.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=180, bbox_inches='tight')
            print(f'Saved figure: {destination.resolve()}')

        if args.gif:
            destination = args.gif.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            azims = np.linspace(args.azim_start, args.azim_stop, args.frames, endpoint=False)

            def update(frame_index: int):
                ax.view_init(elev=args.elev, azim=float(azims[frame_index]))
                return (ax,)

            anim = animation.FuncAnimation(fig, update, frames=len(azims), interval=1000/args.fps, blit=False)
            try:
                writer = animation.PillowWriter(fps=args.fps)
            except Exception as exc:
                raise ValueError('GIF output requires Pillow. Install it with: python -m pip install pillow') from exc
            anim.save(destination, writer=writer, dpi=140)
            print(f'Saved GIF: {destination.resolve()}')

        if not args.no_show:
            print('Drag in the plot window to rotate; use the toolbar to zoom or save.')
            plt.show()
        plt.close(fig)
    except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
