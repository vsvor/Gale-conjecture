#!/usr/bin/env python3
"""Find a locally optimal arrangement of points on the unit sphere.

The objective is to maximize the smallest Euclidean distance between two
points.  This is a small numerical experiment, not a proof of global
optimality.  For example:

    python sphere_points.py --points 12 --restarts 10 --output points.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def random_points(point_count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Return uniformly sampled directions (up to numerical precision)."""
    points = rng.normal(size=(point_count, dimension))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return the condensed vector of pairwise Euclidean distances."""
    first, second = np.triu_indices(len(points), k=1)
    return np.linalg.norm(points[first] - points[second], axis=1)


def optimize_arrangement(
    point_count: int,
    dimension: int = 3,
    restarts: int = 8,
    max_iterations: int = 1000,
    seed: int | None = 0,
) -> tuple[np.ndarray, float, bool]:
    """Optimize an arrangement and return ``(points, min_distance, success)``.

    SLSQP is started from several random arrangements because this problem has
    many local optima.  The returned score is recomputed after normalizing the
    final coordinates, rather than trusting the optimizer's auxiliary value.
    """
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    first, second = np.triu_indices(point_count, k=1)
    pairs = np.column_stack((first, second))
    pair_rows = np.arange(len(pairs))[:, None]
    coordinate_offsets = np.arange(dimension)
    first_columns = pairs[:, 0, None] * dimension + coordinate_offsets
    second_columns = pairs[:, 1, None] * dimension + coordinate_offsets
    norm_rows = np.repeat(np.arange(point_count), dimension)
    norm_columns = np.arange(point_count * dimension)
    variable_count = point_count * dimension + 1
    rng = np.random.default_rng(seed)
    best_points: np.ndarray | None = None
    best_score = -np.inf
    all_successful = True

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, float]:
        return vector[:-1].reshape(point_count, dimension), float(vector[-1])

    def objective(vector: np.ndarray) -> float:
        # s is a lower bound for every squared pairwise distance.
        return -vector[-1]

    def objective_jacobian(vector: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(vector)
        gradient[-1] = -1.0
        return gradient

    def norm_constraints(vector: np.ndarray) -> np.ndarray:
        points, _ = unpack(vector)
        return np.sum(points * points, axis=1) - 1.0

    def norm_jacobian(vector: np.ndarray) -> np.ndarray:
        points, _ = unpack(vector)
        jacobian = np.zeros((point_count, variable_count))
        jacobian[norm_rows, norm_columns] = 2.0 * points.ravel()
        return jacobian

    def distance_constraints(vector: np.ndarray) -> np.ndarray:
        points, squared_minimum = unpack(vector)
        differences = points[pairs[:, 0]] - points[pairs[:, 1]]
        squared_distances = np.sum(differences * differences, axis=1)
        return squared_distances - squared_minimum

    def distance_jacobian(vector: np.ndarray) -> np.ndarray:
        points, _ = unpack(vector)
        differences = points[pairs[:, 0]] - points[pairs[:, 1]]
        jacobian = np.zeros((len(pairs), variable_count))
        jacobian[pair_rows, first_columns] = 2.0 * differences
        jacobian[pair_rows, second_columns] = -2.0 * differences
        jacobian[:, -1] = -1.0
        return jacobian

    constraints = (
        {"type": "eq", "fun": norm_constraints, "jac": norm_jacobian},
        {"type": "ineq", "fun": distance_constraints, "jac": distance_jacobian},
    )

    for _ in range(restarts):
        initial_points = random_points(point_count, dimension, rng)
        initial_distances = pairwise_distances(initial_points)
        initial = np.r_[initial_points.ravel(), initial_distances.min() ** 2]
        result = minimize(
            objective,
            initial,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=[(None, None)] * (point_count * dimension) + [(0.0, 4.0)],
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-10, "disp": False},
        )
        all_successful &= bool(result.success)

        points, _ = unpack(result.x)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        score = float(pairwise_distances(points).min())
        if score > best_score:
            best_points = points
            best_score = score

    assert best_points is not None  # point_count >= 2 and restarts >= 1
    return best_points, best_score, all_successful


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--points", type=int, default=12, help="number of points (default: 12)")
    parser.add_argument("-d", "--dimension", type=int, default=3, help="ambient dimension (default: 3)")
    parser.add_argument("-r", "--restarts", type=int, default=8, help="random starts (default: 8)")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, help="optional .npy file for the coordinates")
    args = parser.parse_args()

    points, score, success = optimize_arrangement(
        point_count=args.points,
        dimension=args.dimension,
        restarts=args.restarts,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, points)

    print(f"points: {len(points)}, dimension: {points.shape[1]}")
    print(f"minimum distance: {score:.12f}")
    print(f"optimizer status: {'success' if success else 'at least one restart did not converge'}")
    if args.output is not None:
        print(f"saved: {args.output}")
    np.set_printoptions(precision=8, suppress=True)
    print(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
