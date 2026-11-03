"""Deterministic spatial regions, independent of random delivery order.

Internal boxes are [min, max); the last box includes the dataset maximum.
These IDs describe a regular spatial grid, not semantic object labels.
Coordinates are the same float32 coordinates sent to the viewer.
"""
import numpy as np


def point_region_ids(points, minimum, maximum, grid):
    points = np.asarray(points)
    ids = np.zeros(len(points), dtype=np.int32)
    multiplier = 1
    for axis in range(3):
        span = float(maximum[axis]) - float(minimum[axis])
        if span > 0:
            q = np.floor((points[:, axis].astype(np.float64) - minimum[axis]) / span * grid[axis])
            idx = np.clip(q, 0, grid[axis] - 1).astype(np.int32)
            ids += idx * multiplier
        multiplier *= grid[axis]
    return ids


def build_regions(points, bins=8):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a nonempty N x 3 point array")
    if not np.isfinite(points).all():
        raise ValueError("Region mapping requires finite coordinates")
    minimum = points.min(axis=0).astype(np.float64)
    maximum = points.max(axis=0).astype(np.float64)
    grid = [bins if maximum[a] > minimum[a] else 1 for a in range(3)]
    counts = np.zeros(int(np.prod(grid)), dtype=np.int64)
    # Bound temporary memory when handling the full Waldeck dataset.
    for start in range(0, len(points), 250_000):
        ids = point_region_ids(points[start:start + 250_000], minimum, maximum, grid)
        counts += np.bincount(ids, minlength=len(counts))
    return {"kind": "spatial_grid", "grid": grid,
            "minimum": minimum.tolist(), "maximum": maximum.tolist(),
            "counts": counts.tolist(), "region_count": len(counts),
            "boundary_rule": "Half-open internal boxes; dataset maximum belongs to last box"}


def describe_region(region_id, info):
    nx, ny, nz = info["grid"]
    if not 0 <= region_id < nx * ny * nz:
        raise ValueError("Unknown region")
    ijk = [region_id % nx, (region_id // nx) % ny, region_id // (nx * ny)]
    bounds = []
    for a in range(3):
        lo, hi = info["minimum"][a], info["maximum"][a]
        step = (hi - lo) / info["grid"][a]
        bounds.extend([lo + ijk[a] * step, lo + (ijk[a] + 1) * step])
    return {"region_id": int(region_id), "grid_index": ijk,
            "bounds": bounds, "total_points": info["counts"][region_id]}
