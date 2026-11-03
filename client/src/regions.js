// Matches server/regions.py. IDs depend on coordinates, not arrival order.
export function regionForCoords(x, y, z, info) {
  let id = 0, multiplier = 1;
  for (let axis = 0; axis < 3; axis++) {
    const span = info.maximum[axis] - info.minimum[axis];
    const index = span > 0
      ? Math.max(0, Math.min(info.grid[axis] - 1,
        Math.floor(((axis === 0 ? x : axis === 1 ? y : z) - info.minimum[axis]) / span * info.grid[axis]))) : 0;
    id += index * multiplier;
    multiplier *= info.grid[axis];
  }
  return id;
}

export function regionId(point, info) { return regionForCoords(point[0], point[1], point[2], info); }
