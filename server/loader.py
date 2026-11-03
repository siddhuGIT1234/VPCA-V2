"""Convert VTK files (.vtp, .vtu, .vti, .vtk, .vtm) into point records.

Every dataset type becomes the same thing the rest of the system uses:
    pos  (N, 3) float32   point coordinates
    att  (N, K) float32   one column per scalar attribute
    names                 K attribute names
    label (N,)  int32     zeros unless the file has nothing better

Rules:
- Point data arrays become attributes.
- Cell data arrays are averaged onto points (so a .vtu with only cell data
  still gets attributes). If there are no points' worth of data at all,
  the x, y, z coordinates are used as attributes so coloring still works.
- Scalars (1 component) are kept as-is. 2-3 component arrays (vectors) become
  one column per component plus a magnitude column. Larger tensors keep only
  their magnitude. Text / non-numeric arrays are skipped.
- If the file has more than `max_points` points, a uniform random subset is
  kept (recorded in the manifest so it is never silent).
"""
import numpy as np
import pyvista as pv

COMP = "xyz"


def _columns(name, arr):
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        return []
    arr = arr.astype(np.float64)
    if arr.ndim == 1:
        return [(name, arr)]
    n_comp = arr.shape[1]
    mag = (f"{name}_magnitude", np.linalg.norm(arr, axis=1))
    if n_comp <= 3:
        return [(f"{name}_{COMP[i]}", arr[:, i]) for i in range(n_comp)] + [mag]
    return [mag]


def load_vtk(path, max_points=2_000_000, seed=0):
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if mesh.n_points == 0:
        raise ValueError("File contains no points")

    if mesh.cell_data.keys():
        # Average cell values onto points; keeps existing point arrays.
        mesh = mesh.cell_data_to_point_data(pass_cell_data=False)

    pos = np.asarray(mesh.points, dtype=np.float64)

    columns, skipped = [], []
    for name in mesh.point_data.keys():
        cols = _columns(name, mesh.point_data[name])
        if not cols:
            skipped.append(name)
        columns.extend(cols)
    if not columns:
        columns = [(c, pos[:, i]) for i, c in enumerate(COMP)]

    names = [c[0] for c in columns]
    att = np.stack([c[1] for c in columns], axis=1)
    att = np.nan_to_num(att, nan=0.0, posinf=0.0, neginf=0.0)

    n_total = len(pos)
    if n_total > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(n_total, max_points, replace=False))
        pos, att = pos[keep], att[keep]

    n = len(pos)
    info = {
        "source_type": type(mesh).__name__,
        "points_in_file": int(n_total),
        "points_kept": int(n),
        "subsampled": bool(n_total > n),
        "skipped_arrays": skipped,
    }
    return {
        "record_id": np.arange(n, dtype=np.int32),
        "pos": pos.astype(np.float32),
        "att": att.astype(np.float32),
        "label": np.zeros(n, dtype=np.int32),
        "attribute_names": np.array(names),
    }, info
