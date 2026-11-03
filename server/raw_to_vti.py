"""Convert an Open SciVis .raw volume into a .vti file the app can upload.

The file name carries the size and type, e.g.  bonsai_256x256x256_uint8.raw
so usually you only pass the file:

    python raw_to_vti.py C:\\Downloads\\bonsai_256x256x256_uint8.raw

Optional:
    --step 2     keep every 2nd voxel in each direction (8x fewer points).
                 Use this for big volumes; the app keeps at most 2,000,000
                 points anyway (it subsamples above that).
    --name NAME  attribute name in the output (default: "value")

Output: same folder, same name with .vti (e.g. bonsai_256x256x256_uint8_s2.vti)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyvista as pv

DTYPES = {"uint8": "<u1", "uint16": "<u2", "int16": "<i2", "float32": "<f4", "float64": "<f8"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("--dims", help="WxHxD if not in the file name, e.g. 256x256x256")
    p.add_argument("--dtype", choices=DTYPES, help="if not in the file name")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--name", default="value")
    a = p.parse_args()

    path = Path(a.raw)
    m = re.search(r"_(\d+)x(\d+)x(\d+)_(uint8|uint16|int16|float32|float64)", path.name)
    dims = tuple(int(v) for v in (a.dims.split("x") if a.dims else m.groups()[:3]))
    dtype = a.dtype or m.group(4)

    w, h, d = dims
    expected = w * h * d * np.dtype(DTYPES[dtype]).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"Size mismatch: file is {actual:,} bytes, {dims} {dtype} needs {expected:,}")

    # Memory-mapped: the whole file is never loaded at once.
    vol = np.memmap(path, dtype=DTYPES[dtype], mode="r", shape=(d, h, w))   # [z][y][x]
    s = a.step
    sub = np.ascontiguousarray(vol[::s, ::s, ::s]).astype(np.float32)
    dz, dy, dx = sub.shape

    grid = pv.ImageData(dimensions=(dx, dy, dz), spacing=(s, s, s))
    grid.point_data[a.name] = sub.ravel(order="C")   # x varies fastest = VTK order

    out = path.with_name(path.stem + (f"_s{s}" if s > 1 else "") + ".vti")
    grid.save(out)
    print(f"{path.name}: {dims} {dtype} -> {(dx, dy, dz)} = {grid.n_points:,} points")
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if grid.n_points > 2_000_000:
        print("Note: over 2M points, the app will randomly subsample on upload. "
              "Consider --step 2 or higher.")


if __name__ == "__main__":
    main()
