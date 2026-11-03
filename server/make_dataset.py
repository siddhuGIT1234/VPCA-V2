"""Generate a synthetic scientific point dataset with a known rare group.

Each record: record_id, x, y, z, temperature, pressure, velocity, density, label.
- Three spatial blobs make up the dominant structure (labels 0, 1, 2).
- A rare group (~1%, label 3) is scattered across ALL blobs spatially but has
  extreme temperature and pressure. Spatial-only sampling tends to miss it;
  attribute-aware methods should find it. This is the ground truth for
  rare-group recall later.

Usage:  python make_dataset.py --n 200000 --seed 42
Output: data/synthetic.npz
"""
import argparse
from pathlib import Path

import numpy as np

ATTRIBUTES = ["temperature", "pressure", "velocity", "density"]


def make(n: int, seed: int, rare_frac: float = 0.01):
    rng = np.random.default_rng(seed)
    n_rare = int(n * rare_frac)
    n_main = n - n_rare

    centers = np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 2.0]])
    scales = np.array([1.2, 0.8, 1.0])
    blob = rng.choice(3, size=n_main, p=[0.5, 0.3, 0.2])
    pos_main = centers[blob] + rng.normal(size=(n_main, 3)) * scales[blob, None]

    # Rare points live inside the same blobs, so space alone can't separate them.
    rare_blob = rng.choice(3, size=n_rare)
    pos_rare = centers[rare_blob] + rng.normal(size=(n_rare, 3)) * scales[rare_blob, None]

    def attrs(pos, blob_ids, rare):
        m = len(pos)
        temp = 300 + 20 * pos[:, 0] + 10 * blob_ids + rng.normal(0, 5, m)
        pres = 100 + 5 * pos[:, 1] + rng.normal(0, 2, m)
        vel = np.linalg.norm(pos, axis=1) * 2 + rng.normal(0, 0.5, m)
        dens = 1.0 + 0.1 * pos[:, 2] + rng.normal(0, 0.05, m)
        if rare:
            temp += 250  # extreme hot spots
            pres += 60
        return np.stack([temp, pres, vel, dens], axis=1)

    a_main = attrs(pos_main, blob, rare=False)
    a_rare = attrs(pos_rare, rare_blob, rare=True)

    pos = np.vstack([pos_main, pos_rare]).astype(np.float32)
    att = np.vstack([a_main, a_rare]).astype(np.float32)
    label = np.concatenate([blob, np.full(n_rare, 3)]).astype(np.int32)

    # Shuffle so record order carries no information.
    perm = rng.permutation(n)
    record_id = np.arange(n, dtype=np.int32)
    return record_id, pos[perm], att[perm], label[perm]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rid, pos, att, label = make(args.n, args.seed)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    np.savez(out / "synthetic.npz", record_id=rid, pos=pos, att=att, label=label,
             attribute_names=np.array(ATTRIBUTES))
    print(f"Wrote {out / 'synthetic.npz'}: {args.n} records, "
          f"{(label == 3).sum()} rare (label 3)")


if __name__ == "__main__":
    main()
