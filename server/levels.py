"""Turn a dataset into nested refinement levels L0 ⊂ L1 ⊂ ... ⊂ D.

A delivery *method* is just a function that returns an ordering (a permutation
of row indices). Levels are prefixes of that ordering, so nesting is
guaranteed by construction and every method gets identical point budgets.

Right now only `random` exists. AAHR, spatial-only and VPCA plug in later as
new entries in METHODS with the same signature.
"""
import numpy as np


def order_random(pos, att, seed=0):
    return np.random.default_rng(seed).permutation(len(pos))


METHODS = {
    "random": order_random,
    # "spatial": order_spatial,   # TODO: hierarchy on xyz only
    # "aahr":    order_aahr,      # TODO: hierarchy on xyz + attributes
}


def level_sizes(n, first=1000, factor=4):
    """Cumulative sizes, e.g. 1000, 4000, 16000, 64000, n."""
    sizes, s = [], first
    while s < n:
        sizes.append(s)
        s *= factor
    sizes.append(n)
    return sizes


# Wire format: one float32 row per record, stride = 4 + n_attributes
#   [record_id, x, y, z, attr_0, ..., attr_k]
# record_id fits exactly in float32 up to 16.7M records.
def pack(record_id, pos, att):
    rows = np.empty((len(pos), 4 + att.shape[1]), dtype=np.float32)
    rows[:, 0] = record_id
    rows[:, 1:4] = pos
    rows[:, 4:] = att
    return rows


def build_levels(data, method="random", seed=0):
    order = METHODS[method](data["pos"], data["att"], seed)
    sizes = level_sizes(len(order))
    levels, prev = [], 0
    for size in sizes:
        idx = order[prev:size]           # only the NEW records for this level
        levels.append(pack(data["record_id"][idx], data["pos"][idx], data["att"][idx]))
        prev = size
    return levels, sizes
