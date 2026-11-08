import json
from pathlib import Path
import numpy as np

SEED = 42
RAW = Path("data/raw")
OUT = Path("results")


def gains(score, y, weight=None, points=20):
    """Cumulative share of positives captured when targeting the top-k share by score."""
    w = np.ones_like(y, dtype=float) if weight is None else weight
    o = np.argsort(-score, kind="stable")
    cum_pos = np.cumsum((y * w)[o]); cum_n = np.cumsum(w[o])
    out = []
    for k in np.linspace(0, 1, points + 1):
        if k == 0:
            out.append([0.0, 0.0]); continue
        i = np.searchsorted(cum_n, k * cum_n[-1])
        i = min(i, len(o) - 1)
        out.append([round(float(k), 3), round(float(cum_pos[i] / cum_pos[-1]), 4)])
    return out


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    json.dump(obj, open(OUT / f"{name}.json", "w"), indent=2, default=float)
