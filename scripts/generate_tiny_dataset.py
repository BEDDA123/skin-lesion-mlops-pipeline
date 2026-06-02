

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/tiny_synthetic.csv"))
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n_pix = 28 * 28 * 3
    X = rng.integers(0, 256, size=(args.rows, n_pix), dtype=np.int32)
    y = rng.integers(0, 3, size=args.rows, dtype=np.int32)
    cols = [f"pixel{i:04d}" for i in range(n_pix)] + ["label"]
    df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Écrit: {args.out.resolve()} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
