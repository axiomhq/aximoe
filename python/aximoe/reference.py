"""Reference scores: the numbers every student table is compared against, computed from a
teacher store.

Per tile: Winkler (interval score of the q10 to q90 band, width plus 2/alpha times the
distance of every point outside, lower is better), pinball (the quantile loss summed over
the nine knots), the share of targets below q10 and above q90, and the same two scores
for persistence, the no-model baseline that forecasts every point of the next patch as
the last observed value, scored as a zero-width band. All scores live in the teacher's
asinh-standardised space, so a channel in bytes and a channel in seconds are comparable.

Run from the repository root:

    python python/aximoe/reference.py --data data --root derived/<store> --store teacher-313m

--root names the directory under --data holding tiles.parquet, the stores and reference/;
it defaults to aximoe.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from aximoe import store as teacher_store
from aximoe.store import PATCH, POSITIONS

KNOTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ALPHA = 0.2


def standardise(values: np.ndarray, loc: np.ndarray, scale: np.ndarray):
    """Targets of positions 0..62, patches 1..63, in asinh space: ys and mask [B, 63, 32]."""
    y = values[:, PATCH: (POSITIONS + 1) * PATCH].reshape(-1, POSITIONS, PATCH).astype(np.float64)
    m = np.isfinite(y)
    ys = np.arcsinh((np.nan_to_num(y) - loc[:, :, None]) / scale[:, :, None])
    return ys, m


def score_batch(values: np.ndarray, q: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> dict:
    """values [B, 2048], q [B, 9, 63, 32], loc and scale [B, 63]. Winkler is the masked mean
    over all points; pinball is the masked mean per position, summed over knots, averaged
    over positions."""
    ys, m = standardise(values, loc, scale)
    q = q.astype(np.float64)
    lo, hi = q[:, 0], q[:, -1]
    below, above = (ys < lo) & m, (ys > hi) & m
    wk = (hi - lo) + (2.0 / ALPHA) * ((lo - ys) * below + (ys - hi) * above)
    mf = m.astype(np.float64)
    n = mf.sum(axis=(1, 2))
    denom = np.maximum(n, 1.0)
    tau = np.array(KNOTS)[None, :, None, None]
    ysq, mq = ys[:, None], mf[:, None]
    pl = 2.0 * np.abs((ysq - q) * ((ysq <= q) - tau)) * mq
    pinball = (pl.sum(-1) / np.maximum(mq.sum(-1), 1.0)).sum(1).mean(-1)
    return {
        "winkler": (wk * mf).sum(axis=(1, 2)) / denom,
        "pinball": pinball,
        "points": n.astype(np.int64),
        "below_q10": below.sum(axis=(1, 2)) / denom,
        "above_q90": above.sum(axis=(1, 2)) / denom,
    }


def score_tile(values, q, loc, scale) -> dict:
    return {k: v[0].item() for k, v in score_batch(values[None], q[None], loc[None], scale[None]).items()}


def persistence_quantiles(values: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The last observed value at the end of each conditioning patch, forward-filled over
    gaps, as a zero-width band [B, 9, 63, 32] in asinh space."""
    filled = pd.DataFrame(values.T).ffill().to_numpy().T
    last = filled[:, (np.arange(POSITIONS) + 1) * PATCH - 1]
    fhat = np.arcsinh((np.nan_to_num(last, nan=0.0) - np.where(np.isnan(last), 0.0, loc)) / scale)
    return np.broadcast_to(fhat[:, None, :, None], (values.shape[0], len(KNOTS), POSITIONS, PATCH))


def score_persistence(values, loc, scale) -> dict:
    s = score_batch(values[None], persistence_quantiles(values[None], loc[None], scale[None]), loc[None], scale[None])
    return {"winkler": s["winkler"][0].item(), "pinball": s["pinball"][0].item()}


def score_store(store: teacher_store.Store, manifest: pd.DataFrame, chunk: int = 1024, log=lambda m: None) -> pd.DataFrame:
    frames = []
    n = store.n
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        values = np.asarray(store.values[i:j])
        loc, scale = np.asarray(store.loc[i:j], dtype=np.float64), np.asarray(store.scale[i:j], dtype=np.float64)
        s = score_batch(values, np.asarray(store.quantiles[i:j], dtype=np.float32), loc, scale)
        p = score_batch(values, persistence_quantiles(values, loc, scale), loc, scale)
        part = manifest.iloc[i:j]
        frames.append(pd.DataFrame({
            "tile": np.arange(i, j), "env": part.env.to_numpy(), "channel": part.channel.to_numpy(),
            "series": part.series.to_numpy(), "start": part.start.to_numpy(), "split": part.split.to_numpy(),
            "bucket": part.bucket.to_numpy(), **s,
            "winkler_persistence": p["winkler"], "pinball_persistence": p["pinball"],
        }))
        log(f"chunk {j}/{n}")
    return pd.concat(frames, ignore_index=True)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["winkler", "winkler_persistence", "pinball", "pinball_persistence", "below_q10", "above_q90"]
    g = scores.groupby(["split", "bucket"], observed=True)
    out = g[cols].mean()
    out.insert(0, "tiles", g.size())
    return out


def report(table: pd.DataFrame, model: str) -> str:
    lines = [f"mean per tile, {model} against persistence (zero-width band); lower is better",
             f"| {'split':<17} | {'bucket':<6} | {'tiles':>7} | {'Winkler':>8} | {'persist.':>8} | {'pinball':>8} | {'persist.':>8} | {'< q10':>6} | {'> q90':>6} |",
             f"|{'-' * 19}|{'-' * 8}|{'-' * 8}:|{'-' * 9}:|{'-' * 9}:|{'-' * 9}:|{'-' * 9}:|{'-' * 7}:|{'-' * 7}:|"]
    for (split, bucket), r in table.iterrows():
        lines.append(f"| {split:<17} | {bucket:<6} | {int(r.tiles):>7,} | {r.winkler:>8.4f} | {r.winkler_persistence:>8.4f} | "
                     f"{r.pinball:>8.4f} | {r.pinball_persistence:>8.4f} | {r.below_q10:>6.3f} | {r.above_q90:>6.3f} |")
    return "\n".join(lines)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=os.path.normpath(os.path.join(here, "..", "..", "data")))
    ap.add_argument("--root", default="aximoe", help="directory under --data holding tiles.parquet and the stores")
    ap.add_argument("--store", default=None, help="store directory name under <data>/<root>, e.g. teacher or teacher-4m")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    out_dir = os.path.join(args.data, args.root, "reference")
    os.makedirs(out_dir, exist_ok=True)
    if args.store:
        store = teacher_store.Store.open(os.path.join(args.data, args.root, args.store))
        manifest = pd.read_parquet(os.path.join(args.data, args.root, "tiles.parquet"))
        assert len(manifest) == store.n, f"manifest has {len(manifest)} tiles, store {store.n}"
        t0 = time.perf_counter()
        scores = score_store(store, manifest, log=lambda m: print(m, flush=True))
        path = os.path.join(out_dir, f"scores_{args.store}.parquet")
        scores.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        table = summarise(scores)
        table.to_csv(os.path.join(out_dir, f"summary_{args.store}.csv"))
        print(f"\n{report(table, store.meta.get('model', args.store))}\nwrote {path} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
