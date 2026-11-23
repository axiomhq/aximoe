"""Pick a channel tag from the series itself. The student runs under every channel row it has,
and the row whose in-context forecasts fit the observed patches best, by the trainer's pinball
loss, is the pick; `unknown` competes like any other row and is the back-off.

    python python/aximoe/tagpick.py --model <dir> --store <teacher store> [--split heldout_container]
        [--tiles 2048] [--seed 0] [--prefix 31] [--batch 256] [--device cuda]

Scores the trainer's evaluation sample of the split: pick accuracy against the true channel at
several context lengths, the per-channel confusion, and the Winkler of every tile's positions
from `--prefix` on under the true row, the row picked from the positions before it and the
`unknown` row, beside the teacher and persistence. The study scores on the store's grid, patches
of 32 steps; the forecaster's pick (`forecast.pick_tags`: `candidates`, `pinball_by_position`,
`picks`) runs at the model's patch width.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aximoe import batches, reference, student  # noqa: E402
from aximoe import store as teacher_store  # noqa: E402
from aximoe.student import FAMILIES, LEVELS, PATCH, POSITIONS, SHARED, TAG_BASE, family_of  # noqa: E402

# positions in LEVELS of the 0.1 and 0.9 knots that bound the Winkler band
LOW, HIGH = 2, 10
PREFIXES = (4, 8, 16, 32, 62)


def candidates(description: dict) -> list[tuple[str, int]]:
    """Every channel row the model trained, by row, then `unknown`."""
    rows = sorted(description["channels"].items(), key=lambda kv: kv[1])
    return rows + [("unknown", description["config"]["tags"] - 1)]


def resolution_rows(description: dict, seconds: np.ndarray) -> np.ndarray:
    """The resolution row of each bin width; a width without training tiles gets `unknown`."""
    widths = description["widths"]
    return np.array([widths.index(int(s)) if int(s) in widths else len(widths) for s in seconds], dtype=np.int64)


def pinball_by_position(pred: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """The trainer's pinball over all levels, summed within each position: `[n, POSITIONS]`
    from quantiles `[n, SEQ, 15, PATCH]`."""
    err = targets[:, :, None] - pred[:, :POSITIONS]
    tau = torch.tensor(LEVELS, device=pred.device, dtype=pred.dtype).reshape(1, 1, -1, 1)
    return (torch.maximum(err * tau, err * (tau - 1.0)) * target_mask[:, :, None]).sum(dim=(2, 3))


def winkler_by_position(lo: np.ndarray, hi: np.ndarray, ys: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The reference scorer's Winkler summed within each position: `[n, POSITIONS]` from a band
    `[n, POSITIONS, PATCH]` and the targets of `reference.standardise`."""
    below, above = (ys < lo) & m, (ys > hi) & m
    wk = (hi - lo) + (2.0 / reference.ALPHA) * ((lo - ys) * below + (ys - hi) * above)
    return (wk * m).sum(-1)


def in_context(model, batch: dict, rows: list[int], res: torch.Tensor, ys: np.ndarray, m: np.ndarray):
    """The batch under each candidate row: pinball sums `[C, n, POSITIONS]` for the pick and
    Winkler sums `[C, n, POSITIONS]` for the scoring."""
    n = batch["inputs"].shape[0]
    pin = torch.zeros(len(rows), n, POSITIONS, device=batch["inputs"].device)
    wink = np.zeros((len(rows), n, POSITIONS))
    with torch.no_grad():
        for c, row in enumerate(rows):
            tags = torch.full((n,), row, dtype=torch.int64, device=res.device) + (res + 1) * TAG_BASE
            pred = model(batch["inputs"], batch["input_mask"], tags)
            pin[c] = pinball_by_position(pred, batch["targets"], batch["target_mask"])
            band = pred[:, :POSITIONS, [LOW, HIGH]].double().cpu().numpy()
            wink[c] = winkler_by_position(band[:, :, 0], band[:, :, 1], ys, m)
    return pin.cpu().numpy(), wink


def picks(pin: np.ndarray, count: np.ndarray, prefix: int) -> np.ndarray:
    """The candidate with the lowest pinball per present point over the first `prefix`
    positions, `[n]`."""
    loss = pin[:, :, :prefix].sum(-1) / np.maximum(count[:, :prefix].sum(-1), 1.0)
    return loss.argmin(0)


def table(rows: list[list[str]], align: str) -> str:
    """A markdown table padded for monospace reading; `align` holds `l` or `r` per column."""
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(r):
        return "| " + " | ".join(c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(r, width, align)) + " |"

    sep = "|" + "|".join("-" * (w + 1) + ":" if a == "r" else "-" * (w + 2) for w, a in zip(width, align)) + "|"
    return "\n".join([line(rows[0]), sep] + [line(r) for r in rows[1:]])


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True, help="teacher store directory, with tiles.index.* beside or above it")
    ap.add_argument("--split", default="heldout_container")
    ap.add_argument("--tiles", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0, help="seed of the trainer's evaluation sample")
    ap.add_argument("--prefix", type=int, default=31, help="positions the pick sees; the later ones are scored")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if PATCH != teacher_store.PATCH:
        sys.exit(f"the study scores against the store's statistics and teacher forecasts at patches of {teacher_store.PATCH} steps; "
                 f"this process runs at {PATCH}")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    description, model = student.load(args.model, device)
    model.eval()
    store = teacher_store.Store.open(args.store)
    index = batches.Index(args.store)
    tags = batches.Tags(index)
    tiles = batches.eval_tiles(index, args.split, args.tiles, args.seed)
    cand = candidates(description)
    names, rows = [c[0] for c in cand], [c[1] for c in cand]
    slot = {row: i for i, row in enumerate(rows)}
    unknown = len(cand) - 1
    prefixes = tuple(sorted(set(PREFIXES + (args.prefix,))))
    tail = slice(args.prefix, POSITIONS)

    t0 = time.perf_counter()
    truth, bucket, picked = [], [], {p: [] for p in prefixes}
    winkler = {k: [] for k in ("true", "picked", "unknown", "teacher", "persistence")}
    check = 0.0
    total = -(-len(tiles) // args.batch)
    for i in range(0, len(tiles), args.batch):
        chunk = tiles[i : i + args.batch]
        b = batches.to_device(batches.make_batch(store, index, tags, chunk), device)
        info = index.tiles[chunk]
        res = torch.from_numpy(resolution_rows(description, info["width"])).to(device)
        values = np.asarray(store.values[chunk])
        loc = np.asarray(store.loc[chunk], dtype=np.float64)
        scale = np.maximum(np.asarray(store.scale[chunk], dtype=np.float64), batches.MIN_SCALE)
        ys, m = reference.standardise(values, loc, scale)
        pin, wink = in_context(model, b, rows, res, ys, m)
        count = b["target_mask"].sum(-1).cpu().numpy()
        points = m.sum(-1)
        n = len(chunk)
        true_slot = np.array([slot[description["channels"].get(index.channels[c], rows[unknown])] for c in info["channel"]])
        truth.extend(true_slot.tolist())
        bucket.extend(index.buckets[k] for k in info["bucket"])
        for p in prefixes:
            picked[p].extend(picks(pin, count, p).tolist())
        chosen = np.array(picked[args.prefix][-n:])
        tail_points = np.maximum(points[:, tail].sum(-1), 1.0)
        arange = np.arange(n)
        winkler["true"].extend((wink[true_slot, arange][:, tail].sum(-1) / tail_points).tolist())
        winkler["picked"].extend((wink[chosen, arange][:, tail].sum(-1) / tail_points).tolist())
        winkler["unknown"].extend((wink[unknown][:, tail].sum(-1) / tail_points).tolist())
        tq = b["teacher"].double().cpu().numpy()
        winkler["teacher"].extend((winkler_by_position(tq[:, 0], tq[:, -1], ys, m)[:, tail].sum(-1) / tail_points).tolist())
        pq = reference.persistence_quantiles(values, loc, scale)
        winkler["persistence"].extend((winkler_by_position(pq[:, 0], pq[:, -1], ys, m)[:, tail].sum(-1) / tail_points).tolist())
        # the per-position Winkler of the true row, summed over every position, is the reference scorer's number
        with torch.no_grad():
            pred = model(b["inputs"], b["input_mask"], b["tags"])[:, :POSITIONS, list(SHARED)]
        ref = reference.score_batch(values, np.ascontiguousarray(pred.float().cpu().numpy().transpose(0, 2, 1, 3)), loc, scale)
        mine = wink[true_slot, arange].sum(-1) / np.maximum(points.sum(-1), 1.0)
        check = max(check, float(np.abs(mine - ref["winkler"]).max()))
        print(f"batch {i // args.batch + 1}/{total}", flush=True)
    seconds = time.perf_counter() - t0

    truth = np.array(truth)
    bucket = np.array(bucket)
    family = np.array([family_of(names[t]) for t in truth])
    groups = [("all", np.ones(len(truth), bool))] + [(bk, bucket == bk) for bk in index.buckets]
    counts = np.bincount(truth, minlength=len(cand))
    top = counts.argmax()
    fam_counts = np.bincount(family, minlength=len(FAMILIES))
    print(f"\n{args.split}: {len(tiles)} tiles, {len(cand)} candidate rows ({len(cand) - 1} channels and unknown), "
          f"{len(tiles) * len(cand)} forward passes in {seconds:.0f} s on {args.device}")
    print(f"reference check: largest difference between the per-position Winkler summed over all positions and "
          f"reference.score_batch, true row: {check:.2e}")
    print(f"baselines: chance 1/{len(cand) - 1} = {pct(1 / (len(cand) - 1))} per channel; the most common channel "
          f"({names[top]}) is {pct(counts[top] / len(truth))} of the tiles; the most common family "
          f"({FAMILIES[fam_counts.argmax()]}) is {pct(fam_counts.max() / len(truth))}\n")
    print("pick accuracy by context length (positions the pick sees; one position is one 32-step patch)")
    rows_a = [["context, patches"] + [f"channel, {g}" for g, _ in groups] + [f"family, {g}" for g, _ in groups] + ["unknown picked"]]
    for p in prefixes:
        pk = np.array(picked[p])
        fam_pk = np.array([family_of(names[k]) if k != unknown else -1 for k in pk])
        rows_a.append([str(p)] + [pct((pk == truth)[g].mean()) for _, g in groups]
                      + [pct((fam_pk == family)[g].mean()) for _, g in groups] + [pct((pk == unknown).mean())])
    print(table(rows_a, "l" + "r" * (len(rows_a[0]) - 1)))
    print(f"\nper channel, pick from {args.prefix} positions")
    pk = np.array(picked[args.prefix])
    rows_b = [["channel", "tiles", "picked right", "most frequent pick", "its share"]]
    for t in np.argsort(names[:-1]):
        sel = truth == t
        if not sel.any():
            continue
        freq = np.bincount(pk[sel], minlength=len(cand))
        rows_b.append([names[t], str(sel.sum()), pct((pk[sel] == t).mean()), names[freq.argmax()], pct(freq.max() / sel.sum())])
    print(table(rows_b, "lrrlr"))
    print(f"\nWinkler of positions {args.prefix} to {POSITIONS - 1} (lower is better), the pick from the positions before")
    rows_c = [["bucket", "tiles", "true row", "picked row", "unknown row", "teacher", "persistence", "picked vs true", "unknown vs true"]]
    w = {k: np.array(v) for k, v in winkler.items()}
    for g, sel in groups:
        t, pkd, u = w["true"][sel].mean(), w["picked"][sel].mean(), w["unknown"][sel].mean()
        rows_c.append([g, str(sel.sum()), f"{t:.4f}", f"{pkd:.4f}", f"{u:.4f}", f"{w['teacher'][sel].mean():.4f}",
                       f"{w['persistence'][sel].mean():.4f}", f"{100 * (pkd / t - 1):+.2f} %", f"{100 * (u / t - 1):+.2f} %"])
    print(table(rows_c, "l" + "r" * 8))


if __name__ == "__main__":
    main()
