"""Batches for the student from a teacher store: the tile index, inputs standardised per patch
with the stored scaler statistics, targets and teacher quantiles as tensors, a seeded train
sampler, and a balanced held-out sample for evaluation. The sampler, the evaluation sample and
the tag dropout draw from a small seeded generator, so a run with the same seeds draws the same
tiles.
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import multiprocessing
import os
import queue
import threading

import numpy as np
import torch

from aximoe.student import PATCH, POSITIONS, SEQ, TAG_BASE, family_of

MASK64 = (1 << 64) - 1
# smallest scale, matching the precompute's clamp
MIN_SCALE = 1e-6
# environment name of generated series, and the prefix of their channel names
SYNTHETIC = "synthetic"
RECORD = np.dtype([("env", "u1"), ("channel", "<u2"), ("series", "<u4"), ("start", "<u4"),
                   ("bucket", "u1"), ("split", "u1"), ("width", "<u2")])


class Index:
    """`tiles.index.bin` and its code tables, found in the store directory or its parent."""

    def __init__(self, store_dir: str):
        for d in (store_dir, os.path.dirname(os.path.abspath(store_dir))):
            if os.path.exists(os.path.join(d, "tiles.index.bin")):
                break
        else:
            raise FileNotFoundError(f"no tiles.index.bin beside {store_dir}")
        with open(os.path.join(d, "tiles.index.json")) as f:
            self.meta = json.load(f)
        self.tiles = np.fromfile(os.path.join(d, "tiles.index.bin"), dtype=RECORD)
        self.channels: list[str] = self.meta["channels"]
        self.buckets: list[str] = self.meta["buckets"]
        self.splits: list[str] = self.meta["splits"]

    def with_split(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.tiles["split"] == self.splits.index(split))


class Tags:
    """Which channel codes and bin widths the model has a trained row for. Any other channel,
    such as a held-out one, uses the extra `unknown` id, the last row of its table; so does
    any width without training tiles. With `groups`, a map from channel name to group index,
    channels of one group share one row and the table has one row per group plus `unknown`."""

    def __init__(self, index: Index, groups: dict[str, int] | None = None):
        train = index.tiles[index.with_split("train")]
        self.known = np.zeros(len(index.channels), dtype=bool)
        self.known[np.unique(train["channel"])] = True
        # generated series carry no channel identity, so they train the `unknown` row
        self.known &= np.array([not name.startswith(SYNTHETIC) for name in index.channels])
        self.widths = sorted(int(w) for w in np.unique(train["width"]))
        self.rows = len(self.known) if groups is None else max(groups.values()) + 1
        self.row_of_code = np.arange(len(self.known), dtype=np.int64)
        if groups is not None:
            self.row_of_code = np.full(len(self.known), self.rows, dtype=np.int64)
            for code, name in enumerate(index.channels):
                if name in groups:
                    self.row_of_code[code] = groups[name]
            self.known &= self.row_of_code < self.rows

    def count(self) -> int:
        return self.rows + 1

    def unknown(self) -> int:
        return self.rows

    def ids(self, channels: np.ndarray) -> np.ndarray:
        return np.where(self.known[channels], self.row_of_code[channels], self.unknown()).astype(np.int64)

    def resolution_count(self) -> int:
        return len(self.widths) + 1

    def resolution_unknown(self) -> int:
        return len(self.widths)

    def resolution_ids(self, widths: np.ndarray) -> np.ndarray:
        table = {w: i for i, w in enumerate(self.widths)}
        return np.array([table.get(int(w), self.resolution_unknown()) for w in widths], dtype=np.int64)

    def known_channels(self, index: Index) -> dict[str, int]:
        return {name: int(self.row_of_code[code]) for code, name in enumerate(index.channels) if self.known[code]}


def pack_tags(channels: np.ndarray, resolutions: np.ndarray) -> np.ndarray:
    return channels + (resolutions + 1) * TAG_BASE


class Lcg:
    """The Rust trainer's generator, bit for bit, so seeds mean the same here."""

    def __init__(self, seed: int):
        self.state = (seed * 0x9E3779B97F4A7C15 + 1) & MASK64

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.state >> 11

    def below(self, n: int) -> int:
        return self.next_u64() % n

    def unit(self) -> float:
        return self.next_u64() / float(1 << 53)

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def keeps_series(series: int, fraction: float) -> bool:
    """Whether a training series is in the kept `fraction`: a fixed hash of its id, so smaller
    fractions nest inside larger ones and the choice is independent of the run seed."""
    x = (series + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return (x >> 11) / float(1 << 53) < fraction


class Sampler:
    """Draws training tiles with replacement from the `train` split, from the `series_fraction`
    of its series that `keeps_series` selects. With `width_shares`, a map from bin width in
    seconds to its share of the draws, each draw picks the width by share first and then a tile
    of that width; without it a draw is uniform over the training tiles. Tiles of the `synthetic`
    environment are drawn only through `synthetic_share`, the probability that a draw is one of
    them, uniform over them; the width shares and the series fraction apply to the other tiles."""

    def __init__(self, index: Index, seed: int, series_fraction: float = 1.0,
                 width_shares: dict[int, float] | None = None, synthetic_share: float = 0.0):
        train = index.with_split("train")
        envs = index.meta.get("envs", [])
        generated = (index.tiles["env"][train] == envs.index(SYNTHETIC)) if SYNTHETIC in envs else np.zeros(len(train), dtype=bool)
        self.synthetic = train[generated]
        self.synthetic_share = synthetic_share
        if synthetic_share > 0.0 and len(self.synthetic) == 0:
            raise ValueError("the store has no synthetic training tiles")
        train = train[~generated]
        if series_fraction < 1.0:
            keep = np.array([keeps_series(int(s), series_fraction) for s in index.tiles["series"][train]])
            train = train[keep]
        if len(train) == 0:
            raise ValueError("the store has no training tiles for this series fraction")
        self.train = train
        self.series = len(np.unique(index.tiles["series"][train]))
        self.rng = Lcg(seed)
        self.groups: list[tuple[int, float, np.ndarray]] = []
        if width_shares:
            widths = index.tiles["width"][train]
            total = sum(width_shares.values())
            for width, share in sorted(width_shares.items()):
                rows = train[widths == width]
                if len(rows) == 0:
                    raise ValueError(f"the store has no training tiles of width {width} s")
                self.groups.append((width, share / total, rows))

    def __len__(self) -> int:
        return len(self.train)

    def next(self, n: int) -> np.ndarray:
        if not self.groups and self.synthetic_share == 0.0:
            size = len(self.train)
            return self.train[[self.rng.below(size) for _ in range(n)]]
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            if self.synthetic_share > 0.0 and self.rng.unit() < self.synthetic_share:
                out[i] = self.synthetic[self.rng.below(len(self.synthetic))]
                continue
            if not self.groups:
                out[i] = self.train[self.rng.below(len(self.train))]
                continue
            u = self.rng.unit()
            for _, share, rows in self.groups:
                u -= share
                if u < 0.0:
                    break
            out[i] = rows[self.rng.below(len(rows))]
        return out


class Siblings:
    """The group of every real training tile: the training tiles of its channel in its
    environment and namespace (a series' name up to its slash) at its width and start, within
    its source store (`source_sizes`, the tiles of each in index order). `pick` draws other
    tiles of a tile's group."""

    def __init__(self, index: Index, source_sizes: list[int], seed: int):
        t = index.tiles
        spaces: dict[str, int] = {}
        namespace = np.array([spaces.setdefault(str(name).split("/")[0], len(spaces)) for name in index.meta["series"]],
                             dtype=np.int64)
        source = np.repeat(np.arange(len(source_sizes)), source_sizes)
        if len(source) != len(t):
            raise ValueError("the source sizes do not add up to the index")
        train = index.with_split("train")
        envs = index.meta.get("envs", [])
        if SYNTHETIC in envs:
            train = train[t["env"][train] != envs.index(SYNTHETIC)]
        keys = (t["start"][train], t["width"][train], t["channel"][train], namespace[t["series"][train]],
                t["env"][train], source[train])
        order = np.lexsort(keys)
        new = np.ones(len(order), dtype=bool)
        for key in keys:
            v = key[order]
            new[1:] &= v[1:] == v[:-1]
        new = ~new
        new[:1] = True
        self.members = train[order]
        self.first = np.append(np.flatnonzero(new), len(order))
        self.group = np.cumsum(new) - 1
        self.place = np.full(len(t), -1, dtype=np.int64)
        self.place[self.members] = np.arange(len(order))
        self.rng = np.random.default_rng([seed, 7])

    def pick(self, tiles: np.ndarray, k: int, dropout: float = 0.0) -> np.ndarray:
        """Up to `k` other tiles of each tile's group, `[len(tiles), k]`; -1 fills a row whose
        group holds fewer, the row of a tile without a group and, with probability `dropout`,
        the row of any tile, so a model also learns to forecast a series alone."""
        out = np.full((len(tiles), k), -1, dtype=np.int64)
        for i, tile in enumerate(tiles):
            p = self.place[tile] if tile >= 0 else -1
            if p < 0 or (dropout and self.rng.random() < dropout):
                continue
            a, b = self.first[self.group[p]], self.first[self.group[p] + 1]
            others = b - a - 1
            if others <= 0:
                continue
            at = a + self.rng.choice(others, size=min(k, others), replace=False)
            # the draw skips the tile's own place
            out[i, : len(at)] = self.members[at + (at >= p)]
        return out

    def regroup(self, tiles: np.ndarray, widths: np.ndarray, size: int, dropout: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """A draw of `tiles` (with their bin `widths`) turned into groups of up to `size`: each
        tile that has a group and still stands as drawn hands the places of later grouped draws
        of its width to other tiles of its group. A share `dropout` of the grouped draws takes
        no part and stays alone as drawn. Returns the tiles and, per place, the place of its
        group's first tile; a tile alone names itself. The number of tiles per width and the
        places of tiles without a group stay as drawn."""
        tiles = np.array(tiles, dtype=np.int64)
        lead = np.arange(len(tiles), dtype=np.int64)
        free = (tiles >= 0) & (self.place[np.maximum(tiles, 0)] >= 0)
        if dropout:
            free &= self.rng.random(len(tiles)) >= dropout
        for i in range(len(tiles)):
            if not free[i]:
                continue
            free[i] = False
            p = self.place[tiles[i]]
            a, b = self.first[self.group[p]], self.first[self.group[p] + 1]
            later = np.flatnonzero(free & (widths == widths[i]))[: size - 1]
            count = min(len(later), b - a - 1)
            if count <= 0:
                continue
            at = a + self.rng.choice(b - a - 1, size=count, replace=False)
            later = later[:count]
            # the draw skips the tile's own place
            tiles[later], lead[later], free[later] = self.members[at + (at >= p)], i, False
        return tiles, lead


class History:
    """The tile before every tile: the tile of the same source store (`source_sizes`, the tiles
    of each in index order), environment, channel, series and width that ends where the tile
    starts. `before` follows that chain back."""

    def __init__(self, index: Index, source_sizes: list[int]):
        t = index.tiles
        source = np.repeat(np.arange(len(source_sizes)), source_sizes)
        if len(source) != len(t):
            raise ValueError("the source sizes do not add up to the index")
        keys = (t["start"], t["width"], t["series"], t["channel"], t["env"], source)
        order = np.lexsort(keys)
        same = np.ones(len(order) - 1, dtype=bool)
        for key in keys[1:]:
            v = key[order]
            same &= v[1:] == v[:-1]
        start = t["start"][order].astype(np.int64)
        same &= start[1:] - start[:-1] == SEQ * PATCH
        self.previous = np.full(len(t), -1, dtype=np.int64)
        self.previous[order[1:][same]] = order[:-1][same]

    def before(self, tiles: np.ndarray, count: int) -> np.ndarray:
        """The `count` tiles before each tile, `[len(tiles), count]`, the nearest last; -1 where
        the chain ends and for a tile number below zero."""
        out = np.full((len(tiles), count), -1, dtype=np.int64)
        at = np.asarray(tiles, dtype=np.int64)
        for j in range(count - 1, -1, -1):
            at = np.where(at >= 0, self.previous[np.maximum(at, 0)], -1)
            out[:, j] = at
        return out


def history_for(store, index: Index, scales: tuple) -> tuple[History, tuple] | None:
    """What `make_batch` takes as `history` for a model with `scales`: the chain of the store's
    tiles and the scales; nothing without scales."""
    if not scales:
        return None
    sizes = [s["tiles"] for s in store.meta["sources"]] if "sources" in store.meta else [store.n]
    return History(index, sizes), tuple(scales)


def history_values(store, chain: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Tiles `values [n, SEQ * PATCH]` with the values of the tiles `chain [n, count]` before
    them (`History.before`) in front, missing where the chain has no tile."""
    n, count = chain.shape
    steps = SEQ * PATCH
    out = np.full((n, (count + 1) * steps), np.nan, np.float32)
    out[:, count * steps :] = values
    valid = chain >= 0
    if valid.any():
        ids, inverse = np.unique(chain[valid], return_inverse=True)
        read = np.asarray(store.rows(ids, ("values",))["values"], np.float32)
        rows, cols = np.nonzero(valid)
        for row, col, k in zip(rows, cols, inverse):
            out[row, col * steps : (col + 1) * steps] = read[k]
    return out


def history_channel(values: np.ndarray, patches: int, loc: np.ndarray, scale: np.ndarray, scales: tuple) -> np.ndarray:
    """The history input `[n, SEQ, len(scales) * 2 * PATCH]` of tiles whose first `patches`
    patches are the last `patches * PATCH` steps of `values [n, steps]` (NaN where missing). For
    each of those positions and each scale `r`: the means of the 32 bins of `r` steps that end
    with the position's patch, in the asinh space of the position's statistics (`loc`, `scale`,
    `[n, >= patches]`), then the share of each bin's steps that are present. A bin without a
    present step is zero, as is every later position."""
    n, steps = values.shape
    present = np.isfinite(values)
    total = np.concatenate([np.zeros((n, 1)), np.cumsum(np.where(present, values, 0.0), axis=1, dtype=np.float64)], axis=1)
    count = np.concatenate([np.zeros((n, 1), dtype=np.int64), np.cumsum(present, axis=1)], axis=1)
    ends = steps - (patches - 1 - np.arange(patches)) * PATCH
    out = np.zeros((n, SEQ, len(scales), 2, PATCH), np.float32)
    location = loc[:, :patches].astype(np.float64)[:, :, None]
    spread = np.maximum(scale[:, :patches].astype(np.float64), MIN_SCALE)[:, :, None]
    for c, r in enumerate(scales):
        edges = np.clip(ends[:, None] - (PATCH - np.arange(PATCH + 1))[None, :] * r, 0, steps)
        held = count[:, edges[:, 1:]] - count[:, edges[:, :-1]]
        mean = (total[:, edges[:, 1:]] - total[:, edges[:, :-1]]) / np.maximum(held, 1)
        out[:, :patches, c, 0] = np.where(held > 0, np.arcsinh((mean - location) / spread), 0.0)
        out[:, :patches, c, 1] = held / r
    return out.reshape(n, SEQ, len(scales) * 2 * PATCH)


def sibling_channel(store, picks: np.ndarray) -> np.ndarray:
    """The sibling input `[n, SEQ, 2 * PATCH]` for `picks [n, k]` (tile numbers, -1 for none):
    at every step the mean of the picked tiles' standardised inputs over those that hold the
    step, and beside it the share of the `k` picks that hold it."""
    n, k = picks.shape
    out = np.zeros((n, SEQ, 2 * PATCH), np.float32)
    valid = picks >= 0
    if not valid.any():
        return out
    ids, inverse = np.unique(picks[valid], return_inverse=True)
    read = store.rows(ids, ("values", "loc", "scale"))
    inputs, mask, _, _ = standardise(read["values"], read["loc"], read["scale"])
    rows = np.nonzero(valid)[0]
    total = np.zeros((n, SEQ, PATCH), np.float32)
    count = np.zeros((n, SEQ, PATCH), np.float32)
    np.add.at(total, rows, inputs[inverse] * mask[inverse])
    np.add.at(count, rows, mask[inverse])
    out[:, :, :PATCH] = total / np.maximum(count, 1.0)
    out[:, :, PATCH:] = count / k
    return out


def eval_tiles(index: Index, split: str, n: int, seed: int) -> np.ndarray:
    """Up to `n` tiles of `split`, an equal share from each bucket where possible, in tile
    order; empty for a split the store has no tiles of."""
    held = index.with_split(split)
    rng = Lcg(seed)
    out: list[int] = []
    per_bucket = -(-n // len(index.buckets))
    for code in range(len(index.buckets)):
        ours = [int(t) for t in held[index.tiles["bucket"][held] == code]]
        rng.shuffle(ours)
        out.extend(ours[:per_bucket])
    return np.array(sorted(out), dtype=np.int64)


def standardise(values: np.ndarray, loc: np.ndarray, scale: np.ndarray):
    """Inputs and targets of tiles `values [n, 2048]` in the teacher's asinh space. Input
    patch `s` uses the statistics at the end of patch `s`, the last patch reusing the last
    position's; target position `s` is patch `s + 1` with the statistics at the end of patch
    `s`. Missing values are zero with a zero mask."""
    n = values.shape[0]
    # float32 throughout: the precision the model computes in
    loc = np.concatenate([loc, loc[:, -1:]], axis=1).astype(np.float32)[:, :, None]
    scale = np.maximum(np.concatenate([scale, scale[:, -1:]], axis=1).astype(np.float32), np.float32(MIN_SCALE))[:, :, None]
    v = values.reshape(n, SEQ, PATCH).astype(np.float32, copy=False)
    input_mask = np.isfinite(v)
    inputs = np.where(input_mask, np.arcsinh((np.nan_to_num(v) - loc) / scale), np.float32(0.0))
    y = v[:, 1:]
    target_mask = input_mask[:, 1:]
    targets = np.where(target_mask, np.arcsinh((np.nan_to_num(y) - loc[:, :POSITIONS]) / scale[:, :POSITIONS]), np.float32(0.0))
    return inputs, input_mask.astype(np.float32), targets, target_mask.astype(np.float32)


def causal_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`forecast.causal_stats` by cumulative sums in the same order of additions: the running
    mean and the Bessel-corrected deviation of the finite values of `values [n, steps]`, read at
    the last step of each patch."""
    v = values.astype(np.float64)
    present = np.isfinite(v)
    x = np.where(present, v, 0.0)
    count = np.maximum(np.cumsum(present, axis=1, dtype=np.float64), 1.0)
    loc = np.cumsum(x, axis=1) / count
    before = np.concatenate([np.zeros((v.shape[0], 1)), loc[:, :-1]], axis=1)
    m2 = np.cumsum(np.where(present, (x - before) * (x - loc), 0.0), axis=1)
    scale = np.sqrt(m2 / np.maximum(count - 1.0, 1.0))
    ends = slice(PATCH - 1, None, PATCH)
    return loc[:, ends].astype(np.float32), np.maximum(scale[:, ends], MIN_SCALE).astype(np.float32)


def _generate(seed: int, worker: int, chunk: int, family: str, out, siblings: int = 0, state: dict | None = None,
              late: float = 0.0) -> None:
    from aximoe import synthetic
    rng = np.random.default_rng([seed, worker])
    if state is not None:
        rng.bit_generator.state = state

    def draw(width: int) -> np.ndarray:
        # a draw the generator cannot finish is drawn again, so one bad draw does not end the stream
        while True:
            try:
                y = synthetic.series(rng, width, family)
                return synthetic.late_start(rng, y, PATCH) if late and rng.random() < late else y
            except (ArithmeticError, IndexError, ValueError):
                continue

    while True:
        widths = rng.choice(synthetic.WIDTHS, size=chunk)
        values = np.stack([draw(int(w)) for w in widths])
        if siblings:
            # the tile is one member of its group, the others are its siblings
            group = synthetic.members(rng, values, siblings + 1)
            out.put((group[:, 0], widths.astype(np.int64), group[:, 1:], rng.bit_generator.state))
        else:
            out.put((values, widths.astype(np.int64), rng.bit_generator.state))


class SyntheticStream:
    """Generated tiles (`synthetic.series`) from worker processes, each drawn once and never
    again. `take(n)` returns `values [n, 2048]` and their nominal bin widths; the workers are
    read in turn, so a seed gives the same tiles in the same order. With `siblings`, every tile
    is one of a group of generated series that share a shape (`synthetic.members`), and `take`
    also returns that many others of its group, `[n, siblings, 2048]`. `late` is the share of the
    tiles that start late inside one of the model's patches (`synthetic.late_start`). A stream started with the
    `state()` of another continues that one's tiles."""

    def __init__(self, seed: int, workers: int = 6, chunk: int = 32, family: str = "any", siblings: int = 0,
                 state: dict | None = None, late: float = 0.0):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        # each worker's generator state after the last chunk read from it
        self.after = list(state["after"]) if state else [None] * workers
        self.queues = [ctx.Queue(maxsize=4) for _ in range(workers)]
        self.processes = [ctx.Process(target=_generate, args=(seed, w, chunk, family, q, siblings, self.after[w], late), daemon=True)
                          for w, q in enumerate(self.queues)]
        for p in self.processes:
            p.start()
        self.turn = state["turn"] if state else 0
        self.values = state["values"] if state else np.zeros((0, SEQ * PATCH), np.float32)
        self.widths = state["widths"] if state else np.zeros(0, np.int64)
        self.others = (state["others"] if state else np.zeros((0, siblings, SEQ * PATCH), np.float32)) if siblings else None

    def state(self) -> dict:
        """What a new stream needs to continue this one: the workers' generators, whose turn it is and the tiles read but not taken."""
        return {"after": list(self.after), "turn": self.turn, "values": self.values, "widths": self.widths, "others": self.others}

    def take(self, n: int) -> tuple:
        while len(self.widths) < n:
            # a worker that died would otherwise stall the trainer without a word
            values, widths, *others, self.after[self.turn] = self.queues[self.turn].get(timeout=600)
            self.turn = (self.turn + 1) % len(self.queues)
            self.values, self.widths = np.concatenate([self.values, values]), np.concatenate([self.widths, widths])
            if self.others is not None:
                self.others = np.concatenate([self.others, others[0]])
        out = (self.values[:n], self.widths[:n]) + ((self.others[:n],) if self.others is not None else ())
        self.values, self.widths = self.values[n:], self.widths[n:]
        if self.others is not None:
            self.others = self.others[n:]
        return out


def synthetic_batch(values: np.ndarray, widths: np.ndarray, tags: Tags, quantile_dtype, dropout: tuple[float, Lcg] | None = None,
                    raw: bool = False, sibling: bool = False, others: np.ndarray | None = None,
                    alone: tuple[float, np.random.Generator] | None = None, first_place: int | None = None,
                    history: tuple | None = None, past: int | None = None) -> dict:
    """Generated tiles `values [n, 2048]` at nominal bin `widths` as a batch in `make_batch`'s
    layout: statistics from `causal_stats`, the `unknown` channel row, the resolution row of each
    width under the same dropout, zero teacher quantiles and a zero `distill_mask [n, POSITIONS]`,
    so these tiles learn from the truth alone. Under `sibling` the batch holds the sibling input:
    the mean of each tile's `others [n, k, 2048]`, standardised each by its own statistics, and
    zero without them or, with `alone` (a share and its generator), for that share of the tiles.
    With `first_place`, the batch place of the first of these tiles, each tile is a group of its own.
    With `history`, the scales, the batch holds the history input from the tile's own steps.
    With `past`, a count of tiles, the batch holds that many tiles of missing steps as the series before each tile."""
    n = len(widths)
    loc, scale = causal_stats(values)
    loc, scale = loc[:, :POSITIONS], scale[:, :POSITIONS]
    inputs, input_mask, targets, target_mask = standardise(values, loc, scale)
    resolution = tags.resolution_ids(widths)
    if dropout is not None:
        p, rng = dropout
        for i in range(n):
            if rng.unit() < p:
                resolution[i] = tags.resolution_unknown()
    extra = {"values": torch.from_numpy(values), "loc": torch.from_numpy(loc), "scale": torch.from_numpy(scale)} if raw else {}
    if sibling:
        channel = np.zeros((n, SEQ, 2 * PATCH), np.float32)
        if others is not None:
            k = others.shape[1]
            flat = others.reshape(n * k, -1)
            o_loc, o_scale = causal_stats(flat)
            o_inputs, o_mask, _, _ = standardise(flat, o_loc[:, :POSITIONS], o_scale[:, :POSITIONS])
            o_inputs, o_mask = o_inputs.reshape(n, k, SEQ, PATCH), o_mask.reshape(n, k, SEQ, PATCH)
            count = o_mask.sum(1)
            channel[:, :, :PATCH] = (o_inputs * o_mask).sum(1) / np.maximum(count, 1.0)
            channel[:, :, PATCH:] = count / k
            if alone is not None:
                channel[alone[1].random(n) < alone[0]] = 0.0
        extra["sibling"] = torch.from_numpy(channel)
    if first_place is not None:
        extra["group"] = torch.arange(first_place, first_place + n)
    if history is not None:
        extra["history"] = torch.from_numpy(history_channel(values, SEQ, np.concatenate([loc, loc[:, -1:]], axis=1),
                                                            np.concatenate([scale, scale[:, -1:]], axis=1), history))
    if past:
        extra["past"] = torch.full((n, past * SEQ * PATCH), float("nan"), dtype=torch.float32)
    return extra | {
        "inputs": torch.from_numpy(inputs),
        "input_mask": torch.from_numpy(input_mask),
        "targets": torch.from_numpy(targets),
        "target_mask": torch.from_numpy(target_mask),
        "teacher": torch.zeros(n, 9, POSITIONS, PATCH, dtype=quantile_dtype),
        "tags": torch.from_numpy(pack_tags(np.full(n, tags.unknown(), dtype=np.int64), resolution)),
        "families": torch.full((n,), family_of(SYNTHETIC), dtype=torch.int64),
        "tiles": np.full(n, -1, dtype=np.int64),
        "distill_mask": torch.zeros(n, POSITIONS),
    }


def drop_empty_contexts(batch: dict) -> dict:
    """`batch` with the target mask zero at every position whose context holds no observed point."""
    seen = batch["input_mask"].sum(2).cumsum(1)[:, :POSITIONS] > 0
    return batch | {"target_mask": batch["target_mask"] * seen[:, :, None]}


def join(a: dict, b: dict) -> dict:
    """Two batches of the same layout as one, `a`'s tiles first."""
    return {k: (torch.cat([v, b[k]]) if isinstance(v, torch.Tensor) else np.concatenate([v, b[k]])) for k, v in a.items()}


def ahead_targets(values: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, patches: int):
    """Targets of the `patches` patches after each position of tiles `values [n, 2048]`, with the
    scaler statistics `loc`, `scale [n, POSITIONS]`: `[n, POSITIONS, patches * PATCH]` and their
    mask, where position `s` holds patches `s + 1` to `s + patches` in the asinh space of the
    statistics at the end of patch `s`, as `standardise` holds patch `s + 1`; patches past the tile
    are masked."""
    n = values.shape[0]
    v = values.reshape(n, SEQ, PATCH)
    present = torch.isfinite(v)
    v = torch.where(present, v, torch.zeros((), dtype=v.dtype, device=v.device))
    loc = loc[:, :POSITIONS, None].to(v.dtype)
    scale = scale[:, :POSITIONS, None].to(v.dtype).clamp_min(MIN_SCALE)
    ys, ms = [], []
    for k in range(1, patches + 1):
        idx = torch.arange(POSITIONS, device=v.device) + k
        m = present[:, idx.clamp(max=SEQ - 1)] & (idx < SEQ)[None, :, None]
        ys.append(torch.where(m, torch.asinh((v[:, idx.clamp(max=SEQ - 1)] - loc) / scale), 0.0))
        ms.append(m)
    return torch.cat(ys, 2), torch.cat(ms, 2).to(v.dtype)


# steps per patch of a teacher store
STORE_PATCH = 32


class ShiftLabels:
    """A shifted teacher pass: the teacher's forecast of the 32 steps from tile step
    `shift + 32 (position + 1)` on, per training tile, with the statistics of its asinh space."""

    def __init__(self, directory: str):
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        self.tiles = np.fromfile(os.path.join(directory, "tiles.i64"), "<i8")
        n = len(self.tiles)
        self.positions = SEQ * PATCH // STORE_PATCH - 1
        self.loc = np.memmap(os.path.join(directory, "loc.f32"), "<f4", "r", shape=(n, self.positions))
        self.scale = np.memmap(os.path.join(directory, "scale.f32"), "<f4", "r", shape=(n, self.positions))
        self.fd = os.open(os.path.join(directory, "quantiles.f16"), os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)

    def rows(self, tiles: np.ndarray):
        """`found [n]`, and for the found tiles in order `quantiles [m, 9, positions, 32]`, `loc`, `scale [m, positions]`."""
        at = np.minimum(np.searchsorted(self.tiles, tiles), len(self.tiles) - 1)
        found = self.tiles[at] == tiles
        rows = at[found]
        q = np.empty((len(rows), 9, self.positions, STORE_PATCH), "<f2")
        if len(rows) == 0:
            return found, q, np.empty((0, self.positions), np.float32), np.empty((0, self.positions), np.float32)
        size = q[0].nbytes
        flat = memoryview(q).cast("B")
        for k, r in enumerate(rows):
            os.posix_fadvise(self.fd, int(r) * size, size, os.POSIX_FADV_WILLNEED)
        for k, r in enumerate(rows):
            if os.preadv(self.fd, [flat[k * size:(k + 1) * size]], int(r) * size) != size:
                raise IOError(f"short read of row {r}")
        order = np.sort(rows)
        back = np.searchsorted(order, rows)
        return found, q, np.asarray(self.loc[order])[back], np.asarray(self.scale[order])[back]


_shift_labels: ShiftLabels | None = None


def _asinh_half(z: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """asinh of the float64 tensor `z` rounded to float16, zero where `keep` (broadcast over `z`) is false. The
    values are computed in float32, and in float64 where float32's error could change the rounding or `z` lies
    past float32's range, so every result equals the float64 one."""
    y = z.to(torch.float32)
    a = y.abs()
    big = a > 1e18
    square = a * a
    root = torch.sqrt(square + 1.0).add_(1.0)
    y = torch.copysign(torch.log1p(square.div_(root).add_(a)), y)
    y.masked_fill_(~keep, 0.0)
    half = y.to(torch.float16)
    unsure = (y * (1.0 - 1e-6)).to(torch.float16) != (y * (1.0 + 1e-6)).to(torch.float16)
    unsure |= big
    unsure &= keep
    at = unsure.nonzero(as_tuple=True)
    if len(at[0]):
        half[at] = torch.from_numpy(np.arcsinh(z[at].numpy()).astype("<f2"))
    return half


def fine_teacher(tiles: np.ndarray, stored: np.ndarray, loc: np.ndarray, scale: np.ndarray):
    """Teacher labels on a 16-step grid, `[n, 9, POSITIONS, PATCH]` in the frames `loc`, `scale
    [n, POSITIONS]`, and `distill_mask [n, POSITIONS]`. An odd position ends on a store boundary
    and takes the first 16 steps of the store's forecast `stored [n, 9, 63, 32]`; an even position
    from 2 on takes those of the pass in `AXIMOE_SHIFT_TEACHER`, moved into the position's frame."""
    global _shift_labels
    if _shift_labels is None:
        _shift_labels = ShiftLabels(os.environ["AXIMOE_SHIFT_TEACHER"])
    n = len(tiles)
    out = np.zeros((n, 9, POSITIONS, PATCH), stored.dtype)
    mask = np.zeros((n, POSITIONS), np.float32)
    out[:, :, 1::2] = stored[:, :, :, :PATCH]
    mask[:, 1::2] = 1.0
    found, q, t_loc, t_scale = _shift_labels.rows(tiles)
    own_loc, own_scale = loc[found][:, 2::2].astype(np.float64), scale[found][:, 2::2].astype(np.float64)
    shift = torch.from_numpy((t_loc.astype(np.float64) - own_loc) / own_scale)
    stretch = torch.from_numpy(t_scale.astype(np.float64) / own_scale)
    z = torch.sinh(torch.from_numpy(np.ascontiguousarray(q[:, :, :, :PATCH])).to(torch.float64))
    z.mul_(stretch[:, None, :, None]).add_(shift[:, None, :, None])
    # z - z is zero for a finite value and NaN otherwise, so a position is whole when its sum is zero
    whole = (z - z).sum(dim=(1, 3)) == 0
    moved = _asinh_half(z, whole[:, None, :, None]).numpy().astype(stored.dtype, copy=False)
    whole = whole.numpy()
    if found.all():
        out[:, :, 2::2] = moved
        mask[:, 2::2] = whole
    else:
        out[np.flatnonzero(found)[:, None], :, np.arange(2, POSITIONS, 2)[None]] = moved.transpose(0, 2, 1, 3)
        mask[np.flatnonzero(found)[:, None], np.arange(2, POSITIONS, 2)[None]] = whole
    return out, mask


def make_batch(store, index: Index, tags: Tags, tiles: np.ndarray, dropout: tuple[float, Lcg] | None = None,
               raw: bool = False, siblings: tuple[Siblings, int, float] | None = None,
               groups: tuple[Siblings, int, float] | None = None, history: tuple[History, tuple] | None = None,
               past: tuple[History, int] | None = None) -> dict:
    """One batch of `tiles` as CPU tensors. With `dropout`, each tile's channel tag and,
    independently, its resolution tag become `unknown` with that probability, so the unknown
    rows are trained and the model learns to forecast without a specific tag. With `raw`, the
    batch also holds the raw values and the scaler statistics, for `ahead_targets` on the device.
    With `siblings`, the groups, how many tiles to draw from each and the share of tiles that
    draw none, the batch holds the sibling input of `sibling_channel`. With `groups`, the
    groups, their largest size and the share of tiles that stay alone, the draw is regrouped
    (`Siblings.regroup`) and the batch holds `group`, each tile's group as the place of its first tile.
    With `history`, the chain of tiles and the scales, the batch holds the history input of
    `history_channel`, read from each tile and as many tiles before it as the widest scale spans.
    With `past`, the chain of tiles and a count, the batch holds the raw values of that many tiles
    before each tile, `[n, count * 2048]`, missing where the chain has none.
    At the store's patch width the teacher labels are the store's. At 16 steps with a pass in
    `AXIMOE_SHIFT_TEACHER` they are `fine_teacher`'s, and the batch holds their `distill_mask`;
    at any other width the batch holds a zero `distill_mask`."""
    tiles = np.asarray(tiles)
    lead = None
    if groups is not None:
        tiles, lead = groups[0].regroup(tiles, index.tiles["width"][tiles], groups[1], groups[2])
    read = store.rows(tiles)
    values = read["values"]
    # statistics on the model's own patch grid
    loc, scale = causal_stats(values)
    loc, scale = loc[:, :POSITIONS], scale[:, :POSITIONS]
    distill_mask = None
    if PATCH == STORE_PATCH:
        teacher = read["quantiles"]
    elif PATCH == 16 and os.environ.get("AXIMOE_SHIFT_TEACHER"):
        teacher, distill_mask = fine_teacher(tiles, read["quantiles"], loc, scale)
    else:
        teacher = np.zeros((len(tiles), 9, POSITIONS, PATCH), read["quantiles"].dtype)
        distill_mask = np.zeros((len(tiles), POSITIONS), np.float32)
    inputs, input_mask, targets, target_mask = standardise(values, loc, scale)
    info = index.tiles[tiles]
    channel = tags.ids(info["channel"])
    resolution = tags.resolution_ids(info["width"])
    if dropout is not None:
        p, rng = dropout
        for i in range(len(tiles)):
            if rng.unit() < p:
                channel[i] = tags.unknown()
            if rng.unit() < p:
                resolution[i] = tags.resolution_unknown()
    extra = {}
    if siblings is not None:
        extra["sibling"] = torch.from_numpy(sibling_channel(store, siblings[0].pick(tiles, siblings[1], siblings[2])))
    if lead is not None:
        extra["group"] = torch.from_numpy(lead)
    if history is not None:
        chain = history[0].before(tiles, -(-PATCH * max(history[1]) // (SEQ * PATCH)))
        position_loc = np.concatenate([loc, loc[:, -1:]], axis=1)
        position_scale = np.concatenate([scale, scale[:, -1:]], axis=1)
        extra["history"] = torch.from_numpy(history_channel(history_values(store, chain, values), SEQ, position_loc, position_scale, history[1]))
    if past is not None:
        steps = past[1] * SEQ * PATCH
        extra["past"] = torch.from_numpy(np.ascontiguousarray(history_values(store, past[0].before(tiles, past[1]), values)[:, :steps]))
    if raw:
        extra |= {"values": torch.from_numpy(np.asarray(values, np.float32)),
                  "loc": torch.from_numpy(np.asarray(loc, np.float32)),
                  "scale": torch.from_numpy(np.asarray(scale, np.float32))}
    if distill_mask is not None:
        extra["distill_mask"] = torch.from_numpy(distill_mask)
    return extra | {
        "inputs": torch.from_numpy(inputs),
        "input_mask": torch.from_numpy(input_mask),
        "targets": torch.from_numpy(targets),
        "target_mask": torch.from_numpy(target_mask),
        "teacher": torch.from_numpy(teacher),
        "tags": torch.from_numpy(pack_tags(channel, resolution)),
        "families": torch.tensor([family_of(index.channels[c]) for c in info["channel"]], dtype=torch.int64),
        "tiles": tiles,
    }


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class Prefetcher:
    """Assembles the next batches on a thread while the device trains on the current one; with
    `pin`, their tensors sit in page-locked memory, so the copy to a card does not wait."""

    def __init__(self, make, depth: int = 4, pin: bool = False):
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.make = (lambda: {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in make().items()}) if pin else make
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            self.queue.put(self.make())

    def next(self) -> dict:
        return self.queue.get()


class Drawn:
    """The random draws of one batch, made where the run's generators live, handed out again in
    their order where the batch is assembled: `units` as an `Lcg` gives them, `sides` as a
    generator's `integers` gives them."""

    def __init__(self, units: list, sides: np.ndarray | None = None):
        self.units, self.sides, self.at = units, sides, 0

    def unit(self) -> float:
        self.at += 1
        return self.units[self.at - 1]

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        if size != len(self.sides):
            raise ValueError(f"{len(self.sides)} sides were drawn, {size} are asked for")
        return self.sides


class Builders:
    """Assembles batches in `workers` processes and hands them out in the order of their jobs.
    Each process runs `start(spec)` once; `assemble(job)` there returns the batch as
    `(name, value, tensor)` rows, arrays where `tensor` says the batch holds a tensor."""

    def __init__(self, workers: int, start, spec: dict, assemble):
        self.pool = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                                           initializer=start, initargs=(spec,))
        self.assemble, self.ahead = assemble, 2 * workers
        self.pending: collections.deque = collections.deque()

    def next(self, draw) -> dict:
        """The next batch; `draw()` returns the next job and what the batch holds beside the job's rows."""
        while len(self.pending) <= self.ahead:
            job, beside = draw()
            self.pending.append((self.pool.submit(self.assemble, job), beside))
        made, beside = self.pending.popleft()
        return {name: (torch.from_numpy(value) if tensor else value) for name, value, tensor in made.result()} | beside

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


class AheadLink:
    """The preparing process's end of an `Ahead`: the two weight slots, the steps asked for, the
    batches handed back and the trainer's word to stop."""

    def __init__(self, weights: list, go, made, stop):
        self.weights, self.go, self.made, self.stop = weights, go, made, stop

    def wanted(self) -> bool:
        trainer = multiprocessing.parent_process()
        return not self.stop.is_set() and trainer is not None and trainer.is_alive()

    def serve(self, load, prepare) -> None:
        """While the trainer runs: for each step it asks for, `load(weights)` with the weights
        that step's batch reads, then `prepare(step)`, handed to the trainer once the card holds it."""
        while self.wanted():
            try:
                step = self.go.get(timeout=0.2)
            except queue.Empty:
                continue
            load(self.weights[step % 2])
            item = prepare(step)
            if self.weights[0].is_cuda:
                torch.cuda.synchronize()
            while self.wanted():
                try:
                    self.made.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass


class Ahead:
    """Prepares the batches of coming steps in a process of its own, started as `target(*args, link)`
    with an `AheadLink`. The batch of step s is prepared from the weights published after step
    s - 2, the batches of the steps `first` and `first + 1` from `weights`, so a run prepares the
    same batches however long a step or a preparation takes."""

    def __init__(self, target, args: tuple, weights: torch.Tensor, first: int = 0):
        ctx = torch.multiprocessing.get_context("spawn")
        self.weights = [weights.detach().clone().share_memory_() for _ in range(2)]
        self.go, self.made, self.stop = ctx.Queue(), ctx.Queue(maxsize=2), ctx.Event()
        self.process = ctx.Process(target=target, args=(*args, AheadLink(self.weights, self.go, self.made, self.stop)))
        self.process.start()
        self.go.put(first)
        self.go.put(first + 1)

    def next(self):
        """What the preparing process made for the next step."""
        while True:
            try:
                return self.made.get(timeout=0.2)
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError("the preparing process ended") from None

    def publish(self, step: int, weights: torch.Tensor) -> None:
        """Hands `weights`, the ones after `step`, to the batch of step `step + 2`."""
        self.weights[step % 2].copy_(weights)
        if weights.is_cuda:
            # the preparing process reads the slot on its own stream, so the copy is complete before it is told
            torch.cuda.synchronize()
        self.go.put(step + 2)

    def close(self) -> None:
        self.stop.set()
        self.process.join(30)
        if self.process.is_alive():
            self.process.terminate()
