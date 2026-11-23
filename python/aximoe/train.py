"""Train the student against a teacher store.

    python python/aximoe/train.py --store data/derived/<store>/teacher-313m --batch 64 --lr 1.2e-3 \\
        --warmup 200 --steps 100000 --eval-every 10000 --eval-tiles 2048 --eval-seed 0 \\
        --save-every 50000 --seed 0 --lambda 1.0:1.0 --save data/models/<widths>-<steps>/seed<n>

A run is named by its data and recipe, one directory per seed. The published model's command
is in the README, section Training.

The loss is pinball on the truth at all fifteen levels plus a Huber distance between the
student's nine shared knots and the teacher's, weighted by `--lambda a:b`, annealed from a to b
over the run. `--tail-teacher DIR`, the output of a Chronos-2 pass on this store's
training tiles, adds a Huber distance between the student's 0.01, 0.05, 0.95 and 0.99 and
Chronos-2's at each tile's stored positions, weighted by `--tail-lambda a:b` the same way; it is
divided by its own points times nine, so in expectation over positions each of its four levels
weighs as each of the distillation term's nine. `--relative-loss ALPHA` weighs both terms of a
stored synthetic tile by its teacher's pinball on the truth to the power minus ALPHA, so a tile
the teacher forecasts closely counts as much as one it cannot. `--ahead-teacher DIRS`, far-label passes
over this store's training tiles, gives a tile of theirs the tail mask its pass
drew and the teacher's forecast of the hidden patches as the teacher term at the positions after
the cut, which otherwise learn from the truth alone; a tile that trains the rollout and whose
stored forecast hides no more patches than the step unrolls is unrolled that many patches and
learns the same forecast at the patches fed back. Every `--eval-every` steps the held-out-time
sample is scored against the teacher and persistence, and every split in `--eval-splits` at the
end. `--save DIR` writes `student.pt`
and `model.json` into DIR (snapshots as `student-<step>.pt`); `--load DIR` starts from a saved
model, so `--load DIR --steps 0` only evaluates it. `--compile off` runs the model eagerly;
`--dtype bf16` runs the forward and backward under bfloat16 autocast, `--dtype fp16` under
float16 autocast with the loss scaled for the backward pass. `--device` is `cuda` by default,
`cpu` or `mps` on a Mac.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aximoe import batches, fastpass, guide, periodic, reference, student  # noqa: E402
from aximoe import store as teacher_store  # noqa: E402
from aximoe.student import LEVELS, MEDIAN, PATCH, POSITIONS, SEQ, SHARED, Student, StudentConfig  # noqa: E402

TAIL_LEVELS = (0, 1, 11, 12, 13, 14)


class TailTeacher:
    """A Chronos-2 pass over this store's tiles: per tile its stored forecast positions
    and the teacher's quantiles there, in the store's asinh space."""

    def __init__(self, directory: str, store_tiles: int):
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        self.levels = [LEVELS.index(v) for v in self.meta["levels"]]
        tiles = np.fromfile(os.path.join(directory, "tiles.i64"), "<i8")
        k = self.meta["positions_per_tile"]
        self.positions = np.fromfile(os.path.join(directory, "positions.u8"), np.uint8).reshape(len(tiles), k)
        self.quantiles = np.memmap(os.path.join(directory, "quantiles.f16"), "<f2", "r",
                                   shape=(len(tiles), len(self.levels), k, PATCH))
        self.row = np.full(store_tiles, -1, np.int64)
        self.row[tiles] = np.arange(len(tiles))

    def lookup(self, tiles: np.ndarray) -> dict:
        """The teacher's quantiles `tail_q [n, levels, k, PATCH]` at positions `tail_pos [n, k]` of
        `tiles`, and `tail_mask [n, k, PATCH]`, zero for a tile outside the pass or a non-finite quantile."""
        rows = self.row[np.asarray(tiles)]
        found = rows >= 0
        k = self.positions.shape[1]
        q = np.zeros((len(rows), len(self.levels), k, PATCH), np.float32)
        pos = np.zeros((len(rows), k), np.int64)
        q[found] = self.quantiles[rows[found]]
        pos[found] = self.positions[rows[found]]
        mask = np.isfinite(q).all(axis=1) & found[:, None, None]
        return {"tail_q": torch.from_numpy(np.nan_to_num(q)), "tail_pos": torch.from_numpy(pos),
                "tail_mask": torch.from_numpy(mask.astype(np.float32))}


class AheadTeacher:
    """Far-label passes over this store's tiles: per tile the count of hidden tail
    patches its forecast was made under and the teacher's quantiles at the last positions.

    On the 16-step grid a pass made from tile step `shift` (0 or 16) with `j` hidden 32-step
    patches cuts the tile at step `shift + 32 (63 - j)`: a hidden count of `2 j + 1 - shift / 16`
    patches of 16 steps. Its forecasts are halved into the positions from `FIRST + shift / 16`
    on. A tile held by a pass of either shift takes one of the two per draw."""

    STORED = 32
    FIRST = 67

    def __init__(self, directories: list[str], store_tiles: int, seed: int = 0):
        self.metas, self.fds = [], []
        # per shift (0, 16) and store tile: the pass that holds it (-1 for none), its row there and its count of hidden patches
        self.part = np.full((2, store_tiles), -1, np.int64)
        self.row = np.zeros((2, store_tiles), np.int64)
        self.hidden = np.zeros((2, store_tiles), np.int64)
        self.rng = np.random.default_rng([seed, 7])
        for k, directory in enumerate(directories):
            with open(os.path.join(directory, "meta.json")) as f:
                self.metas.append(json.load(f))
            shift = self.metas[-1].get("shift", 0)
            if shift not in ((0,) if PATCH == self.STORED else (0, 16)):
                raise ValueError(f"{directory} was made from step {shift}")
            side = shift // 16
            tiles = np.fromfile(os.path.join(directory, "tiles.i64"), "<i8")
            self.part[side, tiles], self.row[side, tiles] = k, np.arange(len(tiles))
            self.hidden[side, tiles] = np.fromfile(os.path.join(directory, "hidden.u8"), np.uint8)
            fd = os.open(os.path.join(directory, "quantiles.f16"), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self.fds.append(fd)
        self.ahead = self.metas[0]["ahead"]
        if any(m["ahead"] != self.ahead for m in self.metas):
            raise ValueError("the passes hide different numbers of patches")
        self.shifted = batches.ShiftLabels(os.environ["AXIMOE_SHIFT_TEACHER"]) if (self.part[1] >= 0).any() else None
        if self.shifted is not None and self.shifted.meta["shift"] != 16:
            raise ValueError("the pass in AXIMOE_SHIFT_TEACHER does not read from step 16")

    def held(self, tiles: np.ndarray) -> np.ndarray:
        """Which of `tiles` a pass holds."""
        return (self.part[:, tiles] >= 0).any(axis=0)

    def lookup(self, tiles: np.ndarray) -> dict:
        """`ahead_q`, the teacher's quantiles at the last positions of `tiles`, and `ahead_hidden
        [n]`, the count its forecast was made under: zero for a tile outside the passes, a
        generated tile (number -1) or a forecast with a non-finite quantile. On the store's grid
        `ahead_q` is `[n, 9, ahead, PATCH]`. On the 16-step grid it is `[n, 9, 2 ahead, PATCH]`
        for the positions from `FIRST` on, with `ahead_first [n]`, the first position that holds a
        forecast, and `ahead_loc`, `ahead_scale [n]`, the statistics of the forecast's asinh
        space where they are those of the pass read from step 16, else NaN."""
        tiles = np.asarray(tiles)
        known = np.maximum(tiles, 0)
        part = np.where(tiles[None] >= 0, self.part[:, known], -1)
        both = (part >= 0).all(axis=0)
        side = np.where(both, self.rng.integers(0, 2, len(tiles)), (part[1] >= 0).astype(np.int64))
        at = np.arange(len(tiles))
        part, row, count = part[side, at], self.row[side, known], self.hidden[side, known]
        q = np.zeros((len(tiles), len(SHARED), self.ahead, self.STORED), np.dtype("<f2"))
        size = q.itemsize * math.prod(q.shape[1:])
        flat = memoryview(q).cast("B")
        teacher_store.read_rows([(self.fds[part[i]], int(row[i]) * size, flat[i * size:(i + 1) * size])
                           for i in np.flatnonzero(part >= 0)])
        hidden = np.where(part >= 0, count, 0)
        # a float16 is non-finite when its exponent bits are all set; the bit test runs on integers
        wild = (q.view("<u2") & 0x7C00) == 0x7C00
        hidden[wild.any(axis=(1, 2, 3))] = 0
        q[wild] = 0.0
        if PATCH == self.STORED:
            return {"ahead_q": torch.from_numpy(q), "ahead_hidden": torch.from_numpy(hidden)}
        halves = q.reshape(len(tiles), len(SHARED), 2 * self.ahead, PATCH)
        moved = side == 1
        fine = halves.copy()
        # read from step 16, a forecast starts one position later and its last half lies past the tile
        fine[moved, :, 1:] = halves[moved, :, :-1]
        fine[moved, :, 0] = 0.0
        loc = np.full(len(tiles), np.nan, np.float32)
        scale = np.full(len(tiles), np.nan, np.float32)
        late = np.flatnonzero(moved & (hidden > 0))
        if len(late):
            rows = np.minimum(np.searchsorted(self.shifted.tiles, tiles[late]), len(self.shifted.tiles) - 1)
            found = self.shifted.tiles[rows] == tiles[late]
            cut = self.shifted.positions - 1 - hidden[late]
            hidden[late[~found]] = 0
            for i in np.flatnonzero(found):
                loc[late[i]], scale[late[i]] = self.shifted.loc[rows[i], cut[i]], self.shifted.scale[rows[i], cut[i]]
        return {"ahead_q": torch.from_numpy(fine), "ahead_hidden": torch.from_numpy(np.where(hidden > 0, 2 * hidden + 1 - side, 0)),
                "ahead_first": torch.from_numpy(self.FIRST + side), "ahead_loc": torch.from_numpy(loc),
                "ahead_scale": torch.from_numpy(scale)}


def prepare_main(argv: list, link) -> None:
    """The preparing process of `--unroll-ahead on`: the trainer's setup on the same command line,
    then the batches of the steps the trainer asks for through `link`."""
    sys.stdout = open(os.devnull, "w")
    main(argv, link)


# what a builder process assembles its batches from: `start_builder` fills it
BUILD: dict = {}


def start_builder(spec: dict) -> None:
    """Opens in a builder process the store and the label passes that `spec` names."""
    torch.set_num_threads(1)
    store = teacher_store.Store.open(spec["store"])
    index = batches.Index(spec["store"])
    groups = None
    if spec["tag_groups"]:
        with open(spec["tag_groups"]) as f:
            groups = json.load(f)["map"]
    BUILD.update(spec, store=store, index=index, tags=batches.Tags(index, groups),
                 ahead=AheadTeacher(spec["ahead"].split(","), store.n) if spec["ahead"] else None,
                 tail=TailTeacher(spec["tail"], store.n) if spec["tail"] else None)


def draw_batch(sampler, tag_rng, stream, ahead, batch: int, streamed: int) -> dict:
    """The job of one batch: its stored tiles, its generated tiles and every random draw its
    assembly reads, each made here in the order `assemble_batch` reads them."""
    tiles = sampler.next(batch - streamed)
    units = [tag_rng.unit() for _ in range(2 * len(tiles))]
    values = widths = None
    if stream:
        values, widths = stream.take(streamed)
        units += [tag_rng.unit() for _ in range(streamed)]
    return {"tiles": tiles, "units": units, "values": values, "widths": widths,
            "sides": ahead.rng.integers(0, 2, batch) if ahead else None}


def assemble_batch(job: dict) -> list:
    """The batch of `job` from the sources in `BUILD`, as `(name, value, tensor)` rows for `batches.Builders`."""
    drawn = batches.Drawn(job["units"], job["sides"])
    b = batches.make_batch(BUILD["store"], BUILD["index"], BUILD["tags"], job["tiles"], (BUILD["tag_dropout"], drawn), raw=BUILD["raw"])
    if job["values"] is not None:
        b.setdefault("distill_mask", torch.ones(len(job["tiles"]), POSITIONS))
        b = batches.join(b, batches.synthetic_batch(job["values"], job["widths"], BUILD["tags"], b["teacher"].dtype,
                                                    (BUILD["tag_dropout"], drawn), raw=BUILD["raw"]))
    if not BUILD["empty_context_loss"]:
        b = batches.drop_empty_contexts(b)
    if BUILD["ahead"]:
        BUILD["ahead"].rng = drawn
        b = b | BUILD["ahead"].lookup(b["tiles"])
    if BUILD["tail"]:
        b = b | BUILD["tail"].lookup(b["tiles"])
    return [(name, value.numpy() if isinstance(value, torch.Tensor) else value, isinstance(value, torch.Tensor))
            for name, value in b.items()]


def ahead_labels(batch: dict, j: torch.Tensor, truth: float = 1.0, own: bool = False) -> dict:
    """`batch` after `mask_tail(batch, j)` with the teacher's forecast of the hidden patches as
    the teacher term at the positions after the cut, for each tile whose `j` is the count its
    stored forecast (`ahead_q`, `ahead_hidden`) was made under. `truth_weight [n, POSITIONS]`
    is `truth` at those positions and what the batch held elsewhere: the weight of the truth's
    pinball there. Under `own`, `batch` comes from `unroll` with the depths `j`, where a position
    after the cut keeps its own statistics: the stored forecast, made in the asinh space of the
    statistics at the cut, is put into each position's."""
    span = batch["ahead_q"].shape[2]
    cut = POSITIONS - 1 - j
    stored = (batch["ahead_hidden"] == j) & (j > 0)
    at = torch.arange(POSITIONS, device=j.device)[None]
    after = (at > cut[:, None]) & stored[:, None]
    if "ahead_first" in batch:
        after = after & (at >= batch["ahead_first"][:, None])
    teacher_q = batch["teacher"].clone()
    last = teacher_q[:, :, POSITIONS - span:]
    q = batch["ahead_q"]
    if own:
        rows = torch.arange(len(j), device=j.device)
        loc, scale = batch["loc"].double(), batch["scale"].double().clamp_min(batches.MIN_SCALE)
        space_loc, space_scale = loc[rows, cut], scale[rows, cut]
        if "ahead_loc" in batch:
            given = torch.isfinite(batch["ahead_loc"])
            space_loc = torch.where(given, batch["ahead_loc"].double(), space_loc)
            space_scale = torch.where(given, batch["ahead_scale"].double(), space_scale)
        raw = space_loc[:, None, None, None] + space_scale[:, None, None, None] * torch.sinh(q.double())
        q = torch.asinh((raw - loc[:, None, POSITIONS - span:, None]) / scale[:, None, POSITIONS - span:, None])
    teacher_q[:, :, POSITIONS - span:] = torch.where(after[:, None, POSITIONS - span:, None], q.to(last.dtype), last)
    kind = batch["distill_mask"].dtype
    weight = torch.where(after, truth, 1.0).to(kind)
    return batch | {"teacher": teacher_q, "distill_mask": torch.maximum(batch["distill_mask"], after.to(kind)),
                    "truth_weight": weight * batch["truth_weight"] if "truth_weight" in batch else weight}


def tail_loss(pred: torch.Tensor, batch: dict, levels: list[int]) -> torch.Tensor:
    """`pred`: `[n, SEQ, 15, PATCH]`. The Huber distance (delta 1, in asinh space) between the
    student's `levels` and the tail teacher's at each tile's stored positions, masked to present
    targets, to positions whose context is all truth, and by `tail_mask`; divided by the masked
    points times nine."""
    pos = batch["tail_pos"]
    n, k = pos.shape
    rows = torch.arange(n, device=pred.device)[:, None].expand(n, k)
    diff = pred[rows, pos][:, :, levels] - batch["tail_q"].transpose(1, 2)
    huber = torch.where(diff.abs() < 1.0, 0.5 * diff * diff, diff.abs() - 0.5)
    m = batch["target_mask"][rows, pos] * batch["tail_mask"]
    if "distill_mask" in batch:
        m = m * batch["distill_mask"][rows, pos][:, :, None]
    return (huber * m[:, :, None]).sum() / (m.sum().clamp_min(1.0) * len(SHARED))


def loss_parts(pred: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """`pred`: `[n, SEQ, 15, PATCH]`. Pinball on the truth over all levels, and the Huber
    distance (delta 1, in asinh space) between the shared knots and the teacher's; both
    masked and averaged over the present target points."""
    q = pred[:, :POSITIONS]
    y = batch["targets"][:, :, None]
    m = batch["target_mask"][:, :, None]
    if "weight" in batch:
        m = m * batch["weight"][:, None, None, None]
    count = batch["target_mask"].sum().clamp_min(1.0)
    tau = torch.tensor(LEVELS, device=pred.device, dtype=pred.dtype).reshape(1, 1, -1, 1)
    err = y - q
    pinball = torch.maximum(err * tau, err * (tau - 1.0)) * m
    if "truth_weight" in batch:
        pinball = pinball * batch["truth_weight"][:, :, None, None]
    truth = pinball.sum() / (count * len(LEVELS))
    diff = q[:, :, list(SHARED)] - batch["teacher"].transpose(1, 2)
    huber = torch.where(diff.abs() < 1.0, 0.5 * diff * diff, diff.abs() - 0.5)
    if "distill_mask" in batch:
        # the teacher saw the truth, so positions whose context holds forecasts learn from the truth alone
        m = m * batch["distill_mask"][:, :, None, None]
        count = (batch["target_mask"] * batch["distill_mask"][:, :, None]).sum().clamp_min(1.0)
    distill = (huber * m).sum() / (count * len(SHARED))
    return truth, distill


def level_loss(level: torch.Tensor, batch: dict) -> torch.Tensor:
    """The absolute distance, in asinh space, between `level [n, SEQ]` and the median of each
    position's target patch over its present points, averaged over the positions whose target
    patch holds at least a quarter of its points; a tile's `weight` scales its positions."""
    present = batch["target_mask"] > 0
    target = torch.where(present, batch["targets"], torch.nan).nanmedian(dim=2).values
    held = (present.sum(dim=2) >= PATCH // 4).to(level.dtype)
    if "weight" in batch:
        held = held * batch["weight"][:, None]
    return ((level[:, :POSITIONS] - torch.nan_to_num(target)).abs() * held).sum() / held.sum().clamp_min(1.0)


def relative_weights(batch: dict, alpha: float, floor: float, scale: float) -> torch.Tensor:
    """`[n]`, a weight per tile for `loss_parts`: for the tiles of `batch["relative"]`,
    `(t / scale + floor) ** -alpha` with `t` the stored teacher forecast's pinball on the truth
    over its nine levels in the tile's own space, scaled so that these tiles' weighted `t` sums
    to their plain `t`; one for every other tile."""
    q = batch["teacher"].transpose(1, 2).float()
    y, m = batch["targets"][:, :, None].float(), batch["target_mask"][:, :, None].float()
    tau = torch.tensor([LEVELS[k] for k in SHARED], device=q.device).reshape(1, 1, -1, 1)
    err = y - q
    points = batch["target_mask"].float().sum((1, 2)).clamp_min(1.0) * len(SHARED)
    t = (torch.maximum(err * tau, err * (tau - 1.0)) * m).sum((1, 2, 3)) / points
    group = batch["relative"]
    w = (t / scale + floor) ** -alpha * group
    w = w * (t * group).sum() / (w * t).sum().clamp_min(1e-12)
    return torch.where(group, w, torch.ones_like(w))


def fixed_cache(cached_pass):
    """`cached_pass` with its cache copied into tensors kept at fixed addresses, one set per
    shape and type, which a CUDA graph of `Student.next_rows` reads in place."""
    kept = {}

    def run(*args, **kwargs):
        full, states = cached_pass(*args, **kwargs)
        key = tuple((s.shape, s.dtype) for s in states)
        if key not in kept:
            kept[key] = [torch.empty_like(s) for s in states]
            for s in kept[key]:
                torch._dynamo.mark_static_address(s)
        for fixed, s in zip(kept[key], states):
            fixed.copy_(s)
        return full, kept[key]

    return run


FIXED: dict = {}


def fixed_copy(name: str, source: torch.Tensor) -> torch.Tensor:
    """`source` copied into a tensor kept at a fixed address, one per name, shape and type, which
    a CUDA graph reads and writes in place."""
    key = (name, source.shape, source.dtype, source.device)
    if key not in FIXED:
        FIXED[key] = torch.empty_like(source)
        torch._dynamo.mark_static_address(FIXED[key])
    FIXED[key].copy_(source)
    return FIXED[key]


@functools.lru_cache(maxsize=None)
def band_picks(device: torch.device) -> torch.Tensor:
    """The places among `LEVELS` of the 0.1, 0.5 and 0.9 levels, on `device`."""
    return torch.tensor([LEVELS.index(0.1), MEDIAN, LEVELS.index(0.9)], device=device)


def unroll_depths(batch: dict, depth: int, generator: torch.Generator, keep: torch.Tensor | None = None) -> torch.Tensor:
    """How many of its last context patches `unroll` replaces per tile: drawn from 0 to `depth`,
    the tiles of a group (`batch["group"]`) sharing their first tile's draw, zero for the tiles of `keep`."""
    inputs = batch["inputs"]
    n, device = inputs.shape[0], inputs.device
    j = torch.randint(0, depth + 1, (n,), generator=generator)
    # a pinned source lets the copy run without waiting for the GPU
    j = (j.pin_memory() if device.type == "cuda" else j).to(device, non_blocking=True)
    group = batch.get("group")
    if group is not None:
        j = j[group]
    if keep is not None:
        j = j * (~keep)
    return j


def guide_input(cached_pass, next_position, batch: dict, cut: torch.Tensor, frozen: torch.Tensor, rows: tuple, patches: int,
                kept: torch.Tensor) -> torch.Tensor:
    """The guide input of `batch` (`guide.channel`) when each tile's context ends with patch `cut
    [n]`: the model's forecast of `patches` patches at the tile's coarser width (`rows`, the
    model's `guide`) from the steps up to that patch, the tile's own and those of `batch["past"]`
    before it. `frozen [n]` marks the tiles whose later positions keep the statistics at `cut`,
    as `mask_tail` leaves them; a tile outside `kept [n]` gets no guide."""
    series = torch.cat([batch["past"], batch["values"]], dim=1) if "past" in batch else batch["values"]
    end = series.shape[1] - SEQ * PATCH + (cut + 1) * PATCH
    k, tags = guide.plan(rows, batch["tags"])
    k = k * kept
    levels, share = guide.paths(cached_pass, next_position, series, end, k, tags, patches, batch["inputs"].dtype)
    return guide.channel(levels, share, k, cut, batch["loc"], batch["scale"], frozen).to(batch["inputs"].dtype)


def fed_back(q: torch.Tensor, loc_p: torch.Tensor, scale_p: torch.Tensor, loc_s: torch.Tensor, scale_s: torch.Tensor,
             limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The patch a forecast feeds back: `q [n, 3, PATCH]`, its 0.1, 0.5 and 0.9 levels in the asinh
    space of `loc_p, scale_p [n]`, as the median `[n, PATCH]` in the space of `loc_s, scale_s` and
    the band `[n, 2 * PATCH]` around it; with `limit`, through double precision and clamped."""
    wide = torch.float64 if limit else q.dtype
    raw = loc_p[:, None, None].to(wide) + scale_p[:, None, None].to(wide) * torch.sinh(q.to(wide))
    a = torch.asinh((raw - loc_s[:, None, None].to(wide)) / scale_s[:, None, None].to(wide))
    if limit:
        a = a.clamp(-limit, limit)
    a = a.to(q.dtype)
    return a[:, 1], torch.cat([a[:, 0] - a[:, 1], a[:, 2] - a[:, 1]], dim=1)


def unroll_step(next_rows):
    """Pass `t` of an unroll as one function, so compiled it replays as one graph: `next_rows`'s
    pass at the position `p` that tiles cut at `cut` have reached, the row each layer keeps
    written into `states` at `p`, and its forecast written into `inputs`, `mask` and `band` at
    `p + 1` for the tiles that replace `t` patches or more of their `j`. `states` and those
    three are tensors at fixed addresses (`fixed_cache`, `fixed_copy`)."""

    def run(inputs, mask, band, tags, t, cut, j, states, group, history, state, loc, scale, picks, limit):
        p = (cut + t - 1).clamp(max=POSITIONS - 2)
        q, kept_rows = next_rows(inputs, mask, band, tags, p, states, group, history, state)
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        median, around = fed_back(q[:, picks].float(), loc[rows, p], scale[rows, p], loc[rows, p + 1], scale[rows, p + 1], limit)
        keep, s = (j < t)[:, None], p + 1
        inputs[rows, s] = torch.where(keep, inputs[rows, s], median.to(inputs.dtype))
        mask[rows, s] = torch.where(keep, mask[rows, s], 1.0)
        band[rows, s] = torch.where(keep, band[rows, s], around.to(band.dtype))
        for cache, row in zip(states, kept_rows):
            cache[rows, p] = row

    return run


def unroll(cached_pass, next_position, batch: dict, depth: int, generator: torch.Generator,
           keep: torch.Tensor | None = None, limit: float = 0.0, j: torch.Tensor | None = None, frozen: bool = False,
           own: bool = False, step=None) -> dict:
    """`batch` with the last `j` context patches of each tile replaced by the model's own
    forecasts, `j` as given or drawn per tile (`unroll_depths`): patch by patch from the context so far,
    without gradients, the median becomes the input and the 0.1 and 0.9 levels its band, in the
    asinh space of the replaced patch's statistics. The tiles of a group (`batch["group"]`, each
    tile's group as the place of its first tile) share that tile's `j`. The first pass runs every position and
    keeps each layer's inputs (`Student.cached_pass`), each later pass runs only the position
    whose patch was just replaced (`Student.next_position`). `distill_mask [n, POSITIONS]` keeps
    the positions whose context is all truth. With `limit`, the forecast goes to raw units and
    back in double precision and the patch fed back stays within `limit` asinh units, so a very
    large knot stays a finite input. A guide input in the batch (`batch["guide"]`) is read at
    every pass beside the history input. Under `frozen`, the positions after the cut keep the
    statistics at the cut: a forecast is fed back as it was made, the targets after the cut
    (from `batch["values"]`) and the batch's `loc` and `scale` there are in that space. Under
    `own`, a position after the cut holds the running statistics of the observed values up to
    the cut and the medians fed back up to its own patch, which the forecaster computes over
    its window; the targets after the cut and the batch's `loc` and `scale` follow. `step`, an
    `unroll_step`, runs each later pass and writes the patch it feeds back."""
    in_graph = step is not None and not own
    n, device = batch["inputs"].shape[0], batch["inputs"].device
    band = torch.zeros(n, SEQ, 2 * PATCH, device=device, dtype=batch["inputs"].dtype)
    if in_graph:
        # `step` writes these three in place
        inputs, mask, band = fixed_copy("inputs", batch["inputs"]), fixed_copy("mask", batch["input_mask"]), fixed_copy("band", band)
    else:
        inputs, mask = batch["inputs"].clone(), batch["input_mask"].clone()
    if j is None:
        j = unroll_depths(batch, depth, generator, keep)
    group = batch.get("group")
    cut = POSITIONS - 1 - j
    history = batch.get("history")
    if history is not None:
        # a position whose patch is a forecast reads no history
        history = history * (torch.arange(SEQ, device=device)[None] <= cut[:, None])[:, :, None]
    if "guide" in batch:
        history = batch["guide"] if history is None else torch.cat([history, batch["guide"]], dim=2)
    rows = torch.arange(n, device=device)
    loc, scale = batch["loc"], batch["scale"].clamp_min(batches.MIN_SCALE)
    if frozen:
        held = torch.minimum(torch.arange(loc.shape[1], device=device)[None], cut[:, None])
        loc, scale = loc.gather(1, held), scale.gather(1, held)
    if own:
        loc, scale = loc.clone(), scale.clone()
        # count, mean and summed squared deviations of the observed values up to the cut
        count = torch.isfinite(batch["values"].reshape(n, SEQ, PATCH)).sum(2).cumsum(1).double()[rows, cut]
        mean = loc[rows, cut].double()
        m2 = scale[rows, cut].double() ** 2 * (count - 1.0).clamp_min(1.0)
    picks = band_picks(device)
    passes = torch.arange(depth + 1, device=device)
    if in_graph:
        # `step` reads these in place, so a replay copies none of them
        read = {name: value if value is None else fixed_copy(name, value)
                for name, value in (("tags", batch["tags"]), ("cut", cut), ("j", j), ("group", group), ("history", history),
                                    ("state", batch.get("state")), ("loc", loc), ("scale", scale), ("picks", picks))}
    with torch.no_grad():
        for t in range(1, depth + 1):
            if t > 1 and in_graph:
                step(inputs, mask, band, read["tags"], passes[t], read["cut"], read["j"], states, read["group"],
                     read["history"], read["state"], read["loc"], read["scale"], read["picks"], limit)
                continue
            p = (cut + t - 1).clamp(max=POSITIONS - 2)
            if t == 1:
                full, states = cached_pass(inputs, mask, batch["tags"], band, 0, batch.get("sibling"), group, history,
                                           state=batch.get("state"))
                q = full[rows, p]
            else:
                q = next_position(inputs, mask, band, batch["tags"], p, states, group, history, state=batch.get("state"))
            q = q[:, picks].float()
            if own:
                wide = torch.float64 if limit else q.dtype
                raw = loc[rows, p, None, None].to(wide) + scale[rows, p, None, None].to(wide) * torch.sinh(q.to(wide))
                x = raw[:, 1].double()
                total, centre, fed = count + PATCH, x.mean(1), j >= t
                joined = m2 + ((x - centre[:, None]) ** 2).sum(1) + count * PATCH / total * (mean - centre) ** 2
                mean = torch.where(fed, mean + (centre - mean) * PATCH / total, mean)
                count, m2 = torch.where(fed, total, count), torch.where(fed, joined, m2)
                spread = torch.sqrt(m2 / (count - 1.0).clamp_min(1.0)).clamp_min(batches.MIN_SCALE)
                loc[rows, p + 1] = torch.where(fed, mean.to(loc.dtype), loc[rows, p + 1])
                scale[rows, p + 1] = torch.where(fed, spread.to(scale.dtype), scale[rows, p + 1])
            median, around = fed_back(q, loc[rows, p], scale[rows, p], loc[rows, p + 1], scale[rows, p + 1], limit)
            keep, s = (j < t)[:, None], p + 1
            inputs[rows, s] = torch.where(keep, inputs[rows, s], median)
            mask[rows, s] = torch.where(keep, mask[rows, s], 1.0)
            band[rows, s] = torch.where(keep, band[rows, s], around)
    if in_graph:
        # the next unroll writes the fixed tensors again
        inputs, mask, band = inputs.clone(), mask.clone(), band.clone()
    positions = torch.arange(POSITIONS, device=device)
    kept = (positions[None] <= cut[:, None]).to(inputs.dtype)
    out = batch | {"inputs": inputs, "input_mask": mask, "band": band, "context_end": cut,
                   "distill_mask": kept * batch["distill_mask"] if "distill_mask" in batch else kept}
    if frozen or own:
        v = batch["values"].reshape(n, SEQ, PATCH)[:, 1:]
        present = torch.isfinite(v)
        placed = torch.asinh((torch.nan_to_num(v) - loc[:, :POSITIONS, None]) / scale[:, :POSITIONS, None])
        targets = torch.where(present, placed, 0.0).to(batch["targets"].dtype)
        out |= {"targets": torch.where((positions[None] > cut[:, None])[:, :, None], targets, batch["targets"]),
                "loc": loc, "scale": scale}
    return out


def backtest_errors(cached_pass, next_position, batch: dict, known: torch.Tensor, span: int, frozen: bool,
                    one_step: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """The backtest half of the error input (`student.backtest_channel`) for tiles whose last
    observed position is `known [n]`: `unroll` replaces the `span` patches before it by a rollout
    from the context `span` patches shorter, which reads the one-patch errors `one_step [n, SEQ,
    2 * PATCH]` up to its own end, and each replaced patch is compared with its truth in the
    space it was fed back in."""
    n, device = known.shape[0], known.device
    cut = known - span
    seen = one_step * (torch.arange(SEQ, device=device)[None] <= cut[:, None])[:, :, None]
    rolled = unroll(cached_pass, next_position, batch | {"sibling": torch.cat([seen, torch.zeros_like(seen)], dim=2)},
                    span, None, None, limit, POSITIONS - 1 - cut, frozen)
    loc = rolled.get("loc", batch["loc"])[:, :POSITIONS, None]
    scale = rolled.get("scale", batch["scale"]).clamp_min(batches.MIN_SCALE)[:, :POSITIONS, None]
    v = batch["values"].reshape(n, SEQ, PATCH)[:, :POSITIONS]
    present = torch.isfinite(v)
    truth = torch.where(present, torch.asinh((torch.nan_to_num(v) - loc) / scale), 0.0)
    return student.backtest_channel(rolled["inputs"][:, :POSITIONS], truth, present, cut, span)


def refine_state(cached_pass, next_position, update, start: torch.Tensor, batch: dict, known: torch.Tensor, rounds: int,
                 depth: int, frozen: bool, limit: float = 0.0, read: str = "position") -> torch.Tensor:
    """The series embedding at each position `[n, SEQ, d_model]` of tiles whose last observed
    position is `known [n]`, from `start [d_model]` in `rounds` rounds. Round `i` ends its prefix
    at patch `(known + 1) * i // (rounds + 1) - 1`, forecasts up to `depth` patches after it by
    `unroll` under the embeddings so far, and `update` (`Student.update`) writes the embedding
    anew from the tile with that forecast beside the truth, both in the space the forecast was
    fed back in. A position reads the embedding of the last round whose forecast ended at or
    before it, `start` before the first, so an embedding is read only where every patch it was
    made from is an input. With `read` "forecast" only the positions from a context end on read
    one: in a round those from its prefix's end, in the result those from `known`; the positions
    before read `start`. Gradients run through the updates; the rollouts are made without."""
    n, device = known.shape[0], known.device
    rows = torch.arange(n, device=device)

    def reading(state: torch.Tensor, cut: torch.Tensor) -> torch.Tensor:
        if read != "forecast":
            return state
        return torch.where((every >= cut[:, None])[:, :, None], state[rows, cut][:, None], start[None, None].to(state.dtype))

    e = start[None].expand(n, -1)
    state = start[None, None].expand(n, SEQ, -1)
    every = torch.arange(SEQ, device=device)[None]
    at = torch.arange(POSITIONS, device=device)[None]
    v = batch["values"].reshape(n, SEQ, PATCH)[:, :POSITIONS]
    present = torch.isfinite(v)
    for i in range(1, rounds + 1):
        cut = ((known + 1) * i // (rounds + 1) - 1).clamp_min(0)
        end = torch.minimum(cut + depth, known)
        # no tile has more patches left than the longest rest
        span = min(depth, int((known - cut).max()))
        read_now = reading(state, cut)
        rolled = unroll(cached_pass, next_position, batch | {"sibling": None, "state": read_now.detach()}, span, None, None, limit,
                        POSITIONS - 1 - cut, frozen)
        loc = rolled.get("loc", batch["loc"])[:, :POSITIONS, None]
        scale = rolled.get("scale", batch["scale"]).clamp_min(batches.MIN_SCALE)[:, :POSITIONS, None]
        truth = torch.where(present, torch.asinh((torch.nan_to_num(v) - loc) / scale), 0.0).to(batch["inputs"].dtype)
        shown = ((at > cut[:, None]) & (at <= end[:, None]))[:, :, None]
        held = (present & shown).to(truth.dtype)
        side = torch.cat([rolled["inputs"][:, :POSITIONS] * held, held], dim=2)
        side = torch.cat([side, torch.zeros_like(side[:, :1])], dim=1)
        inputs = batch["inputs"].clone()
        inputs[:, :POSITIONS] = torch.where(shown, truth, inputs[:, :POSITIONS])
        e = update(inputs, batch["input_mask"], batch["tags"], side, end, read_now, e)
        state = torch.where((every >= end[:, None])[:, :, None], e[:, None], state)
    return reading(state, known)


def mask_tail(batch: dict, j: torch.Tensor) -> dict:
    """`batch` with the last input patches of each tile missing: `j [n]` counts them per tile and
    `cut = POSITIONS - 1 - j` is the last position whose input patch stays. A position after `cut`
    then forecasts its next patch from the context up to `cut` alone, so its target is put in the
    asinh space of the statistics at `cut`, the last the model can know. `distill_mask [n,
    POSITIONS]` keeps the positions up to `cut`, where the teacher saw the same context."""
    inputs, mask = batch["inputs"], batch["input_mask"]
    n, device = inputs.shape[0], inputs.device
    cut = POSITIONS - 1 - j
    rows = torch.arange(n, device=device)
    hidden = torch.arange(SEQ, device=device)[None] > cut[:, None]
    shown = (~hidden)[:, :, None].to(inputs.dtype)
    v = batch["values"].reshape(n, SEQ, PATCH)[:, 1:]
    present = torch.isfinite(v)
    loc = batch["loc"][rows, cut, None, None]
    scale = batch["scale"][rows, cut, None, None].clamp_min(batches.MIN_SCALE)
    frozen = torch.where(present, torch.asinh((torch.nan_to_num(v) - loc) / scale), 0.0).to(inputs.dtype)
    ahead = hidden[:, :POSITIONS]
    kept = (~ahead).to(inputs.dtype)
    return batch | {"inputs": inputs * shown, "input_mask": mask * shown,
                    "context_end": torch.minimum(batch["context_end"], cut) if "context_end" in batch else cut,
                    "targets": torch.where(ahead[:, :, None], frozen, batch["targets"]),
                    "distill_mask": kept * batch["distill_mask"] if "distill_mask" in batch else kept}


def ahead_loss(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`pred`: the knots of the patches after the next, `[n, SEQ, 15, steps]`; `targets` and
    `mask`: `[n, POSITIONS, steps]`. Pinball on the truth over all levels, masked and averaged
    over the present target points."""
    q = pred[:, :POSITIONS]
    y, m = targets[:, :, None], mask[:, :, None]
    tau = torch.tensor(LEVELS, device=pred.device, dtype=pred.dtype).reshape(1, 1, -1, 1)
    err = y - q
    pinball = torch.maximum(err * tau, err * (tau - 1.0)) * m
    return pinball.sum() / (mask.sum().clamp_min(1.0) * len(LEVELS))


class RoutingStats:
    """Items each expert received, per channel and MoE layer, from one evaluation."""

    def __init__(self, experts: int, layers: int):
        self.experts, self.layers = experts, layers
        self.per_channel: dict[str, np.ndarray] = {}

    def add(self, names: list[str], chosen: list[torch.Tensor]) -> None:
        for layer, c in enumerate(chosen):
            ids = c.cpu().numpy().reshape(len(names), -1)
            for name, row in zip(names, ids):
                counts = self.per_channel.setdefault(name, np.zeros((self.layers, self.experts), np.int64))
                counts[layer] += np.bincount(row, minlength=self.experts)

    def mutual_information(self, layer: int) -> float:
        """Between channel and chosen expert, in bits: zero when routing ignores the channel,
        up to the channel entropy when it determines the expert."""
        table = np.stack([c[layer] for c in self.per_channel.values()]).astype(np.float64)
        total = table.sum()
        if total == 0:
            return 0.0
        joint = table / total
        pc, pe = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        nz = joint > 0
        return float((joint[nz] * np.log2(joint[nz] / (pc @ pe)[nz])).sum())

    def format_table(self, layer: int) -> str:
        width = max([len(k) for k in self.per_channel] + [7])
        head = f"| {'channel':<{width}} |" + "".join(f" {'exp ' + str(e):>6} |" for e in range(self.experts))
        lines = [head, f"|{'-' * (width + 2)}|" + "-------:|" * self.experts]
        for name, counts in sorted(self.per_channel.items()):
            total = max(counts[layer].sum(), 1)
            lines.append(f"| {name:<{width}} |" + "".join(f" {100 * c / total:>5.1f}% |" for c in counts[layer]))
        return "\n".join(lines)


def regroup(q: np.ndarray, values: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The model's forecasts `q [n, POSITIONS, levels, PATCH]`, each in the asinh frame of its own position, as
    `[n, teacher_store.POSITIONS, levels, teacher_store.PATCH]` in the store's frames `loc`, `scale [n, teacher_store.POSITIONS]`:
    a store position holds the forecasts of the model positions whose patches make up its patch."""
    r = teacher_store.PATCH // PATCH
    own_loc, own_scale = batches.causal_stats(values)
    span = slice(r - 1, r - 1 + teacher_store.POSITIONS * r)
    # shift and stretch relative to the store's frame, so a level far above its scale keeps its digits
    shift = (own_loc[:, span].astype(np.float64) - np.repeat(loc, r, axis=1)) / np.repeat(scale, r, axis=1)
    stretch = own_scale[:, span].astype(np.float64) / np.repeat(scale, r, axis=1)
    z = shift[:, :, None, None] + stretch[:, :, None, None] * np.sinh(q[:, span].astype(np.float64))
    n, levels = z.shape[0], z.shape[2]
    z = z.reshape(n, teacher_store.POSITIONS, r, levels, PATCH).transpose(0, 1, 3, 2, 4).reshape(n, teacher_store.POSITIONS, levels, teacher_store.PATCH)
    return np.arcsinh(z).astype(np.float32)


def evaluate(model, forward_full, store, index: batches.Index, tags: batches.Tags, tiles: np.ndarray, batch: int, device,
             history: tuple | None = None):
    """Scores `tiles` in batches and aggregates by bucket plus an `all` row: Winkler and
    pinball of the student, the teacher and persistence, and the share of targets outside the
    student's tail levels; for a mixture of experts also the routing statistics. `history` is
    what `make_batch` takes for a model with a history input. A model of finer patches than the
    store's is scored on the store's grid (`regroup`), where each store position joins the
    forecasts of its model positions, the later ones made after reading the earlier ones' steps."""
    rows: dict[str, dict] = {}
    cfg = model.cfg
    stats = RoutingStats(cfg.experts, cfg.layers) if cfg.experts else None
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i : i + batch]
            b = batches.make_batch(store, index, tags, chunk, history=history)
            d = batches.to_device(b, device)
            pred, _, chosen = forward_full(d["inputs"], d["input_mask"], d["tags"], d["families"], history=d.get("history"))
            if stats is not None:
                stats.add([index.channels[c] for c in index.tiles["channel"][chunk]], chosen)
            q = pred[:, :POSITIONS].float().cpu().numpy()
            values = np.asarray(store.values[chunk])
            loc = np.asarray(store.loc[chunk], dtype=np.float64)
            scale = np.maximum(np.asarray(store.scale[chunk], dtype=np.float64), batches.MIN_SCALE)
            if PATCH != teacher_store.PATCH:
                q = regroup(q, values, loc, scale)
            shared = np.ascontiguousarray(q[:, :, list(SHARED)].transpose(0, 2, 1, 3))
            s = reference.score_batch(values, shared, loc, scale)
            t = reference.score_batch(values, np.asarray(store.quantiles[chunk], dtype=np.float32), loc, scale)
            p = reference.score_batch(values, reference.persistence_quantiles(values, loc, scale), loc, scale)
            ys, mask = reference.standardise(values, loc, scale)
            points = np.maximum(mask.sum(axis=(1, 2)), 1)
            tails = np.stack([(((ys < q[:, :, k]) if j < 2 else (ys > q[:, :, k])) & mask).sum(axis=(1, 2)) / points
                              for j, k in enumerate(TAIL_LEVELS)], axis=1)
            for n, tile in enumerate(chunk):
                bucket = index.buckets[index.tiles["bucket"][tile]]
                for name in (bucket, "all"):
                    r = rows.setdefault(name, {"tiles": 0, "student": np.zeros(2), "teacher": np.zeros(2),
                                               "persistence": np.zeros(2), "tails": np.zeros(len(TAIL_LEVELS))})
                    r["tiles"] += 1
                    r["student"] += (s["winkler"][n], s["pinball"][n])
                    r["teacher"] += (t["winkler"][n], t["pinball"][n])
                    r["persistence"] += (p["winkler"][n], p["pinball"][n])
                    r["tails"] += tails[n]
    model.train()
    out = []
    for name in sorted(rows, key=lambda k: (k != "all", k)):
        r = rows[name]
        n = r["tiles"]
        out.append({"bucket": name, "tiles": n, "student": r["student"] / n, "teacher": r["teacher"] / n,
                    "persistence": r["persistence"] / n, "tails": r["tails"] / n})
    return out, stats


def format_table(rows: list[dict], teacher_name: str) -> str:
    head = (f"| {'bucket':<7} | {'tiles':>7} | {'W student':>9} | {'W ' + teacher_name:>9} | {'W persist':>9} | "
            f"{'P student':>9} | {'P ' + teacher_name:>9} | {'P persist':>9} | {'<.01':>6} | {'<.05':>6} | "
            f"{'>.95':>6} | {'>.99':>6} | {'>.995':>6} | {'>.999':>6} |")
    lines = [head, "|---------|--------:|----------:|----------:|----------:|----------:|----------:|----------:|"
                   "-------:|-------:|-------:|-------:|-------:|-------:|"]
    for r in rows:
        s, t, p = r["student"], r["teacher"], r["persistence"]
        tails = " | ".join(f"{v:>6.3f}" for v in r["tails"])
        lines.append(f"| {r['bucket']:<7} | {r['tiles']:>7} | {s[0]:>9.4f} | {t[0]:>9.4f} | {p[0]:>9.4f} | "
                     f"{s[1]:>9.4f} | {t[1]:>9.4f} | {p[1]:>9.4f} | {tails} |")
    return "\n".join(lines)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", required=True, help="teacher store directory, with tiles.index.* beside or above it")
    ap.add_argument("--tags", default="on", choices=["on", "off"], help="the channel tag as an input")
    ap.add_argument("--tag-dropout", type=float, default=0.1)
    ap.add_argument("--tag-groups", default=None, help="a groups.json: channels of one group share one row")
    ap.add_argument("--resolution-tags", default="on", choices=["on", "off"], help="the bin width as an input")
    ap.add_argument("--d-model", type=int, default=256, help="width of the backbone; the embedding and the head scale with it")
    ap.add_argument("--d-ff", type=int, default=688, help="hidden width of the feed-forward block, per expert under MoE")
    ap.add_argument("--layers", type=int, default=4, help="transformer layers of the backbone")
    ap.add_argument("--rope", default="off", choices=["on", "off"], help="rotary positions in every attention, no position embedding")
    ap.add_argument("--qk-norm", default="off", choices=["on", "off"], help="normalised queries and keys with a learned gain per head")
    ap.add_argument("--loops", type=int, default=1, help="times the layers run in sequence with the same weights")
    ap.add_argument("--canon", type=int, default=0,
                    help="positions, the position itself last, each layer mixes into its input and into its feed-forward input")
    ap.add_argument("--canon-held", default="off", choices=["on", "off"],
                    help="with --canon: the mix reads only positions whose input patch holds a value")
    ap.add_argument("--recurrent", type=int, default=0, help="state width of a recurrent layer before the first layer")
    ap.add_argument("--level-head", default="off", choices=["on", "off"],
                    help="a second head forecasts each next patch's level and a gate moves the patch's knots towards it")
    ap.add_argument("--level-weight", type=float, default=0.0,
                    help="weight of the level loss: the level head's level against the target patch's median, "
                         "or without a level head the median knot's own patch level")
    ap.add_argument("--history-scales", default="",
                    help="steps per bin of each scale of the history input, as 32,128,384; each position reads 32 bins per scale")
    ap.add_argument("--history-clip", type=float, default=0.0,
                    help="with --history-scales: the input's values are held within this distance of zero")
    ap.add_argument("--across", type=int, default=0,
                    help="last layers that attend over the tiles of a group at each position instead of over positions")
    ap.add_argument("--group-size", type=int, default=8, help="with --across: the most tiles of one group in a batch")
    ap.add_argument("--group-dropout", type=float, default=0.25, help="with --across: the share of grouped tiles that stay alone")
    ap.add_argument("--io-hidden", type=int, default=-1,
                    help="hidden width of the patch embedding and the head; -1 is four times d_model, 0 makes both linear")
    ap.add_argument("--flow-head", default="off", choices=["on", "off"], help="a flow-matching head beside the quantile head")
    ap.add_argument("--flow-weight", type=float, default=1.0, help="weight of the flow-matching loss")
    ap.add_argument("--flow-samples", type=int, default=128, help="draws per position in the flow head's final evaluation")
    ap.add_argument("--flow-steps", type=int, default=8, help="Euler steps per draw in the flow head's final evaluation")
    ap.add_argument("--horizon-patches", type=int, default=1, help="patches the quantile head forecasts from each position")
    ap.add_argument("--ahead-weight", type=float, default=1.0, help="weight of the pinball on the patches after the next")
    ap.add_argument("--unroll", type=int, default=0,
                    help="band as input: up to this many last context patches replaced by the model's own forecasts")
    ap.add_argument("--unroll-depths", default="",
                    help="with --unroll: the unroll depth of a step is drawn from these, as depth:share,... (3:0.5,8:0.3,30:0.2)")
    ap.add_argument("--frozen-feedback", default="off", choices=["on", "off"],
                    help="with --unroll: the patches fed back and the targets after them are in the asinh space of the last observed patch")
    ap.add_argument("--fed-statistics", default="tile", choices=["tile", "own"],
                    help="with --unroll: the running statistics after the cut are the tile's, or those of the context and the "
                         "medians fed back (own), which the forecaster computes")
    ap.add_argument("--guide", default="off", choices=["on", "off"],
                    help="a forecasting position also reads the model's own forecast at the next coarser trained width (guide.py)")
    ap.add_argument("--guide-patches", type=int, default=6, help="coarse patches forecast per tile for the guide")
    ap.add_argument("--guide-dropout", type=float, default=0.2, help="share of tiles that get no guide")
    ap.add_argument("--guide-past", type=int, default=7, help="stored tiles before each tile that the coarse forecast may read")
    ap.add_argument("--knot-limit", type=float, default=0.0,
                    help="with --unroll: a patch fed back is computed in double precision and held within this many asinh units")
    ap.add_argument("--mask-tail", type=int, default=0,
                    help="single-pass decoding: up to this many last input patches of a tile missing, their positions forecasting further ahead")
    ap.add_argument("--mask-min", type=int, default=0,
                    help="with --mask-tail: a masked tile misses at least this many trailing patches")
    ap.add_argument("--copy-input", default="off", choices=["on", "off"],
                    help="with --mask-tail: positions from the cut on read their forecast patch copied from one detected period back")
    ap.add_argument("--sibling-input", type=int, default=0,
                    help="each tile reads the mean of up to this many other training tiles of its group over its context")
    ap.add_argument("--stream-groups", default="off", choices=["on", "off"],
                    help="with --sibling-input and --stream-share: each generated tile comes with generated siblings")
    ap.add_argument("--diff-attention", default="off", choices=["on", "off"],
                    help="each head reads one attention map minus a learned share of a second")
    ap.add_argument("--error-input", default="off", choices=["on", "off"],
                    help="each position reads the error of the model's own forecast of its patch, from a pass without this input")
    ap.add_argument("--error-dropout", type=float, default=0.5, help="with --error-input: the share of tiles that read no error")
    ap.add_argument("--refine-rounds", type=int, default=0,
                    help="the model keeps a series embedding and writes it anew in this many rounds, each a forecast of "
                         "known context patches from a shorter prefix read beside the truth")
    ap.add_argument("--refine-depth", type=int, default=8, help="with --refine-rounds: the patches a round forecasts")
    ap.add_argument("--refine-read", default="position", choices=["position", "forecast"],
                    help="with --refine-rounds: the series embedding is read at every position after the round that made it, "
                         "or only from the context end on, so that the forecast's loss alone trains the update")
    ap.add_argument("--error-backtest", type=int, default=0,
                    help="with --error-input: the last this many observed positions also read the error of a rollout "
                         "that started that many patches before the context end")
    ap.add_argument("--sibling-dropout", type=float, default=0.0,
                    help="with --sibling-input: the share of tiles that read no other tile")
    ap.add_argument("--mask-share", type=float, default=0.5,
                    help="with --unroll and --mask-tail together: the share of tiles that get the tail mask, the others the unroll")
    ap.add_argument("--ahead-teacher", default=None,
                    help="comma-separated far-label passes over this store's training tiles: a tile of theirs gets "
                         "its tail mask as stored and the teacher's forecast of the hidden patches as targets")
    ap.add_argument("--anchors", default="off", choices=["on", "off"],
                    help="the head adds learned weights of each position's last input value and input-patch median")
    ap.add_argument("--ahead-truth", type=float, default=1.0,
                    help="weight of the truth's pinball at the positions that learn a stored hidden-patch forecast")
    ap.add_argument("--experts", type=int, default=0, help="experts per layer; zero keeps the dense block")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--router-tags", default="off", choices=["on", "off"], help="the channel tag as a router bias")
    ap.add_argument("--router-resolution", default="off", choices=["on", "off"], help="the resolution row as a router bias")
    ap.add_argument("--hard-routing", default="off", choices=["on", "off"], help="the channel family picks the expert")
    ap.add_argument("--balance", type=float, default=0.02, help="weight of the Switch balance loss")
    ap.add_argument("--train-series", type=float, default=1.0, help="fraction of the training series drawn from")
    ap.add_argument("--width-shares", default=None,
                    help="width:share pairs, 10:1,30:1,60:1; absent draws widths by tile count")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr-decay", default="none", choices=["none", "cosine"],
                    help="the learning rate after warm-up: constant, or a half cosine down to --lr-floor times --lr")
    ap.add_argument("--lr-floor", type=float, default=0.0, help="the cosine's end, as a fraction of --lr")
    ap.add_argument("--relative-loss", type=float, default=0.0,
                    help="exponent of the loss weight of stored synthetic tiles by their teacher's pinball; 0 weighs every tile alike")
    ap.add_argument("--relative-floor", type=float, default=0.1,
                    help="added to the teacher's pinball over --relative-scale before the power, which bounds the weight")
    ap.add_argument("--relative-scale", type=float, default=0.1, help="the teacher pinball the weight is measured against")
    ap.add_argument("--synthetic-share", type=float, default=0.0,
                    help="probability that a training draw is a tile of the synthetic environment")
    ap.add_argument("--stream-share", type=float, default=0.0,
                    help="share of every batch generated afresh by synthetic.py and trained on the truth alone")
    ap.add_argument("--stream-family", default="any", choices=["any", "kernel", "components", "sparse", "ramp", "pulse", "any-pulse", "plateau", "any-plateau"],
                    help="the generator family of the stream")
    ap.add_argument("--stream-late", type=float, default=0.0,
                    help="share of the stream's tiles that start late inside a patch, the first patch with values holding 2 to 31")
    ap.add_argument("--empty-context-loss", default="on", choices=["on", "off"],
                    help="off: a position whose context holds no observed point has no loss")
    ap.add_argument("--lambda", dest="lambda_", default="1.0:1.0", help="distillation weight, start:end")
    ap.add_argument("--tail-teacher", default=None, help="a Chronos-2 pass over this store's training tiles")
    ap.add_argument("--tail-lambda", default="1.0:1.0", help="weight of the tail teacher's term, start:end")
    ap.add_argument("--eval-tiles", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-splits", default="heldout_time,heldout_container,heldout_channel")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=None, help="seed of the evaluation sample; defaults to --seed")
    ap.add_argument("--save", default=None, help="model directory to write")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume-every", type=int, default=5000,
                    help="with --save: every this many steps the run's whole state (weights, optimiser, random generators, "
                         "position in the data) is written to resume.pt in the model directory; zero writes none")
    ap.add_argument("--resume", default=None,
                    help="a resume.pt: the run it was written by is continued from its step, under the same arguments")
    ap.add_argument("--load", default=None, help="model directory to start from")
    ap.add_argument("--compile", default="on", choices=["on", "off"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "f32", "bf16", "fp16"],
                    help="the passes' precision; auto is half precision on a card, bf16 where the card computes it natively "
                         "and fp16 elsewhere, and f32 on other devices")
    ap.add_argument("--tf32", default="off", choices=["on", "off"], help="TensorFloat-32 matrix products on cards that have them")
    ap.add_argument("--fused", default="on", choices=["on", "off"], help="AdamW's update as one operation per parameter tensor")
    ap.add_argument("--pin", default="on", choices=["on", "off"], help="batches in page-locked memory")
    ap.add_argument("--unroll-ahead", default="off", choices=["on", "off"],
                    help="a step's batch is built and unrolled in a process of its own while the trainer trains on the batch "
                         "before it; the unroll reads the weights of two steps before")
    ap.add_argument("--builders", type=int, default=0,
                    help="assemble the batches in this many processes, the same batches in the same order; "
                         "for batches without siblings, groups, history, past tiles and the relative loss")
    ap.add_argument("--profile", type=int, default=0,
                    help="after 300 steps, print where this many steps spend their CPU and card time, then stop")
    ap.add_argument("--fast-pass", default="off", choices=["on", "off"],
                    help="the unroll's one-position passes through fastpass.Bundle: the same numbers in fewer kernels; "
                         "on a card with compiled passes, for the models it covers")
    ap.add_argument("--pass-check", type=int, default=0,
                    help="with --fast-pass on: for this many steps the unroll also runs through the model's own pass and "
                         "in single precision; printed are the count of values that differ and each pass's distance to single precision")
    ap.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead"],
                    help="torch.compile's mode; reduce-overhead replays the compiled passes as CUDA graphs")
    ap.add_argument("--device", default="cuda", help="cuda, cpu, or mps on a Mac")
    args = ap.parse_args(argv)
    a, b = args.lambda_.split(":")
    args.lambda_ = (float(a), float(b))
    a, b = args.tail_lambda.split(":")
    args.tail_lambda = (float(a), float(b))
    if args.steps is None and args.minutes is None:
        ap.error("--steps or --minutes is required")
    if args.unroll and args.horizon_patches > 1:
        ap.error("--unroll needs a head of one patch")
    if args.frozen_feedback == "on" and not args.unroll:
        ap.error("--frozen-feedback needs --unroll")
    if args.fed_statistics == "own" and (not args.unroll or args.frozen_feedback == "on" or args.guide == "on"):
        ap.error("--fed-statistics own needs --unroll and runs without --frozen-feedback and --guide")
    if args.unroll_ahead == "on" and (not args.unroll or args.pass_check):
        ap.error("--unroll-ahead needs --unroll and runs without --pass-check")
    if args.unroll and args.experts:
        ap.error("--unroll needs a dense student")
    if args.mask_tail and args.horizon_patches > 1:
        ap.error("--mask-tail needs a head of one patch")
    if args.stream_share and args.tail_teacher:
        ap.error("--stream-share runs without --tail-teacher")
    if not 0 <= args.mask_min <= args.mask_tail:
        ap.error("--mask-min lies between zero and --mask-tail")
    if args.copy_input == "on" and not args.mask_tail:
        ap.error("--copy-input needs --mask-tail")
    if args.ahead_teacher and not (args.mask_tail or args.unroll):
        ap.error("--ahead-teacher needs --mask-tail or --unroll")
    return args


def main(argv=None, link=None):
    args = parse(argv)
    device = torch.device(args.device)
    if args.dtype == "auto":
        native = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        args.dtype = "f32" if device.type != "cuda" else "bf16" if native else "fp16"
    torch.manual_seed(args.seed)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"

    store = teacher_store.Store.open(args.store)
    index = batches.Index(args.store)
    groups = None
    if args.tag_groups:
        with open(args.tag_groups) as f:
            groups = json.load(f)
    tags = batches.Tags(index, groups["map"] if groups else None)
    teacher_name = store.meta.get("model", "teacher").rsplit("-", 1)[-1]
    tagged = args.tags == "on" or args.router_tags == "on"
    cfg = StudentConfig(d_model=args.d_model, d_ff=args.d_ff, layers=args.layers, tags=tags.count() if tagged else 0, backbone_tags=args.tags == "on",
                        experts=args.experts, top_k=args.top_k, router_tags=args.router_tags == "on",
                        router_resolution=args.router_resolution == "on",
                        resolutions=tags.resolution_count() if args.resolution_tags == "on" else 0,
                        resolution_tags=args.resolution_tags == "on", hard_routing=args.hard_routing == "on",
                        flow_head=args.flow_head == "on", horizon_patches=args.horizon_patches,
                        band_input=args.unroll > 0, mask_tail=args.mask_tail, mask_min=args.mask_min,
                        anchors=args.anchors == "on", copy_input=args.copy_input == "on",
                        sibling_input=args.sibling_input, loops=args.loops, io_hidden=args.io_hidden,
                        rope=args.rope == "on", qk_norm=args.qk_norm == "on", canon=args.canon, recurrent=args.recurrent,
                        canon_held=args.canon_held == "on",
                        across=args.across, level_head=args.level_head == "on",
                        history_scales=tuple(int(r) for r in args.history_scales.split(",") if r),
                        history_clip=args.history_clip,
                        guide=guide.rows(tags.widths) if args.guide == "on" else (),
                        frozen_feedback=args.frozen_feedback == "on", error_input=args.error_input == "on",
                        error_backtest=args.error_backtest,
                        refine_rounds=args.refine_rounds, refine_depth=args.refine_depth, refine_read=args.refine_read,
                        diff_attention=args.diff_attention == "on")
    if args.load:
        description, model = student.load(args.load, device)
        if args.sibling_input and not model.cfg.sibling_input:
            # the loaded weights go on unchanged; the sibling input starts from zero weights
            model = student.widen(model, dataclasses.replace(model.cfg, sibling_input=args.sibling_input)).to(device)
        if args.refine_rounds and not model.cfg.refine_rounds:
            # the loaded weights go on unchanged; the series embedding and its input start from zero weights
            model = student.widen(model, dataclasses.replace(model.cfg, refine_rounds=args.refine_rounds,
                                                             refine_depth=args.refine_depth, refine_read=args.refine_read)).to(device)
        if args.error_input == "on" and not model.cfg.error_input:
            # the loaded weights go on unchanged; the error input starts from zero weights
            model = student.widen(model, dataclasses.replace(model.cfg, error_input=True)).to(device)
        cfg = model.cfg
    else:
        model = Student(cfg).to(device)
    if resume:
        model.load_state_dict(resume["model"])
    description = student.describe(cfg, store.meta.get("model", ""), tags.known_channels(index) if cfg.tags else {},
                                   tags.widths if cfg.resolutions else [], groups["groups"] if groups else None)
    print(f"{device}\nstore {args.store} ({store.n} tiles, teacher {store.meta.get('model', '')}), "
          f"student {model.num_params() / 1e6:.2f} M parameters, layers {cfg.layers}, d_ff {cfg.d_ff}, backbone tags "
          f"{'on' if cfg.backbone_tags else 'off'}{f' in {tags.rows} groups' if groups else ''}, experts {cfg.experts} top-{cfg.top_k} router tags "
          f"{'on' if cfg.router_tags else 'off'} hard routing {'on' if cfg.hard_routing else 'off'}"
          f"{' resolution tags on' if cfg.resolution_tags else ''}{' router resolution on' if cfg.router_resolution else ''}"
          f"{f' flow head on (weight {args.flow_weight})' if cfg.flow_head else ''}"
          f"{f' horizon patches {cfg.horizon_patches} (ahead weight {args.ahead_weight})' if cfg.horizon_patches > 1 else ''}"
          f"{f' band input, unroll up to {args.unroll} patches' if cfg.band_input else ''}"
          f"{' fed back in the last observed patch statistics' if cfg.frozen_feedback else ''}"
          f"{' fed back in the statistics of the context and the patches fed back' if args.fed_statistics == 'own' else ''}"
          f"{f' tail mask up to {cfg.mask_tail} patches' if cfg.mask_tail else ''}"
          f"{f' from {cfg.mask_min}' if cfg.mask_min else ''}"
          f"{' anchors on' if cfg.anchors else ''}"
          f"{' copy input on' if cfg.copy_input else ''}"
          f"{f' sibling input of {cfg.sibling_input}' if cfg.sibling_input else ''}"
          f"{f' layers run {cfg.loops} times' if cfg.loops > 1 else ''}"
          f"{' rotary positions' if cfg.rope else ''}{' normalised queries and keys' if cfg.qk_norm else ''}"
          f"{f' mix of {cfg.canon} positions per layer' if cfg.canon else ''}{' from held positions' if cfg.canon_held else ''}"
          f"{f' recurrent first layer of width {cfg.recurrent}' if cfg.recurrent else ''}"
          f"{f' last {cfg.across} layers across tiles' if cfg.across else ''}"
          f"{' level head' if cfg.level_head else ''}{f' level loss weight {args.level_weight}' if args.level_weight else ''}"
          f"{f' history input at {list(cfg.history_scales)} steps a bin' if cfg.history_scales else ''}"
          f"{f' within {cfg.history_clip}' if cfg.history_clip else ''}"
          f"{f' guide from the coarser width {list(cfg.guide)}, {args.guide_patches} coarse patches, dropout {args.guide_dropout}' if cfg.guide else ''}"
          f"{'' if cfg.io_hidden < 0 else f' embedding and head hidden width {cfg.io_hidden}'}"
          f" balance {args.balance}, batch {args.batch}, "
          f"lr {args.lr} warm-up {args.warmup}, lambda {args.lambda_[0]}->{args.lambda_[1]}, compile {args.compile}, "
          f"dtype {args.dtype}", flush=True)

    shares = ({int(w): float(x) for w, x in (pair.split(":") for pair in args.width_shares.split(","))}
              if args.width_shares else None)
    sampler = batches.Sampler(index, args.seed, args.train_series, shares, args.synthetic_share)
    tag_rng = batches.Lcg(args.seed + 1)
    siblings = None
    if cfg.sibling_input:
        sizes = [s["tiles"] for s in store.meta["sources"]] if "sources" in store.meta else [store.n]
        siblings = (batches.Siblings(index, sizes, args.seed), cfg.sibling_input, args.sibling_dropout)
        grouped = np.diff(siblings[0].first)
        print(f"sibling input: {len(grouped)} groups over {int(grouped.sum())} real training tiles, "
              f"{int((grouped > 1).sum())} with more than one tile, median size {int(np.median(grouped))}, "
              f"dropout {args.sibling_dropout}", flush=True)
    groups = None
    if cfg.across:
        sizes = [s["tiles"] for s in store.meta["sources"]] if "sources" in store.meta else [store.n]
        groups = (batches.Siblings(index, sizes, args.seed), args.group_size, args.group_dropout)
        grouped = np.diff(groups[0].first)
        print(f"groups: {len(grouped)} over {int(grouped.sum())} real training tiles, {int((grouped > 1).sum())} with more than one "
              f"tile, up to {args.group_size} tiles of a group in a batch, {args.group_dropout} of the grouped tiles alone", flush=True)
    history = batches.history_for(store, index, cfg.history_scales)
    past = None
    if cfg.guide and args.guide_past:
        sizes = [s["tiles"] for s in store.meta["sources"]] if "sources" in store.meta else [store.n]
        past = (batches.History(index, sizes), args.guide_past)
    if history:
        reach = (history[0].before(index.with_split("train"), 1) >= 0).mean()
        print(f"history input: {100 * reach:.0f} % of the training tiles have a tile before them in the store", flush=True)
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    held = batches.eval_tiles(index, "heldout_time", args.eval_tiles, eval_seed)
    # a finer model is scored on the store's grid, where its later forecast of a store patch has read the earlier one's steps
    regrouped = "" if PATCH == teacher_store.PATCH else (f", each store position scored with the forecasts of its {teacher_store.PATCH // PATCH} "
                                                   f"model positions, the later made after reading the earlier one's steps")
    print(f"train tiles {len(sampler)} from {sampler.series} series (fraction {args.train_series}), "
          f"periodic evaluation on {len(held)} held-out-time tiles{regrouped}", flush=True)
    real = 1.0 - args.synthetic_share
    for width, share, rows in sampler.groups:
        passes = f", {args.steps * args.batch * real * share / len(rows):.2f} passes" if args.steps else ""
        print(f"width {width} s: {real * share:.1%} of the draws over {len(rows)} training tiles{passes}", flush=True)
    if args.synthetic_share > 0.0:
        passes = f", {args.steps * args.batch * args.synthetic_share / len(sampler.synthetic):.2f} passes" if args.steps else ""
        print(f"synthetic: {args.synthetic_share:.1%} of the draws over {len(sampler.synthetic)} training tiles{passes}", flush=True)
    print(f"learning rate {args.lr}, warm-up {args.warmup} steps, decay {args.lr_decay}"
          f"{f' to {args.lr_floor} of it' if args.lr_decay != 'none' else ''}", flush=True)
    tail = None
    if args.tail_teacher:
        tail = TailTeacher(args.tail_teacher, store.n)
        if not os.path.samefile(tail.meta["store"], args.store):
            sys.exit(f"{args.tail_teacher} was made on {tail.meta['store']}")
        covered = sum(int((tail.row[rows] >= 0).sum()) for _, _, rows in sampler.groups)
        print(f"tail teacher {args.tail_teacher}: {tail.meta['model']} at levels {tail.meta['levels']}, "
              f"{tail.meta['positions_per_tile']} positions per tile, on {covered} of {len(sampler)} training tiles, "
              f"weight {args.tail_lambda[0]}->{args.tail_lambda[1]}", flush=True)

    ahead = None
    if args.ahead_teacher:
        ahead = AheadTeacher(args.ahead_teacher.split(","), store.n, args.seed)
        if PATCH != AheadTeacher.STORED and (cfg.mask_tail or not cfg.band_input):
            sys.exit("on the 16-step grid --ahead-teacher is built for the unroll alone")
        for meta in ahead.metas:
            if not os.path.samefile(meta["store"], args.store):
                sys.exit(f"an --ahead-teacher pass was made on {meta['store']}")
        if cfg.mask_tail and ahead.ahead > cfg.mask_tail:
            sys.exit(f"the --ahead-teacher passes hide up to {ahead.ahead} patches, --mask-tail {cfg.mask_tail}")
        print(f"ahead teacher {args.ahead_teacher}: {ahead.metas[0]['model']}, up to {ahead.ahead} hidden patches", flush=True)
        for width, _, rows in sampler.groups:
            print(f"ahead teacher, width {width} s: {int(ahead.held(rows).sum())} of {len(rows)} training tiles", flush=True)
        if len(sampler.synthetic):
            print(f"ahead teacher, synthetic: {int(ahead.held(sampler.synthetic).sum())} of {len(sampler.synthetic)} "
                  f"training tiles", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.01,
                            **({"fused": True} if args.fused == "on" else {"foreach": True}))
    # float16 gradients underflow unscaled; the scaler passes everything through in the other types
    scaler = torch.amp.GradScaler(device.type, enabled=args.dtype == "fp16")
    if resume:
        opt.load_state_dict(resume["optimiser"])
        scaler.load_state_dict(resume.get("scaler", scaler.state_dict()))
    forward_full = torch.compile(model.forward_full, mode=args.compile_mode) if args.compile == "on" else model.forward_full
    hidden_fn = torch.compile(model.hidden, mode=args.compile_mode) if args.compile == "on" else model.hidden
    head_fn = torch.compile(model.head, mode=args.compile_mode) if args.compile == "on" else model.head
    patches_fn = torch.compile(model.head_patches, mode=args.compile_mode) if args.compile == "on" else model.head_patches
    loss_fn = torch.compile(loss_parts, mode=args.compile_mode) if args.compile == "on" else loss_parts
    ahead_fn = torch.compile(ahead_loss, mode=args.compile_mode) if args.compile == "on" else ahead_loss
    cached_fn = torch.compile(model.cached_pass, mode=args.compile_mode) if args.compile == "on" else model.cached_pass
    # the unroll's one-position passes are many small kernels, so on a card each replays as one CUDA graph
    graphs = args.compile == "on" and device.type == "cuda"
    compiled_cached = cached_fn
    if graphs:
        cached_fn = fixed_cache(cached_fn)
    rows_fn = (torch.compile(model.next_rows, mode="reduce-overhead" if graphs else args.compile_mode)
               if args.compile == "on" else model.next_rows)
    next_fn = lambda *a, **k: model.next_position(*a, **k, rows_fn=rows_fn)
    step_fn = (torch.compile(unroll_step(model.next_rows), mode="reduce-overhead" if graphs else args.compile_mode)
               if args.compile == "on" else None)
    half = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype)
    bundle = fast_cached = fast_step = None
    if args.fast_pass == "on" and graphs and cfg.band_input and fastpass.supports(model):
        bundle = fastpass.Bundle(model, half or torch.float32)
        fast_cached = bundle.cache(compiled_cached)
        fast_step = torch.compile(bundle.step(fed_back), mode="reduce-overhead")
    print(f"unroll passes: {'fastpass.Bundle' if bundle is not None else 'the model\'s own'}", flush=True)
    autocast = torch.autocast(device.type, dtype=half) if half else torch.autocast(device.type, enabled=False)
    multi = cfg.horizon_patches > 1
    unroll_rng = torch.Generator().manual_seed(args.seed + 2)
    error_rng = torch.Generator().manual_seed(args.seed + 5)
    depth_rng = torch.Generator().manual_seed(args.seed + 3)
    depths = None
    if args.unroll_depths:
        pairs = [pair.split(":") for pair in args.unroll_depths.split(",")]
        depths = ([int(d) for d, _ in pairs], torch.tensor([float(share) for _, share in pairs]))
        shares = (depths[1] / depths[1].sum()).tolist()
        print("unroll depth per step drawn from " + ", ".join(f"{d} at {s:.2f}" for d, s in zip(depths[0], shares)), flush=True)
    raw = multi or cfg.band_input or cfg.mask_tail > 0 or bool(cfg.guide)
    guide_rng = torch.Generator().manual_seed(args.seed + 4)
    streamed = round(args.stream_share * args.batch)
    # with --unroll-ahead the preparing process builds the batches
    feeding = link is not None or args.unroll_ahead == "off"
    stream_groups = cfg.sibling_input if args.stream_groups == "on" else 0
    data = resume["data"] if resume else None
    stream = (batches.SyntheticStream(args.seed, family=args.stream_family, siblings=stream_groups, state=data and data["stream"],
                                       late=args.stream_late)
              if streamed and feeding else None)
    alone_rng = np.random.default_rng([args.seed, 11])
    generators = {"unroll": unroll_rng, "error": error_rng, "depth": depth_rng, "guide": guide_rng}
    drawers = [pair[0].rng for pair in (siblings, groups) if pair]
    if resume:
        for name, g in generators.items():
            g.set_state(resume["generators"][name])
        torch.set_rng_state(resume["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda"])
        sampler.rng.state, tag_rng.state = data["sampler"], data["tag"]
        alone_rng.bit_generator.state = data["alone"]
        for rng, state in zip(drawers, data["drawers"]):
            rng.bit_generator.state = state
        print(f"resumed {args.resume} at step {resume['step']}", flush=True)
    if streamed:
        print(f"stream: {streamed} of every {args.batch} tiles generated afresh ({args.stream_family}), truth only"
              f"{f', {args.stream_late:.2f} of them starting late inside a patch' if args.stream_late else ''}"
              f"{f', each with {stream_groups} generated siblings' if stream_groups else ''}", flush=True)

    if args.empty_context_loss == "off":
        print("positions whose context holds no observed point have no loss", flush=True)
    relative_rows = None
    if args.relative_loss > 0.0:
        envs = index.meta.get("envs", [])
        relative_rows = (index.tiles["env"] == envs.index(batches.SYNTHETIC) if batches.SYNTHETIC in envs
                         else np.zeros(len(index.tiles), dtype=bool))
        print(f"relative loss: stored synthetic tiles weighted by (teacher pinball / {args.relative_scale} + "
              f"{args.relative_floor}) ** -{args.relative_loss}, their teacher loss mass kept", flush=True)

    def next_batch() -> dict:
        b = batches.make_batch(store, index, tags, sampler.next(args.batch - streamed), (args.tag_dropout, tag_rng), raw=raw,
                               siblings=siblings, groups=groups, history=history, past=past)
        if stream:
            b.setdefault("distill_mask", torch.ones(args.batch - streamed, POSITIONS))
            values, widths, *others = stream.take(streamed)
            b = batches.join(b, batches.synthetic_batch(values, widths, tags, b["teacher"].dtype, (args.tag_dropout, tag_rng),
                                                        raw=raw, sibling=siblings is not None, others=others[0] if others else None,
                                                        alone=(args.sibling_dropout, alone_rng),
                                                        first_place=args.batch - streamed if groups else None,
                                                        history=cfg.history_scales or None,
                                                        past=past[1] if past else None))
        if args.empty_context_loss == "off":
            b = batches.drop_empty_contexts(b)
        if relative_rows is not None:
            # generated tiles carry the tile number -1
            b["relative"] = torch.from_numpy((b["tiles"] >= 0) & relative_rows[np.maximum(b["tiles"], 0)])
        if ahead:
            b = b | ahead.lookup(b["tiles"])
        if tail:
            b = b | tail.lookup(b["tiles"])
        # where the data stands after this batch, for a run resumed at the step that trains on it
        return b | {"resume": {"sampler": sampler.rng.state, "tag": tag_rng.state, "alone": alone_rng.bit_generator.state,
                               "drawers": [rng.bit_generator.state for rng in drawers], "stream": stream.state() if stream else None}}

    make = next_batch
    if args.builders:
        if siblings or groups or history or past or relative_rows is not None or (streamed and stream_groups):
            sys.exit("--builders is built for batches without siblings, groups, history, past tiles and the relative loss")
        print(f"batches assembled in {args.builders} processes", flush=True)
    if args.builders and feeding:
        builders = batches.Builders(args.builders, start_builder,
                                    {"store": args.store, "tag_groups": args.tag_groups, "ahead": args.ahead_teacher,
                                     "tail": args.tail_teacher, "raw": raw, "tag_dropout": args.tag_dropout,
                                     "empty_context_loss": args.empty_context_loss != "off"}, assemble_batch)

        def draw() -> tuple:
            job = draw_batch(sampler, tag_rng, stream, ahead, args.batch, streamed)
            return job, {"resume": {"sampler": sampler.rng.state, "tag": tag_rng.state, "alone": alone_rng.bit_generator.state,
                                    "drawers": [rng.bit_generator.state for rng in drawers],
                                    "stream": stream.state() if stream else None}}

        make = lambda: builders.next(draw)
    prefetch = batches.Prefetcher(make, pin=args.pin == "on" and device.type == "cuda") if feeding else None

    budget_steps = args.steps if args.steps is not None else 10**12
    budget_secs = args.minutes * 60 if args.minutes is not None else float("inf")
    t0 = time.perf_counter() - (resume["elapsed"] if resume else 0.0)
    step, loss_sum, flow_sum, ahead_sum, tail_sum, level_sum, count = resume["step"] if resume else 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    profiler = None
    checks = []

    def read_checks(now: int, wait: bool) -> None:
        """Stops the run at the first step whose loss was not finite, among the steps the card
        has reached; with `wait`, among all steps so far."""
        if wait and device.type == "cuda":
            torch.cuda.synchronize()
        while checks and (wait or checks[0][2] is None or checks[0][2].query()):
            at, finite, _, parts, pred, batch = checks.pop(0)
            if finite:
                continue
            seen = {k: bool(torch.isfinite(batch[k]).all()) for k in ("inputs", "band", "history", "targets") if k in batch}
            largest = {k: float(torch.nan_to_num(batch[k].float().abs(), nan=0.0, posinf=0.0).max()) for k in seen}
            weights = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            raise FloatingPointError(
                f"the loss is not finite at step {at}, read at step {now}: " + ", ".join(f"{k} {v.item():.4g}" for k, v in parts.items())
                + f"; the forecast is finite: {bool(torch.isfinite(pred).all())}; the weights are finite at step {now}: {weights}; "
                f"finite per batch array: {seen}; largest finite magnitude per array: {largest}")

    def prepare(step: int) -> tuple:
        """The batch of `step` on the device as the forward pass reads it, where the data stands after it, and its groups."""
        batch = prefetch.next()
        data = batch.pop("resume")
        batch = batches.to_device(batch, device)
        with autocast:
            # a pinned source lets the copies run without waiting for the GPU
            on_device = lambda t: (t.pin_memory() if device.type == "cuda" else t).to(device, non_blocking=True)
            both = cfg.band_input and cfg.mask_tail > 0
            # with both decoders, each tile trains one: the tail mask or the unroll
            masked = on_device(torch.rand(args.batch, generator=unroll_rng) < args.mask_share) if both else None
            group = batch.get("group")
            if masked is not None and group is not None:
                # the tiles of a group train the same decoder with the same context end
                masked = masked[group]
            if relative_rows is not None:
                batch["weight"] = relative_weights(batch, args.relative_loss, args.relative_floor, args.relative_scale)
            def mask_depths() -> torch.Tensor:
                j = on_device(torch.randint(max(1 if both else 0, cfg.mask_min), cfg.mask_tail + 1, (args.batch,),
                                            generator=unroll_rng))
                if group is not None:
                    j = j[group]
                if ahead:
                    # a tile with a stored forecast is masked as that forecast was made
                    j = torch.where(batch["ahead_hidden"] > 0, batch["ahead_hidden"], j)
                return j * masked if both else j

            j_roll = j_mask = None
            if cfg.band_input:
                depth = args.unroll if depths is None else depths[0][int(torch.multinomial(depths[1], 1, generator=depth_rng))]
            if cfg.error_input:
                if cfg.mask_tail or not cfg.band_input or group is not None:
                    raise ValueError("the error input is built for the unroll alone")
                if cfg.error_backtest and (args.fed_statistics == "own" or cfg.history_scales or cfg.guide):
                    raise ValueError("the backtest errors are built for the tile's statistics, without history input or guide")
            if ahead and cfg.band_input:
                j_roll = unroll_depths(batch, depth, unroll_rng, masked)
                # a tile whose stored forecast hides no more patches than this step unrolls is unrolled as it was made
                fits = (batch["ahead_hidden"] > 0) & (batch["ahead_hidden"] <= depth)
                j_roll = torch.where(fits if masked is None else fits & ~masked, batch["ahead_hidden"], j_roll)
                if group is not None:
                    j_roll = j_roll[group]
            if cfg.guide:
                # both decoders' context ends are drawn first, in the order the two draws are made without a guide
                if j_roll is None:
                    j_roll = unroll_depths(batch, depth, unroll_rng, masked) if cfg.band_input else torch.zeros(args.batch, dtype=torch.int64, device=device)
                j_mask = mask_depths() if cfg.mask_tail else torch.zeros_like(j_roll)
                kept = ~on_device(torch.rand(args.batch, generator=guide_rng) < args.guide_dropout)
                batch["guide"] = guide_input(cached_fn, next_fn, batch, POSITIONS - 1 - j_roll - j_mask,
                                             torch.ones_like(kept) if cfg.frozen_feedback else j_mask > 0, cfg.guide,
                                             args.guide_patches, kept)
            if cfg.refine_rounds:
                if cfg.mask_tail or not cfg.band_input or group is not None or args.fed_statistics == "own" or cfg.history_scales or cfg.guide:
                    raise ValueError("the series embedding is built for the unroll alone, with the tile's statistics")
                if j_roll is None:
                    j_roll = unroll_depths(batch, depth, unroll_rng, masked)
                batch["state"] = refine_state(cached_fn, next_fn, model.update, model.series_start, batch, POSITIONS - 1 - j_roll,
                                              cfg.refine_rounds, cfg.refine_depth, cfg.frozen_feedback, args.knot_limit, cfg.refine_read)
            if cfg.error_input:
                if j_roll is None:
                    j_roll = unroll_depths(batch, depth, unroll_rng, masked)
                with torch.no_grad():
                    draft = forward_full(batch["inputs"], batch["input_mask"], batch["tags"], batch["families"])[0]
                    reads = on_device(torch.rand(args.batch, generator=error_rng) >= args.error_dropout)
                    # a position after the context end holds a forecast, which has no error yet
                    known = POSITIONS - 1 - j_roll
                    batch["sibling"] = student.error_channel(draft, batch["targets"], batch["target_mask"],
                                                             known * reads - (~reads).long())
                    if cfg.error_backtest:
                        back = backtest_errors(cached_fn, next_fn, batch, known, cfg.error_backtest, cfg.frozen_feedback,
                                               batch["sibling"], args.knot_limit)
                        batch["sibling"] = torch.cat([batch["sibling"], back * reads[:, None, None]], dim=2)
            if cfg.band_input:
                if bundle is not None and args.fed_statistics != "own":
                    if step < args.pass_check:
                        j_both = unroll_depths(batch, depth, unroll_rng, masked) if j_roll is None else j_roll
                        own = unroll(cached_fn, next_fn, batch, depth, unroll_rng, masked, args.knot_limit, j_both,
                                     cfg.frozen_feedback, False, step_fn)
                        # the same unroll in single precision, the yardstick both half-precision unrolls are held against
                        with torch.autocast(device.type, enabled=False):
                            exact = unroll(model.cached_pass, model.next_position, batch, depth, unroll_rng, masked, args.knot_limit,
                                           j_both, cfg.frozen_feedback)
                        batch = unroll(fast_cached, next_fn, batch, depth, unroll_rng, masked, args.knot_limit, j_both,
                                       cfg.frozen_feedback, False, fast_step)
                        far = []
                        for other in (own, batch):
                            gap = torch.cat([(other[k].double() - exact[k].double()).abs().nan_to_num().flatten() for k in ("inputs", "band")])
                            far.append((float(gap.mean()), int((gap > 1e-2).sum()), float(gap.max())))
                        print(f"pass yardstick, step {step}: mean distance to single precision, the model's own pass {far[0][0]:.3e}, "
                              f"the leaner pass {far[1][0]:.3e}; values farther than 0.01: {far[0][1]} and {far[1][1]}; "
                              f"largest {far[0][2]:.3e} and {far[1][2]:.3e}", flush=True)
                        differ, largest = 0, 0.0
                        for name in ("inputs", "input_mask", "band"):
                            a, b = own[name].double(), batch[name].double()
                            unequal = (a != b) & ~(a.isnan() & b.isnan())
                            differ += int(unequal.sum())
                            if bool(unequal.any()):
                                largest = max(largest, float((a - b).abs()[unequal].nan_to_num(nan=float("inf")).max()))
                        replaced = int((batch["context_end"] < POSITIONS - 1).sum())
                        print(f"pass check, step {step}: depth {depth}, {replaced} tiles unrolled, "
                              f"{differ} of {sum(own[k].numel() for k in ('inputs', 'input_mask', 'band'))} values differ, "
                              f"largest difference {largest:.3e}", flush=True)
                    else:
                        batch = unroll(fast_cached, next_fn, batch, depth, unroll_rng, masked, args.knot_limit, j_roll,
                                       cfg.frozen_feedback, False, fast_step)
                else:
                    batch = unroll(cached_fn, next_fn, batch, depth, unroll_rng, masked, args.knot_limit, j_roll, cfg.frozen_feedback,
                                   args.fed_statistics == "own", step_fn)
                if ahead:
                    batch = ahead_labels(batch, j_roll, args.ahead_truth, own=True)
            if cfg.mask_tail:
                j = mask_depths() if j_mask is None else j_mask
                batch = mask_tail(batch, j)
                if cfg.copy_input:
                    cut, rows = POSITIONS - 1 - j, torch.arange(args.batch, device=device)
                    # a tile without a tail mask gets no copy
                    lag, strength = periodic.detect_lag(batch["values"], (cut + 1) * PATCH - 1)
                    batch["copy"] = periodic.copy_ahead(batch["values"], lag * (j > 0), strength, cut, batch["loc"][rows, cut],
                                                       batch["scale"][rows, cut].clamp_min(batches.MIN_SCALE))
                if ahead:
                    batch = ahead_labels(batch, j, args.ahead_truth)
            if cfg.history_scales and "context_end" in batch:
                # a position after the tile's true context reads no history
                shown = torch.arange(SEQ, device=device)[None] <= batch["context_end"][:, None]
                batch["history"] = batch["history"] * shown[:, :, None]
            if cfg.guide:
                batch["history"] = batch["guide"] if "history" not in batch else torch.cat([batch["history"], batch["guide"]], dim=2)
            if (cfg.sibling_input or cfg.error_input) and "context_end" in batch:
                # the other series are known as far as the tile's own true context
                known = torch.arange(SEQ, device=device)[None] <= batch["context_end"][:, None]
                batch["sibling"] = batch["sibling"] * known[:, :, None]
        return batch, data, group

    if link is not None:
        def load(weights: torch.Tensor) -> None:
            at = 0
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(weights[at:at + p.numel()].view_as(p))
                    at += p.numel()

        def made(step: int) -> tuple:
            batch, data, _ = prepare(step)
            return batch, data, {name: g.get_state() for name, g in generators.items()}

        link.serve(load, made)
        if args.builders:
            builders.close()
        return
    ahead_run = None
    if args.unroll_ahead == "on":
        ahead_run = batches.Ahead(prepare_main, (sys.argv[1:] if argv is None else list(argv),),
                                  torch.nn.utils.parameters_to_vector(model.parameters()), step)
        print("each step's batch is built and unrolled in a process of its own, from the weights of two steps before", flush=True)
    while True:
        if args.profile and step == 300 and profiler is None:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
            torch.cuda.synchronize()
            profile_start = time.perf_counter()
        if profiler is not None and step == 300 + args.profile:
            torch.cuda.synchronize()
            wall = time.perf_counter() - profile_start
            profiler.__exit__(None, None, None)
            events = profiler.key_averages()
            print(f"profile of {args.profile} steps: {1e3 * wall / args.profile:.2f} ms a step", flush=True)
            print(events.table(sort_by="self_cuda_time_total", row_limit=22, max_name_column_width=44), flush=True)
            print(events.table(sort_by="self_cpu_time_total", row_limit=22, max_name_column_width=44), flush=True)
            return
        elapsed = time.perf_counter() - t0
        # a spent step budget is full progress, so --steps 0 evaluates a loaded model without training
        progress = 1.0 if step >= budget_steps else min(max(step / budget_steps, elapsed / budget_secs), 1.0)
        if progress >= 1.0:
            break
        lam = args.lambda_[0] + (args.lambda_[1] - args.lambda_[0]) * progress
        tail_lam = args.tail_lambda[0] + (args.tail_lambda[1] - args.tail_lambda[0]) * progress
        lr =args.lr if args.warmup == 0 else args.lr * min((step + 1) / args.warmup, 1.0)
        if args.lr_decay == "cosine":
            lr *= args.lr_floor + (1.0 - args.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr
        if ahead_run is not None:
            batch, data, states = ahead_run.next()
            group = batch.get("group")
        else:
            batch, data, group = prepare(step)
        with autocast:
            flow = ahead_term = level_term = torch.zeros((), device=device)
            if cfg.level_head:
                h, balance, _ = hidden_fn(batch["inputs"], batch["input_mask"], batch["tags"], batch["families"], batch.get("band"),
                                          batch.get("copy"), batch.get("sibling"), group, batch.get("history"))
                pred = head_fn(h)
                if args.level_weight:
                    level_term = level_loss(model.level(h), batch)
            elif cfg.flow_head or multi:
                h, balance, _ = hidden_fn(batch["inputs"], batch["input_mask"], batch["tags"], batch["families"], batch.get("band"))
                if multi:
                    full = patches_fn(h)
                    pred = full[..., :PATCH]
                    ys, ms = batches.ahead_targets(batch["values"], batch["loc"], batch["scale"], cfg.horizon_patches)
                    ahead_term = ahead_fn(full[..., PATCH:], ys[..., PATCH:], ms[..., PATCH:])
                else:
                    pred = head_fn(h)
                if cfg.flow_head:
                    flow = model.flow_loss(h, batch["targets"], batch["target_mask"])
            else:
                pred, balance, _ = forward_full(batch["inputs"], batch["input_mask"], batch["tags"], batch["families"],
                                                batch.get("band"), batch.get("copy"), batch.get("sibling"), group, batch.get("history"),
                                                batch.get("state"))
            truth, distill = loss_fn(pred, batch)
            tail_term = tail_loss(pred, batch, tail.levels) if tail else torch.zeros((), device=device)
            if args.level_weight and not cfg.level_head:
                level_term = level_loss(student.patch_level(pred[:, :, MEDIAN]), batch)
        total = (truth + distill * lam + balance * args.balance + flow * args.flow_weight + ahead_term * args.ahead_weight
                 + tail_term * tail_lam + level_term * args.level_weight)
        finite, reached = torch.isfinite(total.detach()), None
        if device.type == "cuda":
            # the flag lands in pinned memory once the card reaches the event, so the step does not wait for the card
            finite = torch.empty((), dtype=torch.bool).pin_memory().copy_(finite, non_blocking=True)
            reached = torch.cuda.Event()
            reached.record()
        checks.append((step, finite, reached,
                       {"truth": truth, "distill": distill, "level": level_term, "flow": flow, "ahead": ahead_term, "tail": tail_term},
                       pred, batch))
        read_checks(step, False)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ahead_run is not None:
            with torch.no_grad():
                ahead_run.publish(step, torch.nn.utils.parameters_to_vector(model.parameters()))
        loss_sum += total.detach()
        flow_sum += flow.detach()
        ahead_sum += ahead_term.detach()
        tail_sum += tail_term.detach()
        level_sum += level_term.detach()
        count += 1
        step += 1
        if step % args.log_every == 0:
            mean = (loss_sum / count).item()
            flow_part = f"  flow {(flow_sum / count).item():.4f}" if cfg.flow_head else ""
            ahead_part = f"  ahead {(ahead_sum / count).item():.4f}" if multi else ""
            tail_part = f"  tail {(tail_sum / count).item():.4f}" if tail else ""
            level_part = f"  level {(level_sum / count).item():.4f}" if args.level_weight else ""
            print(f"step {step:6}  loss {mean:.4f}  lambda {lam:.2f}{flow_part}{ahead_part}{tail_part}{level_part}  "
                  f"{step / max(elapsed, 1e-9):.1f} steps/s  progress {int(progress * 100)}/100", flush=True)
            loss_sum, flow_sum, ahead_sum, tail_sum, level_sum, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        if args.eval_every and step % args.eval_every == 0:
            rows, _ = evaluate(model, forward_full, store, index, tags, held, args.eval_batch, device, history)
            print(f"evaluation at step {step}, {len(held)} held-out-time tiles\n{format_table(rows, teacher_name)}",
                  flush=True)
        if args.save and args.save_every and step % args.save_every == 0:
            read_checks(step, True)
            student.save(model, description, args.save, stem=f"student-{step}")
            print(f"saved snapshot {args.save}/student-{step}.pt", flush=True)
        if args.save and args.resume_every and step % args.resume_every == 0:
            read_checks(step, True)
            os.makedirs(args.save, exist_ok=True)
            torch.save({"step": step, "elapsed": time.perf_counter() - t0, "model": model.state_dict(), "optimiser": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "generators": states if ahead_run is not None else {name: g.get_state() for name, g in generators.items()},
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None, "data": data},
                       os.path.join(args.save, "resume.tmp"))
            os.replace(os.path.join(args.save, "resume.tmp"), os.path.join(args.save, "resume.pt"))
    read_checks(step, True)
    if ahead_run is not None:
        batch = None
        ahead_run.close()
    if args.save:
        student.save(model, description, args.save)
        print(f"saved {args.save}", flush=True)
    print(f"final evaluation after {step} steps in {(time.perf_counter() - t0) / 60:.1f} min{regrouped}", flush=True)
    for split in args.eval_splits.split(","):
        tiles = batches.eval_tiles(index, split, args.eval_tiles, eval_seed)
        if len(tiles) == 0:
            print(f"split {split}: no tiles", flush=True)
            continue
        rows, stats = evaluate(model, forward_full, store, index, tags, tiles, args.eval_batch, device, history)
        if stats is not None:
            mi = ", ".join(f"layer {l} {stats.mutual_information(l):.3f} bits" for l in range(stats.layers))
            print(f"split {split}, routing: mutual information between channel and expert: {mi}\n"
                  f"last layer, share of each channel's items per expert\n{stats.format_table(stats.layers - 1)}", flush=True)
        print(f"split {split}, {len(tiles)} tiles\n{format_table(rows, teacher_name)}", flush=True)
        if cfg.flow_head:
            draws = torch.Generator(device=device).manual_seed(eval_seed)

            def flow_forward(inputs, mask, ids, families, history=None):
                h, balance, chosen = model.hidden(inputs, mask, ids, families, history=history)
                return model.sample_quantiles(h, args.flow_samples, args.flow_steps, draws), balance, chosen

            rows, _ = evaluate(model, flow_forward, store, index, tags, tiles, min(args.eval_batch, 64), device)
            print(f"split {split}, flow head, {args.flow_samples} draws in {args.flow_steps} steps, {len(tiles)} tiles\n"
                  f"{format_table(rows, teacher_name)}", flush=True)
    print("progress 100/100", flush=True)


if __name__ == "__main__":
    main()
