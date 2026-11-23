"""Forecast raw series windows with a saved model, for scoring on public benchmarks and for a
first look at the bands.

    python python/aximoe/forecast.py --model <dir> --in series.f32 --out bands.f32 \\
        [--tag unknown|auto] [--resolution SECONDS] [--horizon N] [--device cpu|cuda]

`<dir>` holds `student.pt` and `model.json` as the trainer writes them; a run from the command
line continues at the model's patch width, and the model's window is 2048 steps less one patch:
2016 steps at 32, 2032 at 16. `--in` holds little-endian float32 windows of `--length` steps
(default: the model's window), NaN where missing; a longer window keeps its last steps, as many
as the model's window holds, a shorter one is padded at the front with missing steps.
`--out` receives `[windows, 15, horizon]` float32 quantiles at the model's levels in the series'
units, and `<out>.json` the layout: the first patch is the direct forecast, later steps are
recursive, each patch's median fed back as context. A window with fewer observed steps than a
patch, or with none in its last patch, is left unanswered: NaN at every level and step, counted
in the layout as `unanswered` (`answerable`). A window with no value below zero has every
quantile clipped at zero (docs/methods/floor.md). `--tag unknown` runs every window on the
`unknown` channel row; `--tag auto` picks each window's channel row by the in-context fit of
its own patches (`tagpick`); `--tags <file>` reads one channel name per window, `unknown` or a
name the model has no row for falling back to the `unknown` row; the layout lists the names
used. `--resolution`, seconds per bin, picks the resolution row when the model has one.
`--widen <file>` multiplies each level's distance from the median by a factor per forecast patch
and level (`widen_band`). Standard output gets one line per window: the
last observed value and the median with its 10 to 90 % band at the first and the last forecast
step.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aximoe import batches, guide, periodic, student, tagpick  # noqa: E402
from aximoe.student import CONTEXT, LEVELS, MEDIAN, PATCH, POSITIONS, SEQ, TAG_BASE  # noqa: E402

# the position whose output forecasts the patch after the context
LAST = POSITIONS - 1
TILE = CONTEXT + PATCH
# a scale within a thousandth of the floor counts as at the floor
FLOORED = np.float32(batches.MIN_SCALE * 1.001)
FRAMES = ["window", "context", "context-at-floor"]


def causal_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Toto 2.0's `PatchedCausalStdScaler` (Apache-2.0, Copyright Datadog, Inc.) read at the
    last step of each patch of `values [n, steps]`: running mean and Bessel-corrected
    deviation of the finite values, accumulated in float64 and rounded to float32."""
    n, steps = values.shape
    patches = steps // PATCH
    total = np.zeros(n)
    count = np.zeros(n)
    m2 = np.zeros(n)
    prev_loc = np.zeros(n)
    locs, scales = np.zeros((n, patches), np.float32), np.zeros((n, patches), np.float32)
    v = values.astype(np.float64)
    for t in range(steps):
        present = np.isfinite(v[:, t])
        x = np.where(present, v[:, t], 0.0)
        total += np.where(present, x, 0.0)
        count += present
        loc = total / np.maximum(count, 1.0)
        m2 += np.where(present, (x - prev_loc) * (x - loc), 0.0)
        prev_loc = loc
        if (t + 1) % PATCH == 0:
            scale = np.sqrt(m2 / np.maximum(np.maximum(count, 1.0) - 1.0, 1.0))
            locs[:, t // PATCH] = loc
            scales[:, t // PATCH] = np.maximum(scale, batches.MIN_SCALE)
    return locs, scales


def band_offsets(inputs: np.ndarray, lo: np.ndarray, hi: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The band input `[n, SEQ, 2 * PATCH]` of tiles whose standardised `inputs` are `[n, SEQ,
    PATCH]`: the raw 0.1 and 0.9 levels `lo`, `hi [n, SEQ * PATCH]` as offsets from each input
    step in its patch's asinh space (`loc`, `scale [n, POSITIONS]`, as `batches.standardise`
    uses them); zero where a level is NaN, as for every observed step."""
    n = inputs.shape[0]
    loc = np.concatenate([loc, loc[:, -1:]], axis=1).astype(np.float32)[:, :, None]
    scale = np.maximum(np.concatenate([scale, scale[:, -1:]], axis=1).astype(np.float32), np.float32(batches.MIN_SCALE))[:, :, None]
    out = []
    for level in (lo, hi):
        v = level.reshape(n, SEQ, PATCH).astype(np.float32)
        present = np.isfinite(v)
        out.append(np.where(present, np.arcsinh((np.nan_to_num(v) - loc) / scale) - inputs, np.float32(0.0)))
    return np.concatenate(out, axis=2)


def sibling_inputs(inputs: np.ndarray, mask: np.ndarray, groups: np.ndarray, k: int, nearest: int = 0,
                   known: int = SEQ) -> np.ndarray:
    """The sibling input `[n, SEQ, 2 * PATCH]` of windows whose standardised `inputs` and `mask`
    are `[n, SEQ, PATCH]`: for each window the mean of the other windows of its group (`groups
    [n]`) at every step, over those that hold the step, and beside it their number as a share of
    `k`, at most one. With `nearest`, the mean is over the `nearest` other windows of the group
    that correlate most with the window over the steps both hold in the first `known` patches."""
    out = np.zeros((len(inputs), SEQ, 2 * PATCH), np.float32)
    order = np.argsort(groups, kind="stable")
    for members in np.split(order, np.flatnonzero(np.diff(groups[order])) + 1):
        held = inputs[members] * mask[members]
        if nearest and len(members) > nearest + 1:
            m = len(members)
            x = held[:, :known].reshape(m, -1).astype(np.float64)
            w = mask[members][:, :known].reshape(m, -1).astype(np.float64)
            both = w @ w.T
            mean = (x @ w.T) / np.maximum(both, 1.0)
            cov = (x @ x.T) / np.maximum(both, 1.0) - mean * mean.T
            var = (x * x) @ w.T / np.maximum(both, 1.0) - mean * mean
            corr = np.where((var > 1e-12) & (var.T > 1e-12) & (both > 1), cov / np.sqrt(np.maximum(var * var.T, 1e-24)), -np.inf)
            np.fill_diagonal(corr, -np.inf)
            pick = np.argsort(-corr, axis=1)[:, :nearest]
            others = mask[members][pick].sum(1)
            out[members, :, :PATCH] = held[pick].sum(1) / np.maximum(others, 1.0)
            out[members, :, PATCH:] = np.minimum(others / k, 1.0)
            continue
        others = mask[members].sum(0)[None] - mask[members]
        out[members, :, :PATCH] = (held.sum(0)[None] - held) / np.maximum(others, 1.0)
        out[members, :, PATCH:] = np.minimum(others / k, 1.0)
    return out


def passes(n: int, batch: int, groups: np.ndarray | None = None) -> list[np.ndarray]:
    """The numbers of `n` windows in passes of at most `batch`; with `groups [n]`, the windows of
    a group share a pass, and a group larger than a pass is cut into passes of its own."""
    if groups is None:
        return [np.arange(c0, min(c0 + batch, n)) for c0 in range(0, n, batch)]
    order = np.argsort(groups, kind="stable")
    out: list[np.ndarray] = []
    held: list[np.ndarray] = []
    for members in np.split(order, np.flatnonzero(np.diff(groups[order])) + 1):
        if held and sum(len(h) for h in held) + len(members) > batch:
            out.append(np.concatenate(held))
            held = []
        if len(members) > batch:
            out.extend(members[c0 : c0 + batch] for c0 in range(0, len(members), batch))
        else:
            held.append(members)
    if held:
        out.append(np.concatenate(held))
    return out


def capped(groups: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """`groups [n]` with every group cut at random into the fewest groups of at most `cap`
    windows, of sizes within one of each other."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(groups), dtype=np.int64)
    count = 0
    for g in np.unique(groups):
        members = rng.permutation(np.flatnonzero(groups == g))
        for part in np.array_split(members, -(-len(members) // cap)):
            out[part] = count
            count += 1
    return out


def coarse_levels(model, series: np.ndarray, tags: np.ndarray, horizon: int, batch: int, device, floor: bool = True):
    """What a model with `guide` reads beside its windows: for every series `[n, steps]` whose
    width has a coarser one, the model's rollout forecast of the series binned at that width
    (`guide.bins` of the whole series, its last `CONTEXT` bins), spread back over the fine steps
    of the horizon rounded up to whole patches: the levels of `guide.PICKS`, `[n, 3, steps]`,
    NaN for a series without a coarser width; the share of its coarse context that is present,
    `[n]`; and whether it has a guide, `[n]`."""
    n = len(series)
    steps = -(-horizon // PATCH) * PATCH
    levels = np.full((n, 3, steps), np.nan, np.float32)
    share = np.zeros(n, np.float32)
    k, coarse_tags = (t.numpy() for t in guide.plan(model.cfg.guide, torch.from_numpy(np.asarray(tags, dtype=np.int64))))
    whole = torch.from_numpy(np.ascontiguousarray(series, dtype=np.float32))
    for width in np.unique(k[k > 0]):
        at = np.flatnonzero(k == width)
        binned = guide.bins(whole[at], torch.full((len(at),), series.shape[1]), torch.full((len(at),), int(width)), CONTEXT).numpy()
        out = forecast(model, binned, coarse_tags[at], -(-steps // int(width)), batch, device, floor, decoder="rollout", coarse=False)
        levels[at] = np.repeat(out[:, list(guide.PICKS)], int(width), axis=2)[:, :, :steps]
        share[at] = np.isfinite(binned).mean(1)
    return levels, share, k > 0


def guide_side(coarse, loc: np.ndarray, scale: np.ndarray, cut: int, frozen: bool, hold: int | None, device) -> torch.Tensor | None:
    """The guide input (`guide.channel`) of windows whose position `cut` forecasts the first patch
    of the horizon: `coarse` is their part of `coarse_levels`, `loc` and `scale [n, positions]`
    their statistics per position, the last column standing for the positions after it. Under
    `frozen`, the positions from `hold` on keep the statistics at `hold`. Nothing without `coarse`."""
    if not coarse:
        return None
    levels, share, has = coarse
    n = len(share)
    pad = POSITIONS - loc.shape[1]
    if pad > 0:
        loc = np.concatenate([loc, np.repeat(loc[:, -1:], pad, axis=1)], axis=1)
        scale = np.concatenate([scale, np.repeat(scale[:, -1:], pad, axis=1)], axis=1)
    whole = lambda v: torch.full((n,), v, dtype=torch.int64)
    out = guide.channel(torch.from_numpy(levels), torch.from_numpy(share), torch.from_numpy(has.astype(np.int64)), whole(cut),
                        torch.from_numpy(np.ascontiguousarray(loc)), torch.from_numpy(np.ascontiguousarray(scale)),
                        torch.full((n,), frozen), None if hold is None else whole(hold))
    return out.to(device)


def next_patch(model, windows: np.ndarray, tags: np.ndarray, device, head: str = "quantile",
               samples: int = 256, steps: int = 16, draws=None, band=None, sibling=None,
               group: np.ndarray | None = None, long: np.ndarray | None = None, fed: int = 0,
               coarse=None, backtest: np.ndarray | None = None, state=None,
               frame: tuple[np.ndarray, np.ndarray] | None = None, at_floor: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The quantiles of the patch after each window `[n, CONTEXT]` under its packed tag id,
    `[n, 15, PATCH]` in asinh space, with the location and scale that map them back to the
    series' units; the flow head gives the levels' quantiles of `samples` draws in `steps`
    Euler steps from the generator `draws`; a quantile head of several patches gives all of
    them, `[n, 15, horizon_patches * PATCH]`. Under `band_input`, `band` holds the raw 0.1 and
    0.9 levels of each context step `[n, CONTEXT]` each, NaN for observed steps. Under
    `sibling_input`, `sibling` is the windows' sibling input `[n, SEQ, 2 * PATCH]`. Under
    `across`, `group [n]` names each window's group. Under `history_scales`, `long [n, steps]`
    is each series up to the window's end, the window its last `CONTEXT` steps, and the history
    input is read from it; the last `fed` patches of the window are forecasts and read none.
    Under `guide`, `coarse` is the windows' part of `coarse_levels`; the last position and the
    `fed` before it hold the coarse forecast of the patch each forecasts. Under
    `frozen_feedback`, the last `fed` patches and the forecast are in the asinh space of the
    patch before them, the last observed one, or in `frame`, a location and scale per window;
    with `at_floor`, `frame` holds for the windows whose own scale there is at the floor.
    Under `error_backtest`, `backtest` is the windows'
    part of `backtest_input`, moved to the windows' positions. Under `refine_rounds`, `state` is
    the windows' part of `series_state`."""
    n = windows.shape[0]
    loc, scale = causal_stats(windows)
    if model.cfg.frozen_feedback and fed:
        cut = LAST - min(fed, LAST)
        held_loc, held_scale = loc[:, cut].copy(), scale[:, cut].copy()
        if frame is not None:
            swap = held_scale <= FLOORED if at_floor else np.ones(n, bool)
            held_loc[swap], held_scale[swap] = frame[0][swap], frame[1][swap]
        # past LAST fed patches the window holds forecasts alone
        start = cut + 1 if fed <= LAST else 0
        loc[:, start:], scale[:, start:] = held_loc[:, None], held_scale[:, None]
    history = None
    if model.cfg.history_scales:
        channel = batches.history_channel(windows if long is None else long, POSITIONS, loc, scale, model.cfg.history_scales)
        channel[:, max(POSITIONS - fed, 0) :] = 0.0
        history = torch.from_numpy(channel).to(device)
    if model.cfg.guide and coarse:
        side = guide_side(coarse, loc, scale, LAST - fed, False, None, device)
        history = side if history is None else torch.cat([history, side], dim=2)
    g = None if group is None else torch.from_numpy(np.asarray(group, dtype=np.int64)).to(device)
    tiles = np.concatenate([windows, np.full((n, PATCH), np.nan, np.float32)], axis=1)
    inputs, mask, targets, target_mask = batches.standardise(tiles, loc, scale)
    ids = torch.from_numpy(np.asarray(tags, dtype=np.int64)).to(device)
    with torch.no_grad():
        x, m = torch.from_numpy(inputs).to(device), torch.from_numpy(mask).to(device)
        if model.cfg.band_input:
            pad = np.full((n, PATCH), np.nan, np.float32)
            lo, hi = ((np.concatenate([b, pad], axis=1) for b in band) if band is not None else (tiles * np.nan, tiles * np.nan))
            b = torch.from_numpy(band_offsets(inputs, lo, hi, loc, scale)).to(device)
            side = None if sibling is None else torch.from_numpy(sibling).to(device)
            if model.cfg.error_input:
                # the errors of a pass without the input, over the observed patches
                draft = model(x, m, ids, b, None, None, g, history)
                side = student.error_channel(draft, torch.from_numpy(targets).to(device), torch.from_numpy(target_mask).to(device),
                                             torch.full((n,), LAST - min(fed, LAST), device=device))
                if model.cfg.error_backtest:
                    back = torch.zeros_like(side) if backtest is None else torch.from_numpy(backtest).to(device)
                    side = torch.cat([side, back], dim=2)
            if state is not None:
                # a position reads the embedding of the patch it holds, `fed` positions later in the window `state` was made from
                state = state[:, (torch.arange(SEQ, device=device) + fed).clamp(max=LAST)]
            q = model(x, m, ids, b, None, side, g, history, state)[:, LAST]
        elif head == "flow":
            h = model.hidden(x, m, ids)[0][:, LAST : LAST + 1]
            q = model.sample_quantiles(h, samples, steps, draws)[:, 0]
        elif model.cfg.horizon_patches > 1:
            q = model.head_patches(model.hidden(x, m, ids)[0])[:, LAST]
        else:
            q = model(x, m, ids, group=g, history=history)[:, LAST]
    return q.float().cpu().numpy(), loc[:, LAST], scale[:, LAST]


def backtest_input(model, windows: np.ndarray, tags: np.ndarray, batch: int, device) -> np.ndarray:
    """The backtest half of the error input `[n, SEQ, 2 * PATCH]` of windows `[n, CONTEXT]` whose
    steps are all observed ones, as the trainer builds it (`train.backtest_errors`): the last
    `error_backtest` patches against a rollout from the context that many patches shorter."""
    from aximoe import train
    span, n = model.cfg.error_backtest, windows.shape[0]
    out = np.zeros((n, SEQ, 2 * PATCH), np.float32)
    for c0 in range(0, n, batch):
        w = windows[c0 : c0 + batch]
        loc, scale = causal_stats(w)
        tiles = np.concatenate([w, np.full((len(w), PATCH), np.nan, np.float32)], axis=1)
        inputs, mask, targets, target_mask = (torch.from_numpy(a).to(device) for a in batches.standardise(tiles, loc, scale))
        ids = torch.from_numpy(np.asarray(tags[c0 : c0 + batch], dtype=np.int64)).to(device)
        known = torch.full((len(w),), LAST, device=device)
        with torch.no_grad():
            one_step = student.error_channel(model(inputs, mask, ids), targets, target_mask, known)
            tile = {"inputs": inputs, "input_mask": mask, "targets": targets, "tags": ids, "values": torch.from_numpy(tiles).to(device),
                    "loc": torch.from_numpy(loc.astype(np.float32)).to(device), "scale": torch.from_numpy(scale.astype(np.float32)).to(device)}
            back = train.backtest_errors(model.cached_pass, model.next_position, tile, known, span, model.cfg.frozen_feedback, one_step)
        out[c0 : c0 + batch] = back.float().cpu().numpy()
    return out


def series_state(model, windows: np.ndarray, tags: np.ndarray, batch: int, device) -> torch.Tensor:
    """The series embedding at each position `[n, SEQ, d_model]` of windows `[n, CONTEXT]` of
    observed steps, by the trainer's rounds (`train.refine_state`) over the whole window."""
    from aximoe import train
    out = []
    for c0 in range(0, windows.shape[0], batch):
        w = windows[c0 : c0 + batch]
        loc, scale = causal_stats(w)
        tiles = np.concatenate([w, np.full((len(w), PATCH), np.nan, np.float32)], axis=1)
        inputs, mask, targets, _ = (torch.from_numpy(a).to(device) for a in batches.standardise(tiles, loc, scale))
        tile = {"inputs": inputs, "input_mask": mask, "targets": targets, "values": torch.from_numpy(tiles).to(device),
                "tags": torch.from_numpy(np.asarray(tags[c0 : c0 + batch], dtype=np.int64)).to(device),
                "loc": torch.from_numpy(loc.astype(np.float32)).to(device), "scale": torch.from_numpy(scale.astype(np.float32)).to(device)}
        with torch.no_grad():
            out.append(train.refine_state(model.cached_pass, model.next_position, model.update, model.series_start, tile,
                                          torch.full((len(w),), LAST, device=device), model.cfg.refine_rounds,
                                          model.cfg.refine_depth, model.cfg.frozen_feedback, read=model.cfg.refine_read))
    return torch.cat(out)


def ahead_tiles(windows: np.ndarray, patches: int):
    """The tiles a pass that forecasts `patches` patches reads: the windows' last `SEQ - patches`
    patches at the first positions with missing patches after them, their standardised inputs and
    mask, the statistics of the context `[n, context patches]` and that number of patches."""
    n = windows.shape[0]
    c = min(SEQ - patches, POSITIONS)
    context = windows[:, windows.shape[1] - c * PATCH :]
    loc, scale = causal_stats(context)
    pad = POSITIONS - c
    loc_all = np.concatenate([loc, np.repeat(loc[:, -1:], pad, axis=1)], axis=1)
    scale_all = np.concatenate([scale, np.repeat(scale[:, -1:], pad, axis=1)], axis=1)
    tiles = np.concatenate([context, np.full((n, (SEQ - c) * PATCH), np.nan, np.float32)], axis=1)
    inputs, mask, _, _ = batches.standardise(tiles, loc_all, scale_all)
    return tiles, inputs, mask, loc, scale, c


def ahead_patches(model, windows: np.ndarray, tags: np.ndarray, device, patches: int,
                  sibling: np.ndarray | None = None, group: np.ndarray | None = None,
                  long: np.ndarray | None = None, fed: int = 0, coarse=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The quantiles of the `patches` patches after each window in one pass, for a model trained
    with `mask_tail`: the window's last `SEQ - patches` patches sit at the first positions with
    missing patches after them, as in training, and the position of the last context patch and
    those after it each forecast their next patch. `[n, 15, patches * PATCH]` in the asinh space
    of the location and scale at the last context patch, which are returned with it. `sibling`
    is the tiles' sibling input `[n, SEQ, 2 * PATCH]` under `sibling_input`, `group [n]` each
    window's group under `across`. Under `history_scales`, `long [n, steps]` is each series up
    to the window's end and the history input of the context patches is read from it; the last
    `fed` context patches are forecasts and read none. Under `guide`, `coarse` is the windows'
    part of `coarse_levels`, and every position from the `fed` last context patches on holds
    the coarse forecast of the patch it forecasts."""
    n = windows.shape[0]
    tiles, inputs, mask, loc, scale, c = ahead_tiles(windows, patches)
    history = None
    if model.cfg.history_scales:
        channel = batches.history_channel(windows if long is None else long, c, loc, scale, model.cfg.history_scales)
        channel[:, max(c - fed, 0) : c] = 0.0
        history = torch.from_numpy(channel).to(device)
    if model.cfg.guide and coarse:
        side = guide_side(coarse, loc, scale, c - 1 - fed, True, c - 1, device)
        history = side if history is None else torch.cat([history, side], dim=2)
    ids = torch.from_numpy(np.asarray(tags, dtype=np.int64)).to(device)
    with torch.no_grad():
        copy = None
        if model.cfg.copy_input:
            raw, cut = torch.from_numpy(tiles).to(device), torch.full((n,), c - 1, dtype=torch.int64, device=device)
            copy = periodic.copy_ahead(raw, *periodic.detect_lag(raw, (cut + 1) * PATCH - 1), cut,
                                       torch.from_numpy(loc[:, c - 1]).to(device), torch.from_numpy(scale[:, c - 1]).to(device))
        q = model(torch.from_numpy(inputs).to(device), torch.from_numpy(mask).to(device), ids, None, copy,
                  None if sibling is None else torch.from_numpy(sibling).to(device),
                  None if group is None else torch.from_numpy(np.asarray(group, dtype=np.int64)).to(device),
                  history)[:, c - 1 : c - 1 + patches]
    q = q.float().permute(0, 2, 1, 3).reshape(n, len(LEVELS), patches * PATCH)
    return q.cpu().numpy(), loc[:, c - 1], scale[:, c - 1]


def forecast_ahead(model, windows: np.ndarray, tags: np.ndarray, horizon: int, batch: int, device,
                   eligible: np.ndarray, block: int = 0, groups: np.ndarray | None = None,
                   past: np.ndarray | None = None, coarse=None) -> np.ndarray:
    """`forecast` for a model trained with `mask_tail`: `block` patches per pass, by default as
    many as the horizon needs up to the `mask_tail + 1` the model was trained for and at least
    the `mask_min + 1` it was trained from; a horizon longer than one pass feeds each pass's
    medians back as context. With `groups [n]` under `sibling_input`, each window reads the
    other windows of its group over the first pass's context; under `across`, in every pass.
    `past [n, steps]` is each series before its window, for the history input; `coarse`,
    the windows' `coarse_levels`, for the guide."""
    n = windows.shape[0]
    limit = model.cfg.mask_tail + 1
    per_pass = min(block, limit) if block else limit
    least = min(model.cfg.mask_min + 1, per_pass) if model.cfg.mask_min else 1
    context = windows.astype(np.float32)
    long = None
    if model.cfg.history_scales:
        long = context if past is None else np.concatenate([past.astype(np.float32), context], axis=1)
    out = np.zeros((n, len(LEVELS), horizon), np.float32)
    done = 0
    while done < horizon:
        patches = max(min(per_pass, -(-(horizon - done) // PATCH)), least)
        raws = np.zeros((n, len(LEVELS), patches * PATCH), np.float32)
        sibling = None
        if model.cfg.sibling_input and groups is not None and done == 0:
            _, inputs, mask, _, _, _ = ahead_tiles(context, patches)
            sibling = sibling_inputs(inputs, mask, groups, model.cfg.sibling_input)
        together = groups if model.cfg.across else None
        for at in passes(n, batch, together):
            q, loc, scale = ahead_patches(model, context[at], tags[at], device, patches,
                                          None if sibling is None else sibling[at], None if together is None else together[at],
                                          None if long is None else long[at], done // PATCH,
                                          tuple(c[at] for c in coarse) if coarse else None)
            raw = (loc[:, None, None].astype(np.float64) + scale[:, None, None].astype(np.float64) * np.sinh(q.astype(np.float64))).astype(np.float32)
            raws[at] = floor_at_zero(raw, eligible[at])
        steps = min(patches * PATCH, horizon - done)
        out[:, :, done : done + steps] = raws[:, :, :steps]
        context = np.concatenate([context[:, patches * PATCH :], raws[:, MEDIAN]], axis=1)
        if long is not None:
            long = np.concatenate([long, raws[:, MEDIAN]], axis=1)
        done += patches * PATCH
    return out


def pick_tags(model, windows: np.ndarray, description: dict, resolution: int | None, batch: int,
              device) -> tuple[np.ndarray, list[str]]:
    """One packed tag id per window `[n, CONTEXT]`: the channel row, `unknown` among the
    candidates, whose in-context forecasts of the window's own patches have the lowest pinball
    loss, at the resolution row of `resolution`; and the picked channel names."""
    cand = tagpick.candidates(description)
    rows = [row for _, row in cand]
    res = student.tag_id(description, None, resolution) // TAG_BASE - 1
    chosen = np.zeros(windows.shape[0], dtype=np.int64)
    for c0 in range(0, windows.shape[0], batch):
        w = windows[c0 : c0 + batch]
        n = w.shape[0]
        loc, scale = causal_stats(w)
        tiles = np.concatenate([w, np.full((n, PATCH), np.nan, np.float32)], axis=1)
        inputs, mask, targets, target_mask = (torch.from_numpy(a).to(device) for a in batches.standardise(tiles, loc, scale))
        pin = torch.zeros(len(rows), n, POSITIONS, device=device)
        with torch.no_grad():
            for c, row in enumerate(rows):
                ids = torch.full((n,), row + (res + 1) * TAG_BASE, dtype=torch.int64, device=device)
                pin[c] = tagpick.pinball_by_position(model(inputs, mask, ids), targets, target_mask)
        chosen[c0 : c0 + n] = tagpick.picks(pin.cpu().numpy(), target_mask.sum(-1).cpu().numpy(), POSITIONS)
    tags = np.array([rows[k] + (res + 1) * TAG_BASE for k in chosen], dtype=np.int64)
    return tags, [cand[k][0] for k in chosen]


def answerable(windows: np.ndarray) -> np.ndarray:
    """`[n]`: whether each window gets a forecast: it holds at least a patch of observed steps and
    at least one in its last patch, the one the forecast is read from."""
    present = np.isfinite(windows)
    return (present.sum(axis=1) >= PATCH) & present[:, -PATCH:].any(axis=1)


def floor_eligible(windows: np.ndarray) -> np.ndarray:
    """Per window `[n, steps]`, whether none of its finite values is below zero."""
    return ~(np.where(np.isfinite(windows), windows, 0.0) < 0).any(axis=1)


def floor_at_zero(raw: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """`raw` quantiles `[n, ...]` in the series' units, clipped at zero in the eligible windows."""
    return np.where(eligible.reshape((-1,) + (1,) * (raw.ndim - 1)), np.maximum(raw, 0), raw)


def widen_band(out: np.ndarray, windows: np.ndarray, factors: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """`out [n, 15, steps]` with each level's distance from the median multiplied by `factors
    [patches, 15]`, row `r` for forecast patch `r + 1` and the last row for every later patch, in
    the asinh space of each window's scaler statistics at the forecast origin; the levels are
    sorted after and the floor applied again."""
    loc, scale = causal_stats(windows)
    loc = loc[:, LAST, None, None].astype(np.float64)
    scale = scale[:, LAST, None, None].astype(np.float64)
    rows = np.minimum(np.arange(out.shape[2]) // PATCH, len(factors) - 1)
    f = np.asarray(factors, np.float64)[rows].T[None]
    a = np.arcsinh((out.astype(np.float64) - loc) / scale)
    mid = a[:, MEDIAN : MEDIAN + 1]
    wide = np.sort(mid + f * (a - mid), axis=1)
    return floor_at_zero((loc + scale * np.sinh(wide)).astype(np.float32), eligible)


def draw_from_knots(knots: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One value per step of `knots [n, 15, steps]` at the probability `u [n]`, linear between
    the two levels around `u`; outside the levels the end knots hold."""
    levels = np.asarray(LEVELS, np.float64)
    j = np.clip(np.searchsorted(levels, u), 1, len(levels) - 1)
    lo = j - 1
    w = np.clip((u - levels[lo]) / (levels[j] - levels[lo]), 0.0, 1.0).astype(np.float32)
    rows = np.arange(knots.shape[0])
    return knots[rows, lo] + w[:, None] * (knots[rows, j] - knots[rows, lo])


def mixture_quantiles(knots: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """The levels' quantiles of the equal-weight mixture of the quantile functions `knots [rows,
    paths, 15]`, each linear between its knots and flat beyond its end knots: the mixture's
    distribution function is the mean of the paths', evaluated at every knot and inverted
    linearly."""
    rows, paths, k = knots.shape
    grid, _ = knots.reshape(rows, paths * k).sort(dim=1)
    points = grid.shape[1]
    g = grid[:, None, :].expand(rows, paths, points)
    idx = torch.searchsorted(knots.contiguous(), g.contiguous(), right=True)
    lo, hi = (idx - 1).clamp(0, k - 1), idx.clamp(0, k - 1)
    k_lo, k_hi = knots.gather(2, lo), knots.gather(2, hi)
    w = ((g - k_lo) / (k_hi - k_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    f = levels[lo] + w * (levels[hi] - levels[lo])
    f = torch.where(idx == 0, levels[0], torch.where(idx >= k, levels[-1], f))
    cdf = f.mean(dim=1)
    t = levels[None, :].expand(rows, k)
    j = torch.searchsorted(cdf.contiguous(), t.contiguous())
    jlo, jhi = (j - 1).clamp(0, points - 1), j.clamp(0, points - 1)
    c_lo, c_hi = cdf.gather(1, jlo), cdf.gather(1, jhi)
    x_lo, x_hi = grid.gather(1, jlo), grid.gather(1, jhi)
    x = x_lo + ((t - c_lo) / (c_hi - c_lo).clamp_min(1e-12)).clamp(0.0, 1.0) * (x_hi - x_lo)
    return torch.where(j == 0, grid[:, :1], torch.where(j >= points, grid[:, -1:], x))


def forecast(model, windows: np.ndarray, tags: np.ndarray, horizon: int, batch: int, device,
             floor: bool = True, paths: int = 0, seed: int = 0, head: str = "quantile", samples: int = 256,
             steps: int = 16, hybrid: bool = False, widen: np.ndarray | None = None, block: int = 0,
             decoder: str = "auto", groups: np.ndarray | None = None, coarse=None, nearest: int = 0,
             frame: str = "window") -> np.ndarray:
    """`horizon` steps of all levels after each window, `[n, 15, horizon]` in the series'
    units, under one packed tag id per window, in forward passes of at most `batch` windows.
    Steps from `PATCH` on are recursive: each patch's median joins the context, the oldest patch
    leaves it. Under `floor`, a window with no value below zero has each patch clipped at zero
    as it is produced, so the value fed back is the clipped one. With `paths` above zero, that
    many paths each feed a draw from their own quantile function back, one probability per path
    and patch from `seed` shared by the patch's steps, and every step's levels are those of the
    equal-weight mixture of the paths' quantile functions; under `hybrid` the median rollout runs
    beside them and each step's levels are shifted so the 0.5 level is its median. Under `head`
    `flow`, each patch's quantiles are those of `samples` draws of the flow head in `steps`
    Euler steps, seeded by `seed`. With `widen`, factors per forecast patch and level, the
    finished band goes through `widen_band`. A quantile head of several patches forecasts them
    all in one pass, and their medians join the context together. Under `band_input`, each
    forecast patch's 0.1 and 0.9 levels join the context beside its median as its band. A model
    trained with `mask_tail` decodes `block` patches per pass instead (`forecast_ahead`); for a
    model trained with both, `decoder` picks: `rollout`, `direct`, or `splice:K`, the rollout's
    first K patches followed by the direct forecast of the rest; `auto` is `splice:9` for such a
    model: the rollout is the better of the two for about nine patches, the direct forecast after.
    Under `sibling_input`, `groups [n]` names each window's group, and a window reads the mean of
    the other windows of its group over the true context (with `nearest`, of that many that
    correlate most with it there); patches fed back read none. Under
    `across`, the windows of a group are forecast in one pass and read each other at every patch.
    A window longer than `CONTEXT` steps is forecast from its last `CONTEXT`; the steps before
    them are the series' past, which a model with `history_scales` reads through its history input.
    A shorter window starts with missing steps.
    A model with `guide` first forecasts each series at its coarser width from the whole window
    (`coarse_levels`) and reads that forecast at every patch; `coarse` hands it a forecast made
    before, and `False` forecasts without one. Under `frozen_feedback`, `frame` names the
    statistics the fed-back patches and each forecast are standardised with: `window`, those of
    the last observed patch of the window as it has moved on; `context`, those of the last
    patch of the window as it was given; or `context-at-floor`, the window's, and the given
    context's where the window's scale is at the floor. A window `answerable` rejects is NaN at every level
    and step, and the others are forecast among themselves."""
    n = windows.shape[0]
    answered = answerable(windows)
    if not answered.all():
        out = np.full((n, len(LEVELS), horizon), np.nan, np.float32)
        if answered.any():
            out[answered] = forecast(model, windows[answered], tags[answered], horizon, batch, device, floor, paths, seed,
                                     head, samples, steps, hybrid, widen, block, decoder,
                                     None if groups is None else groups[answered],
                                     tuple(c[answered] for c in coarse) if coarse else coarse, nearest, frame)
        return out
    if coarse is None:
        coarse = coarse_levels(model, windows, tags, horizon, batch, device, floor) if model.cfg.guide and head == "quantile" and not paths else False
    past = None
    if windows.shape[1] > CONTEXT:
        past, windows = windows[:, : windows.shape[1] - CONTEXT], windows[:, windows.shape[1] - CONTEXT :]
    if windows.shape[1] < CONTEXT:
        # a short window starts with missing steps
        windows = np.concatenate([np.full((n, CONTEXT - windows.shape[1]), np.nan, windows.dtype), windows], axis=1)
    if decoder == "auto":
        decoder = ("splice:9" if model.cfg.band_input else "direct") if model.cfg.mask_tail else "rollout"
    near = horizon if decoder == "rollout" else 0 if decoder == "direct" else min(int(decoder.split(":")[1]) * PATCH, horizon)
    if near < horizon and head == "quantile" and not paths and widen is None:
        eligible = floor_eligible(windows) if floor else np.zeros(n, dtype=bool)
        direct = forecast_ahead(model, windows, tags, horizon, batch, device, eligible, block, groups, past, coarse)
        if near:
            whole = windows if past is None else np.concatenate([past, windows], axis=1)
            direct[:, :, :near] = forecast(model, whole, tags, near, batch, device, floor, decoder="rollout", groups=groups, coarse=coarse,
                                          nearest=nearest, frame=frame)
        return direct
    span = PATCH * (model.cfg.horizon_patches if head == "quantile" else 1)
    if (span > PATCH or model.cfg.band_input) and paths:
        raise ValueError("sample paths need a head of one patch without band input")
    banded = model.cfg.band_input and head == "quantile"
    band = np.full((2, n, windows.shape[1]), np.nan, np.float32) if banded else None
    eligible = floor_eligible(windows) if floor else np.zeros(n, dtype=bool)
    hybrid = hybrid and paths > 0
    p_count = max(paths, 1) + (1 if hybrid else 0)
    contexts = np.repeat(windows.astype(np.float32)[None], p_count, axis=0)
    rng = np.random.default_rng(seed)
    draws = torch.Generator(device=device).manual_seed(seed) if head == "flow" else None
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=device)
    out = np.zeros((n, len(LEVELS), horizon), np.float32)
    given = None
    if frame != "window" and model.cfg.frozen_feedback:
        loc, scale = causal_stats(windows.astype(np.float32))
        given = loc[:, LAST], scale[:, LAST]
    read_groups = model.cfg.sibling_input and groups is not None and paths == 0 and head == "quantile"
    long = None
    if model.cfg.history_scales and paths == 0 and head == "quantile":
        long = windows.astype(np.float32) if past is None else np.concatenate([past, windows], axis=1).astype(np.float32)
    together = groups if model.cfg.across and paths == 0 and head == "quantile" else None
    backtest = None
    if model.cfg.error_backtest and paths == 0 and head == "quantile":
        if model.cfg.history_scales or model.cfg.guide:
            raise ValueError("the backtest errors are built without history input or guide")
        backtest = backtest_input(model, windows.astype(np.float32), tags, batch, device)
    state = None
    if model.cfg.refine_rounds and paths == 0 and head == "quantile":
        if model.cfg.history_scales or model.cfg.guide:
            raise ValueError("the series embedding is built without history input or guide")
        state = series_state(model, windows.astype(np.float32), tags, batch, device)
    for r in range(-(-horizon // span)):
        steps = min(span, horizon - r * span)
        raws = np.zeros((p_count, n, len(LEVELS), span), np.float32)
        sibling = None
        if read_groups:
            loc, scale = causal_stats(contexts[0])
            tiles = np.concatenate([contexts[0], np.full((n, PATCH), np.nan, np.float32)], axis=1)
            inputs, mask, _, _ = batches.standardise(tiles, loc, scale)
            sibling = sibling_inputs(inputs, mask, groups, model.cfg.sibling_input, nearest, max(POSITIONS - r, 1))
            # the patches fed back so far are forecasts, which no other series is known beside
            sibling[:, max(POSITIONS - r, 0) :] = 0.0
        for p in range(p_count):
            if r == 0 and p > 0:
                # the paths share the context before the first patch
                raws[p] = raws[0]
                continue
            for at in passes(n, batch, together):
                q, loc, scale = next_patch(model, contexts[p][at], tags[at], device, head, samples, steps, draws,
                                           band[:, at] if banded else None, None if sibling is None else sibling[at],
                                           None if together is None else together[at], None if long is None else long[at],
                                           r * span // PATCH,
                                           tuple(c[at] for c in coarse) if coarse else None,
                                           None if backtest is None else backtest[at],
                                           None if state is None else state[torch.from_numpy(at).to(device)],
                                           None if given is None else (given[0][at], given[1][at]),
                                           frame == "context-at-floor")
                raw = (loc[:, None, None].astype(np.float64) + scale[:, None, None].astype(np.float64) * np.sinh(q.astype(np.float64))).astype(np.float32)
                raws[p, at] = floor_at_zero(raw, eligible[at])
        if paths == 0:
            out[:, :, r * span : r * span + steps] = raws[0][:, :, :steps]
            contexts[0] = np.concatenate([contexts[0, :, span:], raws[0][:, MEDIAN, :]], axis=1)
            if backtest is not None:
                # the errors move with their patches as the window moves on
                backtest = np.concatenate([backtest[:, span // PATCH :], np.zeros_like(backtest[:, : span // PATCH])], axis=1)
            if long is not None:
                long = np.concatenate([long, raws[0][:, MEDIAN, :]], axis=1)
            if banded:
                levels_fed = raws[0][:, [LEVELS.index(0.1), LEVELS.index(0.9)]].transpose(1, 0, 2)
                band = np.concatenate([band[:, :, span:], levels_fed], axis=2)
            continue
        per_step = np.ascontiguousarray(raws[:paths].transpose(1, 3, 0, 2)).reshape(n * PATCH, paths, len(LEVELS))
        mixed = np.zeros((n * PATCH, len(LEVELS)), np.float32)
        for c0 in range(0, n * PATCH, 4096):
            chunk = torch.from_numpy(per_step[c0 : c0 + 4096]).to(device)
            mixed[c0 : c0 + 4096] = mixture_quantiles(chunk, levels).cpu().numpy()
        mixed = mixed.reshape(n, PATCH, len(LEVELS)).transpose(0, 2, 1)
        if hybrid:
            mixed = mixed + (raws[paths][:, MEDIAN, :] - mixed[:, MEDIAN, :])[:, None, :]
        out[:, :, r * PATCH : r * PATCH + steps] = mixed[:, :, :steps]
        u = rng.random((p_count, n))
        if hybrid:
            u[paths] = 0.5
        for p in range(p_count):
            contexts[p] = np.concatenate([contexts[p, :, PATCH:], draw_from_knots(raws[p], u[p])], axis=1)
    return out if widen is None else widen_band(out, windows, widen, eligible)


def read_windows(path: str, length: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size % length:
        sys.exit(f"{path} does not hold whole windows of {length} float32 values")
    w = values.reshape(-1, length)
    if length >= CONTEXT:
        return np.ascontiguousarray(w[:, length - CONTEXT :])
    padded = np.full((w.shape[0], CONTEXT), np.nan, np.float32)
    padded[:, CONTEXT - length :] = w
    return padded


def compact(v: float) -> str:
    return f"{v:.4f}" if v == 0.0 or 1e-3 <= abs(v) < 1e6 else f"{v:.3e}"


def summary(windows: np.ndarray, out: np.ndarray, levels: list[float], horizon: int) -> str:
    q10, q90 = levels.index(0.1), levels.index(0.9)
    last_step = horizon - 1
    lines = [f"{'row':>5} {'last':>12} {'+1 median':>12} {'10-90 % band at +1':>26} {'+' + str(horizon) + ' median':>12} "
             f"{'10-90 % band at +' + str(horizon):>26}"]
    for w in range(windows.shape[0]):
        finite = windows[w][np.isfinite(windows[w])]
        last = compact(float(finite[-1])) if finite.size else "missing"
        band = lambda step: f"{compact(float(out[w, q10, step]))} .. {compact(float(out[w, q90, step]))}"
        lines.append(f"{w:>5} {last:>12} {compact(float(out[w, MEDIAN, 0])):>12} {band(0):>26} "
                     f"{compact(float(out[w, MEDIAN, last_step])):>12} {band(last_step):>26}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="unknown", choices=["unknown", "auto"],
                    help="the channel row of every window, or one picked per window from its own patches")
    ap.add_argument("--tags", default=None, help="file with one channel name per window, overriding --tag")
    ap.add_argument("--resolution", type=int, default=None, help="seconds per bin of the windows")
    ap.add_argument("--length", type=int, default=CONTEXT)
    ap.add_argument("--horizon", type=int, default=PATCH)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps on a Mac")
    ap.add_argument("--paths", type=int, default=0, help="sample-feedback paths in the recursive steps; 0 feeds the median back")
    ap.add_argument("--seed", type=int, default=0, help="seed of the paths' draws and of the flow head's")
    ap.add_argument("--hybrid", action="store_true", help="with --paths, centre each step's levels on the median rollout's median")
    ap.add_argument("--head", default="quantile", choices=["quantile", "flow"], help="the head whose levels are written")
    ap.add_argument("--samples", type=int, default=256, help="draws per patch for the flow head")
    ap.add_argument("--steps", type=int, default=16, help="Euler steps per draw for the flow head")
    ap.add_argument("--widen", default=None, help="JSON file whose `factors` [patches, 15] widen the band per forecast patch")
    ap.add_argument("--block", type=int, default=0, help="patches per pass for a model trained with a tail mask; 0: as many as it was trained for")
    ap.add_argument("--decoder", default="auto", help="auto, rollout, direct or splice:K (the rollout's first K patches, the direct forecast after)")
    ap.add_argument("--frame", default="window", choices=FRAMES,
                    help="the statistics fed-back patches are standardised with: the moved window's last observed patch, "
                         "the given context's last patch, or the given context's where the moved window's scale is at the floor")
    args = ap.parse_args(argv)
    if argv is None:
        # a run from the command line continues at the model's patch width
        student.rerun_at_width(args.model)
    device = torch.device(args.device)
    description, model = student.load(args.model, device)
    model.eval()
    windows = read_windows(args.input, args.length)
    t0 = time.perf_counter()
    picked = None
    if args.tags:
        with open(args.tags) as f:
            picked = [line.strip() for line in f if line.strip()]
        if len(picked) != windows.shape[0]:
            sys.exit(f"{args.tags} names {len(picked)} windows, the input holds {windows.shape[0]}")
        unknown = student.tag_id(description, None, args.resolution)
        tags = np.array([student.tag_id(description, name, args.resolution) if name in description["channels"] else unknown
                         for name in picked], dtype=np.int64)
    elif args.tag == "auto":
        tags, picked = pick_tags(model, windows, description, args.resolution, args.batch, device)
    else:
        tags = np.full(windows.shape[0], student.tag_id(description, None, args.resolution), dtype=np.int64)
    if args.head == "flow" and model.flow is None:
        sys.exit(f"{args.model} has no flow head")
    widen = None
    if args.widen:
        with open(args.widen) as f:
            widen = np.asarray(json.load(f)["factors"], np.float64)
        if widen.ndim != 2 or widen.shape[1] != len(LEVELS):
            sys.exit(f"{args.widen}: factors must be [patches, {len(LEVELS)}], not {list(widen.shape)}")
    out = forecast(model, windows, tags, args.horizon, args.batch, device, paths=args.paths, seed=args.seed,
                   head=args.head, samples=args.samples, steps=args.steps, hybrid=args.hybrid, widen=widen, block=args.block,
                   decoder=args.decoder, frame=args.frame)
    seconds = time.perf_counter() - t0
    out.astype("<f4").tofile(args.out)
    widths = description["widths"]
    row = widths.index(args.resolution) if args.resolution in widths else None
    span = PATCH * (model.cfg.horizon_patches if args.head == "quantile" else 1)
    rounds = -(-args.horizon // span)
    hybrid = args.hybrid and args.paths > 0
    passes = windows.shape[0] * (1 + (rounds - 1) * (max(args.paths, 1) + (1 if hybrid else 0)))
    recursion = (("each forecast patch's median joins the context and the oldest patch leaves it" if span == PATCH else
                  f"each pass forecasts {span // PATCH} patches, whose medians join the context as the oldest leave it")
                 if args.paths == 0 else
                 f"{args.paths} paths each feed a draw from their own quantile function back; the levels are the mixture of the paths' quantile functions"
                 + ("; the median rollout runs beside them and each step's levels are centred on its median" if hybrid else ""))
    layout = {
        "model": args.model, "teacher": description["teacher"], "parameters": model.num_params(),
        "windows": int(windows.shape[0]), "levels": description["levels"], "horizon": args.horizon,
        "layout": "float32 little-endian [window, level, step], series units", "context": CONTEXT, "patch": PATCH,
        "input_length": args.length, "direct_steps": min(span, args.horizon), "recursive_from_step": span,
        "recursion": recursion, "paths": args.paths, "hybrid": hybrid, "draw_seed": args.seed, "head": args.head,
        "flow_samples": args.samples if args.head == "flow" else None, "flow_steps": args.steps if args.head == "flow" else None,
        "floored_windows": int(floor_eligible(windows).sum()), "unanswered": int((~answerable(windows)).sum()),
        "widen": args.widen, "frame": args.frame,
        "channel": "file" if args.tags else args.tag,
        "picked": dict(collections.Counter(picked).most_common()) if picked else None,
        "resolution_seconds": args.resolution,
        "resolution_row": "unknown" if row is None else str(row), "device": args.device, "batch": args.batch,
        "seconds": seconds, "window_patches_per_second": passes / max(seconds, 1e-9),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(layout, f, indent=2)
    print(summary(windows, out, description["levels"], args.horizon))
    print(f"aximoe-forecast: {windows.shape[0]} windows, horizon {args.horizon}, {passes} window-patches in "
          f"{seconds:.2f} s ({passes / max(seconds, 1e-9):.0f}/s) on {args.device}", file=sys.stderr)


if __name__ == "__main__":
    main()
