"""The coarse guide: the model's own forecast of a series at a coarser bin width, handed to the
positions that forecast its fine patches. Bins of `k` steps show a level and a cycle after one
or two coarse patches that a fine forecast reaches only after many patches fed back; a
position that forecasts reads the coarse forecast over the steps of its patch beside its own
context and learns how far to follow it.

`StudentConfig.guide` holds, per resolution row, the steps per coarse bin and the resolution row
of the coarse width, `(0, 0)` for a width without a coarser one (`rows`). The trainer forecasts
the coarse series of every tile on the device (`paths`); the forecaster forecasts it with
`forecast.forecast`; both hand the levels to `channel`, which makes the model's input.
"""
from __future__ import annotations

import torch

from aximoe.batches import MIN_SCALE
from aximoe.student import GUIDE, LEVELS, MEDIAN, PATCH, POSITIONS, SEQ, TAG_BASE

# seconds per bin of a width -> (steps per coarse bin, seconds per coarse bin)
PLAN = {10: (6, 60), 30: (10, 300), 60: (5, 300), 300: (6, 1800), 900: (4, 3600), 1800: (2, 3600)}
# the levels a guide carries: 0.1, the median, 0.9
PICKS = (LEVELS.index(0.1), MEDIAN, LEVELS.index(0.9))
# a guide value stays within this distance of zero in asinh space
LIMIT = 12.0


def rows(widths: list[int]) -> tuple:
    """`StudentConfig.guide` for resolution rows of `widths` seconds per bin."""
    out = []
    for w in widths:
        k, coarse = PLAN.get(int(w), (0, 0))
        out.append((k, widths.index(coarse)) if k and coarse in widths else (0, 0))
    return tuple(out)


def plan(guide: tuple, tags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per packed tag: the steps per coarse bin, zero for a tile without a coarser width (its
    resolution row unknown, or a width without one), and the packed tag of its coarse pass:
    the same channel row at the coarse width's resolution row."""
    table = torch.tensor([list(r) for r in guide] + [[0, 0]], dtype=torch.int64, device=tags.device)
    upper = tags // TAG_BASE
    row = torch.where((upper >= 1) & (upper <= len(guide)), upper - 1, torch.full_like(upper, len(guide)))
    k, coarse = table[row, 0], table[row, 1]
    return k, torch.where(k > 0, tags % TAG_BASE + (coarse + 1) * TAG_BASE, tags)


def bins(series: torch.Tensor, end: torch.Tensor, k: torch.Tensor, steps: int) -> torch.Tensor:
    """The last `steps` bins of `k [n]` steps each before step `end [n]` of `series [n, length]`
    (NaN where missing): the mean of a bin's present steps, NaN for a bin without one, for a
    bin that starts before the series and for every bin of a row with `k` zero."""
    n, length = series.shape
    device = series.device
    widest = max(int(k.max()), 1)
    first = end[:, None] - (steps - torch.arange(steps, device=device))[None] * k[:, None]
    idx = first[:, :, None] + torch.arange(widest, device=device)[None, None]
    inside = (torch.arange(widest, device=device)[None, None] < k[:, None, None]) & (first[:, :, None] >= 0)
    v = series.gather(1, idx.clamp(0, length - 1).reshape(n, -1)).reshape(n, steps, widest)
    present = inside & torch.isfinite(v)
    count = present.sum(2)
    mean = torch.where(present, v, torch.zeros((), dtype=v.dtype, device=device)).double().sum(2) / count.clamp_min(1)
    return torch.where(count > 0, mean.to(series.dtype), torch.full((), float("nan"), dtype=series.dtype, device=device))


def causal_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`batches.causal_stats` on the device: the running mean and Bessel-corrected deviation
    of the present values of `values [n, steps]`, read at the last step of each patch."""
    v = values.double()
    present = torch.isfinite(v)
    x = torch.where(present, v, torch.zeros((), dtype=v.dtype, device=v.device))
    count = present.cumsum(1).double().clamp_min(1.0)
    loc = x.cumsum(1) / count
    before = torch.cat([torch.zeros_like(loc[:, :1]), loc[:, :-1]], dim=1)
    m2 = torch.where(present, (x - before) * (x - loc), torch.zeros((), dtype=v.dtype, device=v.device)).cumsum(1)
    scale = torch.sqrt(m2 / (count - 1.0).clamp_min(1.0))
    ends = slice(PATCH - 1, None, PATCH)
    return loc[:, ends].float(), scale[:, ends].clamp_min(MIN_SCALE).float()


def paths(cached_pass, next_position, series: torch.Tensor, end: torch.Tensor, k: torch.Tensor, tags: torch.Tensor,
          patches: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's forecast of the `patches` coarse patches after step `end [n]` of `series [n,
    length]` at `k [n]` steps per bin under the packed `tags [n]`: the levels of `PICKS` in the
    series' units, `[n, 3, patches * PATCH]`, and the share of the coarse context that is
    present, `[n]`. The context is the last `SEQ - patches` patches of bins before `end`; each
    forecast patch joins it as the next input, its median with the two other levels as its
    band, all in the asinh space of the statistics at the context's last patch. Without
    gradients. `cached_pass` and `next_position` are the model's, compiled or not."""
    n, device = series.shape[0], series.device
    held = SEQ - patches
    context = bins(series, end, k, held * PATCH)
    loc, scale = causal_stats(context)
    v = context.reshape(n, held, PATCH)
    present = torch.isfinite(v)
    inputs = torch.zeros(n, SEQ, PATCH, device=device, dtype=dtype)
    mask = torch.zeros(n, SEQ, PATCH, device=device, dtype=dtype)
    band = torch.zeros(n, SEQ, 2 * PATCH, device=device, dtype=dtype)
    inputs[:, :held] = torch.where(present, torch.asinh((torch.nan_to_num(v) - loc[:, :, None]) / scale[:, :, None]), 0.0).to(dtype)
    mask[:, :held] = present.to(dtype)
    last_loc, last_scale = loc[:, -1, None, None].double(), scale[:, -1, None, None].double()
    out = []
    with torch.no_grad():
        for t in range(patches):
            if t == 0:
                full, states = cached_pass(inputs, mask, tags, band, 0, None, None, None)
                q = full[:, held - 1]
            else:
                q = next_position(inputs, mask, band, tags, torch.full((n,), held - 1 + t, dtype=torch.int64, device=device), states, None, None)
            q = q[:, PICKS].float().clamp(-LIMIT, LIMIT)
            out.append((last_loc + last_scale * torch.sinh(q.double())).float())
            if t + 1 < patches:
                inputs[:, held + t] = q[:, 1].to(dtype)
                mask[:, held + t] = 1.0
                band[:, held + t] = torch.cat([q[:, 0] - q[:, 1], q[:, 2] - q[:, 1]], dim=1).to(dtype)
    return torch.cat(out, dim=2), present.reshape(n, -1).float().mean(1)


def channel(levels: torch.Tensor, share: torch.Tensor, k: torch.Tensor, cut: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor,
            frozen: torch.Tensor, hold: torch.Tensor | None = None) -> torch.Tensor:
    """The guide input `[n, SEQ, GUIDE]` of tiles whose context ends with patch `cut [n]`. A
    position `p >= cut` forecasts the patch `p - cut` patches after the context; it holds the
    coarse forecast `levels [n, 3, bins]` (the levels of `PICKS` per bin of `k [n]` steps after
    the context, in the series' units) at the steps of that patch: the median, then the two
    other levels as offsets from it, in the asinh space of the position's statistics (`loc`,
    `scale [n, POSITIONS]`), for the tiles of `frozen [n]` of those at position `min(p, hold)`
    (`hold [n]`, by default `cut`); then a one and `share [n]`. Zero at the positions before
    `cut`, for a patch the levels do not cover whole, and for tiles with `k` zero."""
    n, device = levels.shape[0], levels.device
    positions = torch.arange(POSITIONS, device=device)[None]
    ahead = positions - cut[:, None]
    step = ahead[:, :, None] * PATCH + torch.arange(PATCH, device=device)[None, None]
    at_bin = torch.div(step, k.clamp_min(1)[:, None, None], rounding_mode="floor")
    held = levels.gather(2, at_bin.clamp(0, levels.shape[2] - 1).reshape(n, 1, -1).expand(n, 3, -1)).reshape(n, 3, POSITIONS, PATCH)
    given = ((ahead >= 0)[:, :, None] & (at_bin < levels.shape[2]) & (k > 0)[:, None, None]).all(2) & torch.isfinite(held).all(3).all(1)
    stats = torch.where(frozen[:, None], torch.minimum(positions, (cut if hold is None else hold)[:, None]), positions.expand(n, -1))
    stats = stats.clamp(0, POSITIONS - 1)
    centre = loc.gather(1, stats)[:, None, :, None].double()
    spread = scale.gather(1, stats).clamp_min(MIN_SCALE)[:, None, :, None].double()
    a = torch.asinh((torch.nan_to_num(held).double() - centre) / spread).float().clamp(-LIMIT, LIMIT)
    out = torch.zeros(n, SEQ, GUIDE, device=device)
    g = given[:, :, None]
    out[:, :POSITIONS, :PATCH] = torch.where(g, a[:, 1], 0.0)
    out[:, :POSITIONS, PATCH : 2 * PATCH] = torch.where(g, a[:, 0] - a[:, 1], 0.0)
    out[:, :POSITIONS, 2 * PATCH : 3 * PATCH] = torch.where(g, a[:, 2] - a[:, 1], 0.0)
    out[:, :POSITIONS, 3 * PATCH] = given.float()
    out[:, :POSITIONS, 3 * PATCH + 1] = given.float() * share[:, None]
    return out
