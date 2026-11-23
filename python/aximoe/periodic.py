"""A tile's own values one detected period back, as an input for the positions a pass forecasts
ahead of its context: `detect_lag` reads the period of the observed part of each tile from its
autocorrelation, `copy_ahead` copies each forecast step from the latest observed step a whole
number of periods before it and marks the copied steps with the height of that autocorrelation.
"""
from __future__ import annotations

import torch

from aximoe.student import PATCH, SEQ

MAX_LAG = 1000
# peaks this close to the highest autocorrelation peak count as the same height
TIE = 0.02


def detect_lag(values: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The period, in steps, of each tile `values [n, steps]` (NaN where missing) over its steps
    up to `end [n]`: the smallest lag from 2 to `MAX_LAG` whose autocorrelation is a local peak
    after the initial decay, within `TIE` of the highest such peak and with two periods inside
    the span; zero where no lag qualifies. With it, the autocorrelation at that lag, between
    zero and one."""
    steps = values.shape[1]
    at = torch.arange(steps, device=values.device)
    seen = torch.isfinite(values) & (at[None] <= end[:, None])
    v = torch.where(seen, values, torch.zeros((), dtype=values.dtype, device=values.device)).double()
    mean = v.sum(1, keepdim=True) / seen.sum(1, keepdim=True).clamp_min(1)
    x = torch.where(seen, v - mean, torch.zeros((), dtype=v.dtype, device=v.device))
    f = torch.fft.rfft(x, 2 * steps, dim=1)
    acf = torch.fft.irfft(f * f.conj(), 2 * steps, dim=1)[:, : MAX_LAG + 1]
    m = torch.fft.rfft(seen.double(), 2 * steps, dim=1)
    pairs = torch.fft.irfft(m * m.conj(), 2 * steps, dim=1)[:, : MAX_LAG + 1]
    acf = acf / pairs.clamp_min(1.0)
    acf = acf / acf[:, :1].clamp_min(1e-12)
    lags = torch.arange(MAX_LAG + 1, device=values.device)
    rise = acf[:, 2:] > acf[:, 1:-1]
    # the first lag after which the autocorrelation rises again ends its initial decay
    trough = torch.where(rise.any(1), rise.to(torch.int8).argmax(1) + 1, MAX_LAG + 1)
    peak = torch.zeros_like(acf, dtype=torch.bool)
    peak[:, 2:-1] = (acf[:, 2:-1] >= acf[:, 1:-2]) & (acf[:, 2:-1] >= acf[:, 3:])
    peak &= (lags[None] > trough[:, None]) & (2 * lags[None] <= end[:, None] + 1)
    height = torch.where(peak, acf, -torch.ones((), dtype=acf.dtype, device=acf.device))
    best = height.max(1).values
    first = (height >= best[:, None] - TIE).to(torch.int8).argmax(1)
    found = best > -1.0
    strength = torch.where(found, height.gather(1, first[:, None])[:, 0], torch.zeros((), dtype=acf.dtype, device=acf.device))
    return torch.where(found, first, 0), strength.clamp(0.0, 1.0).to(values.dtype)


def copy_ahead(values: torch.Tensor, lag: torch.Tensor, strength: torch.Tensor, cut: torch.Tensor, loc: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
    """The copy input `[n, SEQ, 2 * PATCH]` of tiles `values [n, SEQ * PATCH]` whose input
    patches after position `cut [n]` are missing: for each position from `cut` on, the steps of
    the patch it forecasts, each the tile's value the smallest whole number of `lag [n]` steps
    back that lies in the patches up to `cut`, in the asinh space of `loc`, `scale [n]`, followed
    by `strength [n]` at the copied steps. Zero for the positions before `cut`, for a tile whose
    lag is below 2 and where the source step is missing."""
    n, steps = values.shape
    device = values.device
    # step `PATCH + k` belongs to the patch that position `k // PATCH` forecasts
    target = torch.arange(PATCH, steps + PATCH, device=device)
    position = torch.arange(SEQ, device=device).repeat_interleave(PATCH)
    end = (cut + 1) * PATCH - 1
    period = lag.clamp_min(1)[:, None]
    ahead = target[None] - end[:, None]
    source = target[None] - torch.div(ahead + period - 1, period, rounding_mode="floor") * period
    usable = (lag[:, None] >= 2) & (position[None] >= cut[:, None]) & (source >= 0)
    v = values.gather(1, source.clamp(0, steps - 1))
    usable = usable & torch.isfinite(v)
    z = torch.asinh((torch.nan_to_num(v) - loc[:, None]) / scale[:, None])
    zero = torch.zeros((), dtype=values.dtype, device=device)
    copy = torch.where(usable, z.to(values.dtype), zero)
    marks = usable.to(values.dtype) * strength[:, None]
    return torch.cat([copy.reshape(n, SEQ, PATCH), marks.reshape(n, SEQ, PATCH)], dim=2)
