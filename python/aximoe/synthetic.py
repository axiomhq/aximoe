"""Generated series: a prior over periodic, observability-like series, for teacher labelling and
for the trainer's synthetic share.

A series of 2,048 steps at a nominal bin width is a periodic core plus noise, trend and events,
then optionally exponentiated, floored, quantised and given gaps. The core comes from one of two
families: `components`, up to three hand-shaped periodic profiles (harmonics, piecewise linear,
pulses, sawtooth, steps), or `kernel_core`, a draw from a Gaussian process whose kernel is a sum
of products of stationary kernels (periodic, squared exponential, Matern, rational quadratic),
the family of KernelSynth (Ansari et al., Chronos, 2024), sampled by circulant embedding. Periods
are mostly the natural durations (a minute to a week) that fit the bin width, so the width says
which cycles are plausible, as on real telemetry. Two more families have no noise to speak of:
`sparse`, zero most of the time with bursts of events, as counters of rare events are, and
`ramp`, a value that grows linearly and falls back at resets, as an age, an uptime or the time
since a job last ran does. A fifth, `pulse`, is a baseline with events that repeat exactly, as
timers, flushes and scheduled jobs draw them; `any-pulse` draws it half the time and `any` otherwise.
A sixth, `plateau`, holds a level and moves to another, as a pool size, a replica count or a
state does; `any-plateau` draws it half the time and `any` otherwise.

    python python/aximoe/synthetic.py --plot grid.png [--seed 3] [--family kernel]
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.signal import lfilter

TILE = 2048
PATCH = 32
WIDTHS = (10, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600)
# seconds: 1, 5, 10, 15, 30 minutes, 1, 2, 4, 6, 8, 12 hours, a day, a week
NATURAL = (60, 300, 600, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200, 86400, 604800)
MIN_PERIOD, MAX_PERIOD = 3.0, 1536.0
# length of the circulant the stationary draws live on, at least twice the tile
CIRCULANT = 4096


def natural_periods(width: int) -> list[float]:
    return [d / width for d in NATURAL if MIN_PERIOD <= d / width <= MAX_PERIOD]


def draw_period(rng: np.random.Generator, width: int, first: float | None = None) -> float:
    """A period in steps: a multiple or a fraction of `first` half the time when given, else a
    natural duration at this width (70 %) or log-uniform between 3 and 1,024 steps."""
    if first is not None and rng.random() < 0.5:
        m = rng.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 24])
        p = first * m if first * m <= MAX_PERIOD else first / m
        if p >= MIN_PERIOD:
            return float(p)
    if rng.random() < 0.7:
        return float(rng.choice(natural_periods(width)))
    return float(np.exp(rng.uniform(np.log(MIN_PERIOD), np.log(1024.0))))


def shape(rng: np.random.Generator, theta: np.ndarray) -> np.ndarray:
    """One periodic profile at the phases `theta` in [0, 1), zero mean and unit deviation."""
    kind = rng.choice(5, p=[0.35, 0.2, 0.2, 0.1, 0.15])
    if kind == 0:
        # harmonics with a power-law fall-off
        h = np.arange(1, rng.integers(1, 9) + 1)
        a = rng.normal(size=len(h)) / h ** rng.uniform(0.5, 2.5)
        y = (a[:, None] * np.cos(2 * np.pi * (h[:, None] * theta[None] + rng.random(len(h))[:, None]))).sum(0)
    elif kind == 1:
        # piecewise linear through random knots
        k = rng.integers(2, 9)
        x = np.sort(rng.random(k))
        v = rng.normal(size=k)
        y = np.interp(theta, np.concatenate([x - 1, x, x + 1]), np.tile(v, 3))
    elif kind == 2:
        # one or two pulses, boxes or bumps
        y = np.zeros_like(theta)
        for _ in range(rng.integers(1, 3)):
            centre, width = rng.random(), rng.uniform(0.02, 0.3)
            d = np.abs((theta - centre + 0.5) % 1.0 - 0.5)
            y += rng.uniform(0.3, 1.0) * (np.exp(-0.5 * (d / (width / 2.5)) ** 2) if rng.random() < 0.5 else (d < width / 2))
    elif kind == 3:
        # sawtooth, rising or falling
        y = theta ** rng.uniform(0.5, 2.0)
        if rng.random() < 0.5:
            y = 1.0 - y
    else:
        # steps between a few levels
        k = rng.integers(2, 7)
        y = rng.normal(size=k)[np.minimum((theta * k).astype(int), k - 1)]
    y = y - y.mean()
    return y / max(y.std(), 1e-6)


def components(rng: np.random.Generator, width: int) -> tuple[np.ndarray, bool]:
    """Zero to three hand-shaped periodic profiles summed to unit deviation, and whether any was drawn."""
    t = np.arange(TILE, dtype=np.float64)
    y = np.zeros(TILE)
    count = rng.choice(4, p=[0.12, 0.43, 0.3, 0.15])
    first = None
    for _ in range(count):
        p = draw_period(rng, width, first)
        first = first or p
        comp = shape(rng, (t / p + rng.random()) % 1.0)
        mod = rng.random()
        if mod < 0.2:
            # each cycle scaled by its place in a short super-cycle, as weekdays and weekends
            slots = rng.uniform(0.1, 1.2, size=rng.choice([2, 3, 4, 5, 7]))
            comp = comp * slots[(np.floor(t / p + rng.random()).astype(int)) % len(slots)]
        elif mod < 0.4:
            comp = comp * np.exp(np.cumsum(rng.normal(scale=rng.uniform(0.002, 0.02), size=TILE)))
        y += rng.uniform(0.3, 1.0) * comp
    if count:
        y /= max(y.std(), 1e-6)
    return y, bool(count)


def stationary(rng: np.random.Generator, acov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two independent draws `[TILE]` of a zero-mean stationary Gaussian series whose
    autocovariance at lag k is `acov[k]`, k from 0 to CIRCULANT / 2, by circulant embedding;
    negative eigenvalues of the embedding are clipped at zero."""
    ring = np.concatenate([acov, acov[-2:0:-1]])
    eig = np.maximum(np.fft.fft(ring).real, 0.0)
    z = rng.normal(size=CIRCULANT) + 1j * rng.normal(size=CIRCULANT)
    y = np.fft.fft(np.sqrt(eig / CIRCULANT) * z)
    return y.real[:TILE], y.imag[:TILE]


def kernel(rng: np.random.Generator, width: int, lags: np.ndarray, first: float | None) -> tuple[np.ndarray, float | None]:
    """One stationary kernel at `lags` (steps), unit variance, and its period if it has one."""
    kind = rng.choice(5, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    if kind == 0:
        p = draw_period(rng, width, first)
        return np.exp(-2.0 * np.sin(np.pi * lags / p) ** 2 / np.exp(rng.uniform(np.log(0.2), np.log(3.0))) ** 2), p
    scale = float(np.exp(rng.uniform(np.log(2.0), np.log(2000.0))))
    if kind == 1:
        return np.exp(-0.5 * (lags / scale) ** 2), None
    if kind == 2:
        return np.exp(-lags / scale), None
    if kind == 3:
        r = np.sqrt(3.0) * lags / scale
        return (1.0 + r) * np.exp(-r), None
    alpha = float(rng.choice([0.1, 1.0, 10.0]))
    return (1.0 + 0.5 * (lags / scale) ** 2 / alpha) ** -alpha, None


def kernel_core(rng: np.random.Generator, width: int) -> tuple[np.ndarray, bool]:
    """A Gaussian-process draw of unit deviation: one to three groups summed, each a product of
    one or two stationary kernels, a fifth of them times a line; and whether a period is in it."""
    lags = np.arange(CIRCULANT // 2 + 1, dtype=np.float64)
    ramp = np.arange(TILE) / TILE
    y = np.zeros(TILE)
    first = None
    for _ in range(rng.integers(1, 4)):
        acov = np.ones_like(lags)
        for _ in range(rng.integers(1, 3)):
            k, p = kernel(rng, width, lags, first)
            first = first or p
            acov = acov * k
        a, b = stationary(rng, acov)
        if rng.random() < 0.2:
            a = rng.normal() * a + rng.normal() * (ramp - rng.random()) * b
        y += float(np.exp(rng.uniform(np.log(0.1), 0.0))) * a
    return y / max(y.std(), 1e-6), first is not None


def gaps(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """`y` as float32 with a late start (10 %) or up to three holes (5 %) set to NaN."""
    r = rng.random()
    if r < 0.1:
        y[: 32 * rng.integers(1, 13)] = np.nan
    elif r < 0.15:
        for _ in range(rng.integers(1, 4)):
            at = rng.integers(0, TILE - 100)
            y[at : at + rng.integers(5, 100)] = np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        out = y.astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def late_start(rng: np.random.Generator, y: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """`y` with every step before a drawn start set to NaN: the start lies 2 to `patch - 1` steps
    before the end of a patch of `patch` steps drawn among all but the last, so the first patch
    with values holds at least two of them."""
    at = int(rng.integers(0, TILE // patch - 1))
    y[: (at + 1) * patch - int(rng.integers(2, patch))] = np.nan
    return y


def sparse(rng: np.random.Generator, width: int) -> np.ndarray:
    """A series that is zero except for events: arrivals at a random rate, in on and off
    stretches or once per cycle of a period, with skewed sizes, optionally whole counts, and
    optionally smeared over the following steps as a windowed rate is."""
    t = np.arange(TILE)
    kind = rng.random()
    if kind < 0.3:
        # once per cycle, most cycles
        p = draw_period(rng, width)
        phase = rng.uniform(0, p)
        hits = np.floor((t + phase) / p) != np.floor((t - 1 + phase) / p)
        hits &= rng.random(TILE) < rng.uniform(0.5, 1.0)
    else:
        rate = np.exp(rng.uniform(np.log(1e-3), np.log(0.2)))
        if kind < 0.6:
            # on and off stretches
            cuts = np.sort(rng.integers(1, TILE, size=rng.integers(1, 9)))
            on = (np.searchsorted(cuts, t) + rng.integers(0, 2)) % 2 == 0
            rate = rate * np.where(on, 1.0, rng.choice([0.0, 0.05]))
        hits = rng.random(TILE) < rate
    size = np.exp(rng.normal(scale=rng.uniform(0.2, 1.5), size=TILE))
    if rng.random() < 0.4:
        size = np.ceil(size * rng.uniform(0.5, 3.0))
    y = hits * size
    if rng.random() < 0.5:
        tau = rng.uniform(0.5, 6.0)
        y = lfilter([1.0 - np.exp(-1.0 / tau)], [1.0, -np.exp(-1.0 / tau)], y)
        y[y < 1e-3 * max(y.max(), 1e-12)] = 0.0
    return gaps(rng, y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6)))))


def ramp(rng: np.random.Generator, width: int) -> np.ndarray:
    """The time since the last reset, in steps: no reset in the window (an uptime), resets every
    period, resets at some slots of a short super-cycle (a job that skips weekends), or resets at
    random intervals; optionally counting down, capped, or with a little jitter."""
    t = np.arange(TILE, dtype=np.float64)
    kind = rng.random()
    if kind < 0.15:
        resets = np.array([-rng.uniform(0.0, 2000.0)])
    elif kind < 0.8:
        p = draw_period(rng, width)
        resets = np.arange(rng.uniform(-8 * p, -7 * p), TILE, p)
        if kind >= 0.5:
            m = int(rng.choice([2, 3, 4, 5, 7]))
            kept = rng.random(m) < rng.uniform(0.4, 0.9)
            kept[rng.integers(m)] = True
            resets = resets[kept[np.arange(len(resets)) % m]]
    else:
        waits = rng.gamma(rng.uniform(1.0, 20.0), size=600)
        waits = waits / waits.mean() * np.exp(rng.uniform(np.log(8.0), np.log(600.0)))
        resets = np.cumsum(waits) - waits[0] * rng.random()
    # the resets before the window stay, so the first stretch is as old as its neighbours
    resets = np.unique(np.ceil(resets[resets < TILE - 1]))
    if len(resets) == 0 or resets[0] > 0:
        resets = np.concatenate([[-rng.uniform(0.0, 500.0)], resets])
    origin = resets[np.maximum(np.searchsorted(resets, t, side="right") - 1, 0)]
    y = t - origin
    if rng.random() < 0.1:
        y = np.minimum(y, np.quantile(y, rng.uniform(0.5, 0.95)))
    if rng.random() < 0.15:
        y = -y
    if rng.random() < 0.3:
        y = y + rng.normal(scale=rng.uniform(0.01, 0.05) * max(y.std(), 1e-6), size=TILE)
    y = rng.uniform(-5.0, 50.0) * max(y.std(), 1e-6) + y
    return gaps(rng, y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6)))))


def pulse(rng: np.random.Generator, width: int) -> np.ndarray:
    """A baseline with one to four events per cycle of a period, each at its own place in the
    cycle with its own height and shape: a single step, a box, a decaying tail or a rise and
    fall. Three series in five repeat exactly; the others vary a little: the height per cycle,
    a missed cycle, a short super-cycle of heights, noise on the baseline, whole counts or a
    windowed rate. The period is a whole number of steps up to 64, any length from 2 to 700
    steps, or a natural duration at this width."""
    t = np.arange(TILE, dtype=np.float64)
    r = rng.random()
    if r < 0.35:
        p = float(rng.integers(2, 65))
    elif r < 0.6:
        p = float(np.exp(rng.uniform(np.log(2.0), np.log(700.0))))
    else:
        p = draw_period(rng, width)
    start = rng.uniform(0.0, p)
    cycle = np.floor((t + start) / p).astype(np.int64)
    place = (t + start) - cycle * p
    cycle -= cycle.min()
    exact = rng.random() < 0.6
    y = np.zeros(TILE)
    for _ in range(rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.08])):
        # steps since the event's start in this cycle
        d = (place - rng.uniform(0.0, p)) % p
        kind = rng.random()
        if kind < 0.45:
            body = (d < 1.0).astype(np.float64)
        elif kind < 0.7:
            body = (d < rng.uniform(1.0, max(1.5, min(0.3 * p, 12.0)))).astype(np.float64)
        elif kind < 0.85:
            tau = rng.uniform(0.5, max(1.0, min(0.2 * p, 8.0)))
            body = np.exp(-d / tau) * (d < 6.0 * tau)
        else:
            span = rng.uniform(2.0, max(3.0, min(0.5 * p, 30.0)))
            body = np.clip(1.0 - np.abs(2.0 * d / span - 1.0), 0.0, None) * (d < span)
        height = (1.0 if rng.random() < 0.85 else -1.0) * float(np.exp(rng.normal(scale=0.7)))
        if not exact:
            v = rng.random()
            if v < 0.4:
                body = body * np.exp(rng.normal(scale=rng.uniform(0.02, 0.3), size=cycle.max() + 1))[cycle]
            elif v < 0.6:
                body = body * (rng.random(cycle.max() + 1) < rng.uniform(0.85, 1.0))[cycle]
            elif v < 0.8:
                slots = rng.uniform(0.2, 1.5, size=rng.choice([2, 3, 4, 5, 7]))
                body = body * slots[cycle % len(slots)]
        y += height * body
    top = max(np.abs(y).max(), 1e-6)
    if not exact:
        v = rng.random()
        if v < 0.4:
            y = y + rng.normal(scale=float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))) * top, size=TILE)
        elif v < 0.55:
            y = np.round(y / top * rng.integers(2, 40))
        elif v < 0.75:
            tau = rng.uniform(0.5, 4.0)
            y = lfilter([1.0 - np.exp(-1.0 / tau)], [1.0, -np.exp(-1.0 / tau)], y)
    if rng.random() < 0.5:
        y = y + rng.uniform(-5.0, 50.0) * max(y.std(), 1e-6)
    return gaps(rng, y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6)))))


def plateau(rng: np.random.Generator, width: int) -> np.ndarray:
    """A value that holds a level and moves to another: two to five levels visited in a fixed
    order or at random, or a new level near the last each time. A level is held for its own
    number of steps (6 to 400), the same every time, within 15 % of it, or within 60 %; the move
    is a jump, a straight ramp of 2 to 12 steps or an approach that decays over as many. A third
    of the series carry a little noise, one in seven is rounded to a grid."""
    kind = rng.random()
    count = 2 if kind < 0.5 else int(rng.integers(3, 6))
    levels = np.sort(rng.normal(size=count))
    walk = kind >= 0.85
    cyclic = rng.random() < 0.7
    hold = np.exp(rng.uniform(np.log(6.0), np.log(400.0), size=count))
    spread = float(rng.choice([0.0, 0.15, 0.6], p=[0.3, 0.4, 0.3]))
    move = rng.random()
    span = 0 if move < 0.3 else int(rng.integers(2, 13))
    k = int(rng.integers(count))
    value = float(levels[k])
    skipped = int(rng.uniform(0.0, hold.sum()))
    out: list[float] = []
    while len(out) < TILE + skipped:
        steps = max(int(round(hold[k] * rng.uniform(1.0 - spread, 1.0 + spread))), 1)
        k = (k + 1) % count if cyclic else int((k + rng.integers(1, count)) % count)
        after = value + float(rng.normal(scale=0.5)) if walk else float(levels[k])
        out += [value] * steps
        if span:
            part = (np.arange(span) + 1.0) / (span + 1.0)
            if move >= 0.9:
                part = 1.0 - np.exp(-3.0 * part)
            out += list(value + (after - value) * part)
        value = after
    y = np.array(out[skipped : skipped + TILE])
    dev = max(y.std(), 1e-6)
    if rng.random() < 1 / 3:
        y = y + rng.normal(scale=float(np.exp(rng.uniform(np.log(0.003), np.log(0.05)))) * dev, size=TILE)
    elif rng.random() < 0.2:
        y = np.round(y / dev * rng.integers(2, 20))
    y = rng.uniform(-5.0, 50.0) * max(y.std(), 1e-6) + y
    return gaps(rng, y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6)))))


def members(rng: np.random.Generator, shared: np.ndarray, count: int) -> np.ndarray:
    """`count` series per row of `shared [n, TILE]` that follow its shape as the instances of
    one service follow their common load: the row standardised, with a gain, a level and white
    noise of its own per series (deviation 0.02 to 1 of the row's), back in the row's units and
    missing where the row is; `[n, count, TILE]`."""
    n = shared.shape[0]
    present = np.isfinite(shared)
    held = np.maximum(present.sum(1), 1)
    mean = np.where(present, shared, 0.0).sum(1) / held
    dev = np.sqrt(np.where(present, (shared - mean[:, None]) ** 2, 0.0).sum(1) / held)
    dev = np.maximum(dev, 1e-6 * np.abs(mean) + 1e-12)
    z = (shared.astype(np.float64) - mean[:, None]) / dev[:, None]
    gain = np.exp(rng.normal(scale=0.3, size=(n, count, 1)))
    level = rng.normal(scale=0.5, size=(n, count, 1))
    sigma = np.exp(rng.uniform(np.log(0.02), np.log(1.0), size=(n, count, 1)))
    out = (gain * z[:, None] + level + sigma * rng.normal(size=(n, count, shared.shape[1]))) * dev[:, None, None] + mean[:, None, None]
    return out.astype(np.float32)


def series(rng: np.random.Generator, width: int, family: str = "any") -> np.ndarray:
    """One generated series `[TILE]`, float32, NaN where missing; `any` draws the family: 40 %
    Gaussian-process kernels, 40 % hand-shaped components, 10 % sparse, 10 % ramps. A sixth of
    the periodic series of the first two families are clean: no noise and no events. `pulse` and
    `plateau` are families of their own; `any-pulse` and `any-plateau` draw theirs half the time
    and `any` otherwise."""
    if family == "pulse" or (family == "any-pulse" and rng.random() < 0.5):
        return pulse(rng, width)
    if family == "plateau" or (family == "any-plateau" and rng.random() < 0.5):
        return plateau(rng, width)
    t = np.arange(TILE, dtype=np.float64)
    pick = rng.random() if family in ("any", "any-pulse", "any-plateau") else {"kernel": 0.0, "components": 0.5, "sparse": 0.85, "ramp": 0.95}[family]
    if pick >= 0.9:
        return ramp(rng, width)
    if pick >= 0.8:
        return sparse(rng, width)
    if pick < 0.4:
        y, periodic = kernel_core(rng, width)
    else:
        y, periodic = components(rng, width)
    if periodic and rng.random() < 1 / 6:
        y = rng.uniform(-5.0, 50.0) + y
        return gaps(rng, y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6)))))
    # noise: white or autoregressive, Gaussian or heavy-tailed
    sigma = float(np.exp(rng.uniform(np.log(0.02), np.log(2.0)))) if periodic else 1.0
    e = rng.standard_t(3, size=TILE) / np.sqrt(3.0) if rng.random() < 0.25 else rng.normal(size=TILE)
    if rng.random() < 0.7:
        # unit-variance AR(1), started in its stationary state
        phi = rng.uniform(0.3, 0.995)
        gain = np.sqrt(1 - phi * phi)
        e, _ = lfilter([gain], [1.0, -phi], e, zi=[phi * e[0] / gain])
    y = y + sigma * e
    # trend: none, a line, or a line with changes of slope
    r = rng.random()
    if r < 0.5:
        total = rng.normal(scale=float(rng.choice([1.0, 3.0, 10.0], p=[0.5, 0.3, 0.2])))
        if r < 0.15:
            cuts = np.sort(rng.integers(1, TILE - 1, size=rng.integers(1, 4)))
            slopes = rng.normal(size=len(cuts) + 1)
            trend = np.cumsum(np.repeat(slopes, np.diff(np.concatenate([[0], cuts, [TILE]]))))
            y += total * trend / max(np.abs(trend).max(), 1e-6)
        else:
            y += total * t / TILE
    # events: level shifts, spikes, bumps that decay
    for _ in range(rng.poisson(0.3)):
        y[rng.integers(1, TILE):] += rng.normal(scale=2.0)
    if rng.random() < 0.25:
        hits = rng.random(TILE) < np.exp(rng.uniform(np.log(1e-3), np.log(0.05)))
        sign = 1.0 if rng.random() < 0.8 else -1.0
        y += sign * hits * rng.exponential(size=TILE) * rng.uniform(1.0, 10.0)
    if rng.random() < 0.1:
        for _ in range(rng.integers(1, 4)):
            at, tau = rng.integers(0, TILE), rng.uniform(5, 200)
            y[at:] += rng.normal(scale=3.0) * np.exp(-np.arange(TILE - at) / tau)
    # shape of the value range: skewed positive, floored, quantised
    if rng.random() < 0.3:
        y = np.exp(rng.uniform(0.2, 1.0) * np.clip((y - y.mean()) / max(y.std(), 1e-6), -20.0, 20.0))
    if rng.random() < 0.15:
        y = np.maximum(y, np.quantile(y, rng.uniform(0.1, 0.7)))
    if rng.random() < 0.1:
        step = (y.max() - y.min()) / rng.integers(3, 31)
        if step > 0:
            y = np.round(y / step) * step
    y = rng.uniform(-5.0, 50.0) + y
    y = y * float(np.exp(rng.uniform(np.log(1e-3), np.log(1e6))))
    return gaps(rng, y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plot", required=True, help="write a grid of 40 example series")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--family", default="any", choices=["any", "kernel", "components", "sparse", "ramp", "pulse", "any-pulse", "plateau", "any-plateau"])
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(8, 5, figsize=(26, 22))
    for k, ax in enumerate(axes.flat):
        width = WIDTHS[k % len(WIDTHS)]
        ax.plot(series(rng, width, args.family), lw=0.5)
        ax.set_title(f"{width} s", fontsize=8)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=60)


if __name__ == "__main__":
    main()
