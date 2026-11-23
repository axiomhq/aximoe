"""What changed around a moment: five container metrics fetched from MetricsDB at 1-minute
bins, every 5-minute bin of a window around the moment scored with robust z and with the
student, each score calibrated against the same series' own previous day, the bins rarer than
a threshold flagged and their containers ranked. The MCP server and the command line both call
`investigate`.
"""
from __future__ import annotations

import asyncio
import bisect
import dataclasses
import datetime as dt
import json
import os
import re
import time
import tomllib

import httpx
import numpy as np
import torch

from aximoe import batches, forecast, student
from aximoe.student import LEVELS, MEDIAN, PATCH, SEQ

# ----------------------------------------------------------------------------- metrics

Q10, Q90 = LEVELS.index(0.1), LEVELS.index(0.9)


@dataclasses.dataclass(frozen=True)
class Metric:
    """One scored metric: `channel` is its name in the tile index (and so its tag), `mpl` the
    MetricsDB metric, `counter` whether it is turned into a per-second rate, `unit` the unit
    after the rate, `floor` the smallest spread the detectors divide by, `level` the smallest
    normal level an excess over the normal range is measured in."""
    channel: str
    mpl: str
    counter: bool
    unit: str
    floor: float
    level: float


# CPU use, CPU throttling, the memory Kubernetes evicts on and restarts; OOM kills are counted, not scored
METRICS = [
    Metric("cpu_usage_rate", "container_cpu_usage_seconds_total", True, "cores", 0.0, 1e-4),
    Metric("cpu_throttled_rate", "container_cpu_cfs_throttled_periods_total", True, "per_second", 1 / 60, 1 / 60),
    Metric("mem_working_set", "container_memory_working_set_bytes", False, "bytes", 0.0, 1.0),
    Metric("restarts_rate", "kube_pod_container_status_restarts_total", True, "per_second", 1 / 60, 1 / 60),
]
OOM_KILLS = "oom_kills"  # the name OOM kills rank under beside the scored metrics


def query(dataset: str, metric: Metric, containers: str) -> str:
    """The MPL query for one metric, summed over the pods of each namespace and container;
    `containers` is a regular expression on the container name."""
    rate = "| map rate " if metric.counter else ""
    return (f"`{dataset}`:{metric.mpl} | where container is string and container == #/{containers}/ "
            f"{rate}| align to 1m using avg | group by namespace, container using sum")


def check_pattern(containers: str) -> None:
    if not containers:
        raise ValueError("the container pattern is empty; use .* for every container")
    if "/" in containers:
        raise ValueError(f"the container pattern {containers!r} contains '/', which ends an MPL regex")


def sig(v: float) -> str:
    if v == 0.0:
        return "0"
    digits = min(max(2 - int(np.floor(np.log10(abs(v)))), 0), 6)
    return f"{v:.{digits}f}"


def format_value(unit: str, v: float) -> str:
    """A value in the metric's own unit, short enough for a line of text; rates per minute,
    the grid's step."""
    if not np.isfinite(v):
        return "no data"
    if unit == "cores":
        return f"{sig(v)} cores"
    if unit == "per_second":
        return f"{sig(v * 60)}/min"
    a = abs(v)
    for size, name in ((1024 ** 3, "GiB"), (1024 ** 2, "MiB"), (1024, "KiB")):
        if a >= size:
            return f"{sig(v / size)} {name}"
    return f"{sig(v)} B"


# ----------------------------------------------------------------------------- Axiom

@dataclasses.dataclass
class Deployment:
    name: str
    endpoint: str
    token: str
    org_id: str


def deployment(name: str) -> Deployment:
    """Deployment `name` of `~/.axiom.toml`; MPL queries go to its `edge`, else its `url`."""
    path = os.path.join(os.environ.get("HOME", ""), ".axiom.toml")
    with open(path, "rb") as f:
        table = tomllib.load(f)
    d = table.get("deployments", {}).get(name)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: no deployment {name!r}")
    for key in ("url", "token", "org_id"):
        if key not in d:
            raise ValueError(f"{path}: deployment {name!r} has no {key}")
    return Deployment(name, str(d.get("edge", d["url"])).rstrip("/"), d["token"], d["org_id"])


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def mpl(client: httpx.AsyncClient, dep: Deployment, q: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Runs one MPL query; a failed attempt is retried with a growing pause."""
    body = {"mpl": q, "startTime": iso(start), "endTime": iso(end)}
    last = ""
    for attempt in range(6):
        if attempt:
            await asyncio.sleep(2 << min(attempt, 4))
        try:
            r = await client.post(f"{dep.endpoint}/v1/query/_mpl", json=body,
                                  headers={"Authorization": f"Bearer {dep.token}", "X-Axiom-Org-Id": dep.org_id,
                                           "Accept": "application/json"})
        except httpx.HTTPError as e:
            last = str(e)
            continue
        if r.is_success:
            try:
                return r.json()["series"]
            except (ValueError, KeyError) as e:
                raise ValueError(f"MPL response without series ({e}): {r.text[:300]}")
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if 400 <= r.status_code < 500 and r.status_code != 429:
            break
    raise ValueError(f"MPL query failed: {last}\nquery: {q}")


class Grid:
    """Every fetched series on one grid of 1-minute bins starting at `t0`; NaN where missing.
    Keys are (container key `namespace/container`, metric index into METRICS)."""

    def __init__(self, t0: dt.datetime, length: int):
        self.t0, self.len = t0, length
        self.series: dict[tuple[str, int], np.ndarray] = {}

    def add(self, m: int, part: list[dict]) -> None:
        t0 = int(self.t0.timestamp())
        for s in part:
            tags = s.get("tags") or {}
            container = tags.get("container")
            if not isinstance(container, str):
                continue
            if s["resolution"] != 60:
                raise ValueError(f"resolution {s['resolution']} s, expected 60")
            key = (f"{tags.get('namespace', '?')}/{container}", m)
            values = self.series.setdefault(key, np.full(self.len, np.nan))
            i0 = (s["start"] - t0) // 60
            for k, v in enumerate(s["data"]):
                i = i0 + k
                if v is not None and 0 <= i < self.len:
                    values[i] = v

    def containers(self) -> set[str]:
        return {c for c, _ in self.series}


async def fetch(client: httpx.AsyncClient, dep: Deployment, dataset: str, containers: str,
                start: dt.datetime, end: dt.datetime) -> Grid:
    """Every metric for the scope over `[start, end)`, one query per metric, all at once."""
    grid = Grid(start, int((end - start).total_seconds()) // 60)
    parts = await asyncio.gather(*(mpl(client, dep, query(dataset, m, containers), start, end) for m in METRICS))
    for m, part in enumerate(parts):
        grid.add(m, part)
    return grid


# ----------------------------------------------------------------------------- OOM kills

OOM_LOOKBACK = 10  # minutes before the window each pod's counters are read from
OOM_REASON_DELAY = 2  # minutes a restart's termination reason may take to appear


def pod_query(dataset: str, metric: str, containers: str, where: str = "") -> str:
    """The MPL query for a metric's largest value per minute, one series per pod and container."""
    return (f"`{dataset}`:{metric} | where container is string and container == #/{containers}/ {where}"
            f"| align to 1m using max | group by namespace, pod, container using max")


def pod_series(part: list[dict], t0: dt.datetime, length: int) -> dict[tuple[str, str], np.ndarray]:
    """Per (container key `namespace/container`, pod), the values on a 1-minute grid from `t0`;
    NaN where missing."""
    out: dict[tuple[str, str], np.ndarray] = {}
    base = int(t0.timestamp())
    for s in part:
        tags = s.get("tags") or {}
        if not isinstance(tags.get("container"), str):
            continue
        key = (f"{tags.get('namespace', '?')}/{tags['container']}", str(tags.get("pod", "?")))
        values = out.setdefault(key, np.full(length, np.nan))
        i0 = (s["start"] - base) // 60
        for k, v in enumerate(s["data"]):
            if v is not None and 0 <= i0 + k < length:
                values[i0 + k] = v
    return out


def increases(values: np.ndarray) -> np.ndarray:
    """Per minute, how far a counter rose since its last present value; 0 where it fell, where
    it is missing and at its first value."""
    present = np.isfinite(values)
    last = np.maximum.accumulate(np.where(present, np.arange(len(values)), -1))
    prev = np.full(len(values), np.nan)
    prev[1:] = np.where(last[:-1] >= 0, values[np.maximum(last[:-1], 0)], np.nan)
    rise = values - prev
    return np.where(present & np.isfinite(prev) & (rise > 0), rise, 0.0)


async def oom_kills(client: httpx.AsyncClient, dep: Deployment, dataset: str, containers: str,
                    start: dt.datetime, end: dt.datetime) -> list[dict]:
    """The OOM kills in `[start, end)` per container key and minute: restarts of a pod whose last
    termination reason is `OOMKilled`, and rises of the kernel's OOM counter away from a restart of
    the same pod."""
    t0 = start - dt.timedelta(minutes=OOM_LOOKBACK)
    window = int((end - start).total_seconds()) // 60
    length = OOM_LOOKBACK + window + OOM_REASON_DELAY
    queries = [pod_query(dataset, "kube_pod_container_status_restarts_total", containers),
               pod_query(dataset, "kube_pod_container_status_last_terminated_reason", containers,
                         '| where reason == "OOMKilled" '),
               pod_query(dataset, "container_oom_events_total", containers)]
    parts = await asyncio.gather(*(mpl(client, dep, q, t0, t0 + dt.timedelta(minutes=length)) for q in queries))
    restarts, reason, counter = (pod_series(p, t0, length) for p in parts)
    kills: dict[tuple[str, int], float] = {}
    restarted: set[tuple[str, str, int]] = set()
    for (key, pod), values in restarts.items():
        oom = reason.get((key, pod))
        if oom is None:
            continue
        rise = increases(values)
        for t in np.nonzero(rise)[0]:
            if OOM_LOOKBACK <= t < OOM_LOOKBACK + window and np.any(oom[t : t + OOM_REASON_DELAY + 1] >= 1):
                kills[(key, int(t))] = kills.get((key, int(t)), 0.0) + rise[t]
                restarted.add((key, pod, int(t)))
    for (key, pod), values in counter.items():
        rise = increases(values)
        for t in np.nonzero(rise)[0]:
            t = int(t)
            # a kill that restarts its pod can show in both, up to the reason's delay apart
            near = any((key, pod, u) in restarted for u in range(t - OOM_REASON_DELAY, t + OOM_REASON_DELAY + 1))
            if OOM_LOOKBACK <= t < OOM_LOOKBACK + window and not near:
                kills[(key, t)] = kills.get((key, t), 0.0) + rise[t]
    return [{"container": key, "minute": iso(t0 + dt.timedelta(minutes=t)), "kills": int(round(n))}
            for (key, t), n in sorted(kills.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


# ----------------------------------------------------------------------------- the student

class Scaler:
    """`forecast.causal_stats` kept running on the device: each row's float64 running mean and
    Bessel-corrected deviation of its finite values, read at the end of every patch."""

    def __init__(self, n: int, device: torch.device):
        self.total, self.count, self.m2, self.prev = (torch.zeros(n, dtype=torch.float64, device=device) for _ in range(4))

    def extend(self, values: torch.Tensor, locs: torch.Tensor, scales: torch.Tensor, first: int) -> None:
        """Takes in `values [n, k * PATCH]` and writes the statistics at the end of each of its
        patches into `locs`, `scales [n, positions]` from position `first`."""
        v = values.double()
        for t in range(v.shape[1]):
            present = torch.isfinite(v[:, t])
            x = torch.where(present, v[:, t], 0.0)
            self.total += x
            self.count += present.double()
            loc = self.total / self.count.clamp_min(1.0)
            self.m2 += torch.where(present, (x - self.prev) * (x - loc), 0.0)
            self.prev = loc
            if (t + 1) % PATCH == 0:
                scale = torch.sqrt(self.m2 / (self.count.clamp_min(1.0) - 1.0).clamp_min(1.0))
                locs[:, first + t // PATCH] = loc.float()
                scales[:, first + t // PATCH] = scale.clamp_min(batches.MIN_SCALE).float()


def fill_inputs(x: torch.Tensor, m: torch.Tensor, offsets: torch.Tensor | None, values: torch.Tensor,
                levels: torch.Tensor | None, locs: torch.Tensor, scales: torch.Tensor, first: int) -> None:
    """Writes `batches.standardise`'s inputs and mask, and `forecast.band_offsets` for the raw
    0.1 and 0.9 `levels [2, n, k * PATCH]`, of the patches in `values [n, k * PATCH]` into
    positions `first..first + k`, each patch in the statistics at its own end."""
    n, k = values.shape[0], values.shape[1] // PATCH
    centre = locs[:, first : first + k, None]
    spread = scales[:, first : first + k, None].clamp_min(batches.MIN_SCALE)
    v = values.reshape(n, k, PATCH)
    present = torch.isfinite(v)
    inputs = torch.where(present, torch.asinh((torch.nan_to_num(v) - centre) / spread), 0.0)
    x[:, first : first + k], m[:, first : first + k] = inputs, present.float()
    if offsets is not None and levels is not None:
        sides = []
        for side in levels:
            b = side.reshape(n, k, PATCH)
            sides.append(torch.where(torch.isfinite(b), torch.asinh((torch.nan_to_num(b) - centre) / spread) - inputs, 0.0))
        offsets[:, first : first + k] = torch.cat(sides, dim=2)


class Model:
    """A saved model as a detector: forecasting a window patch by patch from the context before
    it, each patch's median fed back as input, beside its 0.1 to 0.9 band for a band-input model.
    The server's grid is one minute per bin."""

    BIN_SECONDS = 60
    BATCH = 1024

    def __init__(self, directory: str, device: str):
        self.model_dir, self.device_name = directory, device
        self.device = torch.device(device)
        self.description, self.student = student.load(directory, self.device)
        self.student.eval()

    def forecast(self, items: list[tuple[str, np.ndarray]], patches: int) -> list[np.ndarray]:
        """`patches` patches after each item's context, an item being (channel, context) with
        every context the same whole number of patches. Returns, per item, `[steps, 3]`: the
        0.1, 0.5 and 0.9 forecast of each step in the series' units, clipped at zero for an item
        whose context has no value below zero (`forecast.floor_at_zero`)."""
        if not items:
            return []
        length = len(items[0][1])
        if length % PATCH or any(len(c) != length for _, c in items):
            raise ValueError("every context must hold the same whole number of patches")
        context = length // PATCH
        if context + patches > SEQ:
            raise ValueError(f"{context} context patches and {patches} forecast patches exceed the student's {SEQ}")
        n, dev = len(items), self.device
        contexts = np.stack([c.astype(np.float32) for _, c in items])
        eligible = torch.from_numpy(forecast.floor_eligible(contexts)).to(dev)
        tags = torch.tensor([student.tag_id(self.description, ch, self.BIN_SECONDS) for ch, _ in items],
                            dtype=torch.int64, device=dev)
        # every position's inputs stay on the device; each pass adds the patch it fed back
        positions = context + patches - 1
        x = torch.zeros((n, positions, PATCH), dtype=torch.float32, device=dev)
        m = torch.zeros_like(x)
        offsets = torch.zeros((n, positions, 2 * PATCH), dtype=torch.float32, device=dev) if self.student.cfg.band_input else None
        locs = torch.zeros((n, positions), dtype=torch.float32, device=dev)
        scales = torch.zeros_like(locs)
        scaler = Scaler(n, dev)
        values = torch.from_numpy(contexts).to(dev)
        scaler.extend(values, locs, scales, 0)
        fill_inputs(x, m, None, values, None, locs, scales, 0)
        out = torch.zeros((n, patches * PATCH, 3), dtype=torch.float64, device=dev)
        # per batch, each layer's attention inputs: later patches run only their own position
        caches: dict[int, list[torch.Tensor]] = {}
        for r in range(patches):
            pos, s = context + r - 1, context + r
            medians = torch.empty((n, PATCH), dtype=torch.float32, device=dev)
            fed = torch.empty((2, n, PATCH), dtype=torch.float32, device=dev)
            for c0 in range(0, n, self.BATCH):
                c1 = min(c0 + self.BATCH, n)
                xs, ms = x[c0:c1, :s], m[c0:c1, :s]
                bs = None if offsets is None else offsets[c0:c1, :s]
                with torch.no_grad():
                    if r == 0:
                        q, caches[c0] = self.student.cached_pass(xs, ms, tags[c0:c1], bs, room=patches - 1)
                        q = q[:, pos]
                    else:
                        q = self.student.next_position(xs, ms, bs, tags[c0:c1], torch.full((c1 - c0,), pos, device=dev), caches[c0])
                raw = locs[c0:c1, pos, None, None].double() + scales[c0:c1, pos, None, None].double() * torch.sinh(q.float().double())
                raw = torch.where(eligible[c0:c1, None, None], raw.clamp_min(0.0), raw)
                out[c0:c1, r * PATCH : (r + 1) * PATCH] = raw[:, [Q10, MEDIAN, Q90]].transpose(1, 2)
                medians[c0:c1] = raw[:, MEDIAN].float()
                fed[:, c0:c1] = raw[:, [Q10, Q90]].float().transpose(0, 1)
            if r + 1 < patches:
                scaler.extend(medians, locs, scales, s)
                fill_inputs(x, m, offsets, medians, fed, locs, scales, s)
        return list(out.cpu().numpy())


# ----------------------------------------------------------------------------- detection

@dataclasses.dataclass
class Settings:
    threshold: float = 0.99      # rarity a bin must reach in its own history to be reported
    bin: int = 5                 # minutes per reported bin
    ref_z: int = 240             # minutes of reference before the window for robust z
    ref_model: int = 480         # minutes of context before the window for the student
    history: int = 24 * 60       # minutes of history each score is calibrated against
    gate: float = 0.5            # smallest share of present points in a reference or context
    min_null: int = 50           # fewest history bins a rarity is computed from
    min_z: float = 3.0           # smallest robust z a reported bin needs
    min_band: float = 0.5        # smallest distance outside the band a reported bin needs, in band widths


@dataclasses.dataclass
class Bin:
    """The largest point score of one bin and where it happened."""
    score: float
    point: int
    observed: float
    low: float
    high: float
    up: bool


@dataclasses.dataclass
class Scored:
    """One detector's verdict on one series."""
    bins: list = dataclasses.field(default_factory=list)
    rarity: list = dataclasses.field(default_factory=list)
    null_size: int = 0
    skipped: str | None = None


@dataclasses.dataclass
class Outcome:
    container: str
    metric: int
    robust_z: Scored
    aximoe: Scored | None = None


def coverage(values: np.ndarray) -> float:
    return float(np.isfinite(values).mean()) if len(values) else 0.0


def finite_sorted(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return np.sort(v[np.isfinite(v)])


def sorted_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row of `values` sorted, NaN last, and its number of finite values."""
    return np.sort(values, axis=1), np.isfinite(values).sum(axis=1)


def row_quantile(srt: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """The `q` quantile of every row of `srt` (sorted, NaN last, `counts` finite values), linear
    between neighbours; NaN for a row without one."""
    pos = q * (counts - 1)
    lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
    rows = np.arange(len(srt))
    a, b = srt[rows, np.maximum(lo, 0)], srt[rows, np.maximum(hi, 0)]
    return np.where(counts > 0, a + (b - a) * (pos - lo), np.nan)


def best_per_bin(score: np.ndarray, bin_: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row of `score [n, length]` (-inf where a point has no value), each bin's largest score,
    NaN for a bin without one, and the offset of the first point reaching it."""
    n, length = score.shape
    bins = -(-length // bin_)
    padded = np.full((n, bins * bin_), -np.inf)
    padded[:, :length] = score
    per = padded.reshape(n, bins, bin_)
    at = per.argmax(axis=2)
    top = np.take_along_axis(per, at[..., None], axis=2)[..., 0]
    return np.where(np.isfinite(top), top, np.nan), at + np.arange(bins) * bin_


def window_rows(values: np.ndarray, starts: np.ndarray, offset: int, length: int) -> np.ndarray:
    """The `length` points from `offset` after each of `starts`, one row per start."""
    return values[starts[:, None] + offset + np.arange(length)]


def robust_z(values: np.ndarray, starts: np.ndarray, length: int, s: Settings, floor: float):
    """Robust z of each point of the windows at `starts` against the `ref_z` minutes before
    each: distance from the reference median in units of 1.4826 MAD, floored at 5 % of the
    median absolute level and at `floor`. Per window its reference coverage, each bin's largest
    score and grid index, and per window and bin the low, high and direction of that point."""
    reference = window_rows(values, starts, -s.ref_z, s.ref_z)
    cov = np.isfinite(reference).mean(axis=1)
    srt, counts = sorted_rows(reference)
    med = row_quantile(srt, counts, 0.5)
    spread = np.maximum(np.maximum(1.4826 * row_quantile(np.sort(np.abs(srt - med[:, None]), axis=1), counts, 0.5),
                                   0.05 * row_quantile(np.sort(np.abs(srt), axis=1), counts, 0.5)),
                        np.maximum(floor, 1e-12))
    y = window_rows(values, starts, 0, length)
    score = np.where(np.isfinite(y), np.abs(y - med[:, None]) / spread[:, None], -np.inf)
    top, at = best_per_bin(score, s.bin)
    point = starts[:, None] + at
    low = np.repeat(row_quantile(srt, counts, 0.05)[:, None], top.shape[1], axis=1)
    high = np.repeat(row_quantile(srt, counts, 0.95)[:, None], top.shape[1], axis=1)
    return cov, top, point, low, high, values[point] > med[:, None]


def band_exceedance(y: np.ndarray, bands: np.ndarray, context: np.ndarray, floors: np.ndarray, s: Settings):
    """Normalised band exceedance of the windows `y [n, length]` against `bands [n, length, 3]`:
    distance outside the 0.1 to 0.9 band in units of its width, the width floored at 5 % of the
    `context [n, ref_model]`'s median absolute level and at each window's `floors`. Each bin's
    largest score and its offset, and per window and bin the low, high and direction of that point."""
    lo, mid, hi = bands[..., 0], bands[..., 1], bands[..., 2]
    srt, counts = sorted_rows(np.abs(context))
    width = np.maximum(hi - lo, np.maximum(np.maximum(0.05 * row_quantile(srt, counts, 0.5), floors), 1e-9)[:, None])
    score = np.maximum(np.maximum(y - hi, lo - y), 0.0) / width
    score = np.where(np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi), score, -np.inf)
    top, at = best_per_bin(score, s.bin)
    pick = lambda a: np.take_along_axis(a, at, axis=1)  # noqa: E731
    return top, at, pick(lo), pick(hi), pick(y) > pick(mid)


def bins_from(top, point, observed, low, high, up) -> list[Bin | None]:
    """One window's bins: a `Bin` where the bin has a score, `None` where it has none."""
    return [Bin(float(top[b]), int(point[b]), float(observed[b]), float(low[b]), float(high[b]), bool(up[b]))
            if np.isfinite(top[b]) else None for b in range(len(top))]


def fires(scored: Scored, b: int, threshold: float, min_score: float) -> bool:
    """Whether bin `b` is reported: rarer than `threshold` in its own history and at least
    `min_score` from normal, so a drift in the last decimals of a flat series never qualifies."""
    rare = b < len(scored.rarity) and scored.rarity[b] >= threshold
    return rare and b < len(scored.bins) and scored.bins[b] is not None and scored.bins[b].score >= min_score


def history_starts(start: int, length: int, reference: int, s: Settings) -> list[int]:
    """Starts of the history windows: back to back before `start`, each with `reference`
    minutes before it on the grid, spanning at most `s.history` minutes."""
    out = []
    for k in range(1, s.history // length + 1):
        h = start - k * length
        if h < reference:
            break
        out.append(h)
    return out


def rarity(bins: list, null: list, min_null: int) -> list[float]:
    """Per scanned bin, the share of history bin scores below it (ties count half)."""
    srt = finite_sorted(null).tolist()
    out = []
    for b in bins:
        if b is None or len(srt) < min_null:
            out.append(float("nan"))
        else:
            below = bisect.bisect_left(srt, b.score)
            equal = bisect.bisect_right(srt, b.score) - below
            out.append((below + 0.5 * equal) / len(srt))
    return out


def scored(scan: list, null: list, s: Settings) -> Scored:
    return Scored(scan, rarity(scan, null, s.min_null), len(finite_sorted(null)), None)


def analyse(grid: Grid, start: int, length: int, s: Settings) -> list[Outcome]:
    """Robust z for the window `start..start + length` of every series of the grid."""
    outcomes = []
    for (container, metric), values in sorted(grid.series.items()):
        if start < s.ref_z:
            outcomes.append(Outcome(container, metric, Scored(skipped="no reference before the window")))
            continue
        starts = np.array([start] + history_starts(start, length, s.ref_z, s))
        cov, top, point, low, high, up = robust_z(values, starts, length, s, METRICS[metric].floor)
        if cov[0] < s.gate:
            skipped = f"{100 * cov[0]:.0f} % of the {s.ref_z} reference minutes have data"
            outcomes.append(Outcome(container, metric, Scored(skipped=skipped)))
            continue
        history = top[1:][cov[1:] >= s.gate]
        scan = bins_from(top[0], point[0], values[point[0]], low[0], high[0], up[0])
        outcomes.append(Outcome(container, metric, scored(scan, history[np.isfinite(history)].tolist(), s)))
    return outcomes


def context_gap(context: np.ndarray, s: Settings) -> str | None:
    """Why the student gets no forecast from `context`, None when it does: under `s.gate` of its
    minutes have data, or none of the last patch, the one the forecast is read from."""
    cov = coverage(context)
    if cov < s.gate:
        return f"{100 * cov:.0f} % of the {len(context)} context minutes have data"
    if not np.isfinite(context[-PATCH:]).any():
        return f"no data in the {PATCH} minutes before the window"
    return None


def aximoe_scores(grid: Grid, start: int, length: int, s: Settings, model: Model, outcomes: list[Outcome]) -> None:
    """The student's scores for every outcome, all windows of all series forecast together; a
    history window joins the calibration when `context_gap` finds none in its context."""
    patches = -(-length // PATCH)
    windows: list[tuple[int, int]] = []
    for o, out in enumerate(outcomes):
        values = grid.series[(out.container, out.metric)]
        if start < s.ref_model:
            out.aximoe = Scored(skipped="no context before the window")
            continue
        reason = context_gap(values[start - s.ref_model : start], s)
        if reason:
            out.aximoe = Scored(skipped=reason)
            continue
        windows.append((o, start))
        for h in history_starts(start, length, s.ref_model, s):
            if context_gap(values[h - s.ref_model : h], s) is None:
                windows.append((o, h))
    if not windows:
        return
    series = [grid.series[(o.container, o.metric)] for o in outcomes]
    items = [(METRICS[outcomes[o].metric].channel, series[o][w - s.ref_model : w]) for o, w in windows]
    bands = np.stack([b[:length] for b in model.forecast(items, patches)])
    y = np.stack([series[o][w : w + length] for o, w in windows])
    context = np.stack([series[o][w - s.ref_model : w] for o, w in windows])
    floors = np.array([METRICS[outcomes[o].metric].floor for o, _ in windows])
    top, at, low, high, up = band_exceedance(y, bands, context, floors, s)
    null: dict[int, list[float]] = {}
    for i, (o, w) in enumerate(windows):
        if w != start:
            null.setdefault(o, []).extend(top[i][np.isfinite(top[i])].tolist())
    for i, (o, w) in enumerate(windows):
        if w == start:
            scan = bins_from(top[i], w + at[i], np.take_along_axis(y[i], at[i], axis=0), low[i], high[i], up[i])
            outcomes[o].aximoe = scored(scan, null.get(o, []), s)


# ----------------------------------------------------------------------------- the report

def hit(sc: Scored, b: int, threshold: float, min_score: float, t0: dt.datetime) -> dict | None:
    if b >= len(sc.bins) or sc.bins[b] is None:
        return None
    x = sc.bins[b]
    return {"triggered": bool(fires(sc, b, threshold, min_score)), "rarity": float(sc.rarity[b]), "score": x.score,
            "direction": "up" if x.up else "down", "observed": x.observed,
            "observed_at": iso(t0 + dt.timedelta(minutes=x.point)), "normal_low": x.low, "normal_high": x.high}


def triggers(outcomes: list[Outcome], s: Settings, t0: dt.datetime, start: int) -> tuple[list[dict], list[dict]]:
    """The triggered bins of `outcomes`, in time order then by container, and what was skipped."""
    by_bin: dict[tuple[int, str], list[dict]] = {}
    skipped = []
    for o in outcomes:
        metric = METRICS[o.metric].channel
        for name, sc in (("robust_z", o.robust_z), ("aximoe", o.aximoe)):
            if sc is not None and sc.skipped:
                skipped.append({"container": o.container, "metric": metric, "detector": name, "reason": sc.skipped})
        n_bins = max(len(o.robust_z.bins), len(o.aximoe.bins) if o.aximoe else 0)
        for b in range(n_bins):
            z = hit(o.robust_z, b, s.threshold, s.min_z, t0)
            a = hit(o.aximoe, b, s.threshold, s.min_band, t0) if o.aximoe else None
            if (z and z["triggered"]) or (a and a["triggered"]):
                by_bin.setdefault((b, o.container), []).append({"metric": metric, "robust_z": z, "aximoe": a})
    out = []
    for (b, container), metrics in sorted(by_bin.items()):
        start_t = t0 + dt.timedelta(minutes=start + b * s.bin)
        out.append({"container": container, "bin_start": iso(start_t),
                    "bin_end": iso(start_t + dt.timedelta(minutes=s.bin)), "metrics": metrics})
    return out, skipped


def history_span(days: float) -> str:
    hours = round(days * 24)
    d = hours / 24
    return "day" if d == 1 else (f"{d:g} days" if d == int(d) else f"{hours} hours")


TOP = 10  # containers the report lists in full


def excess(h: dict, metric: Metric) -> float:
    """How far a hit's observed value lies beyond its normal range, in multiples of the range's
    nearer edge, that edge at least `metric.level`."""
    up = h["direction"] == "up"
    edge = h["normal_high"] if up else h["normal_low"]
    beyond = h["observed"] - edge if up else edge - h["observed"]
    return max(beyond, 0.0) / max(abs(edge), metric.level)


def ranking(trig: list[dict], oom: list[dict] = (), top: int = TOP) -> tuple[list[dict], int]:
    """The containers of the triggered bins and the OOM kills, most distinct metrics flagged first
    (OOM kills counting as one), then the largest excess over the normal level; the first `top`
    of them and how many more there are."""
    channels = {m.channel: m for m in METRICS}
    by: dict[str, dict] = {}
    for e in oom:
        c = by.setdefault(e["container"], {"container": e["container"], "start": e["minute"], "end": e["minute"],
                                           "metrics": [OOM_KILLS], "excess": -1.0, "largest": None})
        c["start"], c["end"] = min(c["start"], e["minute"]), max(c["end"], e["minute"])
    for t in trig:
        c = by.setdefault(t["container"], {"container": t["container"], "start": t["bin_start"], "end": t["bin_end"],
                                           "metrics": [], "excess": -1.0, "largest": None})
        c["start"], c["end"] = min(c["start"], t["bin_start"]), max(c["end"], t["bin_end"])
        for m in t["metrics"]:
            for h in (m.get("robust_z"), m.get("aximoe")):
                if not (h and h["triggered"]):
                    continue
                if m["metric"] not in c["metrics"]:
                    c["metrics"].append(m["metric"])
                e = excess(h, channels[m["metric"]])
                if e > c["excess"]:
                    c["excess"] = e
                    c["largest"] = {"metric": m["metric"], **{k: h[k] for k in
                                    ("direction", "observed", "observed_at", "normal_low", "normal_high")}}
    ranked = sorted(by.values(), key=lambda c: (len(c["metrics"]), c["excess"]), reverse=True)
    order = [m.channel for m in METRICS] + [OOM_KILLS]
    for c in ranked:
        c["metrics"].sort(key=order.index)
    return ranked[:top], max(len(ranked) - top, 0)


def text(r: dict) -> str:
    """The report as text an agent can read: a summary line, then the ranked containers with
    their flagged metrics and largest excess, then what was skipped."""
    q = r["query"]
    history = history_span(q["history_days"])
    out = (f"Scanned {r['scanned']['series']} series ({r['scanned']['containers']} containers matching "
           f"/{q['containers']}/, metrics {', '.join(r['scanned']['metrics'])}) in {q['dataset']} from "
           f"{q['window_start'][:16].replace('T', ' ')} to {q['window_end'][11:16]} UTC, 5-minute bins, flagged "
           f"when rarer than {100 * q['threshold']:.1f} % of the same series' bins over the previous {history} and "
           f"at least {q['min_robust_z']:g} robust standard deviations (robust z) or {q['min_band_widths']:g} band "
           f"widths outside the forecast band (aximoe) from normal.\n")
    student_part = (f"; aximoe {q['aximoe_run']} ({q['aximoe_device']}) forecasting the window from the "
                    f"{q['aximoe_context_minutes']} minutes before it" if q.get("aximoe_run") else "; aximoe unavailable")
    out += f"Detectors: robust z against the {q['robust_z_reference_minutes']} minutes before the window{student_part}.\n"
    if not r["triggers"]:
        out += "No bin of any scanned series was rare enough to flag.\n"
    else:
        out += (f"{len(r['triggers'])} flagged bins in {len(r['ranked']) + r['more']} containers, ranked by distinct "
                f"metrics flagged, then by the largest excess over the normal level:\n")
        for i, c in enumerate(r["ranked"], 1):
            out += f"{i}. {c['container']} {c['start'][11:16]}-{c['end'][11:16]}: {', '.join(c['metrics'])}"
            h = c["largest"]
            if h is not None:
                unit = next(x.unit for x in METRICS if x.channel == h["metric"])
                out += (f"; largest {h['metric']} {h['direction']} {format_value(unit, h['observed'])} at "
                        f"{h['observed_at'][11:16]} (normal {format_value(unit, h['normal_low'])} to "
                        f"{format_value(unit, h['normal_high'])})")
            out += "\n"
        if r["more"]:
            out += f"{r['more']} more containers have flagged bins; a narrower `containers` pattern lists them.\n"
    per: dict[str, list[dict]] = {}
    for e in r["oom_kills"]:
        per.setdefault(e["container"], []).append(e)
    listed = [f"{c} {sum(e['kills'] for e in es)} ("
              + ", ".join(e["minute"][11:16] + (f" x{e['kills']}" if e["kills"] > 1 else "") for e in es) + ")"
              for c, es in per.items()]
    if listed:
        out += f"OOM kills in the window: {'; '.join(listed)}.\n"
    if r["skipped"]:
        out += f"{len(r['skipped'])} series-detector pairs were not scored: too little data before the window.\n"
    for n in r["notes"]:
        out += f"Note: {n}\n"
    return out


# ----------------------------------------------------------------------------- the call

@dataclasses.dataclass
class State:
    model: Model | None
    model_error: str | None
    log: str
    text_only: bool


async def investigate(state: State, dataset: str, containers: str, at: str, duration_minutes: int | None = None,
                      threshold: float | None = None, deployment_name: str | None = None,
                      history_days: float | None = None) -> dict:
    """Fetches the window around `at` with the history it is calibrated against, scores it
    with both detectors and builds the report."""
    check_pattern(containers)
    try:
        at_t = dt.datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"`at` {at!r} is not an RFC 3339 time: {e}")
    if at_t.tzinfo is None:
        raise ValueError(f"`at` {at!r} has no time zone")
    duration = duration_minutes if duration_minutes is not None else 60
    if not 5 <= duration <= 240:
        raise ValueError(f"duration_minutes {duration} is outside 5 to 240")
    s = Settings()
    if threshold is not None:
        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"threshold {threshold} is outside [0, 1)")
        s.threshold = threshold
    if history_days is not None:
        if not 0.25 <= history_days <= 14.0:
            raise ValueError(f"history_days {history_days} is outside 0.25 to 14")
        s.history = round(history_days * 24 * 60)
    history_bins = s.history // duration * -(-duration // s.bin)
    if history_bins < s.min_null:
        raise ValueError(f"{s.history / 60:g} hours of history give {history_bins} history bins for a "
                         f"{duration}-minute window; at least {s.min_null} are needed")
    if not deployment_name:
        raise ValueError("no deployment named; `deployment` is a table of ~/.axiom.toml")
    dep = deployment(deployment_name)
    at_minute = dt.datetime.fromtimestamp(int(at_t.timestamp()) // 60 * 60, tz=dt.timezone.utc)
    window_start = at_minute - dt.timedelta(minutes=duration // 2)
    window_end = window_start + dt.timedelta(minutes=duration)
    lead = s.history + max(s.ref_model, s.ref_z)
    fetch_start = window_start - dt.timedelta(minutes=lead)

    t = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        grid, oom = await asyncio.gather(fetch(client, dep, dataset, containers, fetch_start, window_end),
                                         oom_kills(client, dep, dataset, containers, window_start, window_end))
    timing = {"fetch": int((time.perf_counter() - t) * 1000), "robust_z": 0, "aximoe": 0}

    def score():
        t = time.perf_counter()
        outcomes = analyse(grid, lead, duration, s)
        timing["robust_z"] = int((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        if state.model is not None:
            aximoe_scores(grid, lead, duration, s, state.model, outcomes)
        timing["aximoe"] = int((time.perf_counter() - t) * 1000)
        return outcomes

    outcomes = await asyncio.to_thread(score)
    trig, skipped = triggers(outcomes, s, grid.t0, lead)
    ranked, more = ranking(trig, oom)
    notes = []
    if state.model_error:
        notes.append(f"aximoe is not loaded: {state.model_error}")
    if not grid.series:
        notes.append("no series matched the dataset, the container pattern and the time range")
    return {
        "query": {"dataset": dataset, "deployment": dep.name, "containers": containers, "at": iso(at_t),
                  "window_start": iso(window_start), "window_end": iso(window_end), "bin_minutes": s.bin,
                  "threshold": s.threshold, "min_robust_z": s.min_z, "min_band_widths": s.min_band,
                  "robust_z_reference_minutes": s.ref_z, "aximoe_context_minutes": s.ref_model,
                  "history_days": s.history / 1440,
                  "aximoe_run": state.model.model_dir if state.model else None,
                  "aximoe_device": state.model.device_name if state.model else None},
        "scanned": {"containers": len(grid.containers()), "series": len(grid.series),
                    "metrics": [m.channel for m in METRICS]},
        "triggers": trig, "ranked": ranked, "more": more, "oom_kills": oom, "skipped": skipped, "timing_ms": timing,
        "notes": notes,
    }


def log_call(path: str, started: dt.datetime, params: dict, result: dict | None, error: str | None) -> None:
    """Appends one call to the log: when, the parameters, and the report or the error."""
    entry = {"time": iso(started), "params": params}
    entry.update({"report": result} if error is None else {"error": error})
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"aximoe-mcp: cannot log to {path}: {e}", file=__import__("sys").stderr)
