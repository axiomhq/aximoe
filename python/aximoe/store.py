"""The teacher store: flat little-endian arrays of tiles and their teacher forecasts, indexed by
the tile's row in tiles.parquet, with their reader and writer."""
from __future__ import annotations

import json
import math
import os

import numpy as np


TILE = 2048
PATCH = 32
QUANTILES = 9
POSITIONS = TILE // PATCH - 1
SUFFIX = {"float16": "f16", "float32": "f32"}
ARRAYS = ("values", "quantiles", "loc", "scale")


def read_rows(plan: list) -> None:
    """Reads each (fd, offset, buffer) of `plan`: every row's WILLNEED first, so the kernel
    fetches all rows outside the page cache at once, then each row's `preadv`."""
    for fd, offset, buf in plan:
        os.posix_fadvise(fd, offset, len(buf), os.POSIX_FADV_WILLNEED)
    for fd, offset, buf in plan:
        got = os.preadv(fd, [buf], offset)
        if got != len(buf):
            raise IOError(f"read {got} of {len(buf)} bytes at offset {offset}")


class Spanned:
    """One array of a linked store: a row index resolves to a source store's array by offset,
    so a read gathers from the sources and returns a plain array."""

    def __init__(self, sources: list, name: str, offsets: np.ndarray, n: int):
        self.sources, self.name, self.offsets = sources, name, np.asarray(offsets, dtype=np.int64)
        self.arrays = [getattr(s, name) for s in sources]
        self.shape = (n,) + tuple(self.arrays[0].shape[1:])
        self.dtype = self.arrays[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        head, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        single = isinstance(head, (int, np.integer))
        if isinstance(head, slice):
            rows = np.arange(*head.indices(self.shape[0]))
        else:
            rows = np.atleast_1d(np.asarray(head, dtype=np.int64))
        rows = np.where(rows < 0, rows + self.shape[0], rows)
        source = np.searchsorted(self.offsets, rows, side="right") - 1
        out = np.empty((len(rows),) + self.shape[1:], self.dtype)
        for s in np.unique(source):
            here = source == s
            out[here] = self.arrays[s][rows[here] - self.offsets[s]]
        if single:
            out = out[0]
            return out[rest] if rest else out
        return out[(slice(None),) + tuple(rest)] if rest else out

    def plan(self, ids: np.ndarray, out: np.ndarray, slots: np.ndarray) -> list:
        """The reads of rows `ids` into rows `slots` of `out`, each from its source store."""
        ids = np.where(ids < 0, ids + self.shape[0], ids)
        source = np.searchsorted(self.offsets, ids, side="right") - 1
        plan = []
        for s in np.unique(source):
            here = source == s
            plan += self.sources[s].plan(self.name, ids[here] - self.offsets[s], out, slots[here])
        return plan


class Store:
    """The four memory-mapped arrays of a teacher store, plus its manifest. A linked store holds
    no arrays of its own: its manifest names source stores and every read goes through to them."""

    def __init__(self, directory: str, n: int, quantile_dtype: str, mode: str, meta: dict | None = None):
        self.directory, self.n, self.quantile_dtype, self.meta = directory, n, quantile_dtype, meta or {}
        shapes = self.shapes(n)
        self.values = np.memmap(os.path.join(directory, "values.f32"), np.float32, mode, shape=shapes["values"])
        self.quantiles = np.memmap(os.path.join(directory, f"quantiles.{SUFFIX[quantile_dtype]}"),
                                   np.dtype(quantile_dtype), mode, shape=shapes["quantiles"])
        self.loc = np.memmap(os.path.join(directory, "loc.f32"), np.float32, mode, shape=shapes["loc"])
        self.scale = np.memmap(os.path.join(directory, "scale.f32"), np.float32, mode, shape=shapes["scale"])
        self.fds: dict[str, int] = {}
        for name in ARRAYS:
            fd = os.open(getattr(self, name).filename, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self.fds[name] = fd

    def plan(self, name: str, ids: np.ndarray, out: np.ndarray, slots: np.ndarray) -> list:
        """(fd, offset, buffer) for each row `ids[k]` of array `name`, into row `slots[k]` of `out`."""
        array = getattr(self, name)
        if isinstance(array, Spanned):
            return array.plan(ids, out, slots)
        if len(ids) == 0:
            return []
        size = out.itemsize * math.prod(out.shape[1:])
        flat = memoryview(out).cast("B")
        return [(self.fds[name], int(i) * size, flat[k * size:(k + 1) * size]) for i, k in zip(ids, slots)]

    def rows(self, ids, names: tuple = ARRAYS) -> dict[str, np.ndarray]:
        """Rows `ids` of the arrays `names`, in that order, as plain arrays: one fetch of every
        row outside the page cache, then a `preadv` per row, so a miss costs one read of the
        row's own size."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        slots = np.arange(len(ids))
        out, plan = {}, []
        for name in names:
            array = getattr(self, name)
            out[name] = np.empty((len(ids),) + array.shape[1:], array.dtype)
            plan += self.plan(name, ids, out[name], slots)
        read_rows(plan)
        return out

    @staticmethod
    def shapes(n: int) -> dict:
        return {"values": (n, TILE), "quantiles": (n, QUANTILES, POSITIONS, PATCH),
                "loc": (n, POSITIONS), "scale": (n, POSITIONS)}

    @classmethod
    def create(cls, directory: str, n: int, quantile_dtype: str = "float16") -> "Store":
        os.makedirs(directory, exist_ok=True)
        marker = os.path.join(directory, "manifest.json")
        if os.path.exists(marker):
            os.remove(marker)
        return cls(directory, n, quantile_dtype, "w+")

    @classmethod
    def open(cls, directory: str) -> "Store":
        with open(os.path.join(directory, "manifest.json")) as f:
            meta = json.load(f)
        if meta.get("arrays") == "linked":
            return cls.linked(directory, meta)
        return cls(directory, meta["tiles"], meta["quantile_dtype"], "r", meta)

    @classmethod
    def linked(cls, directory: str, meta: dict) -> "Store":
        """A store whose rows read through to the source stores its manifest names, in that order;
        each source root is resolved against the store's parent directory."""
        store = cls.__new__(cls)
        store.directory, store.n, store.quantile_dtype, store.meta = directory, meta["tiles"], meta["quantile_dtype"], meta
        store.fds = {}
        parent = os.path.dirname(os.path.abspath(directory))
        sources = [cls.open(os.path.join(parent, s["root"], s["store"])) for s in meta["sources"]]
        offsets = np.cumsum([0] + [s.n for s in sources])
        if offsets[-1] != store.n:
            raise ValueError(f"{directory}: the sources hold {offsets[-1]} tiles, the manifest says {store.n}")
        for name in ARRAYS:
            setattr(store, name, Spanned(sources, name, offsets[:-1], store.n))
        return store

    @staticmethod
    def is_complete(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "manifest.json"))

    def write(self, ids: np.ndarray, values, quantiles, loc, scale):
        self.values[ids] = values
        self.quantiles[ids] = quantiles.astype(self.quantile_dtype)
        self.loc[ids] = loc
        self.scale[ids] = scale

    def finish(self, meta: dict):
        for arr in (self.values, self.quantiles, self.loc, self.scale):
            arr.flush()
        self.meta = dict(meta, tiles=self.n, quantile_dtype=self.quantile_dtype, patch=PATCH,
                         positions=POSITIONS, quantiles=QUANTILES, tile=TILE, byte_order="little",
                         shapes={k: list(v) for k, v in self.shapes(self.n).items()},
                         files={"values": "values.f32", "quantiles": f"quantiles.{SUFFIX[self.quantile_dtype]}",
                                "loc": "loc.f32", "scale": "scale.f32"})
        tmp = os.path.join(self.directory, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=1)
        os.replace(tmp, os.path.join(self.directory, "manifest.json"))
