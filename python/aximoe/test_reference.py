"""Tests for the reference scores: coverage, the persistence baseline and scoring a store,
driven by a fake teacher.

    python -m unittest discover -s python -p "test_reference.py" -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from aximoe import reference
from aximoe import store as teacher_store


def fake_tile(rng, scale=1.0):
    values = (rng.normal(size=teacher_store.TILE).cumsum() * scale).astype(np.float32)
    values[100:140] = np.nan
    return values


def fake_forecast(values):
    """Forecasts of tiles `values [B, 2048]` in a store's shapes: quantile k at position s is the
    mean of patch s plus k, loc at position s is the last step of patch s, and scale is 2."""
    b, positions, patch = len(values), teacher_store.POSITIONS, teacher_store.PATCH
    means = np.nan_to_num(values).reshape(b, -1, patch)[:, :positions].mean(-1)
    q = np.broadcast_to(means[:, None, :, None] + np.arange(9, dtype=np.float32)[None, :, None, None],
                        (b, 9, positions, patch)).copy()
    loc = np.tile(((np.arange(positions) + 1) * patch - 1).astype(np.float32), (b, 1))
    return q, loc, np.full((b, positions), 2.0, np.float32)


class Coverage(unittest.TestCase):
    def test_coverage_counts_targets_outside_the_band(self):
        rng = np.random.default_rng(1)
        values = fake_tile(rng)
        q, loc, scale = fake_forecast(values[None])
        s = reference.score_tile(values, q[0], loc[0], scale[0])
        self.assertTrue(0.0 <= s["below_q10"] <= 1.0 and 0.0 <= s["above_q90"] <= 1.0)
        wide = q[0].copy()
        wide[0] -= 1e6
        wide[-1] += 1e6
        s = reference.score_tile(values, wide, loc[0], scale[0])
        self.assertEqual(s["below_q10"], 0.0)
        self.assertEqual(s["above_q90"], 0.0)


class PersistenceBaseline(unittest.TestCase):
    def test_point_forecast_scores(self):
        rng = np.random.default_rng(2)
        values = fake_tile(rng)
        _, loc, scale = fake_forecast(values[None])
        s = reference.score_persistence(values, loc[0], scale[0])
        # a zero-width band scores 2/alpha times the distance, and a perfect forecast scores 0
        self.assertGreater(s["winkler"], 0.0)
        flat = np.full(teacher_store.TILE, 5.0, dtype=np.float32)
        s0 = reference.score_persistence(flat, np.full(teacher_store.POSITIONS, 5.0, np.float32),
                                         np.ones(teacher_store.POSITIONS, np.float32))
        self.assertEqual(s0["winkler"], 0.0)
        self.assertEqual(s0["pinball"], 0.0)


class ScoreStore(unittest.TestCase):
    def test_scores_every_tile_and_summarises_by_split(self):
        rng = np.random.default_rng(3)
        n = 6
        manifest = pd.DataFrame({
            "env": ["staging"] * n, "channel": ["cpu_usage_rate"] * 3 + ["mem_rss"] * 3,
            "series": [f"ns/c{i}" for i in range(n)], "start": [0] * n,
            "finite_frac": [1.0] * n, "bucket": ["dense"] * n,
            "split": ["train", "heldout_time", "heldout_container", "train", "heldout_time", "heldout_channel"],
        })
        values = np.stack([fake_tile(rng) for _ in range(n)])
        q, loc, scale = fake_forecast(values)
        with tempfile.TemporaryDirectory() as d:
            store = teacher_store.Store.create(d, n, "float32")
            store.write(np.arange(n), values, q, loc, scale)
            store.finish({"model": "fake"})
            scores = reference.score_store(teacher_store.Store.open(d), manifest)
        self.assertEqual(len(scores), n)
        self.assertEqual(set(scores.columns) >= {"tile", "split", "channel", "winkler", "pinball", "points",
                                                 "below_q10", "above_q90", "winkler_persistence", "pinball_persistence"}, True)
        table = reference.summarise(scores)
        self.assertEqual(set(table.index.get_level_values("split")), set(manifest.split))
        self.assertIn("winkler", table.columns)


class Root(unittest.TestCase):
    def test_main_scores_a_store_under_root(self):
        rng = np.random.default_rng(5)
        n = 4
        manifest = pd.DataFrame({
            "env": ["dev"] * n, "channel": ["mem_rss"] * n, "series": [f"ns/c{i}" for i in range(n)],
            "start": np.zeros(n, dtype=np.int32), "finite_frac": np.ones(n, dtype=np.float32),
            "bucket": ["dense"] * n, "split": ["train", "heldout_time", "heldout_container", "train"],
        })
        values = np.stack([fake_tile(rng) for _ in range(n)])
        q, loc, scale = fake_forecast(values)
        with tempfile.TemporaryDirectory() as root:
            base = os.path.join(root, "aximoe-1m")
            store = teacher_store.Store.create(os.path.join(base, "teacher-313m"), n, "float32")
            store.write(np.arange(n), values, q, loc, scale)
            store.finish({"model": "fake"})
            manifest.to_parquet(os.path.join(base, "tiles.parquet"))
            reference.main(["--data", root, "--root", "aximoe-1m", "--store", "teacher-313m"])
            self.assertEqual(sorted(os.listdir(root)), ["aximoe-1m"])
            scores = pd.read_parquet(os.path.join(base, "reference", "scores_teacher-313m.parquet"))
            self.assertTrue(os.path.exists(os.path.join(base, "reference", "summary_teacher-313m.csv")))
        self.assertEqual(len(scores), n)


if __name__ == "__main__":
    unittest.main()
