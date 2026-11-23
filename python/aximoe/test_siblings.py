"""Tests of the sibling input: the groups tiles are drawn from, the mean the trainer builds
against a loop, its agreement with the mean the forecaster builds, and the model's passes with it.

    python -I python/aximoe/test_siblings.py
"""
import os
import sys
import types
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, forecast, student
from aximoe.student import CONTEXT, PATCH, POSITIONS, SEQ

STEPS = SEQ * PATCH


def index_of(rows: list[tuple], names: list[str]) -> types.SimpleNamespace:
    """A stand-in for `batches.Index` over `rows` of (env, channel, series, start, split, width)."""
    tiles = np.zeros(len(rows), dtype=batches.RECORD)
    for i, (env, channel, series, start, split, width) in enumerate(rows):
        tiles[i]["env"], tiles[i]["channel"], tiles[i]["series"] = env, channel, series
        tiles[i]["start"], tiles[i]["split"], tiles[i]["width"] = start, split, width
    index = types.SimpleNamespace(tiles=tiles, meta={"series": names, "envs": ["prod", batches.SYNTHETIC]}, splits=["train", "heldout_time"])
    index.with_split = lambda split: np.flatnonzero(tiles["split"] == index.splits.index(split))
    return index


class FakeStore:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.loc, self.scale = batches.causal_stats(values)
        self.loc, self.scale = self.loc[:, :POSITIONS], self.scale[:, :POSITIONS]

    def rows(self, ids, names):
        return {name: getattr(self, name)[np.asarray(ids)] for name in names}


class Groups(unittest.TestCase):
    def test_picks_come_from_the_tiles_own_group_and_leave_the_tile_out(self):
        names = ["shop/web", "shop/api", "shop/db", "bank/web", batches.SYNTHETIC + "/1"]
        rows = [
            (0, 3, 0, 5, 0, 60),   # 0: shop, channel 3, start 5
            (0, 3, 1, 5, 0, 60),   # 1: same group
            (0, 3, 2, 5, 0, 60),   # 2: same group
            (0, 3, 3, 5, 0, 60),   # 3: another namespace
            (0, 3, 1, 6, 0, 60),   # 4: another start
            (0, 4, 1, 5, 0, 60),   # 5: another channel
            (0, 3, 2, 5, 1, 60),   # 6: held out
            (0, 3, 1, 5, 0, 30),   # 7: another width
            (1, 3, 4, 5, 0, 60),   # 8: synthetic
            (0, 3, 0, 5, 0, 60),   # 9: the first group's key in the second source store
            (0, 3, 1, 5, 0, 60),   # 10
        ]
        groups = batches.Siblings(index_of(rows, names), [9, 2], seed=0)
        for _ in range(20):
            picks = groups.pick(np.array([0, 1, 3, 6, 8, 9, -1]), 4)
            self.assertEqual(sorted(picks[0][picks[0] >= 0].tolist()), [1, 2])
            self.assertEqual(sorted(picks[1][picks[1] >= 0].tolist()), [0, 2])
            self.assertEqual(sorted(picks[5][picks[5] >= 0].tolist()), [10])
            for row in (2, 3, 4, 6):
                self.assertTrue((picks[row] == -1).all())
        seen = {int(groups.pick(np.array([0]), 1)[0, 0]) for _ in range(40)}
        self.assertEqual(seen, {1, 2})
        self.assertTrue((groups.pick(np.array([0, 1, 9]), 4, dropout=1.0) == -1).all())
        kept = np.mean([(groups.pick(np.array([0]), 1, dropout=0.5) >= 0).any() for _ in range(400)])
        self.assertAlmostEqual(float(kept), 0.5, delta=0.1)


class Channel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(5, STEPS)).astype(np.float32) * np.array([1.0, 3.0, 0.5, 2.0, 1.0], np.float32)[:, None]
        values[rng.random(values.shape) < 0.1] = np.nan
        self.store = FakeStore(values)
        self.inputs, self.mask, _, _ = batches.standardise(values, self.store.loc, self.store.scale)

    def test_mean_of_the_picked_tiles_against_a_loop(self):
        picks = np.array([[1, 2, 4, -1], [-1, -1, -1, -1], [0, -1, -1, -1]])
        got = batches.sibling_channel(self.store, picks)
        want = np.zeros((3, SEQ, 2 * PATCH), np.float32)
        for i, row in enumerate(picks):
            ids = [t for t in row if t >= 0]
            for s in range(SEQ):
                for j in range(PATCH):
                    held = [self.inputs[t, s, j] for t in ids if self.mask[t, s, j] > 0]
                    if held:
                        want[i, s, j] = np.mean(held)
                    want[i, s, PATCH + j] = len(held) / 4
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_the_forecaster_builds_the_same_input_from_a_whole_group(self):
        groups = np.array([0, 0, 0, 1, 1])
        picks = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1], [4, -1, -1], [3, -1, -1]])
        trained = batches.sibling_channel(self.store, picks)
        served = forecast.sibling_inputs(self.inputs, self.mask, groups, 3)
        np.testing.assert_allclose(served, trained, rtol=1e-5, atol=1e-5)


class Generated(unittest.TestCase):
    def test_members_share_the_shape_and_the_batch_reads_their_mean(self):
        from aximoe import synthetic
        rng = np.random.default_rng(0)
        t = np.arange(STEPS)
        shared = np.stack([np.sin(2 * np.pi * t / 50) * 4 + 100, np.full(STEPS, 7.0)]).astype(np.float32)
        shared[0, 300:340] = np.nan
        group = synthetic.members(rng, shared, 5)
        self.assertEqual(group.shape, (2, 5, STEPS))
        self.assertTrue(np.isnan(group[0, :, 300:340]).all() and np.isfinite(group[0, :, :300]).all())
        held = np.isfinite(shared[0])
        correlations = [np.corrcoef(group[0, i, held], shared[0, held])[0, 1] for i in range(5)]
        self.assertGreater(min(correlations), 0.5)
        index = types.SimpleNamespace(channels=[batches.SYNTHETIC + "_x"], meta={"widths": [60]})
        tags = types.SimpleNamespace(resolution_ids=lambda w: np.zeros(len(w), dtype=np.int64), unknown=lambda: 0,
                                     resolution_unknown=lambda: 0)
        batch = batches.synthetic_batch(group[:, 0], np.array([60, 60]), tags, torch.float16, None, raw=True, sibling=True,
                                        others=group[:, 1:])
        flat = group[:, 1:].reshape(8, STEPS)
        loc, scale = batches.causal_stats(flat)
        inputs, mask, _, _ = batches.standardise(flat, loc[:, :POSITIONS], scale[:, :POSITIONS])
        inputs, mask = inputs.reshape(2, 4, SEQ, PATCH), mask.reshape(2, 4, SEQ, PATCH)
        want = (inputs * mask).sum(1) / np.maximum(mask.sum(1), 1.0)
        np.testing.assert_allclose(batch["sibling"][:, :, :PATCH].numpy(), want, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(batch["sibling"][:, :, PATCH:].numpy(), mask.sum(1) / 4, rtol=0, atol=1e-6)
        none = batches.synthetic_batch(group[:, 0], np.array([60, 60]), tags, torch.float16, None, raw=True, sibling=True,
                                       others=group[:, 1:], alone=(1.0, np.random.default_rng(0)))
        self.assertEqual(float(none["sibling"].abs().max()), 0.0)


class Passes(unittest.TestCase):
    def test_cached_pass_and_forecasts_read_the_input(self):
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, mask_tail=30, sibling_input=4)
        model = student.Student(cfg).eval()
        n = 6
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        sibling = torch.randn(n, SEQ, 2 * PATCH)
        with torch.no_grad():
            full = model.forward_full(inputs * mask, mask, None, None, None, None, sibling)[0]
            q, _ = model.cached_pass(inputs * mask, mask, None, None, 0, sibling)
            torch.testing.assert_close(q, full, rtol=1e-5, atol=1e-5)
            self.assertGreater(float((full - model.forward_full(inputs * mask, mask)[0]).abs().max()), 1e-3)
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(n, CONTEXT)).astype(np.float32)
        tags = np.zeros(n, dtype=np.int64)
        groups = np.array([0, 0, 0, 1, 1, 1])
        device = torch.device("cpu")
        horizon = 3 * PATCH
        alone = forecast.forecast(model, windows, tags, horizon, 4, device, floor=False, decoder="splice:1")
        together = forecast.forecast(model, windows, tags, horizon, 4, device, floor=False, decoder="splice:1", groups=groups)
        self.assertGreater(float(np.abs(together[:, :, :PATCH] - alone[:, :, :PATCH]).max()), 1e-4)
        self.assertGreater(float(np.abs(together[:, :, PATCH:] - alone[:, :, PATCH:]).max()), 1e-4)
        # the direct pass with the input built by hand from the group
        _, tile_inputs, tile_mask, _, _, _ = forecast.ahead_tiles(windows, 3)
        by_hand = forecast.sibling_inputs(tile_inputs, tile_mask, groups, 4)
        q, loc, scale = forecast.ahead_patches(model, windows, tags, device, 3, by_hand)
        raw = loc[:, None, None] + scale[:, None, None] * np.sinh(q.astype(np.float64))
        np.testing.assert_allclose(together[:, :, PATCH:], raw[:, :, PATCH:], rtol=1e-4, atol=1e-4)
        # a model widened for the input computes what it did before
        narrow = student.Student(student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, mask_tail=30)).eval()
        wide = student.widen(narrow, student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, mask_tail=30,
                                                           sibling_input=4)).eval()
        with torch.no_grad():
            torch.testing.assert_close(wide.forward_full(inputs * mask, mask, None, None, None, None, sibling)[0],
                                       narrow.forward_full(inputs * mask, mask)[0], rtol=1e-6, atol=1e-6)
        # a group of one reads nothing
        single = forecast.forecast(model, windows, tags, horizon, 4, device, floor=False, decoder="splice:1", groups=np.arange(n))
        np.testing.assert_allclose(single, alone, rtol=1e-5, atol=1e-5)


class Nearest(unittest.TestCase):
    def test_the_nearest_windows_are_those_that_correlate_most(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(2, SEQ, PATCH)).astype(np.float32)
        noise = rng.normal(size=(6, SEQ, PATCH)).astype(np.float32)
        # windows 0 and 1 follow one shape, 2 and 3 another, 4 and 5 are unrelated
        inputs = np.stack([base[0], base[0] + 0.1 * noise[1], base[1], base[1] + 0.1 * noise[3], noise[4], noise[5]])
        mask = np.ones_like(inputs)
        mask[1, :3] = 0.0
        groups = np.zeros(6, dtype=np.int64)
        near = forecast.sibling_inputs(inputs, mask, groups, 4, nearest=1)
        np.testing.assert_allclose(near[0, 3:, :PATCH], inputs[1, 3:], atol=1e-6)
        np.testing.assert_allclose(near[0, :3], 0.0, atol=1e-6)
        np.testing.assert_allclose(near[2, :, :PATCH], inputs[3], atol=1e-6)
        np.testing.assert_allclose(near[3, :, :PATCH], inputs[2], atol=1e-6)
        np.testing.assert_allclose(near[2, :, PATCH:], 0.25, atol=1e-6)
        # as many nearest as there are others is the mean of them all
        every = forecast.sibling_inputs(inputs, mask, groups, 4)
        np.testing.assert_allclose(forecast.sibling_inputs(inputs, mask, groups, 4, nearest=5), every, atol=1e-6)
        # the choice reads the known patches only
        changed = inputs.copy()
        changed[1, SEQ - 8 :] = -inputs[0, SEQ - 8 :] * 50.0
        late = forecast.sibling_inputs(changed, mask, groups, 4, nearest=1, known=SEQ - 8)
        np.testing.assert_allclose(late[0, 3 : SEQ - 8, :PATCH], inputs[1, 3 : SEQ - 8], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
