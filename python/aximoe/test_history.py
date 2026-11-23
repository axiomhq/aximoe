"""Tests of the history input: the chain of tiles before a tile, the bin means against a loop,
that a position reads nothing after its own patch, the model's passes with the input, and the
forecaster reading the steps before a window.

    python -I python/aximoe/test_history.py
"""
import os
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, forecast
from aximoe.student import CONTEXT, PATCH, POSITIONS, SEQ, Student, StudentConfig
from aximoe.test_siblings import index_of

STEPS = SEQ * PATCH
SCALES = (4, 40)


class Chain(unittest.TestCase):
    def test_before_follows_the_tiles_that_end_where_a_tile_starts(self):
        names = ["shop/web", "shop/api"]
        rows = [
            (0, 3, 0, 0, 0, 60),            # 0: the first tile of a series
            (0, 3, 0, STEPS, 0, 60),        # 1: the next
            (0, 3, 0, 2 * STEPS, 0, 60),    # 2: the next
            (0, 3, 0, 4 * STEPS, 0, 60),    # 3: after a gap
            (0, 3, 1, STEPS, 0, 60),        # 4: another series
            (0, 4, 0, 3 * STEPS, 0, 60),    # 5: another channel
            (0, 3, 0, 3 * STEPS, 0, 30),    # 6: another width
            (0, 3, 0, 3 * STEPS, 0, 60),    # 7: the same key as tile 2's successor, in the second source store
        ]
        chain = batches.History(index_of(rows, names), [7, 1])
        got = chain.before(np.array([2, 1, 0, 3, 4, 7, -1]), 3)
        self.assertEqual(got.tolist(), [[-1, 0, 1], [-1, -1, 0], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])


class Channel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.values = rng.normal(size=(3, 5 * PATCH + STEPS)).astype(np.float32).cumsum(axis=1)
        self.values[rng.random(self.values.shape) < 0.1] = np.nan
        self.values[1, : 5 * PATCH + 300] = np.nan
        tile = self.values[:, -STEPS:]
        loc, scale = batches.causal_stats(tile)
        self.loc, self.scale = loc[:, :SEQ], scale[:, :SEQ]

    def test_bin_means_against_a_loop(self):
        got = batches.history_channel(self.values, SEQ, self.loc, self.scale, SCALES).reshape(3, SEQ, len(SCALES), 2, PATCH)
        steps = self.values.shape[1]
        for i in range(3):
            for s in (0, 1, 7, 40, SEQ - 1):
                end = steps - (SEQ - 1 - s) * PATCH
                for c, r in enumerate(SCALES):
                    for b in (0, 5, PATCH - 1):
                        lo, hi = max(end - (PATCH - b) * r, 0), max(end - (PATCH - 1 - b) * r, 0)
                        held = self.values[i, lo:hi]
                        held = held[np.isfinite(held)]
                        share = len(held) / r
                        want = np.arcsinh((held.astype(np.float64).mean() - self.loc[i, s]) / max(self.scale[i, s], batches.MIN_SCALE)) if len(held) else 0.0
                        self.assertAlmostEqual(float(got[i, s, c, 0, b]), float(want), delta=2e-4)
                        self.assertAlmostEqual(float(got[i, s, c, 1, b]), share, delta=1e-6)

    def test_a_position_reads_nothing_after_its_own_patch(self):
        base = batches.history_channel(self.values, SEQ, self.loc, self.scale, SCALES)
        changed = self.values.copy()
        s = 20
        end = changed.shape[1] - (SEQ - 1 - s) * PATCH
        changed[:, end:] += 100.0
        moved = batches.history_channel(changed, SEQ, self.loc, self.scale, SCALES)
        np.testing.assert_allclose(moved[:, : s + 1], base[:, : s + 1], rtol=0, atol=1e-6)
        self.assertGreater(float(np.abs(moved[:, s + 1 :] - base[:, s + 1 :]).max()), 1e-3)

    def test_later_positions_stay_empty(self):
        got = batches.history_channel(self.values[:, : -PATCH], 30, self.loc, self.scale, SCALES)
        self.assertEqual(float(np.abs(got[:, 30:]).max()), 0.0)
        self.assertGreater(float(np.abs(got[:, :30]).max()), 0.0)


class Passes(unittest.TestCase):
    def test_cached_passes_match_a_full_pass_and_the_input_starts_silent(self):
        torch.manual_seed(0)
        base = dict(d_model=32, heads=4, layers=2, d_ff=48, band_input=True)
        model = Student(StudentConfig(**base, history_scales=SCALES)).eval()
        plain = Student(StudentConfig(**base)).eval()
        plain.load_state_dict(model.state_dict(), strict=False)
        n = 5
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        history = torch.randn(n, SEQ, len(SCALES) * 2 * PATCH)
        rows = torch.arange(n)
        at = torch.randint(0, POSITIONS - 1, (n,))
        with torch.no_grad():
            start = model.forward_full(inputs, mask, None, None, band, None, None, None, history)[0]
            torch.testing.assert_close(start, plain.forward_full(inputs, mask, None, None, band)[0], rtol=1e-6, atol=1e-6)
            torch.nn.init.normal_(model.history_in.weight, std=0.1)
            full = model.forward_full(inputs, mask, None, None, band, None, None, None, history)[0]
            self.assertGreater(float((full - start).abs().max()), 1e-3)
            # a clip holds the input's values: beyond it the forecast no longer follows them
            held = Student(StudentConfig(**base, history_scales=SCALES, history_clip=0.5)).eval()
            held.load_state_dict(model.state_dict())
            far = history * 10.0
            self.assertGreater(float((model.forward_full(inputs, mask, None, None, band, None, None, None, far)[0] - full).abs().max()), 1e-3)
            torch.testing.assert_close(held.forward_full(inputs, mask, None, None, band, None, None, None, far)[0],
                                       held.forward_full(inputs, mask, None, None, band, None, None, None, far.clamp(-0.5, 0.5))[0], rtol=1e-6, atol=1e-6)
            q, states = model.cached_pass(inputs, mask, None, band, 0, None, None, history)
            torch.testing.assert_close(q, full, rtol=1e-5, atol=1e-5)
            inputs[rows, at], mask[rows, at] = torch.randn(n, PATCH), 1.0
            history[rows, at] = torch.randn(n, len(SCALES) * 2 * PATCH)
            got = model.next_position(inputs, mask, band, None, at, states, None, history)
            want = model.forward_full(inputs, mask, None, None, band, None, None, None, history)[0][rows, at]
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


class Forecasts(unittest.TestCase):
    def test_the_steps_before_a_window_reach_a_model_with_the_input_and_no_other(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        long = rng.normal(size=(5, 3 * CONTEXT)).astype(np.float32).cumsum(axis=1) * 0.1
        long[1, : 2 * CONTEXT + 200] = np.nan
        short = long[:, -CONTEXT:]
        other = long.copy()
        other[:, : 2 * CONTEXT] += 3.0
        tags = np.zeros(5, dtype=np.int64)
        device = torch.device("cpu")
        base = dict(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, mask_tail=30)
        model = Student(StudentConfig(**base, history_scales=(8, 96))).eval()
        torch.nn.init.normal_(model.history_in.weight, std=0.1)
        plain = Student(StudentConfig(**base)).eval()
        for decoder in ("rollout", "direct", "splice:2"):
            with_past = forecast.forecast(model, long, tags, 4 * PATCH, 64, device, decoder=decoder)
            narrow = forecast.forecast(model, long, tags, 4 * PATCH, 2, device, decoder=decoder)
            np.testing.assert_allclose(narrow, with_past, rtol=1e-4, atol=1e-4)
            moved = forecast.forecast(model, other, tags, 4 * PATCH, 64, device, decoder=decoder)
            self.assertGreater(float(np.abs(moved[[0, 2, 3, 4]] - with_past[[0, 2, 3, 4]]).max()), 1e-3)
            # window 1 has no step before its last CONTEXT, so its past changes nothing
            np.testing.assert_allclose(moved[1], with_past[1], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(forecast.forecast(plain, long, tags, 4 * PATCH, 64, device, decoder=decoder),
                                       forecast.forecast(plain, short, tags, 4 * PATCH, 64, device, decoder=decoder), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
