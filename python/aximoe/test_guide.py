"""Tests of the coarse guide: the bins against a loop, the statistics against the batch code,
what each position holds, that a tile's guide reads nothing after its context, the model's
passes with the input, and the forecaster's two passes.

    python -I python/aximoe/test_guide.py
"""
import os
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, forecast, guide, train
from aximoe.student import CONTEXT, GUIDE, PATCH, POSITIONS, SEQ, TAG_BASE, Student, StudentConfig

WIDTHS = [10, 30, 60, 300, 900, 1800, 3600]


def small(**keys) -> Student:
    torch.manual_seed(0)
    cfg = StudentConfig(d_model=32, heads=2, layers=2, d_ff=48, resolutions=len(WIDTHS) + 1, resolution_tags=True,
                        band_input=True, mask_tail=30, guide=guide.rows(WIDTHS), **keys)
    model = Student(cfg).eval()
    with torch.no_grad():
        model.history_in.weight.normal_(std=0.2)
    return model


def tag(seconds: int | None) -> int:
    return ((WIDTHS.index(seconds) if seconds in WIDTHS else len(WIDTHS)) + 1) * TAG_BASE


class Bins(unittest.TestCase):
    def test_bins_are_the_means_of_whole_bins_before_the_end(self):
        rng = np.random.default_rng(0)
        series = rng.normal(size=(3, 500)).astype(np.float32)
        series[rng.random(series.shape) < 0.2] = np.nan
        series[0, 470:482] = np.nan
        end, k, steps = np.array([500, 437, 300]), np.array([6, 5, 0]), 90
        got = guide.bins(torch.from_numpy(series), torch.from_numpy(end), torch.from_numpy(k), steps).numpy()
        want = np.full((3, steps), np.nan, np.float32)
        for row in range(3):
            for c in range(steps):
                first = end[row] - (steps - c) * k[row]
                if k[row] and first >= 0:
                    held = series[row, first : first + k[row]]
                    if np.isfinite(held).any():
                        want[row, c] = np.nanmean(held)
        self.assertTrue(np.array_equal(np.isfinite(got), np.isfinite(want)))
        self.assertLess(np.nanmax(np.abs(got - want)), 1e-5)
        # a bin without a present step, a bin before the series and a row without a coarser width are missing
        self.assertTrue(np.isnan(got[0, steps - 4]) and np.isnan(got[1, 0]) and np.isnan(got[2]).all())

    def test_statistics_agree_with_the_batch_code(self):
        rng = np.random.default_rng(1)
        values = (rng.normal(size=(4, CONTEXT)) * [[1.0], [1e3], [1e-3], [5.0]] + [[0.0], [1e6], [3.0], [-40.0]]).astype(np.float32)
        values[0, :700] = np.nan
        values[3, rng.random(CONTEXT) < 0.5] = np.nan
        loc, scale = guide.causal_stats(torch.from_numpy(values))
        want_loc, want_scale = batches.causal_stats(values)
        self.assertLess(np.abs(loc.numpy() - want_loc).max() / np.abs(want_loc).max(), 1e-6)
        self.assertLess(np.abs(scale.numpy() / want_scale - 1.0).max(), 1e-4)

    def test_plan_names_the_coarser_width_of_each_tag(self):
        tags = torch.tensor([tag(10), tag(60), tag(3600), tag(None), 5 + tag(300)])
        k, coarse = guide.plan(guide.rows(WIDTHS), tags)
        self.assertEqual(k.tolist(), [6, 5, 0, 0, 6])
        self.assertEqual(coarse.tolist(), [tag(60), tag(300), tag(3600), tag(None), 5 + tag(1800)])


class Channel(unittest.TestCase):
    def test_a_position_holds_the_coarse_levels_of_the_patch_it_forecasts(self):
        rng = np.random.default_rng(2)
        n, held = 4, 70
        levels = rng.normal(size=(n, 3, held)).astype(np.float32) * 3 + 10
        k, cut = np.array([6, 2, 0, 1]), np.array([62, 55, 60, 40])
        frozen, share = np.array([False, True, False, True]), np.array([0.5, 1.0, 0.3, 0.8], np.float32)
        loc = rng.normal(size=(n, POSITIONS)).astype(np.float32) + 9
        scale = rng.uniform(0.5, 4.0, size=(n, POSITIONS)).astype(np.float32)
        got = guide.channel(*(torch.from_numpy(v) for v in (levels, share, k, cut, loc, scale, frozen))).numpy()
        want = np.zeros((n, SEQ, GUIDE), np.float32)
        for row in range(n):
            for p in range(cut[row], POSITIONS):
                at = ((p - cut[row]) * PATCH + np.arange(PATCH)) // max(k[row], 1)
                if k[row] == 0 or at.max() >= held:
                    continue
                s = cut[row] if frozen[row] else p
                a = np.arcsinh((levels[row][:, at] - loc[row, s]) / scale[row, s])
                want[row, p] = np.concatenate([a[1], a[0] - a[1], a[2] - a[1], [1.0, share[row]]])
        self.assertLess(np.abs(got - want).max(), 1e-5)
        self.assertLess(float(np.abs(got[:, SEQ - 1]).max()), 1e-12)
        # the last tile's levels end after 70 steps: its third forecast patch gets no guide
        self.assertLess(float(np.abs(got[3, 40:44, 3 * PATCH] - np.array([1.0, 1.0, 0.0, 0.0])).max()), 1e-12)


class Passes(unittest.TestCase):
    def inputs(self, n=3):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(n, SEQ, PATCH, generator=g)
        m = torch.ones(n, SEQ, PATCH)
        band = torch.zeros(n, SEQ, 2 * PATCH)
        side = torch.randn(n, SEQ, GUIDE, generator=g)
        tags = torch.tensor([tag(10), tag(60), tag(300)][:n])
        return x, m, band, side, tags

    def test_the_guide_is_silent_at_zero_weights_and_read_at_its_position_only(self):
        model = small()
        x, m, band, side, tags = self.inputs()
        with torch.no_grad():
            plain = model(x, m, tags, band)
            moved = model(x, m, tags, band, None, None, None, side)
            self.assertGreater(float((moved - plain).abs().max()), 1e-3)
            later = side.clone()
            later[:, 40:] = 0.0
            # a position's output reads the guide at itself and before, never after
            self.assertLess(float((model(x, m, tags, band, None, None, None, later)[:, :40] - moved[:, :40]).abs().max()), 1e-6)
            model.history_in.weight.zero_()
            self.assertLess(float((model(x, m, tags, band, None, None, None, side) - plain).abs().max()), 1e-6)

    def test_cached_and_single_position_passes_equal_a_full_pass(self):
        model = small()
        x, m, band, side, tags = self.inputs()
        with torch.no_grad():
            full = model(x, m, tags, band, None, None, None, side)
            got, states = model.cached_pass(x, m, tags, band, 0, None, None, side)
            self.assertLess(float((got - full).abs().max()), 1e-5)
            p = torch.tensor([62, 50, 31])
            q = model.next_position(x, m, band, tags, p, states, None, side)
            self.assertLess(float((q - full[torch.arange(3), p]).abs().max()), 1e-5)


class Training(unittest.TestCase):
    def batch(self, n=4):
        rng = np.random.default_rng(4)
        values = np.cumsum(rng.normal(size=(n, SEQ * PATCH)), axis=1).astype(np.float32)
        past = np.cumsum(rng.normal(size=(n, 2 * SEQ * PATCH)), axis=1).astype(np.float32)
        past[1] = np.nan
        loc, scale = batches.causal_stats(values)
        inputs, mask, _, _ = batches.standardise(values, loc[:, :POSITIONS], scale[:, :POSITIONS])
        tags = torch.tensor([tag(10), tag(60), tag(3600), tag(1800)])
        return {"values": torch.from_numpy(values), "past": torch.from_numpy(past), "tags": tags, "inputs": torch.from_numpy(inputs),
                "input_mask": torch.from_numpy(mask), "loc": torch.from_numpy(loc[:, :POSITIONS]), "scale": torch.from_numpy(scale[:, :POSITIONS])}

    def test_a_tiles_guide_reads_nothing_after_its_context_end(self):
        model = small()
        batch = self.batch()
        cut = torch.tensor([62, 50, 55, 40])
        frozen = torch.tensor([False, True, False, True])
        kept = torch.ones(4, dtype=torch.bool)
        first = train.guide_input(model.cached_pass, model.next_position, batch, cut, frozen, model.cfg.guide, 4, kept)
        self.assertEqual(tuple(first.shape), (4, SEQ, GUIDE))
        # the hourly tile has no coarser width; the others are guided from their context end on
        self.assertLess(float(first[2].abs().max()), 1e-12)
        for row in (0, 1, 3):
            self.assertLess(float(first[row, : int(cut[row])].abs().max()), 1e-12)
            self.assertLess(abs(float(first[row, int(cut[row]), 3 * PATCH]) - 1.0), 1e-12)
        after = dict(batch)
        values = batch["values"].clone()
        for row in range(4):
            values[row, (int(cut[row]) + 1) * PATCH :] += 50.0
        after["values"] = values
        second = train.guide_input(model.cached_pass, model.next_position, after, cut, frozen, model.cfg.guide, 4, kept)
        self.assertLess(float((second - first).abs().max()), 1e-6)
        before = dict(batch)
        values = batch["values"].clone()
        values[:, 900:1000] += 5.0
        before["values"] = values
        third = train.guide_input(model.cached_pass, model.next_position, before, cut, frozen, model.cfg.guide, 4, kept)
        self.assertGreater(float((third[0] - first[0]).abs().max()), 1e-3)
        # a tile that is dropped gets none
        none = train.guide_input(model.cached_pass, model.next_position, batch, cut, frozen, model.cfg.guide, 4, torch.tensor([False, True, True, True]))
        self.assertLess(float(none[0].abs().max()), 1e-12)

    def test_the_unroll_reads_the_guide_and_takes_given_depths(self):
        model = small()
        batch = self.batch()
        j = torch.tensor([0, 3, 2, 1])
        batch["guide"] = train.guide_input(model.cached_pass, model.next_position, batch, POSITIONS - 1 - j, torch.zeros(4, dtype=torch.bool),
                                           model.cfg.guide, 2, torch.ones(4, dtype=torch.bool))
        g = torch.Generator().manual_seed(0)
        with_guide = train.unroll(model.cached_pass, model.next_position, batch, 3, g, None, 0.0, j)
        self.assertEqual(with_guide["context_end"].tolist(), (POSITIONS - 1 - j).tolist())
        without = train.unroll(model.cached_pass, model.next_position, {k: v for k, v in batch.items() if k != "guide"}, 3, g, None, 0.0, j)
        # the tiles with a guide and a patch fed back are continued differently, the others alike
        self.assertGreater(float((with_guide["inputs"][1] - without["inputs"][1]).abs().max()), 1e-4)
        self.assertLess(float((with_guide["inputs"][2] - without["inputs"][2]).abs().max()), 1e-6)
        self.assertLess(float((with_guide["inputs"][0] - without["inputs"][0]).abs().max()), 1e-6)


class Forecaster(unittest.TestCase):
    def test_a_guided_forecast_runs_with_every_decoder_at_any_pass_size(self):
        model = small()
        rng = np.random.default_rng(5)
        t = np.arange(4 * CONTEXT)
        windows = (np.sin(2 * np.pi * t / 700.0)[None] * rng.uniform(1, 3, size=(5, 1)) + rng.normal(scale=0.2, size=(5, len(t))) + 5).astype(np.float32)
        tags = np.array([tag(10), tag(10), tag(60), tag(None), tag(3600)])
        device = torch.device("cpu")
        for decoder in ("rollout", "direct", "splice:2"):
            a = forecast.forecast(model, windows, tags, 100, 2, device, decoder=decoder)
            b = forecast.forecast(model, windows, tags, 100, 64, device, decoder=decoder)
            self.assertLess(float(np.abs(a - b).max()), 1e-3, decoder)
            plain = forecast.forecast(model, windows, tags, 100, 64, device, decoder=decoder, coarse=False)
            # windows with a coarser width read their guide, the others forecast as without one
            self.assertGreater(float(np.abs(a[:3] - plain[:3]).max()), 1e-3, decoder)
            self.assertLess(float(np.abs(a[3:] - plain[3:]).max()), 1e-5, decoder)
        short = forecast.forecast(model, windows[:, -CONTEXT:], tags, 100, 64, device, decoder="rollout")
        long = forecast.forecast(model, windows, tags, 100, 64, device, decoder="rollout")
        # the coarse forecast reads the steps before the window
        self.assertGreater(float(np.abs(short[:3] - long[:3]).max()), 1e-3)

    def test_the_coarse_levels_cover_the_horizon_in_whole_patches(self):
        model = small()
        rng = np.random.default_rng(6)
        windows = rng.normal(size=(3, CONTEXT)).astype(np.float32)
        levels, share, has = forecast.coarse_levels(model, windows, np.array([tag(10), tag(3600), tag(1800)]), 50, 8, torch.device("cpu"))
        self.assertEqual(levels.shape, (3, 3, 64))
        self.assertEqual(has.tolist(), [True, False, True])
        self.assertTrue(np.isfinite(levels[0]).all() and np.isnan(levels[1]).all())
        # six fine steps share a coarse bin's level
        self.assertLess(float(np.abs(np.diff(levels[0, 1, :6])).max()), 1e-12)
        self.assertLess(abs(float(share[0]) - (CONTEXT // 6) / CONTEXT), 1e-6)


if __name__ == "__main__":
    unittest.main()
