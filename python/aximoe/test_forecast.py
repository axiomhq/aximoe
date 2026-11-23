"""Tests of the forecaster: the floor at the raw zero, the sample paths, the head of several
patches, the band input, the widened band, the windows left unanswered, a window shorter than
the model's and the patch width of a run from the command line.

    python -m unittest discover -s python -p "test_forecast.py" -v
    python -I python/aximoe/test_forecast.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import forecast, student
from aximoe.student import CONTEXT, PATCH


class Floor(unittest.TestCase):
    def test_floor_clips_only_windows_without_a_negative_value(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64)).eval()
        rng = np.random.default_rng(0)
        zeros = np.zeros(CONTEXT, np.float32)
        spikes = np.where(rng.random(CONTEXT) < 0.02, rng.exponential(3.0, CONTEXT), 0.0).astype(np.float32)
        spikes[:100] = np.nan
        signed = rng.normal(size=CONTEXT).astype(np.float32)
        windows = np.stack([zeros, spikes, signed])
        tags = np.zeros(len(windows), np.int64)
        args = (model, windows, tags, 2 * PATCH, 2, torch.device("cpu"))
        plain = forecast.forecast(*args, floor=False)
        floored = forecast.forecast(*args)
        # the untrained model forecasts below zero where the floor has to act
        self.assertTrue((plain[:2] < 0).any())
        self.assertTrue((floored[:2] >= 0).all())
        np.testing.assert_array_equal(floored[:2, :, :PATCH], np.maximum(plain[:2, :, :PATCH], 0.0))
        np.testing.assert_array_equal(floored[2], plain[2])


class Paths(unittest.TestCase):
    def test_mixture_of_identical_functions_is_the_function(self):
        torch.manual_seed(0)
        knots = torch.sort(torch.randn(5, 1, 15), dim=2).values.expand(5, 4, 15).contiguous()
        levels = torch.tensor(student.LEVELS)
        torch.testing.assert_close(forecast.mixture_quantiles(knots, levels), knots[:, 0], atol=1e-5, rtol=0)

    def test_mixture_of_two_shifted_uniforms(self):
        # uniform on [0, 1] beside uniform on [1, 2]: within the levels the mixture's distribution
        # function is 0.5 x + 0.005 on the first and 0.4995 + 0.5 (x - 1) on the second
        levels = torch.tensor(student.LEVELS, dtype=torch.float64)
        knots = torch.stack([levels, 1.0 + levels])[None]
        mixed = forecast.mixture_quantiles(knots, levels)[0]
        at = {0.1: 0.19, 0.5: 0.99, 0.9: 1.801}
        for level, value in at.items():
            self.assertAlmostEqual(float(mixed[student.LEVELS.index(level)]), value, places=6)

    def test_paths_change_only_the_recursive_steps(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64)).eval()
        windows = np.random.default_rng(0).normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(len(windows), np.int64)
        args = (model, windows, tags, 2 * PATCH, 2, torch.device("cpu"))
        median = forecast.forecast(*args, floor=False)
        sampled = forecast.forecast(*args, floor=False, paths=3)
        np.testing.assert_allclose(sampled[:, :, :PATCH], median[:, :, :PATCH], rtol=1e-5, atol=1e-5)
        self.assertTrue((np.diff(sampled, axis=1) >= -1e-6).all())
        self.assertFalse(np.allclose(sampled[:, :, PATCH:], median[:, :, PATCH:]))
        hybrid = forecast.forecast(*args, floor=False, paths=3, hybrid=True)
        np.testing.assert_allclose(hybrid[:, student.MEDIAN], median[:, student.MEDIAN], rtol=1e-5, atol=1e-5)
        self.assertTrue((np.diff(hybrid, axis=1) >= -1e-6).all())
        self.assertFalse(np.allclose(hybrid[:, :, PATCH:], median[:, :, PATCH:]))


class HorizonPatches(unittest.TestCase):
    def test_one_pass_gives_every_patch_of_the_head(self):
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, horizon_patches=2)
        model = student.Student(cfg).eval()
        windows = np.random.default_rng(0).normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(len(windows), np.int64)
        out = forecast.forecast(model, windows, tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        q, loc, scale = forecast.next_patch(model, windows, tags, torch.device("cpu"))
        self.assertEqual(q.shape, (3, len(student.LEVELS), 2 * PATCH))
        direct = loc[:, None, None] + scale[:, None, None] * np.sinh(q.astype(np.float64))
        np.testing.assert_allclose(out[:, :, : 2 * PATCH], direct, rtol=1e-5, atol=1e-5)
        # the first patch of the head is what the model's one-patch forward gives
        x, m = forecast.batches.standardise(np.concatenate([windows, np.full((3, PATCH), np.nan, np.float32)], 1),
                                            *forecast.causal_stats(windows))[:2]
        with torch.no_grad():
            first = model(torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(tags))[:, forecast.LAST].numpy()
        np.testing.assert_allclose(q[:, :, :PATCH], first, rtol=0, atol=1e-6)
        # the third patch comes from a second pass whose context ends with the two medians
        fed = np.concatenate([windows[:, 2 * PATCH :], out[:, student.MEDIAN, : 2 * PATCH]], axis=1)
        q2, loc2, scale2 = forecast.next_patch(model, fed, tags, torch.device("cpu"))
        second = loc2[:, None, None] + scale2[:, None, None] * np.sinh(q2[:, :, :PATCH].astype(np.float64))
        np.testing.assert_allclose(out[:, :, 2 * PATCH :], second, rtol=1e-5, atol=1e-5)


class BandInput(unittest.TestCase):
    def test_each_forecast_patch_feeds_its_band_back(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True)).eval()
        windows = np.random.default_rng(0).normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(len(windows), np.int64)
        cpu = torch.device("cpu")
        out = forecast.forecast(model, windows, tags, 2 * PATCH, 2, cpu, floor=False)
        q, loc, scale = forecast.next_patch(model, windows, tags, cpu)
        first = loc[:, None, None] + scale[:, None, None] * np.sinh(q.astype(np.float64))
        np.testing.assert_allclose(out[:, :, :PATCH], first, rtol=1e-5, atol=1e-5)
        fed = np.concatenate([windows[:, PATCH:], out[:, student.MEDIAN, :PATCH]], axis=1)
        lo = np.full_like(fed, np.nan)
        hi = np.full_like(fed, np.nan)
        lo[:, -PATCH:] = out[:, student.LEVELS.index(0.1), :PATCH]
        hi[:, -PATCH:] = out[:, student.LEVELS.index(0.9), :PATCH]
        q2, loc2, scale2 = forecast.next_patch(model, fed, tags, cpu, band=(lo, hi))
        second = loc2[:, None, None] + scale2[:, None, None] * np.sinh(q2.astype(np.float64))
        np.testing.assert_allclose(out[:, :, PATCH:], second, rtol=1e-5, atol=1e-5)
        # without the band the second patch differs: the band reaches the model
        q0, _, _ = forecast.next_patch(model, fed, tags, cpu)
        self.assertFalse(np.allclose(q0, q2))


class Frame(unittest.TestCase):
    def test_the_context_frame_holds_once_the_window_has_turned_constant(self):
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True, frozen_feedback=True)
        model = student.Student(cfg).eval()
        windows = np.full((2, CONTEXT), 0.5, np.float32)
        windows[:, 3] = 40.0
        tags = np.zeros(len(windows), np.int64)
        cpu = torch.device("cpu")
        loc, scale = forecast.causal_stats(windows)
        given = loc[:, -1], scale[:, -1]
        out = forecast.forecast(model, windows, tags, 2 * PATCH, 2, cpu, floor=False, frame="context")
        fed = np.concatenate([windows[:, PATCH:], out[:, student.MEDIAN, :PATCH]], axis=1)
        lo = np.full_like(fed, np.nan)
        hi = np.full_like(fed, np.nan)
        lo[:, -PATCH:] = out[:, student.LEVELS.index(0.1), :PATCH]
        hi[:, -PATCH:] = out[:, student.LEVELS.index(0.9), :PATCH]
        # the spike has left the moved window, whose own last observed patch is at the scale floor
        _, _, moved = forecast.next_patch(model, fed, tags, cpu, band=(lo, hi), fed=1)
        self.assertLess(float(moved.max()), 1e-5)
        q, loc2, scale2 = forecast.next_patch(model, fed, tags, cpu, band=(lo, hi), fed=1, frame=given)
        np.testing.assert_allclose(scale2, given[1], rtol=1e-6)
        np.testing.assert_allclose(loc2, given[0], rtol=1e-6, atol=1e-6)
        second = loc2[:, None, None] + scale2[:, None, None] * np.sinh(q.astype(np.float64))
        np.testing.assert_allclose(out[:, :, PATCH:], second, rtol=1e-5, atol=1e-5)

    def test_the_context_frame_at_the_floor_leaves_a_window_with_a_scale_of_its_own(self):
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True, frozen_feedback=True)
        model = student.Student(cfg).eval()
        fed = np.full((2, CONTEXT), 0.5, np.float32)
        fed[1] = np.random.default_rng(0).normal(0.5, 0.1, CONTEXT)
        tags = np.zeros(2, np.int64)
        cpu = torch.device("cpu")
        band = (np.full_like(fed, np.nan), np.full_like(fed, np.nan))
        given = np.array([0.7, 0.7], np.float32), np.array([3.0, 3.0], np.float32)
        _, loc, scale = forecast.next_patch(model, fed, tags, cpu, band=band, fed=1)
        _, loc2, scale2 = forecast.next_patch(model, fed, tags, cpu, band=band, fed=1, frame=given, at_floor=True)
        np.testing.assert_allclose([loc2[0], scale2[0]], [0.7, 3.0], rtol=1e-6)
        np.testing.assert_allclose([loc2[1], scale2[1]], [loc[1], scale[1]], rtol=1e-6)


class MaskMin(unittest.TestCase):
    def test_a_short_horizon_decodes_with_the_patches_the_model_was_trained_from(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, mask_tail=30, mask_min=12)).eval()
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(3, dtype=np.int64)
        horizon = 2 * PATCH + 5
        got = forecast.forecast_ahead(model, windows, tags, horizon, 8, torch.device("cpu"), np.zeros(3, dtype=bool))
        q, loc, scale = forecast.ahead_patches(model, windows, tags, torch.device("cpu"), 13)
        want = loc[:, None, None] + scale[:, None, None] * np.sinh(q.astype(np.float64))
        np.testing.assert_allclose(got, want[:, :, :horizon], rtol=1e-5, atol=1e-5)
        plain = forecast.ahead_patches(model, windows, tags, torch.device("cpu"), 3)[0]
        self.assertGreater(float(np.abs(plain - q[:, :, : 3 * PATCH]).max()), 1e-4)


class Widen(unittest.TestCase):
    def test_factors_scale_the_distance_from_the_median_at_the_origin(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64)).eval()
        windows = np.random.default_rng(0).normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(len(windows), np.int64)
        args = (model, windows, tags, 2 * PATCH, 2, torch.device("cpu"))
        plain = forecast.forecast(*args, floor=False)
        ones = np.ones((2, len(student.LEVELS)))
        np.testing.assert_allclose(forecast.forecast(*args, floor=False, widen=ones), plain, rtol=1e-5, atol=1e-5)
        factors = ones.copy()
        factors[1] = 2.0
        wide = forecast.forecast(*args, floor=False, widen=factors)
        np.testing.assert_allclose(wide[:, :, :PATCH], plain[:, :, :PATCH], rtol=1e-5, atol=1e-5)
        loc, scale = forecast.causal_stats(windows)
        to_asinh = lambda x: np.arcsinh((x.astype(np.float64) - loc[:, forecast.LAST, None, None]) / scale[:, forecast.LAST, None, None])
        a, b = to_asinh(plain[:, :, PATCH:]), to_asinh(wide[:, :, PATCH:])
        mid = slice(student.MEDIAN, student.MEDIAN + 1)
        np.testing.assert_allclose(b - b[:, mid], 2.0 * (a - a[:, mid]), rtol=1e-4, atol=1e-4)


class Unanswered(unittest.TestCase):
    def test_a_window_without_a_patch_of_data_or_with_an_empty_last_patch_gets_no_forecast(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64)).eval()
        full = np.random.default_rng(0).normal(size=CONTEXT).astype(np.float32)
        short, newest, stale = np.full(CONTEXT, np.nan, np.float32), np.full(CONTEXT, np.nan, np.float32), full.copy()
        short[-(PATCH - 1) :] = full[-(PATCH - 1) :]
        newest[-PATCH:] = full[-PATCH:]
        stale[-PATCH:] = np.nan
        windows = np.stack([full, short, newest, stale])
        tags = np.zeros(len(windows), np.int64)
        cpu = torch.device("cpu")
        np.testing.assert_array_equal(forecast.answerable(windows), [True, False, True, False])
        out = forecast.forecast(model, windows, tags, 2 * PATCH, 2, cpu)
        self.assertTrue(np.isnan(out[[1, 3]]).all())
        self.assertTrue(np.isfinite(out[[0, 2]]).all())
        # an answered window's forecast is the one it gets among answered windows alone
        alone = forecast.forecast(model, windows[[0, 2]], tags[[0, 2]], 2 * PATCH, 2, cpu)
        np.testing.assert_allclose(out[[0, 2]], alone, rtol=1e-6, atol=1e-6)


class ShortWindow(unittest.TestCase):
    def test_a_window_shorter_than_the_models_starts_with_missing_steps(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True)).eval()
        short = np.random.default_rng(0).normal(size=(3, CONTEXT - PATCH)).astype(np.float32)
        padded = np.concatenate([np.full((3, PATCH), np.nan, np.float32), short], axis=1)
        tags = np.zeros(3, np.int64)
        cpu = torch.device("cpu")
        np.testing.assert_allclose(forecast.forecast(model, short, tags, 2 * PATCH, 2, cpu, floor=False),
                                   forecast.forecast(model, padded, tags, 2 * PATCH, 2, cpu, floor=False), rtol=0, atol=1e-6)


class CommandLine(unittest.TestCase):
    def test_a_run_from_the_command_line_takes_the_patch_width_from_the_model(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        other = 16 if PATCH != 16 else 32
        save = ("import sys; sys.path.insert(0, sys.argv[1]); import torch; from aximoe import student as s; "
                "torch.manual_seed(0); cfg = s.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64); "
                "s.save(s.Student(cfg), s.describe(cfg, 'none', {}, []), sys.argv[2])")
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-I", "-c", save, root, d], env={**os.environ, "AXIMOE_PATCH": str(other)}, check=True)
            series, out = os.path.join(d, "series.f32"), os.path.join(d, "bands.f32")
            np.random.default_rng(0).normal(size=(2, 256)).astype("<f4").tofile(series)
            # the process starts at this test's width, which the model does not have
            subprocess.run([sys.executable, "-I", os.path.join(root, "aximoe", "forecast.py"), "--model", d, "--in", series,
                            "--out", out, "--length", "256"], env={**os.environ, "AXIMOE_PATCH": str(PATCH)}, check=True,
                           capture_output=True)
            with open(out + ".json") as f:
                layout = json.load(f)
        self.assertEqual((layout["patch"], layout["horizon"]), (other, other))


if __name__ == "__main__":
    unittest.main()
