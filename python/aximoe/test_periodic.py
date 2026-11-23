"""Tests of the copy input: the detected lag, the copied steps against a step-by-step loop, and
that the trainer's tile and the forecaster's window give the same copy.

    python -I python/aximoe/test_periodic.py
"""
import math
import os
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import forecast, periodic, student
from aximoe.student import CONTEXT, PATCH, POSITIONS, SEQ

STEPS = SEQ * PATCH


class DetectLag(unittest.TestCase):
    def test_lag_of_clean_periodic_tiles_and_none_for_a_constant(self):
        t = torch.arange(STEPS, dtype=torch.float32)
        tiles = torch.stack([torch.sin(2 * math.pi * t / 18), (t % 60 < 7).float() * 3.0 + 1.0, torch.full((STEPS,), 2.5)])
        tiles[0, 100:140] = float("nan")
        lag, strength = periodic.detect_lag(tiles, torch.full((3,), STEPS - 1))
        self.assertEqual(lag.tolist(), [18, 60, 0])
        torch.testing.assert_close(strength, torch.tensor([1.0, 1.0, 0.0]), rtol=0, atol=2e-2)

    def test_only_the_steps_up_to_the_end_count(self):
        t = torch.arange(STEPS, dtype=torch.float32)
        tile = torch.where(t < 1200, torch.sin(2 * math.pi * t / 24), torch.sin(2 * math.pi * t / 50))[None]
        self.assertEqual(periodic.detect_lag(tile, torch.tensor([1199]))[0].tolist(), [24])


class CopyAhead(unittest.TestCase):
    def test_copied_steps_match_a_loop(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(3, STEPS)).astype(np.float32)
        values[rng.random(values.shape) < 0.1] = np.nan
        lag, cut = np.array([7, 45, 1]), np.array([40, 33, 50])
        loc, scale = np.array([0.5, -1.0, 0.0], np.float32), np.array([2.0, 0.5, 1.0], np.float32)
        strength = np.array([0.8, 0.3, 0.9], np.float32)
        got = periodic.copy_ahead(torch.from_numpy(values), torch.from_numpy(lag), torch.from_numpy(strength), torch.from_numpy(cut),
                                  torch.from_numpy(loc), torch.from_numpy(scale)).numpy()
        want = np.zeros((3, SEQ, 2 * PATCH), np.float32)
        for n in range(3):
            end = (cut[n] + 1) * PATCH - 1
            for s in range(cut[n], SEQ):
                for i in range(PATCH):
                    step = (s + 1) * PATCH + i
                    source = step
                    while source > end:
                        source -= lag[n]
                    if lag[n] >= 2 and source >= 0 and np.isfinite(values[n, source]):
                        want[n, s, i] = np.arcsinh((values[n, source] - loc[n]) / scale[n])
                        want[n, s, PATCH + i] = strength[n]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        self.assertEqual(float(np.abs(got[2]).max()), 0.0)

    def test_the_forecaster_feeds_the_copy_the_trainer_builds(self):
        torch.manual_seed(0)
        t = np.arange(STEPS)
        tile = (np.sin(2 * math.pi * t / 45) + 0.3 * np.sin(2 * math.pi * t / 15)).astype(np.float32)[None]
        patches = 24
        c = SEQ - patches
        cut = torch.tensor([c - 1])
        context = tile[:, : c * PATCH]
        loc, scale = forecast.causal_stats(context)
        # the trainer reads the whole tile, the forecaster the context with missing steps after it
        whole = torch.from_numpy(tile)
        padded = torch.from_numpy(np.concatenate([context, np.full((1, patches * PATCH), np.nan, np.float32)], axis=1))
        stats = torch.from_numpy(loc[:, c - 1]), torch.from_numpy(scale[:, c - 1])
        copies = [periodic.copy_ahead(v, *periodic.detect_lag(v, (cut + 1) * PATCH - 1), cut, *stats) for v in (whole, padded)]
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=1e-6)
        self.assertGreater(float(copies[0][0, c - 1 :, PATCH:].mean()), 0.97)
        truth = np.arcsinh((tile[0, c * PATCH :] - loc[0, c - 1]) / scale[0, c - 1])
        np.testing.assert_allclose(copies[0][0, c - 1 : SEQ - 1, :PATCH].reshape(-1).numpy(), truth, atol=1e-4)
        model = student.Student(student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, mask_tail=30, copy_input=True)).eval()
        window = np.concatenate([np.full((1, CONTEXT - c * PATCH), np.nan, np.float32), context], axis=1)
        q, _, _ = forecast.ahead_patches(model, window, np.zeros(1, dtype=np.int64), torch.device("cpu"), patches)
        frozen = [np.concatenate([a, np.repeat(a[:, -1:], POSITIONS - c, axis=1)], axis=1) for a in (loc, scale)]
        inputs, mask, _, _ = forecast.batches.standardise(padded.numpy(), *frozen)
        with torch.no_grad():
            fed = model(torch.from_numpy(inputs), torch.from_numpy(mask), torch.zeros(1, dtype=torch.int64), None, copies[0])[:, c - 1 : c - 1 + patches]
            bare = model(torch.from_numpy(inputs), torch.from_numpy(mask), torch.zeros(1, dtype=torch.int64))[:, c - 1 : c - 1 + patches]
        fed = fed.permute(0, 2, 1, 3).reshape(1, -1, patches * PATCH).numpy()
        np.testing.assert_allclose(q, fed, rtol=1e-5, atol=1e-5)
        self.assertGreater(float(np.abs(fed - bare.permute(0, 2, 1, 3).reshape(1, -1, patches * PATCH).numpy()).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
