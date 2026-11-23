"""The error input: what each position holds, and that the forecaster reads it."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import forecast, student
from aximoe.student import CONTEXT, MEDIAN, PATCH, POSITIONS, SEQ


class ErrorInput(unittest.TestCase):
    def test_a_position_holds_the_error_of_the_forecast_of_its_own_patch(self):
        torch.manual_seed(0)
        pred = torch.randn(3, SEQ, 15, PATCH)
        targets = torch.randn(3, POSITIONS, PATCH)
        mask = (torch.rand(3, POSITIONS, PATCH) > 0.2).float()
        known = torch.tensor([POSITIONS - 1, 10, -1])
        channel = student.error_channel(pred, targets, mask, known)
        self.assertEqual(tuple(channel.shape), (3, SEQ, 2 * PATCH))
        self.assertEqual(float(channel[:, 0].abs().max()), 0.0)
        # position 7 holds patch 7, which position 6 forecast
        want = (targets[0, 6] - pred[0, 6, MEDIAN]).clamp(-8, 8) * mask[0, 6]
        torch.testing.assert_close(channel[0, 7, :PATCH], want, rtol=0, atol=1e-6)
        torch.testing.assert_close(channel[0, 7, PATCH:], mask[0, 6], rtol=0, atol=1e-6)
        # nothing after the last observed position, and nothing at all for a tile that reads none
        self.assertEqual(float(channel[1, 11:].abs().max()), 0.0)
        self.assertGreater(float(channel[1, 10].abs().max()), 0.0)
        self.assertEqual(float(channel[2].abs().max()), 0.0)

    def test_the_forecaster_reads_the_errors_of_its_own_first_pass(self):
        torch.manual_seed(1)
        cfg = student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, error_input=True)
        model = student.Student(cfg).eval()
        rng = np.random.default_rng(2)
        windows = rng.normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(3, dtype=np.int64)
        with_errors, _, _ = forecast.next_patch(model, windows, tags, torch.device("cpu"))
        # the same weights with the input's weights at zero forecast as a model that reads no error
        blind = student.Student(cfg).eval()
        blind.load_state_dict(model.state_dict())
        with torch.no_grad():
            blind.embed1.weight[:, -2 * PATCH :] = 0.0
        without, _, _ = forecast.next_patch(blind, windows, tags, torch.device("cpu"))
        self.assertGreater(float(np.abs(with_errors - without).max()), 1e-4)
        # a context whose last patches are forecasts reads no error for them
        fed, _, _ = forecast.next_patch(model, windows, tags, torch.device("cpu"), fed=POSITIONS)
        torch.manual_seed(1)
        self.assertTrue(np.isfinite(fed).all())

    def test_the_backtest_holds_the_errors_of_a_rollout_from_the_shorter_context(self):
        from aximoe import batches, train
        torch.manual_seed(3)
        span = 4
        cfg = student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, error_input=True,
                                    error_backtest=span, frozen_feedback=True)
        model = student.Student(cfg).eval()
        with torch.no_grad():
            # a model that reads no error forecasts in the rollout as in the pass without the input
            model.embed1.weight[:, -4 * PATCH :] = 0.0
        rng = np.random.default_rng(4)
        windows = rng.normal(size=(3, CONTEXT)).astype(np.float32).cumsum(axis=1)
        windows[1, 5:40] = np.nan
        tags = np.zeros(3, dtype=np.int64)
        back = forecast.backtest_input(model, windows, tags, 2, torch.device("cpu"))
        self.assertEqual(back.shape, (3, SEQ, 2 * PATCH))
        cut = POSITIONS - 1 - span
        self.assertEqual(float(np.abs(back[:, : cut + 1]).max()), 0.0)
        self.assertEqual(float(np.abs(back[:, POSITIONS:]).max()), 0.0)
        self.assertEqual(float(back[0, cut + 1 : POSITIONS, PATCH:].min()), 1.0)
        # the rollout's first patch is the forecast of the pass without the input, in the space at the cut
        loc, scale = forecast.causal_stats(windows)
        tiles = np.concatenate([windows, np.full((3, PATCH), np.nan, np.float32)], axis=1)
        inputs, mask, targets, target_mask = (torch.from_numpy(a) for a in batches.standardise(tiles, loc, scale))
        with torch.no_grad():
            draft = model(inputs, mask, torch.from_numpy(tags))
        want = (targets[:, cut] - draft[:, cut, MEDIAN]).clamp(-8, 8) * target_mask[:, cut]
        np.testing.assert_allclose(back[:, cut + 1, :PATCH], want.numpy(), rtol=0, atol=1e-4)
        # a later patch follows a forecast, so its error is another than the one-patch error
        later = (targets[:, cut + 2] - draft[:, cut + 2, MEDIAN]).clamp(-8, 8) * target_mask[:, cut + 2]
        self.assertGreater(float(np.abs(back[:, cut + 3, :PATCH] - later.numpy()).max()), 1e-3)

    def test_the_forecaster_reads_the_backtest(self):
        torch.manual_seed(5)
        cfg = student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, error_input=True,
                                    error_backtest=4, frozen_feedback=True)
        model = student.Student(cfg).eval()
        rng = np.random.default_rng(6)
        windows = rng.normal(size=(3, CONTEXT)).astype(np.float32)
        tags = np.zeros(3, dtype=np.int64)
        read = forecast.forecast(model, windows, tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        blind = student.Student(cfg).eval()
        blind.load_state_dict(model.state_dict())
        with torch.no_grad():
            blind.embed1.weight[:, -2 * PATCH :] = 0.0
        without = forecast.forecast(blind, windows, tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        self.assertTrue(np.isfinite(read).all())
        self.assertGreater(float(np.abs(read - without).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
