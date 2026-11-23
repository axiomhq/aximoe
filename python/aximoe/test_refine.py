"""The series embedding: a widened model computes what it did, and the rounds reach the forecast."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import numpy as np
import torch

from aximoe import forecast, student
from aximoe.student import CONTEXT, PATCH


class Refine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = student.StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, frozen_feedback=True)
        self.base = student.Student(self.cfg).eval()
        rng = np.random.default_rng(1)
        self.windows = rng.normal(size=(3, CONTEXT)).astype(np.float32).cumsum(axis=1)
        self.tags = np.zeros(3, dtype=np.int64)

    def test_a_widened_model_forecasts_as_the_model_it_came_from(self):
        wide = student.widen(self.base, dataclasses.replace(self.cfg, refine_rounds=3, refine_depth=4)).eval()
        a = forecast.forecast(self.base, self.windows, self.tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        b = forecast.forecast(wide, self.windows, self.tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)

    def test_the_rounds_change_the_embedding_and_the_forecast_and_carry_gradients(self):
        from aximoe import train
        wide = student.widen(self.base, dataclasses.replace(self.cfg, refine_rounds=3, refine_depth=4)).eval()
        with torch.no_grad():
            wide.series_out.weight.normal_(std=0.5)
            wide.embed1.weight[:, -2 * PATCH :].normal_(std=0.5)
        state = forecast.series_state(wide, self.windows, self.tags, 2, torch.device("cpu"))
        self.assertEqual(tuple(state.shape), (3, student.SEQ, 32))
        self.assertGreater(float((state[0, -1] - state[1, -1]).abs().max()), 1e-4)
        a = forecast.forecast(self.base, self.windows, self.tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        b = forecast.forecast(wide, self.windows, self.tags, 3 * PATCH, 2, torch.device("cpu"), floor=False)
        self.assertGreater(float(np.abs(a - b).max()), 1e-4)
        # the loss of a forecast under the embedding reaches the update's weights through every round
        loc, scale = forecast.causal_stats(self.windows)
        tiles = np.concatenate([self.windows, np.full((3, PATCH), np.nan, np.float32)], axis=1)
        from aximoe import batches
        inputs, mask, targets, _ = (torch.from_numpy(x) for x in batches.standardise(tiles, loc, scale))
        tile = {"inputs": inputs, "input_mask": mask, "targets": targets, "values": torch.from_numpy(tiles),
                "tags": torch.from_numpy(self.tags), "loc": torch.from_numpy(loc.astype(np.float32)),
                "scale": torch.from_numpy(scale.astype(np.float32))}
        e = train.refine_state(wide.cached_pass, wide.next_position, wide.update, wide.series_start, tile,
                               torch.full((3,), student.POSITIONS - 1), 3, 4, True)
        wide(inputs, mask, torch.from_numpy(self.tags), state=e).sum().backward()
        self.assertGreater(float(wide.series_out.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(wide.series_start.grad.abs().max()), 0.0)
        self.assertGreater(float(wide.embed1.weight.grad[:, -2 * PATCH :].abs().max()), 0.0)

    def test_a_position_reads_no_embedding_made_from_a_later_patch(self):
        from aximoe import batches, train
        wide = student.widen(self.base, dataclasses.replace(self.cfg, refine_rounds=3, refine_depth=4)).eval()
        with torch.no_grad():
            wide.series_out.weight.normal_(std=0.5)
            wide.embed1.weight[:, -2 * PATCH :].normal_(std=0.5)
        cut = 20
        changed = self.windows.copy()
        changed[:, (cut + 1) * PATCH :] += 3.0
        out = []
        for w in (self.windows, changed):
            loc, scale = forecast.causal_stats(w)
            tiles = np.concatenate([w, np.full((3, PATCH), np.nan, np.float32)], axis=1)
            inputs, mask, targets, _ = (torch.from_numpy(x) for x in batches.standardise(tiles, loc, scale))
            tile = {"inputs": inputs, "input_mask": mask, "targets": targets, "values": torch.from_numpy(tiles),
                    "tags": torch.from_numpy(self.tags), "loc": torch.from_numpy(loc.astype(np.float32)),
                    "scale": torch.from_numpy(scale.astype(np.float32))}
            with torch.no_grad():
                state = train.refine_state(wide.cached_pass, wide.next_position, wide.update, wide.series_start, tile,
                                           torch.full((3,), student.POSITIONS - 1), 3, 4, True)
                out.append((state, wide(inputs, mask, torch.from_numpy(self.tags), state=state)))
        # the patches after `cut` differ: the embeddings and forecasts up to `cut` are the same, later ones are not
        self.assertLess(float((out[0][0][:, : cut + 1] - out[1][0][:, : cut + 1]).abs().max()), 1e-5)
        self.assertLess(float((out[0][1][:, : cut + 1] - out[1][1][:, : cut + 1]).abs().max()), 1e-4)
        self.assertGreater(float((out[0][0][:, -1] - out[1][0][:, -1]).abs().max()), 1e-4)
        self.assertGreater(float((out[0][0][:, cut] - wide.series_start).abs().max()), 1e-4)

    def test_read_at_the_forecast_only_the_forecast_loss_reaches_the_update(self):
        from aximoe import batches, train
        wide = student.widen(self.base, dataclasses.replace(self.cfg, refine_rounds=3, refine_depth=4, refine_read="forecast")).eval()
        with torch.no_grad():
            wide.series_out.weight.normal_(std=0.5)
            wide.embed1.weight[:, -2 * PATCH :].normal_(std=0.5)
        known = 40
        loc, scale = forecast.causal_stats(self.windows)
        tiles = np.concatenate([self.windows, np.full((3, PATCH), np.nan, np.float32)], axis=1)
        inputs, mask, targets, _ = (torch.from_numpy(x) for x in batches.standardise(tiles, loc, scale))
        tile = {"inputs": inputs, "input_mask": mask, "targets": targets, "values": torch.from_numpy(tiles),
                "tags": torch.from_numpy(self.tags), "loc": torch.from_numpy(loc.astype(np.float32)),
                "scale": torch.from_numpy(scale.astype(np.float32))}
        for part, reaches in ((slice(0, known), False), (slice(known, known + 1), True)):
            wide.zero_grad()
            state = train.refine_state(wide.cached_pass, wide.next_position, wide.update, wide.series_start, tile,
                                       torch.full((3,), known), 3, 4, True, read="forecast")
            wide(inputs, mask, torch.from_numpy(self.tags), state=state)[:, part].sum().backward()
            grad = float(wide.series_out.weight.grad.abs().max()) if wide.series_out.weight.grad is not None else 0.0
            self.assertEqual(grad > 0.0, reaches)


if __name__ == "__main__":
    unittest.main()
