"""Tests of the flow head: the sampler integrates a known velocity field exactly, the quantile
head is unchanged by the backbone split, and the flow loss counts present points only.

    python -m unittest discover -s python -p "test_flow.py" -v
    python -I python/aximoe/test_flow.py
"""
import os
import sys
import tempfile
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aximoe import student
from aximoe.student import PATCH, SEQ


class Transport(torch.nn.Module):
    """The velocity field that carries every point to `target` along a straight line in unit time."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target

    def state(self, h):
        return h

    def forward(self, state, x, t):
        return (self.target - x) / (1.0 - t)


class Flow(unittest.TestCase):
    def test_sampler_reaches_the_target_of_a_straight_field(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, flow_head=True))
        target = torch.randn(2, 3, 1, PATCH)
        model.flow = Transport(target)
        draws = model.sample(torch.zeros(2, 3, 32), samples=5, steps=8)
        torch.testing.assert_close(draws, target.expand(2, 3, 5, PATCH), atol=1e-5, rtol=0)

    def test_quantile_head_is_the_head_of_the_hidden_state(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, flow_head=True)).eval()
        inputs, mask = torch.randn(2, SEQ, PATCH), torch.ones(2, SEQ, PATCH)
        with torch.no_grad():
            full = model.forward_full(inputs, mask)[0]
            split = model.head(model.hidden(inputs, mask)[0])
        torch.testing.assert_close(full, split)
        self.assertEqual(tuple(model.sample_quantiles(model.hidden(inputs, mask)[0][:, :4], 16, 2).shape),
                         (2, 4, len(student.LEVELS), PATCH))

    def test_knot_draws_at_a_level_are_that_knot(self):
        torch.manual_seed(0)
        knots = torch.sort(torch.randn(2, 3, len(student.LEVELS), PATCH), dim=2).values
        for m, level in enumerate(student.LEVELS):
            u = torch.full((2, 3, 4, PATCH), level, dtype=knots.dtype)
            torch.testing.assert_close(student.knot_draws(knots, u), knots[:, :, None, m].expand(2, 3, 4, PATCH),
                                       atol=1e-5, rtol=0)
        between = student.knot_draws(knots, torch.full((2, 3, 1, PATCH), 0.45))
        self.assertTrue((between >= knots[:, :, None, 5]).all() and (between <= knots[:, :, None, 6]).all())

    def test_flow_loss_counts_present_points_only(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, flow_head=True))
        h, targets = torch.randn(2, SEQ, 32), torch.randn(2, SEQ - 1, PATCH)
        self.assertEqual(float(model.flow_loss(h, targets, torch.zeros_like(targets))), 0.0)
        self.assertTrue(torch.isfinite(model.flow_loss(h, targets, torch.ones_like(targets))))

    def test_model_without_the_flow_head_keeps_the_quantile_output(self):
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, flow_head=True)
        model = student.Student(cfg).eval()
        description = student.describe(cfg, "teacher", {}, [60])
        with tempfile.TemporaryDirectory() as out:
            stripped, plain = student.without_flow(description, model)
            student.save(plain, stripped, out)
            saved, plain = student.load(out)
        self.assertFalse(saved["config"]["flow_head"])
        self.assertIsNone(plain.flow)
        self.assertEqual(plain.num_params(), model.num_params() - sum(p.numel() for p in model.flow.parameters()))
        inputs, mask = torch.randn(2, SEQ, PATCH), torch.ones(2, SEQ, PATCH)
        with torch.no_grad():
            torch.testing.assert_close(plain.eval().forward_full(inputs, mask)[0], model.forward_full(inputs, mask)[0],
                                       atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
