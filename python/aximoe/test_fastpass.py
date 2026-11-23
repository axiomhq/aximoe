"""Tests of the unroll's leaner one-position pass: `fastpass.Bundle.rows` against `Student.next_rows`."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aximoe import fastpass
from aximoe.student import PATCH, POSITIONS, SEQ, TAG_BASE, Student, StudentConfig

# single-precision passes on the CPU differ by the order of their sums
TOLERANCE = 2e-5


def model_of(**changes) -> Student:
    torch.manual_seed(3)
    cfg = StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, tags=8, backbone_tags=True, top_k=1, resolutions=3,
                        resolution_tags=True, band_input=True, canon=4, frozen_feedback=True, **changes)
    model = Student(cfg)
    with torch.no_grad():
        # the parameters that start at zero get values, so every term of a pass counts
        for layer in model.layers:
            for p in (layer.attn.in_proj_bias, layer.attn.out_proj.bias, layer.canon_a, layer.canon_c):
                p.normal_(0.0, 0.2)
    return model.eval()


class Rows(unittest.TestCase):
    def passes(self, model: Student) -> None:
        n = 5
        inputs, mask = torch.randn(n, SEQ, PATCH), torch.ones(n, SEQ, PATCH)
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        # a channel row plus, times TAG_BASE, one more than a resolution row (zero: the last row)
        tags = torch.tensor([0, 1, 2, 3, 4]) + TAG_BASE * torch.tensor([0, 1, 2, 3, 0])
        self.assertTrue(fastpass.supports(model))
        bundle = fastpass.Bundle(model, torch.float32)
        with torch.no_grad():
            _, own = model.cached_pass(inputs, mask, tags, band, 0)
            _, held = bundle.cache(model.cached_pass)(inputs, mask, tags, band, 0)
            rows = torch.arange(n)
            p = torch.tensor([POSITIONS - 9, POSITIONS - 5, 3, 0, POSITIONS - 2])
            for _ in range(3):
                # the patch at p changed since the caches were filled, as in an unroll
                inputs[rows, p] = torch.randn(n, PATCH)
                want, want_rows = model.next_rows(inputs, mask, band, tags, p, own)
                got, got_rows = bundle.rows(inputs, mask, band, tags, p, held)
                self.assertLessEqual(float((want - got).abs().max()), TOLERANCE)
                for cache, row in zip(own, want_rows):
                    cache[rows, p] = row
                for cache, row in zip(held, got_rows):
                    cache[rows, p] = row
                for i, cache in enumerate(own):
                    self.assertLessEqual(float((cache - torch.cat([held[2 * i], held[2 * i + 1]], dim=2)).abs().max()), TOLERANCE)
                p = p + 1

    def test_a_pass_gives_the_models_quantiles_and_cache_rows(self):
        self.passes(model_of())

    def test_with_rotary_positions_too(self):
        self.passes(model_of(rope=True))


if __name__ == "__main__":
    unittest.main()
