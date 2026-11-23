"""Tests of the layers across tiles: a tile reads the tiles of its group and no other, the cached
passes agree with a full pass, a batch's groups are true groups at the drawn widths, the
forecaster keeps a group in one pass, and the unroll gives a group one context end.

    python -I python/aximoe/test_across.py
"""
import os
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, forecast, train
from aximoe.student import CONTEXT, PATCH, POSITIONS, SEQ, Student, StudentConfig
from aximoe.test_siblings import index_of

BASE = dict(d_model=32, heads=4, layers=3, d_ff=48, band_input=True)


def tiles(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, SEQ, PATCH, generator=g)
    mask = (torch.rand(n, SEQ, PATCH, generator=g) > 0.2).float()
    band = torch.randn(n, SEQ, 2 * PATCH, generator=g) * 0.1
    return inputs * mask, mask, band


class Layers(unittest.TestCase):
    def test_a_tile_reads_its_group_and_no_other_tile(self):
        torch.manual_seed(0)
        model = Student(StudentConfig(**BASE, across=1)).eval()
        inputs, mask, band = tiles(6)
        group = torch.tensor([0, 0, 0, 1, 1, 2])
        with torch.no_grad():
            alone = model.forward_full(inputs, mask, None, None, band)[0]
            each = model.forward_full(inputs, mask, None, None, band, None, None, torch.arange(6))[0]
            torch.testing.assert_close(each, alone, rtol=1e-5, atol=1e-5)
            together = model.forward_full(inputs, mask, None, None, band, None, None, group)[0]
            self.assertGreater(float((together[:5] - alone[:5]).abs().max()), 1e-3)
            torch.testing.assert_close(together[5], alone[5], rtol=1e-5, atol=1e-5)
            changed = inputs.clone()
            changed[3] = torch.randn(SEQ, PATCH) * mask[3]
            moved = model.forward_full(changed, mask, None, None, band, None, None, group)[0]
            torch.testing.assert_close(moved[:3], together[:3], rtol=1e-5, atol=1e-5)
            self.assertGreater(float((moved[4] - together[4]).abs().max()), 1e-3)

    def test_a_tile_is_read_from_its_first_present_patch_on(self):
        torch.manual_seed(0)
        model = Student(StudentConfig(**BASE, across=1)).eval()
        inputs, mask, band = tiles(3)
        mask[1, :20] = 0.0
        mask[2] = 0.0
        inputs = inputs * mask
        with torch.no_grad():
            alone = model.forward_full(inputs[:1], mask[:1], None, None, band[:1])[0]
            pair = model.forward_full(inputs[:2], mask[:2], None, None, band[:2], None, None, torch.zeros(2, dtype=torch.int64))[0]
            torch.testing.assert_close(pair[0, :20], alone[0, :20], rtol=1e-5, atol=1e-5)
            self.assertGreater(float((pair[0, 20:] - alone[0, 20:]).abs().max()), 1e-3)
            three = model.forward_full(inputs, mask, None, None, band, None, None, torch.zeros(3, dtype=torch.int64))[0]
            torch.testing.assert_close(three[:2], pair, rtol=1e-5, atol=1e-5)

    def test_cached_passes_match_a_full_pass_in_groups(self):
        torch.manual_seed(0)
        for extra in (dict(across=1), dict(across=1, rope=True, qk_norm=True), dict(across=2, canon=3, recurrent=8),
                      dict(across=1, loops=2)):
            model = Student(StudentConfig(**BASE, **extra)).eval()
            for w in [w for layer in model.layers for w in (layer.canon_a, layer.canon_c) if w is not None]:
                torch.nn.init.normal_(w, std=0.3)
            if model.recurrent is not None:
                torch.nn.init.normal_(model.recurrent_out.weight, std=0.3)
            n, s = 7, 15
            inputs, mask, band = tiles(n, 1)
            group = torch.tensor([0, 0, 0, 3, 3, 5, 6])
            with torch.no_grad():
                full = model.forward_full(inputs, mask, None, None, band, None, None, group)[0]
                self.assertGreater(float((full - model.forward_full(inputs, mask, None, None, band)[0]).abs().max()), 1e-3)
                q, states = model.cached_pass(inputs, mask, None, band, 0, None, group)
                torch.testing.assert_close(q, full, rtol=1e-5, atol=1e-5)
                # the tiles of a group stand at one position, the groups at different ones
                at = torch.tensor([40, 40, 40, 12, 12, 0, 61])
                rows = torch.arange(n)
                for step in (0, 1):
                    p = at + step
                    inputs[rows, p], mask[rows, p] = torch.randn(n, PATCH), 1.0
                    got = model.next_position(inputs, mask, band, None, p, states, group)
                    want = model.forward_full(inputs, mask, None, None, band, None, None, group)[0][rows, p]
                    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
                q, states = model.cached_pass(inputs[:, :s], mask[:, :s], None, band[:, :s], 2, None, group)
                torch.testing.assert_close(q, model.forward_full(inputs, mask, None, None, band, None, None, group)[0][:, :s],
                                           rtol=1e-5, atol=1e-5)
                for p in (s, s + 1):
                    inputs[:, p], mask[:, p] = torch.randn(n, PATCH), 1.0
                    got = model.next_position(inputs, mask, band, None, torch.full((n,), p), states, group)
                    want = model.forward_full(inputs, mask, None, None, band, None, None, group)[0][:, p]
                    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_the_layers_across_tiles_are_the_last_and_cost_no_parameter(self):
        plain = Student(StudentConfig(**BASE))
        model = Student(StudentConfig(**BASE, across=1))
        self.assertEqual([layer.across for layer in model.layers], [False, False, True])
        self.assertEqual(model.num_params(), plain.num_params())


class Batches(unittest.TestCase):
    def setUp(self):
        names = [f"shop/{k}" for k in range(6)] + ["bank/web", batches.SYNTHETIC + "/1"]
        rows = [(0, 3, k, 5, 0, 60) for k in range(6)]          # 0..5: one group of six at 60 s
        rows += [(0, 3, k, 5, 0, 30) for k in range(3)]         # 6..8: one group of three at 30 s
        rows += [(0, 3, 6, 5, 0, 60)]                           # 9: another namespace, a group of one
        rows += [(1, 3, 7, 5, 0, 60), (1, 3, 7, 6, 0, 60)]      # 10, 11: synthetic
        self.index = index_of(rows, names)
        self.groups = batches.Siblings(self.index, [len(rows)], seed=0)

    def test_regrouped_draws_keep_the_widths_and_name_true_groups(self):
        width = self.index.tiles["width"]
        for trial in range(50):
            rng = np.random.default_rng(trial)
            drawn = rng.integers(0, 12, size=16)
            tiles_, lead = self.groups.regroup(drawn, width[drawn], 4)
            self.assertEqual(sorted(width[tiles_].tolist()), sorted(width[drawn].tolist()))
            generated = np.isin(drawn, [10, 11])
            self.assertTrue((tiles_[generated] == drawn[generated]).all() and (lead[generated] == np.flatnonzero(generated)).all())
            self.assertTrue((lead <= np.arange(16)).all() and (lead[lead] == lead).all())
            self.assertTrue((tiles_[lead == np.arange(16)] == drawn[lead == np.arange(16)]).all())
            for first in np.unique(lead):
                members = tiles_[lead == first]
                self.assertLessEqual(len(members), 4)
                self.assertEqual(len(set(members.tolist())), len(members))
                place = self.groups.place[members]
                self.assertTrue(len(members) == 1 or ((place >= 0).all() and len(set(self.groups.group[place].tolist())) == 1))
        sizes = [np.bincount(self.groups.regroup(np.arange(6), width[:6], 4)[1]).max() for _ in range(20)]
        self.assertEqual(max(sizes), 4)
        same, lead = self.groups.regroup(np.arange(12), width[:12], 4, dropout=1.0)
        self.assertTrue((same == np.arange(12)).all() and (lead == np.arange(12)).all())


class Forecasts(unittest.TestCase):
    def test_passes_keep_a_group_together_and_cover_every_window_once(self):
        groups = np.array([3, 1, 3, 2, 1, 3, 2, 9, 9, 9, 9, 9])
        for batch in (4, 5, 64):
            got = forecast.passes(len(groups), batch, groups)
            self.assertEqual(sorted(np.concatenate(got).tolist()), list(range(len(groups))))
            self.assertTrue(all(len(p) <= batch for p in got))
            for g in (1, 2, 3):
                self.assertEqual(sum(bool((groups[p] == g).any()) for p in got), 1)
        self.assertEqual([p.tolist() for p in forecast.passes(5, 2)], [[0, 1], [2, 3], [4]])
        cut = forecast.capped(groups, 2)
        sizes = np.bincount(cut)
        # groups of 2, 2, 3 and 5 windows in parts of at most two
        self.assertEqual(sorted(sizes.tolist()), [1, 1, 2, 2, 2, 2, 2])
        self.assertTrue(all(len(set(groups[cut == c].tolist())) == 1 for c in np.unique(cut)))

    def test_a_group_changes_the_forecast_whatever_the_pass_size(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(9, CONTEXT)).astype(np.float32).cumsum(axis=1) * 0.1
        windows[2, :300] = np.nan
        tags = np.zeros(9, dtype=np.int64)
        groups = np.array([0, 0, 0, 1, 1, 1, 1, 2, 3])
        device = torch.device("cpu")
        model = Student(StudentConfig(**BASE, mask_tail=30, across=1)).eval()
        for decoder in ("rollout", "direct", "splice:2"):
            alone = forecast.forecast(model, windows, tags, 5 * PATCH, 64, device, decoder=decoder)
            wide = forecast.forecast(model, windows, tags, 5 * PATCH, 64, device, decoder=decoder, groups=groups)
            narrow = forecast.forecast(model, windows, tags, 5 * PATCH, 4, device, decoder=decoder, groups=groups)
            np.testing.assert_allclose(narrow, wide, rtol=1e-4, atol=1e-4)
            self.assertGreater(float(np.abs(wide[:7] - alone[:7]).max()), 1e-3)
            np.testing.assert_allclose(wide[7:], alone[7:], rtol=1e-4, atol=1e-4)
        plain = Student(StudentConfig(**BASE, mask_tail=30)).eval()
        np.testing.assert_allclose(forecast.forecast(plain, windows, tags, 3 * PATCH, 64, device, groups=groups),
                                   forecast.forecast(plain, windows, tags, 3 * PATCH, 64, device), rtol=1e-5, atol=1e-5)


class Unroll(unittest.TestCase):
    def test_the_tiles_of_a_group_share_their_context_end(self):
        torch.manual_seed(0)
        model = Student(StudentConfig(**BASE, across=1)).eval()
        n = 12
        inputs, mask, _ = tiles(n, 2)
        group = torch.tensor([0, 0, 0, 0, 4, 4, 4, 7, 8, 8, 8, 8])
        batch = {"inputs": inputs, "input_mask": mask, "tags": None, "group": group,
                 "loc": torch.zeros(n, POSITIONS), "scale": torch.ones(n, POSITIONS)}
        ends = set()
        for seed in range(6):
            out = train.unroll(model.cached_pass, model.next_position, batch, 3, torch.Generator().manual_seed(seed))
            end = out["context_end"]
            self.assertTrue((end == end[group]).all())
            ends |= set(end.tolist())
        self.assertGreater(len(ends), 2)


if __name__ == "__main__":
    unittest.main()
