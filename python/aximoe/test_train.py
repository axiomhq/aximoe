"""Tests of the trainer's unroll for the band input, of the tail teacher's term and of the tail
mask's agreement with the single-pass forecaster.

    python -m unittest discover -s python -p "test_train.py" -v
    python -I python/aximoe/test_train.py
"""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, student, train
from aximoe.student import CONTEXT, LEVELS, MEDIAN, PATCH, POSITIONS, SEQ


class TailMask(unittest.TestCase):
    """What a position after the cut is trained on is what the forecaster asks of it: the same
    inputs, the same positions and the same statistics."""

    def setUp(self):
        from aximoe import forecast
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        self.forecast = forecast
        self.values = (50.0 + 10.0 * rng.normal(size=(3, SEQ * PATCH)).cumsum(axis=1) / 30).astype(np.float32)
        self.values[0, 100:140] = np.nan
        loc, scale = forecast.causal_stats(self.values)
        self.loc, self.scale = loc[:, :POSITIONS], scale[:, :POSITIONS]
        inputs, mask, targets, target_mask = batches.standardise(self.values, self.loc, self.scale)
        self.batch = {"inputs": torch.from_numpy(inputs), "input_mask": torch.from_numpy(mask),
                      "targets": torch.from_numpy(targets), "target_mask": torch.from_numpy(target_mask),
                      "values": torch.from_numpy(self.values), "loc": torch.from_numpy(self.loc),
                      "scale": torch.from_numpy(self.scale)}
        self.model = student.Student(student.StudentConfig(mask_tail=30)).eval()

    def test_targets_and_masks_past_the_cut(self):
        j = torch.tensor([0, 5, 30])
        out = train.mask_tail(self.batch, j)
        for t, cut in enumerate((POSITIONS - 1 - j).tolist()):
            self.assertAlmostEqual(float(out["input_mask"][t, cut + 1 :].abs().sum()), 0.0, delta=1e-9)
            self.assertAlmostEqual(float(out["inputs"][t, cut + 1 :].abs().sum()), 0.0, delta=1e-9)
            torch.testing.assert_close(out["inputs"][t, : cut + 1], self.batch["inputs"][t, : cut + 1], rtol=0, atol=1e-6)
            torch.testing.assert_close(out["targets"][t, : cut + 1], self.batch["targets"][t, : cut + 1], rtol=0, atol=1e-6)
            want = torch.tensor([1.0] * (cut + 1) + [0.0] * (POSITIONS - cut - 1))
            torch.testing.assert_close(out["distill_mask"][t], want, rtol=0, atol=1e-9)
            for s in range(cut + 1, POSITIONS):
                raw = self.values[t, (s + 1) * PATCH : (s + 2) * PATCH]
                want = np.arcsinh((raw - self.loc[t, cut]) / self.scale[t, cut])
                np.testing.assert_allclose(out["targets"][t, s].numpy(), want, rtol=0, atol=1e-5)

    def test_the_forecaster_sees_what_training_showed(self):
        for j in (1, 7, 28):
            cut = POSITIONS - 1 - j
            out = train.mask_tail(self.batch, torch.full((3,), j))
            with torch.no_grad():
                trained = self.model(out["inputs"], out["input_mask"], None)[:, cut:POSITIONS]
            window = self.values[:, : (cut + 1) * PATCH]
            q, loc, scale = self.forecast.ahead_patches(self.model, window, np.zeros(3, np.int64), torch.device("cpu"), j + 1)
            want = trained.permute(0, 2, 1, 3).reshape(3, len(LEVELS), (j + 1) * PATCH).numpy()
            np.testing.assert_allclose(q, want, rtol=0, atol=1e-5)
            np.testing.assert_allclose(loc, self.loc[:, cut], rtol=1e-6, atol=0)
            np.testing.assert_allclose(scale, self.scale[:, cut], rtol=1e-6, atol=0)


class TailTeacher(unittest.TestCase):
    def test_the_term_reads_each_tiles_row_and_stored_positions(self):
        # quarter steps are exact in float16, so the teacher can equal the student bit for bit
        pred = torch.randint(-8, 8, (4, SEQ, len(LEVELS), PATCH)).float() / 4
        covered = np.array([1, 4, 7])
        positions = np.array([[3, 40], [0, 62], [10, 11]], np.uint8)
        batch_tiles = np.array([7, 2, 1, 4])
        levels = [LEVELS.index(v) for v in (0.01, 0.05, 0.95, 0.99)]
        q = np.zeros((len(covered), len(levels), 2, PATCH), np.float16)
        for b, t in enumerate(batch_tiles):
            if t in covered:
                r = list(covered).index(t)
                q[r] = pred[b, positions[r].astype(int)][:, levels].numpy().transpose(1, 0, 2)
        q[list(covered).index(4), 2, 1, 5] += 0.5
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"levels": [0.01, 0.05, 0.95, 0.99], "positions_per_tile": 2}, f)
            covered.astype("<i8").tofile(os.path.join(d, "tiles.i64"))
            positions.tofile(os.path.join(d, "positions.u8"))
            q.astype("<f2").tofile(os.path.join(d, "quantiles.f16"))
            teacher = train.TailTeacher(d, 10)
            batch = {"target_mask": torch.ones(4, POSITIONS, PATCH), "distill_mask": torch.ones(4, POSITIONS)}
            batch["distill_mask"][0, 11] = 0.0
            batch |= teacher.lookup(batch_tiles)
            loss = train.tail_loss(pred, batch, teacher.levels)
        # tiles 7, 1 and 4 at two positions each, less tile 7's position 11: 160 points; one is off by 0.5
        self.assertAlmostEqual(loss.item() / (0.5 * 0.5 ** 2 / (160 * 9)), 1.0, places=6)


class Unroll(unittest.TestCase):
    def test_forecasts_replace_the_last_patches_in_their_own_statistics(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True)).eval()
        rng = np.random.default_rng(0)
        n = 16
        values = rng.normal(5.0, 2.0, size=(n, 2048)).astype(np.float32)
        loc = rng.normal(5.0, 1.0, size=(n, POSITIONS)).astype(np.float32)
        scale = rng.uniform(0.5, 2.0, size=(n, POSITIONS)).astype(np.float32)
        inputs, input_mask, targets, target_mask = batches.standardise(values, loc, scale)
        batch = {"inputs": torch.from_numpy(inputs), "input_mask": torch.from_numpy(input_mask),
                 "targets": torch.from_numpy(targets), "target_mask": torch.from_numpy(target_mask),
                 "tags": torch.zeros(n, dtype=torch.int64), "families": torch.zeros(n, dtype=torch.int64),
                 "loc": torch.from_numpy(loc), "scale": torch.from_numpy(scale)}
        out = train.unroll(model.cached_pass, model.next_position, batch, 3, torch.Generator().manual_seed(0))
        # a limit above every knot leaves the unroll as it is
        same = train.unroll(model.cached_pass, model.next_position, batch, 3, torch.Generator().manual_seed(0), None, 60.0)
        torch.testing.assert_close(same["inputs"], out["inputs"], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(same["band"], out["band"], rtol=1e-4, atol=1e-4)
        # knots of 200 asinh units: infinite inputs without the limit, inputs within it with
        huge = lambda *a, **k: (model.cached_pass(*a, **k)[0] * 0 + 200.0, model.cached_pass(*a, **k)[1])
        far = lambda *a, **k: model.next_position(*a, **k) * 0 + 200.0
        broken = train.unroll(huge, far, batch, 3, torch.Generator().manual_seed(0))
        self.assertFalse(bool(torch.isfinite(broken["inputs"]).all()))
        held = train.unroll(huge, far, batch, 3, torch.Generator().manual_seed(0), None, 60.0)
        self.assertTrue(bool(torch.isfinite(held["inputs"]).all()) and bool(torch.isfinite(held["band"]).all()))
        self.assertLessEqual(float(held["inputs"].abs().max()), max(60.0, float(batch["inputs"].abs().max())) + 1e-4)
        cut = (out["distill_mask"].sum(1) - 1).long()
        j = POSITIONS - 1 - cut
        self.assertTrue(set(j.tolist()) <= {0, 1, 2, 3} and {0, 3} <= set(j.tolist()))
        for i in range(n):
            c = int(cut[i])
            # the truth context up to the cut is untouched and carries no band
            torch.testing.assert_close(out["inputs"][i, : c + 1], batch["inputs"][i, : c + 1])
            self.assertEqual(float(out["band"][i, : c + 1].abs().sum()), 0.0)
            self.assertEqual(float(out["band"][i, POSITIONS:].abs().sum()), 0.0)
        # every forecast patch: the prediction at the position before it from the unrolled context,
        # mapped from that position's statistics into the replaced patch's, as its input and band
        with torch.no_grad():
            q = model.forward_full(out["inputs"], out["input_mask"], out["tags"], out["families"],
                                   out["band"])[0].double()
        lo, hi = LEVELS.index(0.1), LEVELS.index(0.9)
        for i in range(n):
            for s in range(int(cut[i]) + 1, POSITIONS):
                raw = loc[i, s - 1] + scale[i, s - 1] * torch.sinh(q[i, s - 1])
                a = torch.asinh((raw - float(loc[i, s])) / float(scale[i, s]))
                torch.testing.assert_close(out["inputs"][i, s].double(), a[MEDIAN], rtol=0, atol=1e-5)
                torch.testing.assert_close(out["band"][i, s, :PATCH].double(), a[lo] - a[MEDIAN], rtol=0, atol=1e-5)
                torch.testing.assert_close(out["band"][i, s, PATCH:].double(), a[hi] - a[MEDIAN], rtol=0, atol=1e-5)
                self.assertEqual(float(out["input_mask"][i, s].min()), 1.0)

    def test_frozen_feedback_trains_what_the_forecaster_shows(self):
        from aximoe import forecast
        torch.manual_seed(0)
        cfg = student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True, frozen_feedback=True)
        model = student.Student(cfg).eval()
        rng = np.random.default_rng(0)
        n = 6
        values = (50.0 + 10.0 * rng.normal(size=(n, SEQ * PATCH)).cumsum(axis=1) / 30).astype(np.float32)
        values[0, 100:140] = np.nan
        values[1, 2020:2030] = np.nan
        stats = forecast.causal_stats(values)
        loc, scale = stats[0][:, :POSITIONS], stats[1][:, :POSITIONS]
        inputs, input_mask, targets, target_mask = batches.standardise(values, loc, scale)
        batch = {"inputs": torch.from_numpy(inputs), "input_mask": torch.from_numpy(input_mask),
                 "targets": torch.from_numpy(targets), "target_mask": torch.from_numpy(target_mask),
                 "tags": torch.zeros(n, dtype=torch.int64), "families": torch.zeros(n, dtype=torch.int64),
                 "values": torch.from_numpy(values), "loc": torch.from_numpy(loc), "scale": torch.from_numpy(scale)}
        lo, hi = LEVELS.index(0.1), LEVELS.index(0.9)
        for j in (1, 3):
            cut = POSITIONS - 1 - j
            out = train.unroll(model.cached_pass, model.next_position, batch, 3, torch.Generator().manual_seed(0),
                               j=torch.full((n,), j), frozen=True)
            with torch.no_grad():
                q = model.forward_full(out["inputs"], out["input_mask"], out["tags"], out["families"], out["band"])[0]
            torch.testing.assert_close(out["inputs"][:, : cut + 1], batch["inputs"][:, : cut + 1], rtol=0, atol=1e-6)
            torch.testing.assert_close(out["targets"][:, : cut + 1], batch["targets"][:, : cut + 1], rtol=0, atol=1e-6)
            for s in range(cut + 1, POSITIONS):
                # a forecast is fed back as it was made, and its target is in the space of the cut
                torch.testing.assert_close(out["inputs"][:, s], q[:, s - 1, MEDIAN], rtol=0, atol=1e-4)
                torch.testing.assert_close(out["band"][:, s, :PATCH], q[:, s - 1, lo] - q[:, s - 1, MEDIAN], rtol=0, atol=1e-4)
                torch.testing.assert_close(out["band"][:, s, PATCH:], q[:, s - 1, hi] - q[:, s - 1, MEDIAN], rtol=0, atol=1e-4)
                raw = values[:, (s + 1) * PATCH : (s + 2) * PATCH]
                want = np.where(np.isfinite(raw), np.arcsinh((np.nan_to_num(raw) - loc[:, cut, None]) / scale[:, cut, None]), 0.0)
                np.testing.assert_allclose(out["targets"][:, s].numpy(), want, rtol=0, atol=1e-5)
            # the forecaster, handed the same context with the forecasts as its last patches, reads and answers the same
            centre, spread = loc[:, cut, None].astype(np.float64), scale[:, cut, None].astype(np.float64)
            window = values[:, :CONTEXT].copy()
            low, high = np.full_like(window, np.nan), np.full_like(window, np.nan)
            fed = out["inputs"][:, cut + 1 : POSITIONS].double().numpy().reshape(n, -1)
            offsets = out["band"][:, cut + 1 : POSITIONS].double().numpy()
            window[:, (cut + 1) * PATCH :] = centre + spread * np.sinh(fed)
            low[:, (cut + 1) * PATCH :] = centre + spread * np.sinh(fed + offsets[:, :, :PATCH].reshape(n, -1))
            high[:, (cut + 1) * PATCH :] = centre + spread * np.sinh(fed + offsets[:, :, PATCH:].reshape(n, -1))
            got, at, by = forecast.next_patch(model, window, np.zeros(n, np.int64), torch.device("cpu"), band=(low, high), fed=j)
            np.testing.assert_allclose(got, q[:, POSITIONS - 1].numpy(), rtol=0, atol=1e-3)
            np.testing.assert_allclose(at, loc[:, cut], rtol=1e-6, atol=0)
            np.testing.assert_allclose(by, scale[:, cut], rtol=1e-6, atol=0)
            # without it the forecaster's statistics include the patches fed back
            model.cfg.frozen_feedback = False
            _, _, wide = forecast.next_patch(model, window, np.zeros(n, np.int64), torch.device("cpu"), band=(low, high), fed=j)
            model.cfg.frozen_feedback = True
            self.assertTrue((np.abs(wide / by - 1.0) > 1e-4).all())

    def test_own_statistics_are_those_the_forecaster_computes(self):
        from aximoe import forecast
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True)).eval()
        rng = np.random.default_rng(0)
        n = 6
        values = (50.0 + 10.0 * rng.normal(size=(n, SEQ * PATCH)).cumsum(axis=1) / 30).astype(np.float32)
        values[0, 100:140] = np.nan
        values[1, 2020:2030] = np.nan
        stats = forecast.causal_stats(values)
        loc, scale = stats[0][:, :POSITIONS], stats[1][:, :POSITIONS]
        inputs, input_mask, targets, target_mask = batches.standardise(values, loc, scale)
        batch = {"inputs": torch.from_numpy(inputs), "input_mask": torch.from_numpy(input_mask),
                 "targets": torch.from_numpy(targets), "target_mask": torch.from_numpy(target_mask),
                 "tags": torch.zeros(n, dtype=torch.int64), "families": torch.zeros(n, dtype=torch.int64),
                 "values": torch.from_numpy(values), "loc": torch.from_numpy(loc), "scale": torch.from_numpy(scale)}
        for j in (1, 3):
            cut = POSITIONS - 1 - j
            out = train.unroll(model.cached_pass, model.next_position, batch, 3, torch.Generator().manual_seed(0),
                               j=torch.full((n,), j), own=True)
            torch.testing.assert_close(out["loc"][:, : cut + 1], batch["loc"][:, : cut + 1], rtol=0, atol=0)
            # the window a forecaster holds after feeding these forecasts back, in the series' units
            window = values[:, :CONTEXT].copy()
            low, high = np.full_like(window, np.nan), np.full_like(window, np.nan)
            for s in range(cut + 1, POSITIONS):
                centre, spread = out["loc"][:, s, None].double().numpy(), out["scale"][:, s, None].double().numpy()
                fed, offsets = out["inputs"][:, s].double().numpy(), out["band"][:, s].double().numpy()
                steps = slice(s * PATCH, (s + 1) * PATCH)
                window[:, steps] = centre + spread * np.sinh(fed)
                low[:, steps] = centre + spread * np.sinh(fed + offsets[:, :PATCH])
                high[:, steps] = centre + spread * np.sinh(fed + offsets[:, PATCH:])
            want_loc, want_scale = forecast.causal_stats(window)
            np.testing.assert_allclose(out["loc"][:, cut + 1 :].numpy(), want_loc[:, cut + 1 :], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(out["scale"][:, cut + 1 :].numpy(), want_scale[:, cut + 1 :], rtol=1e-4, atol=0)
            self.assertTrue((np.abs(want_scale[:, cut + 1 :] / scale[:, cut + 1 :] - 1.0) > 1e-4).all())
            for s in range(cut + 1, POSITIONS):
                raw = values[:, (s + 1) * PATCH : (s + 2) * PATCH]
                want = np.where(np.isfinite(raw), np.arcsinh((np.nan_to_num(raw) - want_loc[:, s, None]) / want_scale[:, s, None]), 0.0)
                np.testing.assert_allclose(out["targets"][:, s].numpy(), want, rtol=0, atol=1e-4)
            # the forecaster reads that window as the trainer's last position read the tile
            with torch.no_grad():
                q = model.forward_full(out["inputs"], out["input_mask"], out["tags"], out["families"], out["band"])[0]
            got, at, by = forecast.next_patch(model, window, np.zeros(n, np.int64), torch.device("cpu"), band=(low, high), fed=j)
            np.testing.assert_allclose(got, q[:, POSITIONS - 1].numpy(), rtol=0, atol=1e-3)
            np.testing.assert_allclose(at, out["loc"][:, POSITIONS - 1].numpy(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(by, out["scale"][:, POSITIONS - 1].numpy(), rtol=1e-4, atol=0)


class DistillMask(unittest.TestCase):
    def test_the_mask_drops_positions_from_the_teacher_term_only(self):
        torch.manual_seed(0)
        n = 4
        pred = torch.randn(n, student.SEQ, len(LEVELS), PATCH).sort(dim=2).values
        batch = {"targets": torch.randn(n, POSITIONS, PATCH), "target_mask": (torch.rand(n, POSITIONS, PATCH) > 0.1).float(),
                 "teacher": torch.randn(n, len(student.SHARED), POSITIONS, PATCH)}
        truth, distill = train.loss_parts(pred, batch)
        t1, d1 = train.loss_parts(pred, batch | {"distill_mask": torch.ones(n, POSITIONS)})
        torch.testing.assert_close((t1, d1), (truth, distill))
        keep = torch.zeros(n, POSITIONS)
        keep[:, :40] = 1.0
        t2, d2 = train.loss_parts(pred, batch | {"distill_mask": keep})
        cut = {"targets": batch["targets"][:, :40], "target_mask": batch["target_mask"][:, :40],
               "teacher": batch["teacher"][:, :, :40]}
        _, want = train.loss_parts(pred[:, :40], cut)
        torch.testing.assert_close(t2, truth)
        torch.testing.assert_close(d2, want)


class RelativeWeights(unittest.TestCase):
    def test_a_closely_forecast_tile_weighs_more_and_the_loss_applies_the_weight_per_tile(self):
        from aximoe.student import SHARED
        torch.manual_seed(0)
        n = 4
        targets = torch.randn(n, POSITIONS, PATCH)
        mask = torch.ones(n, POSITIONS, PATCH)
        mask[1, :5] = 0.0
        levels = torch.tensor([LEVELS[k] for k in SHARED])
        spread = torch.tensor([0.05, 0.5, 2.0, 1.0])
        teacher = targets[:, None] + spread[:, None, None, None] * (levels[None, :, None, None] - 0.5)
        group = torch.tensor([True, True, True, False])
        batch = {"targets": targets, "target_mask": mask, "teacher": teacher, "relative": group}
        w = train.relative_weights(batch, 1.0, 0.1, 0.1)
        t = torch.zeros(n)
        for i in range(n):
            err = targets[i][:, None] - teacher[i].transpose(0, 1)
            tau = levels.reshape(1, -1, 1)
            pin = torch.maximum(err * tau, err * (tau - 1.0)) * mask[i][:, None]
            t[i] = pin.sum() / (mask[i].sum() * len(SHARED))
        raw = (t / 0.1 + 0.1) ** -1.0
        want = raw * t[group].sum() / (raw * t)[group].sum()
        torch.testing.assert_close(w[group], want[group], rtol=0, atol=1e-5)
        self.assertAlmostEqual(float(w[3]), 1.0, delta=1e-9)
        self.assertGreater(float(w[0]), float(w[1]))
        self.assertGreater(float(w[1]), float(w[2]))
        self.assertAlmostEqual(float((w * t)[group].sum()), float(t[group].sum()), delta=1e-5)

        pred = torch.randn(n, SEQ, len(LEVELS), PATCH)
        truth, distill = train.loss_parts(pred, batch | {"weight": w})
        truth_sum, distill_sum, points = 0.0, 0.0, 0.0
        for i in range(n):
            one = {k: v[i : i + 1] for k, v in batch.items()}
            ti, di = train.loss_parts(pred[i : i + 1], one)
            truth_sum += float(w[i] * ti * mask[i].sum())
            distill_sum += float(w[i] * di * mask[i].sum())
            points += float(mask[i].sum())
        self.assertAlmostEqual(float(truth), truth_sum / points, delta=1e-5)
        self.assertAlmostEqual(float(distill), distill_sum / points, delta=1e-5)


class AheadLabels(unittest.TestCase):
    def test_a_tile_unrolled_as_its_stored_forecast_learns_it_in_each_positions_own_space(self):
        from aximoe.student import SHARED
        rng = np.random.default_rng(1)
        n, span = 3, 30
        loc = rng.normal(5.0, 1.0, size=(n, POSITIONS))
        scale = rng.uniform(0.5, 2.0, size=(n, POSITIONS))
        own = rng.normal(size=(n, len(SHARED), POSITIONS, PATCH)).astype(np.float32)
        stored = rng.normal(size=(n, len(SHARED), span, PATCH)).astype(np.float32)
        # tile 0 is unrolled the 4 patches its forecast hides, tile 1 another depth, tile 2 holds no forecast
        j = torch.tensor([4, 2, 3])
        kept = (torch.arange(POSITIONS)[None] <= (POSITIONS - 1 - j)[:, None]).float()
        batch = {"teacher": torch.from_numpy(own), "ahead_q": torch.from_numpy(stored), "ahead_hidden": torch.tensor([4, 5, 0]),
                 "loc": torch.from_numpy(loc.astype(np.float32)), "scale": torch.from_numpy(scale.astype(np.float32)), "distill_mask": kept}
        out = train.ahead_labels(batch, j, own=True)
        cut = POSITIONS - 1 - 4
        raw = loc[0, cut] + scale[0, cut] * np.sinh(stored[0, :, span - 4:].astype(np.float64))
        want = np.arcsinh((raw - loc[0, None, cut + 1:, None]) / scale[0, None, cut + 1:, None])
        self.assertLess(float(np.abs(out["teacher"][0, :, cut + 1:].numpy() - want).max()), 1e-4)
        self.assertLess(float((out["teacher"][0, :, : cut + 1] - batch["teacher"][0, :, : cut + 1]).abs().max()), 1e-7)
        self.assertLess(float((out["teacher"][1:] - batch["teacher"][1:]).abs().max()), 1e-7)
        want_mask = kept.clone()
        want_mask[0] = 1.0
        self.assertLess(float((out["distill_mask"] - want_mask).abs().max()), 1e-7)
        # a second call for the masked tiles keeps the first call's weights
        again = train.ahead_labels(out | {"truth_weight": torch.full((n, POSITIONS), 0.5)}, torch.tensor([0, 0, 0]), 0.25)
        self.assertLess(float((again["truth_weight"] - 0.5).abs().max()), 1e-7)

    @unittest.skipUnless(PATCH == 32, "the tail mask takes far labels at the store's patch width")
    def test_a_tile_masked_as_its_stored_forecast_learns_it_after_the_cut(self):
        from aximoe.student import SHARED
        rng = np.random.default_rng(0)
        n, span = 5, 30
        values = rng.normal(5.0, 2.0, size=(n, 2048)).astype(np.float32)
        loc = rng.normal(5.0, 1.0, size=(n, POSITIONS)).astype(np.float32)
        scale = rng.uniform(0.5, 2.0, size=(n, POSITIONS)).astype(np.float32)
        inputs, input_mask, targets, target_mask = batches.standardise(values, loc, scale)
        own = rng.normal(size=(n, len(SHARED), POSITIONS, PATCH)).astype(np.float16)
        stored = rng.normal(size=(3, len(SHARED), span, PATCH)).astype(np.float16)
        stored[2, 0, 0, 0] = np.inf
        with tempfile.TemporaryDirectory() as d:
            # store tiles 7, 9 and 12 hold a forecast, with 5, 6 and 4 patches hidden
            np.array([7, 9, 12], "<i8").tofile(os.path.join(d, "tiles.i64"))
            np.array([5, 6, 4], np.uint8).tofile(os.path.join(d, "hidden.u8"))
            stored.tofile(os.path.join(d, "quantiles.f16"))
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"ahead": span, "model": "test"}, f)
            found = train.AheadTeacher([d], 20).lookup(np.array([7, 9, 3, -1, 12]))
        # the forecast with a non-finite quantile counts as none
        self.assertEqual(found["ahead_hidden"].tolist(), [5, 6, 0, 0, 0])
        np.testing.assert_allclose(found["ahead_q"][:2].numpy(), stored[:2], rtol=0, atol=1e-7)
        batch = {"inputs": torch.from_numpy(inputs), "input_mask": torch.from_numpy(input_mask),
                 "targets": torch.from_numpy(targets), "target_mask": torch.from_numpy(target_mask),
                 "values": torch.from_numpy(values), "loc": torch.from_numpy(loc), "scale": torch.from_numpy(scale),
                 "teacher": torch.from_numpy(own)} | found
        # tile 0 is masked as stored, tile 1 is left to the unroll, tiles 2 to 4 hold no forecast
        j = torch.tensor([5, 0, 3, 2, 4])
        out = train.ahead_labels(train.mask_tail(batch, j), j)
        cut = POSITIONS - 1 - 5
        np.testing.assert_allclose(out["teacher"][0, :, cut + 1:].numpy(), stored[0, :, span - 5:], rtol=0, atol=1e-7)
        np.testing.assert_allclose(out["teacher"][0, :, : cut + 1].numpy(), own[0, :, : cut + 1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(out["teacher"][1:].numpy(), own[1:], rtol=0, atol=1e-7)
        want = (torch.arange(POSITIONS)[None] <= (POSITIONS - 1 - j)[:, None]).float()
        want[0] = 1.0
        np.testing.assert_allclose(out["distill_mask"].numpy(), want.numpy(), rtol=0, atol=1e-7)
        # the truth's pinball is weighted down at the positions that learn the stored forecast, and nowhere else
        quarter = train.ahead_labels(train.mask_tail(batch, j), j, 0.25)
        weight = torch.ones(n, POSITIONS)
        weight[0, cut + 1:] = 0.25
        np.testing.assert_allclose(quarter["truth_weight"].numpy(), weight.numpy(), rtol=0, atol=1e-7)
        pred = torch.randn(n, SEQ, len(LEVELS), PATCH)
        plain = {k: v for k, v in quarter.items() if k != "truth_weight"}
        full, d0 = train.loss_parts(pred, plain)
        part, d1 = train.loss_parts(pred, quarter)
        only = plain | {"target_mask": plain["target_mask"] * (weight < 1.0).float()[:, :, None]}
        there, _ = train.loss_parts(pred, only)
        points = plain["target_mask"].sum()
        self.assertAlmostEqual(float(part), float(full - 0.75 * there * only["target_mask"].sum() / points), delta=1e-5)
        self.assertAlmostEqual(float(d1), float(d0), delta=1e-7)


@unittest.skipUnless(PATCH == 16, "runs at patches of 16 steps")
class FarLabelsAtSixteen(unittest.TestCase):
    def far_pass(self, directory: str, tiles: list[int], hidden: list[int], q: np.ndarray, shift: int | None) -> None:
        os.makedirs(directory)
        np.array(tiles, "<i8").tofile(os.path.join(directory, "tiles.i64"))
        np.array(hidden, np.uint8).tofile(os.path.join(directory, "hidden.u8"))
        q.astype("<f2").tofile(os.path.join(directory, "quantiles.f16"))
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"ahead": 30, "model": "test"} | ({} if shift is None else {"shift": shift}), f)

    def test_a_pass_of_either_shift_lands_on_the_positions_whose_steps_it_forecasts(self):
        # every level of a pass holds, at stored position k and step s, the number 32 k + s
        steps = (32 * np.arange(30)[:, None] + np.arange(32)[None]).astype(np.float32)
        plain = np.stack([steps, steps + 0.5])[:, None].repeat(9, axis=1)
        moved = -1.0 - plain
        loc = np.arange(2 * 63, dtype=np.float32).reshape(2, 63)
        with tempfile.TemporaryDirectory() as d:
            # tiles 3 and 5 hold a forecast made from step 0, tiles 5 and 8 one made from step 16
            self.far_pass(os.path.join(d, "a"), [3, 5], [2, 30], plain, None)
            self.far_pass(os.path.join(d, "b"), [5, 8], [4, 7], moved, 16)
            shifted = os.path.join(d, "shifted")
            os.makedirs(shifted)
            np.array([5, 8], "<i8").tofile(os.path.join(shifted, "tiles.i64"))
            np.zeros((2, 9, 63, 32), "<f2").tofile(os.path.join(shifted, "quantiles.f16"))
            loc.tofile(os.path.join(shifted, "loc.f32"))
            (loc + 1000).tofile(os.path.join(shifted, "scale.f32"))
            with open(os.path.join(shifted, "meta.json"), "w") as f:
                json.dump({"shift": 16}, f)
            os.environ["AXIMOE_SHIFT_TEACHER"] = shifted
            try:
                found = train.AheadTeacher([os.path.join(d, "a"), os.path.join(d, "b")], 20, seed=0).lookup(np.array([3, 8, 9, -1, 5]))
            finally:
                del os.environ["AXIMOE_SHIFT_TEACHER"]
        q = found["ahead_q"].numpy()
        self.assertEqual(q.shape, (5, 9, 60, PATCH))
        halves = (16 * np.arange(60)[:, None] + np.arange(16)[None]).astype(np.float32)
        # from step 0, 2 hidden patches of 32 steps are 5 of 16 and the forecasts start at position 67
        self.assertEqual((int(found["ahead_hidden"][0]), int(found["ahead_first"][0])), (5, 67))
        np.testing.assert_allclose(q[0, 4], halves, rtol=0, atol=1e-6)
        self.assertTrue(bool(torch.isnan(found["ahead_loc"][0])))
        # from step 16, 7 hidden patches are 14 of 16, the forecasts start one position later, in the pass's frame at its cut
        self.assertEqual((int(found["ahead_hidden"][1]), int(found["ahead_first"][1])), (14, 68))
        np.testing.assert_allclose(q[1, 4, 1:], -1.5 - halves[:-1], rtol=0, atol=1e-6)
        self.assertEqual(float(np.abs(q[1, :, 0]).max()), 0.0)
        self.assertEqual((float(found["ahead_loc"][1]), float(found["ahead_scale"][1])), (float(loc[1, 62 - 7]), float(loc[1, 62 - 7] + 1000)))
        self.assertEqual(found["ahead_hidden"][2:4].tolist(), [0, 0])
        # a tile both passes hold takes one of them whole
        both = (int(found["ahead_hidden"][4]), int(found["ahead_first"][4]))
        self.assertIn(both, ((61, 67), (8, 68)))
        if both == (61, 67):
            np.testing.assert_allclose(q[4, 4], halves + 0.5, rtol=0, atol=1e-6)
        else:
            np.testing.assert_allclose(q[4, 4, 1:], -1.0 - halves[:-1], rtol=0, atol=1e-6)
            self.assertEqual(float(found["ahead_loc"][4]), float(loc[0, 62 - 4]))

    def test_an_unrolled_tile_learns_its_forecast_from_the_first_position_that_holds_one(self):
        from aximoe.student import SHARED
        rng = np.random.default_rng(2)
        n, span = 3, 60
        loc = rng.normal(5.0, 1.0, size=(n, POSITIONS))
        scale = rng.uniform(0.5, 2.0, size=(n, POSITIONS))
        own = rng.normal(size=(n, len(SHARED), POSITIONS, PATCH)).astype(np.float32)
        stored = rng.normal(size=(n, len(SHARED), span, PATCH)).astype(np.float32)
        # tile 0 holds a forecast from step 0, tiles 1 and 2 one from step 16 in the pass's own frame
        j = torch.tensor([5, 14, 60])
        kept = (torch.arange(POSITIONS)[None] <= (POSITIONS - 1 - j)[:, None]).float()
        space_loc, space_scale = np.array([np.nan, 2.5, -1.0]), np.array([np.nan, 1.5, 0.75])
        batch = {"teacher": torch.from_numpy(own), "ahead_q": torch.from_numpy(stored), "ahead_hidden": j.clone(),
                 "ahead_first": torch.tensor([67, 68, 68]), "ahead_loc": torch.from_numpy(space_loc.astype(np.float32)),
                 "ahead_scale": torch.from_numpy(space_scale.astype(np.float32)),
                 "loc": torch.from_numpy(loc.astype(np.float32)), "scale": torch.from_numpy(scale.astype(np.float32)), "distill_mask": kept}
        out = train.ahead_labels(batch, j, own=True)
        for row, first in ((0, 67), (1, 68), (2, 68)):
            cut = POSITIONS - 1 - int(j[row])
            start = max(cut + 1, first)
            frame = (loc[row, cut], scale[row, cut]) if row == 0 else (space_loc[row], space_scale[row])
            raw = frame[0] + frame[1] * np.sinh(stored[row, :, start - 67:].astype(np.float64))
            want = np.arcsinh((raw - loc[row, None, start:, None]) / scale[row, None, start:, None])
            self.assertLess(float(np.abs(out["teacher"][row, :, start:].numpy() - want).max()), 1e-4)
            self.assertLess(float((out["teacher"][row, :, :start] - batch["teacher"][row, :, :start]).abs().max()), 1e-7)
            want_mask = kept[row].clone()
            want_mask[start:] = 1.0
            self.assertLess(float((out["distill_mask"][row] - want_mask).abs().max()), 1e-7)


@unittest.skipUnless(PATCH == 16, "runs at patches of 16 steps")
class RegroupAtSixteen(unittest.TestCase):
    def test_forecasts_of_the_truth_at_16_steps_are_the_truth_on_the_stores_grid(self):
        from aximoe import reference
        rng = np.random.default_rng(3)
        values = rng.normal(5.0, 2.0, size=(2, 2048)).cumsum(axis=1).astype(np.float32)
        own_loc, own_scale = batches.causal_stats(values)
        # the store's statistics are those at the end of every second patch of 16 steps
        loc, scale = own_loc[:, 1::2][:, :63].astype(np.float64), own_scale[:, 1::2][:, :63].astype(np.float64)
        truth = values.reshape(2, SEQ, PATCH)[:, 1:].astype(np.float64)
        q = np.arcsinh((truth - own_loc[:, :POSITIONS, None]) / own_scale[:, :POSITIONS, None])
        q = np.repeat(q[:, :, None], 3, axis=2).astype(np.float32)
        out = train.regroup(q, values, loc, scale)
        want, _ = reference.standardise(values, loc, scale)
        self.assertEqual(out.shape, (2, 63, 3, 32))
        for level in range(3):
            np.testing.assert_allclose(out[:, :, level], want, rtol=0, atol=1e-4)


@unittest.skipUnless(PATCH == 32, "starts the tests of the 16-step grid")
class AtSixteenSteps(unittest.TestCase):
    def test_the_tests_of_the_16_step_grid_pass(self):
        names = ["FarLabelsAtSixteen", "RegroupAtSixteen", "Unroll.test_frozen_feedback_trains_what_the_forecaster_shows",
                 "Unroll.test_own_statistics_are_those_the_forecaster_computes",
                 "AheadLabels.test_a_tile_unrolled_as_its_stored_forecast_learns_it_in_each_positions_own_space"]
        done = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), *names],
                              env={**os.environ, "AXIMOE_PATCH": "16"}, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        self.assertIn("Ran 6 tests", done.stderr)


class TakenStream:
    """A stream of generated tiles that hands out the same tiles to every twin of a seed."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def take(self, n: int) -> tuple:
        return self.rng.normal(size=(n, SEQ * PATCH)).cumsum(axis=1).astype(np.float32), self.rng.choice([10, 60], size=n)


@unittest.skipUnless(PATCH == 32, "runs on the store's grid")
class BuilderBatches(unittest.TestCase):
    def sources(self, d: str, seed: int):
        rng = np.random.default_rng(100 + seed)
        values = rng.normal(size=(40, SEQ * PATCH)).cumsum(axis=1).astype(np.float32)
        values[rng.random(values.shape) < 0.1] = np.nan
        stored = np.full((40, 63), np.nan, np.float32)
        quantiles = rng.normal(size=(40, 9, 63, 32)).astype(np.float16)
        store = types.SimpleNamespace(rows=lambda ids: {"values": values[ids], "loc": stored[ids], "scale": stored[ids],
                                                        "quantiles": quantiles[ids]})
        tiles = np.zeros(40, dtype=batches.RECORD)
        tiles["width"] = np.where(np.arange(40) % 2, 10, 60)
        tiles["channel"] = np.arange(40) % 3
        index = types.SimpleNamespace(tiles=tiles, channels=["cpu", "memory", "disk"], splits=["train"], meta={},
                                      with_split=lambda split: np.arange(40))
        far = os.path.join(d, f"far{seed}")
        if not os.path.exists(far):
            os.makedirs(far)
            held = np.arange(0, 40, 3)
            held.astype("<i8").tofile(os.path.join(far, "tiles.i64"))
            rng.integers(1, 30, len(held)).astype(np.uint8).tofile(os.path.join(far, "hidden.u8"))
            rng.normal(size=(len(held), 9, 30, 32)).astype("<f2").tofile(os.path.join(far, "quantiles.f16"))
            with open(os.path.join(far, "meta.json"), "w") as f:
                json.dump({"ahead": 30, "model": "test"}, f)
        return store, index, batches.Tags(index), train.AheadTeacher([far], 40, seed=seed)

    def test_a_batch_assembled_from_its_draws_is_the_batch_the_trainer_makes_in_one_go(self):
        # the property: for any seed, every batch of a run is the same number for number, since a job carries every draw
        with tempfile.TemporaryDirectory() as d:
            for seed in range(4):
                store, index, tags, ahead = self.sources(d, seed)
                sampler, tag_rng, stream = batches.Sampler(index, seed), batches.Lcg(seed + 1), TakenStream(seed)
                twin = (batches.Sampler(index, seed), batches.Lcg(seed + 1), TakenStream(seed), self.sources(d, seed)[3])
                train.BUILD.clear()
                train.BUILD.update(store=store, index=index, tags=tags, ahead=self.sources(d, seed)[3], tail=None, raw=True,
                                   tag_dropout=0.5, empty_context_loss=False)
                for _ in range(5):
                    want = batches.make_batch(store, index, tags, sampler.next(11), (0.5, tag_rng), raw=True)
                    want.setdefault("distill_mask", torch.ones(11, POSITIONS))
                    taken, widths = stream.take(5)
                    want = batches.join(want, batches.synthetic_batch(taken, widths, tags, want["teacher"].dtype, (0.5, tag_rng), raw=True))
                    want = batches.drop_empty_contexts(want)
                    want = want | ahead.lookup(want["tiles"])
                    got = {k: torch.from_numpy(v) if tensor else v
                           for k, v, tensor in train.assemble_batch(train.draw_batch(*twin, batch=16, streamed=5))}
                    self.assertEqual(set(got), set(want))
                    for k, v in want.items():
                        if isinstance(v, torch.Tensor):
                            self.assertEqual(got[k].dtype, v.dtype, k)
                            self.assertTrue(torch.equal(got[k].nan_to_num(nan=7.0), v.nan_to_num(nan=7.0)), k)
                        else:
                            np.testing.assert_array_equal(got[k], v, k)
                # after the same batches the twins' generators stand where the trainer's stand
                self.assertEqual((twin[0].rng.state, twin[1].state), (sampler.rng.state, tag_rng.state))
                self.assertEqual(twin[3].rng.bit_generator.state, ahead.rng.bit_generator.state)


class LevelLoss(unittest.TestCase):
    def test_distance_to_the_target_patch_median_against_a_loop(self):
        torch.manual_seed(0)
        n = 3
        targets = torch.randn(n, POSITIONS, PATCH)
        mask = (torch.rand(n, POSITIONS, PATCH) > 0.3).float()
        mask[0, 5] = 0.0
        mask[1, 7, 5:] = 0.0
        level = torch.randn(n, SEQ)
        got = train.level_loss(level, {"targets": targets * mask, "target_mask": mask})
        total, count = 0.0, 0
        for i in range(n):
            for s in range(POSITIONS):
                held = targets[i, s][mask[i, s] > 0]
                if len(held) < PATCH // 4:
                    continue
                # torch's median of an even count is the lower of the two middle values
                total += abs(float(level[i, s]) - float(held.sort().values[(len(held) - 1) // 2]))
                count += 1
        self.assertAlmostEqual(float(got), total / count, delta=1e-5)
        weighted = train.level_loss(level, {"targets": targets * mask, "target_mask": mask, "weight": torch.tensor([1.0, 0.0, 0.0])})
        alone = train.level_loss(level[:1], {"targets": (targets * mask)[:1], "target_mask": mask[:1]})
        self.assertAlmostEqual(float(weighted), float(alone), delta=1e-5)


if __name__ == "__main__":
    unittest.main()
