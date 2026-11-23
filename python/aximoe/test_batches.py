"""The training sampler draws widths by their share when asked, and by tile count otherwise;
synthetic tiles are drawn at their own share and never without one. A stored tile is framed by
statistics read at the end of each of the model's patches. At patches of 16 steps the teacher
labels come from the store at odd positions and from the shifted pass at even ones; the tests of
that grid run in a process started with `AXIMOE_PATCH=16`, which the default run starts."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches
from aximoe.student import PATCH, POSITIONS, SEQ


class FakeIndex:
    """300 training tiles: 150 at 10 s, 100 at 30 s, 50 at 60 s, one series each."""

    def __init__(self):
        widths = np.array([10] * 150 + [30] * 100 + [60] * 50, dtype=np.uint16)
        self.tiles = np.zeros(len(widths), dtype=batches.RECORD)
        self.tiles["width"] = widths
        self.tiles["series"] = np.arange(len(widths))
        self.splits = ["train"]
        self.meta = {"envs": ["prod", batches.SYNTHETIC]}

    def with_split(self, split):
        return np.flatnonzero(self.tiles["split"] == self.splits.index(split))


class WidthShares(unittest.TestCase):
    def test_equal_shares_draw_each_width_a_third(self):
        index = FakeIndex()
        draws = batches.Sampler(index, 0, 1.0, {10: 1, 30: 1, 60: 1}).next(3000)
        widths = index.tiles["width"][draws]
        for w in (10, 30, 60):
            self.assertAlmostEqual(float(np.mean(widths == w)), 1 / 3, delta=0.05)
        self.assertTrue(np.all(index.tiles["split"][draws] == 0))

    def test_no_shares_draw_by_tile_count(self):
        index = FakeIndex()
        widths = index.tiles["width"][batches.Sampler(index, 0).next(3000)]
        self.assertAlmostEqual(float(np.mean(widths == 10)), 0.5, delta=0.05)
        self.assertAlmostEqual(float(np.mean(widths == 60)), 1 / 6, delta=0.05)


class SyntheticShare(unittest.TestCase):
    def index(self):
        index = FakeIndex()
        index.tiles["env"][:30] = 1
        return index

    def test_synthetic_tiles_are_drawn_at_their_share(self):
        index = self.index()
        draws = batches.Sampler(index, 0, 1.0, {10: 1, 30: 1, 60: 1}, synthetic_share=0.2).next(4000)
        synthetic = index.tiles["env"][draws] == 1
        self.assertAlmostEqual(float(synthetic.mean()), 0.2, delta=0.03)
        widths = index.tiles["width"][draws[~synthetic]]
        for w in (10, 30, 60):
            self.assertAlmostEqual(float(np.mean(widths == w)), 1 / 3, delta=0.05)

    def test_no_share_draws_no_synthetic_tile(self):
        index = self.index()
        for shares in (None, {10: 1, 30: 1, 60: 1}):
            draws = batches.Sampler(index, 0, 1.0, shares).next(3000)
            self.assertEqual(int((index.tiles["env"][draws] == 1).sum()), 0)


class Generated(unittest.TestCase):
    def test_the_vectorised_scaler_is_the_forecasters(self):
        from aximoe import forecast

        rng = np.random.default_rng(1)
        values = (rng.normal(3.0, 2.0, size=(5, 2048)) * np.exp(rng.normal(size=(5, 1)) * 5)).astype(np.float32)
        values[rng.random(values.shape) < 0.1] = np.nan
        values[2, :700] = np.nan
        loc, scale = batches.causal_stats(values)
        want_loc, want_scale = forecast.causal_stats(values)
        np.testing.assert_allclose(loc, want_loc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(scale, want_scale, rtol=1e-6, atol=0)

    def test_generated_tiles_learn_from_the_truth_on_the_unknown_row(self):
        import torch

        index = FakeIndex()
        index.channels = ["cpu_usage_rate"]
        tags = batches.Tags(index)
        rng = np.random.default_rng(2)
        values = rng.normal(size=(4, 2048)).astype(np.float32)
        b = batches.synthetic_batch(values, np.array([10, 30, 60, 3600]), tags, torch.float16)
        self.assertAlmostEqual(float(b["distill_mask"].abs().sum()), 0.0, delta=1e-9)
        self.assertAlmostEqual(float(b["teacher"].float().abs().sum()), 0.0, delta=1e-9)
        self.assertEqual((b["tags"] % batches.TAG_BASE).tolist(), [tags.unknown()] * 4)
        self.assertEqual((b["tags"] // batches.TAG_BASE - 1).tolist(), [0, 1, 2, tags.resolution_unknown()])


    def test_a_late_start_leaves_a_partly_filled_first_patch_and_a_target_after_it(self):
        from aximoe import synthetic

        rng = np.random.default_rng(3)
        for _ in range(200):
            y = synthetic.late_start(rng, np.ones(2048, np.float32), PATCH)
            _, input_mask, _, target_mask = batches.standardise(y[None], *[s[:, :POSITIONS] for s in batches.causal_stats(y[None])])
            held = input_mask[0].sum(1)
            first = int(np.argmax(held > 0))
            self.assertTrue(2 <= held[first] <= PATCH - 1)
            self.assertAlmostEqual(float(input_mask[0, first, -int(held[first]):].sum()), float(held[first]), delta=1e-9)
            self.assertAlmostEqual(float(held[first + 1:].min()), float(PATCH), delta=1e-9)
            self.assertAlmostEqual(float(target_mask[0, first].sum()), float(PATCH), delta=1e-9)


    def test_the_stream_starts_a_tile_late_inside_a_patch_of_the_models_width(self):
        from aximoe import synthetic

        class Once:
            def put(self, item):
                raise StopIteration

        seen, original = [], synthetic.late_start
        synthetic.late_start = lambda rng, y, patch=None: seen.append(patch) or original(rng, y, patch)
        try:
            with self.assertRaises(StopIteration):
                batches._generate(0, 0, 4, "any", Once(), late=1.0)
        finally:
            synthetic.late_start = original
        self.assertEqual(seen, [PATCH] * 4)

    def test_a_position_with_an_empty_context_loses_its_target(self):
        import torch

        index = FakeIndex()
        index.channels = ["cpu_usage_rate"]
        tags = batches.Tags(index)
        values = np.ones((2, 2048), np.float32)
        values[0, : 5 * PATCH - 4] = np.nan
        b = batches.drop_empty_contexts(batches.synthetic_batch(values, np.array([60, 60]), tags, torch.float16))
        per_position = b["target_mask"].sum(2)
        self.assertAlmostEqual(float(per_position[0, :4].sum()), 0.0, delta=1e-9)
        self.assertAlmostEqual(float(per_position[0, 4:].min()), float(PATCH), delta=1e-9)
        self.assertAlmostEqual(float(per_position[1].min()), float(PATCH), delta=1e-9)


class AheadTargets(unittest.TestCase):
    def test_patches_ahead_align_with_the_next_patch_targets(self):
        import torch

        rng = np.random.default_rng(0)
        values = rng.normal(5.0, 2.0, size=(3, 2048)).astype(np.float32)
        values[rng.random(values.shape) < 0.1] = np.nan
        loc = rng.normal(5.0, 1.0, size=(3, 63)).astype(np.float32)
        scale = rng.uniform(0.5, 2.0, size=(3, 63)).astype(np.float32)
        _, _, targets, target_mask = batches.standardise(values, loc, scale)
        ys, ms = batches.ahead_targets(torch.from_numpy(values), torch.from_numpy(loc), torch.from_numpy(scale), 3)
        ys, ms = ys.numpy(), ms.numpy()
        np.testing.assert_allclose(ys[:, :, :32], targets, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(ms[:, :, :32], target_mask)
        # position 10, third patch ahead: patch 13 in the statistics at the end of patch 10
        raw = values[:, 13 * 32 : 14 * 32]
        want = np.arcsinh((raw - loc[:, 10, None]) / scale[:, 10, None])
        present = np.isfinite(raw)
        np.testing.assert_allclose(ys[:, 10, 64:][present], want[present], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(ms[:, 10, 64:], present)
        # the last position's patches past the tile are masked
        self.assertEqual(ms[:, 62, 32:].sum(), 0)


class OwnGrid(unittest.TestCase):
    def test_a_stored_tile_is_framed_by_statistics_computed_at_the_end_of_each_model_patch(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(3, SEQ * PATCH)).cumsum(axis=1).astype(np.float32)
        values[1, 40:90] = np.nan
        tiles = np.zeros(3, dtype=batches.RECORD)
        tiles["width"] = 60
        index = types.SimpleNamespace(tiles=tiles, channels=["cpu"], splits=["train"], with_split=lambda split: np.arange(3))
        batch = batches.make_batch(stand_in_store(values), index, batches.Tags(index), np.arange(3), raw=True)
        loc, scale = (stat[:, :POSITIONS] for stat in batches.causal_stats(values))
        for name, want in zip(("inputs", "input_mask", "targets", "target_mask"), batches.standardise(values, loc, scale)):
            np.testing.assert_allclose(batch[name].numpy(), want, rtol=0, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(batch["loc"].numpy(), loc, rtol=0, atol=1e-6)
        np.testing.assert_allclose(batch["scale"].numpy(), scale, rtol=0, atol=1e-6)


def stand_in_store(values: np.ndarray):
    """A store of `values` whose statistics columns hold no number and whose forecasts are zero."""
    stored = np.full((len(values), 63), np.nan, np.float32)
    return types.SimpleNamespace(rows=lambda ids: {"values": values[ids], "loc": stored[ids], "scale": stored[ids],
                                                   "quantiles": np.zeros((len(ids), 9, 63, 32), np.float16)})


@unittest.skipUnless(PATCH == 16, "runs at patches of 16 steps")
class FineTeacher(unittest.TestCase):
    def shifted_pass(self, directory: str, tiles: np.ndarray, q: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> None:
        tiles.astype("<i8").tofile(os.path.join(directory, "tiles.i64"))
        q.astype("<f2").tofile(os.path.join(directory, "quantiles.f16"))
        loc.astype("<f4").tofile(os.path.join(directory, "loc.f32"))
        scale.astype("<f4").tofile(os.path.join(directory, "scale.f32"))
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"shift": 16}, f)

    def with_pass(self, directory: str, call):
        os.environ["AXIMOE_SHIFT_TEACHER"], batches._shift_labels = directory, None
        try:
            return call()
        finally:
            del os.environ["AXIMOE_SHIFT_TEACHER"]
            batches._shift_labels = None

    def test_odd_positions_take_the_stores_forecast_and_even_ones_the_shifted_pass_in_their_own_frame(self):
        rng = np.random.default_rng(0)
        tiles = np.array([9, 7, 4])
        loc = rng.normal(size=(3, POSITIONS)).astype(np.float32)
        scale = rng.uniform(0.5, 2.0, size=(3, POSITIONS)).astype(np.float32)
        stored = rng.normal(size=(3, 9, 63, 32)).astype(np.float16)
        # the pass holds tiles 4 and 9: forecasts in series units, kept in the pass's own frames
        raw = rng.normal(size=(2, 9, 63, 32))
        pass_loc = rng.normal(size=(2, 63)).astype(np.float32)
        pass_scale = rng.uniform(0.5, 2.0, size=(2, 63)).astype(np.float32)
        q = np.arcsinh((raw - pass_loc[:, None, :, None]) / pass_scale[:, None, :, None])
        q[1, :, 5] = np.inf
        with tempfile.TemporaryDirectory() as d:
            self.shifted_pass(d, np.array([4, 9]), q, pass_loc, pass_scale)
            out, mask = self.with_pass(d, lambda: batches.fine_teacher(tiles, stored, loc, scale))
        self.assertEqual(out.shape, (3, 9, POSITIONS, PATCH))
        np.testing.assert_allclose(out[:, :, 1::2], stored[:, :, :, :PATCH], rtol=0, atol=1e-6)
        self.assertTrue(bool((mask[:, 1::2] == 1).all()) and bool((mask[:, 0] == 0).all()))
        even = np.arange(2, POSITIONS, 2)
        for row, held in ((2, 0), (0, 1)):
            want = np.arcsinh((raw[held, :, :, :PATCH] - loc[row, even][None, :, None]) / scale[row, even][None, :, None])
            finite = np.arange(63) != 5 if held == 1 else np.ones(63, bool)
            # the pass and the labels are float16
            np.testing.assert_allclose(out[row][:, even[finite]], want[:, finite], rtol=0, atol=2e-2)
            self.assertTrue(bool((mask[row, even[finite]] == 1).all()))
        self.assertEqual(float(mask[0, even[5]]), 0.0)
        self.assertEqual(float(np.abs(out[0][:, even[5]]).max()), 0.0)
        self.assertEqual(float(mask[1, even].max()), 0.0)

    def test_a_batch_holds_the_labels_mask_and_a_zero_mask_without_a_pass(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(2, SEQ * PATCH)).cumsum(axis=1).astype(np.float32)
        tiles = np.zeros(2, dtype=batches.RECORD)
        tiles["width"] = 60
        index = types.SimpleNamespace(tiles=tiles, channels=["cpu"], splits=["train"], with_split=lambda split: np.arange(2))
        batch = batches.make_batch(stand_in_store(values), index, batches.Tags(index), np.arange(2))
        self.assertEqual(tuple(batch["teacher"].shape), (2, 9, POSITIONS, PATCH))
        self.assertEqual(float(batch["distill_mask"].max()), 0.0)
        with tempfile.TemporaryDirectory() as d:
            self.shifted_pass(d, np.array([1]), np.zeros((1, 9, 63, 32)), np.zeros((1, 63)), np.ones((1, 63)))
            batch = self.with_pass(d, lambda: batches.make_batch(stand_in_store(values), index, batches.Tags(index), np.arange(2)))
        self.assertEqual(batch["distill_mask"][:, 1::2].min().item(), 1.0)
        self.assertEqual(batch["distill_mask"][0, 2::2].max().item(), 0.0)
        self.assertEqual(batch["distill_mask"][1, 2::2].min().item(), 1.0)


STARTED: dict = {}


def note_start(spec: dict) -> None:
    """Keeps what a builder process was started with."""
    STARTED.update(spec)


def slow_echo(job: dict) -> list:
    """The job's number after the job's own delay, with the mark its process was started with."""
    time.sleep(job["delay"])
    return [("number", np.array([job["number"]]), True), ("mark", STARTED["mark"], False)]


class DrawnDraws(unittest.TestCase):
    def test_units_come_back_in_the_order_they_were_drawn(self):
        made = batches.Lcg(3)
        units = [made.unit() for _ in range(40)]
        drawn = batches.Drawn(units)
        self.assertEqual([drawn.unit() for _ in range(40)], units)

    def test_sides_come_back_for_the_size_they_were_drawn_at(self):
        sides = np.random.default_rng(0).integers(0, 2, 9)
        drawn = batches.Drawn([], sides)
        np.testing.assert_array_equal(drawn.integers(0, 2, 9), sides)
        with self.assertRaises(ValueError):
            drawn.integers(0, 2, 8)


class BuildersOrder(unittest.TestCase):
    def test_batches_come_out_in_the_order_of_their_jobs_whatever_each_takes(self):
        # the property: neither the count of processes nor how long a job takes changes the order or what rides along
        for workers, seed in ((1, 0), (2, 1), (4, 2)):
            delays = np.random.default_rng(seed).uniform(0.0, 0.02, 30)
            numbers = iter(range(30))

            def draw():
                k = next(numbers)
                return {"number": k, "delay": float(delays[k])}, {"resume": k}

            pool = batches.Builders(workers, note_start, {"mark": f"mark {seed}"}, slow_echo)
            try:
                out = [pool.next(draw) for _ in range(30 - pool.ahead)]
            finally:
                pool.close()
            self.assertEqual([int(b["number"][0]) for b in out], list(range(len(out))))
            self.assertEqual([b["resume"] for b in out], list(range(len(out))))
            self.assertTrue(all(b["mark"] == f"mark {seed}" for b in out))
            self.assertTrue(all(isinstance(b["number"], torch.Tensor) for b in out))


def echo_weights(mark: float, link) -> None:
    """Prepares, for each step asked for, the step's number with the weights it read and `mark`."""
    read = {}
    link.serve(lambda weights: read.update(weights=weights.clone()),
               lambda step: {"step": step, "weights": read["weights"], "mark": mark})


def fail_at_three(mark: float, link) -> None:
    def prepare(step: int) -> dict:
        if step == 3:
            raise ValueError("no fourth batch")
        return {"step": step}

    link.serve(lambda weights: None, prepare)


class AheadWeights(unittest.TestCase):
    def test_a_batch_is_prepared_from_the_weights_two_steps_before_it_whatever_the_timing(self):
        # the property: batch s reads the weights published after step s - 2, the starting ones for the first two,
        # however long a step or a preparation takes
        for seed, first in ((0, 0), (1, 7)):
            waits = np.random.default_rng(seed).uniform(0.0, 0.01, 12)
            ahead = batches.Ahead(echo_weights, (float(seed),), torch.full((5,), -1.0), first)
            try:
                for k in range(12):
                    made = ahead.next()
                    self.assertEqual((made["step"], made["mark"]), (first + k, float(seed)))
                    self.assertTrue(torch.equal(made["weights"], torch.full((5,), float(k - 2) if k >= 2 else -1.0)), k)
                    time.sleep(float(waits[k]))
                    ahead.publish(first + k, torch.full((5,), float(k)))
            finally:
                ahead.close()
            self.assertFalse(ahead.process.is_alive())

    def test_the_trainer_stops_when_the_preparing_process_ends(self):
        ahead = batches.Ahead(fail_at_three, (0.0,), torch.zeros(2), 0)
        try:
            with self.assertRaises(RuntimeError):
                for k in range(6):
                    ahead.next()
                    ahead.publish(k, torch.zeros(2))
        finally:
            ahead.close()
        self.assertFalse(ahead.process.is_alive())


@unittest.skipUnless(PATCH == 32, "starts the tests of the 16-step grid")
class AtSixteenSteps(unittest.TestCase):
    def test_the_tests_of_the_16_step_grid_pass(self):
        done = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "FineTeacher", "Generated", "OwnGrid"],
                              env={**os.environ, "AXIMOE_PATCH": "16"}, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        self.assertIn("Ran 8 tests", done.stderr)


if __name__ == "__main__":
    unittest.main()
