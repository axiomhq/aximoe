"""Test of the MCP server's rollout against the forecaster's.

    python -m unittest discover -s python -p "test_investigate.py" -v
    python -I python/aximoe/test_investigate.py
"""
import asyncio
import datetime as dt
import os
import sys
import unittest
from unittest import mock

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aximoe import batches, forecast, investigate, student
from aximoe.student import CONTEXT, MEDIAN, PATCH, SEQ


class BandRollout(unittest.TestCase):
    def test_second_patch_is_the_forecasters_patch_with_the_band_fed_back(self):
        torch.manual_seed(0)
        model = student.Student(student.StudentConfig(d_model=32, heads=2, layers=1, d_ff=64, band_input=True)).eval()
        cpu = torch.device("cpu")
        detector = investigate.Model.__new__(investigate.Model)
        detector.device, detector.student = cpu, model
        detector.description = {"config": {"tags": 0}, "channels": {}, "widths": []}
        # 62 context patches put the second forecast patch at the forecaster's last position;
        # normal draws go below zero, so the floor stays off
        contexts = np.random.default_rng(0).normal(size=(3, CONTEXT - PATCH)).astype(np.float32)
        out = np.stack(detector.forecast([("x", c) for c in contexts], 2))
        fed = np.concatenate([contexts, out[:, :PATCH, 1].astype(np.float32)], axis=1)
        lo, hi = np.full_like(fed, np.nan), np.full_like(fed, np.nan)
        lo[:, -PATCH:], hi[:, -PATCH:] = out[:, :PATCH, 0], out[:, :PATCH, 2]
        tags = np.full(len(contexts), student.tag_id(detector.description, "x", investigate.Model.BIN_SECONDS))
        q, loc, scale = forecast.next_patch(model, fed, tags, cpu, band=(lo, hi))
        second = loc[:, None, None] + scale[:, None, None] * np.sinh(q.astype(np.float64))
        levels = [student.LEVELS.index(v) for v in (0.1, 0.5, 0.9)]
        np.testing.assert_allclose(out[:, PATCH:], second[:, levels].transpose(0, 2, 1), rtol=1e-5, atol=1e-5)
        # without the band the second patch differs: the band reaches the model
        q0, _, _ = forecast.next_patch(model, fed, tags, cpu)
        self.assertFalse(np.allclose(q0, q))


def full_pass_rollout(detector, contexts, patches):
    """The rollout with every patch read from a full pass over the 64 positions of its tile."""
    model, n, length = detector.student, contexts.shape[0], contexts.shape[1]
    seqs, band = contexts.copy(), np.full((2, n, length), np.nan, np.float32)
    eligible = forecast.floor_eligible(seqs)
    tags = torch.tensor([student.tag_id(detector.description, "x", detector.BIN_SECONDS)] * n, dtype=torch.int64)
    out = np.zeros((n, patches * PATCH, 3))
    for r in range(patches):
        pos = length // PATCH + r - 1
        tile = np.full((n, SEQ * PATCH), np.nan, np.float32)
        tile[:, : seqs.shape[1]] = seqs
        loc, scale = forecast.causal_stats(tile[:, : (SEQ - 1) * PATCH])
        inputs, mask, _, _ = batches.standardise(tile, loc, scale)
        offsets = None
        if model.cfg.band_input:
            lo, hi = np.full_like(tile, np.nan), np.full_like(tile, np.nan)
            lo[:, : band.shape[2]], hi[:, : band.shape[2]] = band
            offsets = torch.from_numpy(forecast.band_offsets(inputs, lo, hi, loc, scale))
        with torch.no_grad():
            q = model(torch.from_numpy(inputs), torch.from_numpy(mask), tags, offsets)[:, pos].float().numpy()
        raw = loc[:, pos, None, None].astype(np.float64) + scale[:, pos, None, None] * np.sinh(q.astype(np.float64))
        raw = forecast.floor_at_zero(raw, eligible)
        out[:, r * PATCH : (r + 1) * PATCH] = raw[:, [investigate.Q10, MEDIAN, investigate.Q90]].transpose(0, 2, 1)
        seqs = np.concatenate([seqs, raw[:, MEDIAN].astype(np.float32)], axis=1)
        band = np.concatenate([band, raw[:, [investigate.Q10, investigate.Q90]].astype(np.float32).transpose(1, 0, 2)], axis=2)
    return out


class Rollout(unittest.TestCase):
    def test_matches_full_passes_of_the_whole_tile(self):
        for band_input in (True, False):
            with self.subTest(band_input=band_input):
                torch.manual_seed(0)
                config = student.StudentConfig(d_model=32, heads=2, layers=2, d_ff=64, band_input=band_input)
                detector = investigate.Model.__new__(investigate.Model)
                detector.device, detector.student = torch.device("cpu"), student.Student(config).eval()
                detector.description = {"config": {"tags": 0}, "channels": {}, "widths": []}
                # more series than a batch holds, so each batch keeps its own state
                detector.BATCH = 3
                rng = np.random.default_rng(0)
                contexts = (rng.normal(size=(7, 15 * PATCH)) + 3.0).astype(np.float32)
                contexts[rng.random(contexts.shape) < 0.1] = np.nan
                out = np.stack(detector.forecast([("x", c) for c in contexts], 4))
                np.testing.assert_allclose(out, full_pass_rollout(detector, contexts, 4), rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs an MPS device")
    def test_on_mps_matches_full_passes_on_the_cpu(self):
        # MPS has no float64, which the scaler and the inverse transform keep
        torch.manual_seed(0)
        config = student.StudentConfig(d_model=32, heads=2, layers=2, d_ff=64, band_input=True)
        detector = investigate.Model.__new__(investigate.Model)
        detector.device, detector.student = torch.device("cpu"), student.Student(config).eval()
        detector.description = {"config": {"tags": 0}, "channels": {}, "widths": []}
        rng = np.random.default_rng(0)
        # values near 1e9, like memory bytes, where float32 sums lose the deviation
        contexts = (rng.normal(size=(5, 15 * PATCH)) * 1e6 + 1e9).astype(np.float32)
        contexts[rng.random(contexts.shape) < 0.1] = np.nan
        expected = full_pass_rollout(detector, contexts, 4)
        detector.device, detector.student = torch.device("mps"), detector.student.to("mps")
        out = np.stack(detector.forecast([("x", c) for c in contexts], 4))
        np.testing.assert_allclose(out, expected, rtol=1e-4)


class Recorder:
    """Stands in for `investigate.Model` and keeps the contexts it is asked to forecast."""

    def __init__(self):
        self.items = []

    def forecast(self, items, patches):
        self.items = items
        return [np.tile(np.array([0.0, 1.0, 2.0]), (patches * PATCH, 1)) for _ in items]


class ContextGate(unittest.TestCase):
    def test_a_context_with_an_empty_newest_patch_is_not_forecast(self):
        s = investigate.Settings()
        length, start = 30, s.ref_model + 90
        steps = start + length
        grid = investigate.Grid(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), steps)
        # each series holds its own minute plus 10,000 times its number, so a context names its window
        full, gap32, gap31, history = (np.arange(steps, dtype=np.float64) + 10_000 * k for k in range(4))
        gap32[start - PATCH : start] = np.nan
        gap31[start - PATCH + 1 : start] = np.nan
        history[start - 60 - PATCH : start - 60] = np.nan
        for k, values in enumerate((full, gap32, gap31, history)):
            grid.series[(f"ns/c{k}", 0)] = values
        outcomes = [investigate.Outcome(f"ns/c{k}", 0, investigate.Scored()) for k in range(4)]
        model = Recorder()
        investigate.aximoe_scores(grid, start, length, s, model, outcomes)
        self.assertEqual(outcomes[1].aximoe.skipped, f"no data in the {PATCH} minutes before the window")
        for k in (0, 2, 3):
            self.assertIsNone(outcomes[k].aximoe.skipped)
            self.assertTrue(outcomes[k].aximoe.bins)
        seen: dict[int, list[int]] = {}
        for _, context in model.items:
            seen.setdefault(int(context[0]) // 10_000, []).append(int(context[0]) % 10_000 + s.ref_model)
        # the scanned window at 570 and the history windows before it, less those with an empty newest patch
        self.assertEqual({k: sorted(v) for k, v in seen.items()},
                         {0: [480, 510, 540, 570], 2: [480, 510, 540, 570], 3: [480, 540, 570]})


def flagged(metric: str, observed: float, high: float, triggered: bool = True) -> dict:
    """One metric of a triggered bin, the aximoe detector's hit only, its normal range 0 to `high`."""
    return {"metric": metric, "robust_z": None,
            "aximoe": {"triggered": triggered, "direction": "up", "observed": observed,
                       "observed_at": "2026-10-26T06:01:00Z", "normal_low": 0.0, "normal_high": high}}


class Ranking(unittest.TestCase):
    def test_distinct_metrics_first_then_largest_excess(self):
        def bin_(container, *metrics):
            return {"container": container, "bin_start": "2026-10-26T06:00:00Z",
                    "bin_end": "2026-10-26T06:05:00Z", "metrics": list(metrics)}
        trig = [
            # one metric, excess 1; its memory hit is below the threshold and counts for nothing
            bin_("a", flagged("cpu_usage_rate", 2.0, 1.0), flagged("mem_working_set", 900.0, 100.0, triggered=False)),
            # two metrics, excess 0.1
            bin_("b", flagged("cpu_usage_rate", 1.1, 1.0), flagged("mem_working_set", 110.0, 100.0)),
            # one metric, excess 8
            bin_("c", flagged("cpu_usage_rate", 9.0, 1.0)),
        ]
        ranked, more = investigate.ranking(trig, top=2)
        self.assertEqual([c["container"] for c in ranked], ["b", "c"])
        self.assertEqual(more, 1)
        ranked, _ = investigate.ranking(trig)
        self.assertEqual([c["container"] for c in ranked], ["b", "c", "a"])
        self.assertEqual(ranked[2]["metrics"], ["cpu_usage_rate"])


class Increases(unittest.TestCase):
    def test_rises_since_the_last_present_value(self):
        # first value, a gap, a rise across the gap, a fall to a reset counter, a rise after the reset
        values = np.array([np.nan, 3.0, 3.0, np.nan, 5.0, 5.0, 0.0, 2.0])
        np.testing.assert_allclose(investigate.increases(values), [0, 0, 0, 0, 2, 0, 0, 2])



class OomKills(unittest.TestCase):
    def test_a_restart_hides_only_its_own_pods_kernel_rise(self):
        start = dt.datetime(2026, 11, 5, 11, 55, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(minutes=10)
        t0 = int(start.timestamp()) - 60 * investigate.OOM_LOOKBACK
        length = investigate.OOM_LOOKBACK + 10 + investigate.OOM_REASON_DELAY

        def counter(pod: str, rise_at: int) -> dict:
            at = investigate.OOM_LOOKBACK + rise_at
            data = [0.0 if t < at else 1.0 for t in range(length)]
            return {"tags": {"namespace": "h", "pod": pod, "container": "c"}, "start": t0, "data": data}

        # h-1 is OOM-restarted at minute 2 and its kernel counter sees the same kill at 3; h-2 is
        # killed at 3 without a restart
        replies = {"kube_pod_container_status_restarts_total": [counter("h-1", 2)],
                   "kube_pod_container_status_last_terminated_reason": [counter("h-1", 2)],
                   "container_oom_events_total": [counter("h-1", 3), counter("h-2", 3)]}

        async def mpl(client, dep, q, query_start, query_end):
            return next(part for metric, part in replies.items() if f":{metric} " in q)

        with mock.patch.object(investigate, "mpl", mpl):
            kills = asyncio.run(investigate.oom_kills(None, None, "d", ".*", start, end))
        self.assertEqual([(k["container"], k["minute"][11:16], k["kills"]) for k in kills],
                         [("h/c", "11:57", 1), ("h/c", "11:58", 1)])


if __name__ == "__main__":
    unittest.main()
