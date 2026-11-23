"""Tests of the student's cached pass: `cached_pass` gives `forward_full`'s quantiles, and
`next_position` over two consecutive changed patches gives what a full pass on the changed
inputs gives at those positions. And of the patch grid a process takes from `AXIMOE_PATCH`.

    python -m unittest discover -s python -p "test_student.py" -v
    python -I python/aximoe/test_student.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

from aximoe.student import (MEDIAN, PATCH, POSITIONS, SEQ, TAG_BASE, Layer, Student, StudentConfig, anchor_values, load,
                            mix_back, patch_level, save)


class NextPosition(unittest.TestCase):
    def test_matches_a_full_pass_on_the_changed_inputs(self):
        torch.manual_seed(0)
        cfg = StudentConfig(d_model=32, heads=4, layers=3, d_ff=48, tags=5, backbone_tags=True,
                            resolutions=4, resolution_tags=True, band_input=True)
        model = Student(cfg).eval()
        n = 16
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        tags = torch.randint(0, 4, (n,)) * TAG_BASE + torch.randint(0, 5, (n,))
        rows = torch.arange(n)
        p = torch.randint(0, POSITIONS - 1, (n,))
        with torch.no_grad():
            q, states = model.cached_pass(inputs, mask, tags, band)
            self.assertTrue(torch.equal(q, model.forward_full(inputs, mask, tags, None, band)[0]))
            for step in (0, 1):
                at = p + step
                inputs[rows, at] = torch.randn(n, PATCH)
                mask[rows, at] = 1.0
                band[rows, at] = torch.randn(n, 2 * PATCH) * 0.1
                got = model.next_position(inputs, mask, band, tags, at, states)
                want = model.forward_full(inputs, mask, tags, None, band)[0][rows, at]
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_leading_positions_match_a_full_pass(self):
        torch.manual_seed(0)
        cfg = StudentConfig(d_model=32, heads=4, layers=3, d_ff=48, tags=5, backbone_tags=True,
                            resolutions=4, resolution_tags=True, band_input=True)
        model = Student(cfg).eval()
        n, s = 16, 15
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        tags = torch.randint(0, 4, (n,)) * TAG_BASE + torch.randint(0, 5, (n,))
        with torch.no_grad():
            q, states = model.cached_pass(inputs[:, :s], mask[:, :s], tags, band[:, :s], room=2)
            want = model.forward_full(inputs, mask, tags, None, band)[0][:, :s]
            torch.testing.assert_close(q, want, rtol=1e-5, atol=1e-5)
            for at in (s, s + 1):
                inputs[:, at] = torch.randn(n, PATCH)
                mask[:, at] = 1.0
                band[:, at] = torch.randn(n, 2 * PATCH) * 0.1
                got = model.next_position(inputs, mask, band, tags, torch.full((n,), at), states)
                want = model.forward_full(inputs, mask, tags, None, band)[0][:, at]
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


class Anchors(unittest.TestCase):
    def test_values_are_the_last_present_value_and_the_median_carried_forward(self):
        inputs, mask = torch.zeros(2, 4, PATCH), torch.zeros(2, 4, PATCH)
        # tile 0: a full patch, a patch whose tail is missing, an empty patch, a full patch
        inputs[0, 0], mask[0, 0] = torch.arange(PATCH, dtype=torch.float32), 1.0
        inputs[0, 1, :5], mask[0, 1, :5] = torch.tensor([3.0, 1.0, 2.0, 9.0, 4.0]), 1.0
        inputs[0, 3], mask[0, 3] = -2.0, 1.0
        # tile 1: its first value is in patch 2
        inputs[1, 2, 7], mask[1, 2, 7] = 5.0, 1.0
        last, median = anchor_values(inputs, mask)
        torch.testing.assert_close(last, torch.tensor([[PATCH - 1.0, 4.0, 4.0, -2.0], [0.0, 0.0, 5.0, 5.0]]), rtol=0, atol=1e-6)
        torch.testing.assert_close(median, torch.tensor([[PATCH / 2 - 1.0, 3.0, 3.0, -2.0], [0.0, 0.0, 5.0, 5.0]]), rtol=0, atol=1e-6)

    def test_cached_passes_match_a_full_pass(self):
        torch.manual_seed(0)
        cfg = StudentConfig(d_model=32, heads=4, layers=3, d_ff=48, band_input=True, anchors=True)
        model = Student(cfg).eval()
        torch.nn.init.normal_(model.anchor_head.weight, std=0.1)
        n = 8
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        mask[:, 40:44] = 0.0
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        rows = torch.arange(n)
        at = torch.randint(0, POSITIONS - 1, (n,))
        with torch.no_grad():
            plain = Student(StudentConfig(d_model=32, heads=4, layers=3, d_ff=48, band_input=True)).eval()
            plain.load_state_dict(model.state_dict(), strict=False)
            full = model.forward_full(inputs * mask, mask, None, None, band)[0]
            self.assertGreater(float((full - plain.forward_full(inputs * mask, mask, None, None, band)[0]).abs().max()), 1e-3)
            q, states = model.cached_pass(inputs * mask, mask, None, band)
            torch.testing.assert_close(q, full, rtol=1e-5, atol=1e-5)
            inputs[rows, at], mask[rows, at] = torch.randn(n, PATCH), 1.0
            got = model.next_position(inputs * mask, mask, band, None, at, states)
            torch.testing.assert_close(got, model.forward_full(inputs * mask, mask, None, None, band)[0][rows, at], rtol=1e-5, atol=1e-5)


class Shapes(unittest.TestCase):
    def test_looped_and_linear_students_keep_the_cached_passes(self):
        torch.manual_seed(0)
        base = dict(d_model=32, heads=4, layers=2, d_ff=48, band_input=True)
        plain = Student(StudentConfig(**base))
        for extra in (dict(loops=3), dict(io_hidden=0), dict(io_hidden=24, loops=2), dict(rope=True), dict(qk_norm=True),
                      dict(rope=True, qk_norm=True, loops=2), dict(canon=4), dict(recurrent=16),
                      dict(canon=3, recurrent=8, rope=True, loops=2), dict(level_head=True), dict(canon=4, canon_held=True),
                      dict(diff_attention=True), dict(diff_attention=True, canon=4)):
            model = Student(StudentConfig(**base, **extra)).eval()
            if model.level_head is not None:
                torch.nn.init.normal_(model.level_head.weight, std=0.1)
            mixes = [w for layer in model.layers for w in (layer.canon_a, layer.canon_c) if w is not None]
            for w in mixes + ([model.recurrent_out.weight] if model.recurrent is not None else []):
                torch.nn.init.normal_(w, std=0.3)
            if set(extra) <= {"loops"}:
                self.assertEqual(model.num_params(), plain.num_params())
            n = 6
            inputs = torch.randn(n, SEQ, PATCH)
            mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
            band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
            rows = torch.arange(n)
            at = torch.randint(0, POSITIONS - 1, (n,))
            with torch.no_grad():
                full = model.forward_full(inputs, mask, None, None, band)[0]
                q, states = model.cached_pass(inputs, mask, None, band)
                self.assertEqual(len(states), 2 * model.cfg.loops + (model.cfg.recurrent > 0))
                torch.testing.assert_close(q, full, rtol=1e-5, atol=1e-5)
                inputs[rows, at], mask[rows, at] = torch.randn(n, PATCH), 1.0
                got = model.next_position(inputs, mask, band, None, at, states)
                torch.testing.assert_close(got, model.forward_full(inputs, mask, None, None, band)[0][rows, at], rtol=1e-5, atol=1e-5)
        linear = Student(StudentConfig(**base, io_hidden=0))
        self.assertEqual(plain.num_params() - linear.num_params(),
                         (4 * PATCH + 32 + 1) * 128 + 128 * (15 * PATCH + 32 + 1) + 15 * PATCH + 32 - (4 * PATCH * 32 + 32 + 32 * 15 * PATCH + 15 * PATCH))

    def test_mixed_and_recurrent_students_extend_leading_positions(self):
        torch.manual_seed(0)
        base = dict(d_model=32, heads=4, layers=2, d_ff=48, band_input=True)
        model = Student(StudentConfig(**base, canon=4, recurrent=16)).eval()
        n, s = 6, 15
        inputs = torch.randn(n, SEQ, PATCH)
        mask = (torch.rand(n, SEQ, PATCH) > 0.2).float()
        band = torch.randn(n, SEQ, 2 * PATCH) * 0.1
        with torch.no_grad():
            plain = Student(StudentConfig(**base)).eval()
            plain.load_state_dict(model.state_dict(), strict=False)
            start = model.forward_full(inputs, mask, None, None, band)[0]
            torch.testing.assert_close(start, plain.forward_full(inputs, mask, None, None, band)[0], rtol=1e-6, atol=1e-6)
            for layer in model.layers:
                torch.nn.init.normal_(layer.canon_a, std=0.3)
                torch.nn.init.normal_(layer.canon_c, std=0.3)
            torch.nn.init.normal_(model.recurrent_out.weight, std=0.3)
            self.assertGreater(float((model.forward_full(inputs, mask, None, None, band)[0] - start).abs().max()), 1e-3)
            q, states = model.cached_pass(inputs[:, :s], mask[:, :s], None, band[:, :s], room=2)
            torch.testing.assert_close(q, model.forward_full(inputs, mask, None, None, band)[0][:, :s], rtol=1e-5, atol=1e-5)
            for at in (s, s + 1):
                inputs[:, at] = torch.randn(n, PATCH)
                mask[:, at] = 1.0
                got = model.next_position(inputs, mask, band, None, torch.full((n,), at), states)
                torch.testing.assert_close(got, model.forward_full(inputs, mask, None, None, band)[0][:, at], rtol=1e-5, atol=1e-5)
            _, states = model.cached_pass(inputs, mask, None, band)
            inputs[:, 0] = torch.randn(n, PATCH)
            got = model.next_position(inputs, mask, band, None, torch.zeros(n, dtype=torch.int64), states)
            torch.testing.assert_close(got, model.forward_full(inputs, mask, None, None, band)[0][:, 0], rtol=1e-5, atol=1e-5)

    def test_level_head_moves_each_patch_to_its_level_as_far_as_the_gate_is_open(self):
        torch.manual_seed(0)
        base = dict(d_model=32, heads=4, layers=2, d_ff=48, band_input=True)
        model = Student(StudentConfig(**base, level_head=True)).eval()
        plain = Student(StudentConfig(**base)).eval()
        plain.load_state_dict(model.state_dict(), strict=False)
        inputs = torch.randn(4, SEQ, PATCH)
        mask = (torch.rand(4, SEQ, PATCH) > 0.2).float()
        band = torch.randn(4, SEQ, 2 * PATCH) * 0.1
        with torch.no_grad():
            before = plain.forward_full(inputs, mask, None, None, band)[0]
            own = before[:, :, MEDIAN].median(dim=2).values
            # the gate starts at sigmoid(-4) and the level at zero
            start = model.forward_full(inputs, mask, None, None, band)[0]
            torch.testing.assert_close(start, before - (torch.sigmoid(torch.tensor(-4.0)) * own)[:, :, None, None], rtol=1e-4, atol=1e-5)
            torch.nn.init.normal_(model.level_head.weight, std=0.1)
            model.level_head.bias.copy_(torch.tensor([0.3, 20.0]))
            h = model.hidden(inputs, mask, None, None, band)[0]
            moved = model.head(h)
            torch.testing.assert_close(moved[:, :, MEDIAN].median(dim=2).values, model.level(h), rtol=1e-4, atol=1e-4)
            # every knot of a patch moves by the same amount
            shift = moved - before
            self.assertLess(float((shift - shift[:, :, :1, :1]).abs().max()), 1e-4)

    def test_a_held_mix_reads_no_position_without_data(self):
        torch.manual_seed(0)
        x, weight = torch.randn(2, 12, 8), torch.randn(8, 1, 4)
        held = torch.ones(2, 12, 1)
        held[:, 5:8] = 0.0
        base = mix_back(x, weight, held)
        changed = x.clone()
        changed[:, 5:8] += 3.0
        moved = mix_back(changed, weight, held)
        # a position without data keeps its own state and hands nothing on
        torch.testing.assert_close(moved - changed, base - x, rtol=1e-5, atol=1e-5)
        self.assertGreater(float((mix_back(changed, weight) - changed - (mix_back(x, weight) - x)).abs().max()), 1e-3)
        model = Student(StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, band_input=True, canon=4, canon_held=True)).eval()
        for layer in model.layers:
            torch.nn.init.normal_(layer.canon_a, std=0.3)
            torch.nn.init.normal_(layer.canon_c, std=0.3)
        inputs, mask = torch.randn(4, SEQ, PATCH), torch.ones(4, SEQ, PATCH)
        mask[:, 40:] = 0.0
        band = torch.zeros(4, SEQ, 2 * PATCH)
        with torch.no_grad():
            q, states = model.cached_pass(inputs * mask, mask, None, band)
            torch.testing.assert_close(q, model.forward_full(inputs * mask, mask, None, None, band)[0], rtol=1e-5, atol=1e-5)
            # the patch at 40 arrives: the single-position pass equals a full pass there
            mask[:, 40] = 1.0
            got = model.next_position(inputs * mask, mask, band, None, torch.full((4,), 40), states)
            torch.testing.assert_close(got, model.forward_full(inputs * mask, mask, None, None, band)[0][:, 40], rtol=1e-5, atol=1e-5)

    def test_patch_level_is_the_median_and_its_gradient_the_means(self):
        path = torch.randn(3, 5, PATCH, requires_grad=True)
        level = patch_level(path)
        torch.testing.assert_close(level, path.median(dim=2).values, rtol=0, atol=1e-6)
        level.sum().backward()
        torch.testing.assert_close(path.grad, torch.full_like(path, 1.0 / PATCH), rtol=0, atol=1e-7)

    def test_rotary_attention_reads_offsets_between_positions_only(self):
        torch.manual_seed(0)
        layer = Layer(StudentConfig(d_model=32, heads=4, layers=1, d_ff=48, rope=True)).eval()
        x = torch.randn(3, 10, 32)
        open_ = torch.zeros(10, 10, dtype=torch.bool)
        at = torch.arange(10)
        with torch.no_grad():
            here = layer.attend(x, x, open_, at, at)
            torch.testing.assert_close(layer.attend(x, x, open_, at + 17, at + 17), here, rtol=1e-4, atol=1e-5)
            self.assertGreater(float((layer.attend(x, x, open_, at * 2, at * 2) - here).abs().max()), 1e-3)


class Weights(unittest.TestCase):
    def test_safetensors_weights_load_the_same_student_as_the_pickle(self):
        torch.manual_seed(0)
        cfg = StudentConfig(d_model=32, heads=4, layers=2, d_ff=48, tags=5, backbone_tags=True,
                            resolutions=4, resolution_tags=True, band_input=True)
        model = Student(cfg).eval()
        with tempfile.TemporaryDirectory() as pt_dir, tempfile.TemporaryDirectory() as st_dir:
            save(model, {"config": cfg.to_json(), "weights": "student.pt"}, pt_dir)
            save_file(model.state_dict(), os.path.join(st_dir, "model.safetensors"))
            with open(os.path.join(st_dir, "model.json"), "w") as f:
                json.dump({"config": cfg.to_json(), "weights": "model.safetensors"}, f)
            _, from_pt = load(pt_dir)
            _, from_st = load(st_dir)
        want, got = from_pt.state_dict(), from_st.state_dict()
        self.assertEqual(want.keys(), got.keys())
        for name in want:
            self.assertTrue(torch.equal(want[name], got[name]), name)
        inputs = torch.randn(3, SEQ, PATCH)
        mask = torch.ones(3, SEQ, PATCH)
        band = torch.zeros(3, SEQ, 2 * PATCH)
        tags = torch.zeros(3, dtype=torch.long)
        with torch.no_grad():
            self.assertTrue(torch.equal(from_pt.forward_full(inputs, mask, tags, None, band)[0],
                                        from_st.forward_full(inputs, mask, tags, None, band)[0]))


class PatchWidth(unittest.TestCase):
    def test_a_process_takes_its_grid_from_the_environment(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from aximoe import student as s; "
                "print(s.PATCH, s.SEQ, s.POSITIONS, s.CONTEXT)")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-I", "-c", code, root], env={**os.environ, "AXIMOE_PATCH": "16"},
                             check=True, capture_output=True, text=True).stdout
        self.assertEqual(out.split(), ["16", "128", "127", "2032"])


if __name__ == "__main__":
    unittest.main()
