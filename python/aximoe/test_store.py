"""Tests for the teacher store's writer and reader.

    python -m unittest discover -s python -p "test_store.py" -v
"""
import os
import tempfile
import unittest

import numpy as np

from aximoe import store as teacher_store


class StoreRoundTrip(unittest.TestCase):
    def test_write_read_and_completion_marker(self):
        n = 5
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            store = teacher_store.Store.create(d, n, quantile_dtype="float16")
            values = rng.normal(size=(n, teacher_store.TILE)).astype(np.float32)
            q = rng.normal(size=(n, 9, teacher_store.POSITIONS, 32)).astype(np.float32)
            loc = rng.normal(size=(n, teacher_store.POSITIONS)).astype(np.float32) * 1e9
            scale = rng.uniform(1, 2, size=(n, teacher_store.POSITIONS)).astype(np.float32)
            store.write(np.array([3, 1, 4]), values[:3], q[:3], loc[:3], scale[:3])
            store.write(np.array([0, 2]), values[3:], q[3:], loc[3:], scale[3:])
            self.assertFalse(teacher_store.Store.is_complete(d))
            store.finish({"model": "fake"})
            self.assertTrue(teacher_store.Store.is_complete(d))
            back = teacher_store.Store.open(d)
            self.assertEqual(back.meta["model"], "fake")
            self.assertEqual(back.meta["tiles"], n)
            np.testing.assert_array_equal(back.values[3], values[0])
            np.testing.assert_array_equal(back.values[2], values[4])
            np.testing.assert_array_equal(back.loc[3], loc[0])      # loc keeps float32: bytes survive
            self.assertEqual(back.quantiles.dtype, np.float16)
            np.testing.assert_allclose(back.quantiles[4].astype(np.float32), q[2], rtol=1e-3, atol=1e-3)
            picks = np.array([4, 0, 4, 2])
            for name in teacher_store.ARRAYS:
                np.testing.assert_array_equal(back.rows(picks)[name], np.asarray(getattr(back, name)[picks]))
                self.assertEqual(back.rows([])[name].shape, (0,) + getattr(back, name).shape[1:])
            self.assertEqual(sorted(os.listdir(d)), ["loc.f32", "manifest.json", "quantiles.f16", "scale.f32", "values.f32"])


if __name__ == "__main__":
    unittest.main()
