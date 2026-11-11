import unittest

import torch

from subtext.residuals import projection_spec, quantized_delta_fingerprint


class ResidualProjectionTests(unittest.TestCase):
    def test_projection_is_deterministic_and_uses_every_bin(self):
        first = projection_spec(256, 32)
        second = projection_spec(256, 32)
        self.assertEqual(first, second)
        self.assertEqual(set(first[0]), set(range(32)))
        self.assertEqual(set(first[1]), {-1, 1})

    def test_normalizers_match_bucket_population(self):
        buckets, _signs, scales = projection_spec(128, 16)
        counts = {bucket: buckets.count(bucket) for bucket in set(buckets)}
        for bucket, scale in zip(buckets, scales):
            self.assertAlmostEqual(scale, counts[bucket] ** -0.5)

    def test_rejects_more_bins_than_coordinates(self):
        with self.assertRaises(ValueError):
            projection_spec(8, 16)

    def test_equal_width_and_bin_count_remains_defined(self):
        buckets, _signs, scales = projection_spec(16, 16)
        self.assertEqual(buckets, tuple(range(16)))
        self.assertTrue(all(scale == 1.0 for scale in scales))

    def test_delta_is_formed_before_normalization(self):
        delta = torch.tensor([1.0, -1.0, 2.0, -2.0])
        projection = torch.eye(4)
        from_origin = quantized_delta_fingerprint(
            torch.zeros(4), delta, projection
        )
        shifted = quantized_delta_fingerprint(
            torch.full((4,), 1000.0),
            torch.full((4,), 1000.0) + delta,
            projection,
        )
        self.assertEqual(from_origin, shifted)
        self.assertAlmostEqual(from_origin[0], (2.5) ** 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
