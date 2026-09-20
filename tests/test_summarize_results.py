import unittest

from graphkv.metrics import percentile


class PercentileTests(unittest.TestCase):
    def test_p95_uses_nearest_rank(self) -> None:
        values = [float(value) for value in range(33)]
        self.assertEqual(percentile(values, 0.95), 31.0)

    def test_percentile_accepts_unsorted_values(self) -> None:
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
