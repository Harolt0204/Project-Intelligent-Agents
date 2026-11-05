import unittest
import numpy as np
from experiments.intervalos import mean_ci95


class ConfidenceIntervalTests(unittest.TestCase):
    def test_student_t_small_sample_not_normal_or_episode_count(self):
        mean, low, high = mean_ci95([1, 2, 3])
        self.assertAlmostEqual(float(mean), 2.)
        self.assertAlmostEqual(float(high), 4.484137711719546)
        self.assertAlmostEqual(float(low), -0.484137711719546)

    def test_constant_and_missing_replicate(self):
        mean, low, high = mean_ci95([[500, 1], [500, 2], [500, np.nan]])
        np.testing.assert_equal([mean[0], low[0], high[0]], [500]*3)
        self.assertTrue(all(np.isnan(v[1]) for v in (mean, low, high)))
        with self.assertRaises(ValueError):
            mean_ci95([1, 2])


if __name__ == '__main__':
    unittest.main()
