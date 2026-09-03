import unittest

from experiment_tracker import scoring


class TestScoring(unittest.TestCase):
    def test_rmse(self) -> None:
        self.assertAlmostEqual(0.0, scoring.rmse([1.0, 2.0], [1.0, 2.0]))
        self.assertAlmostEqual(1.0, scoring.rmse([1.0, 3.0], [2.0, 2.0]))

    def test_mae(self) -> None:
        self.assertAlmostEqual(0.25, scoring.mae([1.0, 2.0], [1.5, 2.0]))

    def test_mape_skips_zero_actuals(self) -> None:
        self.assertAlmostEqual(0.5, scoring.mape([1.5, 9.0], [1.0, 0.0]))

    def test_mape_needs_a_non_zero_actual(self) -> None:
        with self.assertRaises(ValueError):
            scoring.mape([1.0], [0.0])

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            scoring.mae([1.0, 2.0], [1.0])

    def test_an_empty_sequence_raises(self) -> None:
        with self.assertRaises(ValueError):
            scoring.mae([], [])

    def test_score_returns_a_mapping_shaped_for_log_metrics(self) -> None:
        result = scoring.score([1.0, 3.0], [2.0, 2.0])
        self.assertEqual({"rmse", "mae"}, set(result))
        self.assertAlmostEqual(1.0, result["rmse"])

    def test_score_names_its_metrics(self) -> None:
        self.assertEqual({"mae"}, set(scoring.score([1.0], [2.0], metrics=["mae"])))

    def test_an_unknown_metric_raises(self) -> None:
        with self.assertRaises(ValueError):
            scoring.score([1.0], [2.0], metrics=["r2"])


if __name__ == "__main__":
    unittest.main()
