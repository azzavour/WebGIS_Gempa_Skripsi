import pandas as pd
from django.test import SimpleTestCase

from .aftershock import classify_aftershock_risk
from .monthly_aggregation import (
    MonthlyAggregationError,
    aggregate_weekly_predictions,
    classify_monthly_risk,
)
from .seismic_cause import SeismicCauseAnalyzer
from .views import _classify_risk, _get_mitigation_actions


class MonthlyAggregationTests(SimpleTestCase):
    def test_combines_weekly_probabilities_per_month(self):
        df = pd.DataFrame(
            {
                "grid_id": ["0_0"] * 4,
                "grid_lat": [0.0] * 4,
                "grid_lon": [100.0] * 4,
                "prediction_date": [
                    "2025-01-03",
                    "2025-01-10",
                    "2025-02-02",
                    "2025-02-16",
                ],
                "rf_prob": [0.1, 0.25, 0.5, 0.5],
            }
        )

        result = aggregate_weekly_predictions(df, "rf_prob")
        january = result[result["month_label"] == "2025-01"].iloc[0]
        february = result[result["month_label"] == "2025-02"].iloc[0]

        self.assertAlmostEqual(january["probability"], 1 - (0.9 * 0.75), places=6)
        self.assertEqual(january["weekly_count"], 2)
        self.assertEqual(january["risk_classification"], "Medium")

        self.assertAlmostEqual(february["probability"], 1 - (0.5 * 0.5), places=6)
        self.assertEqual(february["risk_classification"], "High")

    def test_requires_time_information(self):
        df = pd.DataFrame(
            {
                "grid_id": ["0_0"],
                "grid_lat": [0.0],
                "grid_lon": [100.0],
                "rf_prob": [0.2],
            }
        )
        with self.assertRaises(MonthlyAggregationError):
            aggregate_weekly_predictions(df, "rf_prob")

    def test_classification_thresholds_follow_spec(self):
        self.assertEqual(classify_monthly_risk(0.1), "Low")
        self.assertEqual(classify_monthly_risk(0.3), "Medium")
        self.assertEqual(classify_monthly_risk(0.6), "Medium")
        self.assertEqual(classify_monthly_risk(0.61), "High")


class SeismicCauseTests(SimpleTestCase):
    def setUp(self):
        self.analyzer = SeismicCauseAnalyzer()

    def test_fault_rule_has_priority_when_within_threshold(self):
        result = self.analyzer.describe_properties(-6.85, 107.65)
        self.assertEqual(result["seismic_cause"], "Active Fault Movement")

    def test_volcano_rule_triggers_when_faults_are_farther(self):
        result = self.analyzer.describe_properties(-7.25, 108.05)
        self.assertEqual(result["seismic_cause"], "Volcanic Activity")

    def test_default_when_no_structure_within_threshold(self):
        result = self.analyzer.describe_properties(-4.0, 130.0)
        self.assertEqual(result["seismic_cause"], "Regional Tectonic Activity")


class AftershockEstimatorTests(SimpleTestCase):
    def test_rule_high(self):
        self.assertEqual(classify_aftershock_risk(6.1), "High")

    def test_rule_medium(self):
        self.assertEqual(classify_aftershock_risk(5.5), "Medium")

    def test_rule_low(self):
        self.assertEqual(classify_aftershock_risk(4.9), "Low")


class ExposureClassificationTests(SimpleTestCase):
    def test_extreme_threshold(self):
        self.assertEqual(_classify_risk(1_000_001), "Extreme")

    def test_high_threshold(self):
        self.assertEqual(_classify_risk(350_000), "High")

    def test_medium_threshold(self):
        self.assertEqual(_classify_risk(120_000), "Medium")

    def test_low_threshold_and_guidance(self):
        self.assertEqual(_classify_risk(80_000), "Low")
        guidance = _get_mitigation_actions("Low")
        self.assertIn("Monitoring", guidance[0])
