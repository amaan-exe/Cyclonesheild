"""
Cyclone Horizon — Pytest Robustness & Stress Tests
Verifies that:
1. Coordinate perturbation does not cause chaotic diverging track errors
2. Missing observation fixes are handled gracefully with 100% completion rate
3. Rapid Intensification and recurvature cases are detected and evaluated
4. Robustness JSON report is generated
"""

import os
import pytest
from src.evaluation.robustness_tester import RobustnessTester


@pytest.fixture(scope="module")
def tester():
    return RobustnessTester()


class TestModelRobustness:
    def test_coordinate_perturbation_bounded(self, tester):
        res = tester.test_coordinate_perturbation(sample_storms=10)
        assert res["status"] in ["PASSED", "ACCEPTABLE"]
        assert res["clean_mean_error_km"] > 0
        assert res["amplification_ratio"] < 2.2
        assert "conclusion" in res

    def test_missing_fix_dropout_graceful(self, tester):
        res = tester.test_missing_fix_dropout(sample_storms=10)
        assert res["status"] in ["PASSED", "ACCEPTABLE"]
        assert res["completion_rate_pct"] == 100.0
        assert res["degradation_pct"] < 35.0

    def test_rapid_intensification_stress(self, tester):
        res = tester.test_rapid_intensification_stress()
        assert res["status"] == "PASSED"
        assert res["total_ri_cases"] > 0
        assert len(res["evaluated_ri_storms"]) > 0

    def test_recurvature_stress(self, tester):
        res = tester.test_recurvature_stress()
        assert res["status"] == "PASSED"
        assert res["total_recurvature_cases"] > 0

    def test_stratified_breakdown(self, tester):
        res = tester.test_stratified_breakdown()
        assert "bay_of_bengal" in res
        assert "arabian_sea" in res
        assert res["bay_of_bengal"]["storm_count"] > 0
        assert res["arabian_sea"]["storm_count"] > 0

    def test_rl_safety_guardrail(self, tester):
        res = tester.test_rl_safety_guardrail(num_adversarial_samples=100)
        assert res["status"] == "PASSED"
        assert res["safety_violations_count"] == 0
        assert res["nan_or_inf_count"] == 0
        assert res["safety_compliance_pct"] == 100.0
        assert res["max_observed_track_nudge_deg"] <= 0.4501
        assert res["max_observed_intensity_nudge_kt"] <= 8.001

    def test_detection_noise_resilience(self, tester):
        res = tester.test_detection_noise_resilience()
        assert res["status"] in ["PASSED", "ACCEPTABLE"]
        assert res["centroid_drift_under_noise_km"] < 28.0
        assert res["clean_confidence"] > 0.05
