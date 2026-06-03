"""
Test for PC. all data is synthetic.
"""


import pytest
import numpy as np
from datetime import datetime, timezone
 
from src.conjunction.conjunction_models import ConjunctionEvent
from src.probability.pc_calculator import PcCalculator, PcResult
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def make_event(
    miss_km: float = 1.0,
    v_rel: float = 7.5,
    radial: float = 0.5,
    transverse: float = 0.1,
    normal: float = 0.866,
) -> ConjunctionEvent:
    """Build a synthetic ConjunctionEvent for testing."""
    return ConjunctionEvent(
        primary_id=25544,
        secondary_id=99001,
        primary_name="ISS",
        secondary_name="DEBRIS",
        tca=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        miss_distance=miss_km,
        relative_velocity=v_rel,
        radial_miss=radial,
        transverse_miss=transverse,
        normal_miss=normal,
        screening_threshold_km=5.0,
    )
 
 
def make_covariance(sigma_km: float = 1.0) -> np.ndarray:
    """Isotropic 3×3 covariance with given 1-sigma (km)."""
    return np.eye(3) * sigma_km**2
 
 
# ── PcResult Tests ─────────────────────────────────────────────────────────────
 
class TestPcResult:
 
    def test_red_threshold(self):
        result = PcResult(pc=1e-3, miss_distance=0.5, r_hbr=0.01,
                          sigma_x=1.0, sigma_y=0.3, risk_level="red")
        assert result.risk_level == "red"
        assert result.is_actionable is True
 
    def test_yellow_threshold(self):
        result = PcResult(pc=5e-5, miss_distance=1.0, r_hbr=0.01,
                          sigma_x=1.0, sigma_y=0.3, risk_level="yellow")
        assert result.risk_level == "yellow"
        assert result.is_actionable is True
 
    def test_green_threshold(self):
        result = PcResult(pc=1e-7, miss_distance=4.0, r_hbr=0.01,
                          sigma_x=1.0, sigma_y=0.3, risk_level="green")
        assert result.risk_level == "green"
        assert result.is_actionable is False
 
 
# ── PcCalculator Basic Tests ───────────────────────────────────────────────────
 
class TestPcCalculator:
 
    def test_returns_pc_result(self):
        calc = PcCalculator(r_hbr_km=0.01)
        event = make_event(miss_km=1.0)
        result = calc.compute(event)
        assert isinstance(result, PcResult)
 
    def test_pc_between_zero_and_one(self):
        calc = PcCalculator(r_hbr_km=0.01)
        for miss in [0.1, 0.5, 1.0, 3.0, 5.0]:
            result = calc.compute(make_event(miss_km=miss))
            assert 0.0 <= result.pc <= 1.0, f"Pc={result.pc} out of range for miss={miss}"
 
    def test_pc_written_to_event(self):
        """compute() should write Pc back into event.pc."""
        calc = PcCalculator(r_hbr_km=0.01)
        event = make_event(miss_km=1.0)
        assert event.pc is None
        calc.compute(event)
        assert event.pc is not None
 
    def test_pc_decreases_with_miss_distance(self):
        """Larger miss distance → lower Pc."""
        calc = PcCalculator(r_hbr_km=0.01)
        cov = make_covariance(1.0)
        results = [calc.compute(make_event(miss_km=d), cov, cov) for d in [0.2, 0.5, 1.0, 2.0]]
        pcs = [r.pc for r in results]
        assert pcs == sorted(pcs, reverse=True), f"Pc should decrease with miss distance: {pcs}"
 
    def test_pc_increases_with_larger_hbr(self):
        """Larger hard-body radius → higher Pc."""
        cov = make_covariance(0.1)
        small_hbr = PcCalculator(r_hbr_km=0.001).compute(make_event(miss_km=3.0), cov, cov)
        large_hbr = PcCalculator(r_hbr_km=0.100).compute(make_event(miss_km=3.0), cov, cov)
        assert large_hbr.pc > small_hbr.pc

    def test_sigma_values_positive(self):
        calc = PcCalculator(r_hbr_km=0.01)
        result = calc.compute(make_event())
        assert result.sigma_x > 0
        assert result.sigma_y > 0
 
    def test_default_covariance_flagged(self):
        """When no covariance provided, used_default_covariance should be True."""
        calc = PcCalculator()
        result = calc.compute(make_event())
        assert result.used_default_covariance is True
 
    def test_real_covariance_not_flagged(self):
        """When real covariance provided, used_default_covariance should be False."""
        calc = PcCalculator()
        cov = make_covariance(1.0)
        result = calc.compute(make_event(), cov, cov)
        assert result.used_default_covariance is False
 
 
# ── Risk Classification Tests ─────────────────────────────────────────────────
 
class TestRiskClassification:
 
    def test_classify_red(self):
        assert PcCalculator._classify_risk(1e-3) == "red"
        assert PcCalculator._classify_risk(1e-4) == "red"
 
    def test_classify_yellow(self):
        assert PcCalculator._classify_risk(5e-5) == "yellow"
        assert PcCalculator._classify_risk(1e-5) == "yellow"
 
    def test_classify_green(self):
        assert PcCalculator._classify_risk(9.9e-6) == "green"
        assert PcCalculator._classify_risk(1e-10) == "green"
        assert PcCalculator._classify_risk(0.0) == "green"
 
 
# ── Batch Tests ───────────────────────────────────────────────────────────────
 
class TestBatchCompute:
 
    def test_batch_returns_same_count(self):
        calc = PcCalculator()
        events = [make_event(miss_km=d) for d in [1.0, 2.0, 3.0, 4.0]]
        results = calc.compute_batch(events)
        assert len(results) == len(events)
 
    def test_batch_writes_pc_to_all_events(self):
        calc = PcCalculator()
        events = [make_event(miss_km=d) for d in [1.0, 2.0, 3.0]]
        calc.compute_batch(events)
        assert all(e.pc is not None for e in events)
 
    def test_batch_sorted_by_input_order(self):
        """Results should be in same order as input events."""
        calc = PcCalculator()
        miss_distances = [4.0, 1.0, 2.5]
        events = [make_event(miss_km=d) for d in miss_distances]
        results = calc.compute_batch(events)
        for i, (event, result) in enumerate(zip(events, results)):
            assert abs(result.miss_distance - event.miss_distance) < 1e-9
 
 
# ── Foster Approximation Tests ────────────────────────────────────────────────
 
class TestFosterApproximation:
 
    def test_foster_returns_positive(self):
        calc = PcCalculator()
        pc = calc._foster_approximation(
            x_miss=1.0, y_miss=0.5,
            sigma_x=2.0, sigma_y=1.0,
            r_hbr=0.01,
        )
        assert pc > 0
 
    def test_foster_zero_miss_gives_max_pc(self):
        """At zero miss distance, Pc should be at its maximum."""
        calc = PcCalculator()
        pc_zero = calc._foster_approximation(0.0, 0.0, 1.0, 1.0, 0.01)
        pc_far  = calc._foster_approximation(5.0, 0.0, 1.0, 1.0, 0.01)
        assert pc_zero > pc_far
 