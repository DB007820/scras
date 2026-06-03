## This file Tests comjunction detection

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from src.models import Trajectory, StateVector
from src.conjunction.conjunction_models import ConjunctionEvent
from src.conjunction.detector import ConjunctionDetector, _eci_to_rrn

def make_circular_trajectory(
        norad_id: int,
        name: str,
        altitude_km: float,
        inclination_deg: float,
        t_start: datetime,
        t_end: datetime,
        dt_seconds: float = 60.0,
        raan_deg: float = 0.0

) -> Trajectory:
    ## Builds a synthetic circular orbit trajectory for testing purposes
    ## Uses Keplerian motion for tests
    """Glossary
    MU - standard gravitational parameter
    v - circular orbital speed
    T - Orbital periods in seconds
    """
    MU = 398600 
    R_Earth = 6371
    r = R_Earth + altitude_km
    v = np.sqrt(MU/r)
    T = 2* np.pi * np.sqrt(r**3 / MU)

    inc = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)

    states = []
    t = t_start
    step = timedelta(seconds=dt_seconds)

    while t <= t_end:
        elapsed = (t - t_start).total_seconds()
        n = 2 * np.pi / T
        theta = n * elapsed

        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # rotate to ECI using raan & inclination

        x = (np.cos(raan) * x_orb - np.sin(raan)* np.cos(inc) * y_orb)
        y = (np.sin(raan) * x_orb + np.cos(raan) * np.cos(inc) * y_orb)
        z = np.sin(inc) * y_orb

        vx_orb = -v * np.sin(theta)
        vy_orb = v * np.cos(theta)

        vx = (np.cos(raan) * vx_orb - np.sin(raan) * np.cos(inc) * vy_orb)
        vy = (np.sin(raan) * vx_orb + np.cos(raan) * np.cos(inc) * vy_orb)
        vz = np.sin(inc) * vy_orb

        states.append(StateVector(
            norad_id=norad_id,
            epoch=t,
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
        ))
        t += step
    return Trajectort(
        norad_id=norad_id,
        name=name,
        states=states,
        propagator="syntetic Keppler Motion"
    )

## Create parallel trajectory

def make_parallel_trajectory(
        norad_id: int,
        name: str,
        reference: Trajectory,
        offset_km: float,

) -> Trajectory:
    states = []
    states = []
    for sv in reference.states:
        h = np.cross(sv.position, sv.velocity)
        n_hat = h / np.linalg.norm(h)

    new_pos = sv.position + offset_km * n_hat
    states.append(StateVector(
        norad_id=norad_id,
        epoch=sv.epoch,
        position=new_pos,
        velocity=sv.velocity.copy(),
    ))

    return Trajectory(
        norad_id=norad_id,
        name=name,
        states=states,
        propagator="synthetic_offset",
    )

#Fixture Section
@pytest.fixture
def t_start():
    return datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def t_end():
    return datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc) # (2 for 2 hours)

@pytest.fixture
def detector():
    return ConjunctionDetector(threshold_km=5.0)

@pytest.fixture
def iss_like(t_start, t_end):
    return make_circular_trajectory(
        norad_id=25544,
        name="ISS_LIKE",
        altitude_km=410.0,
        inclination_deg=51.6,
        t_start=t_start,
        t_end=t_end,

    )

pytest.fixture
def near_miss_sat(iss_like):
    return make_parallel_trajectory(
        norad_id=99001,
        name="NEAR_MISS",
        reference=iss_like,
        offset_km=3.0,
    )

@pytest.fixture
def far_sat(t_start,t_end):
    return make_circular_trajectory(
        norad_id=99002,
        name="FAR_SAT",
        altitude_km=800.0,
        inclination_deg=98.0,
        t_start=t_start,
        t_end=t_end,
    )

#Perigee/Apogee Filter test

class TestPerigeeApogeeFilter:
 
    def test_same_altitude_passes(self, detector, iss_like, near_miss_sat):
        """Satellites at nearly the same altitude should pass the filter."""
        result = detector._perigee_apogee_filter(iss_like, near_miss_sat)
        assert result is True
 
    def test_well_separated_altitudes_filtered(self, detector, iss_like, far_sat):
        """410 km vs 800 km — altitude bands don't overlap, should be filtered."""
        result = detector._perigee_apogee_filter(iss_like, far_sat)
        assert result is False
 
    def test_filter_is_symmetric(self, detector, iss_like, far_sat):
        """Filter result should be same regardless of which satellite is A or B."""
        assert (
            detector._perigee_apogee_filter(iss_like, far_sat) ==
            detector._perigee_apogee_filter(far_sat, iss_like)
        )
 
    def test_threshold_buffer_included(self, t_start, t_end):
        """Satellites just outside threshold should still pass with buffer."""
        detector = ConjunctionDetector(threshold_km=10.0)
        sat_410 = make_circular_trajectory(99, "A", 410.0, 51.6, t_start, t_end)
        sat_415 = make_circular_trajectory(98, "B", 415.0, 51.6, t_start, t_end)
        # 5 km apart, 10 km threshold — should pass
        assert detector._perigee_apogee_filter(sat_410, sat_415) is True
class TestPerigeeApogeeFilter:
 
    def test_same_altitude_passes(self, detector, iss_like, near_miss_sat):
        """Satellites at nearly the same altitude should pass the filter."""
        result = detector._perigee_apogee_filter(iss_like, near_miss_sat)
        assert result is True
 
    def test_well_separated_altitudes_filtered(self, detector, iss_like, far_sat):
        """410 km vs 800 km — altitude bands don't overlap, should be filtered."""
        result = detector._perigee_apogee_filter(iss_like, far_sat)
        assert result is False
 
    def test_filter_is_symmetric(self, detector, iss_like, far_sat):
        """Filter result should be same regardless of which satellite is A or B."""
        assert (
            detector._perigee_apogee_filter(iss_like, far_sat) ==
            detector._perigee_apogee_filter(far_sat, iss_like)
        )
 
    def test_threshold_buffer_included(self, t_start, t_end):
        """Satellites just outside threshold should still pass with buffer."""
        detector = ConjunctionDetector(threshold_km=10.0)
        sat_410 = make_circular_trajectory(99, "A", 410.0, 51.6, t_start, t_end)
        sat_415 = make_circular_trajectory(98, "B", 415.0, 51.6, t_start, t_end)
        # 5 km apart, 10 km threshold — should pass
        assert detector._perigee_apogee_filter(sat_410, sat_415) is True
 
 
# ── TCA Computation Tests ─────────────────────────────────────────────────────
 
class TestTCAComputation:
 
    def test_near_miss_detected(self, detector, iss_like, near_miss_sat):
        """3 km offset satellite should be detected as a conjunction."""
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
 
    def test_near_miss_distance_accurate(self, detector, iss_like, near_miss_sat):
        """Miss distance should be approximately 3 km (within 0.5 km)."""
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        assert abs(event.miss_distance - 3.0) < 0.5, (
            f"Expected ~3.0 km miss distance, got {event.miss_distance:.3f} km"
        )
 
    def test_far_satellite_not_detected(self, detector, iss_like, far_sat):
        """800 km satellite should not generate a conjunction event with ISS."""
        event = detector.screen_pair(iss_like, far_sat)
        assert event is None
 
    def test_event_fields_populated(self, detector, iss_like, near_miss_sat):
        """All ConjunctionEvent fields should be filled after detection."""
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        assert event.primary_id == iss_like.norad_id
        assert event.secondary_id == near_miss_sat.norad_id
        assert event.primary_name == "ISS_LIKE"
        assert event.secondary_name == "NEAR_MISS"
        assert isinstance(event.tca, datetime)
        assert event.miss_distance > 0
        assert event.relative_velocity >= 0
        assert event.tca.tzinfo is not None   # must be UTC-aware
 
    def test_relative_velocity_plausible(self, detector, iss_like, near_miss_sat):
        """
        Parallel orbits at same altitude have ~0 relative velocity.
        Should be < 0.1 km/s for nearly-identical orbits.
        """
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        assert event.relative_velocity < 0.5
 
    def test_rtn_components_sum_to_miss_distance(self, detector, iss_like, near_miss_sat):
        """
        RTN components should satisfy: sqrt(R² + T² + N²) ≈ miss_distance.
        """
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        reconstructed = np.sqrt(
            event.radial_miss**2 +
            event.transverse_miss**2 +
            event.normal_miss**2
        )
        assert abs(reconstructed - event.miss_distance) < 0.01, (
            f"RTN magnitude {reconstructed:.3f} != miss distance {event.miss_distance:.3f}"
        )
 
    def test_normal_miss_dominates_for_cross_track_offset(self, detector, iss_like, near_miss_sat):
        """
        Since near_miss_sat is offset in the normal (cross-track) direction,
        the normal component should be the largest RTN component.
        """
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        assert abs(event.normal_miss) > abs(event.radial_miss)
        assert abs(event.normal_miss) > abs(event.transverse_miss)
 
    def test_pc_initially_none(self, detector, iss_like, near_miss_sat):
        """Pc should be None after detection — it's filled in Step 3."""
        event = detector.screen_pair(iss_like, near_miss_sat)
        assert event is not None
        assert event.pc is None
# Batch Screening test
class TestBatchScreening:
 
    def test_batch_finds_known_conjunction(
        self, detector, iss_like, near_miss_sat, far_sat, t_start, t_end
    ):
        """Batch screen should find the near-miss but not the far satellite."""
        trajectories = {
            iss_like.norad_id: iss_like,
            near_miss_sat.norad_id: near_miss_sat,
            far_sat.norad_id: far_sat,
        }
        events = detector.screen(trajectories)
        assert len(events) >= 1
        norad_pairs = {(e.primary_id, e.secondary_id) for e in events}
        assert (iss_like.norad_id, near_miss_sat.norad_id) in norad_pairs
 
    def test_batch_sorted_by_miss_distance(
        self, detector, iss_like, t_start, t_end
    ):
        """Events should be returned sorted by miss distance, closest first."""
        sat_2km = make_parallel_trajectory(99010, "SAT_2KM", iss_like, 2.0)
        sat_4km = make_parallel_trajectory(99011, "SAT_4KM", iss_like, 4.0)
        trajectories = {
            iss_like.norad_id: iss_like,
            sat_2km.norad_id: sat_2km,
            sat_4km.norad_id: sat_4km,
        }
        events = detector.screen(trajectories)
        assert len(events) >= 2
        distances = [e.miss_distance for e in events]
        assert distances == sorted(distances)
 
    def test_stats_populated_after_screen(
        self, detector, iss_like, near_miss_sat, far_sat
    ):
        """Stats dict should be populated after screening."""
        trajectories = {
            iss_like.norad_id: iss_like,
            near_miss_sat.norad_id: near_miss_sat,
            far_sat.norad_id: far_sat,
        }
        detector.screen(trajectories)
        stats = detector.stats
        assert stats["total_pairs"] == 3
        assert stats["events_found"] >= 1
 
    def test_single_satellite_no_pairs(self, detector, iss_like):
        """Single satellite should produce no events."""
        events = detector.screen({iss_like.norad_id: iss_like})
        assert events == []
 
    def test_no_duplicate_pairs(self, detector, iss_like, near_miss_sat):
        """Each pair should only appear once — not (A,B) and (B,A)."""
        trajectories = {
            iss_like.norad_id: iss_like,
            near_miss_sat.norad_id: near_miss_sat,
        }
        events = detector.screen(trajectories)
        pairs = [(e.primary_id, e.secondary_id) for e in events]
        assert len(pairs) == len(set(pairs))

# RTN Frame test

class TestRTNFrame:
 
    def test_radial_vector_projects_correctly(self):
        """A radial offset should show up entirely in the R component."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        offset = np.array([1.0, 0.0, 0.0])   # purely radial
 
        rad, trans, norm = _eci_to_rtn(r, v, offset)
        assert abs(rad - 1.0) < 1e-6
        assert abs(trans) < 1e-6
        assert abs(norm) < 1e-6
 
    def test_normal_vector_projects_correctly(self):
        """A cross-track offset should show up entirely in the N component."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        # h = r × v = [0,0,7000*7.5] → n_hat = [0,0,1]
        offset = np.array([0.0, 0.0, 1.0])   # purely normal
 
        rad, trans, norm = _eci_to_rtn(r, v, offset)
        assert abs(norm - 1.0) < 1e-6
        assert abs(rad) < 1e-6
        assert abs(trans) < 1e-6
 
    def test_rtn_preserves_magnitude(self):
        """RTN decomposition should preserve the vector magnitude."""
        r = np.array([7000.0, 1000.0, 500.0])
        v = np.array([-1.0, 7.0, 0.5])
        vec = np.array([0.5, 0.3, 0.8])
 
        rad, trans, norm = _eci_to_rtn(r, v, vec)
        magnitude_rtn = np.sqrt(rad**2 + trans**2 + norm**2)
        magnitude_orig = np.linalg.norm(vec)
 
        assert abs(magnitude_rtn - magnitude_orig) < 1e-6