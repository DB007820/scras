import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
 
from src.models import StateVector, Trajectory
from src.kalman.kalman_filter import (
    ExtendedKalmanFilter, KalmanState, RadarObservation,
    _rk4_propagate, _two_body_j2, _simulate_observation,
    kalman_covariances_to_eci,
)
 
R_EARTH = 6371.0
MU      = 398600.4418
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def circular_state(altitude_km: float = 410.0) -> np.ndarray:
    """ISS-like circular orbit state vector in ECI."""
    r = R_EARTH + altitude_km
    v = np.sqrt(MU / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])
 
 
def make_trajectory(
    norad_id: int = 25544,
    altitude_km: float = 410.0,
    n_states: int = 30,
    dt_seconds: float = 60.0,
) -> Trajectory:
    """Build a synthetic circular trajectory for testing."""
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    x0 = circular_state(altitude_km)
    states = []
    x = x0.copy()
    for i in range(n_states):
        t = t0 + timedelta(seconds=i * dt_seconds)
        states.append(StateVector(
            norad_id=norad_id,
            epoch=t,
            position=x[:3].copy(),
            velocity=x[3:].copy(),
        ))
        if i < n_states - 1:
            x = _rk4_propagate(x, dt_seconds)
    return Trajectory(norad_id=norad_id, name="TEST_SAT", states=states,
                      propagator="synthetic")
 
 
# ── Dynamics Tests ─────────────────────────────────────────────────────────────
 
class TestOrbitalDynamics:
 
    def test_rk4_preserves_energy(self):
        """Energy should be approximately conserved over short propagation."""
        x0 = circular_state(410.0)
        r0 = np.linalg.norm(x0[:3])
        v0 = np.linalg.norm(x0[3:])
        E0 = 0.5 * v0**2 - MU / r0
 
        x1 = _rk4_propagate(x0, 60.0)
        r1 = np.linalg.norm(x1[:3])
        v1 = np.linalg.norm(x1[3:])
        E1 = 0.5 * v1**2 - MU / r1
 
        assert abs(E1 - E0) / abs(E0) < 1e-6, "Energy not conserved"
 
    def test_rk4_preserves_altitude(self):
        """Circular orbit altitude should stay approximately constant."""
        x0 = circular_state(410.0)
        x1 = _rk4_propagate(x0, 60.0)
        alt0 = np.linalg.norm(x0[:3]) - R_EARTH
        alt1 = np.linalg.norm(x1[:3]) - R_EARTH
        assert abs(alt1 - alt0) < 0.01, f"Altitude changed: {alt0:.3f} → {alt1:.3f} km"
 
    def test_two_body_acceleration_direction(self):
        """Acceleration should point toward Earth center (negative radial)."""
        x = circular_state(410.0)
        dxdt = _two_body_j2(x)
        acc = dxdt[3:]
        pos_hat = x[:3] / np.linalg.norm(x[:3])
        # Acceleration should be anti-parallel to position
        dot = np.dot(acc / np.linalg.norm(acc), pos_hat)
        assert dot < -0.99, "Acceleration not pointing toward Earth"
 
 
# ── Initialization Tests ──────────────────────────────────────────────────────
 
class TestInitialization:
 
    def test_initialize_returns_kalman_state(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        assert isinstance(state, KalmanState)
 
    def test_initial_state_matches_input(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        np.testing.assert_array_almost_equal(state.position, x0[:3])
        np.testing.assert_array_almost_equal(state.velocity, x0[3:])
 
    def test_initial_covariance_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        eigenvalues = np.linalg.eigvalsh(state.P)
        assert np.all(eigenvalues > 0), "Initial covariance not positive definite"
 
    def test_initial_updates_zero(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        assert state.n_updates == 0
 
 
# ── Predict Step Tests ────────────────────────────────────────────────────────
 
class TestPredictStep:
 
    def test_predict_advances_epoch(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=60)
        state0 = ekf.initialize(25544, x0[:3], x0[3:], t0)
        state1 = ekf.predict(state0, t1)
        assert state1.epoch == t1
 
    def test_predict_grows_covariance(self):
        """Uncertainty should increase after prediction (no measurement)."""
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=600)
        state0 = ekf.initialize(25544, x0[:3], x0[3:], t0)
        state1 = ekf.predict(state0, t1)
        trace0 = np.trace(state0.P)
        trace1 = np.trace(state1.P)
        assert trace1 > trace0, "Covariance should grow during prediction"
 
    def test_predict_covariance_stays_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=300)
        state0 = ekf.initialize(25544, x0[:3], x0[3:], t0)
        state1 = ekf.predict(state0, t1)
        eigenvalues = np.linalg.eigvalsh(state1.P)
        assert np.all(eigenvalues > 0)
 
    def test_predict_zero_dt_unchanged(self):
        """Predicting to same time should not change state."""
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state0 = ekf.initialize(25544, x0[:3], x0[3:], t0)
        state1 = ekf.predict(state0, t0)
        np.testing.assert_array_almost_equal(state0.x, state1.x)
 
 
# ── Update Step Tests ─────────────────────────────────────────────────────────
 
class TestUpdateStep:
 
    def test_update_shrinks_covariance(self):
        """Covariance should decrease after a measurement update."""
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        radar_pos = np.array([R_EARTH, 0.0, 0.0])
 
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        sv = StateVector(norad_id=25544, epoch=t0,
                         position=x0[:3], velocity=x0[3:])
        obs = _simulate_observation(sv, radar_pos)
 
        state_updated = ekf.update(state, obs, radar_pos)
        trace_before = np.trace(state.P)
        trace_after  = np.trace(state_updated.P)
        assert trace_after < trace_before, "Covariance should shrink after update"
 
    def test_update_increments_counter(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        radar_pos = np.array([R_EARTH, 0.0, 0.0])
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        sv = StateVector(norad_id=25544, epoch=t0,
                         position=x0[:3], velocity=x0[3:])
        obs = _simulate_observation(sv, radar_pos)
        state_updated = ekf.update(state, obs, radar_pos)
        assert state_updated.n_updates == 1
 
    def test_update_covariance_stays_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        x0 = circular_state()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        radar_pos = np.array([R_EARTH, 0.0, 0.0])
        state = ekf.initialize(25544, x0[:3], x0[3:], t0)
        sv = StateVector(norad_id=25544, epoch=t0,
                         position=x0[:3], velocity=x0[3:])
        obs = _simulate_observation(sv, radar_pos)
        state_updated = ekf.update(state, obs, radar_pos)
        eigenvalues = np.linalg.eigvalsh(state_updated.P)
        assert np.all(eigenvalues > 0)
 
 
# ── Full Trajectory Processing Tests ─────────────────────────────────────────
 
class TestTrajectoryProcessing:
 
    def test_process_trajectory_returns_kalman_state(self):
        ekf = ExtendedKalmanFilter()
        traj = make_trajectory()
        result = ekf.process_trajectory(traj, n_observations=5)
        assert isinstance(result, KalmanState)
 
    def test_process_trajectory_reduces_uncertainty(self):
        """Final covariance should be smaller than initial after observations."""
        ekf = ExtendedKalmanFilter(initial_pos_std=3.0)
        traj = make_trajectory(n_states=30)
        result = ekf.process_trajectory(traj, n_observations=10)
        # Final sigma should be less than initial 3 km
        final_sigmas = result.sigma_position
        assert np.all(final_sigmas < 3.0), (
            f"Uncertainty did not reduce: σ={final_sigmas}"
        )
 
    def test_process_batch_returns_all_satellites(self):
        ekf = ExtendedKalmanFilter()
        trajectories = {
            25544: make_trajectory(25544, 410.0),
            99001: make_trajectory(99001, 500.0),
            99002: make_trajectory(99002, 600.0),
        }
        results = ekf.process_batch(trajectories, n_observations=5)
        assert len(results) == 3
        assert all(k in results for k in [25544, 99001, 99002])
 
    def test_kalman_covariances_extractable(self):
        """kalman_covariances_to_eci should return 3×3 matrices."""
        ekf = ExtendedKalmanFilter()
        trajectories = {
            25544: make_trajectory(25544),
            99001: make_trajectory(99001),
        }
        kalman_states = ekf.process_batch(trajectories, n_observations=5)
        covs = kalman_covariances_to_eci(kalman_states)
        for norad_id, cov in covs.items():
            assert cov.shape == (3, 3), f"Covariance shape wrong for {norad_id}"
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues > 0), f"Covariance not PD for {norad_id}"