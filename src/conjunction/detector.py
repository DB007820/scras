"""
Step 2: Conjunction Detection

Two Stage Sequence:
Stage 1: Perigee/Apogee filter
    Mechanics: If the apogee of Sat. A is below the perigee of Sat. B then they will never be in the same altitude (discard immediately)

Stage 2: TCA computation
    For pairs that pass stage 1, find TCA (Time Closest Approach)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
 
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
 
from src.models import Trajectory, StateVector
from src.conjunction.conjunction_models import ConjunctionEvent
 
logger = logging.getLogger(__name__)
 
R_EARTH_KM = 6371.0
DEFAULT_SCREENING_THRESHOLD_KM = 5.0
NASA_RED_LINE_KM = 1.0
 
# Minimum relative velocity to consider a pair — filters docked/co-orbiting objects
MIN_RELATIVE_VELOCITY_KMS = 0.005  # km/s — below this they are essentially co-orbiting
 
 
class ConjunctionDetector:
    """
    Screens satellite trajectories for conjunction events.
 
    Usage:
        detector = ConjunctionDetector(threshold_km=5.0)
        events = detector.screen(trajectories)
    """
 
    def __init__(
        self,
        threshold_km: float = DEFAULT_SCREENING_THRESHOLD_KM,
        min_vrel_kms: float = MIN_RELATIVE_VELOCITY_KMS,
    ):
        self.threshold_km = threshold_km
        self.min_vrel_kms = min_vrel_kms
        self._stats = {
            "total_pairs": 0,
            "passed_altitude_filter": 0,
            "skipped_co_orbiting": 0,
            "events_found": 0,
        }
 
    # ── Public API ─────────────────────────────────────────────────────────────
 
    def screen(
        self,
        trajectories: dict[int, Trajectory],
    ) -> list[ConjunctionEvent]:
        """
        Screen all satellite pairs for conjunctions.
 
        Parameters
        ----------
        trajectories : dict mapping norad_id → Trajectory
 
        Returns
        -------
        List of ConjunctionEvent, sorted by miss distance (closest first).
        """
        traj_list = list(trajectories.values())
        n = len(traj_list)
        total_pairs = n * (n - 1) // 2
 
        logger.info(
            "Screening %d satellites -> %d pairs, threshold=%.1f km",
            n, total_pairs, self.threshold_km,
        )
 
        self._stats["total_pairs"] = total_pairs
        events = []
        altitude_filtered = 0
        co_orbit_skipped  = 0

    
       # Precomputes perigee/apogee once per satellite
        perigee_apogee = {}
        for traj in traj_list:
            alts = np.linalg.norm(traj.positions(), axis=1) - R_EARTH_KM
            perigee_apogee[traj.norad_id] = (alts.min(), alts.max())

        for i, traj_a in enumerate(traj_list):
            for j, traj_b in enumerate(traj_list):
                if j <= i:
                    continue
                # Stage 1: altitude filter
                perigee_a, apogee_a = perigee_apogee[traj_a.norad_id]
                perigee_b, apogee_b = perigee_apogee[traj_b.norad_id]
                if perigee_a > apogee_b + self.threshold_km or perigee_b > apogee_a + self.threshold_km:
                    altitude_filtered += 1
                    continue
                # Stage 2: TCA computation
                event = self._compute_conjunction(traj_a, traj_b)
                if event is None:
                    continue
 
                # Post-filter: skip co-orbiting / docked objects
                if event.relative_velocity < self.min_vrel_kms:
                    co_orbit_skipped += 1
                    logger.debug(
                        "Skipping co-orbiting pair %s / %s (v_rel=%.4f km/s)",
                        traj_a.name, traj_b.name, event.relative_velocity,
                    )
                    continue
 
                events.append(event)
 
        self._stats["passed_altitude_filter"] = total_pairs - altitude_filtered
        self._stats["skipped_co_orbiting"]    = co_orbit_skipped
        self._stats["events_found"]           = len(events)
 
        logger.info(
            "Screening complete — %d passed altitude filter, "
            "%d co-orbiting skipped, %d real events found",
            self._stats["passed_altitude_filter"],
            co_orbit_skipped,
            len(events),
        )
 
        events.sort(key=lambda e: e.miss_distance)
        return events
 
    def screen_pair(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> Optional[ConjunctionEvent]:
        """Screen a single pair. Useful for testing."""
        if not self._perigee_apogee_filter(traj_a, traj_b):
            return None
        return self._compute_conjunction(traj_a, traj_b)
 
    @property
    def stats(self) -> dict:
        return self._stats.copy()
 
    # ── Stage 1: Perigee / Apogee Filter ──────────────────────────────────────
 
    def _perigee_apogee_filter(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> bool:
        """
        Fast altitude-band overlap check.
        Returns True if the pair COULD have a conjunction.
        Returns False if geometrically impossible.
        """
        altitudes_a = np.linalg.norm(traj_a.positions(), axis=1) - R_EARTH_KM
        altitudes_b = np.linalg.norm(traj_b.positions(), axis=1) - R_EARTH_KM
 
        perigee_a, apogee_a = altitudes_a.min(), altitudes_a.max()
        perigee_b, apogee_b = altitudes_b.min(), altitudes_b.max()
 
        buffer = self.threshold_km
 
        if perigee_a > apogee_b + buffer:
            return False
        if perigee_b > apogee_a + buffer:
            return False
 
        return True
 
    # ── Stage 2: TCA Computation ───────────────────────────────────────────────
 
    def _compute_conjunction(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> Optional[ConjunctionEvent]:
        """
        Find the Time of Closest Approach (TCA) between two trajectories.
 
        Steps:
        1. Build cubic interpolators over the overlapping time span
        2. Coarse scan to find approximate minimum distance
        3. Refine with scipy.optimize.minimize_scalar
        4. If miss distance < threshold, build ConjunctionEvent
        """
        epochs_a = traj_a.epochs()
        epochs_b = traj_b.epochs()
 
        t_start = max(epochs_a[0], epochs_b[0])
        t_end   = min(epochs_a[-1], epochs_b[-1])
 
        if t_start >= t_end:
            return None
 
        t0 = t_start
 
        # Filter states to overlapping window
        states_a = [sv for sv in traj_a.states if t_start <= sv.epoch <= t_end]
        states_b = [sv for sv in traj_b.states if t_start <= sv.epoch <= t_end]
 
        if len(states_a) < 2 or len(states_b) < 2:
            return None
 
        times_a = np.array([(sv.epoch - t0).total_seconds() for sv in states_a])
        times_b = np.array([(sv.epoch - t0).total_seconds() for sv in states_b])
        pos_a   = np.array([sv.position for sv in states_a])
        pos_b   = np.array([sv.position for sv in states_b])
        vel_a   = np.array([sv.velocity for sv in states_a])
        vel_b   = np.array([sv.velocity for sv in states_b])
 
        # Build interpolators
        interp_pos_a = [interp1d(times_a, pos_a[:, i], kind="cubic") for i in range(3)]
        interp_pos_b = [interp1d(times_b, pos_b[:, i], kind="cubic") for i in range(3)]
        interp_vel_a = [interp1d(times_a, vel_a[:, i], kind="cubic") for i in range(3)]
        interp_vel_b = [interp1d(times_b, vel_b[:, i], kind="cubic") for i in range(3)]
 
        t_min_s = 0.0
        t_max_s = (t_end - t_start).total_seconds()
 
        def distance_at(t_s: float) -> float:
            r_a = np.array([f(t_s) for f in interp_pos_a])
            r_b = np.array([f(t_s) for f in interp_pos_b])
            return float(np.linalg.norm(r_a - r_b))
 
        # Coarse scan — use denser grid for better minimum detection
        n_scan = min(len(states_a), len(states_b), 200)
        common_times = np.linspace(t_min_s, t_max_s, n_scan)
        distances = np.array([distance_at(t) for t in common_times])
 
        coarse_min_idx  = int(np.argmin(distances))
        coarse_min_dist = distances[coarse_min_idx]
 
        # Early exit — if coarse minimum is way above threshold, skip
        if coarse_min_dist > self.threshold_km * 3:
            return None
 
        # ── Fix: detect TCA snap to t=0 bug ──────────────────────────────────
        # If minimum is at the very first point AND distance is near-zero,
        # this is likely a co-orbiting pair whose "minimum" is just the start
        # epoch. Skip it — the co-orbit filter will catch it anyway.
        if coarse_min_idx == 0 and coarse_min_dist < 0.01:
            return None
 
        # Refine around coarse minimum
        dt = common_times[1] - common_times[0] if len(common_times) > 1 else t_max_s
        t_search_lo = max(t_min_s, common_times[coarse_min_idx] - 3 * dt)
        t_search_hi = min(t_max_s, common_times[coarse_min_idx] + 3 * dt)
 
        try:
            result = minimize_scalar(
                distance_at,
                bounds=(t_search_lo, t_search_hi),
                method="bounded",
                options={"xatol": 1.0},
            )
            tca_s = float(result.x)
            miss_distance = float(result.fun)
        except Exception as e:
            logger.warning("minimize_scalar failed for %s/%s: %s",
                           traj_a.name, traj_b.name, e)
            tca_s = float(common_times[coarse_min_idx])
            miss_distance = coarse_min_dist
 
        if miss_distance > self.threshold_km:
            return None
 
        # Build event
        tca = t0 + timedelta(seconds=tca_s)
 
        r_a = np.array([f(tca_s) for f in interp_pos_a])
        r_b = np.array([f(tca_s) for f in interp_pos_b])
        v_a = np.array([f(tca_s) for f in interp_vel_a])
        v_b = np.array([f(tca_s) for f in interp_vel_b])
 
        rel_pos   = r_b - r_a
        rel_vel   = v_b - v_a
        rel_speed = float(np.linalg.norm(rel_vel))
 
        radial, transverse, normal = _eci_to_rtn(r_a, v_a, rel_pos)
 
        logger.info(
            "CONJUNCTION: %s / %s  TCA=%s  miss=%.3f km  v_rel=%.3f km/s",
            traj_a.name, traj_b.name,
            tca.strftime("%Y-%m-%d %H:%M:%S"),
            miss_distance, rel_speed,
        )
 
        return ConjunctionEvent(
            primary_id=traj_a.norad_id,
            secondary_id=traj_b.norad_id,
            primary_name=traj_a.name,
            secondary_name=traj_b.name,
            tca=tca,
            miss_distance=miss_distance,
            relative_velocity=rel_speed,
            radial_miss=radial,
            transverse_miss=transverse,
            normal_miss=normal,
            screening_threshold_km=self.threshold_km,
        )
 
 
# ── RTN Frame ─────────────────────────────────────────────────────────────────
 
def _eci_to_rtn(
    r_primary: np.ndarray,
    v_primary: np.ndarray,
    vector_eci: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose a vector into RTN frame of the primary satellite."""
    r_hat = r_primary / np.linalg.norm(r_primary)
    h     = np.cross(r_primary, v_primary)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
 
    return (
        float(np.dot(vector_eci, r_hat)),
        float(np.dot(vector_eci, t_hat)),
        float(np.dot(vector_eci, n_hat)),
    )
 
 
def _seconds_to_epoch(t0: datetime, seconds: float) -> datetime:
    return t0 + timedelta(seconds=seconds)
 