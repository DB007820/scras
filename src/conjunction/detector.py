"""
Step 2: Conjunction Detection

Two Stage Sequence:
Stage 1: Perigee/Apogee filter
    Mechanics: If the apogee of Sat. A is below the perigee of Sat. B then they will never be in the same altitude (discard immediately)

Stage 2: TCA computation
    For pairs that pass stage 1, find TCA (Time Closest Approach)
"""

import logging
import itertools
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from src.models import Trajectory, StateVector
from src.conjunction.conjunction_models import ConjunctionEvent

logger = logging.getLogger(__name__)

## Constants
R_EARTH_KM = 6371.0
DEFAULT_SCREENING_THRESHOLD_KM = 5.0    # Flag if miss distance < 5 km
NASA_RED_LINE_KM = 1.0                  # NASA hard-body threshold

class ConjunctionDetector:
    '''
    Screens satellite trajectories for conjunction events.
    The threshold_km controls which events get flagged — default 5 km
    '''
    def __init__(self, threshold_km: float = DEFAULT_SCREENING_THRESHOLD_KM):
        self.threshold_km = threshold_km
        self._stats = {
            "total_pairs": 0,
            "passed_filter": 0,
            "events_found": 0,
        }
    
    ## Public API

    def screen(
         self,
        trajectories: dict[int, Trajectory],   
    ) -> list[ConjunctionEvent]:
        """
        Screen all satellite pairs for conjunctions.

        Parameters
        ----------
        trajectories : dict mapping norad_id → Trajectory
                       All satellites must cover the same time span.

        Returns
        -------
        List of ConjunctionEvent, sorted by miss distance (closest first).
        """
        traj_list = list(trajectories.values())
        n = len(traj_list)
        total_pairs = n * (n - 1) // 2

        logger.info(
            "Screening %d satellites → %d pairs, threshold=%.1f km",
            n, total_pairs, self.threshold_km
        )

        self._stats["total_pairs"] = total_pairs
        events = []
        filtered_out = 0

        for i, traj_a in enumerate(traj_list):
            for j, traj_b in enumerate(traj_list):
                if j <= i:
                    continue  # avoids duplicate pairs (A,B) == (B,A)

                # ── Stage 1: Perigee/Apogee filter ───────────────────────────
                if not self._perigee_apogee_filter(traj_a, traj_b):
                    filtered_out += 1
                    continue

                # ── Stage 2: TCA computation ──────────────────────────────────
                event = self._compute_conjunction(traj_a, traj_b)
                if event is not None:
                    events.append(event)

        self._stats["passed_filter"] = total_pairs - filtered_out
        self._stats["events_found"] = len(events)

        logger.info(
            "Screening complete — %d / %d pairs passed filter, %d events found",
            self._stats["passed_filter"],
            total_pairs,
            len(events),
        )

        # Sort by miss distance — closest approaches first
        events.sort(key=lambda e: e.miss_distance)
        return events

    def screen_pair(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> Optional[ConjunctionEvent]:
        """
        Screen a single pair of trajectories.
        Useful for testing and targeted analysis.
        """
        if not self._perigee_apogee_filter(traj_a, traj_b):
            return None
        return self._compute_conjunction(traj_a, traj_b)
    
    @property
    def stats(self) -> dict:
        """Screening statistics from the last run."""
        return self._stats.copy()

    # ── Stage 1: Perigee / Apogee Filter ──────────────────────────────────────

    def _perigee_apogee_filter(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> bool:
        """
        Fast altitude-band overlap check.

        If satellite A's trajectory stays above satellite B's
        maximum altitude (or vice versa), !THEY CAN NEVER MEET!.
        Eliminates ~99% of pairs in microseconds.

        Returns True if the pair COULD have a conjunction (keep it).
        Returns False if conjunction is geometrically impossible (discard).
        """
        altitudes_a = np.linalg.norm(traj_a.positions(), axis=1) - R_EARTH_KM
        altitudes_b = np.linalg.norm(traj_b.positions(), axis=1) - R_EARTH_KM

        perigee_a, apogee_a = altitudes_a.min(), altitudes_a.max()
        perigee_b, apogee_b = altitudes_b.min(), altitudes_b.max()

        # Perigee/Apogee -> Screening Buffer (Go/no go) -> Phase 2
        # if they come within threshold_km of each other's altitude band, keep the pair:
        buffer = self.threshold_km

        # Discards if A's entire range is above B's maximum + buffer
        if perigee_a > apogee_b + buffer:
            return False

        # Discards if B's entire range is above A's maximum + buffer
        if perigee_b > apogee_a + buffer:
            return False

        return True
    
# ── Stage 2: TCA Computation ─────────────────────────

    def _compute_conjunction(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
    ) -> Optional[ConjunctionEvent]:
        """
        Find the Time of Closest Approach (TCA) between two trajectories.

        Method:
        1. Compute distance at every time step (coarse grid)
        2. Find the time step with minimum distance (coarse TCA estimate)
        3. Refine with scipy.optimize.minimize_scalar around the coarse minimum
        4. If miss distance < threshold, build a ConjunctionEvent

        The coarse + refine approach is much faster than running the optimizer
        cold over the entire time span.
        """
        # ── Build interpolators over the common time span ─────────────────────
        epochs_a = traj_a.epochs()
        epochs_b = traj_b.epochs()

        # Find overlapping time range
        t_start = max(epochs_a[0], epochs_b[0])
        t_end   = min(epochs_a[-1], epochs_b[-1])

        if t_start >= t_end:
            logger.debug(
                "No time overlap between %s and %s",
                traj_a.name, traj_b.name
            )
            return None

        # Convert epochs to seconds since t_start for numerical stability
        t0 = t_start
        times_a = np.array([
            (e - t0).total_seconds() for e in epochs_a
            if t_start <= e <= t_end
        ])
        times_b = np.array([
            (e - t0).total_seconds() for e in epochs_b
            if t_start <= e <= t_end
        ])

        pos_a = np.array([
            sv.position for sv in traj_a.states
            if t_start <= sv.epoch <= t_end
        ])
        pos_b = np.array([
            sv.position for sv in traj_b.states
            if t_start <= sv.epoch <= t_end
        ])
        vel_a = np.array([
            sv.velocity for sv in traj_a.states
            if t_start <= sv.epoch <= t_end
        ])
        vel_b = np.array([
            sv.velocity for sv in traj_b.states
            if t_start <= sv.epoch <= t_end
        ])

        if len(times_a) < 2 or len(times_b) < 2:
            return None

        # ── Build cubic interpolators for each position/velocity component ────
        interp_pos_a = [interp1d(times_a, pos_a[:, i], kind="cubic") for i in range(3)]
        interp_pos_b = [interp1d(times_b, pos_b[:, i], kind="cubic") for i in range(3)]
        interp_vel_a = [interp1d(times_a, vel_a[:, i], kind="cubic") for i in range(3)]
        interp_vel_b = [interp1d(times_b, vel_b[:, i], kind="cubic") for i in range(3)]

        t_min_s = 0.0
        t_max_s = (t_end - t_start).total_seconds()

        def distance_at(t_s: float) -> float:
            """Interpolated distance between the two satellites at time t_s."""
            r_a = np.array([f(t_s) for f in interp_pos_a])
            r_b = np.array([f(t_s) for f in interp_pos_b])
            return float(np.linalg.norm(r_a - r_b))

        # ── Coarse scan: find approximate minimum ─────────────────────────────
        # Use the common time grid (whichever is coarser)
        common_times = np.linspace(t_min_s, t_max_s, min(len(times_a), len(times_b)))
        distances = np.array([distance_at(t) for t in common_times])
        coarse_min_idx = int(np.argmin(distances))
        coarse_min_dist = distances[coarse_min_idx]

        # Early exit — if even the coarse minimum is way above threshold, skip
        if coarse_min_dist > self.threshold_km * 3:
            return None

        # ── Refine: golden-section search around the coarse minimum ──────────
        # Search window: ±2 time steps around the coarse minimum
        dt = common_times[1] - common_times[0] if len(common_times) > 1 else t_max_s
        t_search_lo = max(t_min_s, common_times[coarse_min_idx] - 2 * dt)
        t_search_hi = min(t_max_s, common_times[coarse_min_idx] + 2 * dt)

        result = minimize_scalar(
            distance_at,
            bounds=(t_search_lo, t_search_hi),
            method="bounded",
            options={"xatol": 1.0},   # 1-second accuracy on TCA
        )

        tca_s = float(result.x)
        miss_distance = float(result.fun)

        # ── Threshold check ───────────────────────────────────────────────────
        if miss_distance > self.threshold_km:
            return None

        # ── Build ConjunctionEvent ────────────────────────────────────────────
        tca = _seconds_to_epoch(t0, tca_s)

        r_a = np.array([f(tca_s) for f in interp_pos_a])
        r_b = np.array([f(tca_s) for f in interp_pos_b])
        v_a = np.array([f(tca_s) for f in interp_vel_a])
        v_b = np.array([f(tca_s) for f in interp_vel_b])

        rel_pos = r_b - r_a              # miss vector
        rel_vel = v_b - v_a              # relative velocity vector
        rel_speed = float(np.linalg.norm(rel_vel))

        # ── RTN decomposition of miss vector ─────────────────────────────────
        radial, transverse, normal = _eci_to_rtn(r_a, v_a, rel_pos)

        logger.info(
            "CONJUNCTION: %s / %s  TCA=%s  miss=%.3f km  v_rel=%.3f km/s",
            traj_a.name, traj_b.name,
            tca.strftime("%Y-%m-%d %H:%M:%S"),
            miss_distance,
            rel_speed,
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


# ── RTN Frame Decomposition ────────────────────────────────────────────────────

def _eci_to_rtn(
    r_primary: np.ndarray,
    v_primary: np.ndarray,
    vector_eci: np.ndarray,
) -> tuple[float, float, float]:
    """
    Decompose a vector from ECI into RTN frame of the primary satellite.

    RTN (Radial-Transverse-Normal) is the standard frame for describing
    conjunction geometry because it aligns with the orbital motion:
        R — radial, points from Earth center through the satellite
        T — transverse, along the velocity direction (along-track)
        N — normal, perpendicular to orbital plane (h = r × v direction)

    Parameters
    ----------
    r_primary : position vector of primary satellite in ECI (km)
    v_primary : velocity vector of primary satellite in ECI (km/s)
    vector_eci : vector to decompose (e.g., miss vector r_b - r_a)

    Returns
    -------
    (radial, transverse, normal) components in km
    """
    # Unit vectors of RTN frame
    r_hat = r_primary / np.linalg.norm(r_primary)                    # radial
    h     = np.cross(r_primary, v_primary)
    n_hat = h / np.linalg.norm(h)                                     # normal
    t_hat = np.cross(n_hat, r_hat)                                    # transverse

    radial     = float(np.dot(vector_eci, r_hat))
    transverse = float(np.dot(vector_eci, t_hat))
    normal     = float(np.dot(vector_eci, n_hat))

    return radial, transverse, normal


def _seconds_to_epoch(t0: datetime, seconds: float) -> datetime:
    """Convert seconds offset from t0 back to a UTC datetime."""
    from datetime import timedelta
    return t0 + timedelta(seconds=seconds)
