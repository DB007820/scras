"""
SGP4 Orbit Propagator - converts TLE Recrds into time-series Statevectors

SGP4 is the standard analytical propagator used by NORAD for Earth satellites.
It accounts for:
  - J2, J3, J4 zonal harmonics (Earth oblateness)
  - Atmospheric drag
  - Solar radiation pressure (partially)
  - Lunar/solar gravity (partially, via deep-space extension SDP4)

"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np

from sgp4.api import Satrec, WGS84
from sgp4.conveniences import sat_epoch_datetime

from src.models import TLERecord, StateVector, Trajectory

logger = logging.getLogger(__name__)

class SGP4Propagator:
    """
    Propagates TLERecords forwards (or backwards, as one chooses) in time using SGP4/SDP4

    Note: 
    It's important to do this because TLERecords range from a few hours to a few days old
    The Epoch (snapshot of position) must undergo perturbations, the physics must be sequenced forward.

    Forward: Prediction purposes
    Backward: Historical analysis
    """

    def __init__(self, frame: str = "teme"):
        #teme - raw SGP4 output
        # eci - convert to ECI J2000 (slower than teme)
        self.frame = frame
    
    # Public API Section

    def propagate(
            self,
            tle: TLERecord,
            t_start: datetime,
            t_end: datetime,
            dt_seconds: float = 60.0
    ) -> Optional[Trajectory]:
        """
        Propagate a single TLE over a time span.

        Parameters
        ----------
        tle         : TLERecord to propagate
        t_start     : UTC datetime, propagation start
        t_end       : UTC datetime, propagation end
        dt_seconds  : time step in seconds (default 60 s)

        Returns
        -------
        Trajectory with one StateVector per time step
        """
        sat = self._build_satrec(tle)
        if sat is None:
            return None
        
        epochs = self._time_grid(t_start, t_end, dt_seconds)
        states = []

        for epoch in epochs:
            sv = self._propagate_to(sat, tle.norad_id, epoch)
            if sv is not None:
                states.append(sv)

        if not states:
            logger.error("No valid states for NORAD ID %d", tle.norad_id)
            return None
        
        traj = Trajectory(
            norad_id=tle.norad_id,
            name=tle.name,
            states=states,
            propagator=f"sgp4/{self.frame}",
        )
        logger.debug(
            "Propagated %s: %d steps over %.1f hours",
            tle.name, len(states), traj.duration_hours
        )
        return traj
        
    def propagate_batch(
        self,
        tles: list[TLERecord],
        t_start: datetime,
        t_end: datetime,
        dt_seconds: float = 60.0,
    ) -> dict[int, Trajectory]:
        """
        Propagate multiple TLEs over the same time span.
        Returns {norad_id: Trajectory} — failed satellites are excluded.
        """
        results = {}
        for i, tle in enumerate(tles):
            traj = self.propagate(tle, t_start, t_end, dt_seconds)
            if traj is not None:
                results[tle.norad_id] = traj
            if (i + 1) % 100 == 0:
                logger.info("Propagated %d / %d satellites", i + 1, len(tles))
        logger.info(
            "Batch propagation complete: %d / %d succeeded",
            len(results), len(tles)
        )
        return results

    def state_at(
        self,
        tle: TLERecord,
        epoch: datetime,
    ) -> Optional[StateVector]:
        """Propagate to a single epoch — useful for conjunction point calculation."""
        sat = self._build_satrec(tle)
        if sat is None:
            return None
        return self._propagate_to(sat, tle.norad_id, epoch)
    
    # Internal: SGP4

    def _build_satrec(self, tle: TLERecord) -> Optional[Satrec]:
        # Parse TLE strings into sgp4 Satrec object
        try:
            sat = Satrec.twoline2rv(tle.line1 , tle.line2, WGS84)
            return sat
        except Exception as e:
            logger.error("Failed to initialize SGP4 for NORAD %d: %s", tle.norad_id, e)
            return None
        
    def _propagate_to(
            self,
            sat: Satrec,
            norad_id: int,
            epoch: datetime,
    ) -> Optional[StateVector]:
        """
        Propagate Satrec to a specific UTC epoch.
        sgp4 expects Julian date split into whole + fractional parts.
        """
        jd, fr = _datetime_to_jd(epoch)
        error_code, r_teme, v_teme = sat.sgp4(jd, fr)

        if error_code != 0:
            # SGP4 error codes: 1=eccentricity, 2=mean motion, 3=decay, 4=decay,
            # 5=max iterations, 6=satellite number
            logger.debug(
                "SGP4 error code %d for NORAD %d at %s",
                error_code, norad_id, epoch.isoformat()
            )
            return None

        r = np.array(r_teme)   # km
        v = np.array(v_teme)   # km/s

        if self.frame == "eci":
            r, v = _teme_to_eci(r, v, epoch)

        return StateVector(
            norad_id=norad_id,
            epoch=epoch,
            position=r,
            velocity=v,
            covariance=None,   # SGP4 doesn't produce covariance; added later via filters
        )
    @staticmethod
    def _time_grid(
        t_start: datetime,
        t_end: datetime,
        dt_seconds: float,
    ) -> list[datetime]:
        """Generate evenly spaced UTC datetimes from t_start to t_end."""
        t_start = _ensure_utc(t_start)
        t_end   = _ensure_utc(t_end)

        epochs = []
        t = t_start
        step = timedelta(seconds=dt_seconds)
        while t <= t_end:
            epochs.append(t)
            t += step
        return epochs

# Frame conversion: TEME → ECI (J2000 / GCRS approximation)

def _teme_to_eci(
        r_teme: np.ndarray,
        v_teme: np.ndarray,
        epoch: datetime,
) -> tuple[np.ndarray, np.ndarray]:

    # This command converts TEME to ECI J2000 via GMST rotation (z-axis rotation)

    gmst = _compute_gmst(epoch)
    c, s = np.cos(gmst), np.sin(gmst)

    # Rotation matrix:
    R = np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1],
        ])

    # Earth rotation rate:
    omega_E = 7.2921150e-5  # rad/s

    r_eci = R @ r_teme

    # Velocity correction:
    omega_vec = np.array([0, 0, omega_E])
    v_eci = R @ (v_teme + np.cross(omega_vec, r_teme))

    return r_eci, v_eci

def _compute_gmst(epoch: datetime) -> float:
    # greenwich Mean sidereal time in radians

    epoch = _ensure_utc(epoch)

    # Julian date of J2000.0
    J2000 = 2451545.0
    jd, fr = _datetime_to_jd(epoch)
    T = ((jd - J2000) + fr) / 36525.0   # Julian centuries from J2000

    # IAU 1982 GMST formula (degrees)
    gmst_deg = (
        280.46061837
        + 360.98564736629 * ((jd - J2000) + fr)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
  
    # Normalize to [0, 360) then convert to radians
    gmst_deg = gmst_deg % 360.0
    return np.deg2rad(gmst_deg)

#Julian Date Utilities

def _datetime_to_jd(dt: datetime) -> tuple[float, float]:
    """
    Convert a UTC datetime to Julian date split as (JD_whole, JD_fraction).
    sgp4.api.Satrec.sgp4() requires this split form for numerical precision.

    Julian date of J2000 (2000 Jan 1.5 TT) = 2451545.0
    """
    dt = _ensure_utc(dt)

    # Reference: Julian date of 2000-01-01 00:00:00 UTC = 2451544.5
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    J2000_JD = 2451545.0

    delta = dt - J2000_EPOCH
    days = delta.total_seconds() / 86400.0

    jd_full = J2000_JD + days
    jd_whole = float(int(jd_full))
    jd_frac  = jd_full - jd_whole
    return jd_whole, jd_frac


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone if datetime is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt