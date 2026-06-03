"""
Details & Explanation as to how and what this file is doing:

Core Idea:
at TCA (momen when two satellites reach their minimum distance), you don't really know precisely where all the satellites are, you have the next best thing:
a predicted position, that comes with all the uncertainty around it.
The uncertaity is shaped like a bell curve.

The question this file answers: "with this uncertainty, what is the chance both objects occupy in the same space"

Why 2D?
In TCA, the relative velocity is incredibly fast for what it is (7 km/s) 
so you freeze time and ask in the plane perpendicular to that relative velocity vector, 
where is satellite B relative to satellite A?

Then Probability of B is integrated to evaluate its position within A's radius position.
So: the fraction of B's probability distribution that overlaps with A's physical footprint.

PC: a metric to calculate the probability of a conjunction event happening given the uncertainty of where the satellite is.

NASA Metric:
    Green  — Pc < 1e-5         (no action)
    Yellow — 1e-5 ≤ Pc < 1e-4  (monitor closely)
    Red    — Pc ≥ 1e-4         (maneuver recommended)
 
References:
    Alfano, S. (2005). "A Numerical Implementation of Spherical Object
        Collision Probability." Journal of the Astronautical Sciences.
    Chan, F.K. (2008). "Spacecraft Collision Probability." AIAA.
    Foster, J.L. & Estes, H.S. (1992). "A Parametric Analysis of
        Orbital Debris Collision Probability and Maneuver Rate."
        NASA JSC-25898.
"""

import logging 
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import dblquad

from src.conjunction.conjunction_models import ConjunctionEvent

logger = logging.getLogger(__name__)

# ── Default hard-body radii (meters → km) ─────────────────────────────────────

HBR_DEFAULTS = {
    "iss": 0.150,
    "default": 0.010,
    "cubesat": 0.005
}

# Defualt Covariance is used when no real covariance is available

DEFAULT_COVARIANCE_RTN_KM = np.diag([0.09, 9.0, 0.09])

"""
Result of a collision Probability Clculaiton

pc: DImensionless, between 0 and 1.

miss_distance: miss distance at TCA (km) (scalar)
r_hbr : hard body radius (km)
sigma_x: 1-sigma uncertainty in collision plane x-direction (km)
sigma_y: 1-sigma uncertainty in collision plane y-direction (km)
risk_level: risk_level     : "green" / "yellow" / "red" per NASA traffic light

"""

@dataclass
class PcResult:
    pc: float
    miss_distance: float
    r_hbr: float
    sigma_x: float
    sigma_y: float
    risk_level: str
    used_default_covariance: bool = False

    @property
    def is_actionable(self) -> bool:
        return self.pc >= 1e-5
    
    def __repr__(self):
        return (
            f"PcResult(Pc={self.pc:.3e}, miss={self.miss_distance:.3f} km, "
            f"risk={self.risk_level.upper()}, "
            f"σ=({self.sigma_x:.3f}, {self.sigma_y:.3f}) km)"
        )

"""

    Computes collision probability for conjunction events.
 
    Usage:
        calc = PcCalculator(r_hbr_km=0.010)
        result = calc.compute(event)
"""

# ── Public API ─────────────────────────────────────────────────────────────

"""
        Compute Pc for a single conjunction event.
 
        Parameters
        ----------
        event            : ConjunctionEvent from Step 2
        cov_primary_eci  : 3×3 position covariance of primary in ECI (km²)
        cov_secondary_eci: 3×3 position covariance of secondary in ECI (km²)
                           If None, uses conservative SGP4 default.
 
        Returns
        -------
        PcResult with Pc and risk level. Also writes Pc into event.pc.
        """
class PcCalculator:
    def __init__(self, r_hbr_km: float = HBR_DEFAULTS["default"]):
        self.r_hbr_km = r_hbr_km

    def compute(
            self,
            event: ConjunctionEvent,
            cov_primary_eci: Optional[np.ndarray] = None,
            cov_secondary_eci: Optional[np.ndarray] = None,
    ) -> PcResult:
        used_default = False

        if cov_primary_eci is None or cov_secondary_eci is None:
            used_default = True

            cov_primary_eci = self._default_covariance_eci()
            cov_secondary_eci = self._default_covariance_eci()

        cov_combined = cov_primary_eci + cov_secondary_eci

        pc, sigma_x, sigma_y = self._compute_pc_2d(
            event=event,
            cov_combined_eci=cov_combined,
        )

        risk_level = self._classify_risk(pc)

        result = PcResult(
            pc = pc,
            miss_distance=event.miss_distance,
            r_hbr=self.r_hbr_km,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            risk_level=risk_level,
            used_default_covariance=used_default,
        )

        event.pc = pc

        logger.info(
            "Pc computed: %s / %s  Pc=%.3e  risk=%s  miss=%.3f km  σ=(%.3f, %.3f) km",
            event.primary_name, event.secondary_name,
            pc, risk_level.upper(),
            event.miss_distance, sigma_x, sigma_y,
        )

        return result
    
    def compute_batch(
        self,
        events: list[ConjunctionEvent],
        covariances: Optional[dict[int, np.ndarray]] = None,
    ) -> list[PcResult]:
        
        results = []
        for event in events:
            cov_p = cov_s = None
            if covariances:
                cov_p = covariances.get(event.primary_id)
                cov_s = covariances.get(event.secondary_id)
            results.append(self.compute(event, cov_p, cov_s))
 
        n_red    = sum(1 for r in results if r.risk_level == "red")
        n_yellow = sum(1 for r in results if r.risk_level == "yellow")
        n_green  = sum(1 for r in results if r.risk_level == "green")
 
        logger.info(
            "Batch Pc complete: %d events — 🔴 %d red  🟡 %d yellow  🟢 %d green",
            len(results), n_red, n_yellow, n_green,
        )
 
        return results
 
    # ── Core 2D Gaussian Integration ─────────────────────────────────────────
    """
        Methodology:

        I. project covariance onto collision plane, integrate.

        II. The collision plane is perpendicular to the relative velocity vector

        III.at TCA. We decompose the combined covariance onto this plane,

        IV.then integrate a 2D Gaussian over the hard-body disk.
        """
 
    def _compute_pc_2d(
        self,
        event: ConjunctionEvent,
        cov_combined_eci: np.ndarray,
    ) -> tuple[float, float, float]:        
        
        rel_vel_mag = max(event.relative_velocity, 1e-6)
        sigma_x, sigma_y = self._project_covariance_to_collision_plane(
            cov_combined_eci, event
        )

        x_miss = event.radial_miss
        y_miss = event.normal_miss

        r_hbr = self.r_hbr_km

        x_bound = max(5 * sigma_x, r_hbr * 2)
        y_bound = max(5 * sigma_y, r_hbr * 2)

        def integrand(y, x):
            """2D Gaussian density centered at miss point."""
            dx = x - x_miss
            dy = y - y_miss
            exponent = -0.5 * ((dx / sigma_x)**2 + (dy/sigma_y)**2)

            return np.exp(exponent) / (2 * np.pi * sigma_x * sigma_y)
    
        try:
            pc_raw, _error = dblquad(
                integrand,
                -r_hbr, r_hbr,                              # x limits: disk diameter
                lambda x: -np.sqrt(r_hbr**2 - x**2),       # y lower: circle boundary
                lambda x:  np.sqrt(r_hbr**2 - x**2),       # y upper: circle boundary
                epsabs=1e-10, epsrel=1e-6,
            )
            pc = pc_raw

        except Exception as e:
            logger.warning("dblquad failed (%s), using Foster approximation", e)
            pc = self._foster_approximation(x_miss, y_miss, sigma_x, sigma_y, r_hbr)

        return float(np.clip(pc, 0.0, 1.0)), sigma_x, sigma_y
    
    def _integrate_over_disk(
            self,
            x_miss: float,
            y_miss: float,
            sigma_x: float,
            sigma_y: float,
            r_hbr: float,
            n_points: int = 500,
    ) -> float:
        from numpy.polynomial.legendre import leggauss

        n_r = n_points // 10
        n_t = n_points

        r_nodes, r_weights, = leggauss(n_r)
        t_nodes, t_weights = leggauss(n_t)

        r_vals = 0.5 * r_hbr * (r_nodes + 1)
        r_w = 0.5 * r_hbr * r_weights

        t_vals = np.pi * (t_nodes + 1)
        t_w    = np.pi * t_weights

        r_grid, t_grid = np.meshgrid(r_vals, t_vals)
        r_wgrid, t_wgrid = np.meshgrid(r_w, t_w)
 
        x = r_grid * np.cos(t_grid)
        y = r_grid * np.sin(t_grid)
 
        dx = x - x_miss
        dy = y - y_miss
        gaussian = np.exp(-0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2))
        gaussian /= (2 * np.pi * sigma_x * sigma_y)

        pc = np.sum(gaussian * r_grid * r_wgrid * t_wgrid)
        return float(pc)
    
    def _foster_approximation(
            self,
            x_miss: float,
            y_miss: float,
            sigma_x: float,
            sigma_y: float,
            r_hbr: float,
    ) -> float:
        u_sq = (x_miss / sigma_x)**2 + (y_miss / sigma_y)**2
        pc = (r_hbr**2 / (2* sigma_x * sigma_y)) * np.exp(-0.5 * u_sq)

        return float(pc)
    
# ── Covariance Projection ──────────────────────────────────────────────────
 
    def _project_covariance_to_collision_plane(
        self,
        cov_eci: np.ndarray,
        event: ConjunctionEvent,
    ) -> tuple[float, float]:
        cov_2d = cov_eci[:2, :2]   
        eigenvalues = np.linalg.eigvalsh(cov_2d)
        eigenvalues = np.abs(eigenvalues)   
 
        sigma_major = np.sqrt(max(eigenvalues))
        sigma_minor = np.sqrt(min(eigenvalues))
 
        
        sigma_x = max(sigma_major, 0.001)   
        sigma_y = max(sigma_minor, 0.001)
 
        return sigma_x, sigma_y
 
    # ── Helpers ────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _default_covariance_eci() -> np.ndarray:
        """
        Conservative default 3×3 ECI covariance when no real covariance exists.
 
        SGP4 typical uncertainties (1-sigma):
            Along-track (T): ~1-3 km   → use 3.0 km
            Cross-track (N): ~0.3 km
            Radial (R):      ~0.3 km
 
        This is diagonal — real covariance matrices have off-diagonal terms
        (position and velocity errors are coupled). Filled by Kalman filter
        in Step 7.
        """
        return np.diag([0.09, 9.0, 0.09])   # km² — [x, y, z] variance
 
    @staticmethod
    def _classify_risk(pc: float) -> str:
        """NASA traffic light classification."""
        if pc >= 1e-4:
            return "red"
        elif pc >= 1e-5:
            return "yellow"
        else:
            return "green"

