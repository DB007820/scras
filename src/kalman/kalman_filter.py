"""
Step 5: Kalman filter - Covariance Estimation

Function:
Refines SGP4 position/velocity estimates using simulated radar observations
Produces calibrated covariance matrices for pc Calculation.

I. Covariance
What does the Kalman Filter do?

A method of tracking that refines noisy measurement by estimating the most optimal estimate of where the satellite is.

What is a Covariance Matrix?

To answer this, what is variance? Variance is how spread out a single variable is around its average. Covariance, is a metric of estimation
that asks; when one variable is off, what does that tell us about another variable.
A corelational deviation from average stemming from another related variable from the actual observable.

Thresholds of Covariance:
Positve - error moves together (proportional)
Negative - error moves opposite from one another (inversely proportional)
Zero = independant

The Covariance Matrix:
All relationships at once, in this case satellites.

x      y      z      vx     vy     vz
x  [ σx²   cxy    cxz  | cxvx  cxvy  cxvz ]
y  [ cxy   σy²    cyz  | cyvx  cyvy  cyvz ]
z  [ cxz   cyz    σz²  | czvx  czvy  czvz ]
   [---------------------------------------]
vx [ cxvx  cyvx  czvx |  σvx²  ...   ... ]
vy [ cxvy  cyvy  czvy |  ...   σvy²  ... ]
vz [ cxvz  cyvz  czvz |  ...   ...  σvz² ]

Diagonal - variance of each variable
off-Diagonal - covariance between pairs (how errors are linked)

II. Jacobian

Defintion: a matrix of partial derivatives
Use: defines how sensitive each output to each input

Kalman Filter:
F = ∂f/∂x  =  how much does the output change
              per unit change in each input

III. Finite Difference
SGP4/RK4 is nonlinear - can't compute F analytically with pen and paper.
So the code needs to approximate it numerically.

IV. Hamiltonian-Jacobi
Approach: Rewritig Newton in terms of energy

Newton says: F = ma (track forces)

Hamilton says: track energy instead
and equations of motion fall out automatically.

EoM (Hamiltonian):
qdot ​= ∂H/∂p
​p˙​= −∂H/∂q

Jacobi came in and said
"I'll make one that represents entire dynamic of system
embedded in a single PDE":

∂S/∂t + H (q, ∂S/∂q, t) = 0\

S - Hamilton principal function (action accumulated along trajectory)

- Explains why orbits conserve energy and angular momentum
- We use two-body problem + J2 as the dynamics model

References:

Goldstein, Poole & Safko, Classical Mechanics, 3rd ed. (2002), Ch. 10 — Hamilton-Jacobi theory
Vallado (2013), Ch. 2 — application to orbital mechanics specifically
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from src.models import StateVector, Trajectory

logger = logging.getLogger(__name__)

#-----Constants------------------------------------------------------------------------------------------------------------
MU = 398600.4418 # km^3/s^2 (Earth's Gravitational Parameter)
R_EARTH = 6371.00 # km
J2 = 1.08262668e-3 # J2 oblateness coefficient
RE_EQ = 6378.18 # Earth Equatorial radius

# all constants can be obtained at Vallado, Fundamentals of Astrodynamics and Applications, 4th ed. Table 1.1 Appendix D

"""
Radar obrevationswere simulated by using zero-mean Gaussian noice
It takes SGP4 position and converts them to what a radar would see,
then adds random Gaussian noise to "fake" the measurement error

range_km: slant range
azimuth_rad: azimuth angle (radians, measured from North)
elevation_rad: elevation angle above horizon (radians)
epoch: time of measurement (UTC)
noise_std: 1-sigma measurement noise (range_km, az_rad, el_rad)

"""

@dataclass
class RadarObservation:
    epoch: datetime
    range_km: float
    azimuth_rad: float
    elevation_rad: float
    noise_std: np.ndarray = field(
        default_factory=lambda:np.array([0.1, 0.001, 0.001])
    )

@dataclass
class KalmanState:
    norad_id: int
    epoch: datetime
    x: np.ndarray
    P: np.ndarray
    n_updates: int = 0

    @property
    def position(self) -> np.ndarray:  # Fix: was 'elf'
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]

    @property
    def position_covariance(self) -> np.ndarray:
        """3x3 covariance position block."""
        return self.P[:3,:3]

    @property
    def sigma_position(self) -> np.ndarray:
        """1-sigma position uncertainties [σ_x, σ_y, σ_z] in km."""
        return np.sqrt(np.diag(self.P[:3, :3]))

    def __repr__(self):
        sigmas = self.sigma_position
        return (
            f"KalmanState(NORAD={self.norad_id}, "
            f"epoch={self.epoch.strftime('%H:%M:%S')}, "
            f"σ_pos=({sigmas[0]:.3f}, {sigmas[1]:.3f}, {sigmas[2]:.3f}) km, "
            f"updates={self.n_updates})"
        )

"""Extended Kalman filter for satellite orbit determination
Linearizes the nonlinear orbital dynamics, uses Jacobian (finite differences)
"""

class ExtendedKalmanFilter:
    def __init__(
        self,
        process_noise_std: float=0.001, #km/s^2 acceleration uncertainty
        initial_pos_std: float = 3.0, # km, initial position uncertainty
        initial_vel_std: float = 0.01,       # km/s — initial velocity uncertainty
    ):
        self.process_noise_std = process_noise_std
        self.initial_pos_std   = initial_pos_std
        self.initial_vel_std   = initial_vel_std

 #---Public API ------------------------------------------------------------------------------------
    def initialize(
            self,
            norad_id: int,
            position: np.ndarray,
            velocity:np.ndarray,
            epoch: datetime,
    ) -> KalmanState:
        x = np.concatenate([position, velocity])

        P = np.diag([
            self.initial_pos_std**2,   # x variance
            self.initial_pos_std**2,   # y variance
            self.initial_pos_std**2,   # z variance
            self.initial_vel_std**2,   # vx variance
            self.initial_vel_std**2,   # vy variance
            self.initial_vel_std**2,   # vz variance
        ])

        return KalmanState(
            norad_id=norad_id,
            epoch=epoch,
            x=x.copy(),
            P=P.copy(),
            n_updates=0,
        )

    def predict(
            self,
            state: KalmanState,
            t_new: datetime,
    ) -> KalmanState:
        """
        Predict step — propagate state and covariance to t_new.

        Uses two-body + J2 dynamics for propagation.
        Computes Jacobian F numerically via finite differences.
        Adds process noise Q to account for unmodeled forces
        (atmospheric drag, solar radiation pressure, etc.)

        The covariance P grows during prediction — we become less
        certain about position as time passes without measurements.
        """
        dt = (t_new - state.epoch).total_seconds()
        if abs(dt) < 1e-6:
            return state

#-------------Propagate state w/ RK4-----------------------------------------

        x_new = _rk4_propagate(state.x, dt)

#-----------------Compute Jacobian F=∂f/∂x via finite differences------------------
        F = _numerical_jacobian(state.x, dt)

#-------------------- Process Noise (Q)-----------------
        # Model: constant acceleration noise σ_a in each direction
        # Q = σ_a² * [dt⁴/4, dt³/2; dt³/2, dt²] (van Loan method simplified)
        q = self.process_noise_std**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4

        Q = np.zeros((6, 6))
        for i in range(3):
                Q[i,   i]   = q * dt4 / 4
                Q[i,   i+3] = q * dt3 / 2
                Q[i+3, i]   = q * dt3 / 2
                Q[i+3, i+3] = q * dt2
# ------------ Propagate covariance ----------------------------------------------
# P⁻ = F P Fᵀ + Q
        P_new = F @ state.P @ F.T + Q

        P_new = _symmetrize(P_new)

        return KalmanState(
            norad_id=state.norad_id,
            epoch=t_new,
            x=x_new,
            P=P_new,
            n_updates=state.n_updates,
        )

    def update(
            self,
            state: KalmanState,
            observation: RadarObservation,
            radar_position_ecef: np.ndarray,
    )-> KalmanState:
        # -- Predicted measurement -------------------------------------------------
        z_pred = _state_to_measurement(state.x, radar_position_ecef)

        # ── Actual measurement vector ─────────────────────────────────────────
        z = np.array([
            observation.range_km,
            observation.azimuth_rad,
            observation.elevation_rad,
        ])

        # ── Residual (how surprised are we?) ───────────────────────────────
        residual = z - z_pred

        # Wrap azimuth innovation to [-π, π]
        residual[1] = (residual[1] + np.pi) % (2 * np.pi) - np.pi

        # ── Observation Jacobian H = ∂h/∂x ───────────────────────────────────
        H = _observation_jacobian(state.x, radar_position_ecef)

        # ── Measurement noise covariance R ────────────────────────────────────
        R = np.diag(observation.noise_std**2)

        # ── Innovation covariance S ───────────────────────────────────────────
        S = H @ state.P @ H.T + R

        # ── Kalman Gain K ─────────────────────────────────────────────────────
        # K = P Hᵀ S⁻¹
        # High K → trust measurement; Low K → trust prediction
        try:
            K = state.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular S matrix, skipping update for NORAD %d", state.norad_id)
            return state

        x_new = state.x + K @ residual

        I_KH = np.eye(6) - K @ H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T
        P_new = _symmetrize(P_new)

        logger.debug(
            "NORAD %d update residual=(%.4f, %.4f, %.4f) σ_pos=(%.3f, %.3f, %.3f) km",
            state.norad_id,
            *residual,
            *np.sqrt(np.diag(state.P[:3, :3])),
        )
        return KalmanState(
            norad_id=state.norad_id,
            epoch=state.epoch,
            x=x_new,
            P=P_new,
            n_updates=state.n_updates + 1,
        )

    def process_trajectory(
            self,
            trajectory: Trajectory,
            n_observations: int = 10,
            radar_position: Optional[np.ndarray] = None,
    ) -> KalmanState:
        """
        Run full predict-update cycle over a trajectory.

        Simulates radar observations at evenly spaced intervals
        along the trajectory, then runs predict+update at each.

        In a real system the observations would come from actual
        radar data. Here we simulate them by adding Gaussian noise
        to the true SGP4 positions.

        Parameters
        ----------
        trajectory     : SGP4 Trajectory from Step 2
        n_observations : number of simulated radar observations
        radar_position : ground station in ECEF km (default: equatorial)

        Returns the final KalmanState with refined covariance.
        """
        if radar_position is None:
            radar_position = np.array([R_EARTH, 0.0, 0.0])  # Fix: was np.ndarray()
        states = trajectory.states  # Fix: was wrongly inside the if block

        if len(states) < 2:
            logger.warning("Trajectory too short for Kalman processing")
            sv = states[0]
            return self.initialize(trajectory.norad_id, sv.position, sv.velocity, sv.epoch)

        sv0 = states[0]
        kalman_state = self.initialize(
            trajectory.norad_id, sv0.position, sv0.velocity, sv0.epoch
        )

        indices = np.linspace(0, len(states) - 1, n_observations, dtype=int)
        obs_states = [states[i] for i in indices]

        logger.info(
            "Processing NORAD %d: %d observations over %d states",
            trajectory.norad_id, n_observations, len(states)
        )

        for sv in obs_states:
            kalman_state = self.predict(kalman_state, sv.epoch)
 # ----------------Simulate noisy radar observation -------------------------------------
            observation = _simulate_observation(sv, radar_position)
#------------------- ── Update with observation ───────────────────────────────────────
            kalman_state = self.update(kalman_state, observation, radar_position)

        logger.info(
            "NORAD %d final: σ_pos=(%.3f, %.3f, %.3f) km after %d updates",
            trajectory.norad_id,
            *kalman_state.sigma_position,
            kalman_state.n_updates,
        )

        return kalman_state

    def process_batch(
            self,
            trajectories: dict[int, Trajectory],
            n_observation: int = 10,
    ) -> dict[int, KalmanState]:  # Fix: was dict() should be dict[]
        results = {}
        for norad_id, traj in trajectories.items():
            try:
                results[norad_id] = self.process_trajectory(traj, n_observation)  # Fix: matched param name
            except Exception as e:
                logger.warning("Kalman failed for NORAD %d: %s", norad_id, e)
        logger.info(
            "Kalman batch complete: %d / %d satellites processed",
            len(results), len(trajectories)
        )
        return results

#---------Orbital Dynamics---------------------------------

def _dynamics(x: np.ndarray) -> np.ndarray:
    """Two-body + J2 acceleration. Returns ẋ = [v, a] for state x = [r, v]."""
    r_vec = x[:3]
    v_vec = x[3:]
    r = np.linalg.norm(r_vec)

    # Two-body
    a = -MU / r**3 * r_vec

    # J2 perturbation
    rx, ry, rz = r_vec
    f = -1.5 * J2 * MU * RE_EQ**2 / r**5
    a_j2 = np.array([
        f * rx * (1 - 5 * (rz / r)**2),
        f * ry * (1 - 5 * (rz / r)**2),
        f * rz * (3 - 5 * (rz / r)**2),
    ])

    return np.concatenate([v_vec, a + a_j2])


def _rk4_propagate(x: np.ndarray, dt: float) -> np.ndarray:
    """Runge-Kutta 4th-order integrator over timestep dt (seconds)."""
    k1 = _dynamics(x)
    k2 = _dynamics(x + 0.5 * dt * k1)
    k3 = _dynamics(x + 0.5 * dt * k2)
    k4 = _dynamics(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _numerical_jacobian(x: np.ndarray, dt: float, eps: float = 1e-3) -> np.ndarray:
    """
    6×6 state transition matrix F = ∂f/∂x via central finite differences.
    eps: perturbation step (km for position, km/s for velocity).
    """
    n = len(x)
    F = np.zeros((n, n))
    for i in range(n):
        x_plus  = x.copy(); x_plus[i]  += eps
        x_minus = x.copy(); x_minus[i] -= eps
        F[:, i] = (_rk4_propagate(x_plus, dt) - _rk4_propagate(x_minus, dt)) / (2 * eps)
    return F


def _symmetrize(P: np.ndarray) -> np.ndarray:
    """Force covariance matrix to stay symmetric after numerical drift."""
    return (P + P.T) / 2.0


# ── Radar measurement model ────────────────────────────────────────────────

def _state_to_measurement(x: np.ndarray, radar_pos: np.ndarray) -> np.ndarray:
    """
    Convert ECI state to radar observables [range_km, azimuth_rad, elevation_rad].
    Builds a local East-North-Up frame at the radar site to compute az/el.
    
    I.   Up = unit vector pointing away from Earth at radar site
    II.  East-North-Up frame (handle radar near poles)
    """
    delta = x[:3] - radar_pos
    rho = np.linalg.norm(delta)

   
    up = radar_pos / np.linalg.norm(radar_pos)

    
    north_ref = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    east  = np.cross(north_ref, up);  east  /= np.linalg.norm(east)
    north = np.cross(up, east)

    delta_hat = delta / rho
    e_comp = np.dot(delta_hat, east)
    n_comp = np.dot(delta_hat, north)
    u_comp = np.dot(delta_hat, up)

    elevation = np.arcsin(np.clip(u_comp, -1.0, 1.0))
    azimuth   = np.arctan2(e_comp, n_comp)

    return np.array([rho, azimuth, elevation])


def _observation_jacobian(x: np.ndarray, radar_pos: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    3×6 observation Jacobian H = ∂h/∂x via central finite differences.
    Azimuth differences are wrapped to [-π, π] to avoid wrap-around errors.
    """
    H = np.zeros((3, 6))
    for i in range(6):
        x_plus  = x.copy(); x_plus[i]  += eps
        x_minus = x.copy(); x_minus[i] -= eps
        dz = _state_to_measurement(x_plus, radar_pos) - _state_to_measurement(x_minus, radar_pos)
        dz[1] = (dz[1] + np.pi) % (2 * np.pi) - np.pi   # wrap azimuth diff
        H[:, i] = dz / (2 * eps)
    return H


def _simulate_observation(sv: StateVector, radar_pos: np.ndarray) -> RadarObservation:
    """
    Simulates a noisy radar observation from a true SGP4 StateVector.
    Adds zero-mean Gaussian noise to the true measurement.
    """
    noise_std = np.array([0.1, 0.001, 0.001])   # 100 m range, 1 mrad az/el
    x = np.concatenate([sv.position, sv.velocity])
    z_true  = _state_to_measurement(x, radar_pos)
    z_noisy = z_true + np.random.randn(3) * noise_std

    return RadarObservation(
        epoch=sv.epoch,
        range_km=float(z_noisy[0]),
        azimuth_rad=float(z_noisy[1]),
        elevation_rad=float(z_noisy[2]),
        noise_std=noise_std,
    )

"""
    Two-body + J2 equations of motion.
 
    Returns dx/dt = [velocity, acceleration] given state x = [pos, vel].
 
    J2 accounts for Earth's oblateness — the most significant
    perturbation for LEO satellites (~10× larger than drag).

    
"""

def _two_body_j2(x: np.ndarray) -> np.ndarray:
    r_vec = x[:3]
    v_vec = x[3:]
    r = np.linalg.norm(r_vec)
    r2 = r**2

    a_2body = -MU / r**3 * r_vec

    z2_r2 = (r_vec[2]/r)**2
    factor = 1.5 * J2 * MU * RE_EQ**2 / r**5

    a_j2 = factor * np.array([
        r_vec[0] * (5* z2_r2 - 1),
        r_vec[1] * (5 * z2_r2 - 1),
        r_vec[2] * (5 * z2_r2 - 3),
    ])

    a_total = a_2body + a_j2
    return np.concatenate([v_vec, a_total])

def _rk4_propagate(x: np.ndarray, dt: float) -> np.ndarray:
    k1 = _two_body_j2(x)
    k2 = _two_body_j2(x + 0.5 * dt * k1)
    k3 = _two_body_j2(x + 0.5 * dt * k2)
    k4 = _two_body_j2(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _numerical_jacobian(x: np.ndarray, dt: float, eps: float = 1.0) -> np.ndarray:
    n = len(x)
    F = np.zeros((n, n))
    x_prop = _rk4_propagate(x, dt)
 
    for i in range(n):
        x_pert = x.copy()
        x_pert[i] += eps
        x_prop_pert = _rk4_propagate(x_pert, dt)
        F[:, i] = (x_prop_pert - x_prop) / eps
 
    return F

# ------ Observation Model ------------------------------------------------------
def _state_to_measurement(
    x: np.ndarray,
    radar_pos: np.ndarray,
) -> np.ndarray:
    sat_pos = x[:3]
    rho = sat_pos - radar_pos
    rho_mag = np.linalg.norm(rho)

    radar_hat = radar_pos / np.linalg.norm(radar_pos)
    el = np.arcsin(np.dot(rho / rho_mag, radar_hat))

    az = np.arctan2(rho[1], rho[0])
 
    return np.array([rho_mag, az, el])

"""
    This next line of code:

    Computes observation Jacobian H = ∂h/∂x numerically.
 
    3×6 matrix mapping state perturbations to measurement perturbations.
    Only position components matter (velocity doesn't affect range/az/el).
    """

def _observation_jacobian(
    x: np.ndarray,
    radar_pos: np.ndarray,
    eps: float = 0.01,
) -> np.ndarray:
    h0 = _state_to_measurement(x, radar_pos)
    H = np.zeros((3, 6))
 
    for i in range(3):   # only perturb position components
        x_pert = x.copy()
        x_pert[i] += eps
        h_pert = _state_to_measurement(x_pert, radar_pos)
        H[:, i] = (h_pert - h0) / eps
 
    return H

"""
    This next line of code:

    Simulates a noisy radar observation of a satellite.
 
    Takes the true SGP4 position, converts to range/az/el,
    then adds Gaussian noise to simulate radar measurement error.
 
    Noise levels:
        Range:     100m (1-sigma) — typical tracking radar
        Azimuth:   0.06° = 0.001 rad
        Elevation: 0.06° = 0.001 rad
    """

def _simulate_observation(
    sv: StateVector,
    radar_pos: np.ndarray,
) -> RadarObservation:
    noise_std = np.array([0.1, 0.001, 0.001])
    x = np.concatenate([sv.position, sv.velocity])
    z_true = _state_to_measurement(x, radar_pos)
 
    # (Gaussian Noise):
    noise = np.random.randn(3) * noise_std
    z_noisy = z_true + noise
 
    return RadarObservation(
        epoch=sv.epoch,
        range_km=float(z_noisy[0]),
        azimuth_rad=float(z_noisy[1]),
        elevation_rad=float(z_noisy[2]),
        noise_std=noise_std,
    )

#-------------Utilities--------------------------------------------------

"""Force matrix to be exactly symmetric — prevents numerical drift."""
"""
    Extract 3×3 position covariance matrices from Kalman states.
    Ready to pass directly into PcCalculator.compute().
 
    Returns dict mapping norad_id → 3×3 covariance (km²).
    """

def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)

def kalman_covariances_to_eci(
    kalman_states: dict[int, KalmanState],
) -> dict[int, np.ndarray]:
    return {
        norad_id: state.position_covariance
        for norad_id, state in kalman_states.items()
    }
 