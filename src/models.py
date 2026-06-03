from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass
class TLERecord:
   # raw 2 line element from Celestrak
   norad_id: int
   name: str
   line1:str
   line2: str
   epoch: datetime
   source: str = "celestrak"

   def __repr__(self):
      return f"TLERecord(norad_id={self.norad_id}, name={self.name!r}, epoch={self.epoch.isoformat()})"
   
@dataclass
class StateVector:
    norad_id: int
    epoch: datetime
    position: np.ndarray
    velocity: np.ndarray
    covariance: Optional[np.ndarray] = None
    source_tle: Optional["TLERecord"] = None

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude(self) -> float:
        R_EARTH_KM = 6371.0
        return float(np.linalg.norm(self.position)) - R_EARTH_KM
   

@dataclass
class Trajectory:
    # Time-series of StateVectors for a single satellite.
    # Used for conjunction detection — do not mix satellites.
    norad_id: int
    name: str
    states: list[StateVector] = field(default_factory=list)
    propagator: str = "sgp4"

    @property
    def t_start(self) -> Optional[datetime]:
        return self.states[0].epoch if self.states else None

    @property
    def t_end(self) -> Optional[datetime]:
        return self.states[-1].epoch if self.states else None

    @property
    def duration_hours(self) -> float:
        if not self.states or len(self.states) < 2:
            return 0.0
        delta = self.states[-1].epoch - self.states[0].epoch
        return delta.total_seconds() / 3600

    def positions(self) -> np.ndarray:
        return np.array([s.position for s in self.states])

    def velocities(self) -> np.ndarray:
        return np.array([s.velocity for s in self.states])   # ← was s.position, wrong

    def epochs(self) -> list[datetime]:
        return [s.epoch for s in self.states]

    def __repr__(self):
        return (
            f"Trajectory(norad_id={self.norad_id}, name={self.name!r}, "
            f"steps={len(self.states)}, "
            f"duration={self.duration_hours:.2f}h)"   # ← closing ) was missing
        )