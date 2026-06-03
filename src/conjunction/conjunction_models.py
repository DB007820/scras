"""


Detector -> ConjunctionEvent (dataclass) -> PC calculator -> probability.py
                                       |
                                    detector.py
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ConjunctionEvent:
    """
    A detected close approach between two satellites.

    Distances in km, velocities in km/s.

    RTN frame (Radial-Transverse-Normal) centered on the primary:
        Radial     — along position vector (toward/away from Earth)
        Transverse — along velocity vector (along-track)
        Normal     — perpendicular to orbital plane (cross-track)
    """
    primary_id: int
    secondary_id: int
    primary_name: str
    secondary_name: str

    # TCA: (Scalar Distance), ((relative velocity at TCA))
    tca: datetime               
    miss_distance: float        
    relative_velocity: float    # km/s — magnitude of relative velocity at TCA

    # Miss vector decomposed into RTN components
    radial_miss: float          # km
    transverse_miss: float      # km
    normal_miss: float          # km

    screening_threshold_km: float = 5.0
    pc: Optional[float] = None  # Filled by Step 3 (Pc calculator)

    @property
    def is_high_risk(self) -> bool:
        """NASA red line: Pc > 1e-4, or miss distance < 200m if Pc unknown."""
        if self.pc is not None:
            return self.pc > 1e-4
        return self.miss_distance < 0.2

    def __repr__(self):
        return (
            f"ConjunctionEvent("
            f"{self.primary_name} / {self.secondary_name}, "
            f"TCA={self.tca.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"miss={self.miss_distance:.3f} km, "
            f"v_rel={self.relative_velocity:.3f} km/s)"
        )
    