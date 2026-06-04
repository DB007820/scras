# scras
SCRAS - Satellite Collision Risk Analysis System
A five step end-to-end pipeline for satellite conjunction detection and collision probability estimation, built on open orbital data from CelesTrak and a few astrodynamics references.

Overview:
Low Earth Orbit (LEO) congestion is a growing operational challenge. SCRAS implements a complete collision risk assessment pipeline, from live TLE ingestion through Extended Kalman Filter state estimation and Pc computation using only open data sources and standard Python libraries. The system is designed as a modular research platform. Each step is independently runnable, testable, and extensible.

Theoretical Background:
The pipeline follows the methodology listed below:
1. Orbit Propagation: SGP4/SDP4 analytical propagator (Vallado 2013) applied to Two-Line Element sets. Suitable for LEO objects over short time horizons (days).
2. Conjunction Screening:  Checks every pair of satellites for close approaches by comparing their positions every 60 seconds. An altitude pre-filter removes pairs that are too far apart vertically to ever be a threat, reducing the number of comparisons needed.
3. Collision Probability: Estimates the likelihood two satellites will physically collide during a close approach, assuming both travel in straight lines and their position uncertainties follow a Gaussian distribution. The probability is computed by integrating that uncertainty over the combined collision cross-section (Alfano 2005).
4. State Estimation: Extended Kalman Filter using two-body + J2 dynamics, integrated with RK4. State transition Jacobian computed numerically via finite differences (Tapley et al. 2004).

Flow Chart of Project:
TLE Ingestion → SGP4 Propagation → Conjunction Detection → Collision Probability → EKF Refinement → Visualization

Notes on the data:
TLEs are fetched live from CelesTrak, 
no API key needed for the stations catalog. 
The covariance matrices used in the Pc calculation are simulated (Gaussian noise added to SGP4 states) since real CDM data from Space-Track requires an account.

Background:
This started as a personal project turned class project exploring whether a functional collision risk pipeline could be built on open data. The orbital mechanics follow standard references (Vallado 2013, Alfano 2005, Tapley et al. 2004). The EKF uses two-body + J2 dynamics integrated with RK4, with the Jacobian computed numerically via finite differences.
Still a work in progress, maneuver planning and real CDM integration are the obvious next steps.

References

Vallado, D.A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed.
Alfano, S. (2005). A Numerical Implementation of Spherical Object Collision Probability. Journal of the Astronautical Sciences.
Tapley, B.D., Schutz, B.E. & Born, G.H. (2004). Statistical Orbit Determination.
Kelso, T.S. CelesTrak. https://celestrak.org
