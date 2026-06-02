# scras
Satellite Collision Risk Analysis System

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
