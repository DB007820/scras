"""
demo_step5.py
 
End-to-end demo of the complete 6-step pipeline:
    Step 1 — Ingest TLEs
    Step 2 — Propagate trajectories (SGP4)
    Step 3 — Detect conjunctions
    Step 4 — Calculate Pc (initially with default covariance)
    Step 5 — Kalman filter (refine covariance)
           — Recompute Pc with real covariance
    Step 6 — Visualize (3D orbit + dashboard + timeline)
 
"""

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_step5")
 
 
def main():
    from src.ingestion.fetcher import CelesTrakFetcher
    from src.propagation.propagator import SGP4Propagator
    from src.conjunction.detector import ConjunctionDetector
    from src.probability.pc_calculator import PcCalculator
    from src.kalman.kalman_filter import ExtendedKalmanFilter, kalman_covariances_to_eci
    from src.visualization.visualizer import SatelliteVisualizer
 
    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    logger.info("=== STEP 1: Ingest TLEs ===")
    fetcher = CelesTrakFetcher()
    tles = fetcher.fetch_catalog("stations")
    if not tles:
        logger.error("Failed to fetch TLEs")
        return
    logger.info("Fetched %d TLEs", len(tles))
 
    # ── Step 2: Propagate ─────────────────────────────────────────────────────
    logger.info("\n=== STEP 2: Propagate (SGP4) ===")
    prop = SGP4Propagator(frame="teme")
    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end   = t_start + timedelta(hours=24)
    trajectories = prop.propagate_batch(tles, t_start, t_end, dt_seconds=60)
    logger.info("Propagated %d trajectories", len(trajectories))
 
    # ── Step 3: Detect conjunctions ───────────────────────────────────────────
    logger.info("\n=== STEP 3: Detect Conjunctions ===")
    detector = ConjunctionDetector(threshold_km=5.0)
    events = detector.screen(trajectories)
    logger.info("Found %d conjunction events", len(events))
 
    # ── Step 4: Initial Pc (synthetic covariance) ─────────────────────────────
    logger.info("\n=== STEP 4: Initial Pc (synthetic covariance) ===")
    calc = PcCalculator(r_hbr_km=0.010)
    calc.compute_batch(events)
 
    pc_before = {e.primary_id: e.pc for e in events if e.pc is not None}
    logger.info("Initial Pc computed for %d events", len(pc_before))
 
    # ── Step 5: Kalman Filter ─────────────────────────────────────────────────
    logger.info("\n=== STEP 5: Kalman Filter ===")
    ekf = ExtendedKalmanFilter(
        process_noise_std=0.001,
        initial_pos_std=3.0,
        initial_vel_std=0.01,
    )
 
    logger.info("Processing %d satellites through Kalman filter...", len(trajectories))
    kalman_states = ekf.process_batch(trajectories, n_observations=10)
 
    # Extract refined covariances
    covariances = kalman_covariances_to_eci(kalman_states)
    logger.info("Refined covariances for %d satellites", len(covariances))
 
    # Show covariance improvement for first satellite
    sample_id = next(iter(kalman_states))
    sample_state = kalman_states[sample_id]
    sigmas = sample_state.sigma_position
    print(f"\n  Sample satellite NORAD {sample_id}:")
    print(f"    Before Kalman: σ = 3.000 km (synthetic default)")
    print(f"    After  Kalman: σ = ({sigmas[0]:.3f}, {sigmas[1]:.3f}, {sigmas[2]:.3f}) km")
    print(f"    Observations assimilated: {sample_state.n_updates}")
 
    # ── Recompute Pc with real covariance ─────────────────────────────────────
    logger.info("\n=== STEP 5b: Recompute Pc with Kalman Covariance ===")
 
    # Reset Pc values so we can recompute
    for event in events:
        event.pc = None
 
    results_kalman = calc.compute_batch(events, covariances=covariances)
 
    # ── Compare before/after ──────────────────────────────────────────────────
    print(f"\n  Pc comparison (synthetic vs Kalman covariance):")
    print(f"  {'Primary':<20} {'Secondary':<20} {'Pc (synthetic)':<16} {'Pc (Kalman)':<16} {'Change'}")
    print(f"  {'-'*20} {'-'*20} {'-'*16} {'-'*16} {'-'*10}")
 
    for event, result in zip(events[:8], results_kalman[:8]):
        pc_syn = pc_before.get(event.primary_id, 0)
        pc_kal = result.pc
        if pc_syn and pc_kal:
            ratio = pc_kal / pc_syn if pc_syn > 0 else 0
            direction = "↑" if ratio > 1.1 else "↓" if ratio < 0.9 else "≈"
            print(
                f"  {event.primary_name:<20} {event.secondary_name:<20} "
                f"{pc_syn:<16.3e} {pc_kal:<16.3e} {direction}"
            )
 
    # ── Step 6: Visualize ─────────────────────────────────────────────────────
    logger.info("\n=== STEP 6: Visualization ===")
    viz = SatelliteVisualizer(output_dir="data/visualizations")
 
    path1 = viz.orbit_plot(trajectories, events)
    path2 = viz.risk_dashboard(events)
    path3 = viz.timeline_view(events)
 
    print(f"\n  Visualizations saved:")
    print(f"    {Path(path1).resolve()}")
    print(f"    {Path(path2).resolve()}")
    print(f"    {Path(path3).resolve()}")
 
    print(f"\n✅ Full 6-step pipeline complete.")
    print(f"   Steps: Ingest → Propagate → Detect → Pc → Kalman → Visualize")
 
 
if __name__ == "__main__":
    main()