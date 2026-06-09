import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_step5")
 
 
def main():
    from src.ingestion.fetcher import SpaceTrackFetcher
    from src.propagation.propagator import SGP4Propagator
    from src.conjunction.detector import ConjunctionDetector
    from src.probability.pc_calculator import PcCalculator
    from src.kalman.kalman_filter import ExtendedKalmanFilter, kalman_covariances_to_eci
    from src.visualization.visualizer import SatelliteVisualizer
 
     # ── Step 1: Ingest ────────────────────────────────────────────────────────
    logger.info("=== STEP 1: Ingest TLEs ===")
    ST_USER = "fadel.longhorn@gmail.com"   # ← your Space-Track credentials
    ST_PASS = "Fadellonghorn123"      # ← your Space-Track credentials
    fetcher = SpaceTrackFetcher(username=ST_USER, password=ST_PASS)
    MAX_SATS = 50
    tles = fetcher.fetch_active(limit=MAX_SATS)
    if not tles:
        logger.error("Failed to fetch TLEs")
        return
    logger.info("Fetched %d TLEs", len(tles))
    # ── Step 2: Propagate ─────────────────────────────────────────────────────
    logger.info("\n=== STEP 2: Propagate (SGP4) ===")
    prop    = SGP4Propagator(frame="teme")
    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end   = t_start + timedelta(hours=24)
    trajectories = prop.propagate_batch(tles, t_start, t_end, dt_seconds=60)
    logger.info("Propagated %d trajectories", len(trajectories))
 
    # ── Step 3: Detect conjunctions ───────────────────────────────────────────
    logger.info("\n=== STEP 3: Detect Conjunctions ===")
    detector = ConjunctionDetector(threshold_km=5.0)
    events   = detector.screen(trajectories)

    stats = detector.stats
    print(f"\n  Satellites:            {len(trajectories)}")
    print(f"  Total pairs checked:   {stats['total_pairs']}")
    print(f"  Passed altitude filter:{stats['passed_altitude_filter']}")
    print(f"  Co-orbiting skipped:   {stats['skipped_co_orbiting']}")
    print(f"  Real conjunction events:{stats['events_found']}")

    if not events:
        print("\n  No conjunctions found. Skipping Steps 4 & 5, proceeding to visualization.")
    else:
        # ── Step 4: Initial Pc (synthetic covariance) ─────────────────────────
        logger.info("\n=== STEP 4: Collision Probability (Pc) ===")
        calc = PcCalculator(r_hbr_km=0.010)
        calc.compute_batch(events)

        # ── Step 5: Extended Kalman Filter ─────────────────────────────────────
        logger.info("\n=== STEP 5: Extended Kalman Filter ===")
        ekf = ExtendedKalmanFilter(
            process_noise_std=0.001,
            initial_pos_std=3.0,
            initial_vel_std=0.01,
        )
        kalman_states = ekf.process_batch(trajectories, n_observation=10)
        covariances   = kalman_covariances_to_eci(kalman_states)

        # Recompute Pc with Kalman covariance
        for event in events:
            event.pc = None
        calc.compute_batch(events, covariances=covariances)

        # ── Print results ─────────────────────────────────────────────────────
        print(f"\n  Top {min(10, len(events))} closest approaches:\n")
        print(f"  {'Risk':<8} {'Primary':<22} {'Secondary':<22} {'Miss (km)':<12} {'Pc':<12} {'TCA (UTC)'}")
        print(f"  {'-'*8} {'-'*22} {'-'*22} {'-'*12} {'-'*12} {'-'*20}")

        icons = {"red": "🔴", "yellow": "🟡", "green": "🟢"}
        for event in events[:10]:
            risk = "red" if event.pc and event.pc >= 1e-4 else "yellow" if event.pc and event.pc >= 1e-5 else "green"
            icon = icons[risk]
            pc_str = f"{event.pc:.3e}" if event.pc else "N/A"
            print(
                f"  {icon} {risk:<6} "
                f"{event.primary_name:<22} "
                f"{event.secondary_name:<22} "
                f"{event.miss_distance:<12.3f} "
                f"{pc_str:<12} "
                f"{event.tca.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    # ── Step 6: Visualize ─────────────────────────────────────────────────────
    logger.info("\n=== STEP 6: Visualization ===")
    viz  = SatelliteVisualizer(output_dir="data/visualizations")
    path = viz.render(trajectories, events, output_file="mission_control.html")

    print(f"\n  Dashboard saved. Open in browser:")
    print(f"  {Path(path).resolve()}\n")
    print("✅ Full 6-step pipeline complete.")
    print("   Ingest → Propagate → Detect → Pc → Kalman → Visualize")


if __name__ == "__main__":
    main()