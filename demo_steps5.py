import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_step8")
 
 
def main():
    from src.ingestion.fetcher import CelesTrakFetcher
    from src.propagation.propagator import SGP4Propagator
    from src.conjunction.detector import ConjunctionDetector
    from src.probability.pc_calculator import PcCalculator
    from src.visualization.visualizer import SatelliteVisualizer
 
    # ── Steps 1-3: Full pipeline ──────────────────────────────────────────────
    logger.info("=== Running Steps 1-3 Pipeline ===")
 
    fetcher = CelesTrakFetcher()
    tles = fetcher.fetch_catalog("stations")
    if not tles:
        logger.error("Failed to fetch TLEs")
        return
    logger.info("Fetched %d TLEs", len(tles))
 
    prop = SGP4Propagator(frame="teme")
    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end   = t_start + timedelta(hours=24)
    trajectories = prop.propagate_batch(tles, t_start, t_end, dt_seconds=60)
    logger.info("Propagated %d trajectories", len(trajectories))
 
    detector = ConjunctionDetector(threshold_km=5.0)
    events = detector.screen(trajectories)
    logger.info("Found %d conjunction events", len(events))
 
    calc = PcCalculator(r_hbr_km=0.010)
    results = calc.compute_batch(events)
    logger.info("Computed Pc for %d events", len(results))
 
    # ── Step 8: Visualize ─────────────────────────────────────────────────────
    logger.info("\n=== STEP 8: Generating Visualizations ===")
    viz = SatelliteVisualizer(output_dir="data/visualizations")
 
    # 1. 3D Orbit Plot
    logger.info("Generating 3D orbit plot...")
    path1 = viz.orbit_plot(trajectories, events)
    print(f"\n  ✅ 3D Orbit Plot     → {path1}")
 
    # 2. Risk Dashboard
    logger.info("Generating risk dashboard...")
    path2 = viz.risk_dashboard(events)
    print(f"  ✅ Risk Dashboard    → {path2}")
 
    # 3. Timeline
    logger.info("Generating timeline...")
    path3 = viz.timeline_view(events)
    print(f"  ✅ Timeline View     → {path3}")
 
    print(f"\n  Open these files in your browser:")
    print(f"    {Path(path1).resolve()}")
    print(f"    {Path(path2).resolve()}")
    print(f"    {Path(path3).resolve()}")
    print(f"\n✅ Step 8 complete.")
 
 
if __name__ == "__main__":
    main()