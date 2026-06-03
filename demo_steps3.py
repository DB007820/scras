"""
Collision Probability Calculation
Builds on STEP 2
Computes Pc for each event and displays a risk summary table.
"""


import logging
from datetime import datetime, timezone, timedelta
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_step3")
 
RISK_ICONS = {"red": "🔴", "yellow": "🟡", "green": "🟢"}
 
 
def main():
    from src.ingestion.fetcher import CelesTrakFetcher
    from src.propagation.propagator import SGP4Propagator
    from src.conjunction.detector import ConjunctionDetector
    from src.probability.pc_calculator import PcCalculator
 
    # ── Step 2A: Fetch TLEs ────────────────────────────────────────────────────
    logger.info("=== STEP 3A: Fetch + Propagate (from Step 2) ===")
    fetcher = CelesTrakFetcher()
    tles = fetcher.fetch_catalog("stations")
    if not tles:
        logger.error("Failed to fetch TLEs")
        return
    logger.info("Fetched %d TLEs", len(tles))
 
    # ── Step 2B: Propagate ─────────────────────────────────────────────────────
    prop = SGP4Propagator(frame="teme")
    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end   = t_start + timedelta(hours=24)
    trajectories = prop.propagate_batch(tles, t_start, t_end, dt_seconds=60)
 
    # ── Step 2C: Conjunction Screening ────────────────────────────────────────
    logger.info("\n=== STEP 3B: Conjunction Screening ===")
    detector = ConjunctionDetector(threshold_km=5.0)
    events = detector.screen(trajectories)
    logger.info("Found %d conjunction events", len(events))
 
    if not events:
        print("\n  No conjunctions found. Try a larger catalog (e.g. 'active').")
        return
 
    # ── Step 3: Compute Pc ────────────────────────────────────────────────────
    logger.info("\n=== STEP 3C: Collision Probability Calculation ===")
    calc = PcCalculator(r_hbr_km=0.010)   # 10m combined hard-body radius
    results = calc.compute_batch(events)
 
    # ── Display results ───────────────────────────────────────────────────────
    print(f"\n{'─'*95}")
    print(f"  {'Risk':<6} {'Primary':<20} {'Secondary':<20} {'TCA (UTC)':<22} {'Miss (km)':<10} {'Pc':<12} {'σ_x, σ_y (km)'}")
    print(f"{'─'*95}")
 
    for event, result in zip(events, results):
        icon = RISK_ICONS[result.risk_level]
        print(
            f"  {icon:<6} "
            f"{event.primary_name:<20} "
            f"{event.secondary_name:<20} "
            f"{event.tca.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{event.miss_distance:<10.3f} "
            f"{result.pc:<12.3e} "
            f"({result.sigma_x:.3f}, {result.sigma_y:.3f})"
        )
 
    print(f"{'─'*95}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    n_red    = sum(1 for r in results if r.risk_level == "red")
    n_yellow = sum(1 for r in results if r.risk_level == "yellow")
    n_green  = sum(1 for r in results if r.risk_level == "green")
 
    print(f"\n  Risk Summary:")
    print(f"    🔴 Red    (Pc ≥ 1e-4) : {n_red}")
    print(f"    🟡 Yellow (Pc ≥ 1e-5) : {n_yellow}")
    print(f"    🟢 Green  (Pc < 1e-5) : {n_green}")
    print(f"\n  Note: Using conservative default SGP4 covariance.")

 

 
if __name__ == "__main__":
    main()