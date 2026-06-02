"""
**INstructions & REminder for DELL!**

This is: 
End-to-end demo of Phase 2: Conjunction Detection.

Fetches a small set of satellites from CelesTrak, propagates them,
then screens all pairs for conjunctions.

Run with:
    python demo_step2.py
"""

import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_step2")


def main():
    from src.ingestion.fetcher import CelesTrakFetcher
    from src.propagation.propagator import SGP4Propagator
    from src.conjunction.detector import ConjunctionDetector

    # ── 1. Fetch a small catalog ───────────────────────────────────────────────
    logger.info("=== STEP 2A: Fetch TLEs ===")
    fetcher = CelesTrakFetcher()

    # Use ISS debris + stations — small set, known to have interesting orbits
    logger.info("Fetching space stations catalog...")
    tles = fetcher.fetch_catalog("stations")

    if not tles:
        logger.error("Failed to fetch TLEs — check CelesTrak URL")
        return

    logger.info("Fetched %d TLEs", len(tles))

    # ── 2. Propagate 24 hours ─────────────────────────────────────────────────
    logger.info("\n=== STEP 2B: Propagate ===")
    prop = SGP4Propagator(frame="teme")

    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end   = t_start + timedelta(hours=24)

    logger.info("Propagating %d satellites for 24 hours...", len(tles))
    trajectories = prop.propagate_batch(tles, t_start, t_end, dt_seconds=60)
    logger.info("Propagated %d trajectories successfully", len(trajectories))

    # ── 3. Screen for conjunctions ────────────────────────────────────────────
    logger.info("\n=== STEP 2C: Conjunction Screening ===")
    detector = ConjunctionDetector(threshold_km=5.0)
    events = detector.screen(trajectories)

    stats = detector.stats
    print(f"\n  Satellites screened:  {len(trajectories)}")
    print(f"  Total pairs checked:  {stats['total_pairs']}")
    print(f"  Passed altitude filter: {stats['passed_filter']}")
    print(f"  Conjunction events:   {stats['events_found']}")

    # ── 4. Print results ──────────────────────────────────────────────────────
    logger.info("\n=== STEP 2D: Results ===")
    if not events:
        print("\n  No conjunctions found in this catalog/timespan.")
        print("  Try a larger catalog (e.g. 'active') or wider threshold.")
    else:
        print(f"\n  Top {min(10, len(events))} closest approaches:\n")
        print(f"  {'Primary':<20} {'Secondary':<20} {'TCA (UTC)':<22} {'Miss (km)':<12} {'V_rel (km/s)'}")
        print(f"  {'-'*20} {'-'*20} {'-'*22} {'-'*12} {'-'*12}")
        for event in events[:10]:
            print(
                f"  {event.primary_name:<20} "
                f"  {event.secondary_name:<20} "
                f"  {event.tca.strftime('%Y-%m-%d %H:%M:%S'):<22} "
                f"  {event.miss_distance:<12.3f} "
                f"  {event.relative_velocity:.3f}"
            )

    print(f"\n✅ Step 2 complete.")
    print(f"   Ready for Step 3: Collision Probability →")


if __name__ == "__main__":
    main()