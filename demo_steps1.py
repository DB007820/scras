import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s"
)
logger = logging.getLogger("demo_step1")


def main():
    from src.ingestion.fetcher import CelesTrakFetcher
    from src.propagation.propagator import SGP4Propagator

    cache_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch TLE(s)
    logger.info("===Fetch TLE(s)===")
    fetcher = CelesTrakFetcher(cache_dir=cache_dir)

    logger.info("Fetching ISS by Norad ID")
    iss = fetcher.fetch_by_norad_id(25544)
    print(f"\n Fetched: {iss}")

    logger.info("Fetching Starlink Catalog...")
    starlink = fetcher.fetch_catalog("starlink")[:10]
    print(f" Fetched {len(starlink)} Starlink TLE(s)")
    for tle in starlink[:3]:
        print(f"    {tle}")
    
    # Propagate
    logger.info("\n=== Propagate ===")
    prop = SGP4Propagator(frame="teme")

    t_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t_end = t_start + timedelta(hours=24)

    logger.info("Propagating ISS for 24 hours at 60s steps...")
    iss_traj = prop.propagate(iss, t_start, t_end, dt_seconds=60)
    print(f"\n {iss_traj}")

    sv0 = iss_traj.states[0]
    pos = ", ".join(f"{v:.2f}" for v in sv0.position)
    vel = ", ".join(f"{v:.4f}" for v in sv0.velocity)
    print(f"  Position (km):   [{pos}]")
    print(f"  Velocity (km/s): [{vel}]")
    print(f"  Altitude: {sv0.altitude:.1f} km")
    print(f"  Speed:    {sv0.speed:.4f} km/s")

    logger.info("Batch propagating %d Starlink sats for 24 hours...", len(starlink))
    sl_trajs = prop.propagate_batch(starlink, t_start, t_end, dt_seconds=60)
    print(f"\n  Propagated {len(sl_trajs)} Starlink trajectories")

    # Altitude Stats
    logger.info("\n=== Trajectory Statistics ===")
    import numpy as np

    all_trajs = {iss.norad_id: iss_traj, **sl_trajs}
    for norad_id, traj in list(all_trajs.items())[:5]:
        altitudes = [sv.altitude for sv in traj.states]
        print(
            f"  {traj.name:30s}  "
            f"alt: {min(altitudes):.1f}–{max(altitudes):.1f} km  "
            f"steps: {len(traj.states)}"
        )
    
    # Saving Sample
    logger.info("\n=== Save Sample Trajectory ===")

    iss_data = {
        "norad_id": iss_traj.norad_id,
        "name": iss_traj.name,
        "propagator": iss_traj.propagator,
        "t_start": iss_traj.t_start.isoformat(),
        "t_end": iss_traj.t_end.isoformat(),
        "num_states": len(iss_traj.states),
        "sample_states":[
            {
                "epoch": sv.epoch.isoformat(),
                "position_km": sv.position.tolist(),
                "velocity_km_s": sv.velocity.tolist(),
                "altitude_km": round(sv.altitude, 3),
                "speed_km_s": round(sv.speed, 6),
            }
            for sv in iss_traj.states[::60]
        ],
    }

    out_path = output_dir / "iss_trajectory_sample.json"
    out_path.write_text(json.dumps(iss_data, indent=2))
    logger.info("Saved ISS sample trajectory → %s", out_path)

    print(f"\n✅ Step 1 complete.")
    print(f"   TLEs fetched:    {1 + len(starlink)}")
    print(f"   Trajectories:    {len(all_trajs)}")
    print(f"   Sample output:   {out_path}")
    print(f"\n   Ready for Step 2: Conjunction Detection →")

if __name__ == "__main__":
    main()