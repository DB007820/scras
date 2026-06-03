import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.models import TLERecord, StateVector, Trajectory
from src.ingestion.fetcher import (
    parse_tle_text,
    _validate_tle_checksum,
    _parse_tle_epoch,
    load_tle_file,
)
from src.propagation.propagator import (
    SGP4Propagator,
    _datetime_to_jd,
    _compute_gmst,
    _teme_to_eci,
)


# ─────────────────────────────────────────────────────────────
# Test fixtures — real TLEs for well-known objects
# ─────────────────────────────────────────────────────────────

# ISS (ZARYA) — NORAD 25544
# Checksums verified programmatically
ISS_TLE_TEXT = """\
ISS (ZARYA)
1 25544U 98067A   23334.42825231  .00019288  00000+0  34645-3 0  9994
2 25544  51.6402  37.8432 0000715 107.4985 252.6274 15.50166148428979
"""

# Hubble Space Telescope — NORAD 20580
HST_TLE_TEXT = """\
HST
1 20580U 90037B   23334.42825231  .00001234  00000+0  12345-4 0  9994
2 20580  28.4700  95.5712 0002450 320.3453  39.7198 15.09281436123459
"""

# Intentionally broken TLE (bad checksum)
BAD_CHECKSUM_TLE = """\
BAD SAT
1 99999U 24001A   24001.50000000  .00000000  00000-0  00000-0 0  9009
2 99999  51.6000 000.0000 0000000   0.0000 000.0000 15.00000000 00009
"""


@pytest.fixture
def iss_tle() -> TLERecord:
    records = parse_tle_text(ISS_TLE_TEXT)
    assert records, "ISS TLE failed to parse"
    return records[0]


@pytest.fixture
def propagator() -> SGP4Propagator:
    return SGP4Propagator(frame="teme")


@pytest.fixture
def t_start() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def t_end() -> datetime:
    return datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)  # 1 hour


# ─────────────────────────────────────────────────────────────
# TLE Parsing Tests
# ─────────────────────────────────────────────────────────────

class TestTLEParsing:

    def test_parse_valid_3line_tle(self):
        records = parse_tle_text(ISS_TLE_TEXT)
        assert len(records) == 1
        tle = records[0]
        assert tle.norad_id == 25544
        assert tle.name == "ISS (ZARYA)"
        assert tle.line1.startswith("1 25544")
        assert tle.line2.startswith("2 25544")

    def test_parse_multiple_tles(self):
        combined = ISS_TLE_TEXT + "\n" + HST_TLE_TEXT
        records = parse_tle_text(combined)
        assert len(records) == 2
        norad_ids = {r.norad_id for r in records}
        assert 25544 in norad_ids
        assert 20580 in norad_ids

    def test_bad_checksum_skipped(self):
        records = parse_tle_text(BAD_CHECKSUM_TLE)
        assert len(records) == 0, "Bad checksum TLE should be rejected"

    def test_epoch_parsing(self):
        records = parse_tle_text(ISS_TLE_TEXT)
        tle = records[0]
        assert tle.epoch.year == 2023
        assert tle.epoch.tzinfo is not None   # Must be UTC-aware

    def test_epoch_year_rollover_pre_2000(self):
        # Year 98 → 1998
        epoch = _parse_tle_epoch("98001.00000000")
        assert epoch.year == 1998

    def test_epoch_year_rollover_post_2000(self):
        # Year 24 → 2024
        epoch = _parse_tle_epoch("24001.50000000")
        assert epoch.year == 2024
        assert epoch.month == 1
        assert epoch.day == 1

    def test_epoch_fractional_day(self):
        # Day 1.5 = noon on Jan 1
        epoch = _parse_tle_epoch("24001.50000000")
        assert epoch.hour == 12
        assert epoch.minute == 0

    def test_source_tag_preserved(self):
        records = parse_tle_text(ISS_TLE_TEXT, source="celestrak/active")
        assert records[0].source == "celestrak/active"


class TestChecksumValidation:

    def test_valid_checksum_line1(self):
        line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009"
        assert _validate_tle_checksum(line1)

    def test_valid_checksum_line2(self):
        line2 = "2 25544  51.6400 208.9163 0006317  86.9957 273.1802 15.49309402433165"
        assert _validate_tle_checksum(line2)

    def test_invalid_checksum(self):
        line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002"
        assert not _validate_tle_checksum(line1)

    def test_wrong_length_fails(self):
        assert not _validate_tle_checksum("1 25544 short line")


# ─────────────────────────────────────────────────────────────
# Julian Date Tests
# ─────────────────────────────────────────────────────────────

class TestJulianDate:

    def test_j2000_epoch(self):
        """J2000 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0"""
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd, fr = _datetime_to_jd(dt)
        assert abs((jd + fr) - 2451545.0) < 1e-6

    def test_known_date(self):
        """2024-01-01 00:00:00 UTC"""
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd, fr = _datetime_to_jd(dt)
        # JD for 2024-01-01 00:00 UTC = 2460310.5
        assert abs((jd + fr) - 2460310.5) < 1e-5

    def test_naive_datetime_treated_as_utc(self):
        dt_naive = datetime(2024, 1, 1, 12, 0, 0)
        dt_utc   = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd_n, fr_n = _datetime_to_jd(dt_naive)
        jd_u, fr_u = _datetime_to_jd(dt_utc)
        assert abs((jd_n + fr_n) - (jd_u + fr_u)) < 1e-10


# ─────────────────────────────────────────────────────────────
# SGP4 Propagation Tests
# ─────────────────────────────────────────────────────────────

class TestSGP4Propagator:

    def test_propagate_returns_trajectory(self, propagator, iss_tle, t_start, t_end):
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        assert traj is not None
        assert isinstance(traj, Trajectory)

    def test_trajectory_has_correct_norad_id(self, propagator, iss_tle, t_start, t_end):
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        assert traj.norad_id == 25544

    def test_trajectory_step_count(self, propagator, iss_tle, t_start, t_end):
        """1 hour at 60s steps = 61 states (inclusive endpoints)"""
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        assert len(traj.states) == 61

    def test_state_vector_fields(self, propagator, iss_tle, t_start, t_end):
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        sv = traj.states[0]
        assert sv.position.shape == (3,)
        assert sv.velocity.shape == (3,)
        assert isinstance(sv.epoch, datetime)
        assert sv.norad_id == 25544

    def test_iss_altitude_plausible(self, propagator, iss_tle, t_start, t_end):
        """ISS orbits at ~400 km altitude."""
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        altitudes = [sv.altitude for sv in traj.states]
        assert all(350 < alt < 450 for alt in altitudes), (
            f"ISS altitude out of range: min={min(altitudes):.1f}, max={max(altitudes):.1f}"
        )

    def test_iss_orbital_speed_plausible(self, propagator, iss_tle, t_start, t_end):
        """ISS orbital speed ~7.66 km/s."""
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        speeds = [sv.speed for sv in traj.states]
        assert all(7.0 < spd < 8.5 for spd in speeds), (
            f"ISS speed out of range: min={min(speeds):.3f}, max={max(speeds):.3f}"
        )

    def test_positions_array_shape(self, propagator, iss_tle, t_start, t_end):
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        pos = traj.positions()
        assert pos.shape == (61, 3)

    def test_state_at_single_epoch(self, propagator, iss_tle, t_start):
        sv = propagator.state_at(iss_tle, t_start)
        assert sv is not None
        assert sv.epoch == t_start

    def test_batch_propagation(self, propagator, t_start, t_end):
        records = parse_tle_text(ISS_TLE_TEXT + "\n" + HST_TLE_TEXT)
        results = propagator.propagate_batch(records, t_start, t_end, dt_seconds=300)
        assert len(results) == 2
        assert 25544 in results
        assert 20580 in results

    def test_propagation_energy_conservation(self, propagator, iss_tle, t_start, t_end):
        """
        Orbital energy should remain approximately constant over 1 orbit.
        E = 0.5*v² - μ/r  (vis-viva)
        SGP4 includes drag so energy should slightly decrease, not increase.
        """
        MU_EARTH = 398600.4418  # km³/s²
        traj = propagator.propagate(iss_tle, t_start, t_end, dt_seconds=60)
        energies = []
        for sv in traj.states:
            r = np.linalg.norm(sv.position)
            v = np.linalg.norm(sv.velocity)
            E = 0.5 * v**2 - MU_EARTH / r
            energies.append(E)

        # Energy should not increase (drag dissipates energy)
        assert energies[0] >= energies[-1] - 0.1, (
            "Orbital energy increased unexpectedly — propagation error?"
        )
        # Energy variation should be small (<1%)
        energy_variation = (max(energies) - min(energies)) / abs(np.mean(energies))
        assert energy_variation < 0.01


# ─────────────────────────────────────────────────────────────
# Frame conversion test
# ─────────────────────────────────────────────────────────────

class TestFrameConversion:

    def test_teme_to_eci_preserves_magnitude(self):
        """Pure rotation should not change vector magnitude."""
        r_teme = np.array([7000.0, 0.0, 0.0])
        v_teme = np.array([0.0, 7.5, 0.0])
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        r_eci, v_eci = _teme_to_eci(r_teme, v_teme, epoch)

        assert abs(np.linalg.norm(r_eci) - np.linalg.norm(r_teme)) < 1e-6
        # Velocity magnitude changes slightly due to Earth rotation correction
        # but should be within a few m/s
        assert abs(np.linalg.norm(v_eci) - np.linalg.norm(v_teme)) < 1.0

    def test_eci_propagator_returns_trajectory(self, iss_tle, t_start, t_end):
        prop_eci = SGP4Propagator(frame="eci")
        traj = prop_eci.propagate(iss_tle, t_start, t_end, dt_seconds=300)
        assert traj is not None
        assert len(traj.states) > 0


# ─────────────────────────────────────────────────────────────
# Integration tests — require network access
# ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestCelesTrakFetcher:

    def test_fetch_iss_by_norad_id(self):
        from src.ingestion.fetcher import CelesTrakFetcher
        fetcher = CelesTrakFetcher()
        tle = fetcher.fetch_by_norad_id(25544)
        assert tle is not None
        assert tle.norad_id == 25544

    def test_fetch_active_catalog(self):
        from src.ingestion.fetcher import CelesTrakFetcher
        fetcher = CelesTrakFetcher()
        records = fetcher.fetch_catalog("active")
        assert len(records) > 100   # Active catalog has thousands

    def test_fetch_starlink_catalog(self):
        from src.ingestion.fetcher import CelesTrakFetcher
        fetcher = CelesTrakFetcher()
        records = fetcher.fetch_catalog("starlink")
        assert len(records) > 1000  # Starlink has 3000+ satellites
