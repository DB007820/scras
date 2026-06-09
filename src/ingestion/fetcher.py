"""
This file does the following function:

Fetches TLE data from:
    - CelesTrak
    - Space-Track
"""

import re
import logging
import requests
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from src.models import TLERecord

logger = logging.getLogger(__name__)

CELESTRAK_CATALOGS = {
    "active":           "active",
    "stations":         "stations",
    "starlink":         "starlink",
    "oneweb":           "oneweb",
    "iridium":          "iridium",
    "weather":          "weather",
    "debris_iridium33":  "2009-005",
    "debris_cosmos2251": "2009-074",
    "debris_fengyun1c":  "1999-025",
}

CELESTRAK_BASE = "https://celestrak.org/SOCRATES/query.php"
CELESTRAK_TLE_BASE = "https://celestrak.org/SATCAT/TLE.php"
CELESTRAK_GP_BASE = "https://celestrak.org/SATCAT/TLE.php"

CELESTRAK_GP_URL = "https://celestrak.org/SATCAT/TLE.php?CATNR={norad_id}&FORMAT=TLE"
CELESTRAK_CATALOG_URL = "https://celestrak.org/SATCAT/TLE.php?GROUP={group}&FORMAT=TLE"


class CelesTrakFetcher:

    MIN_REQUEST_INTERVAL_S = 2.0

    def __init__(self, cache_dir: Optional[Path] = None):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "SatelliteCollisionRiskTool/1.0 (research; contact@example.com)"
        )
        self._last_request_time = 0.0
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_by_norad_id(self, norad_id: int) -> Optional[TLERecord]:
        url = CELESTRAK_GP_URL.format(norad_id=norad_id)
        raw = self._get(url)
        print(f"DEBUG raw response: {repr(raw[:200]) if raw else 'None'}")
        if not raw:
            return None
        records = parse_tle_text(raw, source="celestrak")
        return records[0] if records else None

    def fetch_catalog(self, catalog: str = "active") -> list[TLERecord]:
        group = CELESTRAK_CATALOGS.get(catalog, catalog)
        url = "https://celestrak.org/pub/TLE/catalog.txt"
        raw = self._get(url, cache_key=f"catalog_{group}")
        print(f"DEBUG catalog raw: {repr(raw[:200]) if raw else 'None'}")
        if not raw:
            return []
        records = parse_tle_text(raw, source=f"celestrak/{group}")
        return records

    def fetch_multiple(self, norad_ids: list[int]) -> list[TLERecord]:
        results = []
        for nid in norad_ids:
            rec = self.fetch_by_norad_id(nid)
            if rec:
                results.append(rec)
            else:
                logger.warning("NORAD ID %d not found on CelesTrak", nid)
        return results

    def _get(self, url: str, cache_key: Optional[str] = None) -> Optional[str]:
        if cache_key and self._cache_dir:
            cached = self._load_cache(cache_key)
            if cached:
                logger.debug("Cache hit for '%s'", cache_key)
                return cached
        self._rate_limit()
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text
            if cache_key and self._cache_dir:
                self._save_cache(cache_key, text)
            return text
        except requests.RequestException as e:
            logger.error("CelesTrak request failed: %s — URL: %s", e, url)
            return None

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL_S:
            time.sleep(self.MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request_time = time.time()

    def _load_cache(self, key: str) -> Optional[str]:
        path = self._cache_dir / f"{key}.tle"
        try:
            if path.exists():
                return path.read_text()
        except OSError as e:
            logger.warning("Failed to read cache for '%s': %s", key, e)
        return None

    def _save_cache(self, key: str, content: str):
        path = self._cache_dir / f"{key}.tle"
        try:
            path.write_text(content)
        except OSError as e:
            logger.warning("Failed to write cache for '%s': %s", key, e)


class SpaceTrackFetcher:
    """
    Fetch TLE and CDM data from Space-Track.org.
    Requires a free account: https://www.space-track.org/auth/createAccount
    """

    BASE_URL = "https://www.space-track.org"
    LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
    TLE_URL   = f"{BASE_URL}/basicspacedata/query/class/gp/NORAD_CAT_ID/{{ids}}/format/tle/orderby/EPOCH%20desc/limit/1/emptyresult/show"
    CDM_URL   = f"{BASE_URL}/basicspacedata/query/class/cdm_public/NORAD_CAT_ID1/{{norad_id}}/format/json/orderby/TCA%20asc"

    def __init__(self, username: str, password: str, cache_dir: Optional[Path] = None):
        self._credentials = {"identity": username, "password": password}
        self._session = requests.Session()
        self._authenticated = False
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def authenticate(self) -> bool:
        try:
            resp = self._session.post(self.LOGIN_URL, data=self._credentials, timeout=30)
            resp.raise_for_status()
            self._authenticated = True
            logger.info("Space-Track authentication successful")
            return True
        except requests.RequestException as e:
            logger.error("Space-Track authentication failed: %s", e)
            return False

    def fetch_by_norad_ids(self, norad_ids: list[int]) -> list[TLERecord]:
        """Fetch latest TLEs for a list of NORAD IDs (efficient batch query)."""
        self._ensure_auth()
        ids_str = ",".join(str(n) for n in norad_ids)
        url = self.TLE_URL.format(ids=ids_str)
        try:
            resp = self._session.get(url, timeout=60)
            resp.raise_for_status()
            records = parse_tle_text(resp.text, source="space-track")
            logger.info("Fetched %d TLEs from Space-Track", len(records))
            return records
        except requests.RequestException as e:
            logger.error("Space-Track TLE fetch failed: %s", e)
            return []

    def fetch_active(self, limit: int = 1000) -> list[TLERecord]:
        """Fetch the most recent TLEs for active objects."""
        self._ensure_auth()
        url = (
            f"{self.BASE_URL}/basicspacedata/query/class/gp"
            f"/EPOCH/>now-30"
            f"/orderby/NORAD_CAT_ID"
            f"/limit/{limit}"
            f"/format/tle"
            f"/emptyresult/show"
        )
        try:
            resp = self._session.get(url, timeout=60)
            resp.raise_for_status()
            records = parse_tle_text(resp.text, source="space-track")
            logger.info("Fetched %d TLEs from Space-Track", len(records))
            return records
        except requests.RequestException as e:
            logger.error("Space-Track fetch failed: %s", e)
            return []

    def fetch_cdm(self, norad_id: int) -> list[dict]:
        """
        Fetch Conjunction Data Messages for a satellite.
        Returns raw JSON list — CDM parsing will be a later module.
        """
        self._ensure_auth()
        url = self.CDM_URL.format(norad_id=norad_id)
        try:
            resp = self._session.get(url, timeout=60)
            resp.raise_for_status()
            cdms = resp.json()
            logger.info("Fetched %d CDMs for NORAD ID %d", len(cdms), norad_id)
            return cdms
        except requests.RequestException as e:
            logger.error("Space-Track CDM fetch failed for %d: %s", norad_id, e)
            return []
        except ValueError as e:
            logger.error("Space-Track CDM response is not valid JSON for NORAD ID %d: %s", norad_id, e)
            return []

    def _ensure_auth(self):
        if not self._authenticated:
            success = self.authenticate()
            if not success:
                raise RuntimeError("Space-Track authentication failed. Check credentials.")


def parse_tle_text(raw_text: str, source: str = "unknown") -> list[TLERecord]:
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    records = []
    i = 0

    while i < len(lines):
        if lines[i].startswith("1 ") and len(lines[i]) == 69:
            name = "UNKNOWN"
            line1, line2 = lines[i], lines[i + 1] if i + 1 < len(lines) else ""
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 "):
            name = lines[i]
            line1, line2 = lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue

        record = _build_tle_record(name, line1, line2, source)
        if record:
            records.append(record)

    return records


def _build_tle_record(name: str, line1: str, line2: str, source: str) -> Optional[TLERecord]:
    try:
        if not _validate_tle_checksum(line1) or not _validate_tle_checksum(line2):
            logger.warning("TLE checksum failed for '%s' — skipping", name)
            return None

        norad_id = int(line1[2:7].strip())
        epoch = _parse_tle_epoch(line1[18:32].strip())

        return TLERecord(
            norad_id=norad_id,
            name=name.strip(),
            line1=line1,
            line2=line2,
            epoch=epoch,
            source=source,
        )
    except (ValueError, IndexError) as e:
        logger.warning("Failed to parse TLE for '%s': %s", name, e)
        return None


def _validate_tle_checksum(line: str) -> bool:
    if len(line) != 69:
        return False
    total = 0
    for ch in line[:-1]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return (total % 10) == int(line[-1])


def _parse_tle_epoch(epoch_str: str) -> datetime:
    year_2d = int(epoch_str[:2])
    year = 1900 + year_2d if year_2d >= 57 else 2000 + year_2d

    day_of_year = float(epoch_str[2:])
    day_int = int(day_of_year)
    frac_day = day_of_year - day_int

    from datetime import timedelta
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int - 1)
    epoch = base + timedelta(days=frac_day)
    return epoch


def load_tle_file(path: Path, source: str = "file") -> list[TLERecord]:
    raw = Path(path).read_text()
    records = parse_tle_text(raw, source=source)
    logger.info("Loaded %d TLEs from '%s'", len(records), path)
    return records