"""Shared helpers for the swim-ranking-comp processing pipeline."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WA_BASE_TIMES_CSV = DATA_DIR / "world_aquatics_base_times_2023_2027.csv"

SEX_MAP = {"M": "men", "F": "women"}

# Championship-program events per course. LCM/SCM mirror the World Aquatics
# points tables (SCM adds the 100 IM); SCY is the U.S. short-course program.
_STROKE_50_100_200 = [
    (d, s) for s in ("backstroke", "breaststroke", "butterfly") for d in (50, 100, 200)
]
STANDARD_EVENTS = {
    "LCM": set(
        [(d, "freestyle") for d in (50, 100, 200, 400, 800, 1500)]
        + _STROKE_50_100_200
        + [(200, "individual_medley"), (400, "individual_medley")]
    ),
    "SCM": set(
        [(d, "freestyle") for d in (50, 100, 200, 400, 800, 1500)]
        + _STROKE_50_100_200
        + [(100, "individual_medley"), (200, "individual_medley"), (400, "individual_medley")]
    ),
    "SCY": set(
        [(d, "freestyle") for d in (50, 100, 200, 500, 1000, 1650)]
        + _STROKE_50_100_200
        + [(100, "individual_medley"), (200, "individual_medley"), (400, "individual_medley")]
    ),
}


def load_wa_base_times(path: Path = WA_BASE_TIMES_CSV) -> pd.DataFrame:
    wa = pd.read_csv(path, dtype={"distance": "int64"})
    wa["time_seconds"] = wa["time_seconds"].astype(float)
    return wa


def wa_base_lookup(wa: pd.DataFrame) -> dict:
    """{(sex, course, distance, stroke): [(valid_from, valid_to, base_seconds), ...]}"""
    lookup: dict = {}
    for row in wa.itertuples(index=False):
        key = (row.sex_category, row.pool_course, row.distance, row.stroke)
        lookup.setdefault(key, []).append((row.valid_from, row.valid_to, row.time_seconds))
    for windows in lookup.values():
        windows.sort()
    return lookup


def wa_base_for(lookup: dict, sex: str, course: str, distance, stroke: str, date: str):
    """Base time from the table in force on `date` (ISO string), else None."""
    for valid_from, valid_to, base in lookup.get((sex, course, distance, stroke), ()):
        if valid_from <= date <= valid_to:
            return base
    return None


def season_of(date: str) -> int:
    """Swimming season (Jul 1 - Jun 30), labeled by its starting year."""
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= 7 else year - 1


def implied_base(time_seconds, points):
    """Invert points = 1000*(base/t)^3 to the base time that produced them."""
    if points is None or points <= 0:
        return None
    return time_seconds * (points / 1000.0) ** (1.0 / 3.0)
