"""Clean and enrich each recruiting-class snapshot into analysis-ready artifacts.

For every data/swimcloud_YYYY/ this writes data/processed/swimcloud_YYYY/:
  swims_clean.csv        every lifetime-best row, typed + flagged + WA-enriched
  rankings_clean.csv     typed rankings
  validation_report.json row reconciliation, WA recompute match rates, flag inventory

Usage: uv run scripts/process_data.py [--classes 2026 ...]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from swimlib import (
    DATA_DIR,
    PROCESSED_DIR,
    SEX_MAP,
    STANDARD_EVENTS,
    load_wa_base_times,
    season_of,
    wa_base_for,
    wa_base_lookup,
)

BOOL_COLS = ["legal", "exhibition", "is_user_inputted", "is_relay_leadoff", "is_extracted_split"]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_csv_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", None), errors="coerce")


def process_class(class_dir: Path, wa_lookup: dict, out_root: Path) -> dict:
    recruiting_class = int(class_dir.name.rsplit("_", 1)[1])
    out_dir = out_root / class_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    swims = read_csv_raw(class_dir / "lifetime_bests_combined.csv")
    rankings = read_csv_raw(class_dir / "rankings_combined.csv")
    n_in = len(swims)

    for col in BOOL_COLS:
        swims[col] = swims[col] == "true"
    t = to_float(swims["time_seconds"])
    sc_points = to_float(swims["performance_points"])
    wa_points = to_float(swims["world_aquatics_points"])
    swims["time_seconds"] = t
    swims["performance_points"] = sc_points
    swims["world_aquatics_points"] = wa_points
    swims["power_index"] = to_float(swims["power_index"])
    swims["distance"] = to_float(swims["distance"]).astype("Int64")
    swims["age_at_swim"] = to_float(swims["age_at_swim"]).astype("Int64")

    swims["recruiting_class"] = recruiting_class
    valid_date = swims["swim_date"].str.match(DATE_RE)
    swims["season"] = swims["swim_date"].where(valid_date).map(
        lambda d: season_of(d) if isinstance(d, str) else None
    ).astype("Int64")

    swims["is_standard_event"] = [
        course in STANDARD_EVENTS and (dist, stroke) in STANDARD_EVENTS[course]
        for course, dist, stroke in zip(swims["course"], swims["distance"], swims["stroke"])
    ]
    swims["is_clean"] = (
        swims["legal"]
        & ~swims["exhibition"]
        & ~swims["is_user_inputted"]
        & ~swims["is_relay_leadoff"]
        & ~swims["is_extracted_split"]
        & swims["is_standard_event"]
        & t.notna()
        & (t > 0)
        & valid_date
    )

    # World Aquatics enrichment: score against the table in force at the swim date.
    wa_base = pd.Series(
        [
            wa_base_for(wa_lookup, SEX_MAP.get(g), c, d, s, dt)
            for g, c, d, s, dt in zip(
                swims["gender"], swims["course"], swims["distance"], swims["stroke"], swims["swim_date"]
            )
        ],
        index=swims.index,
        dtype=float,
    )
    swims["wa_base_time"] = wa_base
    swims["wa_points_recomputed"] = (1000 * (wa_base / t) ** 3).round(2)
    swims["wa_points_match"] = (
        (swims["wa_points_recomputed"].round() - wa_points).abs() <= 1
    ).where(wa_base.notna() & wa_points.notna())

    pos_wa = wa_points.where(wa_points > 0)
    pos_sc = sc_points.where(sc_points > 0)
    swims["implied_wa_base"] = (t * (pos_wa / 1000) ** (1 / 3)).round(4)
    swims["implied_sc_base"] = (t * (pos_sc / 1000) ** (1 / 3)).round(4)
    swims["quality_factor"] = ((1000 / pos_wa) ** (1 / 3)).round(6)

    swims = swims.sort_values(
        ["gender", "swimmer_id", "course", "stroke", "distance", "swim_id"]
    ).reset_index(drop=True)

    rankings["rank"] = to_float(rankings["rank"]).astype("Int64")
    rankings["power_index"] = to_float(rankings["power_index"])
    rankings["recruiting_class"] = recruiting_class
    rankings = rankings.sort_values(["gender", "rank", "swimmer_id"]).reset_index(drop=True)

    write_csv(swims, out_dir / "swims_clean.csv")
    write_csv(rankings, out_dir / "rankings_clean.csv")

    report = build_report(swims, rankings, recruiting_class, n_in)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool or str(out[col].dtype) == "boolean":
            out[col] = out[col].map({True: "true", False: "false"})
    out.to_csv(path, index=False)


def build_report(swims: pd.DataFrame, rankings: pd.DataFrame, recruiting_class: int, n_in: int) -> dict:
    meters = swims[swims["course"].isin(["LCM", "SCM"]) & swims["world_aquatics_points"].notna()]
    joined = meters[meters["wa_base_time"].notna()]

    def match_rate(df: pd.DataFrame):
        return None if df.empty else round(float(df["wa_points_match"].mean()), 4)

    mismatches = joined[joined["wa_points_match"] == False]  # noqa: E712
    nonstandard = swims[~swims["is_standard_event"]]
    return {
        "recruiting_class": recruiting_class,
        "rows": {
            "lifetime_bests_in": n_in,
            "swims_out": len(swims),
            "rankings": len(rankings),
            "is_clean": int(swims["is_clean"].sum()),
        },
        "flags": {
            "exhibition": int(swims["exhibition"].sum()),
            "user_inputted": int(swims["is_user_inputted"].sum()),
            "relay_leadoff": int(swims["is_relay_leadoff"].sum()),
            "extracted_split": int(swims["is_extracted_split"].sum()),
            "nonstandard_event": len(nonstandard),
            "nonstandard_breakdown": {
                f"{c}|{d}|{s}": int(n)
                for (c, d, s), n in nonstandard.groupby(
                    ["course", "distance", "stroke"], dropna=False
                ).size().sort_index().items()
            },
        },
        "wa_recompute": {
            "meter_rows_with_points": len(meters),
            "rows_with_vintage_base": len(joined),
            "vintage_coverage": round(len(joined) / len(meters), 4) if len(meters) else None,
            "match_rate": match_rate(joined),
            "match_rate_by_course": {
                c: match_rate(g) for c, g in joined.groupby("course")
            },
            "match_rate_by_season": {
                str(s): match_rate(g) for s, g in joined.groupby("season")
            },
            "mismatch_samples": [
                {
                    "swim_id": r.swim_id,
                    "course": r.course,
                    "event": f"{r.distance} {r.stroke}",
                    "swim_date": r.swim_date,
                    "stored": r.world_aquatics_points,
                    "recomputed": r.wa_points_recomputed,
                }
                for r in mismatches.sort_values("swim_id").head(10).itertuples()
            ],
        },
        "scy_rows_with_wa_points": int(
            ((swims["course"] == "SCY") & swims["world_aquatics_points"].notna()).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="*", type=int, help="recruiting classes (default: all found)")
    args = parser.parse_args()

    class_dirs = sorted(d for d in DATA_DIR.glob("swimcloud_*") if d.is_dir())
    if args.classes:
        class_dirs = [d for d in class_dirs if int(d.name.rsplit("_", 1)[1]) in args.classes]
    if not class_dirs:
        raise SystemExit("no data/swimcloud_* directories matched")

    wa_lookup = wa_base_lookup(load_wa_base_times())
    for class_dir in class_dirs:
        report = process_class(class_dir, wa_lookup, PROCESSED_DIR)
        wa = report["wa_recompute"]
        print(
            f"{class_dir.name}: {report['rows']['swims_out']} swims "
            f"({report['rows']['is_clean']} clean), WA match "
            f"{wa['match_rate']} on {wa['rows_with_vintage_base']} vintage-joined rows"
        )


if __name__ == "__main__":
    main()
