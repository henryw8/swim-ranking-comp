"""Audit artifacts for the cross-course calibration note.

Reads the processed class snapshots (run scripts/process_data.py first) and writes
data/processed/audit/:
  yard_baselines.csv           implied SCY record baselines, segmented by date
  calibration_table.csv        the note's per-event audit: W_c, C_c, rho_c, delta,
                               and the predicted Power Index gap
  event_conversion_ratios.csv  record-implied, official, and performance-points-implied
                               course conversion ratios per event
  equal_point_pairs.csv        same-event LCM/SCY bests with the equal-point SCY time
  audit_summary.json           rollup and reliability flags

Baseline conventions (the note's Remark on vintages): W_LCM/W_SCM come from the
World Aquatics tables in force on the snapshot date; W_SCY from the U.S. Open
records frozen at the preceding year-end (matching how the WA calendar-year
tables freeze); C_c from SwimCloud's published 2026-27 PI base times.

Usage: uv run scripts/audit_calibration.py [--classes 2026 ...]
Defaults to the 2026 class — the only one whose Power Index is contemporaneous
with the scraped bests and the 2026-27 base times.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from swimlib import DATA_DIR, PROCESSED_DIR, SEX_MAP, load_wa_base_times, wa_base_for, wa_base_lookup

EVENT_KEY = ["gender", "distance", "stroke"]
PI_BASE_TIMES_CSV = DATA_DIR / "swimcloud_pi_base_times_2026_27.csv"
# SCY has no 400/800/1500 free; these are the equivalent yard events
SCY_EQUIVALENT_DISTANCE = {(400, "freestyle"): 500, (800, "freestyle"): 1000, (1500, "freestyle"): 1650}
COURSE_PAIRS = [("LCM", "SCY"), ("SCM", "SCY"), ("LCM", "SCM")]
SEGMENT_REL_GAP = 0.0015  # split implied-base clusters on gaps beyond points-rounding noise
BOOTSTRAP_RESAMPLES = 2000


def load_us_open_asof(snapshot_date: str) -> tuple[dict, str | None]:
    """SCY U.S. Open records frozen at the year-end preceding the snapshot.

    Mirrors the World Aquatics convention: the LCM table valid in calendar year
    Y encodes records as of Dec 31 of Y-1, so the yard-side record baseline is
    frozen at the same reference date (the note's Remark on baseline vintages).
    """
    asof = f"{int(snapshot_date[:4]) - 1}-12-31"
    path = DATA_DIR / f"us_open_records_scy_asof_{asof}.csv"
    if not path.exists():
        print(f"WARNING: {path.name} not found — falling back to implied yard baselines", file=sys.stderr)
        return {}, None
    df = pd.read_csv(path)
    gender = {"men": "M", "women": "F"}
    stroke = {"individual medley": "individual_medley"}
    return {
        (gender[r.gender], int(r.distance), stroke.get(r.stroke, r.stroke)): float(r.time_seconds)
        for r in df.itertuples()
    }, asof


def load_pi_base_times() -> dict:
    """SwimCloud's published PI base times: {(gender, course, distance, stroke): seconds}."""
    if not PI_BASE_TIMES_CSV.exists():
        print(f"WARNING: {PI_BASE_TIMES_CSV.name} not found — skipping calibration table", file=sys.stderr)
        return {}
    df = pd.read_csv(PI_BASE_TIMES_CSV)
    return {
        (r.gender, r.course, int(r.distance), r.stroke): float(r.time_seconds)
        for r in df.itertuples()
    }


def load_processed(classes: list[int]) -> pd.DataFrame:
    frames = []
    for path in sorted(PROCESSED_DIR / f"swimcloud_{c}" / "swims_clean.csv" for c in classes):
        if not path.exists():
            raise SystemExit(f"{path} missing — run scripts/process_data.py first")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ["time_seconds", "world_aquatics_points", "implied_wa_base", "implied_sc_base", "quality_factor"]:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"].replace("", None), errors="coerce").astype("Int64")
        df["is_clean"] = df["is_clean"] == "true"
        frames.append(df)
    if not frames:
        raise SystemExit("no processed classes found — run scripts/process_data.py first")
    return pd.concat(frames, ignore_index=True)


def segment_yard_baselines(swims: pd.DataFrame, us_open: dict) -> pd.DataFrame:
    scy = swims[
        (swims["course"] == "SCY") & swims["is_clean"] & swims["implied_wa_base"].notna()
    ]
    records = []
    for (gender, distance, stroke), grp in scy.groupby(EVENT_KEY):
        vals = grp[["implied_wa_base", "swim_date"]].sort_values("implied_wa_base")
        # cluster on sorted implied values: a real record change moves the base by
        # more than the integer-points rounding noise
        bases = vals["implied_wa_base"].to_numpy()
        splits = np.where(np.diff(bases) / bases[:-1] > SEGMENT_REL_GAP)[0] + 1
        clusters = np.split(np.arange(len(bases)), splits)
        segments = []
        for idx in clusters:
            seg = vals.iloc[idx]
            segments.append(
                {
                    "gender": gender,
                    "distance": int(distance),
                    "stroke": stroke,
                    "n": len(seg),
                    "base_median": round(float(seg["implied_wa_base"].median()), 4),
                    "base_min": round(float(seg["implied_wa_base"].min()), 4),
                    "base_max": round(float(seg["implied_wa_base"].max()), 4),
                    "date_min": seg["swim_date"].min(),
                    "date_max": seg["swim_date"].max(),
                }
            )
        segments.sort(key=lambda s: (s["date_max"], s["base_median"]))
        for i, seg in enumerate(segments):
            seg["segment"] = i
            seg["is_latest"] = i == len(segments) - 1
            seg["overlaps_other_segment"] = any(
                o is not seg and seg["date_min"] <= o["date_max"] and o["date_min"] <= seg["date_max"]
                for o in segments
            )
        records.extend(segments)
    for seg in records:
        record = us_open.get((seg["gender"], seg["distance"], seg["stroke"]))
        seg["us_open_asof_time"] = record
        seg["us_open_diff_pct"] = (
            None if record is None else round((seg["base_median"] - record) / record * 100, 3)
        )
    cols = [
        "gender", "distance", "stroke", "segment", "is_latest", "n",
        "base_median", "base_min", "base_max", "date_min", "date_max", "overlaps_other_segment",
        "us_open_asof_time", "us_open_diff_pct",
    ]
    return pd.DataFrame.from_records(records)[cols].sort_values(
        ["gender", "stroke", "distance", "segment"]
    ).reset_index(drop=True)


def snapshot_record_bases(
    swims: pd.DataFrame, yard: pd.DataFrame, us_open: dict, snapshot_date: str
) -> tuple[dict, dict]:
    """{(gender, distance, stroke, course): base} in force on the scrape date.

    SCY bases come from the year-end U.S. Open records file when the event is
    covered there, else fall back to the implied (inverted) latest segment;
    the second return value records which source each SCY event used.
    """
    lookup = wa_base_lookup(load_wa_base_times())
    bases: dict = {}
    scy_sources: dict = {}
    for gender, distance, stroke in swims[swims["is_clean"]].groupby(EVENT_KEY).groups:
        for course in ("LCM", "SCM"):
            base = wa_base_for(lookup, SEX_MAP[gender], course, int(distance), stroke, snapshot_date)
            if base is not None:
                bases[(gender, int(distance), stroke, course)] = base
    for r in yard[yard["is_latest"]].itertuples():
        key = (r.gender, r.distance, r.stroke)
        if key in us_open:
            bases[(*key, "SCY")] = us_open[key]
            scy_sources[key] = "us_open_asof"
        else:
            bases[(*key, "SCY")] = r.base_median
            scy_sources[key] = "implied_latest"
    return bases, scy_sources


def conversion_ratios(
    swims: pd.DataFrame, record_bases: dict, scy_sources: dict, pi_bases: dict
) -> pd.DataFrame:
    clean = swims[swims["is_clean"] & swims["implied_sc_base"].notna()]
    wide = clean.pivot_table(
        index=["recruiting_class", "swimmer_id"] + EVENT_KEY,
        columns="course",
        values="implied_sc_base",
        aggfunc="first",
    ).reset_index()
    classes = sorted(clean["recruiting_class"].unique())
    rng = np.random.default_rng(0)
    records = []
    for (gender, distance, stroke), grp in wide.groupby(EVENT_KEY):
        for num, den in COURSE_PAIRS:
            if num not in grp.columns or den not in grp.columns:
                continue
            pair = grp[grp[num].notna() & grp[den].notna()]
            ratios = (pair[num] / pair[den]).to_numpy()
            if len(ratios) == 0:
                continue
            w_num = record_bases.get((gender, int(distance), stroke, num))
            w_den = record_bases.get((gender, int(distance), stroke, den))
            c_num = pi_bases.get((gender, num, int(distance), stroke))
            c_den = pi_bases.get((gender, den, int(distance), stroke))
            if len(ratios) >= 5:
                boot = np.median(
                    rng.choice(ratios, size=(BOOTSTRAP_RESAMPLES, len(ratios)), replace=True), axis=1
                )
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            else:
                ci_low = ci_high = None
            rec = {
                "gender": gender,
                "distance": int(distance),
                "stroke": stroke,
                "course_pair": f"{num}/{den}",
                "record_ratio": round(w_num / w_den, 5) if w_num and w_den else None,
                "official_c_ratio": round(c_num / c_den, 5) if c_num and c_den else None,
                "w_scy_source": scy_sources.get((gender, int(distance), stroke), "") if "SCY" in (num, den) else "",
                "n_swimmer_pairs": len(ratios),
                "sc_ratio_median": round(float(np.median(ratios)), 5),
                "sc_ratio_q25": round(float(np.percentile(ratios, 25)), 5),
                "sc_ratio_q75": round(float(np.percentile(ratios, 75)), 5),
                "sc_ratio_ci_low": None if ci_low is None else round(float(ci_low), 5),
                "sc_ratio_ci_high": None if ci_high is None else round(float(ci_high), 5),
            }
            for cls in classes:
                sub = pair[pair["recruiting_class"] == cls]
                med = (sub[num] / sub[den]).median()
                rec[f"sc_ratio_median_{cls}"] = None if pd.isna(med) else round(float(med), 5)
            records.append(rec)
    return pd.DataFrame.from_records(records).sort_values(
        ["gender", "stroke", "distance", "course_pair"]
    ).reset_index(drop=True)


def calibration_table(record_bases: dict, pi_bases: dict, scy_sources: dict) -> pd.DataFrame:
    """The note's per-event audit: rho_c = W_c/C_c per course, 1+delta = rho_num/rho_den,
    and the athlete-independent PI gap dPi = (Pi_den + 99) * ((1+delta)^3 - 1)."""
    events = sorted({(g, d, s) for (g, c, d, s) in pi_bases if c in ("LCM", "SCM")})
    rows = []
    for gender, distance, stroke in events:
        for num, den in COURSE_PAIRS:
            d_num = distance
            d_den = SCY_EQUIVALENT_DISTANCE.get((distance, stroke), distance) if den == "SCY" else distance
            w_num = record_bases.get((gender, d_num, stroke, num))
            w_den = record_bases.get((gender, d_den, stroke, den))
            c_num = pi_bases.get((gender, num, d_num, stroke))
            c_den = pi_bases.get((gender, den, d_den, stroke))
            if None in (w_num, w_den, c_num, c_den):
                continue
            rho_num, rho_den = w_num / c_num, w_den / c_den
            one_plus_delta = rho_num / rho_den
            gap_factor = one_plus_delta**3 - 1  # dPi = gap_factor * (Pi_den + 99)
            rows.append(
                {
                    "gender": gender,
                    "stroke": stroke,
                    "distance": d_num,
                    "distance_den": d_den,
                    "course_pair": f"{num}/{den}",
                    "w_num": w_num,
                    "w_den": w_den,
                    "c_num": c_num,
                    "c_den": c_den,
                    "rho_num": round(rho_num, 5),
                    "rho_den": round(rho_den, 5),
                    "one_plus_delta": round(one_plus_delta, 5),
                    "delta_pct": round((one_plus_delta - 1) * 100, 3),
                    "dpi_at_pi_1": round(gap_factor * 100, 2),
                    "dpi_at_pi_15": round(gap_factor * 114, 2),
                    "w_scy_source": scy_sources.get((gender, d_den, stroke), "") if den == "SCY" else "",
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["gender", "stroke", "distance", "course_pair"]
    ).reset_index(drop=True)


def pi_reconstruction_stats(swims: pd.DataFrame, pi_bases: dict) -> dict:
    """How well does a mean of the k best event scores 100(t/C)^3 - 99 reproduce
    the published swimmer Power Index? Diagnostic only — the aggregation rule is
    not documented; delta identification does not depend on it."""
    clean = swims[swims["is_clean"]]
    out: dict = {}
    for cls, grp in clean.groupby("recruiting_class"):
        scores: dict = {}
        observed: dict = {}
        for r in grp.itertuples():
            base = pi_bases.get((r.gender, r.course, int(r.distance), r.stroke))
            if base is None:
                continue
            scores.setdefault(r.swimmer_id, []).append(100 * (r.time_seconds / base) ** 3 - 99)
            observed[r.swimmer_id] = float(r.power_index)
        for k in (2, 3):
            errs = [
                abs(sum(sorted(v)[:k]) / k - observed[sid])
                for sid, v in scores.items()
                if len(v) >= k
            ]
            if errs:
                out.setdefault(str(cls), {})[f"top{k}_mean"] = {
                    "n": len(errs),
                    "median_abs_err": round(float(np.median(errs)), 3),
                    "share_within_0.5": round(sum(e <= 0.5 for e in errs) / len(errs), 3),
                }
    return out


def equal_point_pairs(swims: pd.DataFrame, record_bases: dict, scy_sources: dict) -> pd.DataFrame:
    clean = swims[
        swims["is_clean"] & swims["world_aquatics_points"].notna() & (swims["world_aquatics_points"] > 0)
    ]
    records = []
    keys = ["recruiting_class", "swimmer_id", "name"] + EVENT_KEY
    wide = clean.pivot_table(
        index=keys, columns="course", values=["time_seconds", "world_aquatics_points"], aggfunc="first"
    )
    for idx, row in wide.iterrows():
        rec = dict(zip(keys, idx))
        t_lcm, t_scy = row.get(("time_seconds", "LCM")), row.get(("time_seconds", "SCY"))
        p_lcm, p_scy = row.get(("world_aquatics_points", "LCM")), row.get(("world_aquatics_points", "SCY"))
        w_scy = record_bases.get((rec["gender"], int(rec["distance"]), rec["stroke"], "SCY"))
        if pd.isna(t_lcm) or pd.isna(t_scy) or pd.isna(p_lcm) or pd.isna(p_scy) or w_scy is None:
            continue
        q_lcm = (1000.0 / p_lcm) ** (1 / 3)
        t_star = q_lcm * w_scy
        records.append(
            {
                **rec,
                "distance": int(rec["distance"]),
                "t_lcm": t_lcm,
                "p_lcm": p_lcm,
                "t_scy": t_scy,
                "p_scy": p_scy,
                "q_lcm": round(q_lcm, 6),
                "w_scy": w_scy,
                "w_scy_source": scy_sources.get((rec["gender"], int(rec["distance"]), rec["stroke"]), ""),
                "t_star_scy": round(t_star, 4),
                "p_star_recomputed": round(1000 * (w_scy / t_star) ** 3, 2),
                "t_star_minus_actual_scy": round(t_star - t_scy, 4),
                "wa_points_lcm_minus_scy": round(p_lcm - p_scy, 2),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["gender", "stroke", "distance", "recruiting_class", "swimmer_id"]
    ).reset_index(drop=True)


def build_summary(
    swims, yard, ratios, pairs, calib, pi_stats, classes, snapshot_date, us_open_asof, scy_sources
) -> dict:
    class_reports = {}
    for cls in classes:
        rep = json.loads((PROCESSED_DIR / f"swimcloud_{cls}" / "validation_report.json").read_text())
        class_reports[str(rep["recruiting_class"])] = {
            "swims": rep["rows"]["swims_out"],
            "is_clean": rep["rows"]["is_clean"],
            "wa_match_rate": rep["wa_recompute"]["match_rate"],
            "wa_vintage_coverage": rep["wa_recompute"]["vintage_coverage"],
        }
    multi_seg = yard[yard["segment"] > 0][EVENT_KEY].drop_duplicates()
    latest = yard[yard["is_latest"]]
    wide_ci = ratios[
        ratios["sc_ratio_ci_low"].notna()
        & ((ratios["sc_ratio_ci_high"] - ratios["sc_ratio_ci_low"]) / ratios["sc_ratio_median"] > 0.01)
    ]

    def event_list(df):
        return sorted(f"{r.gender} {r.distance} {r.stroke}" for r in df.itertuples())

    latest_compared = latest[latest["us_open_diff_pct"].notna()]
    divergent = latest_compared[latest_compared["us_open_diff_pct"].abs() > 0.1]
    return {
        "snapshot_date": snapshot_date,
        "classes": class_reports,
        "w_scy_reference": {
            "us_open_asof_date": us_open_asof,
            "events_from_us_open": sum(1 for s in scy_sources.values() if s == "us_open_asof"),
            "events_from_implied_fallback": sorted(
                f"{g} {d} {s}" for (g, d, s), src in scy_sources.items() if src == "implied_latest"
            ),
            "implied_within_0.1pct_of_us_open": f"{len(latest_compared) - len(divergent)}/{len(latest_compared)}",
            "implied_divergent_from_us_open": {
                f"{r.gender} {r.distance} {r.stroke}": f"{r.us_open_diff_pct:+.3f}%"
                for r in divergent.itertuples()
            },
        },
        "yard_baselines": {
            "events": int(yard[EVENT_KEY].drop_duplicates().shape[0]),
            "events_with_multiple_segments": event_list(multi_seg),
            "latest_segments_low_support": event_list(latest[latest["n"] < 5]),
            "segments_with_date_overlap": event_list(
                yard[yard["overlaps_other_segment"]][EVENT_KEY].drop_duplicates()
            ),
        },
        "conversion_ratios": {
            "rows": len(ratios),
            "wide_ci_events": [
                f"{r.gender} {r.distance} {r.stroke} {r.course_pair}" for r in wide_ci.itertuples()
            ],
        },
        "calibration": None if calib is None else {
            "rows": len(calib),
            "delta_pct_by_pair": {
                pair: {
                    "median": round(float(g["delta_pct"].median()), 3),
                    "min": round(float(g["delta_pct"].min()), 3),
                    "max": round(float(g["delta_pct"].max()), 3),
                    "positive_events": int((g["delta_pct"] > 0).sum()),
                    "negative_events": int((g["delta_pct"] < 0).sum()),
                }
                for pair, g in calib.groupby("course_pair")
            },
        },
        "pi_reconstruction": pi_stats,
        "equal_point_pairs": {"rows": len(pairs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="*", type=int, default=[2026], help="recruiting classes to audit")
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    swims = load_processed(args.classes)
    snapshot_date = max(swims["retrieved_at"])[:10]
    us_open, us_open_asof = load_us_open_asof(snapshot_date)
    pi_bases = load_pi_base_times()
    yard = segment_yard_baselines(swims, us_open)
    record_bases, scy_sources = snapshot_record_bases(swims, yard, us_open, snapshot_date)
    ratios = conversion_ratios(swims, record_bases, scy_sources, pi_bases)
    pairs = equal_point_pairs(swims, record_bases, scy_sources)
    calib = calibration_table(record_bases, pi_bases, scy_sources) if pi_bases else None
    pi_stats = pi_reconstruction_stats(swims, pi_bases) if pi_bases else None

    artifacts = [(yard, "yard_baselines"), (ratios, "event_conversion_ratios"), (pairs, "equal_point_pairs")]
    if calib is not None:
        artifacts.append((calib, "calibration_table"))
    for df, name in artifacts:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == bool:
                out[col] = out[col].map({True: "true", False: "false"})
        out.to_csv(out_dir / f"{name}.csv", index=False)

    summary = build_summary(
        swims, yard, ratios, pairs, calib, pi_stats, args.classes, snapshot_date, us_open_asof, scy_sources
    )
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"audit (classes {args.classes}): {len(yard)} yard-baseline segments, "
        f"{len(ratios)} ratio rows, {len(pairs)} equal-point pairs, "
        f"{0 if calib is None else len(calib)} calibration rows (snapshot {snapshot_date})"
    )


if __name__ == "__main__":
    main()
