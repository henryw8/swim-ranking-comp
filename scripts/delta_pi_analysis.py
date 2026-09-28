"""Per-event Power Index gap curves over P and the P-marginalized delta matrix.

For each event and course pair num/den, Proposition 2 of the calibration note
gives the athlete-independent Power Index gap between equal-point swims as a
curve over World Aquatics points P:

    dPi(P) = K / P,   K = 1e5 * (rho_num^3 - rho_den^3),   rho_c = W_c / C_c

The matrix X marginalizes each curve over the empirical P distribution of the
class: clean best-per-athlete swims in the numerator (meter) course, scored at
the snapshot vintage, P = 1000 * (W_num / t)^3. Entries are E_P[dPi] = K * E[1/P].
--max-rank N restricts that population to swimmers ranked <= N per gender on
the leaderboard (K and delta are athlete-independent and unchanged; only the
P weighting moves). --top-events K further restricts it to swims that fill one
of the swimmer's K best event slots — the scoring events under SwimCloud's
slot-weighted Power Index rule (weights 1/1/0.25/0.05), reconstructed from
each swimmer's lifetime bests (every clean best scored 100(t/C)^3 - 99 against
its course's PI base, each event keeping its best course). Flags redirect
output to delta_pi_topN[_slotsK]/.

Baseline conventions match audit_calibration.py (the note's Remark on
vintages): W_LCM/W_SCM from the World Aquatics tables in force on the snapshot
date, W_SCY from the year-end U.S. Open records (implied fallback), C_c from
SwimCloud's published 2026-27 PI base times.

Stroke 50s (50 back/breast/fly) are excluded: the NCAA championship program
contests the 50 only in freestyle, so they are not recruiting events.

Writes data/processed/audit/delta_pi/:
  delta_pi_by_event.csv            per event x pair: baselines, rho, delta, K,
                                   P quartiles, E_P[dPi] and median dPi
  delta_pi_matrix_M.csv / _F.csv   the matrix X: events x course pairs
  delta_pi_curves_{G}_{PAIR}.png   small-multiple dPi(P) curves per event
  delta_pi_matrix_{G}.png          heatmap of X
  delta_pi_pair_distributions.png  strip plot of E_P[dPi] across events, course
                                   pairs compared side by side
  delta_pi_summary.json            rollup

Usage: uv run scripts/delta_pi_analysis.py [--classes 2027 ...] [--max-rank N] [--top-events K]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from audit_calibration import (
    COURSE_PAIRS,
    SCY_EQUIVALENT_DISTANCE,
    load_pi_base_times,
    load_processed,
    load_us_open_asof,
    segment_yard_baselines,
    snapshot_record_bases,
)
from swimlib import PROCESSED_DIR

STROKE_ORDER = {"freestyle": 0, "backstroke": 1, "breaststroke": 2, "butterfly": 3, "individual_medley": 4}
STROKE_SHORT = {"freestyle": "free", "backstroke": "back", "breaststroke": "breast", "butterfly": "fly", "individual_medley": "IM"}
GENDER_LABEL = {"M": "Men", "F": "Women"}
PAIR_NAMES = [f"{num}/{den}" for num, den in COURSE_PAIRS]
# the NCAA championship program has no stroke 50s — not recruiting events
NON_NCAA_EVENTS = {(50, "backstroke"), (50, "breaststroke"), (50, "butterfly")}
# SwimCloud's published-index slot weights: events 1-2 full, 3rd 25%, 4th 5%
PI_SLOT_WEIGHTS = (1.0, 1.0, 0.25, 0.05)

# minimal chart chrome
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#555555"
MUTED = "#888888"
BASELINE = "#c8c8c8"
CURVE = "#2a78d6"  # categorical slot 1
HIST = "#9ec5f4"  # sequential step 200
DIVERGING = ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7a1f1f"]  # blue <- gray -> red

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.grid": False,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8,
    }
)


def event_label(distance: int, stroke: str) -> str:
    return f"{distance} {STROKE_SHORT[stroke]}"


def program_sorted(events):
    return sorted(events, key=lambda ds: (STROKE_ORDER[ds[1]], ds[0]))


def best_times_by_event(swims: pd.DataFrame) -> pd.Series:
    """Fastest clean time per (gender, course, distance, stroke, swimmer):
    the population whose P distribution the curves are marginalized over."""
    clean = swims[swims["is_clean"] & swims["time_seconds"].notna() & (swims["time_seconds"] > 0)]
    return clean.groupby(["gender", "course", "distance", "stroke", "swimmer_id"])["time_seconds"].min()


def load_processed_datasets(datasets: list[str]) -> pd.DataFrame:
    """Load explicitly named processed snapshots using the audit loader's
    typing conventions."""
    frames = []
    for dataset in datasets:
        path = PROCESSED_DIR / dataset / "swims_clean.csv"
        if not path.exists():
            raise SystemExit(f"{path} missing — run scripts/process_data.py first")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ["time_seconds", "world_aquatics_points", "implied_wa_base", "implied_sc_base", "quality_factor"]:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"].replace("", None), errors="coerce").astype("Int64")
        df["is_clean"] = df["is_clean"] == "true"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def event_course_scores(best: pd.Series, pi_bases: dict) -> pd.DataFrame:
    """Every clean best scored 100(t/C)^3 - 99 against its course's PI base.

    SCY distance-free events fold onto their meter twins (500->400 etc., as in
    SwimCloud's own base-time table) via canon_distance; rows without a PI
    base (e.g. LCM 100 IM) are dropped."""
    df = best.reset_index()
    scy_fold = {(scy_d, s): d for (d, s), scy_d in SCY_EQUIVALENT_DISTANCE.items()}
    df["canon_distance"] = [
        scy_fold.get((int(d), s), int(d)) if c == "SCY" else int(d)
        for c, d, s in zip(df["course"], df["distance"], df["stroke"])
    ]
    df["c_base"] = [
        pi_bases.get((g, c, int(d), s))
        for g, c, d, s in zip(df["gender"], df["course"], df["distance"], df["stroke"])
    ]
    df = df[df["c_base"].notna()].copy()
    df["score"] = 100 * (df["time_seconds"] / df["c_base"]) ** 3 - 99
    return df


def scoring_slots(best: pd.Series, pi_bases: dict, published: pd.Series, k: int) -> tuple[set, dict]:
    """Reconstruct each swimmer's k scoring events: each canonical event keeps
    its best-scoring course, and the k lowest event scores are the slots.
    Returns {(swimmer_id, course, canonical_distance, stroke)} plus
    reconstruction stats against the published Power Index (PI_SLOT_WEIGHTS
    truncated to k, per-event floor of 1)."""
    df = event_course_scores(best, pi_bases)
    df = df.sort_values(["score", "course"], kind="mergesort").drop_duplicates(
        ["swimmer_id", "canon_distance", "stroke"]
    )
    df["slot"] = df.groupby("swimmer_id")["score"].rank(method="first")
    in_slots = df[df["slot"] <= k]
    slot_set = set(zip(in_slots["swimmer_id"], in_slots["course"], in_slots["canon_distance"], in_slots["stroke"]))
    errs = []
    for sid, grp in df.groupby("swimmer_id"):
        pub = published.get(sid)
        if pub is None or pd.isna(pub):
            continue
        sc = [max(x, 1.0) for x in sorted(grp["score"])[:k]]
        weights = PI_SLOT_WEIGHTS[: len(sc)]
        errs.append(abs(sum(w * x for w, x in zip(weights, sc)) / sum(weights) - pub))
    stats = {
        "n_swimmers": len(errs),
        "median_abs_err": round(float(np.median(errs)), 3),
        "share_within_1": round(float(np.mean([e <= 1.0 for e in errs])), 3),
    }
    return slot_set, stats


def delta_pi_rows(best: pd.Series, record_bases: dict, pi_bases: dict, scy_sources: dict, slots: set | None = None):
    """Per gender x event x course pair: the curve coefficient K, delta, and
    dPi marginalized over the numerator-course P distribution.

    Returns the long-table rows, {(gender, pair, distance, stroke): P array}
    for the curve figures, and the event/pair combinations skipped for lack of
    a baseline (with the missing side named)."""
    rows, p_arrays, skipped = [], {}, []
    events = sorted(
        {(g, d, s) for (g, c, d, s) in pi_bases if c in ("LCM", "SCM") and (d, s) not in NON_NCAA_EVENTS}
    )
    for gender, distance, stroke in events:
        for num, den in COURSE_PAIRS:
            d_num = distance
            d_den = SCY_EQUIVALENT_DISTANCE.get((distance, stroke), distance) if den == "SCY" else distance
            w_num = record_bases.get((gender, d_num, stroke, num))
            w_den = record_bases.get((gender, d_den, stroke, den))
            c_num = pi_bases.get((gender, num, d_num, stroke))
            c_den = pi_bases.get((gender, den, d_den, stroke))
            if None in (w_num, w_den, c_num, c_den):
                missing = [
                    name
                    for name, val in [(f"W_{num}", w_num), (f"W_{den}", w_den), (f"C_{num}", c_num), (f"C_{den}", c_den)]
                    if val is None
                ]
                skipped.append(f"{gender} {distance} {stroke} {num}/{den}: no {', '.join(missing)}")
                continue
            rho_num, rho_den = w_num / c_num, w_den / c_den
            one_plus_delta = rho_num / rho_den
            k_coeff = 1e5 * (rho_num**3 - rho_den**3)
            # identity check: the curve at P = 1000*rho_den^3 (i.e. Pi_den = 1) must
            # equal the (Pi_den + 99) * ((1+delta)^3 - 1) form used by audit_calibration
            dpi_pi1 = k_coeff / (1000 * rho_den**3)
            assert abs(dpi_pi1 - 100 * (one_plus_delta**3 - 1)) < 1e-9 * max(1.0, abs(dpi_pi1))
            try:
                times = best.loc[(gender, num, d_num, stroke)]
            except KeyError:
                times = pd.Series(dtype=float)
            if slots is not None and len(times):
                keep = np.array([(sid, num, d_num, stroke) in slots for sid in times.index])
                times = times[keep]
            points = 1000.0 * (w_num / times.to_numpy(float)) ** 3
            dpi = k_coeff / points
            p_arrays[(gender, f"{num}/{den}", distance, stroke)] = points
            q = (lambda a: np.percentile(points, a)) if len(points) else (lambda a: None)
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
                    "delta_pct": round((one_plus_delta - 1) * 100, 3),
                    "curve_k": round(k_coeff, 2),
                    "n_athletes": len(points),
                    "p_p25": None if q(25) is None else round(q(25), 1),
                    "p_p50": None if q(50) is None else round(q(50), 1),
                    "p_p75": None if q(75) is None else round(q(75), 1),
                    "dpi_mean": None if len(points) == 0 else round(float(dpi.mean()), 2),
                    "dpi_median": None if len(points) == 0 else round(float(np.median(dpi)), 2),
                    "w_scy_source": scy_sources.get((gender, d_den, stroke), "") if den == "SCY" else "",
                }
            )
    return rows, p_arrays, skipped


def matrix_frame(long: pd.DataFrame, gender: str, col: str = "dpi_mean") -> pd.DataFrame:
    """The matrix X for one gender: rows = events (program order), columns =
    course pairs, cells = the requested column (E_P[dPi] by default)."""
    sub = long[long["gender"] == gender]
    events = program_sorted({(int(r.distance), r.stroke) for r in sub.itertuples()})
    mat = pd.DataFrame(
        {
            "event": [event_label(d, s) for d, s in events],
            **{
                pair: [
                    next(
                        (getattr(r, col) for r in sub.itertuples()
                         if int(r.distance) == d and r.stroke == s and r.course_pair == pair),
                        None,
                    )
                    for d, s in events
                ]
                for pair in PAIR_NAMES
            },
        }
    )
    return mat


def curves_figure(gender: str, pair: str, sub: pd.DataFrame, p_arrays: dict, cohort: str, out_path) -> None:
    events = program_sorted({(int(r.distance), r.stroke) for r in sub.itertuples()})
    rows = {(int(r.distance), r.stroke): r for r in sub.itertuples()}
    spans = [
        (np.percentile(p, 2), np.percentile(p, 98))
        for d, s in events
        for p in [p_arrays[(gender, pair, d, s)]]
        if len(p)
    ]
    x_lo = max(50.0, min(lo for lo, _ in spans))
    x_hi = max(hi for _, hi in spans)
    grid = np.linspace(x_lo, x_hi, 300)
    ks = [rows[e].curve_k for e in events]
    y_lo = min(0.0, min(k / x_lo for k in ks)) * 1.08
    y_hi = max(0.0, max(k / x_lo for k in ks)) * 1.08

    ncols = 5
    nrows = -(-len(events) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.1 * nrows), sharex=True, sharey=True)
    bins = np.linspace(x_lo, x_hi, 31)
    for ax, (d, s) in zip(axes.flat, events):
        r = rows[(d, s)]
        p = p_arrays[(gender, pair, d, s)]
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
        if len(p):
            counts, _ = np.histogram(p, bins=bins)
            if counts.max() > 0:
                heights = counts / counts.max() * 0.20 * (y_hi - y_lo)
                ax.bar(
                    (bins[:-1] + bins[1:]) / 2, heights, width=bins[1] - bins[0],
                    bottom=y_lo, color=HIST, lw=0, zorder=0.8,
                )
        ax.plot(grid, r.curve_k / grid, color=CURVE, lw=1.5, zorder=2)
        ax.set_title(event_label(d, s), loc="left", fontsize=8)
        note = f"n={int(r.n_athletes):,}"
        if pd.notna(r.dpi_mean):
            note = f"mean {r.dpi_mean:+.1f} · " + note
        ax.text(
            0.97, 0.94, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=INK_2, zorder=3,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(events):]:
        ax.set_visible(False)
    num, den = pair.split("/")
    fig.suptitle(
        f"Power Index gap between equal-point {num} and {den} swims — {cohort}",
        x=0.01, ha="left", fontsize=10,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=CURVE, lw=1.5, label="SC power index gap at equal WA points"),
            Patch(facecolor=HIST, label=f"WA points distribution of {num} swims contributing to power indexes"),
        ],
        loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False, fontsize=7.5,
    )
    fig.supxlabel("WA points", fontsize=8.5)
    fig.supylabel("SC power index gap", x=0.008, fontsize=8.5)
    fig.text(
        0.024, 0.5, f"positive: {num} disadvantaged · negative: {num} advantaged",
        rotation=90, ha="center", va="center", fontsize=7, color=INK_2,
    )
    fig.subplots_adjust(top=0.885, bottom=0.09, left=0.08, right=0.99, hspace=0.42, wspace=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def matrix_figure(mat: pd.DataFrame, n_mat: pd.DataFrame, cohort: str, out_path) -> None:
    vals = mat[PAIR_NAMES].to_numpy(float)
    ns = n_mat[PAIR_NAMES].to_numpy(float)
    vmax = float(np.nanmax(np.abs(vals)))
    cmap = LinearSegmentedColormap.from_list("bwr_ref", DIVERGING)
    norm = Normalize(-vmax, vmax)
    # rows keep height 1; a gap opens wherever the stroke changes
    strokes = [e.split()[-1] for e in mat["event"]]
    gap = 0.55
    row_tops, y = [], 0.0
    for i, stroke in enumerate(strokes):
        if i and stroke != strokes[i - 1]:
            y += gap
        row_tops.append(y)
        y += 1.0
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * y + 1.5))
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            face = SURFACE if np.isnan(v) else cmap(norm(v))
            ax.add_patch(Rectangle((j, row_tops[i]), 1, 1, facecolor=face, edgecolor=SURFACE, lw=2))
            if np.isnan(v):
                ax.text(j + 0.5, row_tops[i] + 0.5, "–", ha="center", va="center", fontsize=7.5, color=MUTED)
            else:
                dark = abs(v) > 0.55 * vmax
                ax.text(
                    j + 0.5, row_tops[i] + 0.4, f"{v:+.1f}", ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if dark else INK,
                )
                ax.text(
                    j + 0.5, row_tops[i] + 0.76, f"n={int(ns[i, j]):,}", ha="center", va="center",
                    fontsize=5.5, color="#ffffff" if dark else INK_2, alpha=0.85,
                )
    ax.set_xlim(0, len(PAIR_NAMES))
    ax.set_ylim(y, 0)
    ax.set_xticks(np.arange(len(PAIR_NAMES)) + 0.5, PAIR_NAMES, fontsize=8)
    ax.set_yticks([t + 0.5 for t in row_tops], mat["event"], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Average power index gap between equal-point swims\n{cohort}",
        loc="left", fontsize=9.5, pad=10,
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.12, shrink=0.7
    )
    cbar.set_ticks([0])
    cbar.ax.tick_params(labelsize=7, length=0)
    cbar.outline.set_visible(False)
    cbar.ax.text(
        0.5, 1.03, "first course\ndisadvantaged", transform=cbar.ax.transAxes,
        ha="center", va="bottom", fontsize=7, color=INK_2,
    )
    cbar.ax.text(
        0.5, -0.03, "first course\nadvantaged", transform=cbar.ax.transAxes,
        ha="center", va="top", fontsize=7, color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def weighted_index(scores: list) -> float:
    """Composite Power Index from event scores: PI_SLOT_WEIGHTS over the four
    lowest, per-event floor of 1."""
    sc = [max(x, 1.0) for x in sorted(scores)[:4]]
    weights = PI_SLOT_WEIGHTS[: len(sc)]
    return sum(w * x for w, x in zip(weights, sc)) / sum(weights)


def swimmer_impact(best: pd.Series, record_bases: dict, pi_bases: dict, swimmer_info: pd.DataFrame) -> pd.DataFrame:
    """Composite-index counterfactual per swimmer: every meter swim is
    re-expressed as its equal-point SCY score (t* = t * W_SCY / W_c, scored
    against C_SCY) and the slot-weighted index is rebuilt two ways.

    Primary (pi_penalty): the portfolio is FIXED to the swimmer's actual top-4
    events — the slots SwimCloud actually scored — and only CONTRIBUTING swims
    are re-valued: a yard-valued event is unchanged, a meter-valued event gets
    its winning swim converted (falling back to the swimmer's own yard swim in
    that event if conversion devalues the meter swim below it). Non-scoring
    meter swims can never surface, so a swimmer is affected iff a meter swim
    actually values one of their scoring events (n_meter_valued_slots > 0).

    Sensitivity (pi_penalty_rerank): all swims converted and all events
    re-ranked before taking the top 4. Captures portfolio-selection distortion
    (a meter event unfairly scored out of the top 4) but can be inflated by
    stale meter swims in high-delta events; always >= pi_penalty.

    Positive = the swimmer's published-style index is worse than its
    course-neutral counterpart. Meter swims with no yard-side baseline
    (100 IM) stay unconverted and are counted per swimmer."""
    df = event_course_scores(best, pi_bases)
    cf_scores, unconverted, is_meter = [], [], []
    for r in df.itertuples():
        if r.course == "SCY":
            cf_scores.append(r.score)
            unconverted.append(False)
            is_meter.append(False)
            continue
        is_meter.append(True)
        scy_d = SCY_EQUIVALENT_DISTANCE.get((int(r.distance), r.stroke), int(r.distance))
        w_c = record_bases.get((r.gender, int(r.distance), r.stroke, r.course))
        w_y = record_bases.get((r.gender, scy_d, r.stroke, "SCY"))
        c_y = pi_bases.get((r.gender, "SCY", scy_d, r.stroke))
        if None in (w_c, w_y, c_y):
            cf_scores.append(r.score)
            unconverted.append(True)
            continue
        t_star = r.time_seconds * w_y / w_c
        cf_scores.append(100 * (t_star / c_y) ** 3 - 99)
        unconverted.append(False)
    df["cf_score"], df["unconverted"], df["is_meter"] = cf_scores, unconverted, is_meter

    rows = []
    for sid, grp in df.groupby("swimmer_id"):
        # per event: the contributing (winning) swim, with deterministic ties
        winners = grp.sort_values(["score", "course"], kind="mergesort").drop_duplicates(
            ["canon_distance", "stroke"]
        )
        yard_best = grp[grp["course"] == "SCY"].groupby(["canon_distance", "stroke"])["score"].min()
        strict_cf = [
            w.score
            if w.course == "SCY"
            else min(w.cf_score, yard_best.get((w.canon_distance, w.stroke), np.inf))
            for w in winners.itertuples()
        ]
        portfolio = winners.iloc[:4]  # already sorted: the actual scoring slots
        actual_idx = weighted_index(winners["score"].tolist())
        cf_fixed_idx = weighted_index(strict_cf[:4])
        cf_rerank_idx = weighted_index(
            grp.groupby(["canon_distance", "stroke"])["cf_score"].min().tolist()
        )
        info = swimmer_info.loc[sid]
        rows.append(
            {
                "swimmer_id": sid,
                "gender": info["gender"],
                "rank": int(info["rank"]),
                "name": info["name"],
                "published_pi": info["published_pi"],
                "actual_index": round(actual_idx, 3),
                "counterfactual_index": round(cf_fixed_idx, 3),
                "pi_penalty": round(actual_idx - cf_fixed_idx, 3),
                "counterfactual_index_rerank": round(cf_rerank_idx, 3),
                "pi_penalty_rerank": round(actual_idx - cf_rerank_idx, 3),
                "n_meter_valued_slots": int((portfolio["course"] != "SCY").sum()),
                "n_meter_swims": int(grp["is_meter"].sum()),
                "n_unconverted_meter_swims": int(grp["unconverted"].sum()),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["gender", "rank", "swimmer_id"]).reset_index(drop=True)


def swimmer_impact_figure(impact: pd.DataFrame, out_path) -> None:
    """Exceedance curve: share of swimmers whose composite-index penalty is at
    least x. Reads directly as 'how many swimmers does the miscalibration hurt
    by this much or more' and absorbs the large unaffected-at-zero mass."""
    colors = {"M": "#2a78d6", "F": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for gender in ("M", "F"):
        v = np.sort(impact[impact["gender"] == gender]["pi_penalty"].to_numpy(float))
        share_at_least = 1.0 - np.arange(len(v)) / len(v)
        ax.step(v, share_at_least, where="post", color=colors[gender], lw=1.5, label=GENDER_LABEL[gender])
    ax.axvline(0, color=BASELINE, lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], fontsize=7.5)
    ax.set_xlabel("PI penalty x (index points)", fontsize=8.5)
    ax.set_ylabel("share of swimmers ≥ x", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(labelsize=7.5)
    ax.set_title("Composite-index penalty vs course-neutral counterfactual", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def swarm_offsets(sorted_vals: np.ndarray, min_gap: float, step: float = 0.13) -> np.ndarray:
    """Deterministic beeswarm: values within min_gap of a lane's last dot are
    bumped to the next lane (0, +1, -1, +2, -2, ...)."""
    lanes: dict = {}
    offsets = np.zeros(len(sorted_vals))
    lane_order = [0, 1, -1, 2, -2, 3, -3]
    for i, v in enumerate(sorted_vals):
        for lane in lane_order:
            if lane not in lanes or v - lanes[lane] > min_gap:
                lanes[lane] = v
                offsets[i] = lane * step
                break
    return offsets


def pair_distribution_figure(long: pd.DataFrame, out_path) -> None:
    pair_colors = {"LCM/SCY": "#2a78d6", "SCM/SCY": "#eb6834", "LCM/SCM": "#1baf7a"}
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True)
    valid = long[long["dpi_mean"].notna()]
    span = valid["dpi_mean"].max() - valid["dpi_mean"].min()
    for ax, gender in zip(axes, ("M", "F")):
        sub = valid[valid["gender"] == gender]
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
        for i, pair in enumerate(PAIR_NAMES):
            g = sub[sub["course_pair"] == pair].sort_values("dpi_mean")
            vals = g["dpi_mean"].to_numpy(float)
            y = len(PAIR_NAMES) - 1 - i
            offs = swarm_offsets(vals, min_gap=span / 30)
            ax.scatter(vals, y + offs, s=30, color=pair_colors[pair], ec=SURFACE, lw=0.8, zorder=3)
            med = float(np.median(vals))
            ax.plot([med, med], [y - 0.24, y + 0.24], color=INK, lw=1.6, zorder=4)
            for row_idx, dy in ((0, offs[0]), (len(g) - 1, offs[-1])):
                r = g.iloc[row_idx]
                ax.annotate(
                    event_label(int(r["distance"]), r["stroke"]),
                    (r["dpi_mean"], y + dy), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK_2, zorder=5,
                )
        ax.set_yticks(range(len(PAIR_NAMES) - 1, -1, -1), PAIR_NAMES, fontsize=8)
        ax.set_ylim(-0.6, len(PAIR_NAMES) - 0.4)
        ax.set_title(GENDER_LABEL[gender], loc="left", fontsize=9)
        ax.tick_params(labelsize=7.5)
    axes[-1].set_xlabel("E_P[ΔΠ] (index points)", fontsize=8.5)
    fig.suptitle("E_P[ΔΠ] across events, by course pair", x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.99, hspace=0.32)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_summary(
    long: pd.DataFrame, skipped: list, classes, snapshot_date, us_open_asof, max_rank, top_events, slot_stats, impact
) -> dict:
    top = long[long["dpi_mean"].notna()].copy()
    top["abs_dpi"] = top["dpi_mean"].abs()
    top = top.sort_values(["abs_dpi", "gender", "stroke", "distance"], ascending=[False, True, True, True]).head(10)
    return {
        "classes": list(classes),
        "snapshot_date": snapshot_date,
        "us_open_asof_date": us_open_asof,
        "marginalization": (
            "E_P[dPi] = mean of K/P over clean best-per-athlete numerator-course swims, "
            "P = 1000*(W_num/t)^3 at the snapshot vintage"
        ),
        "rank_cutoff_per_gender": max_rank,
        "swimmer_impact": None if impact is None else {
            gender: {
                "n_swimmers": int((impact["gender"] == gender).sum()),
                "n_affected": int(((impact["gender"] == gender) & (impact["pi_penalty"].abs() > 1e-9)).sum()),
                "penalty_p50_p90_p99_max": [
                    round(float(x), 3)
                    for x in np.percentile(impact[impact["gender"] == gender]["pi_penalty"], [50, 90, 99]).tolist()
                    + [float(impact[impact["gender"] == gender]["pi_penalty"].max())]
                ],
                "rerank_penalty_p50_p90": [
                    round(float(x), 3)
                    for x in np.percentile(impact[impact["gender"] == gender]["pi_penalty_rerank"], [50, 90]).tolist()
                ],
                "share_rerank_exceeds_fixed_by_1": round(
                    float(
                        (
                            (impact["gender"] == gender)
                            & (impact["pi_penalty_rerank"] - impact["pi_penalty"] > 1.0)
                        ).sum()
                        / (impact["gender"] == gender).sum()
                    ),
                    3,
                ),
                "swimmers_with_unconverted_meter_swims": int(
                    ((impact["gender"] == gender) & (impact["n_unconverted_meter_swims"] > 0)).sum()
                ),
            }
            for gender in ("M", "F")
        },
        "scoring_top_events": None if top_events is None else {
            "k": top_events,
            "reconstruction_vs_published_pi": slot_stats,
        },
        "rows": len(long),
        "excluded_non_ncaa_events": sorted(f"{d} {s}" for d, s in NON_NCAA_EVENTS),
        "skipped_no_baseline": skipped,
        "events_per_pair": {
            f"{gender} {pair}": int(((long["gender"] == gender) & (long["course_pair"] == pair)).sum())
            for gender in ("M", "F")
            for pair in PAIR_NAMES
        },
        "expected_dpi_by_pair": {
            f"{gender} {pair}": {
                "median": round(float(g["dpi_mean"].median()), 2),
                "min": round(float(g["dpi_mean"].min()), 2),
                "max": round(float(g["dpi_mean"].max()), 2),
                "positive_events": int((g["dpi_mean"] > 0).sum()),
                "negative_events": int((g["dpi_mean"] < 0).sum()),
            }
            for (gender, pair), g in long[long["dpi_mean"].notna()].groupby(["gender", "course_pair"])
        },
        "top_expected_dpi_magnitude": [
            {
                "gender": r.gender,
                "event": f"{int(r.distance)} {r.stroke}",
                "course_pair": r.course_pair,
                "delta_pct": r.delta_pct,
                "dpi_mean": r.dpi_mean,
                "n_athletes": int(r.n_athletes),
            }
            for r in top.itertuples()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="*", type=int, default=[2027], help="recruiting classes to analyze")
    parser.add_argument(
        "--datasets", nargs="*",
        help="explicit processed snapshot names, e.g. swimcloud_2027_new (takes precedence over --classes)",
    )
    parser.add_argument(
        "--snapshot-date", help="baseline snapshot date YYYY-MM-DD (default: latest retrieved_at date)",
    )
    parser.add_argument(
        "--output-name", help="output directory name below data/processed/audit (default: delta_pi plus filters)",
    )
    parser.add_argument(
        "--max-rank", type=int, default=None,
        help="restrict the marginalization population to swimmers ranked <= N per gender (default: whole class)",
    )
    parser.add_argument(
        "--top-events", type=int, default=None,
        help="restrict to swims filling one of each swimmer's K best event slots (default: all swims)",
    )
    parser.add_argument(
        "--swimmer-impact", action="store_true",
        help="also compute the exact swimmer-level composite-index counterfactual (meters -> equal-point yards)",
    )
    args = parser.parse_args()

    if args.snapshot_date:
        try:
            date.fromisoformat(args.snapshot_date)
        except ValueError as exc:
            raise SystemExit("--snapshot-date must be a valid ISO date (YYYY-MM-DD)") from exc

    suffix = "".join(
        part for part, on in [(f"_top{args.max_rank}", args.max_rank), (f"_slots{args.top_events}", args.top_events)] if on
    )
    output_name = args.output_name or f"delta_pi{suffix}"
    if output_name in {"", ".", ".."} or output_name != Path(output_name).name:
        raise SystemExit("--output-name must be a single directory name")
    out_dir = PROCESSED_DIR / "audit" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    swims = load_processed_datasets(args.datasets) if args.datasets else load_processed(args.classes)
    snapshot_date = args.snapshot_date or max(swims["retrieved_at"])[:10]
    us_open, us_open_asof = load_us_open_asof(snapshot_date)
    pi_bases = load_pi_base_times()
    if not pi_bases:
        raise SystemExit("PI base times are required for the delta-PI analysis")
    yard = segment_yard_baselines(swims, us_open)
    record_bases, scy_sources = snapshot_record_bases(swims, yard, us_open, snapshot_date)

    # rank cutoff shapes only the marginalization population; baselines, yard
    # segmentation, and the snapshot date always come from the full class
    marg_swims = swims
    if args.max_rank is not None:
        marg_swims = swims[pd.to_numeric(swims["rank"], errors="coerce") <= args.max_rank]
    best = best_times_by_event(marg_swims)
    slots, slot_stats = None, None
    if args.top_events is not None:
        published = pd.to_numeric(
            marg_swims.groupby("swimmer_id")["power_index"].first(), errors="coerce"
        )
        slots, slot_stats = scoring_slots(best, pi_bases, published, args.top_events)
    rows, p_arrays, skipped = delta_pi_rows(best, record_bases, pi_bases, scy_sources, slots)
    long = pd.DataFrame.from_records(rows).sort_values(
        ["gender", "stroke", "distance", "course_pair"]
    ).reset_index(drop=True)
    long.to_csv(out_dir / "delta_pi_by_event.csv", index=False)

    n_figures = 0
    for gender in ("M", "F"):
        cohort = GENDER_LABEL[gender] if args.max_rank is None else f"Top {args.max_rank} {GENDER_LABEL[gender]}"
        mat = matrix_frame(long, gender)
        mat.to_csv(out_dir / f"delta_pi_matrix_{gender}.csv", index=False)
        n_mat = matrix_frame(long, gender, "n_athletes")
        matrix_figure(mat, n_mat, cohort, out_dir / f"delta_pi_matrix_{gender}.png")
        n_figures += 1
        for pair in PAIR_NAMES:
            sub = long[(long["gender"] == gender) & (long["course_pair"] == pair)]
            if sub.empty:
                continue
            curves_figure(gender, pair, sub, p_arrays, cohort, out_dir / f"delta_pi_curves_{gender}_{pair.replace('/', '-')}.png")
            n_figures += 1
    pair_distribution_figure(long, out_dir / "delta_pi_pair_distributions.png")
    n_figures += 1

    impact = None
    if args.swimmer_impact:
        swimmer_info = (
            marg_swims.groupby("swimmer_id")
            .agg(gender=("gender", "first"), rank=("rank", "first"), name=("name", "first"), published_pi=("power_index", "first"))
        )
        swimmer_info["rank"] = pd.to_numeric(swimmer_info["rank"], errors="coerce")
        swimmer_info["published_pi"] = pd.to_numeric(swimmer_info["published_pi"], errors="coerce")
        impact = swimmer_impact(best, record_bases, pi_bases, swimmer_info)
        impact.to_csv(out_dir / "swimmer_impact.csv", index=False)
        swimmer_impact_figure(impact, out_dir / "swimmer_impact_distribution.png")
        n_figures += 1

    summary = build_summary(
        long, skipped, args.classes, snapshot_date, us_open_asof, args.max_rank, args.top_events, slot_stats, impact
    )
    summary["processed_datasets"] = args.datasets or [f"swimcloud_{c}" for c in args.classes]
    (out_dir / "delta_pi_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    slot_note = "" if slot_stats is None else (
        f", top-{args.top_events} slot reconstruction vs published PI: "
        f"median abs err {slot_stats['median_abs_err']} over {slot_stats['n_swimmers']} swimmers"
    )
    print(
        f"delta_pi (datasets {summary['processed_datasets']}, rank cutoff {args.max_rank or 'none'}): "
        f"{len(long)} event-pair rows, 2 matrices, {n_figures} figures "
        f"(snapshot {snapshot_date}){slot_note} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
