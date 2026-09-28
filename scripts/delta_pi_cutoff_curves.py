"""How the expected Power Index gap moves with leaderboard depth.

Reads the delta_pi_topN_slots{K} ladder produced by delta_pi_analysis.py
(--max-rank N --top-events K) and, for each gender, draws one panel per event:
E_P[dPi] against the top-N rank cutoff, one line per course pair. The curve
coefficient K is athlete-independent, so any movement along a line is purely
the P distribution of the marginalization population shifting with depth.

Writes data/processed/audit/:
  delta_pi_cutoff_curves_M.png / _F.png

Usage: uv run scripts/delta_pi_cutoff_curves.py [--slots 4]
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from delta_pi_analysis import (
    BASELINE,
    GENDER_LABEL,
    INK,
    INK_2,
    PAIR_NAMES,
    event_label,
    program_sorted,
)
from swimlib import PROCESSED_DIR

# categorical slots 1-3, fixed order — same assignment as the pair strip plot
PAIR_COLORS = {"LCM/SCY": "#2a78d6", "SCM/SCY": "#eb6834", "LCM/SCM": "#1baf7a"}
AUDIT_DIR = PROCESSED_DIR / "audit"


def tick_label(n: int) -> str:
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def load_ladder(slots: int) -> tuple[pd.DataFrame, list, str]:
    """Stack delta_pi_by_event.csv across the delta_pi_topN_slots{K} ladder.

    Returns the long frame with a numeric cutoff column, the (position, label)
    x-axis ticks — a round-number subset when the ladder is dense — and the
    snapshot date. The deepest rung covers the whole class (ranks cap at 5000)."""
    runs = []
    for d in AUDIT_DIR.glob(f"delta_pi_top*_slots{slots}"):
        m = re.fullmatch(r"delta_pi_top(\d+)_slots\d+", d.name)
        if m:
            runs.append((int(m.group(1)), d))
    runs.sort()
    if not runs:
        raise SystemExit(f"no delta_pi_top*_slots{slots} directories under {AUDIT_DIR}")
    cutoffs = [n for n, _ in runs]
    if len(cutoffs) <= 6:
        ticks = [(n, tick_label(n)) for n in cutoffs]
    else:
        ticks = [(n, tick_label(n)) for n in (500, 1000, 2000, 3000, 4000, 5000) if n in cutoffs]
    frames = []
    for n, d in runs:
        df = pd.read_csv(d / "delta_pi_by_event.csv")
        df["cutoff"] = n
        frames.append(df)
    snapshot = json.loads((runs[0][1] / "delta_pi_summary.json").read_text())["snapshot_date"]
    return pd.concat(frames, ignore_index=True), ticks, snapshot


def cutoff_figure(gender: str, long: pd.DataFrame, ticks: list, slots: int, snapshot: str, out_path) -> None:
    sub = long[long["gender"] == gender]
    events = program_sorted({(int(r.distance), r.stroke) for r in sub.itertuples()})
    y_lo = min(0.0, float(sub["dpi_mean"].min())) * 1.08
    y_hi = max(0.0, float(sub["dpi_mean"].max())) * 1.08
    x_lo, x_hi = sub["cutoff"].min(), sub["cutoff"].max()

    ncols = 5
    nrows = -(-len(events) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.1 * nrows), sharex=True, sharey=True)
    for ax, (d, s) in zip(axes.flat, events):
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
        for pair in PAIR_NAMES:
            g = sub[(sub["distance"] == d) & (sub["stroke"] == s) & (sub["course_pair"] == pair)]
            g = g.sort_values("cutoff")
            ax.plot(g["cutoff"], g["dpi_mean"], color=PAIR_COLORS[pair], lw=1.8, zorder=2)
        ax.set_title(event_label(d, s), loc="left", fontsize=8, fontweight="bold", color=INK)
        ax.set_xticks([p for p, _ in ticks], [t for _, t in ticks])
        pad = 0.04 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(events):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Expected ΔΠ by rank cutoff — {GENDER_LABEL[gender]}",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.945,
        f"E_P[ΔΠ] = mean of K/P over clean best-per-athlete numerator-course swims, swimmers ranked in the top k, "
        f"top-{slots} scoring events only.\nK is athlete-independent — movement along a line is the P distribution "
        f"shifting with leaderboard depth. ΔΠ > 0: meter course penalized.",
        va="top", fontsize=7.5, color=INK_2,
    )
    fig.legend(
        handles=[Line2D([], [], color=PAIR_COLORS[p], lw=1.8, label=p) for p in PAIR_NAMES],
        loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False, fontsize=7.5,
    )
    fig.supxlabel(f"rank cutoff k (top-k swimmers per gender · snapshot {snapshot})", fontsize=8.5, color=INK_2)
    fig.supylabel("E_P[ΔΠ] (index points)", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(top=0.865, bottom=0.1, left=0.065, right=0.99, hspace=0.42, wspace=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=4, help="the slots-K ladder to read (default 4)")
    args = parser.parse_args()

    long, ticks, snapshot = load_ladder(args.slots)
    for gender in ("M", "F"):
        cutoff_figure(gender, long, ticks, args.slots, snapshot, AUDIT_DIR / f"delta_pi_cutoff_curves_{gender}.png")
    n_events = long.drop_duplicates(["gender", "distance", "stroke"]).groupby("gender").size().to_dict()
    cutoffs = sorted(long["cutoff"].unique())
    print(
        f"delta_pi cutoff curves (slots {args.slots}, {len(cutoffs)} cutoffs {cutoffs[0]}-{cutoffs[-1]}): "
        f"2 figures, events per gender {n_events} -> {AUDIT_DIR}"
    )


if __name__ == "__main__":
    main()
