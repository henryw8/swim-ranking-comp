# swim-ranking-comp

Analysis of swimmer rankings and lifetime best times (SwimCloud data).

Managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync          # create the environment / install dependencies
uv run <script>  # run scripts inside it
```

## Data

The `data/` folder is not tracked in git. It contains one folder per SwimCloud
recruiting class — `swimcloud_2024/`, `swimcloud_2025/`, `swimcloud_2026/` —
each covering the top 200 ranked swimmers per gender (400 per class). Every
folder has the same files:

| File | Description |
| --- | --- |
| `rankings_men.csv` | Men's ranking list — one row per swimmer: `gender, rank, swimmer_id, name, location, commitment, power_index, profile_url, retrieved_at` |
| `rankings_women.csv` | Women's ranking list, same columns |
| `rankings_combined.csv` | Men's and women's rankings concatenated, same columns |
| `lifetime_bests_men.csv` | Men's lifetime best swims — one row per swimmer/event/course: ranking columns plus swim details (`swim_id, distance, stroke, course, time_display, time_seconds, swim_date, age_at_swim, round, meet, meet_id, team_id, legal, exhibition, is_user_inputted, is_relay_leadoff, is_extracted_split, flags, performance_points, world_aquatics_points, split_distance, splits, result_url`) |
| `lifetime_bests_women.csv` | Women's lifetime best swims, same columns |
| `lifetime_bests_combined.csv` | Men's and women's lifetime bests concatenated, same columns |
| `rankings.json` | Raw scraped rankings, keyed by gender (`M` / `F`) |
| `profile_bests.jsonl` | Raw per-swimmer profile scrape — one line per swimmer: `{swimmer_id, ok, rows}`, where `rows` holds SwimCloud's raw fastest-time records |
| `summary.json` | Retrieval metadata: recruiting class, `retrieved_at` timestamp, row counts, and scraping notes |

### World Aquatics base times

`data/world_aquatics_base_times_2023_2027.csv` (shared across classes) holds the
official World Aquatics (FINA) points base times used for scoring — 280 rows
covering table years 2022–2026, men and women, SCM and LCM, all individual
events. Columns: `official_table_year, sex_category, pool_course, distance,
stroke, time, time_seconds, valid_from, valid_to, source_url, source_pdf,
source_sha256, retrieved_at`.

- Each table has an explicit validity window (`valid_from` / `valid_to`): SCM
  tables run Sep 1–Aug 31, LCM tables run the calendar year — use these, not
  `official_table_year`, to match a swim date to its base time.
- Every row links its source PDF on the World Aquatics site (`source_url`) with
  a SHA-256 checksum; `source_pdf` points to archived copies under
  `data/reference/world_aquatics_base_time_sources/`.

### US Open records (SCY)

`data/us_open_records_scy.csv` (shared across classes) holds the short course
yards U.S. Open swimming records — fastest times swum on American soil
regardless of nationality — for all individual events (no relays): 28 rows,
men and women. Scraped from [Wikipedia's List of United States records in
swimming](https://en.wikipedia.org/wiki/List_of_United_States_records_in_swimming)
by `scripts/scrape_us_open_records.py` (rerun it to refresh). Columns: `course,
gender, event, distance, stroke, time_display, time_seconds, swimmer,
nationality, team, meet, record_date, date_display, location, record_notes,
same_as_american_record, source_url, retrieved_at`.

- Where Wikipedia marks the U.S. Open record as "same" as the American record,
  the entry is resolved from the American record cell and
  `same_as_american_record` is `True`.
- `nationality` is filled only for non-American record holders.
- `record_notes` carries the page's legend markers when present ("world
  record", "awaiting ratification", "en route to final mark") and, for tied
  records, who equalled the mark and when — the row itself keeps the original
  swim. A meet ending in `(p)`/`(sf)`/`(r)` means the record was set in a
  preliminary/semifinal/relay leadoff.
- `--as-of YYYY-MM-DD` scrapes the last Wikipedia revision on or before that
  date instead — the records as they stood then — and writes
  `us_open_records_scy_asof_<date>.csv`, with `source_url` pinned to the exact
  revision. `data/us_open_records_scy_asof_2025-12-31.csv` (revision of
  2025-12-30) is the year-end-2025 snapshot; it differs from the current file
  on six events whose record fell in Jan–Mar 2026.

### SwimCloud PI base times

`data/swimcloud_pi_base_times_2026_27.csv` holds SwimCloud's published Power
Index base times for the 2026–27 season (the C_c recruiting baselines of the
calibration note): 106 rows — boys and girls, SCY/SCM/LCM, all individual
events. Columns: `season, gender, course, distance, stroke, time_display,
time_seconds`. Transcribed from SwimCloud's published table (screenshot,
2026-08-03). The source table lists distance freestyle as combined rows
(400/500, 800/1000, 1500/1650); they are split here as SCY 500/1000/1650 vs
SCM/LCM 400/800/1500. The LCM 100 IM has no base time (not contested).

Notes on the data:

- Lifetime bests are SwimCloud `profile_fastest_times` records — one fastest
  swim per event and course.
- Ranking order is the first 200 displayed entries per gender; rank numbers
  can tie.
- User-inputted, relay-leadoff, and extracted-split records are retained and
  explicitly flagged.

## Processing

Three scripts turn the raw snapshots into the artifacts used to explore the
cross-course calibration note (`docs/Cross_Course_Calibration_Note_v10.pdf`):

```sh
uv run scripts/process_data.py        # per class: data/processed/swimcloud_YYYY/
uv run scripts/audit_calibration.py   # pooled:    data/processed/audit/
uv run scripts/delta_pi_analysis.py   # curves:    data/processed/audit/delta_pi/
```

Per class, `process_data.py` writes:

| File | Description |
| --- | --- |
| `swims_clean.csv` | Every lifetime-best row, typed and enriched: season, `is_standard_event`, a canonical `is_clean` flag (legal, unflagged, standard event), the World Aquatics base time in force at the swim date with recomputed points and a match flag, implied WA/SwimCloud base times from inverting the stored points, and the quality factor `q` |
| `rankings_clean.csv` | Typed rankings with `recruiting_class` |
| `validation_report.json` | Row reconciliation, WA recompute match rates by course/season, flag inventory |

`audit_calibration.py` audits the 2026 class by default (`--classes` to
override) — the only class whose Power Index is contemporaneous with the
scraped bests and the 2026–27 base times. Baseline conventions follow the
note's Remark on vintages: W_LCM/W_SCM from the World Aquatics tables in force
on the snapshot date, W_SCY from the U.S. Open records frozen at the preceding
year-end (`us_open_records_scy_asof_2025-12-31.csv`, falling back to the
inverted implied baseline for events Wikipedia doesn't track, e.g. SCY 50s of
stroke), and C_c from SwimCloud's published base times. It writes:

| File | Description |
| --- | --- |
| `calibration_table.csv` | The note's central audit, per event and course pair: the four baselines (W, C on each side), ρ_c = W_c/C_c, 1+δ = ρ_num/ρ_den, and the athlete-independent PI gap ΔΠ = (Π+99)·((1+δ)³−1) evaluated at Π = 1 and Π = 15 |
| `yard_baselines.csv` | Implied SCY record baselines recovered by inverting SCY `world_aquatics_points`, segmented by swim date to expose record-vintage changes, with comparison against the year-end U.S. Open records |
| `event_conversion_ratios.csv` | Per event and course pair: record-implied and official-C conversion ratios next to the distribution (median, IQR, bootstrap CI) of within-swimmer ratios implied by `performance_points` |
| `equal_point_pairs.csv` | Swimmers with clean same-event LCM and SCY bests, with the equal-point SCY time `t* = q·W_SCY` (Proposition 1 of the note) |
| `audit_summary.json` | Validation rollup, δ distribution by course pair, PI-reconstruction diagnostics, reliability flags |

`delta_pi_analysis.py` (2027 class by default) turns the calibration table's
two anchor points into the full per-event ΔΠ curve over World Aquatics points,
ΔΠ(P) = K/P with K = 10⁵(ρ³_num − ρ³_den) (Proposition 2 of the note), and
marginalizes each curve over the class's empirical P — clean best-per-athlete
swims in the numerator (meter) course of each pair, scored at the snapshot
vintage P = 1000·(W_num/t)³. Baseline conventions are identical to
`audit_calibration.py`. Stroke 50s (50 back/breast/fly) are excluded — the
NCAA championship program contests the 50 only in freestyle, so they are not
recruiting events. `--max-rank N` restricts the marginalization population to
swimmers ranked ≤ N per gender on the leaderboard (δ and the curves are
athlete-independent and unchanged — only the P weighting moves).
`--top-events K` further restricts it to swims filling one of the swimmer's K
best event slots, reconstructed from lifetime bests: every clean best is
scored 100(t/C)³−99 against its course's PI base, SCY distance-free events
fold onto their meter twins, each event keeps its best course, and the K
lowest scores are the slots. SwimCloud aggregates slots with decayed weights —
events #1–2 at 100%, #3 at 25%, #4 at 5% — which reconstructs the published
index to 0.12 median abs error on the 2027 top-1000 (reported in the
summary), so the published PI is ~87% determined by the top two events.
`--swimmer-impact` adds a swimmer-level counterfactual: every meter swim
re-expressed as its equal-point SCY score and the slot-weighted composite
rebuilt (`swimmer_impact.csv` + `swimmer_impact_distribution.png`; 100 IM
meter swims have no yard baseline and stay unconverted, counted per swimmer).
The primary `pi_penalty` fixes the portfolio to the swimmer's actual top-4
events — the slots SwimCloud actually scored — and re-values each neutrally,
so stale meter swims outside the portfolio can't distort the number; the
`pi_penalty_rerank` sensitivity column instead re-ranks all events under
counterfactual scores (captures meter events unfairly scored out of the top
4, but stale meter swims in high-δ events can be promoted; always ≥ the
fixed-portfolio penalty). The composite is top-4 by construction — the flag
is independent of `--top-events`, which only filters the event-level matrix.
Canonical run: `--max-rank 1000 --top-events 4 --swimmer-impact`, keeping all
slot-aware artifacts together in `delta_pi_top1000_slots4/`. Flags redirect output to
`delta_pi_topN[_slotsK]/`. It writes `data/processed/audit/delta_pi/`:

| File | Description |
| --- | --- |
| `delta_pi_by_event.csv` | Per event and course pair: the four baselines, ρ on each side, δ, the curve coefficient K, the empirical P quartiles, and E_P[ΔΠ] (mean and median of K/P over the population) |
| `delta_pi_matrix_{M,F}.csv` | The matrix X — rows = events, columns = the three course pairs (LCM/SCY, SCM/SCY, LCM/SCM), cells = E_P[ΔΠ] |
| `delta_pi_curves_{G}_{PAIR}.png` | Small-multiple ΔΠ(P) curves, one panel per event, with a histogram of the empirical P distribution along the panel base (per-panel scale) |
| `delta_pi_matrix_{G}.png` | Diverging heatmap of X (ΔΠ > 0: meter course penalized) |
| `delta_pi_pair_distributions.png` | Strip plot of E_P[ΔΠ] across events — the three course pairs compared side by side per gender, with medians and labeled extremes |
| `delta_pi_summary.json` | Rollup: E_P[ΔΠ] distribution by pair, largest-magnitude events, and event/pair combinations skipped for lack of a baseline (the 100 IM, which has no yard-side record) |

Rows are never dropped: analyses subset on `is_clean`. All outputs are
deterministic — rerunning the scripts reproduces byte-identical files.
