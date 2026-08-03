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

Notes on the data:

- Lifetime bests are SwimCloud `profile_fastest_times` records — one fastest
  swim per event and course.
- Ranking order is the first 200 displayed entries per gender; rank numbers
  can tie.
- User-inputted, relay-leadoff, and extracted-split records are retained and
  explicitly flagged.
