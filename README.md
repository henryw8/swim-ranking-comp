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

Notes on the data:

- Lifetime bests are SwimCloud `profile_fastest_times` records — one fastest
  swim per event and course.
- Ranking order is the first 200 displayed entries per gender; rank numbers
  can tie.
- User-inputted, relay-leadoff, and extracted-split records are retained and
  explicitly flagged.
