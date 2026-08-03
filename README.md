# swim-ranking-comp

Analysis of swimmer rankings and lifetime best times (SwimCloud data).

Managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync          # create the environment / install dependencies
uv run <script>  # run scripts inside it
```

## Data

The `data/` folder is not tracked in git. The following files are expected there:

| File | Description |
| --- | --- |
| `rankings_men.csv` | Men's ranking list — one row per swimmer: `gender, rank, swimmer_id, name, location, commitment, power_index, profile_url, retrieved_at` |
| `rankings_women.csv` | Women's ranking list, same columns |
| `lifetime_bests_men.csv` | Men's lifetime best swims — one row per swimmer/event: ranking columns plus swim details (`swim_id, distance, stroke, course, time_display, time_seconds, swim_date, age_at_swim, round, meet, meet_id, team_id, legal, exhibition, is_user_inputted, is_relay_leadoff, is_extracted_split, flags, performance_points, world_aquatics_points, split_distance, splits, result_url`) |
| `lifetime_bests_women.csv` | Women's lifetime best swims, same columns |
