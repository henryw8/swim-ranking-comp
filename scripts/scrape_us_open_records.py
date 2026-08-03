"""Scrape U.S. Open swimming records (individual events) from Wikipedia.

Source: https://en.wikipedia.org/wiki/List_of_United_States_records_in_swimming

Grabs the short course yards (25 yd) men's and women's tables. Each table has
three columns: Event | American Record | U.S. Open Record. A U.S. Open cell
that reads "same" means the U.S. Open record equals the American record, so
the entry is resolved from the American record cell.

Writes data/us_open_records_scy.csv. Relay events are skipped.

With --as-of YYYY-MM-DD, scrapes the last revision of the page on or before
that date instead of the current one (records as they stood then, per
Wikipedia's edit history) and writes data/us_open_records_scy_asof_<date>.csv,
with source_url pointing at the exact revision.

Usage:
    uv run scripts/scrape_us_open_records.py [--as-of 2025-12-31] [--out x.csv]
                                             [--html cached_page.html]
"""

import argparse
import csv
import html as htmllib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAGE_TITLE = "List_of_United_States_records_in_swimming"
PAGE_URL = f"https://en.wikipedia.org/wiki/{PAGE_TITLE}"
REVISIONS_API = (
    "https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
    f"&titles={PAGE_TITLE}&rvlimit=1&rvdir=older&rvprop=ids%7Ctimestamp"
    "&format=json&rvstart={start}"
)
REVISION_HTML = f"https://en.wikipedia.org/api/rest_v1/page/html/{PAGE_TITLE}/{{revid}}"
REVISION_URL = f"https://en.wikipedia.org/w/index.php?title={PAGE_TITLE}&oldid={{revid}}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Heading id -> (course, gender), in page order; Men_3/Women_3 are the
# short course yards sections
SECTIONS = [
    ('id="Men_3"', "SCY", "men"),
    ('id="Women_3"', "SCY", "women"),
]

STROKES = {
    "free": "freestyle",
    "back": "backstroke",
    "breast": "breaststroke",
    "fly": "butterfly",
    "im": "individual medley",
}

TIME_RE = re.compile(r"^(?:(\d{1,2}):)?(?:(\d{1,2}):)?(\d{1,2})\.(\d{2})$")
MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
# "March 22, 2018 / City" or "21 February, 2026 / City"; day may carry an
# ordinal suffix ("November 21st, 2025")
DATE_LOC_RE = re.compile(
    rf"^(?P<date>(?:(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}\s+(?:{MONTHS})),\s+\d{{4}})"
    r"\s*/\s*(?P<loc>.+)$"
)

# Markers attached to the time, per the page legend
TIME_MARKERS = {
    "+": "world record",
    "*": "awaiting ratification",
    "†": "en route to final mark",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "swim-ranking-comp research script"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        # Homebrew Pythons ship without a CA bundle; fall back to the system's
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
        with urllib.request.urlopen(req, context=ctx) as resp:
            return resp.read().decode("utf-8")


def cell_lines(cell_html):
    """Reduce a table cell to plain-text lines (one per <br>)."""
    s = re.sub(r"<sup[^>]*class=\"mw-ref[^\"]*\".*?</sup>", "", cell_html, flags=re.S)
    s = re.sub(r"<br[^>]*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = htmllib.unescape(s).replace("\xa0", " ")
    return [ln.strip() for ln in s.split("\n") if ln.strip()]


def time_to_seconds(t):
    m = TIME_RE.match(t)
    if not m:
        return None
    h_or_m, m2, s, cs = m.groups()
    total = int(s) + int(cs) / 100
    if m2 is not None:  # h:mm:ss.cc (never expected here, but be safe)
        total += int(m2) * 60 + int(h_or_m) * 3600
    elif h_or_m is not None:  # m:ss.cc
        total += int(h_or_m) * 60
    return round(total, 2)


def strip_markers(t):
    for mark in TIME_MARKERS:
        t = t.replace(mark, "")
    return t.strip()


def parse_record_cell(cell_html):
    """Parse one record cell into a dict, or the string 'same', or None if empty.

    A cell can hold several entries when the record has been tied — each entry
    starts with a line matching the time pattern. The first (original) entry
    becomes the record; later ones are summarized in record_notes.
    """
    lines = cell_lines(cell_html)
    if not lines:
        return None
    if lines[0].lower() == "same":
        return "same"

    entries = []
    for ln in lines:
        if TIME_RE.match(strip_markers(ln)):
            entries.append([ln])
        elif entries:
            entries[-1].append(ln)
    if not entries:
        return None

    rec = parse_entry(entries[0], cell_html)
    if rec:
        for extra in entries[1:]:
            tie = parse_entry(extra, "")
            note = f"record equalled by {tie['swimmer']} on {tie['date_display']}"
            if tie["meet"]:
                note += f" ({tie['meet']})"
            rec["record_notes"] = ", ".join(filter(None, [rec["record_notes"], note]))
    return rec


def parse_entry(lines, cell_html):
    # First line: the time, possibly with legend markers (+, *, †) appended
    raw_time = lines[0].strip()
    notes = [note for mark, note in TIME_MARKERS.items() if mark in raw_time]
    time_display = raw_time
    for mark in TIME_MARKERS:
        time_display = time_display.replace(mark, "")
    time_display = time_display.strip()
    time_seconds = time_to_seconds(time_display)

    rec = {
        "time_display": time_display,
        "time_seconds": time_seconds,
        "swimmer": "",
        "nationality": "",
        "team": "",
        "meet": "",
        "record_date": "",
        "date_display": "",
        "location": "",
        "record_notes": ", ".join(notes),
    }

    flag = re.search(r'<img alt="([^"]+)"', cell_html)
    if flag:
        rec["nationality"] = flag.group(1)

    # Last line matching "Month D, YYYY / Location"
    body = lines[1:]
    for i in range(len(body) - 1, -1, -1):
        m = DATE_LOC_RE.match(body[i])
        if m:
            rec["date_display"] = m.group("date")
            rec["location"] = m.group("loc").strip()
            plain = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", m.group("date"))
            for fmt in ("%B %d, %Y", "%d %B, %Y"):
                try:
                    rec["record_date"] = datetime.strptime(plain, fmt).date().isoformat()
                    break
                except ValueError:
                    continue
            body = body[:i]
            break

    if body:
        # Next line: "Swimmer (Team)" — team is optional
        m = re.match(r"^(?P<name>[^()]+?)\s*\((?P<team>.+)\)\s*$", body[0])
        if m:
            rec["swimmer"] = m.group("name").strip()
            rec["team"] = m.group("team").strip()
        else:
            rec["swimmer"] = body[0].strip()
        rec["meet"] = " ".join(ln.strip() for ln in body[1:])

    return rec


def parse_event_label(label):
    """'200 breast' -> (200, 'breaststroke'); returns (None, None) if unknown."""
    m = re.match(r"^(\d+)\s+(free|back|breast|fly|IM)$", label, re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), STROKES[m.group(2).lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", dest="as_of", metavar="YYYY-MM-DD",
                    help="scrape the last page revision on or before this date")
    ap.add_argument("--out", help="output CSV path")
    ap.add_argument("--html", help="parse a cached copy of the page instead of fetching")
    args = ap.parse_args()

    source_url, page_url = PAGE_URL, PAGE_URL
    out_path = DATA_DIR / "us_open_records_scy.csv"
    if args.as_of:
        datetime.strptime(args.as_of, "%Y-%m-%d")  # validate early
        api = json.loads(fetch(REVISIONS_API.format(start=f"{args.as_of}T23:59:59Z")))
        (page_info,) = api["query"]["pages"].values()
        rev = page_info["revisions"][0]
        print(f"as of {args.as_of}: using revision {rev['revid']} from {rev['timestamp']}")
        source_url = REVISION_URL.format(revid=rev["revid"])
        page_url = REVISION_HTML.format(revid=rev["revid"])
        out_path = DATA_DIR / f"us_open_records_scy_asof_{args.as_of}.csv"
    if args.out:
        out_path = Path(args.out)

    page = Path(args.html).read_text() if args.html else fetch(page_url)
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records, warnings = [], []
    bounds = [page.find(marker) for marker, _, _ in SECTIONS]
    for i, (marker, course, gender) in enumerate(SECTIONS):
        start = bounds[i]
        if start == -1:
            sys.exit(f"section heading {marker} not found — page layout changed?")
        end = bounds[i + 1] if i + 1 < len(SECTIONS) else len(page)
        table = page[page.find("<table", start) : page.find("</table>", start)]
        if page.find("<table", start) > end:
            sys.exit(f"no table found inside section {marker}")

        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            if len(cells) != 3:  # header row or stroke-group divider row
                continue
            event = " ".join(cell_lines(cells[0]))
            if not event or "relay" in event.lower():
                continue
            # the table repeats its column headers before each stroke group,
            # using <td> cells — skip those
            if " ".join(cell_lines(cells[1])).startswith("American Record"):
                continue

            us_open = parse_record_cell(cells[2])
            same = us_open == "same"
            if same:
                us_open = parse_record_cell(cells[1])
            if not isinstance(us_open, dict):
                warnings.append(f"{course} {gender} {event}: unparseable US Open cell")
                continue
            if us_open["time_seconds"] is None:
                warnings.append(
                    f"{course} {gender} {event}: bad time {us_open['time_display']!r}"
                )
            if not us_open["record_date"]:
                warnings.append(f"{course} {gender} {event}: no date parsed")

            distance, stroke = parse_event_label(event)
            if distance is None:
                warnings.append(f"{course} {gender} {event}: unrecognized event label")

            records.append(
                {
                    "course": course,
                    "gender": gender,
                    "event": event,
                    "distance": distance,
                    "stroke": stroke,
                    **us_open,
                    "same_as_american_record": same,
                    "source_url": source_url,
                    "retrieved_at": retrieved_at,
                }
            )

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    counts = {}
    for r in records:
        key = f"{r['course']} {r['gender']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(records)} records to {out_path}")
    for key, n in counts.items():
        print(f"  {key}: {n}")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
