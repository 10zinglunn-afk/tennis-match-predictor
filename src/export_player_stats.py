"""
Export pre-computed player stats from all_matches.csv → player_stats.json,
and ELO trajectory history → elo_history_{atp,wta}.json.

Run from the repo root:
    python src/export_player_stats.py                  # ATP player stats (default)
    python src/export_player_stats.py --tour wta       # WTA player stats
    python src/export_player_stats.py --elo            # ELO history for both tours
    python src/export_player_stats.py --elo --tour atp # ELO history ATP only
    python src/export_player_stats.py --elo --tour wta # ELO history WTA only
"""

import csv
import json
import sys
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

_tour = 'wta' if '--tour' in sys.argv and sys.argv[sys.argv.index('--tour') + 1] == 'wta' else 'atp'
_elo_mode = '--elo' in sys.argv

if _tour == 'wta':
    CSV_PATH  = Path("data/processed/all_wta_matches.csv")
    JSON_PATH = Path("data/processed/player_stats_wta.json")
    ELO_PATH  = Path("data/processed/elo_history_wta.json")
else:
    CSV_PATH  = Path("data/processed/all_matches.csv")
    JSON_PATH = Path("data/processed/player_stats.json")
    ELO_PATH  = Path("data/processed/elo_history_atp.json")


def init_player(pid, name, ioc, hand):
    return {
        "id":               pid,
        "name":             name or "",
        "ioc":              ioc  or "—",
        "hand":             hand or "?",
        "wins":             0,
        "matches":          0,
        "surf":             {s: {"w": 0, "m": 0} for s in ("Hard", "Clay", "Grass", "Carpet")},
        "bestRank":         None,   # None = never ranked
        "firstYear":        None,
        "lastYear":         None,
        "gsWins":           0,
        "masterWins":       0,
        "opps":             {},     # opponentName → {"w": int, "l": int}
        "last_active_date": None,   # most recent match date (YYYYMMDD string)
        "peak_form":        None,   # highest 90-day win rate ever achieved (0.0–1.0)
        "peak_elo":         None,   # highest ELO rating ever reached
    }


def process_match(p, row, won):
    p["matches"] += 1
    if won:
        p["wins"] += 1

    surface = row.get("surface") or "Hard"
    if surface not in p["surf"]:
        p["surf"][surface] = {"w": 0, "m": 0}
    p["surf"][surface]["m"] += 1
    if won:
        p["surf"][surface]["w"] += 1

    rank_str = row["winner_rank"] if won else row["loser_rank"]
    try:
        rank = float(rank_str)
        if rank > 0 and (p["bestRank"] is None or rank < p["bestRank"]):
            p["bestRank"] = rank
    except (ValueError, TypeError):
        pass

    year_str = row.get("year") or (row.get("tourney_date") or "")[:4]
    try:
        yr = int(year_str)
        if yr > 1900:
            if p["firstYear"] is None or yr < p["firstYear"]:
                p["firstYear"] = yr
            if p["lastYear"] is None or yr > p["lastYear"]:
                p["lastYear"] = yr
    except (ValueError, TypeError):
        pass

    if won and row.get("round") == "F":
        lvl = row.get("tourney_level", "")
        if lvl == "G":
            p["gsWins"] += 1
        elif lvl == "M":
            p["masterWins"] += 1

    opp = row["loser_name"] if won else row["winner_name"]
    if opp:
        if opp not in p["opps"]:
            p["opps"][opp] = {"w": 0, "l": 0}
        if won:
            p["opps"][opp]["w"] += 1
        else:
            p["opps"][opp]["l"] += 1

    date_str = (row.get("tourney_date") or "").strip()
    if date_str and (p["last_active_date"] is None or date_str > p["last_active_date"]):
        p["last_active_date"] = date_str


def build_players(csv_path):
    players = {}
    match_rows = []  # saved for second pass

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row.get("winner_id", "").strip()
            lid = row.get("loser_id",  "").strip()
            if not wid or not lid:
                continue
            if wid not in players:
                players[wid] = init_player(wid, row["winner_name"].strip(),
                                           row.get("winner_ioc", "").strip(),
                                           row.get("winner_hand", "").strip())
            if lid not in players:
                players[lid] = init_player(lid, row["loser_name"].strip(),
                                           row.get("loser_ioc", "").strip(),
                                           row.get("loser_hand", "").strip())
            process_match(players[wid], row, True)
            process_match(players[lid], row, False)
            match_rows.append(row)

    # ── Second pass: compute ELO history and peak_form ───────────────────────
    match_rows.sort(key=lambda r: r.get("tourney_date", ""))
    ELO_INIT, ELO_K = 1500, 32
    player_elos = {}
    player_matches = defaultdict(list)  # id → [(datetime, won), ...]

    for row in match_rows:
        wid  = row.get("winner_id", "").strip()
        lid  = row.get("loser_id",  "").strip()
        date = row.get("tourney_date", "").strip()
        if not wid or not lid or not date:
            continue

        w_elo = player_elos.get(wid, ELO_INIT)
        l_elo = player_elos.get(lid, ELO_INIT)
        w_exp = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))

        player_elos[wid] = w_elo + ELO_K * (1 - w_exp)
        player_elos[lid] = l_elo + ELO_K * (0 - (1 - w_exp))

        for pid, new_elo in ((wid, player_elos[wid]), (lid, player_elos[lid])):
            if pid in players:
                if players[pid]["peak_elo"] is None or new_elo > players[pid]["peak_elo"]:
                    players[pid]["peak_elo"] = round(new_elo, 1)

        try:
            dt = datetime.strptime(date.zfill(8), "%Y%m%d")
            player_matches[wid].append((dt, 1))
            player_matches[lid].append((dt, 0))
        except ValueError:
            pass

    # Peak form: highest 90-day win rate using a O(n) sliding window per player
    for pid, history in player_matches.items():
        if pid not in players:
            continue
        history.sort(key=lambda x: x[0])
        best = 0.0
        wins = 0
        left = 0
        for right, (d_end, w) in enumerate(history):
            wins += w
            while (d_end - history[left][0]).days > 90:
                wins -= history[left][1]
                left += 1
            wr = wins / (right - left + 1)
            if wr > best:
                best = wr
        players[pid]["peak_form"] = round(best, 3) if best > 0 else None

    return sorted(
        (p for p in players.values() if p["matches"] >= 1 and p["name"]),
        key=lambda p: -p["matches"],
    )


def build_elo_history(csv_path: Path) -> dict:
    """
    Walk all matches chronologically, maintaining live ELO for every player.
    At the last match of each calendar year a player appears in, snapshot
    their ELO as their end-of-year rating.

    Returns a dict:  player_id (str) → {
        "name":               str,
        "elo_by_year":        {year_str: rounded_elo, ...},
        "peak_elo":           float,
        "peak_year":          int,
        "career_start":       int,
        "career_end":         int,
        "consistency_score":  float,   # pct of active years within 100 pts of career avg
    }
    Only players with ≥ 2 active years are included (single-year careers
    produce a degenerate consistency score and clutter charts).
    """
    ELO_INIT, ELO_K = 1500.0, 32.0

    # ── Pass 1: collect rows sorted by date ──────────────────────────────────
    rows = []
    player_names: dict[str, str] = {}   # pid → most-recently-seen name
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid  = row.get("winner_id", "").strip()
            lid  = row.get("loser_id",  "").strip()
            date = row.get("tourney_date", "").strip()
            if not wid or not lid or not date:
                continue
            rows.append((date, wid, lid,
                         row.get("winner_name", "").strip(),
                         row.get("loser_name",  "").strip()))
            player_names[wid] = row.get("winner_name", "").strip()
            player_names[lid] = row.get("loser_name",  "").strip()

    rows.sort(key=lambda r: r[0])

    # ── Pass 2: update ELO chronologically, snapshot at year boundary ─────────
    player_elos: dict[str, float]              = {}   # pid → current elo
    last_year_seen: dict[str, int]             = {}   # pid → last year processed
    # elo_snapshots[pid][year] = elo at the last match of that year for this player
    elo_snapshots: dict[str, dict[int, float]] = defaultdict(dict)

    prev_year: dict[str, int] = {}  # pid → year of their last processed match

    for date_str, wid, lid, _, _ in rows:
        try:
            yr = int(date_str[:4])
        except (ValueError, IndexError):
            continue

        w_elo = player_elos.get(wid, ELO_INIT)
        l_elo = player_elos.get(lid, ELO_INIT)
        w_exp = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))

        new_w = w_elo + ELO_K * (1.0 - w_exp)
        new_l = l_elo + ELO_K * (0.0 - (1.0 - w_exp))

        player_elos[wid] = new_w
        player_elos[lid] = new_l

        # For each player: if we've moved into a new year since their last match,
        # the previous match was the last one of the prior year — snapshot already
        # captured below. Now update the running snapshot for the current year.
        for pid, elo in ((wid, new_w), (lid, new_l)):
            elo_snapshots[pid][yr] = round(elo, 1)
            last_year_seen[pid]    = yr

    # ── Pass 3: compute derived fields per player ─────────────────────────────
    result: dict[str, dict] = {}

    for pid, year_map in elo_snapshots.items():
        if len(year_map) < 2:
            continue  # skip single-year careers (degenerate consistency score)

        years        = sorted(year_map.keys())
        elos         = [year_map[y] for y in years]
        peak_elo     = max(elos)
        peak_year    = years[elos.index(peak_elo)]
        career_start = years[0]
        career_end   = years[-1]

        # Consistency: % of active years within 100 pts of career-average ELO
        avg_elo      = sum(elos) / len(elos)
        within_100   = sum(1 for e in elos if abs(e - avg_elo) <= 100)
        consistency  = round(within_100 / len(elos), 3)

        result[pid] = {
            "name":              player_names.get(pid, ""),
            "elo_by_year":       {str(y): year_map[y] for y in years},
            "peak_elo":          round(peak_elo, 1),
            "peak_year":         peak_year,
            "career_start":      career_start,
            "career_end":        career_end,
            "consistency_score": consistency,
        }

    return result


def print_elo_summary(data: dict, tour_label: str) -> None:
    """Print top-5 tables for peak_elo, consistency_score, and career length."""
    players = list(data.values())

    print(f"\n{'═'*62}")
    print(f"  {tour_label} — ELO history export summary")
    print(f"{'═'*62}")
    print(f"  Total players exported: {len(players):,}\n")

    def _table(title: str, ranked: list, key_fn, fmt_fn) -> None:
        print(f"  ── {title} ─────────────────────────────")
        print(f"  {'#':<3}  {'PLAYER':<30}  {'VALUE':>10}")
        print(f"  {'─'*49}")
        for i, p in enumerate(ranked[:5], 1):
            print(f"  {i:<3}  {p['name']:<30}  {fmt_fn(key_fn(p)):>10}")
        print()

    by_peak = sorted(players, key=lambda p: p["peak_elo"], reverse=True)
    _table("Top 5 by Peak ELO", by_peak,
           lambda p: p["peak_elo"], lambda v: f"{v:.1f}")

    # Consistency leaderboard: require ≥ 5 active years AND peak ELO > 1700
    # so fringe players near the 1500 baseline don't dominate with trivially low variance
    eligible = [p for p in players
                if (p["career_end"] - p["career_start"]) >= 5 and p["peak_elo"] > 1700]
    by_consistency = sorted(eligible, key=lambda p: p["consistency_score"], reverse=True)
    _table("Top 5 by Consistency Score (≥5 yrs, peak ELO >1700)", by_consistency,
           lambda p: p["consistency_score"], lambda v: f"{v:.3f}")

    by_career = sorted(
        players,
        key=lambda p: p["career_end"] - p["career_start"],
        reverse=True,
    )
    _table("Top 5 by Career Length", by_career,
           lambda p: p["career_end"] - p["career_start"],
           lambda v: f"{v} yrs")


def export_elo_history(csv_path: Path, json_path: Path, tour_label: str) -> None:
    print(f"Building ELO trajectory for {tour_label} from {csv_path} …")
    data = build_elo_history(csv_path)
    print(f"  {len(data):,} players with ≥ 2 active years")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = json_path.stat().st_size / 1024
    print(f"  Wrote {json_path}  ({size_kb:,.0f} KB)")

    print_elo_summary(data, tour_label)


def main():
    if _elo_mode:
        # Run ELO history export for the requested tour(s)
        if '--tour' in sys.argv:
            # Single tour specified
            if _tour == 'wta':
                wta_csv = Path("data/processed/all_wta_matches.csv")
                export_elo_history(wta_csv, Path("data/processed/elo_history_wta.json"), "WTA")
            else:
                atp_csv = Path("data/processed/all_matches.csv")
                export_elo_history(atp_csv, Path("data/processed/elo_history_atp.json"), "ATP")
        else:
            # No --tour flag → run both
            atp_csv = Path("data/processed/all_matches.csv")
            export_elo_history(atp_csv, Path("data/processed/elo_history_atp.json"), "ATP")

            wta_csv = Path("data/processed/all_wta_matches.csv")
            export_elo_history(wta_csv, Path("data/processed/elo_history_wta.json"), "WTA")
    else:
        # Default: player stats export
        print(f"Reading {CSV_PATH} …")
        players = build_players(CSV_PATH)
        print(f"  {len(players):,} players found")

        JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(players, f, separators=(",", ":"))

        size_kb = JSON_PATH.stat().st_size / 1024
        print(f"Wrote {JSON_PATH}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
