"""
Export pre-computed player stats from all_matches.csv → player_stats.json.

Run from the repo root:
    python src/export_player_stats.py            # ATP (default)
    python src/export_player_stats.py --tour wta # WTA
"""

import csv
import json
import sys
import math
from pathlib import Path

_tour = 'wta' if '--tour' in sys.argv and sys.argv[sys.argv.index('--tour') + 1] == 'wta' else 'atp'

if _tour == 'wta':
    CSV_PATH  = Path("data/processed/all_wta_matches.csv")
    JSON_PATH = Path("data/processed/player_stats_wta.json")
else:
    CSV_PATH  = Path("data/processed/all_matches.csv")
    JSON_PATH = Path("data/processed/player_stats.json")


def init_player(pid, name, ioc, hand):
    return {
        "id":         pid,
        "name":       name or "",
        "ioc":        ioc  or "—",
        "hand":       hand or "?",
        "wins":       0,
        "matches":    0,
        "surf":       {s: {"w": 0, "m": 0} for s in ("Hard", "Clay", "Grass", "Carpet")},
        "bestRank":   None,   # None = never ranked
        "firstYear":  None,
        "lastYear":   None,
        "gsWins":     0,
        "masterWins": 0,
        "opps":       {},     # opponentName → {"w": int, "l": int}
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


def build_players(csv_path):
    players = {}
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

    return sorted(
        (p for p in players.values() if p["matches"] >= 1 and p["name"]),
        key=lambda p: -p["matches"],
    )


def main():
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
