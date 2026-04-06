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
from datetime import datetime, timedelta
from collections import defaultdict

_tour = 'wta' if '--tour' in sys.argv and sys.argv[sys.argv.index('--tour') + 1] == 'wta' else 'atp'

if _tour == 'wta':
    CSV_PATH  = Path("data/processed/all_wta_matches.csv")
    JSON_PATH = Path("data/processed/player_stats_wta.json")
else:
    CSV_PATH  = Path("data/processed/all_matches.csv")
    JSON_PATH = Path("data/processed/player_stats.json")


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
