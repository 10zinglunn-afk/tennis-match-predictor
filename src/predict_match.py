"""
predict_match.py — Predict win probability for a hypothetical ATP match.

Usage:
    python src/predict_match.py "Novak Djokovic" "Carlos Alcaraz"
"""

import math
import os
import sys
import pickle
import difflib
from datetime import datetime

import numpy as np
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import config
from features import FeatureExtractor

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'all_matches.csv')
MODEL_PATH = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model.pkl')

LEVEL_LABELS = {
    'G': 'Grand Slam',
    'M': 'Masters',
    'A': 'Regular (A)',
    'F': 'Tour Finals',
    'D': 'Davis Cup',
}
SEP = '─' * 46


# ═══════════════════════════════════════════════════════════════
# 1. Data loading
# ═══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: {DATA_PATH} not found — run data_loader.py first.")
    df = pd.read_csv(DATA_PATH)
    # Pre-compute a string version of tourney_date once, used by all date helpers
    df['_date_str'] = df['tourney_date'].astype(str).str.zfill(8)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Player name resolution
# ═══════════════════════════════════════════════════════════════

def build_name_index(df: pd.DataFrame) -> dict[str, int]:
    """Return {canonical_name: player_id} using the most frequent spelling per id."""
    tally: dict[str, dict[int, int]] = {}
    for name_col, id_col in [('winner_name', 'winner_id'), ('loser_name', 'loser_id')]:
        for name, pid in zip(df[name_col], df[id_col]):
            if pd.isna(name) or pd.isna(pid):
                continue
            name = str(name).strip()
            pid  = int(pid)
            tally.setdefault(name, {})
            tally[name][pid] = tally[name].get(pid, 0) + 1

    return {name: max(pid_map, key=pid_map.get) for name, pid_map in tally.items()}


def resolve_player(query: str, name_index: dict[str, int]) -> tuple[int, str]:
    """
    Fuzzy-match a query string against all known player names.

    Resolution order:
      1. Exact case-insensitive match → use immediately
      2. difflib close matches (cutoff 0.6) → use if one clear winner
      3. Multiple close matches → print table and exit with instructions
      4. No matches → lower cutoff to 0.35 and retry; exit if still nothing
    """
    names = list(name_index.keys())
    q_low = query.lower()

    # Exact match (case-insensitive)
    exact = [n for n in names if n.lower() == q_low]
    if exact:
        return name_index[exact[0]], exact[0]

    def _score(candidates: list[str]) -> list[tuple[str, float]]:
        return sorted(
            [(n, difflib.SequenceMatcher(None, q_low, n.lower()).ratio()) for n in candidates],
            key=lambda x: x[1], reverse=True
        )

    close = difflib.get_close_matches(query, names, n=8, cutoff=0.6)
    if not close:
        close = difflib.get_close_matches(query, names, n=8, cutoff=0.35)

    if not close:
        print(f"\nNo player found matching '{query}'.")
        print("Check the spelling, or try using just a surname.")
        sys.exit(1)

    scored = _score(close)
    best_name, best_score  = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    # Unambiguous: top match is clearly better than the runner-up
    if best_score >= 0.72 and (best_score - second_score) >= 0.10:
        return name_index[best_name], best_name

    # Ambiguous — show candidates and exit
    print(f"\nAmbiguous name '{query}'. Did you mean:")
    print(f"  {'NAME':<34}  SIMILARITY")
    print(f"  {'-'*46}")
    for name, score in scored[:6]:
        print(f"  {name:<34}  {score:.2f}")
    print(f"\nRe-run with the exact name shown above.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 3. Per-player stat helpers
# ═══════════════════════════════════════════════════════════════

def _player_rows(player_id: int, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (as_winner, as_loser) sub-dataframes sorted by date."""
    as_w = df[df['winner_id'] == player_id].sort_values('tourney_date')
    as_l = df[df['loser_id']  == player_id].sort_values('tourney_date')
    return as_w, as_l


def get_latest_stats(player_id: int, df: pd.DataFrame) -> dict:
    """Most recent rank, age, last match date (int YYYYMMDD), and total matches."""
    as_w, as_l = _player_rows(player_id, df)

    last_w = int(as_w['tourney_date'].iloc[-1]) if not as_w.empty else 0
    last_l = int(as_l['tourney_date'].iloc[-1]) if not as_l.empty else 0
    last_date = max(last_w, last_l)

    # Take rank and age from the most recent match, regardless of win/loss
    if last_w >= last_l and not as_w.empty:
        rank_vals = as_w['winner_rank'].dropna()
        age_vals  = as_w['winner_age'].dropna()
        rank = float(rank_vals.iloc[-1]) if not rank_vals.empty else None
        age  = float(age_vals.iloc[-1])  if not age_vals.empty  else None
    else:
        rank_vals = as_l['loser_rank'].dropna()
        age_vals  = as_l['loser_age'].dropna()
        rank = float(rank_vals.iloc[-1]) if not rank_vals.empty else None
        age  = float(age_vals.iloc[-1])  if not age_vals.empty  else None

    return {
        'rank':         rank,
        'age':          age,
        'last_date':    last_date,
        'total_matches': len(as_w) + len(as_l),
    }


def compute_form(player_id: int, last_date_int: int,
                 df: pd.DataFrame, window_days: int = 90) -> float:
    """
    Win % over the `window_days` days leading up to (and including) last_date_int.

    Uses explicit .astype(str) cast because tourney_date is int64 in the CSV
    and pd.to_datetime with format='%Y%m%d' requires string input.
    """
    ref   = pd.to_datetime(str(int(last_date_int)).zfill(8), format='%Y%m%d')
    start = ref - pd.Timedelta(days=window_days)
    dates = pd.to_datetime(df['_date_str'], format='%Y%m%d')  # pre-cast column

    mask = (
        ((df['winner_id'] == player_id) | (df['loser_id'] == player_id)) &
        (dates >= start) &
        (dates <= ref)
    )
    rows = df[mask]
    if rows.empty:
        return float('nan')
    wins = (rows['winner_id'] == player_id).sum()
    return wins / len(rows)


def compute_fatigue(player_id: int, last_date_int: int,
                    df: pd.DataFrame, window_days: int = 30) -> float:
    """Matches per day in the 30 days up to last_date_int, normalised to [0, 1]."""
    ref   = pd.to_datetime(str(int(last_date_int)).zfill(8), format='%Y%m%d')
    start = ref - pd.Timedelta(days=window_days)
    dates = pd.to_datetime(df['_date_str'], format='%Y%m%d')

    mask = (
        ((df['winner_id'] == player_id) | (df['loser_id'] == player_id)) &
        (dates >= start) &
        (dates <= ref)
    )
    return min(df[mask].shape[0] / window_days, 1.0)


def compute_surface_winrate(player_id: int, surface: str,
                             df: pd.DataFrame, min_matches: int = 5) -> tuple[float, int]:
    """(win_rate, n_surface_matches). Returns 0.5 if fewer than min_matches."""
    surf = df[df['surface'] == surface]
    wins  = (surf['winner_id'] == player_id).sum()
    total = wins + (surf['loser_id'] == player_id).sum()
    if total < min_matches:
        return 0.5, int(total)
    return wins / total, int(total)


def compute_h2h(p1_id: int, p2_id: int, df: pd.DataFrame) -> tuple[int, int]:
    """(p1_wins_vs_p2, p2_wins_vs_p1) across all recorded meetings."""
    p1 = int((  (df['winner_id'] == p1_id) & (df['loser_id']  == p2_id)  ).sum())
    p2 = int((  (df['winner_id'] == p2_id) & (df['loser_id']  == p1_id)  ).sum())
    return p1, p2


def days_since_last(last_date_int: int, cap: int = 30) -> float:
    """Days between last_date_int and today, capped at `cap`."""
    try:
        last  = datetime.strptime(str(int(last_date_int)).zfill(8), '%Y%m%d')
        delta = (datetime.today() - last).days
        return float(min(max(delta, 0), cap))
    except (ValueError, TypeError):
        return float(cap)


# ═══════════════════════════════════════════════════════════════
# 4. Interactive prompts
# ═══════════════════════════════════════════════════════════════

def prompt_surface() -> str:
    canonical = {'hard': 'Hard', 'clay': 'Clay', 'grass': 'Grass', 'carpet': 'Carpet'}
    while True:
        raw = input("Surface  [Hard / Clay / Grass / Carpet]: ").strip().lower()
        if raw in canonical:
            return canonical[raw]
        # accept first letter
        if raw in ('h', 'c', 'g', 'ca'):
            mapping = {'h': 'Hard', 'c': 'Clay', 'g': 'Grass', 'ca': 'Carpet'}
            return mapping[raw]
        print("  Enter one of: Hard, Clay, Grass, Carpet")


def prompt_level() -> str:
    valid = {'g': 'G', 'm': 'M', 'a': 'A', 'f': 'F'}
    while True:
        raw = input("Level    [G=Grand Slam / M=Masters / A=Regular / F=Tour Finals]: ").strip().lower()
        if raw in valid:
            return valid[raw]
        if raw in ('grand slam', 'grand', 'slam'):
            return 'G'
        if raw in ('masters',):
            return 'M'
        print("  Enter one of: G, M, A, F")


# ═══════════════════════════════════════════════════════════════
# 5. Feature vector assembly
# ═══════════════════════════════════════════════════════════════

def build_feature_vector(
    *,
    p1_elo: float, p2_elo: float,
    p1_rank: float, p2_rank: float,
    p1_age: float,  p2_age: float,
    p1_form: float, p2_form: float,
    p1_fatigue: float, p2_fatigue: float,
    p1_swr: float, p2_swr: float,
    h2h_p1: int, h2h_p2: int,
    p1_rest: float, p2_rest: float,
    surface: str, level: str,
) -> dict:
    """
    Assemble the 26-feature dict in the same order the model was trained on.
    All NaN-filling and fallbacks must already be applied by the caller.
    """
    h2h_total = h2h_p1 + h2h_p2
    h2h_diff  = (h2h_p1 - h2h_p2) if h2h_total >= 3 else 0  # zero out thin samples

    return {
        'winner_elo':  p1_elo,
        'loser_elo':   p2_elo,
        'elo_diff':    p1_elo - p2_elo,

        'winner_rank':      p1_rank,
        'loser_rank':       p2_rank,
        'winner_rank_diff': p1_rank - p2_rank,

        'winner_age':      p1_age,
        'loser_age':       p2_age,
        'winner_age_diff': p1_age - p2_age,

        'winner_form':    p1_form,
        'loser_form':     p2_form,
        'winner_fatigue': p1_fatigue,
        'loser_fatigue':  p2_fatigue,

        'winner_surface_winrate':      p1_swr,
        'loser_surface_winrate':       p2_swr,
        'winner_surface_winrate_diff': p1_swr - p2_swr,

        'h2h_winner_wins': h2h_p1,
        'h2h_loser_wins':  h2h_p2,
        'h2h_diff':        h2h_diff,

        'is_grand_slam': int(level == 'G'),
        'is_masters':    int(level == 'M'),
        'is_other':      int(level not in ('G', 'M')),
        'best_of':       5 if level in ('G', 'F') else 3,  # GS and Tour Finals are best-of-5

        'days_rest_winner': p1_rest,
        'days_rest_loser':  p2_rest,
        'days_rest_diff':   p1_rest - p2_rest,
    }


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python src/predict_match.py \"Player One\" \"Player Two\"")
        sys.exit(1)

    p1_query, p2_query = sys.argv[1], sys.argv[2]

    # ── Load data ──────────────────────────────────────────────
    print("Loading match history ...")
    df = load_data()
    total_matches = len(df)
    year_min = int(df['tourney_date'].min()) // 10000
    year_max = int(df['tourney_date'].max()) // 10000

    # ── Resolve player names ───────────────────────────────────
    print("Resolving player names ...")
    name_index = build_name_index(df)

    p1_id, p1_name = resolve_player(p1_query, name_index)
    p2_id, p2_name = resolve_player(p2_query, name_index)

    if p1_id == p2_id:
        sys.exit("ERROR: both names resolved to the same player.")

    print(f"  ✓  \"{p1_query}\"  →  {p1_name}  (ID {p1_id})")
    print(f"  ✓  \"{p2_query}\"  →  {p2_name}  (ID {p2_id})")

    # ── Match context ──────────────────────────────────────────
    print()
    surface = prompt_surface()
    level   = prompt_level()

    # ── ELO ratings (full-history computation) ─────────────────
    print("\nComputing ELO ratings across full history (this takes ~30 s) ...")
    fe = FeatureExtractor()
    player_elos, _ = fe.calculate_elo_ratings(df)
    p1_elo = player_elos.get(p1_id, fe.elo_initial)
    p2_elo = player_elos.get(p2_id, fe.elo_initial)

    # ── Per-player stats ───────────────────────────────────────
    p1_stats = get_latest_stats(p1_id, df)
    p2_stats = get_latest_stats(p2_id, df)

    p1_form    = compute_form(p1_id, p1_stats['last_date'], df)
    p2_form    = compute_form(p2_id, p2_stats['last_date'], df)
    p1_fatigue = compute_fatigue(p1_id, p1_stats['last_date'], df)
    p2_fatigue = compute_fatigue(p2_id, p2_stats['last_date'], df)
    p1_swr, p1_swr_n = compute_surface_winrate(p1_id, surface, df)
    p2_swr, p2_swr_n = compute_surface_winrate(p2_id, surface, df)
    h2h_p1, h2h_p2   = compute_h2h(p1_id, p2_id, df)
    p1_rest = days_since_last(p1_stats['last_date'])
    p2_rest = days_since_last(p2_stats['last_date'])

    # ── Fill missing values + collect warnings ─────────────────
    warnings = []

    if p1_stats['rank'] is None:
        p1_stats['rank'] = 500.0
        warnings.append(f"{p1_name}: no rank found — using 500 (unranked placeholder)")
    if p2_stats['rank'] is None:
        p2_stats['rank'] = 500.0
        warnings.append(f"{p2_name}: no rank found — using 500 (unranked placeholder)")

    if p1_stats['age'] is None:
        p1_stats['age'] = 27.0
        warnings.append(f"{p1_name}: age not found — using 27.0 (tour average)")
    if p2_stats['age'] is None:
        p2_stats['age'] = 27.0
        warnings.append(f"{p2_name}: age not found — using 27.0 (tour average)")

    # ── Skill decay for inactive players ──────────────────────────────────────
    def _months_since(last_date_int: int) -> float:
        if not last_date_int:
            return 0.0
        try:
            last = datetime.strptime(str(int(last_date_int)).zfill(8), '%Y%m%d')
            return max(0.0, (datetime.today() - last).days / 30)
        except (ValueError, TypeError):
            return 0.0

    p1_months = _months_since(p1_stats['last_date'])
    p2_months = _months_since(p2_stats['last_date'])

    if np.isnan(p1_form):
        p1_form = 0.5
        warnings.append(f"{p1_name}: no matches in form window — using 0.5 (neutral)")
    else:
        p1_form = max(0.15, p1_form * math.exp(-0.05 * p1_months))
        if p1_months > 3:
            warnings.append(f"{p1_name}: {p1_months:.1f} months inactive — form decayed to {p1_form*100:.1f}%")

    if np.isnan(p2_form):
        p2_form = 0.5
        warnings.append(f"{p2_name}: no matches in form window — using 0.5 (neutral)")
    else:
        p2_form = max(0.15, p2_form * math.exp(-0.05 * p2_months))
        if p2_months > 3:
            warnings.append(f"{p2_name}: {p2_months:.1f} months inactive — form decayed to {p2_form*100:.1f}%")

    # ELO: decay toward initial rating (1500) for inactive players
    p1_elo = 1500 + (p1_elo - 1500) * math.exp(-0.03 * p1_months)
    p2_elo = 1500 + (p2_elo - 1500) * math.exp(-0.03 * p2_months)
    # ── End skill decay ────────────────────────────────────────────────────────

    if p1_swr_n < 5:
        warnings.append(
            f"{p1_name}: only {p1_swr_n} {surface} matches on record"
            f" — using 0.5 for surface win rate"
        )
    if p2_swr_n < 5:
        warnings.append(
            f"{p2_name}: only {p2_swr_n} {surface} matches on record"
            f" — using 0.5 for surface win rate"
        )

    # ── Build feature vector ───────────────────────────────────
    feat = build_feature_vector(
        p1_elo=p1_elo,     p2_elo=p2_elo,
        p1_rank=p1_stats['rank'],  p2_rank=p2_stats['rank'],
        p1_age=p1_stats['age'],    p2_age=p2_stats['age'],
        p1_form=p1_form,   p2_form=p2_form,
        p1_fatigue=p1_fatigue, p2_fatigue=p2_fatigue,
        p1_swr=p1_swr,     p2_swr=p2_swr,
        h2h_p1=h2h_p1,     h2h_p2=h2h_p2,
        p1_rest=p1_rest,   p2_rest=p2_rest,
        surface=surface,   level=level,
    )

    # ── Load model & predict ───────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: {MODEL_PATH} not found — run train.py first.")

    bundle    = pickle.load(open(MODEL_PATH, 'rb'))
    model     = bundle['model']
    feat_cols = bundle['feature_cols']
    scaler    = bundle['scaler']
    use_scale = bundle['uses_scaling']

    X = np.array([[feat[c] for c in feat_cols]])
    if use_scale:
        X = scaler.transform(X)

    p1_prob = float(model.predict_proba(X)[0][1])  # P(player-1 wins)
    p2_prob = 1.0 - p1_prob

    # ── Output ─────────────────────────────────────────────────
    print()
    if warnings:
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    level_label = LEVEL_LABELS.get(level, level)
    name_width  = max(len(p1_name), len(p2_name))

    print(SEP)
    print(f"{p1_name} vs {p2_name}")
    print(f"Surface: {surface:<10} |  Level: {level_label}")
    print(SEP)
    # Highlight the favourite with an arrow
    fav1 = " ◄" if p1_prob > p2_prob else ""
    fav2 = " ◄" if p2_prob > p1_prob else ""
    print(f"{p1_name:<{name_width}}   {p1_prob * 100:>5.1f}%{fav1}")
    print(f"{p2_name:<{name_width}}   {p2_prob * 100:>5.1f}%{fav2}")
    print(SEP)
    print(f"Based on {total_matches:,} matches ({year_min}–{year_max})")

    # ── Supporting stats table ─────────────────────────────────
    def _pct(v: float) -> str:
        return f"{v * 100:.0f}%" if not np.isnan(v) else "N/A"

    def _last_name(full: str) -> str:
        parts = full.split()
        return parts[-1] if len(parts) > 1 else full

    h2h_total = h2h_p1 + h2h_p2
    h2h_str1  = f"{h2h_p1}W–{h2h_p2}L"
    h2h_str2  = f"{h2h_p2}W–{h2h_p1}L"
    rank1_disp = f"{int(p1_stats['rank'])}" if p1_stats['rank'] != 500.0 else "N/A"
    rank2_disp = f"{int(p2_stats['rank'])}" if p2_stats['rank'] != 500.0 else "N/A"
    short1 = _last_name(p1_name)
    short2 = _last_name(p2_name)
    col = max(len(short1), len(short2), 12)

    print()
    print(f"  {'':>{col}}   {'ELO':>6}  {'Rank':>5}  {'Age':>5}  {surface+' WR':>8}  {'H2H':>7}  {'Form':>5}")
    print(f"  {'─' * (col + 48)}")
    print(
        f"  {short1:>{col}}   {p1_elo:>6.0f}  {rank1_disp:>5}  "
        f"{p1_stats['age']:>5.1f}  {_pct(p1_swr):>8}  {h2h_str1:>7}  {_pct(p1_form):>5}"
    )
    print(
        f"  {short2:>{col}}   {p2_elo:>6.0f}  {rank2_disp:>5}  "
        f"{p2_stats['age']:>5.1f}  {_pct(p2_swr):>8}  {h2h_str2:>7}  {_pct(p2_form):>5}"
    )
    if h2h_total > 0:
        print(f"\n  {h2h_total} recorded H2H match{'es' if h2h_total != 1 else ''} in dataset", end="")
        print(" (diff zeroed — fewer than 3 meetings)" if h2h_total < 3 else "")
    print()


if __name__ == '__main__':
    main()
