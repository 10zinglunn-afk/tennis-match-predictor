"""
Prediction server — runs on port 8001.
Wraps trained GradientBoostingClassifier models for in-browser predictions.
Serves both ATP and WTA tours.

Run from project root:
    PYTHONPATH=. python src/prediction_server.py
"""

import json
import os
import sys
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import config
from features import FeatureExtractor
from predict_match import (
    get_latest_stats, compute_form, compute_fatigue,
    compute_surface_winrate, compute_h2h, days_since_last,
    build_feature_vector,
)

PORT = 8001

# ── Per-tour state ────────────────────────────────────────────────────────────
_df      = {}   # tour → DataFrame
_bundle  = {}   # tour → model pickle
_elo_map = {}   # tour → {player_id (int) → elo}
_players = {}   # tour → list of player dicts (for /players endpoint)


def _load_tour(tour: str, data_file: str, model_file: str, players_file: str):
    print(f"\nLoading {tour.upper()} match history from {data_file} ...")
    df = pd.read_csv(data_file)
    df['_date_str'] = df['tourney_date'].astype(str).str.zfill(8)
    print(f"  {len(df):,} matches loaded")
    _df[tour] = df

    print(f"Computing {tour.upper()} ELO ratings (may take ~30 s) ...")
    fe = FeatureExtractor()
    elo, _ = fe.calculate_elo_ratings(df)
    _elo_map[tour] = elo
    print(f"  ELO computed for {len(elo):,} players")

    print(f"Loading {tour.upper()} model from {model_file} ...")
    with open(model_file, 'rb') as f:
        _bundle[tour] = pickle.load(f)
    mtype = _bundle[tour].get('model_type', 'unknown')
    print(f"  Model loaded: {mtype}")

    if os.path.exists(players_file):
        with open(players_file, 'r', encoding='utf-8') as f:
            _players[tour] = json.load(f)
        print(f"  {len(_players[tour]):,} players loaded from {players_file}")
    else:
        _players[tour] = []
        print(f"  (players file not found: {players_file})")


def startup():
    atp_data    = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'all_matches.csv')
    atp_model   = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model.pkl')
    atp_players = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'player_stats.json')

    wta_data    = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'all_wta_matches.csv')
    wta_model   = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model_wta.pkl')
    wta_players = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'player_stats_wta.json')

    _load_tour('atp', atp_data, atp_model, atp_players)

    if os.path.exists(wta_data) and os.path.exists(wta_model):
        _load_tour('wta', wta_data, wta_model, wta_players)
    else:
        print("\nWTA model not found — run src/train_wta.py to enable WTA predictions")

    tours_ready = ', '.join(t.upper() for t in _bundle)
    print(f"\n✓  Server ready → http://localhost:{PORT}/predict  [{tours_ready}]")
    print("   Press Ctrl+C to stop.\n")


class PredictHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == '/health':
            tours = {t: _bundle[t].get('model_type', '?') for t in _bundle}
            body = json.dumps({
                'status':     'ok',
                'tours':      tours,
                'model_type': _bundle.get('atp', {}).get('model_type', '?'),
                'players':    len(_elo_map.get('atp', {})),
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == '/players':
            tour = (params.get('tour', ['atp'])[0]).lower()
            if tour not in _players:
                self._send_json(404, {'error': f'Tour "{tour}" not available'})
                return
            body = json.dumps(_players[tour]).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        else:
            self._send_json(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/predict':
            self._send_json(404, {'error': 'Not found'})
            return

        length = int(self.headers.get('Content-Length', 0))
        raw    = self.rfile.read(length)
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {'error': 'Invalid JSON body'})
            return

        try:
            result = _build_prediction(req)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {'error': str(e)})
        except Exception as e:
            self._send_json(500, {'error': f'Prediction error: {e}'})

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


def _build_prediction(req: dict) -> dict:
    p1_id_raw = req.get('player1_id')
    p2_id_raw = req.get('player2_id')
    surface   = req.get('surface', 'Hard')
    level     = req.get('level', 'A')
    tour      = req.get('tour', 'atp').lower()

    if not p1_id_raw or not p2_id_raw:
        raise ValueError('player1_id and player2_id are required')

    if tour not in _bundle:
        raise ValueError(f'Tour "{tour}" not available — supported: {list(_bundle.keys())}')

    try:
        p1_id = int(p1_id_raw)
        p2_id = int(p2_id_raw)
    except (ValueError, TypeError):
        raise ValueError('player IDs must be integers')

    if p1_id == p2_id:
        raise ValueError('player1_id and player2_id must be different')

    df      = _df[tour]
    elo_map = _elo_map[tour]
    bundle  = _bundle[tour]

    fe_init = config.ELO_INITIAL_RATING
    p1_elo  = elo_map.get(p1_id, fe_init)
    p2_elo  = elo_map.get(p2_id, fe_init)

    p1_stats = get_latest_stats(p1_id, df)
    p2_stats = get_latest_stats(p2_id, df)

    p1_form    = compute_form(p1_id, p1_stats['last_date'], df)
    p2_form    = compute_form(p2_id, p2_stats['last_date'], df)
    p1_fatigue = compute_fatigue(p1_id, p1_stats['last_date'], df)
    p2_fatigue = compute_fatigue(p2_id, p2_stats['last_date'], df)
    p1_swr, _  = compute_surface_winrate(p1_id, surface, df)
    p2_swr, _  = compute_surface_winrate(p2_id, surface, df)
    h2h_p1, h2h_p2 = compute_h2h(p1_id, p2_id, df)
    p1_rest = days_since_last(p1_stats['last_date'])
    p2_rest = days_since_last(p2_stats['last_date'])

    if p1_stats['rank'] is None: p1_stats['rank'] = 500.0
    if p2_stats['rank'] is None: p2_stats['rank'] = 500.0
    if p1_stats['age']  is None: p1_stats['age']  = 27.0
    if p2_stats['age']  is None: p2_stats['age']  = 27.0
    if np.isnan(p1_form): p1_form = 0.5
    if np.isnan(p2_form): p2_form = 0.5

    feat = build_feature_vector(
        p1_elo=p1_elo,             p2_elo=p2_elo,
        p1_rank=p1_stats['rank'],  p2_rank=p2_stats['rank'],
        p1_age=p1_stats['age'],    p2_age=p2_stats['age'],
        p1_form=p1_form,           p2_form=p2_form,
        p1_fatigue=p1_fatigue,     p2_fatigue=p2_fatigue,
        p1_swr=p1_swr,             p2_swr=p2_swr,
        h2h_p1=h2h_p1,             h2h_p2=h2h_p2,
        p1_rest=p1_rest,           p2_rest=p2_rest,
        surface=surface,           level=level,
    )

    feat_cols = bundle['feature_cols']
    model     = bundle['model']
    scaler    = bundle['scaler']
    use_scale = bundle['uses_scaling']
    mtype     = bundle.get('model_type', 'GradientBoostingClassifier')

    X = np.array([[feat[c] for c in feat_cols]])
    if use_scale:
        X = scaler.transform(X)

    p1_prob = float(model.predict_proba(X)[0][1])
    p2_prob = 1.0 - p1_prob

    p1_rank_disp = None if p1_stats['rank'] == 500.0 else int(p1_stats['rank'])
    p2_rank_disp = None if p2_stats['rank'] == 500.0 else int(p2_stats['rank'])

    match_count = len(df)

    features_used = {
        'p1_elo':          round(p1_elo, 0),
        'p2_elo':          round(p2_elo, 0),
        'elo_diff':        round(feat['elo_diff'], 1),
        'p1_rank':         p1_rank_disp,
        'p2_rank':         p2_rank_disp,
        'rank_diff':       round(feat['winner_rank_diff'], 1),
        'p1_form':         round(p1_form * 100, 1),
        'p2_form':         round(p2_form * 100, 1),
        'form_diff':       round((p1_form - p2_form) * 100, 1),
        'p1_surface_wr':   round(p1_swr * 100, 1),
        'p2_surface_wr':   round(p2_swr * 100, 1),
        'surface_wr_diff': round(feat['winner_surface_winrate_diff'] * 100, 1),
        'h2h_p1':          int(feat['h2h_winner_wins']),
        'h2h_p2':          int(feat['h2h_loser_wins']),
        'h2h_diff':        int(feat['h2h_diff']),
        'rest_diff':       round(feat['days_rest_diff'], 1),
    }

    return {
        'player1_win_prob': round(p1_prob * 100, 1),
        'player2_win_prob': round(p2_prob * 100, 1),
        'features_used':    features_used,
        'model_type':       mtype,
        'tour':             tour,
        'match_count':      match_count,
    }


if __name__ == '__main__':
    startup()
    httpd = HTTPServer(('', PORT), PredictHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')
