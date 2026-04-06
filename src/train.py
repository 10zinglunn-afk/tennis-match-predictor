"""
Model Training Script

Loads the full processed ATP dataset, extracts features, balances the dataset
via perspective mirroring, then trains and compares three model types using a
hard chronological cutoff (train < 2021, test 2021-2024).

Run from project root:
    PYTHONPATH=. python src/train.py
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, ROOT_DIR)  # for config.py
sys.path.insert(0, SRC_DIR)   # for features.py, models.py

import config
from features import FeatureExtractor
from models import TennisModel

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Feature columns passed to the model ──────────────────────────────────────
# Excludes raw identifiers (match_id, winner_id, loser_id, date) and the target.
# Includes both absolute per-player values and their differences so the model
# can learn from both the magnitude and the relative gap between players.
FEATURE_COLS = [
    # ELO
    'winner_elo', 'loser_elo', 'elo_diff',
    # World ranking (500 = unranked placeholder)
    'winner_rank', 'loser_rank', 'winner_rank_diff',
    # Age
    'winner_age', 'loser_age', 'winner_age_diff',
    # Recent form (rolling 90-day win %)
    'winner_form', 'loser_form',
    # Recent match load (rolling 30-day match count, normalised 0-1)
    'winner_fatigue', 'loser_fatigue',
    # Surface-specific win rate (all prior matches on same surface)
    'winner_surface_winrate', 'loser_surface_winrate', 'winner_surface_winrate_diff',
    # Head-to-head history (zeroed when < 3 prior meetings)
    'h2h_winner_wins', 'h2h_loser_wins', 'h2h_diff',
    # Tournament context
    'is_grand_slam', 'is_masters', 'is_other', 'best_of',
    # Days since last match, capped at 30
    'days_rest_winner', 'days_rest_loser', 'days_rest_diff',
]


def mirror_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a loser-perspective mirror of every match row (target = 0).

    extract_all_features() labels every row target=1 (the winner always wins).
    To give the model negative examples we duplicate each row from the loser's
    point of view:
      - Swap winner_* ↔ loser_* columns so the "loser" becomes the focal player
      - Negate all *_diff columns (they are all computed as winner_val - loser_val,
        so after the perspective swap the sign must flip)
      - Set target = 0

    The original DataFrame is unchanged; only the mirrored copy is returned.
    """
    m = features_df.copy()

    # Swap every winner_X ↔ loser_X pair
    winner_cols = [c for c in m.columns if c.startswith('winner_')]
    for w_col in winner_cols:
        l_col = 'loser_' + w_col[len('winner_'):]
        if l_col in m.columns:
            m[w_col] = features_df[l_col].values  # winner slot ← original loser value
            m[l_col] = features_df[w_col].values  # loser slot  ← original winner value

    # Negate diff columns — all are (winner_val - loser_val), sign flips after swap
    for col in [c for c in m.columns if c.endswith('_diff')]:
        m[col] = -m[col]

    m['target'] = 0
    return m


def main():
    # ── 1. Load processed dataset ─────────────────────────────────────────────
    data_path = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'all_matches.csv')
    print(f"Loading {data_path} ...")
    raw_df = pd.read_csv(data_path)
    print(f"  {len(raw_df):,} raw matches loaded  ({raw_df['tourney_date'].min()} – {raw_df['tourney_date'].max()})")

    # ── 2. Extract features ───────────────────────────────────────────────────
    print("\nExtracting features (ELO + rolling stats on full 1991-2024 history) ...")
    fe = FeatureExtractor()
    features_df = fe.extract_all_features(raw_df)
    print(f"  {len(features_df):,} feature rows")

    # ── 3. Balance via mirroring ──────────────────────────────────────────────
    # Without mirroring every label is 1 and the model learns nothing useful.
    # Mirroring doubles the dataset with loser-perspective negatives (target=0).
    mirrored = mirror_features(features_df)
    full_df = (pd.concat([features_df, mirrored], ignore_index=True)
                 .dropna(subset=FEATURE_COLS + ['target'])
                 .sort_values('date')
                 .reset_index(drop=True))
    print(f"  {len(full_df):,} rows after mirroring + dropna")

    # ── 4. Chronological split: train < 2021, test 2021-2024 ─────────────────
    CUTOFF = '20210101'
    train_df = full_df[full_df['date'] <  CUTOFF]
    test_df  = full_df[full_df['date'] >= CUTOFF]

    n_pos_test = int(test_df['target'].sum())
    print(f"\nTrain : {len(train_df):,} rows  (before {CUTOFF})")
    print(f"Test  : {len(test_df):,} rows  ({CUTOFF} onward)  "
          f"pos={n_pos_test:,}  neg={len(test_df)-n_pos_test:,}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['target'].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df['target'].values

    # Fit scaler on training data only — prevents test-set statistics leaking into scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)  # used only for logistic regression
    X_test_sc  = scaler.transform(X_test)

    # ── 5. Train all three models ─────────────────────────────────────────────
    print()
    MODEL_TYPES = ['logistic_regression', 'random_forest', 'gradient_boosting']
    results        = {}
    trained_models = {}

    for model_type in MODEL_TYPES:
        print(f"Training {model_type} ...")

        tm = TennisModel(model_type=model_type)
        tm.feature_names = FEATURE_COLS

        # Logistic regression needs scaled inputs; tree models are scale-invariant
        X_tr = X_train_sc if model_type == 'logistic_regression' else X_train
        X_te = X_test_sc  if model_type == 'logistic_regression' else X_test

        tm.model.fit(X_tr, y_train)

        y_proba = tm.model.predict_proba(X_te)[:, 1]
        y_pred  = tm.model.predict(X_te)

        ll  = log_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # Capture feature importance (RF/GBM) or |coefficient| magnitude (LogReg)
        if hasattr(tm.model, 'feature_importances_'):
            tm.feature_importance = dict(zip(FEATURE_COLS, tm.model.feature_importances_))
        elif hasattr(tm.model, 'coef_'):
            tm.feature_importance = dict(zip(FEATURE_COLS, np.abs(tm.model.coef_[0])))

        results[model_type]        = {'log_loss': ll, 'auc': auc, 'accuracy': acc}
        trained_models[model_type] = tm
        print(f"  log-loss={ll:.4f}   AUC={auc:.4f}   accuracy={acc:.4f}")

    # ── 6. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'MODEL':<26}  {'LOG-LOSS':>9}  {'AUC':>7}  {'ACCURACY':>9}")
    print("-" * 62)
    best_type = min(results, key=lambda k: results[k]['log_loss'])
    for mt, r in results.items():
        marker = " ◄" if mt == best_type else ""
        print(f"{mt:<26}  {r['log_loss']:>9.4f}  {r['auc']:>7.4f}  {r['accuracy']:>9.4f}{marker}")
    print("=" * 62)

    # ── 7. Feature importance for best model ─────────────────────────────────
    best_tm     = trained_models[best_type]
    importances = best_tm.feature_importance

    label = ("feature importances" if hasattr(best_tm.model, 'feature_importances_')
             else "|coefficient| magnitudes")
    print(f"\n{best_type.upper()} — {label} (sorted descending)\n")
    print(f"  {'FEATURE':<35}  {'SCORE':>8}")
    print("  " + "-" * 46)
    for feat, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 200) if score < 0.5 else "█" * 40
        print(f"  {feat:<35}  {score:>8.4f}  {bar}")

    # ── 8. Save best model ────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT_DIR, config.MODELS_PATH), exist_ok=True)
    save_path = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model.pkl')

    # Load old metrics for comparison before overwriting
    old_metrics = None
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                old_bundle = pickle.load(f)
            old_metrics = old_bundle.get('metrics')
        except Exception:
            pass

    new_metrics = results[best_type]
    payload = {
        'model':        best_tm.model,
        'model_type':   best_type,
        'feature_cols': FEATURE_COLS,
        'scaler':       scaler,           # needed if best_type == 'logistic_regression'
        'uses_scaling': best_type == 'logistic_regression',
        'metrics':      new_metrics,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\nBest model saved → {save_path}")

    # ── 9. Before/after comparison ────────────────────────────────────────────
    if old_metrics:
        delta_ll  = new_metrics['log_loss'] - old_metrics['log_loss']
        delta_auc = new_metrics['auc']      - old_metrics['auc']
        delta_acc = new_metrics['accuracy'] - old_metrics['accuracy']
        print("\n── Before/after decay-feature comparison (ATP) ───────────────")
        print(f"  {'METRIC':<12}  {'OLD':>8}  {'NEW':>8}  {'DELTA':>9}")
        print("  " + "-" * 43)
        print(f"  {'log-loss':<12}  {old_metrics['log_loss']:>8.4f}  {new_metrics['log_loss']:>8.4f}  {delta_ll:>+9.4f}  {'▼ better' if delta_ll < 0 else '▲ worse'}")
        print(f"  {'AUC':<12}  {old_metrics['auc']:>8.4f}  {new_metrics['auc']:>8.4f}  {delta_auc:>+9.4f}  {'▲ better' if delta_auc > 0 else '▼ worse'}")
        print(f"  {'accuracy':<12}  {old_metrics['accuracy']:>8.4f}  {new_metrics['accuracy']:>8.4f}  {delta_acc:>+9.4f}  {'▲ better' if delta_acc > 0 else '▼ worse'}")
    else:
        print("\n(No prior ATP metrics found — delta comparison requires a previous run.)")


if __name__ == '__main__':
    main()
