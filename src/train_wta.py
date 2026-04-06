"""
WTA Model Training Script

Loads the full processed WTA dataset, extracts features, balances the dataset
via perspective mirroring, then trains and compares three model types using a
hard chronological cutoff (train < 2021, test 2021-2024).

Prints a side-by-side ATP vs WTA comparison if models/best_model.pkl exists.

Run from project root:
    PYTHONPATH=. python src/train_wta.py
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
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import config
from features import FeatureExtractor
from models import TennisModel
from train import mirror_features, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    # ── 1. Load processed WTA dataset ────────────────────────────────────────
    data_path = os.path.join(ROOT_DIR, config.DATA_PROCESSED_PATH, 'all_wta_matches.csv')
    print(f"Loading {data_path} ...")
    raw_df = pd.read_csv(data_path)
    print(f"  {len(raw_df):,} raw WTA matches loaded  "
          f"({raw_df['tourney_date'].min()} – {raw_df['tourney_date'].max()})")

    # ── 2. Extract features ───────────────────────────────────────────────────
    print("\nExtracting features (ELO + rolling stats on full history) ...")
    fe = FeatureExtractor()
    features_df = fe.extract_all_features(raw_df)
    print(f"  {len(features_df):,} feature rows")

    # ── 3. Balance via mirroring ──────────────────────────────────────────────
    mirrored = mirror_features(features_df)
    full_df = (pd.concat([features_df, mirrored], ignore_index=True)
                 .dropna(subset=FEATURE_COLS + ['target'])
                 .sort_values('date')
                 .reset_index(drop=True))
    print(f"  {len(full_df):,} rows after mirroring + dropna")

    # ── 4. Chronological split ────────────────────────────────────────────────
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

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
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

        X_tr = X_train_sc if model_type == 'logistic_regression' else X_train
        X_te = X_test_sc  if model_type == 'logistic_regression' else X_test

        tm.model.fit(X_tr, y_train)

        y_proba = tm.model.predict_proba(X_te)[:, 1]
        y_pred  = tm.model.predict(X_te)

        ll  = log_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        if hasattr(tm.model, 'feature_importances_'):
            tm.feature_importance = dict(zip(FEATURE_COLS, tm.model.feature_importances_))
        elif hasattr(tm.model, 'coef_'):
            tm.feature_importance = dict(zip(FEATURE_COLS, np.abs(tm.model.coef_[0])))

        results[model_type]        = {'log_loss': ll, 'auc': auc, 'accuracy': acc}
        trained_models[model_type] = tm
        print(f"  log-loss={ll:.4f}   AUC={auc:.4f}   accuracy={acc:.4f}")

    # ── 6. WTA comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'WTA MODEL':<26}  {'LOG-LOSS':>9}  {'AUC':>7}  {'ACCURACY':>9}")
    print("-" * 62)
    best_type = min(results, key=lambda k: results[k]['log_loss'])
    for mt, r in results.items():
        marker = " ◄" if mt == best_type else ""
        print(f"{mt:<26}  {r['log_loss']:>9.4f}  {r['auc']:>7.4f}  {r['accuracy']:>9.4f}{marker}")
    print("=" * 62)

    # ── 7. ATP vs WTA side-by-side comparison ────────────────────────────────
    atp_path = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model.pkl')
    if os.path.exists(atp_path):
        with open(atp_path, 'rb') as f:
            atp_bundle = pickle.load(f)
        atp_type = atp_bundle.get('model_type', '?')
        atp_r    = atp_bundle.get('metrics', {})

        print("\n── ATP vs WTA best-model comparison ─────────────────────────")
        wta_r = results[best_type]
        print(f"{'TOUR':<8}  {'BEST MODEL':<26}  {'LOG-LOSS':>9}  {'AUC':>7}  {'ACCURACY':>9}")
        print("-" * 66)
        if atp_r:
            print(f"{'ATP':<8}  {atp_type:<26}  {atp_r['log_loss']:>9.4f}  "
                  f"{atp_r['auc']:>7.4f}  {atp_r['accuracy']:>9.4f}")
        else:
            print(f"{'ATP':<8}  {atp_type:<26}  (re-run src/train.py to record ATP metrics)")
        print(f"{'WTA':<8}  {best_type:<26}  {wta_r['log_loss']:>9.4f}  "
              f"{wta_r['auc']:>7.4f}  {wta_r['accuracy']:>9.4f}")

    # ── 8. WTA feature importances ────────────────────────────────────────────
    best_tm     = trained_models[best_type]
    importances = best_tm.feature_importance

    label = ("feature importances" if hasattr(best_tm.model, 'feature_importances_')
             else "|coefficient| magnitudes")
    print(f"\nWTA {best_type.upper()} — {label} (sorted descending)\n")
    print(f"  {'FEATURE':<35}  {'SCORE':>8}")
    print("  " + "-" * 46)
    for feat, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 200) if score < 0.5 else "█" * 40
        print(f"  {feat:<35}  {score:>8.4f}  {bar}")

    # ── 9. Save WTA best model ────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT_DIR, config.MODELS_PATH), exist_ok=True)
    save_path = os.path.join(ROOT_DIR, config.MODELS_PATH, 'best_model_wta.pkl')

    # Load old metrics for comparison before overwriting
    old_wta_metrics = None
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                old_wta_bundle = pickle.load(f)
            old_wta_metrics = old_wta_bundle.get('metrics')
        except Exception:
            pass

    new_wta_metrics = results[best_type]
    payload = {
        'model':        best_tm.model,
        'model_type':   best_type,
        'feature_cols': FEATURE_COLS,
        'scaler':       scaler,
        'uses_scaling': best_type == 'logistic_regression',
        'metrics':      new_wta_metrics,
        'tour':         'wta',
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\nWTA best model saved → {save_path}")

    # ── 10. Before/after comparison ───────────────────────────────────────────
    if old_wta_metrics:
        delta_ll  = new_wta_metrics['log_loss'] - old_wta_metrics['log_loss']
        delta_auc = new_wta_metrics['auc']      - old_wta_metrics['auc']
        delta_acc = new_wta_metrics['accuracy'] - old_wta_metrics['accuracy']
        print("\n── Before/after decay-feature comparison (WTA) ───────────────")
        print(f"  {'METRIC':<12}  {'OLD':>8}  {'NEW':>8}  {'DELTA':>9}")
        print("  " + "-" * 43)
        print(f"  {'log-loss':<12}  {old_wta_metrics['log_loss']:>8.4f}  {new_wta_metrics['log_loss']:>8.4f}  {delta_ll:>+9.4f}  {'▼ better' if delta_ll < 0 else '▲ worse'}")
        print(f"  {'AUC':<12}  {old_wta_metrics['auc']:>8.4f}  {new_wta_metrics['auc']:>8.4f}  {delta_auc:>+9.4f}  {'▲ better' if delta_auc > 0 else '▼ worse'}")
        print(f"  {'accuracy':<12}  {old_wta_metrics['accuracy']:>8.4f}  {new_wta_metrics['accuracy']:>8.4f}  {delta_acc:>+9.4f}  {'▲ better' if delta_acc > 0 else '▼ worse'}")
    else:
        print("\n(No prior WTA metrics found — delta comparison requires a previous run.)")

    # Also patch metrics into ATP bundle if it exists
    if os.path.exists(atp_path) and not atp_bundle.get('metrics'):
        print("\n(Hint: run src/train.py again to record ATP metrics for side-by-side comparison)")


if __name__ == '__main__':
    main()
