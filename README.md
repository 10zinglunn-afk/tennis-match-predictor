# Tennis Match Prediction Engine

> ML pipeline predicting ATP & WTA match outcomes with 81.8% accuracy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Model](https://img.shields.io/badge/Model-GradientBoosting-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project builds a full machine learning pipeline on top of Jeff Sackmann's open tennis datasets — 108,375 ATP matches and 93,524 WTA matches spanning 1991–2024. It extracts 11 engineered features per match (ELO ratings, ranking, form, surface win rate, head-to-head history, and more), trains a GradientBoostingClassifier with a strict chronological train/test split, and serves predictions through a local HTTP API. A self-contained interactive dashboard lets you explore every player in the dataset, inspect feature importance, and predict any head-to-head matchup in your browser.

## Model Performance

| Tour | Model | Log-loss | AUC | Accuracy |
|------|-------|----------|-----|----------|
| **ATP** | GradientBoosting | 0.355 | 0.922 | **81.8%** |
| **WTA** | GradientBoosting | 0.366 | 0.917 | **81.5%** |
| Random baseline | — | 0.693 | 0.500 | 50.0% |

The model cuts log-loss by 48.8% versus a naive random baseline. ATP betting markets typically achieve AUC 0.75–0.82; this pipeline reaches **0.922 on four completely unseen years (2021–2024)**.

## Features

The model uses 11 engineered features. All are computed point-in-time — no future data leaks back into any training row.

| Feature | Description |
|---------|-------------|
| `elo_diff` | Difference in Elo ratings; computed pre-match before any update is applied |
| `winner_rank_diff` | Difference in ATP/WTA world ranking (500 = unranked placeholder) |
| `winner_age_diff` | Age gap between the two players in years |
| `winner_form` / `loser_form` | Rolling 90-day win percentage leading up to the match |
| `winner_fatigue` / `loser_fatigue` | Matches played in the prior 30 days, normalised to [0, 1] |
| `winner_surface_winrate_diff` | Cumulative career win rate on the specific surface (Hard / Clay / Grass) |
| `h2h_diff` | Head-to-head win margin; zeroed when fewer than 3 prior meetings exist |
| `is_grand_slam` | 1 if the tournament is a Grand Slam, else 0 |
| `is_masters` | 1 if ATP Masters 1000 / WTA Premier Mandatory, else 0 |
| `best_of` | Sets required to win: 5 for Grand Slams and Tour Finals, 3 otherwise |
| `days_rest_diff` | Difference in days since each player's last match, capped at 30 |

## Architecture

```
data/raw/tennis_atp/         Jeff Sackmann CSVs (ATP, 1991–2024)
data/raw/tennis_wta/         Jeff Sackmann CSVs (WTA, 1991–2024)
         │
         ▼
src/data_loader.py           Loads & concatenates annual CSVs → all_matches.csv
         │
         ▼
src/features.py              Extracts point-in-time features (ELO, form, H2H, …)
         │
         ▼
src/train.py  /  train_wta.py   Trains 3 models, selects best by log-loss → best_model.pkl
         │
         ▼
src/prediction_server.py     HTTP server on :8001 — POST /predict, GET /players
         │
         ▼
outputs/pipeline_dashboard.html   Self-contained browser dashboard (no build step)
```

## Quick Start

**Prerequisites:** Python 3.10+, Git

### 1. Clone the repository

```bash
git clone https://github.com/10zinglunn-afk/tennis-match-predictor.git
cd tennis-match-predictor
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the raw data

```bash
git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp
git clone https://github.com/JeffSackmann/tennis_wta.git data/raw/tennis_wta
```

### 4. Build processed datasets

```bash
PYTHONPATH=. python src/data_loader.py          # ATP → data/processed/all_matches.csv

PYTHONPATH=. python -c "
from src.data_loader import build_wta_processed_dataset
build_wta_processed_dataset()
"                                               # WTA → data/processed/all_wta_matches.csv
```

### 5. Export player stats (required for the dashboard)

```bash
PYTHONPATH=. python src/export_player_stats.py             # ATP
PYTHONPATH=. python src/export_player_stats.py --tour wta  # WTA
```

### 6. Train the models

```bash
PYTHONPATH=. python src/train.py          # ATP  (~5 min) → models/best_model.pkl
PYTHONPATH=. python src/train_wta.py      # WTA  (~5 min) → models/best_model_wta.pkl
```

### 7. Start the prediction server

```bash
PYTHONPATH=. python src/prediction_server.py
# ✓  Server ready → http://localhost:8001/predict  [ATP + WTA]
```

### 8. Open the dashboard

Open `outputs/pipeline_dashboard.html` in your browser. The server status dot in **Section 07** turns green when the prediction server is reachable.

## Predicting a Match via the API

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player1_id": 104925,
    "player2_id": 207989,
    "surface": "Clay",
    "level": "G",
    "tour": "atp"
  }'
```

```json
{
  "player1_win_prob": 62.4,
  "player2_win_prob": 37.6,
  "model_type": "gradient_boosting",
  "tour": "atp",
  "match_count": 108375,
  "features_used": {
    "p1_elo": 2143.0,
    "p2_elo": 2089.0,
    "elo_diff": 54.0,
    "p1_rank": 1,
    "p2_rank": 3
  }
}
```

The `/players?tour=atp` and `/players?tour=wta` endpoints return the full player lists for autocomplete.

## Key Engineering Decisions

### Chronological train/test split — no data leakage
All matches before 2021-01-01 form the training set; 2021–2024 is the held-out test set. A random split would let future match statistics leak backward into training features, inflating reported accuracy without any real predictive signal.

### Point-in-time ELO snapshots
The ELO update loop stores each player's rating *before* applying the match result. Every training row therefore sees the rating the model would have at prediction time — not the post-match rating that encodes the outcome being predicted.

### Vectorized rolling windows with `closed='left'`
Form and fatigue use `groupby + rolling(closed='left')` so the current match is excluded from its own rolling window. Without `closed='left'`, the model would see the outcome of the match it is trying to predict — a subtle but catastrophic leakage bug.

### Perspective mirroring for balanced labels
Raw match data always labels the winner as player 1, so every label is 1. Training on this produces a model that assigns 100% win probability to every player 1. To fix this, every row is duplicated from the loser's perspective (winner/loser columns swapped, all `*_diff` columns negated, `target = 0`). This doubles the dataset and creates perfectly balanced classes without synthetic oversampling.

## Project Structure

```
tennis-predictor/
├── config.py                      Global constants (paths, ELO K-factor, etc.)
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── tennis_atp/            JeffSackmann/tennis_atp (git clone)
│   │   └── tennis_wta/            JeffSackmann/tennis_wta (git clone)
│   └── processed/
│       ├── all_matches.csv
│       ├── all_wta_matches.csv
│       ├── player_stats.json
│       └── player_stats_wta.json
├── models/
│   ├── best_model.pkl             ATP GradientBoostingClassifier bundle
│   └── best_model_wta.pkl         WTA GradientBoostingClassifier bundle
├── outputs/
│   └── pipeline_dashboard.html    Self-contained interactive dashboard
└── src/
    ├── data_loader.py             ATP + WTA data ingestion
    ├── features.py                Point-in-time feature extraction
    ├── models.py                  Model wrappers (LR, RF, GBM)
    ├── train.py                   ATP training pipeline
    ├── train_wta.py               WTA training pipeline
    ├── predict_match.py           CLI match predictor
    ├── prediction_server.py       HTTP prediction API (port 8001)
    └── export_player_stats.py     Pre-compute player stats JSON for dashboard
```

## Data

All match data comes from [Jeff Sackmann's open tennis datasets](https://github.com/JeffSackmann):

- [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — ATP tour matches 1968–present
- [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) — WTA tour matches 1968–present

This project uses 1991–2024 for both tours. The datasets are licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) and are not included in this repository — clone them separately in the Quick Start steps above.

## License

MIT
