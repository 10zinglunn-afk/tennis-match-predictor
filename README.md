# Tennis Match Prediction Project

A machine learning project to predict tennis match outcomes using historical data from Jeff Sackmann's tennis datasets.

## Project Structure

```
tennis-predictor/
├── data/
│   ├── raw/              # Raw data from Jeff Sackmann's repository
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data_loader.py    # Functions to download and load tennis data
│   ├── features.py       # Feature engineering functions
│   └── models.py         # Model training and evaluation
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Trained model artifacts
├── config.py             # Configuration parameters
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

This project uses Jeff Sackmann's tennis datasets:
- ATP matches data (historical match results)
- ATP rankings data (player rankings over time)

Data is automatically downloaded from: https://github.com/JeffSackmann/tennis_atp

## Features

- **ELO Ratings**: Player strength assessment based on match history
- **Form Factor**: Recent performance over the last 90 days
- **Fatigue Factor**: Impact of match frequency and travel

## Usage

```python
from src.data_loader import load_atp_matches
from src.features import FeatureExtractor
from src.models import TennisModel

# Load data
matches_df = load_atp_matches(years=[2023, 2024])

# Extract features
extractor = FeatureExtractor()
features_df = extractor.extract_all_features(matches_df)

# Train model
model = TennisModel()
model.train(features_df)
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- See requirements.txt for full list

## License

This project is for educational purposes.

## Data Source

Tennis data provided by Jeff Sackmann:
- [Tennis ATP](https://github.com/JeffSackmann/tennis_atp)
