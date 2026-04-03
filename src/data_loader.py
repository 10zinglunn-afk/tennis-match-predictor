"""
Data Loader Module

This module provides functions to load Jeff Sackmann's tennis datasets
from the cloned tennis_atp / tennis_wta repositories at data/raw/.
"""

import os
import logging
from typing import List
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Root of the cloned Sackmann repos
SACKMANN_DIR = os.path.join(config.DATA_RAW_PATH, "tennis_atp")
WTA_DIR      = os.path.join(config.DATA_RAW_PATH, "tennis_wta")


def load_matches(year: int) -> pd.DataFrame:
    """
    Load ATP matches for a single year from the cloned Sackmann repo.

    Args:
        year (int): Calendar year (e.g. 2023)

    Returns:
        pd.DataFrame: Match rows for that year.

    Raises:
        FileNotFoundError: If the CSV for that year is not present.
    """
    filepath = os.path.join(SACKMANN_DIR, f"atp_matches_{year}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No match file for {year} at {filepath}. "
            "Make sure the tennis_atp repo is cloned into data/raw/tennis_atp/."
        )
    df = pd.read_csv(filepath)
    df["year"] = year  # add year column so callers can filter after concatenation
    logger.info(f"Loaded {len(df)} matches for {year} from {filepath}")
    return df


def load_all_matches(start: int, end: int) -> pd.DataFrame:
    """
    Load and concatenate ATP matches for a range of years (inclusive).

    Args:
        start (int): First year to include (e.g. 2010)
        end (int): Last year to include (e.g. 2023)

    Returns:
        pd.DataFrame: All matches across the requested range, sorted by tourney_date.
    """
    frames = []
    for year in range(start, end + 1):  # +1 so end year is included
        try:
            frames.append(load_matches(year))
        except FileNotFoundError as e:
            logger.warning(str(e))  # skip missing years rather than aborting the whole load

    if not frames:
        logger.warning(f"No match data loaded for {start}–{end}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "tourney_date" in combined.columns:
        combined = combined.sort_values("tourney_date").reset_index(drop=True)  # chronological order for safe time-series splits
    logger.info(f"Loaded {len(combined)} total matches for {start}–{end}")
    return combined


def load_players() -> pd.DataFrame:
    """
    Load the ATP player registry from the cloned Sackmann repo.

    Returns:
        pd.DataFrame: Player records (player_id, name, hand, dob, country, etc.)

    Raises:
        FileNotFoundError: If atp_players.csv is not present.
    """
    filepath = os.path.join(SACKMANN_DIR, "atp_players.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Player file not found at {filepath}.")
    df = pd.read_csv(
        filepath,
        header=None,  # Sackmann's atp_players.csv has no header row
        names=["player_id", "first_name", "last_name", "hand", "dob", "country"],
    )
    logger.info(f"Loaded {len(df)} players from {filepath}")
    return df


def load_rankings() -> pd.DataFrame:
    """
    Load current ATP rankings from the cloned Sackmann repo.

    Returns:
        pd.DataFrame: Ranking rows (ranking_date, rank, player_id, points)

    Raises:
        FileNotFoundError: If atp_rankings_current.csv is not present.
    """
    filepath = os.path.join(SACKMANN_DIR, "atp_rankings_current.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Rankings file not found at {filepath}.")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} ranking rows from {filepath}")
    return df


def build_processed_dataset(start: int = 1991, end: int = 2024) -> pd.DataFrame:
    """
    Load all ATP matches for the given year range and cache to data/processed/all_matches.csv
    so downstream code can read one file instead of iterating raw CSVs each run.

    Args:
        start (int): First year to include (default 1991, first year in Sackmann repo)
        end (int): Last year to include (default 2024)

    Returns:
        pd.DataFrame: Combined and sorted matches DataFrame.
    """
    df = load_all_matches(start, end)
    os.makedirs(config.DATA_PROCESSED_PATH, exist_ok=True)
    out_path = os.path.join(config.DATA_PROCESSED_PATH, "all_matches.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
    logger.info(f"Processed dataset written to {out_path}")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the structure of a loaded matches DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to validate

    Returns:
        bool: True if all required columns are present and the frame is non-empty.
    """
    required_columns = ["tourney_date", "winner_id", "loser_id"]

    if df.empty:
        logger.warning("Empty dataframe")
        return False

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    logger.info("Data validation passed")
    return True


def load_wta_matches(year: int) -> pd.DataFrame:
    """
    Load WTA matches for a single year from the cloned Sackmann repo.

    Args:
        year (int): Calendar year (e.g. 2023)

    Returns:
        pd.DataFrame: Match rows for that year.

    Raises:
        FileNotFoundError: If the CSV for that year is not present.
    """
    filepath = os.path.join(WTA_DIR, f"wta_matches_{year}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No WTA match file for {year} at {filepath}. "
            "Make sure the tennis_wta repo is cloned into data/raw/tennis_wta/."
        )
    df = pd.read_csv(filepath, low_memory=False)
    df["year"] = year
    logger.info(f"Loaded {len(df)} WTA matches for {year} from {filepath}")
    return df


def load_all_wta_matches(start: int, end: int) -> pd.DataFrame:
    """
    Load and concatenate WTA matches for a range of years (inclusive).

    Args:
        start (int): First year to include (e.g. 1991)
        end (int): Last year to include (e.g. 2024)

    Returns:
        pd.DataFrame: All matches across the requested range, sorted by tourney_date.
    """
    frames = []
    for year in range(start, end + 1):
        try:
            frames.append(load_wta_matches(year))
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not frames:
        logger.warning(f"No WTA match data loaded for {start}–{end}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "tourney_date" in combined.columns:
        combined = combined.sort_values("tourney_date").reset_index(drop=True)
    logger.info(f"Loaded {len(combined)} total WTA matches for {start}–{end}")
    return combined


def build_wta_processed_dataset(start: int = 1991, end: int = 2024) -> pd.DataFrame:
    """
    Load all WTA matches for the given year range and cache to
    data/processed/all_wta_matches.csv.

    Args:
        start (int): First year to include (default 1991)
        end (int): Last year to include (default 2024)

    Returns:
        pd.DataFrame: Combined and sorted matches DataFrame.
    """
    df = load_all_wta_matches(start, end)
    os.makedirs(config.DATA_PROCESSED_PATH, exist_ok=True)
    out_path = os.path.join(config.DATA_PROCESSED_PATH, "all_wta_matches.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
    logger.info(f"WTA processed dataset written to {out_path}")
    return df


if __name__ == "__main__":
    build_processed_dataset()
