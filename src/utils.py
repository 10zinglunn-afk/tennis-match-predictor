"""
Utility Functions Module

This module contains helper functions used across the tennis prediction project.
"""

import logging
from typing import Any, Dict
import pandas as pd
import numpy as np


def setup_logging(
    log_level: str = "INFO",
    log_format: str = None
) -> logging.Logger:
    """
    Setup and configure logging for the project.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format (str, optional): Custom log format string
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )
    
    return logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        
    Returns:
        bool: True if all required columns present, False otherwise
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        logging.warning(f"Missing columns: {missing}")
        return False
    
    return True


def convert_date_format(date_str: str, from_format: str = "%Y%m%d", to_format: str = "%Y-%m-%d") -> str:
    """
    Convert date string between formats.
    
    Args:
        date_str (str): Date string to convert
        from_format (str): Input date format
        to_format (str): Output date format
        
    Returns:
        str: Converted date string
    """
    try:
        date_obj = pd.to_datetime(date_str, format=from_format)
        return date_obj.strftime(to_format)
    except Exception as e:
        logging.error(f"Error converting date {date_str}: {e}")
        return None


def calculate_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate common statistics for a numeric series.
    
    Args:
        series (pd.Series): Numeric series to analyze
        
    Returns:
        Dict: Dictionary with statistics (mean, median, std, min, max, etc.)
    """
    stats = {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'count': series.count(),
        'null_count': series.isna().sum()
    }
    
    return stats


def normalize_series(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a numeric series.
    
    Args:
        series (pd.Series): Series to normalize
        method (str): Normalization method ('minmax' for 0-1, 'zscore' for standardization)
        
    Returns:
        pd.Series: Normalized series
    """
    if method == 'minmax':
        # Min-Max scaling to [0, 1]
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        # Z-score standardization
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    fill_value: Any = None,
    columns: list = None
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): How to handle missing values
                       'drop': Remove rows with missing values
                       'fill': Fill with fill_value
                       'mean': Fill with column mean (numeric only)
                       'forward_fill': Forward fill
        fill_value (Any, optional): Value to fill with when strategy='fill'
        columns (list, optional): Specific columns to apply strategy to
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    elif strategy == 'fill':
        return df_copy.fillna(fill_value)
    elif strategy == 'mean':
        for col in columns:
            if df_copy[col].dtype in [np.float64, np.int64]:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        return df_copy
    elif strategy == 'forward_fill':
        return df_copy.fillna(method='ffill')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_match_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create match pairs from match data (both player perspectives).
    
    Converts each match into two rows - one for winner and one for loser.
    
    Args:
        df (pd.DataFrame): Match data with winner_id and loser_id columns
        
    Returns:
        pd.DataFrame: DataFrame with doubled rows and outcome labels
    """
    # Keep original match data with label 1 for winner
    winner_df = df.copy()
    winner_df['outcome'] = 1
    
    # Create loser perspective by swapping winner/loser
    loser_df = df.copy()
    loser_df.columns = loser_df.columns.str.replace('winner', 'temp_winner')
    loser_df.columns = loser_df.columns.str.replace('loser', 'winner')
    loser_df.columns = loser_df.columns.str.replace('temp_winner', 'loser')
    loser_df['outcome'] = 0
    
    # Combine both perspectives
    combined = pd.concat([winner_df, loser_df], ignore_index=True)
    
    return combined
