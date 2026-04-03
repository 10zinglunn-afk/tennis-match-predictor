"""
Feature Engineering Module

This module contains functions for extracting and computing features used
in the tennis prediction model, including ELO ratings, player form, and fatigue metrics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Class for extracting features from tennis match data.
    """
    
    def __init__(self):
        """Initialize the feature extractor with configuration parameters."""
        self.elo_initial = config.ELO_INITIAL_RATING
        self.elo_k = config.ELO_K_FACTOR
        self.form_window = config.FORM_WINDOW_DAYS
        self.fatigue_weight = config.FATIGUE_WEIGHT
        self.player_elos = {}  # Store ELO ratings over time
        logger.info("FeatureExtractor initialized")
    
    def calculate_elo_ratings(self, matches_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate ELO ratings for all players based on match history.
        
        ELO rating system:
        - Initial rating: 1500
        - K-factor: 32 (determines rating change magnitude)
        - Expected score: 1 / (1 + 10^((opponent_elo - player_elo) / 400))
        - New rating: current_rating + K * (actual_score - expected_score)
        
        Args:
            matches_df (pd.DataFrame): DataFrame with match results containing
                                      winner_id, loser_id columns
                                      
        Returns:
            Dict[int, float]: Dictionary mapping player_id to current ELO rating
        """
        logger.info("Calculating ELO ratings...")

        player_elos = {}
        elo_history = {}  # pre-match snapshot per match so extract_all_features uses rating before, not after, each game
        match_count = 0

        # Sort by date to calculate ELO chronologically
        if 'tourney_date' in matches_df.columns:
            matches_sorted = matches_df.sort_values('tourney_date')
        else:
            matches_sorted = matches_df

        for _, match in matches_sorted.iterrows():
            try:
                winner_id = int(match['winner_id'])
                loser_id = int(match['loser_id'])

                # Get current ELO (or initialize)
                winner_elo = player_elos.get(winner_id, self.elo_initial)
                loser_elo = player_elos.get(loser_id, self.elo_initial)

                # Snapshot ratings BEFORE updating — keyed by match so callers get pre-match ELO, not post-match
                elo_history[f"{winner_id}_{loser_id}_{str(match['tourney_date'])}"] = (winner_elo, loser_elo)

                # Calculate expected scores
                winner_expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
                loser_expected = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))

                # Update ratings (winner gets +1, loser gets 0)
                player_elos[winner_id] = winner_elo + self.elo_k * (1 - winner_expected)
                player_elos[loser_id] = loser_elo + self.elo_k * (0 - loser_expected)

                match_count += 1
            except (KeyError, ValueError) as e:
                logger.debug(f"Error processing match: {e}")
                continue

        logger.info(f"Calculated ELO ratings for {len(player_elos)} players across {match_count} matches")
        return player_elos, elo_history  # return snapshot dict alongside final ratings
    
    def calculate_player_form(
        self,
        matches_df: pd.DataFrame,
        player_id: int,
        reference_date: str,
        window_days: int = None
    ) -> float:
        """
        Calculate player form based on recent match results.
        
        Form metric: Win percentage over the last N days
        - Looks at recent matches within the specified window
        - Returns win percentage (0.0 to 1.0)
        
        Args:
            matches_df (pd.DataFrame): All match data
            player_id (int): Player to calculate form for
            reference_date (str): Reference date (format: 'YYYYMMDD')
            window_days (int, optional): Number of days to look back
            
        Returns:
            float: Win percentage (0.0 to 1.0), or NaN if no matches found
        """
        if window_days is None:
            window_days = self.form_window
        
        try:
            # Parse reference date
            ref_date = pd.to_datetime(reference_date, format='%Y%m%d')
            start_date = ref_date - timedelta(days=window_days)
            
            # Filter matches for this player within time window
            player_matches = matches_df[
                ((matches_df['winner_id'] == player_id) | 
                 (matches_df['loser_id'] == player_id)) &
                (pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d') >= start_date) &
                (pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d') <= ref_date)
            ]
            
            if len(player_matches) == 0:
                return np.nan
            
            # Calculate win percentage
            wins = len(player_matches[player_matches['winner_id'] == player_id])
            win_percentage = wins / len(player_matches)
            
            return win_percentage
        except Exception as e:
            logger.debug(f"Error calculating form for player {player_id}: {e}")
            return np.nan
    
    def calculate_fatigue_factor(
        self,
        matches_df: pd.DataFrame,
        player_id: int,
        reference_date: str,
        window_days: int = 30
    ) -> float:
        """
        Calculate player fatigue based on recent match frequency.
        
        Fatigue metric: Number of matches in recent period
        - Higher number of matches = higher fatigue
        - Normalized by dividing by number of days in the window
        - Returns fatigue score (0.0 to 1.0 scale)
        
        Args:
            matches_df (pd.DataFrame): All match data
            player_id (int): Player to calculate fatigue for
            reference_date (str): Reference date (format: 'YYYYMMDD')
            window_days (int): Number of days to analyze
            
        Returns:
            float: Fatigue score (0.0 = no fatigue, higher = more fatigue), or NaN if error
        """
        try:
            # Parse reference date
            ref_date = pd.to_datetime(reference_date, format='%Y%m%d')
            start_date = ref_date - timedelta(days=window_days)
            
            # Filter matches for this player
            player_matches = matches_df[
                ((matches_df['winner_id'] == player_id) | 
                 (matches_df['loser_id'] == player_id)) &
                (pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d') >= start_date) &
                (pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d') <= ref_date)
            ]
            
            # Calculate matches per day (normalized fatigue)
            match_count = len(player_matches)
            matches_per_day = match_count / max(window_days, 1)
            
            # Normalize to 0-1 scale (assuming 1 match per day = max fatigue)
            fatigue_score = min(matches_per_day, 1.0)
            
            return fatigue_score
        except Exception as e:
            logger.debug(f"Error calculating fatigue for player {player_id}: {e}")
            return np.nan
    
    def extract_all_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features for the dataset.
        
        This is a skeleton function that combines all feature calculations.
        
        Args:
            matches_df (pd.DataFrame): Raw match data
            
        Returns:
            pd.DataFrame: Features dataframe with rows for each match and feature columns
        """
        logger.info("Extracting all features...")

        # Unpack snapshot dict so ELO features use pre-match ratings, not post-match ratings
        player_elos, elo_history = self.calculate_elo_ratings(matches_df)

        # Build base feature columns directly from the dataframe (no iterrows)
        # Include rank, age, and surface now so they're available for new features below
        features_df = matches_df[['winner_id', 'loser_id', 'tourney_date', 'surface',
                                   'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
                                   'tourney_level', 'best_of']].copy()  # include level + best_of so they survive subsequent merges
        features_df['winner_id'] = features_df['winner_id'].astype(int)
        features_df['loser_id'] = features_df['loser_id'].astype(int)
        features_df['date'] = features_df['tourney_date'].astype(str)
        features_df['match_id'] = (features_df['winner_id'].astype(str) + '_' +
                                   features_df['loser_id'].astype(str) + '_' +
                                   features_df['date'])
        features_df['target'] = 1
        features_df = features_df.drop(columns='tourney_date')

        # Feature 1: ranking difference — fill NaN before differencing so unranked players don't produce NaN features
        features_df['winner_rank'] = features_df['winner_rank'].fillna(500)  # 500 = reasonable unranked placeholder
        features_df['loser_rank'] = features_df['loser_rank'].fillna(500)
        features_df['winner_rank_diff'] = features_df['winner_rank'] - features_df['loser_rank']  # positive = loser is ranked higher (underdog winning)

        # Feature 2: age difference — values come directly from the CSV, no computation needed
        features_df['winner_age_diff'] = features_df['winner_age'] - features_df['loser_age']

        # Feature: tournament level one-hot — G=Grand Slam, M=Masters, everything else (F/A/D) → is_other
        features_df['is_grand_slam'] = (features_df['tourney_level'] == 'G').astype(int)
        features_df['is_masters']    = (features_df['tourney_level'] == 'M').astype(int)
        features_df['is_other']      = (~features_df['tourney_level'].isin(['G', 'M'])).astype(int)  # F, A, D all collapse here
        features_df = features_df.drop(columns='tourney_level')  # drop raw code after encoding

        # Feature: best_of — already an integer (3 or 5) in the CSV, carry through as-is
        # (already present in features_df from the initial copy; no transformation needed)

        # Look up pre-match ELO snapshot for each match — prevents post-match ELO from leaking into features
        elo_keys = (features_df['winner_id'].astype(str) + '_' +
                    features_df['loser_id'].astype(str) + '_' +
                    features_df['date'])
        default = (self.elo_initial, self.elo_initial)
        snapshots = elo_keys.map(lambda k: elo_history.get(k, default))  # pre-match snapshot keyed by match id
        features_df['winner_elo'] = snapshots.map(lambda s: s[0])
        features_df['loser_elo'] = snapshots.map(lambda s: s[1])
        features_df['elo_diff'] = features_df['winner_elo'] - features_df['loser_elo']

        # --- Vectorized form & fatigue via rolling window ---
        # Melt into long format: one row per player per match, required for per-player rolling aggregation
        winners = matches_df[['tourney_date', 'winner_id']].copy()
        winners['player_id'] = winners['winner_id'].astype(int)
        winners['won'] = 1
        losers = matches_df[['tourney_date', 'loser_id']].copy()
        losers['player_id'] = losers['loser_id'].astype(int)
        losers['won'] = 0
        long_df = pd.concat(
            [winners[['tourney_date', 'player_id', 'won']],
             losers[['tourney_date', 'player_id', 'won']]],
            ignore_index=True
        )  # one row per player per match, enabling time-based groupby rolling
        long_df['date_dt'] = pd.to_datetime(long_df['tourney_date'].astype(str), format='%Y%m%d')
        long_df = long_df.sort_values(['player_id', 'date_dt']).set_index('date_dt')

        grp = long_df.groupby('player_id')['won']
        # closed='left' is required: excludes the current match from its own rolling window, preventing form/fatigue leakage
        long_df['form'] = grp.rolling(f"{self.form_window}D", closed='left').mean().reset_index(level=0, drop=True)
        long_df['fatigue'] = (  # normalize match count to 0-1 scale matching original calculate_fatigue_factor output
            grp.rolling('30D', closed='left').count().reset_index(level=0, drop=True) / 30
        ).clip(upper=1.0)

        long_df = long_df.reset_index()
        long_df['tourney_date'] = long_df['tourney_date'].astype(str)  # align with features_df['date'] which is also str
        # closed='left' makes all same-day rolling values identical for a player, so deduplication is safe
        stats = (long_df[['player_id', 'tourney_date', 'form', 'fatigue']]
                 .drop_duplicates(subset=['player_id', 'tourney_date']))

        # Merge rolling stats back by player and date — avoids any per-row Python calls for form/fatigue
        winner_stats = stats.rename(columns={
            'player_id': 'winner_id', 'tourney_date': 'date',
            'form': 'winner_form', 'fatigue': 'winner_fatigue'
        })
        features_df = features_df.merge(winner_stats, on=['winner_id', 'date'], how='left')

        loser_stats = stats.rename(columns={
            'player_id': 'loser_id', 'tourney_date': 'date',
            'form': 'loser_form', 'fatigue': 'loser_fatigue'
        })
        features_df = features_df.merge(loser_stats, on=['loser_id', 'date'], how='left')

        # --- Feature 3: vectorized per-player surface win rate ---
        # Build long format carrying surface so we can group by (player, surface)
        winners_s = matches_df[['tourney_date', 'winner_id', 'surface']].rename(columns={'winner_id': 'player_id'})
        winners_s['won'] = 1
        losers_s = matches_df[['tourney_date', 'loser_id', 'surface']].rename(columns={'loser_id': 'player_id'})
        losers_s['won'] = 0
        surface_long = pd.concat([winners_s, losers_s], ignore_index=True)  # one row per player per match, with surface label
        surface_long['player_id'] = surface_long['player_id'].astype(int)

        # Aggregate to (player, surface, date) so all same-day matches see identical pre-day history
        daily_surface = (surface_long
            .groupby(['player_id', 'surface', 'tourney_date'])
            .agg(day_wins=('won', 'sum'), day_matches=('won', 'count'))
            .reset_index()
            .sort_values(['player_id', 'surface', 'tourney_date']))

        grp_s = daily_surface.groupby(['player_id', 'surface'])
        # Subtract today's contribution from cumsum = strictly all matches before today (no current-match leakage)
        daily_surface['cum_wins'] = grp_s['day_wins'].cumsum() - daily_surface['day_wins']
        daily_surface['cum_matches'] = grp_s['day_matches'].cumsum() - daily_surface['day_matches']

        safe_denom = daily_surface['cum_matches'].replace(0, 1)  # avoid ZeroDivisionError; < 5 cases are overridden below anyway
        daily_surface['surface_winrate'] = np.where(
            daily_surface['cum_matches'] < 5,
            0.5,  # fewer than 5 surface matches = no reliable estimate, default to coin flip
            daily_surface['cum_wins'] / safe_denom
        )

        daily_surface['tourney_date'] = daily_surface['tourney_date'].astype(str)  # match 'date' column type in features_df
        surface_stats = daily_surface[['player_id', 'surface', 'tourney_date', 'surface_winrate']]

        # Merge winner and loser surface win rates separately, keyed by player + surface + date
        features_df = features_df.merge(
            surface_stats.rename(columns={'player_id': 'winner_id', 'tourney_date': 'date',
                                          'surface_winrate': 'winner_surface_winrate'}),
            on=['winner_id', 'surface', 'date'], how='left'
        )
        features_df = features_df.merge(
            surface_stats.rename(columns={'player_id': 'loser_id', 'tourney_date': 'date',
                                          'surface_winrate': 'loser_surface_winrate'}),
            on=['loser_id', 'surface', 'date'], how='left'
        )
        # Players with zero prior matches on a surface get NaN from the merge — treat as coin flip
        features_df['winner_surface_winrate'] = features_df['winner_surface_winrate'].fillna(0.5)
        features_df['loser_surface_winrate'] = features_df['loser_surface_winrate'].fillna(0.5)
        features_df['winner_surface_winrate_diff'] = (  # positive = winner has stronger surface record
            features_df['winner_surface_winrate'] - features_df['loser_surface_winrate']
        )

        # --- Feature: head-to-head record ---
        # Build long format with both perspectives of every match so we can groupby (player, opponent)
        h2h_w = matches_df[['tourney_date', 'winner_id', 'loser_id']].rename(
            columns={'winner_id': 'player_id', 'loser_id': 'opponent_id'})
        h2h_w['won'] = 1
        h2h_l = matches_df[['tourney_date', 'winner_id', 'loser_id']].rename(
            columns={'loser_id': 'player_id', 'winner_id': 'opponent_id'})
        h2h_l['won'] = 0
        h2h_long = pd.concat([h2h_w, h2h_l], ignore_index=True)  # one row per player per match, keyed by (player, opponent)
        h2h_long['player_id']   = h2h_long['player_id'].astype(int)
        h2h_long['opponent_id'] = h2h_long['opponent_id'].astype(int)
        h2h_long = h2h_long.sort_values(['player_id', 'opponent_id', 'tourney_date'])

        grp_h2h = h2h_long.groupby(['player_id', 'opponent_id'])
        # cumsum - current won = wins against this opponent strictly before today
        h2h_long['cum_h2h_wins']    = grp_h2h['won'].cumsum() - h2h_long['won']
        # cumcount = 0-based index within group = number of prior meetings with this opponent
        h2h_long['cum_h2h_matches'] = grp_h2h['won'].cumcount()

        h2h_long['tourney_date'] = h2h_long['tourney_date'].astype(str)
        h2h_stats = h2h_long[['player_id', 'opponent_id', 'tourney_date', 'cum_h2h_wins', 'cum_h2h_matches']]

        # Merge winner's h2h record (and total meetings count) against this specific opponent
        features_df = features_df.merge(
            h2h_stats.rename(columns={'player_id': 'winner_id', 'opponent_id': 'loser_id',
                                      'tourney_date': 'date', 'cum_h2h_wins': 'h2h_winner_wins',
                                      'cum_h2h_matches': 'h2h_total'}),
            on=['winner_id', 'loser_id', 'date'], how='left'
        )
        # Merge loser's h2h wins — player_id=loser, opponent_id=winner in h2h_stats
        features_df = features_df.merge(
            h2h_stats[['player_id', 'opponent_id', 'tourney_date', 'cum_h2h_wins']].rename(
                columns={'player_id': 'loser_id', 'opponent_id': 'winner_id',
                         'tourney_date': 'date', 'cum_h2h_wins': 'h2h_loser_wins'}),
            on=['loser_id', 'winner_id', 'date'], how='left'
        )
        features_df['h2h_winner_wins'] = features_df['h2h_winner_wins'].fillna(0).astype(int)
        features_df['h2h_loser_wins']  = features_df['h2h_loser_wins'].fillna(0).astype(int)
        features_df['h2h_total']       = features_df['h2h_total'].fillna(0).astype(int)
        # Zero out diff when there are fewer than 3 h2h matches — not enough signal to use
        features_df['h2h_diff'] = np.where(
            features_df['h2h_total'] < 3,
            0,
            features_df['h2h_winner_wins'] - features_df['h2h_loser_wins']
        )

        # --- Feature: days since last match (rest proxy) ---
        # Long format: one row per player per match so we can shift within each player's timeline
        rest_long = pd.concat([
            matches_df[['tourney_date', 'winner_id']].rename(columns={'winner_id': 'player_id'}),
            matches_df[['tourney_date', 'loser_id']].rename(columns={'loser_id': 'player_id'}),
        ], ignore_index=True)
        rest_long['player_id'] = rest_long['player_id'].astype(int)
        rest_long['date_dt'] = pd.to_datetime(rest_long['tourney_date'].astype(str), format='%Y%m%d')

        # Deduplicate to one row per (player, day): same-day matches share identical rest value
        daily_rest = (rest_long
            .drop_duplicates(subset=['player_id', 'tourney_date'])
            .sort_values(['player_id', 'date_dt']))
        daily_rest['prev_date'] = daily_rest.groupby('player_id')['date_dt'].shift(1)  # previous calendar day this player played
        daily_rest['days_rest'] = (daily_rest['date_dt'] - daily_rest['prev_date']).dt.days
        daily_rest['days_rest'] = daily_rest['days_rest'].fillna(14).clip(upper=30)  # 14 = assumed average; 30+ days all treated the same

        daily_rest['tourney_date'] = daily_rest['tourney_date'].astype(str)
        rest_stats = daily_rest[['player_id', 'tourney_date', 'days_rest']]

        features_df = features_df.merge(
            rest_stats.rename(columns={'player_id': 'winner_id', 'tourney_date': 'date', 'days_rest': 'days_rest_winner'}),
            on=['winner_id', 'date'], how='left'
        )
        features_df = features_df.merge(
            rest_stats.rename(columns={'player_id': 'loser_id', 'tourney_date': 'date', 'days_rest': 'days_rest_loser'}),
            on=['loser_id', 'date'], how='left'
        )
        # Players with no prior match in the dataset (true first match) get NaN — fill with assumed average
        features_df['days_rest_winner'] = features_df['days_rest_winner'].fillna(14)
        features_df['days_rest_loser']  = features_df['days_rest_loser'].fillna(14)
        features_df['days_rest_diff'] = (  # positive = winner is more rested than loser
            features_df['days_rest_winner'] - features_df['days_rest_loser']
        )

        logger.info(f"Extracted features for {len(features_df)} matches")
        return features_df
