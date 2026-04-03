"""
Model Training and Evaluation Module

This module contains the TennisModel class for training and evaluating
machine learning models to predict tennis match outcomes.
"""

import logging
import pickle
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import config

logger = logging.getLogger(__name__)


class TennisModel:
    """
    Tennis match prediction model.
    
    This class handles model training, evaluation, and prediction for tennis matches.
    Supports multiple model types (Random Forest, Gradient Boosting, Logistic Regression).
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the tennis prediction model.
        
        Args:
            model_type (str): Type of model to use.
                             Options: 'random_forest', 'gradient_boosting', 'logistic_regression'
                             Default: 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        logger.info(f"Initialized TennisModel with {model_type}")
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=config.RANDOM_STATE
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=config.RANDOM_STATE
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        test_size: float = None,
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data into training and testing sets.
        
        Args:
            features_df (pd.DataFrame): Features dataframe with 'target' column
            test_size (float, optional): Proportion of data for testing (default from config)
            feature_cols (list, optional): List of feature column names to use
                                          If None, uses all columns except 'target'
                                          
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        logger.info(f"After removing NaN values: {len(features_df)} samples")
        
        # Determine feature columns
        if feature_cols is None:
            exclude_cols = ['target', 'match_id', 'winner_id', 'loser_id', 'date']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        features_df = features_df.sort_values('date')  # sort chronologically so past matches train, future matches test

        X = features_df[feature_cols].values
        y = features_df['target'].values

        # Chronological split: earlier 80% trains, later 20% tests — random split would leak future match outcomes into training
        split_idx = int(len(features_df) * (1 - test_size))  # index where the held-out test period begins
        X_train, X_test = X[:split_idx], X[split_idx:]  # earlier matches for training
        y_train, y_test = y[:split_idx], y[split_idx:]  # later matches for testing

        logger.info(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        features_df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        verbose: bool = True
    ) -> dict:
        """
        Train the tennis prediction model.
        
        Args:
            features_df (pd.DataFrame): Features dataframe with 'target' column
            feature_cols (list, optional): Specific feature columns to use
            verbose (bool): Print training information
            
        Returns:
            dict: Training metrics and results
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            features_df,
            feature_cols=feature_cols
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training and test sets
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Testing accuracy: {test_score:.4f}")
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, F1, ROC-AUC
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate probabilities for ROC-AUC
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features (np.ndarray): Feature array for prediction
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for new data.
        
        Args:
            features (np.ndarray): Feature array for prediction
            
        Returns:
            np.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model type {self.model_type} does not support predict_proba")
        
        return self.model.predict_proba(features)
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str, optional): Path to save the model.
                                     Default: models/tennis_prediction_model.pkl
                                     
        Returns:
            str: Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        if filepath is None:
            os.makedirs(config.MODELS_PATH, exist_ok=True)
            filepath = os.path.join(config.MODELS_PATH, f"{config.MODEL_NAME}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model from disk.
        
        Args:
            filepath (str): Path to the saved model file
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores (if available).
        
        Returns:
            dict: Feature names mapped to importance scores,
                  or None if model doesn't support feature importance
        """
        return self.feature_importance
