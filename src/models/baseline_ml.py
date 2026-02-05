"""
Baseline ML Models for RUL Prediction

Implements traditional ML models:
- XGBoost Regressor (primary)
- Random Forest
- Support Vector Regression
- Gradient Boosting
"""

import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class XGBoostRULPredictor:
    """XGBoost model for RUL prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost RUL predictor.

        Parameters
        ----------
        n_estimators : int, default=200
            Number of boosting rounds
        max_depth : int, default=6
            Maximum tree depth
        learning_rate : float, default=0.1
            Step size shrinkage
        subsample : float, default=0.8
            Subsample ratio of training instances
        colsample_bytree : float, default=0.8
            Subsample ratio of columns per tree
        gamma : float, default=0.1
            Minimum loss reduction for split
        reg_alpha : float, default=0.1
            L1 regularization
        reg_lambda : float, default=1.0
            L2 regularization
        random_state : int, default=42
            Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        self.params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False
        self.feature_importance_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ) -> "XGBoostRULPredictor":
        """
        Fit XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets (RUL)
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        early_stopping_rounds : int, default=20
            Early stopping patience
        verbose : bool, default=True
            Print training progress

        Returns
        -------
        self : XGBoostRULPredictor
            Fitted model
        """
        logger.info(f"Training XGBoost model with {X_train.shape[0]} samples, {X_train.shape[1]} features")

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)

        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        logger.info("✓ XGBoost training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL values.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        predictions : np.ndarray
            Predicted RUL values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        X : np.ndarray
            Features
        y_true : np.ndarray
            True RUL values

        Returns
        -------
        metrics : dict
            Dictionary with RMSE, MAE, R2, MAPE
        """
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
        }
        
        logger.info(f"Evaluation metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        return metrics

    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Parameters
        ----------
        feature_names : list, optional
            Names of features

        Returns
        -------
        importance_df : pd.DataFrame
            Feature importance dataframe
        """
        if self.feature_importance_ is None:
            raise RuntimeError("Model must be fitted first")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        self.model.load_model(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class RandomForestRULPredictor:
    """Random Forest model for RUL prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Random Forest predictor."""
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1,
        }
        self.params.update(kwargs)
        
        self.model = RandomForestRegressor(**self.params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestRULPredictor":
        """Fit Random Forest model."""
        logger.info(f"Training Random Forest with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("✓ Random Forest training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL values."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


class GradientBoostingRULPredictor:
    """Gradient Boosting model for RUL prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Gradient Boosting predictor."""
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'random_state': random_state,
        }
        self.params.update(kwargs)
        
        self.model = GradientBoostingRegressor(**self.params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GradientBoostingRULPredictor":
        """Fit Gradient Boosting model."""
        logger.info(f"Training Gradient Boosting with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("✓ Gradient Boosting training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL values."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
